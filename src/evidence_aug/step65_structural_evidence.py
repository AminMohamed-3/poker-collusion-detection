"""Step 65: training-only audit of STRUCTURAL relational-action features as an evidence ranker, motivated by the
coordinated-isolation gap (audit OOF MAP@5 .30 vs .56 for the other families; listed CI hands have median ranker
rank 7, recall@20 .865). The qualitative dump (step 16) shows listed isolation hands that look ordinary to the 382-feature
ranker: one partner OPENS the pot (often light, often over a limper), the other partner folds, and the opener plays the
pot against at most one victim. The ranker's own isolation events are 're-raise' events (v2_isolation_*), which is a
different shape. This step parses every hand of the 162 audit pairs (all held folds excluded; same audit set as steps
14-18, 45) straight from actions/seats, builds ~45 pair-relative structural features (who opened, over how many
limpers, what the partner did, how many others continued, isolation achieved, c-bet, outcomes, hole strengths,
positions), and asks:
  1. per family, which structural features separate listed from unlisted hands (all hands and within the top-20);
  2. MAP@5 of GroupKFold(5, by table) LightGBM rankers over ALL hands: structural only, structural + chronology,
     structural + chronology + the 382-feature OOF score, per family and pooled; against the audit OOF ranker (.490)
     and the step-14 score+chronology top-20 re-ranker (.545; CI .40);
  3. the same structural features added to the step-14 top-20 re-ranker protocol (apples to apples with 14.16-14.21).
Development labels: validation and model selection only. Output: experiments/label_process_20260911/audit-structural.json
and audit-structural-features.parquet."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step14_audit_truncation import OUT, TOP, CLF
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

RANKS = {r: i for i, r in enumerate('23456789TJQKA', start=2)}
STREET_IX = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
SEEDS = (11, 12, 13)


def log(*a):
    print(*a, flush=True)


def chen(c1, c2):
    if not c1 or not c2:
        return -1.0
    r1, r2 = RANKS[c1[0]], RANKS[c2[0]]
    hi, lo = max(r1, r2), min(r1, r2)
    base = {14: 10, 13: 8, 12: 7, 11: 6}.get(hi, hi / 2)
    if r1 == r2:
        return float(max(5, base * 2))
    s = base + (2 if c1[1] == c2[1] else 0)
    gap = hi - lo - 1
    s -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        s += 1
    return float(s)


def hand_features(acts, seats, p1, p2, bb, button, dealt):
    """acts: list of (street, player_id, action, amount, amount_to, to_call, players_active) in order; seats: dict player -> row."""
    role = {p1: 'A', p2: 'B'}
    f = {}
    nseat = max(r['seat_no'] for r in seats.values()) + 1
    for k, p in (('A', p1), ('B', p2)):
        r = seats[p]
        f[f'{k}_chen'] = chen(r['hole_card_1'], r['hole_card_2'])
        f[f'{k}_pos'] = float((r['seat_no'] - button) % nseat)
        f[f'{k}_stack_bb'] = r['starting_stack'] / bb
        f[f'{k}_net_bb'] = r['net_chips'] / bb
        f[f'{k}_put_bb'] = r['total_contribution'] / bb
        f[f'{k}_folded'] = float(r['folded'])
        f[f'{k}_sd'] = float(r['went_to_showdown'])
        f[f'{k}_won'] = float(r['won_share'] > 0)
    # symmetric versions (the pair is unordered)
    f['chen_max'], f['chen_min'] = max(f['A_chen'], f['B_chen']), min(f['A_chen'], f['B_chen'])
    f['pair_net_bb'] = f['A_net_bb'] + f['B_net_bb']
    f['pair_net_gap'] = abs(f['A_net_bb'] - f['B_net_bb'])
    f['pair_folded'] = f['A_folded'] + f['B_folded']
    f['pair_sd'] = f['A_sd'] + f['B_sd']
    f['pair_won'] = f['A_won'] + f['B_won']
    f['pos_gap'] = abs(f['A_pos'] - f['B_pos'])
    # preflop structure
    pf = [a for a in acts if a[0] == 'preflop']
    opener = None; opener_ix = None; limpers_before = 0; n_raises = 0; raisers = []
    for i, a in enumerate(pf):
        agg = a[2] in ('raise', 'bet') or (a[2] == 'all_in' and (a[3] or 0) > (a[5] or 0))
        if agg:
            n_raises += 1; raisers.append(role.get(a[1], 'o'))
            if opener is None:
                opener, opener_ix = role.get(a[1], 'o'), i
        elif opener is None and a[2] == 'call':
            limpers_before += 1
    f['pf_open_pair'] = float(opener in ('A', 'B'))
    f['pf_open_other'] = float(opener == 'o')
    f['pf_no_open'] = float(opener is None)
    f['pf_limpers_before_open'] = float(limpers_before) if opener is not None else -1.0
    f['pf_iso_raise'] = float(opener in ('A', 'B') and limpers_before >= 1)
    f['pf_n_raises'] = float(n_raises)
    f['pf_pair_raises'] = float(sum(r in ('A', 'B') for r in raisers))
    f['pf_other_raises'] = float(sum(r == 'o' for r in raisers))
    f['pf_pair_vs_pair_raise'] = float('A' in raisers and 'B' in raisers)
    f['pf_pair_3bet_over_other'] = float(opener == 'o' and any(r in ('A', 'B') for r in raisers[1:]))
    f['pf_other_3bet_over_pair'] = float(opener in ('A', 'B') and any(r == 'o' for r in raisers[1:]))
    opener_player = p1 if opener == 'A' else p2 if opener == 'B' else None
    partner = p2 if opener == 'A' else p1 if opener == 'B' else None
    f['pf_opener_chen'] = f['A_chen'] if opener == 'A' else f['B_chen'] if opener == 'B' else -1.0
    f['pf_partner_chen'] = f['B_chen'] if opener == 'A' else f['A_chen'] if opener == 'B' else -1.0
    f['pf_open_size_bb'] = (pf[opener_ix][4] or 0) / bb if opener_ix is not None else 0.0
    f['pf_opener_pos'] = f['A_pos'] if opener == 'A' else f['B_pos'] if opener == 'B' else -1.0
    # partner's response to a pair open, and the others' responses
    partner_resp = 'none'; partner_folded_before = 0.0; others_call = 0; others_fold = 0; others_raise = 0
    if opener_ix is not None:
        folded_before = {a[1] for a in pf[:opener_ix] if a[2] == 'fold'}
        if partner in folded_before:
            partner_folded_before = 1.0
        for a in pf[opener_ix + 1:]:
            r = role.get(a[1], 'o')
            agg = a[2] in ('raise', 'bet') or (a[2] == 'all_in' and (a[3] or 0) > (a[5] or 0))
            if a[1] == partner and partner_resp == 'none':
                partner_resp = 'raise' if agg else a[2]
            if r == 'o':
                if agg:
                    others_raise += 1
                elif a[2] in ('call', 'all_in'):
                    others_call += 1
                elif a[2] == 'fold':
                    others_fold += 1
    for k in ('fold', 'call', 'raise', 'none'):
        f[f'pf_partner_{k}'] = float(partner_resp == k)
    f['pf_partner_folded_before_open'] = partner_folded_before
    f['pf_others_call'] = float(others_call); f['pf_others_fold'] = float(others_fold); f['pf_others_raise'] = float(others_raise)
    folded_pf = {a[1] for a in pf if a[2] == 'fold'}
    post = [a for a in acts if a[0] != 'preflop']
    reached_flop = len(post) > 0
    f['ended_preflop'] = float(not reached_flop)
    f['max_street'] = float(max(STREET_IX[a[0]] for a in acts)) if acts else 0.0
    members_to_flop = float((p1 not in folded_pf) + (p2 not in folded_pf)) if reached_flop else 0.0
    n_to_flop = float(post[0][6]) if reached_flop else 0.0
    f['n_to_flop'] = n_to_flop; f['members_to_flop'] = members_to_flop; f['others_to_flop'] = max(n_to_flop - members_to_flop, 0.0) if reached_flop else 0.0
    f['iso_one_member_one_other'] = float(reached_flop and members_to_flop == 1 and n_to_flop == 2)
    f['iso_one_member_vs_others'] = float(reached_flop and members_to_flop == 1 and n_to_flop >= 2)
    f['hu_pair_only'] = float(reached_flop and members_to_flop == 2 and n_to_flop == 2)
    f['pair_won_preflop'] = float((not reached_flop) and f['pair_won'] > 0)
    f['opener_won'] = float(opener_player is not None and seats[opener_player]['won_share'] > 0)
    f['opener_sd'] = float(opener_player is not None and seats[opener_player]['went_to_showdown'])
    f['opener_net_bb'] = seats[opener_player]['net_chips'] / bb if opener_player is not None else 0.0
    # post-flop relational events
    cbet = 0.0; victim_fold_to_pair_bet = 0.0; partner_fold_to_pair_bet = 0.0; pair_aggr = 0; other_aggr = 0; pair_calls = 0; other_calls = 0
    last_aggr = None
    for a in post:
        r = role.get(a[1], 'o')
        agg = a[2] in ('raise', 'bet') or (a[2] == 'all_in' and (a[3] or 0) > (a[5] or 0))
        if agg:
            if r in ('A', 'B'):
                pair_aggr += 1
                if a[0] == 'flop' and a[1] == opener_player and cbet == 0.0 and last_aggr is None:
                    cbet = 1.0
            else:
                other_aggr += 1
            last_aggr = r
        elif a[2] in ('call', 'all_in'):
            if r in ('A', 'B'):
                pair_calls += 1
            else:
                other_calls += 1
        elif a[2] == 'fold':
            if r == 'o' and last_aggr in ('A', 'B'):
                victim_fold_to_pair_bet += 1
            if r in ('A', 'B') and last_aggr in ('A', 'B') and last_aggr != r:
                partner_fold_to_pair_bet += 1
    f['post_cbet'] = cbet; f['post_victim_fold_to_pair_bet'] = victim_fold_to_pair_bet; f['post_partner_fold_to_pair_bet'] = partner_fold_to_pair_bet
    f['post_pair_aggr'] = float(pair_aggr); f['post_other_aggr'] = float(other_aggr); f['post_pair_calls'] = float(pair_calls); f['post_other_calls'] = float(other_calls)
    f['pair_won_no_sd'] = float(f['pair_won'] > 0 and f['pair_sd'] == 0)
    f['n_actions'] = float(len(acts)); f['dealt'] = float(dealt)
    return f


def build(audit, pair_players=None):
    """audit: frame with pair_id, hand_id; pair_players: frame with pair_id, player_1, player_2 (default: development labels)."""
    t0 = time.time()
    labels = pair_players if pair_players is not None else pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
    pairs = dict(labels.filter(pl.col('pair_id').is_in(audit['pair_id'].unique().implode())).select('pair_id', pl.struct('player_1', 'player_2')).iter_rows())
    hids = audit['hand_id'].unique().implode()
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet', columns=['hand_id', 'big_blind', 'button_seat', 'players_dealt', 'final_pot']).filter(pl.col('hand_id').is_in(hids))
    hinfo = {r['hand_id']: r for r in hands.iter_rows(named=True)}
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet').filter(pl.col('hand_id').is_in(hids))
    srows = {}
    for r in seats.iter_rows(named=True):
        srows.setdefault(r['hand_id'], {})[r['player_id']] = r
    acts = pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').filter(pl.col('hand_id').is_in(hids)).sort('hand_id', 'action_no').collect()
    arows = {}
    for hid, st, pid, act, amt, amt_to, tc, pa in acts.select('hand_id', 'street', 'player_id', 'action', 'amount', 'amount_to', 'to_call', 'players_active').iter_rows():
        arows.setdefault(hid, []).append((st, pid, act, amt, amt_to, tc, pa))
    log('loaded', hands.height, 'hands', seats.height, 'seat rows', acts.height, 'actions', round(time.time() - t0))
    rows = []
    for pid, hid in audit.select('pair_id', 'hand_id').iter_rows():
        h = hinfo[hid]; pp = pairs[pid]
        f = hand_features(arows.get(hid, []), srows[hid], pp['player_1'], pp['player_2'], h['big_blind'], h['button_seat'], h['players_dealt'])
        f['pot_bb'] = h['final_pot'] / h['big_blind']
        f['pair_id'] = pid; f['hand_id'] = hid
        rows.append(f)
    feat = pl.DataFrame(rows)
    log('features', feat.shape, round(time.time() - t0))
    return feat


def oof_scores(X, y, groups, seeds=SEEDS):
    pred = np.zeros(len(y))
    for seed in seeds:
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = lgb.LGBMClassifier(**CLF, random_state=seed)
            clf.fit(X[tr], y[tr])
            pred[te] += clf.predict_proba(X[te])[:, 1] / len(seeds)
    return pred


def summarize(scored, truths, labels):
    m = per_pair_map(scored, truths)
    return {'map5': float(np.mean(list(m.values()))), 'by_family': {f: float(np.mean([v for p, v in m.items() if labels[p] == f])) for f in FAMILIES}}, m


def main():
    t0 = time.time()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    feat = build(audit)
    feat.write_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1')
    d = d.with_columns(*[(pl.col('behavior_family') == f).cast(pl.Float64).alias('family_' + f) for f in FAMILIES])
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    famcols = ['family_' + f for f in FAMILIES]
    chrono = ['chrono_rank', 'chrono_frac', 'n_hands']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    res = {'training_only': True, 'queries': len(truths), 'hands': d.height, 'n_structural': len(struct), 'seconds_build': round(time.time() - t0)}
    # 1. single-feature AUCs per family, all hands and within top-20
    res['auc'] = {}
    for fam in FAMILIES:
        g = d.filter(pl.col('behavior_family') == fam); g20 = g.filter(pl.col('oof_rank') <= TOP)
        y, y20 = g['relevance'].to_numpy(), g20['relevance'].to_numpy()
        res['auc'][fam] = {}
        for c in struct + ['oof', 'chrono_frac']:
            x, x20 = g[c].to_numpy().astype(float), g20[c].to_numpy().astype(float)
            a = roc_auc_score(y, x) if len(set(x)) > 1 else .5; a20 = roc_auc_score(y20, x20) if len(set(x20)) > 1 else .5
            res['auc'][fam][c] = {'all': round(float(a), 3), 'top20': round(float(a20), 3), 'listed_mean': round(float(x[y == 1].mean()), 3), 'unlisted_mean': round(float(x[y == 0].mean()), 3)}
    for fam in FAMILIES:
        top = sorted(res['auc'][fam].items(), key=lambda kv: -abs(kv[1]['all'] - .5))[:12]
        log(fam, [(k, v['all'], v['top20']) for k, v in top])
    # 2. all-hands rankers
    y = d['relevance'].to_numpy().astype(float); groups = d['table_id'].to_numpy()
    base, base_map = summarize(d.select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')), truths, labels)
    res['oof_ranker'] = base
    sets = {'struct': struct + famcols, 'struct_chrono': struct + chrono + famcols, 'struct_chrono_oof': struct + chrono + ['oof', 'oof_rank'] + famcols,
            'chrono_oof': chrono + ['oof', 'oof_rank'] + famcols}
    res['all_hands'] = {}
    for name, cols in sets.items():
        X = d.select(cols).to_numpy().astype(np.float64)
        pred = oof_scores(X, y, groups)
        s, m = summarize(d.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred)), truths, labels)
        s['delta_vs_oof'] = s['map5'] - base['map5']; s['delta_by_family'] = {f: s['by_family'][f] - base['by_family'][f] for f in FAMILIES}
        lst = d.filter(pl.col('relevance') == 1)
        rk = pl.Series(pred).rank('ordinal', descending=True)
        dd = d.with_columns(pl.Series('p', pred)).with_columns(pl.col('p').rank('ordinal', descending=True).over('pair_id').alias('prank'))
        s['listed_recall_at'] = {str(k): {f: round(float((dd.filter((pl.col('relevance') == 1) & (pl.col('behavior_family') == f))['prank'] <= k).mean()), 3) for f in FAMILIES} for k in (5, 10, 20)}
        res['all_hands'][name] = s
        log('all_hands', name, json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}}))
    # per-family models on all hands (family-specific structure), struct + chrono + oof
    res['per_family_models'] = {}
    pred = np.zeros(d.height)
    for fam in FAMILIES:
        mask = (d['behavior_family'] == fam).to_numpy()
        X = d.select(struct + chrono + ['oof', 'oof_rank']).to_numpy().astype(np.float64)[mask]
        pred[mask] = oof_scores(X, y[mask], groups[mask])
    s, m = summarize(d.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred)), truths, labels)
    s['delta_vs_oof'] = s['map5'] - base['map5']
    res['per_family_models']['struct_chrono_oof'] = s
    log('per_family struct_chrono_oof', json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}}))
    # 3. top-20 re-ranker protocol (step 14 5b) with structural features
    cands = d.filter(pl.col('oof_rank') <= TOP)
    tail = d.filter(pl.col('oof_rank') > TOP).select('pair_id', 'hand_id', 'table_id', (pl.col('oof') - 1e6).alias('score'))
    yc = cands['relevance'].to_numpy().astype(float); gc = cands['table_id'].to_numpy()
    res['rerank_top20'] = {}
    for name, cols in {'score_plus_chrono': ['oof', 'oof_rank'] + chrono + famcols, 'score_plus_chrono_plus_struct': ['oof', 'oof_rank'] + chrono + struct + famcols}.items():
        X = cands.select(cols).to_numpy().astype(np.float64)
        pred = oof_scores(X, yc, gc)
        head = cands.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred + 10.0))
        s, m = summarize(pl.concat([head, tail]), truths, labels)
        s['delta_vs_oof'] = s['map5'] - base['map5']
        s['auc_top20'] = float(roc_auc_score(yc, pred))
        res['rerank_top20'][name] = {**s, 'per_pair': m}
        log('rerank', name, json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}, 'auc_top20': round(s['auc_top20'], 4)}))
    # paired pool bootstrap: all-hands struct_chrono_oof vs the top-20 score+chrono re-ranker
    a = res['rerank_top20']['score_plus_chrono'].pop('per_pair'); b = res['rerank_top20']['score_plus_chrono_plus_struct'].pop('per_pair')
    pools = sorted(set(d['table_id'])); ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    rng = np.random.default_rng(65); deltas = []
    for _ in range(2000):
        pick = rng.choice(pools, len(pools), replace=True)
        vals = [b[p] - a[p] for t in pick for p in a if ptab[p] == t]
        deltas.append(np.mean(vals))
    res['bootstrap_rerank_struct_vs_chrono'] = {'mean_delta': float(np.mean([b[p] - a[p] for p in a])), 'ci95': [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
                                                'pairs_better': int(sum(b[p] > a[p] for p in a)), 'pairs_worse': int(sum(b[p] < a[p] for p in a))}
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-structural.json', res)
    log(json.dumps({k: res[k] for k in ['oof_ranker', 'all_hands', 'per_family_models', 'rerank_top20', 'bootstrap_rerank_struct_vs_chrono']}, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
