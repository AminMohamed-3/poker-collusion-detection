"""Step 18: training-only audit of raw action patterns in listed vs highest-scoring unlisted hands.

The competition says every planted evidence hand 'contains a behavior-specific action visible in
actions.parquet'. This script rebuilds simple relational action patterns between the two partners (A =
player_1, B = player_2) straight from the action log for the audit pairs' listed hands and their top-20
out-of-fold candidates, reports listed/unlisted rates per family, and checks whether the patterns add
re-ranking signal beyond score + chronology.
Output: Claude Version/experiments/label_process_20260911/audit-action-patterns.json
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
from evidence_aug.frame import family_columns, FAMILIES
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step14_audit_truncation import fmean, OUT, TOP, CLF, FAMCOLS
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

RANKS = {r: i for i, r in enumerate('23456789TJQKA', start=2)}
STREETS = ['preflop', 'flop', 'turn', 'river']


def chen(c1, c2):
    """Chen preflop strength (rough), 0..20."""
    r1, r2 = RANKS[c1[0]], RANKS[c2[0]]
    hi, lo = max(r1, r2), min(r1, r2)
    base = {14: 10, 13: 8, 12: 7, 11: 6}.get(hi, hi / 2)
    if r1 == r2:
        return max(5, base * 2)
    s = base + (2 if c1[1] == c2[1] else 0)
    gap = hi - lo - 1
    s -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        s += 1
    return s


def patterns_for_hand(acts, a, b, seat_a, seat_b):
    """acts: rows sorted by action_no for one hand. Returns dict of pattern features."""
    f = {}
    last_aggr = {}          # street -> player who made the last bet/raise/all_in-as-raise
    active_partner_hu = False
    for x in ('A', 'B'):
        for k in ('raises', 'bets', 'calls', 'checks', 'folds', 'fold_vs_partner', 'call_vs_partner', 'raise_vs_partner', 'check_hu_partner', 'bet_hu_partner',
                  'aggr_others_present', 'aggr_hu_partner', 'fold_street', 'first_raise_pre', 'threebet_pre', 'max_tocall_from_partner_bb', 'aggr_amount_bb'):
            f[f'{x}_{k}'] = 0.0
    who = {a: 'A', b: 'B'}
    bb = None
    for r in acts:
        st, pid, act, amt, to_call, pa, pot = r['street'], r['player_id'], r['action'], r['amount'], r['to_call'], r['players_active'], r['pot_before']
        if bb is None:
            bb = r['bb']
        aggressive = act in ('bet', 'raise') or (act == 'all_in' and amt > to_call)
        x = who.get(pid)
        partner_last = last_aggr.get(st) in (a, b) and last_aggr.get(st) != pid
        hu = pa == 2 and r['a_in'] and r['b_in']
        if hu:
            active_partner_hu = True
        if x is not None:
            if act == 'raise':
                f[f'{x}_raises'] += 1
            if act == 'bet':
                f[f'{x}_bets'] += 1
            if act == 'call' or (act == 'all_in' and amt <= to_call):
                f[f'{x}_calls'] += 1
                if partner_last:
                    f[f'{x}_call_vs_partner'] += 1
            if act == 'check':
                f[f'{x}_checks'] += 1
                if hu:
                    f[f'{x}_check_hu_partner'] += 1
            if act == 'fold':
                f[f'{x}_folds'] = 1.0
                f[f'{x}_fold_street'] = float(STREETS.index(st) + 1)
                if partner_last:
                    f[f'{x}_fold_vs_partner'] = 1.0
                    f[f'{x}_max_tocall_from_partner_bb'] = max(f[f'{x}_max_tocall_from_partner_bb'], to_call / bb)
            if aggressive:
                f[f'{x}_aggr_amount_bb'] = max(f[f'{x}_aggr_amount_bb'], amt / bb)
                if partner_last:
                    f[f'{x}_raise_vs_partner'] += 1
                if pa >= 3:
                    f[f'{x}_aggr_others_present'] += 1
                if hu:
                    f[f'{x}_aggr_hu_partner'] += 1
                    if act == 'bet':
                        f[f'{x}_bet_hu_partner'] += 1
                if st == 'preflop':
                    if last_aggr.get('preflop') is None:
                        f[f'{x}_first_raise_pre'] = 1.0
                    elif last_aggr.get('preflop') not in (a, b):
                        f[f'{x}_threebet_pre'] += 1
        if aggressive:
            last_aggr[st] = pid
    f['hu_partners_any'] = float(active_partner_hu)
    f['both_raised_pre'] = float(f['A_first_raise_pre'] + f['A_threebet_pre'] > 0 and f['B_first_raise_pre'] + f['B_threebet_pre'] > 0)
    f['partner_vs_partner_aggr'] = f['A_raise_vs_partner'] + f['B_raise_vs_partner']
    f['any_fold_vs_partner'] = float(f['A_fold_vs_partner'] + f['B_fold_vs_partner'] > 0)
    f['any_check_hu_partner'] = float(f['A_check_hu_partner'] + f['B_check_hu_partner'] > 0)
    f['any_aggr_others_present'] = float(f['A_aggr_others_present'] + f['B_aggr_others_present'] > 0)
    return f


def main():
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
    cands = audit.filter((pl.col('oof_rank') <= TOP) | (pl.col('relevance') == 1)).join(labels, on='pair_id')
    hand_ids = cands.select('hand_id').unique()
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet', columns=['hand_id', 'big_blind', 'final_pot', 'players_at_showdown']).join(hand_ids, on='hand_id')
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet').join(hand_ids, on='hand_id')
    acts = pl.read_parquet(ORIGINAL / 'data' / 'actions.parquet').join(hand_ids, on='hand_id').sort('hand_id', 'action_no')
    # fold status per action: a player is 'in' before acting on this action if they have not folded earlier in the hand
    acts = acts.join(hands.select('hand_id', pl.col('big_blind').alias('bb')), on='hand_id')
    folded_before = acts.with_columns((pl.col('action') == 'fold').cast(pl.Int64).cum_sum().over('hand_id', 'player_id').alias('fcum'))
    folded_before = folded_before.with_columns((pl.col('fcum') - (pl.col('action') == 'fold').cast(pl.Int64)).alias('folded_before'))
    rows = []
    seat_map = {(r['hand_id'], r['player_id']): r for r in seats.iter_rows(named=True)}
    act_groups = {k[0]: g for k, g in folded_before.group_by('hand_id')}
    for c in cands.iter_rows(named=True):
        hid, a, b = c['hand_id'], c['player_1'], c['player_2']
        g = act_groups[hid].sort('action_no')
        # a_in / b_in before each action
        fa = {}; fb = {}
        a_folded = False; b_folded = False
        recs = []
        for r in g.iter_rows(named=True):
            r['a_in'] = not a_folded; r['b_in'] = not b_folded
            recs.append(r)
            if r['action'] == 'fold' and r['player_id'] == a:
                a_folded = True
            if r['action'] == 'fold' and r['player_id'] == b:
                b_folded = True
        sa, sb = seat_map[(hid, a)], seat_map[(hid, b)]
        f = patterns_for_hand(recs, a, b, sa['seat_no'], sb['seat_no'])
        bb = hands.filter(pl.col('hand_id') == hid)['big_blind'][0]
        f['A_chen'] = chen(sa['hole_card_1'], sa['hole_card_2']); f['B_chen'] = chen(sb['hole_card_1'], sb['hole_card_2'])
        f['chen_max'] = max(f['A_chen'], f['B_chen']); f['chen_min'] = min(f['A_chen'], f['B_chen'])
        f['folder_chen'] = f['A_chen'] if f['A_fold_vs_partner'] else (f['B_chen'] if f['B_fold_vs_partner'] else -1.0)
        f['A_net_bb'] = sa['net_chips'] / bb; f['B_net_bb'] = sb['net_chips'] / bb
        f['transfer_bb'] = min(max(-f['A_net_bb'], 0.0), max(f['B_net_bb'], 0.0)) + min(max(-f['B_net_bb'], 0.0), max(f['A_net_bb'], 0.0))
        f['both_showdown'] = float(sa['went_to_showdown'] and sb['went_to_showdown'])
        f['pot_bb'] = hands.filter(pl.col('hand_id') == hid)['final_pot'][0] / bb
        f['A_put_bb'] = sa['total_contribution'] / bb; f['B_put_bb'] = sb['total_contribution'] / bb
        rows.append({'pair_id': c['pair_id'], 'hand_id': hid, **f})
    pat = pl.DataFrame(rows)
    cands = cands.join(pat, on=['pair_id', 'hand_id'])
    pcols = [c for c in pat.columns if c not in ('pair_id', 'hand_id')]
    out = {'training_only': True, 'queries': int(cands['pair_id'].n_unique()), 'rows': cands.height}
    top = cands.filter(pl.col('oof_rank') <= TOP)
    out['pattern_means_by_family'] = {}
    for fam in FAMILIES:
        g = top.filter(pl.col('behavior_family') == fam)
        L, U = g.filter(pl.col('relevance') == 1), g.filter(pl.col('relevance') == 0)
        out['pattern_means_by_family'][fam] = {c: {'listed': round(fmean(L[c]), 3), 'unlisted_top20': round(fmean(U[c]), 3)} for c in pcols}
    # single-feature AUCs within top-20 (listed vs unlisted), overall and per family
    y = top['relevance'].to_numpy()
    out['top20_auc_single'] = {c: round(float(roc_auc_score(y, top[c].to_numpy())), 3) for c in pcols}
    out['top20_auc_single_by_family'] = {fam: {c: round(float(roc_auc_score(top.filter(pl.col('behavior_family') == fam)['relevance'].to_numpy(), top.filter(pl.col('behavior_family') == fam)[c].to_numpy())), 3)
                                               for c in pcols if top.filter(pl.col('behavior_family') == fam)[c].n_unique() > 1} for fam in FAMILIES}
    # re-rankers
    truths = {k[0]: set(g['hand_id']) for k, g in audit.filter(pl.col('relevance') == 1).group_by('pair_id')}
    lab = dict(audit.select('pair_id', 'behavior_family').unique().iter_rows())
    fam_top = family_columns(top.sort('pair_id', 'hand_id'), 'behavior_family')
    y = fam_top['relevance'].to_numpy(); groups = fam_top['table_id'].to_numpy()
    base_map = per_pair_map(audit.select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')), truths)
    base = fmean(list(base_map.values()))
    tail = audit.filter(pl.col('oof_rank') > TOP).select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score'))
    S = ['oof', 'oof_rank']; CH = ['chrono_rank', 'chrono_frac', 'n_hands']
    sets = {'score_only': S + FAMCOLS, 'chrono': S + CH + FAMCOLS, 'chrono_plus_patterns': S + CH + pcols + FAMCOLS, 'patterns_only_plus_score': S + pcols + FAMCOLS}
    out['rerank_top20_map5'] = {}
    for name, cols in sets.items():
        X = np.nan_to_num(fam_top.select(cols).to_numpy().astype(np.float64))
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = lgb.LGBMClassifier(**CLF, random_state=11)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        head = fam_top.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred + 10.0))
        m = per_pair_map(pl.concat([head, tail.with_columns(pl.col('score') - 1e6)]), truths)
        s = {'map5': fmean(list(m.values())), 'delta_vs_oof': fmean(list(m.values())) - base, 'auc_top20': float(roc_auc_score(y, pred)), 'by_family': {f: fmean([v for p, v in m.items() if lab[p] == f]) for f in FAMILIES}}
        out['rerank_top20_map5'][name] = s
        print(json.dumps({'rerank': name, 'map5': round(s['map5'], 4), 'delta': round(s['delta_vs_oof'], 4), 'auc': round(s['auc_top20'], 4), 'fam': {k: round(v, 3) for k, v in s['by_family'].items()}}), flush=True)
    write_json(OUT / 'audit-action-patterns.json', out)
    print('AUC overall:', sorted(out['top20_auc_single'].items(), key=lambda kv: -abs(kv[1] - .5))[:20])
    for fam in FAMILIES:
        print(fam, sorted(out['top20_auc_single_by_family'][fam].items(), key=lambda kv: -abs(kv[1] - .5))[:12])


if __name__ == '__main__':
    main()
