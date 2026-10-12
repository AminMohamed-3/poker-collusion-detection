"""Step 48 (label-free, evaluation side): spill-over re-scoring of the v13 block.

Observation (OBSERVATIONS 14.30): 42 % of the 909 evaluation pairs with v13 risk >= .5 share a player with a HIGHER-risk
block pair ("dominated"), and the dominating pair's other player is dealt in .45 of the dominated pair's five evidence
hands (half of them in >= 3 of 5), against a chance rate of 4/28 = .14. On development, ring positives (labelled positives
dominated by another labelled positive) show their third player in the truth evidence hands at the chance rate only.
Hypothesis: many dominated block pairs (A,C) are spill-over false positives: their suspicious hands are (A,B)'s
manipulated hands in which C happened to be dealt.

Test (uses only our own risk, the evidence lists, and who was dealt): re-score every block pair with the production
pair model on subsets of its shared hands:
  noT        : hands in which the third player T is NOT dealt (T = dominating partner for dominated pairs, the most
               co-present third player of the five listed evidence hands otherwise);
  rand       : the same number of hands dropped at random (3 draws)  -> content-free control;
  evdrop     : the five listed evidence hands dropped, random fill to the same count (3 draws) -> "lost its listed hands" control;
  noT_keepEv : T's hands dropped except the five listed evidence hands -> is the signal only in the listed hands?
A spill-over pair collapses under noT and not under rand; a genuine pair with many manipulated hands survives evdrop.
Parity: the full-hand re-score must equal the v13 risk exactly. Per-hand conditional features exist in the inference
package only for its 10,000 screen-selected pairs; the rest are recomputed with the production policy (parity covers them).

Inputs: outputs_v2/hands_evaluation, inference-package/results/batch_*-evidence.parquet, inference-package/{final-models.joblib,
style-counts.parquet}, outputs_v11_conditional/pair-model.joblib, outputs_v13_coverage/submission_conditional_coverage.csv
(protected, read only). No labels exist on this side.
Outputs: Claude Version/experiments/spillover_20260911/eval_rescoring.parquet, eval_rescoring.json."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json, check_protected
import numpy as np
import polars as pl
import joblib
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import aggregate, matrix  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402
from evaluate_conditional_action_screen import extra_aggregate  # noqa: E402

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'spillover_20260911'
SUB = ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv'
INF = ORIGINAL / 'outputs_v11_conditional' / 'inference-package' / 'results'
RAW = ORIGINAL / 'outputs_v2' / 'hands_evaluation'
MODEL = ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib'
THR = .5
SEEDS = [1, 2, 3]
EV = [f'evidence_hand_{i}' for i in range(1, 6)]


def dominance(block):
    """pair_id -> (risk, pair_id, other_player) of the highest-risk OTHER block pair sharing a player, if its risk is higher."""
    best = {}
    for pid, p1, p2, r in block.select('pair_id', 'player_1', 'player_2', 'risk').iter_rows():
        for a, b in ((p1, p2), (p2, p1)):
            if r > best.get(a, (0, None, None))[0]:
                best[a] = (r, pid, b)
    dom = {}
    for pid, p1, p2, r in block.select('pair_id', 'player_1', 'player_2', 'risk').iter_rows():
        cand = [best[a] for a in (p1, p2) if a in best and best[a][0] > r + 1e-12]
        if cand:
            dom[pid] = max(cand)
    return dom


def band(r):
    return '[.99,1]' if r >= .99 else '[.9,.99)' if r >= .9 else '[.5,.9)'


def nanstat(fn, xs):
    xs = [x for x in xs if x == x]
    return float(fn(xs)) if xs else float('nan')


def make_scorer(model, features, names, events, caf):
    def score(raw_sub, ca_sub):
        frame = aggregate(raw_sub, names).join(tails(raw_sub, events), on=['pair_id', 'table_id']).join(extra_aggregate(ca_sub, caf), on=['pair_id', 'table_id'])
        risk = 1 - model.predict_proba(matrix(frame, features))[:, 0]
        pids = frame['pair_id'].to_list()
        return dict(zip(pids, risk.tolist())), dict(zip(pids, frame['shared_hands'].to_list()))
    return score


def variants(raw, ca, score, ids, T_of, ev_of, dealt, seed_base=0):
    """Re-score under noT, rand x3, evdrop x3, noT_keepEv. ev_of[pid] = listed hands in rank order (may be empty)."""
    if set(ids) != set(raw['pair_id'].unique()):
        raw = raw.filter(pl.col('pair_id').is_in(ids)); ca = ca.filter(pl.col('pair_id').is_in(ids))
    hand_pid = raw.select('pair_id', 'hand_id')
    pairs_arr = hand_pid['pair_id'].to_list(); hands_arr = hand_pid['hand_id'].to_list()
    ev_set = {p: set(ev_of.get(p, [])) for p in ids}
    isT = np.array([T_of[p] in dealt[h] for p, h in zip(pairs_arr, hands_arr)])
    isEv = np.array([h in ev_set[p] for p, h in zip(pairs_arr, hands_arr)])
    full, n_full = score(raw, ca)
    noT, n_noT = score(raw.filter(pl.Series(~isT)), ca.filter(pl.Series(~isT)))
    keep_k = ~isT | isEv
    noTk, n_noTk = score(raw.filter(pl.Series(keep_k)), ca.filter(pl.Series(keep_k)))
    n_drop = {p: n_full[p] - n_noT.get(p, 0) for p in ids}
    rows_of, ev_rows_of, nonev_rows_of = defaultdict(list), defaultdict(dict), defaultdict(list)
    for i, (p, h) in enumerate(zip(pairs_arr, hands_arr)):
        rows_of[p].append(i)
        if h in ev_set[p]:
            ev_rows_of[p][h] = i
        else:
            nonev_rows_of[p].append(i)
    rand_runs, ev_runs = [], []
    for seed in SEEDS:
        rs = np.random.default_rng(seed + seed_base)
        keep = np.ones(raw.height, dtype=bool)
        for p, rows in rows_of.items():
            d = n_drop[p]
            if d > 0:
                keep[rs.choice(rows, size=d, replace=False)] = False
        rand_runs.append(score(raw.filter(pl.Series(keep)), ca.filter(pl.Series(keep)))[0])
        rs = np.random.default_rng(seed + seed_base + 1000)
        keep = np.ones(raw.height, dtype=bool)
        for p, rows in rows_of.items():
            d = n_drop[p]
            if d <= 0:
                continue
            listed = [ev_rows_of[p][h] for h in ev_of.get(p, []) if h in ev_rows_of[p]]
            take = listed[:d]
            fill = d - len(take)
            if fill > 0 and nonev_rows_of[p]:
                take = take + list(rs.choice(nonev_rows_of[p], size=min(fill, len(nonev_rows_of[p])), replace=False))
            keep[take] = False
        ev_runs.append(score(raw.filter(pl.Series(keep)), ca.filter(pl.Series(keep)))[0])
    share_T = {p: float(np.mean(isT[rows_of[p]])) for p in ids}
    out = {}
    for p in ids:
        rr = [run.get(p, float('nan')) for run in rand_runs]; ee = [run.get(p, float('nan')) for run in ev_runs]
        out[p] = {'risk_full': full[p], 'n_hands': n_full[p], 'n_noT': n_noT.get(p, 0), 'n_dropped': n_drop[p], 'share_hands_with_T': share_T[p],
                  'risk_noT': noT.get(p, float('nan')), 'risk_noT_keepEv': noTk.get(p, float('nan')), 'n_noT_keepEv': n_noTk.get(p, 0),
                  'risk_rand_mean': nanstat(np.mean, rr), 'risk_rand_min': nanstat(np.min, rr), 'risk_rand_max': nanstat(np.max, rr),
                  'risk_evdrop_mean': nanstat(np.mean, ee), 'risk_evdrop_min': nanstat(np.min, ee), 'risk_evdrop_max': nanstat(np.max, ee)}
    return out


def add_flags(df, risk_col):
    return df.with_columns(pl.col(risk_col).map_elements(band, return_dtype=pl.String).alias('band'),
                           (pl.col('risk_noT') - pl.col('risk_rand_mean')).alias('delta_noT_vs_rand'),
                           ((pl.col('risk_noT').fill_nan(0) < .1) & (pl.col('risk_rand_mean') >= .5)).alias('collapse'),
                           ((pl.col('risk_noT').fill_nan(0) < .1) & (pl.col('risk_evdrop_mean') >= .5)).alias('collapse_vs_evdrop'),
                           ((pl.col('risk_noT').fill_nan(0) < .5) & (pl.col('risk_rand_mean') >= .5)).alias('drops_below_block'))


def group_summary(g):
    return {'n': g.height, 'n_hands_median': float(g['n_hands'].median()), 'n_dropped_median': float(g['n_dropped'].median()),
            'share_hands_with_T_mean': round(float(g['share_hands_with_T'].mean()), 3),
            'risk_noT_median': round(float(g['risk_noT'].fill_nan(0).median()), 4), 'risk_rand_median': round(float(g['risk_rand_mean'].median()), 4),
            'risk_evdrop_median': round(float(g['risk_evdrop_mean'].median()), 4), 'risk_noT_keepEv_median': round(float(g['risk_noT_keepEv'].fill_nan(0).median()), 4),
            'delta_noT_vs_rand_q10_25_50_75_90': [round(float(x), 4) for x in np.nanquantile(g['delta_noT_vs_rand'].to_numpy(), [.1, .25, .5, .75, .9])],
            'share_collapse_noT_lt_.1_rand_ge_.5': round(float(g['collapse'].mean()), 3), 'share_collapse_vs_evdrop': round(float(g['collapse_vs_evdrop'].mean()), 3),
            'share_drops_below_block': round(float(g['drops_below_block'].mean()), 3),
            'share_rand_stays_ge_.5': round(float((g['risk_rand_mean'] >= .5).mean()), 3), 'share_evdrop_stays_ge_.5': round(float((g['risk_evdrop_mean'] >= .5).mean()), 3),
            'share_noT_stays_ge_.5': round(float((g['risk_noT'].fill_nan(0) >= .5).mean()), 3), 'share_noT_keepEv_stays_ge_.5': round(float((g['risk_noT_keepEv'].fill_nan(0) >= .5).mean()), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    check_protected()
    sub = pl.read_csv(SUB)
    ep = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv')
    block = ep.join(sub.select('pair_id', pl.col('risk_score').alias('risk'), *EV), on='pair_id').filter(pl.col('risk') >= THR)
    ids = sorted(block['pair_id'])
    dom = dominance(block)
    raw = pl.concat([pl.read_parquet(b).filter(pl.col('pair_id').is_in(ids)) for b in sorted(RAW.glob('batch_*.parquet'))]).sort('pair_id', 'hand_id')
    ca = pl.concat([pl.read_parquet(b).filter(pl.col('pair_id').is_in(ids)) for b in sorted(INF.glob('batch_*-evidence.parquet'))]).sort('pair_id', 'hand_id')
    caf = [c for c in ca.columns if c.startswith('ca_')]
    missing = sorted(set(ids) - set(ca['pair_id']))
    if missing:
        # production stored per-hand conditional features only for its 10,000 screen-selected pairs; recompute the rest with the production policy
        from pu_pairs.build_allpairs_fold import conditional_features
        policy = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'inference-package' / 'final-models.joblib')
        counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(ORIGINAL / 'outputs_v11_conditional' / 'inference-package' / 'style-counts.parquet').iter_rows()}
        hands_ev = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter(pl.col('phase') == 'evaluation').select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind')
        keys = raw.filter(pl.col('pair_id').is_in(missing)).select('pair_id', 'hand_id', 'table_id').join(block.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        parts = []
        for t in sorted(set(keys['table_id'])):
            part, _, _ = conditional_features(keys.filter(pl.col('table_id') == t), hands_ev.filter(pl.col('table_id') == t), policy, counts, caf)
            parts.append(part.select(ca.columns))
        ca = pl.concat([ca] + parts, how='vertical_relaxed').sort('pair_id', 'hand_id')
        print(json.dumps({'recomputed_conditional_rows_for_pairs': len(missing), 'rows': sum(p.height for p in parts), 'seconds': round(time.time() - t0)}), flush=True)
    assert set(raw['pair_id']) == set(ids) == set(ca['pair_id']), (len(set(raw['pair_id'])), len(ids), len(set(ca['pair_id'])))
    assert raw.select('pair_id', 'hand_id').equals(ca.select('pair_id', 'hand_id')), 'per-hand keys differ between raw and conditional frames'
    raw.write_parquet(OUT / 'eval_block_raw.parquet'); ca.write_parquet(OUT / 'eval_block_ca.parquet')
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    events = [c for c in names if c.startswith('v2_')]
    bundle = joblib.load(MODEL); model = bundle['model']; features = list(bundle['features'])
    score = make_scorer(model, features, names, events, caf)
    hands = sorted(set(raw['hand_id']))
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet', columns=['hand_id', 'player_id']).filter(pl.col('hand_id').is_in(hands))
    dealt = defaultdict(set)
    for h, p in seats.iter_rows():
        dealt[h].add(p)
    tab = dict(raw.select('pair_id', 'table_id').unique().iter_rows())
    tab_players = defaultdict(set)
    for pid, hs in raw.group_by('pair_id').agg(pl.col('hand_id')).iter_rows():
        for h in hs:
            tab_players[tab[pid]].update(dealt[h])

    full, _ = score(raw, ca)
    v13 = dict(block.select('pair_id', 'risk').iter_rows())
    diff = np.array([abs(full[p] - v13[p]) for p in ids])
    parity = {'max_abs_diff': float(diff.max()), 'n_gt_1e-6': int((diff > 1e-6).sum())}
    print(json.dumps({'parity': parity, 'block': len(ids), 'dominated': len(dom), 'pair_hands': raw.height, 'seconds': round(time.time() - t0)}), flush=True)
    assert parity['max_abs_diff'] < 1e-5, parity

    rng = np.random.default_rng(4801)
    info, T_of, ev_of = {}, {}, {}
    for pid, p1, p2, r, *evh in block.select('pair_id', 'player_1', 'player_2', 'risk', *EV).iter_rows():
        ev_of[pid] = list(evh)
        counts = defaultdict(int)
        for h in evh:
            for q in dealt[h] - {p1, p2}:
                counts[q] += 1
        top = max(counts.items(), key=lambda kv: (kv[1], kv[0])) if counts else (None, 0)
        if pid in dom:
            T, kind = dom[pid][2], 'dominator_partner'
        else:
            T, kind = top[0], 'max_copresent'
        T_of[pid] = T
        others = sorted(tab_players[tab[pid]] - {p1, p2})
        rp = others[rng.integers(len(others))]
        info[pid] = {'T': T, 'T_kind': kind, 'cop_T': float(np.mean([T in dealt[h] for h in evh])), 'max_cop_third': top[1] / 5, 'max_cop_player': top[0],
                     'cop_random_player': float(np.mean([rp in dealt[h] for h in evh])), 'dominated': pid in dom,
                     'dom_risk': dom[pid][0] if pid in dom else None, 'dom_pair': dom[pid][1] if pid in dom else None,
                     'dom_partner_is_max_cop': bool(pid in dom and dom[pid][2] == top[0])}
    var = variants(raw, ca, score, ids, T_of, ev_of, dealt)
    print(json.dumps({'variants_done': True, 'seconds': round(time.time() - t0)}), flush=True)
    df = pl.DataFrame([{'pair_id': p, 'table_id': tab[p], 'risk_v13': v13[p], **var[p], **info[p]} for p in ids])
    df = add_flags(df, 'risk_v13')
    df.write_parquet(OUT / 'eval_rescoring.parquet')

    summ = {'parity': parity, 'block': len(ids), 'dominated': len(dom), 'seconds': round(time.time() - t0),
            'cop_random_player_mean': round(float(df['cop_random_player'].mean()), 4), 'share_random_player_ge_3of5': round(float((df['cop_random_player'] >= .6).mean()), 4),
            'dominator_partner_is_max_copresent_share': round(float(df.filter(pl.col('dominated'))['dom_partner_is_max_cop'].mean()), 3)}
    groups = {}
    for dmn in (True, False):
        for b in ('[.5,.9)', '[.9,.99)', '[.99,1]'):
            g = df.filter((pl.col('dominated') == dmn) & (pl.col('band') == b))
            if g.height:
                groups[f'dominated={dmn} band={b}'] = {'cop_T_mean': round(float(g['cop_T'].mean()), 3), 'share_cop_T_ge_3of5': round(float((g['cop_T'] >= .6).mean()), 3), **group_summary(g)}
    summ['groups'] = groups
    by_cop = {}
    for lo, hi in ((0, .2), (.2, .6), (.6, 1.01)):
        g = df.filter(pl.col('dominated') & (pl.col('cop_T') >= lo) & (pl.col('cop_T') < hi))
        if g.height:
            by_cop[f'dominated cop_T in [{lo},{hi})'] = group_summary(g)
    summ['dominated_by_cop_T'] = by_cop
    # label-free candidate rules: how many pairs each would demote (risk_new = risk_noT)
    rules = {}
    for name, cond in [('R1 dominated & risk_noT<.1', pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .1)),
                       ('R1b dominated & risk_noT<.5', pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .5)),
                       ('R2 dominated & cop_T>=.6 & risk_noT<.1', pl.col('dominated') & (pl.col('cop_T') >= .6) & (pl.col('risk_noT').fill_nan(0) < .1)),
                       ('R3 dominated & risk_noT<.1 & risk_evdrop>=.5', pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .1) & (pl.col('risk_evdrop_mean') >= .5))]:
        g = df.filter(cond)
        rules[name] = {'n': g.height, 'by_band': g.group_by('band').len().sort('band').to_dicts(), 'block_after': len(ids) - g.height}
    summ['candidate_rules_eval_counts'] = rules
    write_json(OUT / 'eval_rescoring.json', summ)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()