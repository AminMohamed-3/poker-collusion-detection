"""STEP258 step 4: the six pooled per-pair aggregates of the event probability, for every testbed pair.

Aggregates = evidence_aug/step254_D2_pair.pair_aggs verbatim (count above the honest q99.9, count
above the honest q99, top-1 / top-3 / top-5 mean logit of p_fmax, noisy-or), with each arm's own
stored honest thresholds.

Computed over
  ALL   the pair's prefix-60 hands                                  -> <arm>6
  noT   the prefix-60 hands in which the third player T is NOT dealt -> MECH6noT (mechplus only)
T is the step-48 / step-61 definition already used for this testbed: for a dominated pair the
dominator's other player, otherwise the third player most often co-present with the pair, ties broken
by the larger player_id (pu_pairs/step61_prefix_rescoring.run_fold verbatim, per fold's prefix block,
folds processed 1 then 2 and duplicates dropped keeping the first, as step61 does).
The reconstruction is gated against the stored T_kind / share_hands_with_T / n_noT columns of
experiments/viewdis_20260911/prefix_rescoring.parquet.

Parity gate: on the 149 labelled testbed pairs at FULL development history the six MECHPLUS
aggregates computed here must equal the six computed from the stored mechplus_p_fmax column of
experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet.

Writes aggregates258.parquet + aggregates258_receipt.json.  Development phase only.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
from collections import defaultdict
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step48_spillover_rescoring import dominance

OUT = C + 'experiments/step258_mech_block_view_20260918/'
S254 = C + 'experiments/step254_mechanical_tensor_20260917/'
VD = C + 'experiments/viewdis_20260911/'
EXP = C + 'experiments/pu_pairs_20260911/'
AGG = ['cnt_q999', 'cnt_q99', 'top1_logit', 'top3_logit', 'top5_logit', 'noisy_or']
EMPTY = np.array([0., 0., -13.815510557964274, -13.815510557964274, -13.815510557964274, 0.], np.float32)
ARMS = ('mechplus', 'perm', 'plus')


def log(*a):
    print(*a, flush=True)


def pair_aggs(p, q99, q999):
    """step254_D2_pair.pair_aggs verbatim."""
    if len(p) == 0:
        return EMPTY.copy()
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    s = np.sort(lg)[::-1]
    return np.array([float((p > q999).sum()), float((p > q99).sum()),
                     float(s[:1].mean()), float(s[:3].mean()), float(s[:5].mean()),
                     float(-np.log1p(-np.clip(p, 0, 1 - 1e-9)).sum())], dtype=np.float32)


def build_T(tb):
    """T per testbed pair, exactly as step61 computed it (per fold's prefix block, fold 1 first)."""
    prefix_pairs = pl.read_parquet(OUT + 'scores_mechplus_prefix.parquet').select('pair_id', 'hand_id')
    hands_of = {p: h for p, h in prefix_pairs.group_by('pair_id').agg(pl.col('hand_id')).iter_rows()}
    allh = prefix_pairs.select('hand_id').unique()
    seats = (pl.scan_parquet(DATA + 'seats.parquet').select('hand_id', 'player_id')
             .join(allh.lazy(), on='hand_id').collect())
    dealt = defaultdict(set)
    for h_, p_ in seats.iter_rows():
        dealt[h_].add(p_)
    T_of, kind_of = {}, {}
    for fold in (1, 2):
        scores = pl.read_parquet(EXP + f'fold{fold}-prefix60/scores.parquet')
        block = scores.filter(pl.col('risk') >= .5).sort('pair_id')
        dom = dominance(block)
        for pid, p1, p2 in block.select('pair_id', 'player_1', 'player_2').iter_rows():
            if pid in T_of:                       # step61: unique(keep='first'), fold 1 wins
                continue
            if pid not in hands_of:
                continue
            if pid in dom:
                T_of[pid], kind_of[pid] = dom[pid][2], 'dominator_partner'
            else:
                cnt = defaultdict(int)
                for h_ in hands_of[pid]:
                    for q in dealt[h_] - {p1, p2}:
                        cnt[q] += 1
                T_of[pid] = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
                kind_of[pid] = 'max_copresent'
    ids = tb['pair_id'].to_list()
    assert all(p in T_of for p in ids), [p for p in ids if p not in T_of][:5]
    # gate against the stored columns
    share, nnoT = {}, {}
    for p in ids:
        hs = hands_of[p]
        isT = np.array([T_of[p] in dealt[h] for h in hs])
        share[p] = float(isT.mean())
        nnoT[p] = int((~isT).sum())
    g_kind = int(sum(kind_of[p] != k for p, k in zip(ids, tb['T_kind'].to_list())))
    g_share = float(max(abs(share[p] - s) for p, s in zip(ids, tb['share_hands_with_T'].to_list())))
    g_n = int(sum(nnoT[p] != int(n) for p, n in zip(ids, tb['n_noT'].to_list())))
    gate = {'T_kind_mismatches': g_kind, 'share_hands_with_T_max_abs_diff': g_share,
            'n_noT_mismatches': g_n}
    log('T reconstruction gate', json.dumps(gate))
    assert g_kind == 0 and g_share < 1e-12 and g_n == 0, gate
    isT_row = {}
    for p in ids:
        isT_row[p] = np.array([T_of[p] in dealt[h] for h in hands_of[p]], bool)
    return T_of, kind_of, gate, hands_of, dealt


def grouped(scores):
    """one group_by so the hand_id list and the p_fmax list share the same within-pair order."""
    g = scores.group_by('pair_id').agg(pl.col('hand_id'), pl.col('p_fmax'))
    return {p: (h, np.asarray(v, dtype=np.float64)) for p, h, v in g.iter_rows()}


def aggs_for(grp, ids, thr, subset=None):
    """subset: pair_id -> boolean mask over that pair's rows in the grouped order."""
    q99, q999 = thr['honest_q99'], thr['honest_q999']
    A = np.zeros((len(ids), 6), np.float32)
    for i, p in enumerate(ids):
        v = grp[p][1]
        if subset is not None:
            v = v[subset[p]]
        A[i] = pair_aggs(v, q99, q999)
    return A


def main():
    t0 = time.time()
    tb = pl.read_parquet(OUT + 'testbed32.parquet')
    ids = tb['pair_id'].to_list()
    rec = {'n_pairs': len(ids)}
    cols = {}

    T_of, kind_of, gate, hands_of, dealt = build_T(tb)
    rec['T_gate'] = gate
    rec['T_kind_counts'] = {k: int(sum(1 for p in ids if kind_of[p] == k))
                            for k in ('dominator_partner', 'max_copresent')}

    for arm in ARMS:
        path = OUT + f'scores_{arm}_prefix.parquet'
        if not os.path.exists(path):
            log('MISSING', path, '- arm skipped')
            continue
        sc = pl.read_parquet(path)
        thr = json.load(open(OUT + f'score_receipt_{arm}.json', encoding='utf-8'))['honest_thresholds_stored']
        grp = grouped(sc)
        A = aggs_for(grp, ids, thr)
        for j, nm in enumerate(AGG):
            cols[f'{arm}_{nm}'] = A[:, j]
        rec[f'thresholds_{arm}'] = thr
        if arm == 'mechplus':
            sub = {p: ~np.array([T_of[p] in dealt[h] for h in grp[p][0]], bool) for p in ids}
            rec['pairs_with_zero_noT_hands'] = int(sum(1 for p in ids if not sub[p].any()))
            An = aggs_for(grp, ids, thr, subset=sub)
            for j, nm in enumerate(AGG):
                cols[f'mechplusnoT_{nm}'] = An[:, j]
        log('aggregates', arm, round(time.time() - t0, 1), 's')

    d = pl.DataFrame({'pair_id': ids, **{k: v for k, v in cols.items()}})
    d.write_parquet(OUT + 'aggregates258.parquet')

    # ---- parity gate: full history, 149 labelled testbed pairs, mine vs the stored STEP254 OOF
    lab = tb.filter(pl.col('label') >= 0)['pair_id'].to_list()
    thr = json.load(open(OUT + 'score_receipt_mechplus.json', encoding='utf-8'))['honest_thresholds_stored']
    mine = pl.read_parquet(OUT + 'scores_mechplus_full149.parquet').select('pair_id', 'hand_id', 'p_fmax')
    ref = pl.read_parquet(S254 + 'hand_oof_step254.parquet',
                          columns=['pair_id', 'hand_id', 'mechplus_p_fmax']).join(
        pl.DataFrame({'pair_id': lab}), on='pair_id', how='semi').rename({'mechplus_p_fmax': 'p_fmax'})
    assert ref.height == mine.height, (ref.height, mine.height)
    Am = aggs_for(grouped(mine), lab, thr)
    Ar = aggs_for(grouped(ref), lab, thr)
    par = {nm: float(np.abs(Am[:, j] - Ar[:, j]).max()) for j, nm in enumerate(AGG)}
    prob = mine.join(ref, on=['pair_id', 'hand_id'], suffix='_ref')
    par_p = float(np.abs(prob['p_fmax'].to_numpy() - prob['p_fmax_ref'].to_numpy()).max())
    rec['parity_full149'] = {'n_pairs': len(lab), 'n_hands': int(mine.height),
                             'max_abs_p_fmax_diff': par_p,
                             'max_abs_aggregate_diff': par, 'G_par': 'PASS' if max(par.values()) == 0.0 else 'CHECK'}
    log('parity', json.dumps(rec['parity_full149']))
    rec['seconds'] = round(time.time() - t0, 1)
    rec['columns'] = [c for c in d.columns if c != 'pair_id']
    json.dump(rec, open(OUT + 'aggregates258_receipt.json', 'w', encoding='utf-8'), indent=1)
    log('wrote', OUT + 'aggregates258.parquet', d.shape)


if __name__ == '__main__':
    main()
