"""STEP199 phase A: is the 5-hand evidence list a binding CAP, and how much positive mass does it hide?

340 of 372 labelled development pairs list exactly 5 hands, 21 list 4 and 11 list 3. Because 32
lists are SHORTER than 5, five is a cap and not a quota -- and for the 340 capped pairs the list may
have been truncated by the cap rather than by the evidence running out. If so, hands that would have
ranked 6th and beyond are in training labelled 0, indistinguishable from the 180th-best hand.
STEP133 measured +.034 decoded MAP per doubling of labelled lists.

THE NATURAL EXPERIMENT. In a pair listing 3 or 4, the annotator demonstrably stopped because the
evidence ran out: the cap was never reached, so the stopping point is informative. In a pair listing
exactly 5 the stopping point is uninformative. Compare the two groups on a quantity that should
differ ONLY IF the cap binds:

    s* = the best production out-of-fold score among that pair's UNLISTED candidates
    r  = how many of the pair's LISTED hands s* outranks          rho = r / n_listed

A binding cap means the best unlisted hand is a near-miss that looks like a listed hand and outranks
several. Evidence genuinely running out means it looks clearly worse and outranks few or none.

This guesses nothing and labels nothing -- the three previous attempts at this lever (STEP137
self-labels NULL, STEP196 LLM labels monotone damage, STEP136 synthetic pairs NO-GO) all tried to
guess WHICH hands are the hidden positives. Nothing here can be built into a file.

The test is CONSERVATIVE: s is production's out-of-fold score and production was trained on these
labels, which biases it toward ranking listed above unlisted IN BOTH GROUPS and shrinks the gap.

Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step199_C_censoring_measure.py REGSHA
"""
import os
import sys
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step199_20260917'
REG = OUT / 'registration199.json'
R = lambda x: round(float(x), 4)


def block(r, rho, d, npool):
    """Summary of one group of pairs, with the standard error it must be read against."""
    n = len(r)
    if n == 0:
        return {'n_pairs': 0}
    return {'n_pairs': int(n),
            'mean_r': R(r.mean()), 'se_r': R(r.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            'median_r': R(np.median(r)),
            'frac_r_ge_1': R((r >= 1).mean()), 'frac_r_ge_2': R((r >= 2).mean()),
            'mean_rho': R(rho.mean()), 'mean_score_gap_d': R(d.mean()),
            'pool_size_median': R(np.median(npool)),
            'pool_size_q1_q3': [R(np.percentile(npool, 25)), R(np.percentile(npool, 75))]}


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not (OUT / 'results199.json').exists(), 'preserve a completed read'
    from evidence_aug.step167_C_lp_production import modules
    from evidence_aug.paths import check_protected
    prot = check_protected()
    S108, _ = modules()
    CSP = S108.CSP
    t0 = time.perf_counter()

    frame = CSP.development_frame()[0]
    # the development frame names the out-of-fold stage-1 score s1; S165 aliases it oof later
    cols = ['pair_id', 'hand_id', 'relevance', 'behavior_family', 's1', 'n_hands']
    f = frame.select(cols).unique(subset=['pair_id', 'hand_id'])
    from sklearn.metrics import roc_auc_score
    s1_auc = float(roc_auc_score(f['relevance'].to_numpy().astype(int), f['s1'].to_numpy()))
    print(f'{f.height} candidate rows over {f["pair_id"].n_unique()} pairs; '
          f's1 global AUC vs relevance {s1_auc:.4f}', flush=True)

    rows = []
    for (pid,), g in f.group_by('pair_id'):
        rel = g['relevance'].to_numpy().astype(int)
        s = g['s1'].to_numpy().astype(float)
        nl = int(rel.sum())
        if nl == 0 or (rel == 0).sum() == 0:
            continue
        sl, su = s[rel == 1], s[rel == 0]
        sstar = su.max()
        rows.append({'pair_id': pid, 'family': g['behavior_family'][0], 'n_listed': nl,
                     'n_pool': int(len(s)), 'n_hands': float(g['n_hands'][0]),
                     'r': int((sl < sstar).sum()), 'rho': float((sl < sstar).sum()) / nl,
                     'd': float(sstar - sl.min())})
    P = pl.DataFrame(rows)
    print(f'{P.height} pairs usable  {round(time.perf_counter()-t0)}s', flush=True)

    dist = {int(k): int(v) for k, v in
            zip(*np.unique(P['n_listed'].to_numpy(), return_counts=True))}
    capped = P.filter(pl.col('n_listed') == 5)
    unc = P.filter(pl.col('n_listed') < 5)
    get = lambda T: (T['r'].to_numpy().astype(float), T['rho'].to_numpy(),
                     T['d'].to_numpy(), T['n_pool'].to_numpy())

    res = {'registration_sha256': reg_sha, 'phase': 'A -- measurement only, nothing can be built',
           'n_pairs': P.height, 'list_length_distribution': dist, 's1_global_auc': R(s1_auc),
           'groups': {'capped_len5': block(*get(capped)), 'uncapped_len3_4': block(*get(unc))}}
    a, b = res['groups']['capped_len5'], res['groups']['uncapped_len3_4']
    diff = a['mean_r'] - b['mean_r']
    se = float(np.hypot(a['se_r'] or 0, b['se_r'] or 0))
    res['headline'] = {
        'capped_mean_r': a['mean_r'], 'uncapped_mean_r': b['mean_r'],
        'difference': R(diff), 'se_of_difference': R(se),
        'sigma': R(diff / se) if se > 0 else None,
        'registered_prediction': {'capped': 1.2, 'uncapped': 0.6, 'difference': 0.6,
                                  'band80': [0.0, 1.3], 'P_capped_exceeds': 0.7}}

    # --- confound 1: pool size. s* is a max over U, so a bigger pool inflates it mechanically. ---
    q = np.percentile(P['n_pool'].to_numpy(), [33.3, 66.7])
    strat = {}
    for name, lo, hi in (('small', -np.inf, q[0]), ('mid', q[0], q[1]), ('large', q[1], np.inf)):
        sel = P.filter((pl.col('n_pool') > lo) & (pl.col('n_pool') <= hi))
        c, u = sel.filter(pl.col('n_listed') == 5), sel.filter(pl.col('n_listed') < 5)
        cb, ub = block(*get(c)), block(*get(u))
        strat[name] = {'pool_range': [R(lo) if np.isfinite(lo) else None,
                                      R(hi) if np.isfinite(hi) else None],
                       'capped': cb, 'uncapped': ub,
                       'difference': R(cb['mean_r'] - ub['mean_r']) if ub.get('n_pairs') else None}
    res['stratified_by_pool_size'] = strat
    res['pool_size_overlap_check'] = {
        'capped_median': a['pool_size_median'], 'uncapped_median': b['pool_size_median'],
        'note': 'if these are far apart the pooled comparison is confounded and only the strata count'}

    # --- confound 2: family ---
    fam = {}
    for name in sorted(P['family'].unique().to_list()):
        sel = P.filter(pl.col('family') == name)
        c, u = sel.filter(pl.col('n_listed') == 5), sel.filter(pl.col('n_listed') < 5)
        cb, ub = block(*get(c)), block(*get(u))
        fam[name] = {'capped': cb, 'uncapped': ub,
                     'difference': R(cb['mean_r'] - ub['mean_r']) if ub.get('n_pairs') else None}
    res['by_family'] = fam

    ok = diff > (b['se_r'] or 0)
    strat_signs = [v['difference'] for v in strat.values() if v['difference'] is not None]
    res['verdict'] = ('LEVER_CONFIRMED_AND_SIZED' if ok and all(x > 0 for x in strat_signs)
                      else 'LEVER_SMALLER_THAN_ASSUMED')
    res['stratification_survives'] = bool(strat_signs and all(x > 0 for x in strat_signs))
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    OUT.mkdir(parents=True, exist_ok=True)
    P.write_parquet(OUT / 'per_pair_censoring.parquet')
    (OUT / 'results199.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
