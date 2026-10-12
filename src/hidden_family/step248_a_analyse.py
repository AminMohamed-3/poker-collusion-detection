"""STEP248 Part A analysis: is there a pair-level mechanism that exists in the EVALUATION
phase and not in a length-matched development population?

Populations
  EVAL TAIL : the 112,540 evaluation candidate pairs minus the 909 STEP207 block pairs.
  DEV NULL  : all within-table development pairs from the FIRST 2,000 hands of each table
              (length matched), restricted to n_both >= 38, minus the 372 labelled positives.
              Sensitivity arm additionally trims the top 2 % on the known-family axes
              (contest_both, flow_z), which is where unlisted disclosed positives live.

Test        per statistic, stratified on shared hands (deciles of n_both defined on the EVAL
            TAIL), threshold = the DEV NULL quantile inside the same stratum; excess =
            observed eval-tail count above threshold minus (1-q) x stratum size.
Bootstrap   resamples the DEV NULL (the uncertain quantity is the THRESHOLD, OBS 16.5), 2,000x.
Controls    chain_sym is the declared blind negative control (OBS 16.4: blind at 4x planted
            strength).  copresence / synchrony / seat_conc / adjacency are secondary blind
            controls (selection excluded on development, OBS 16.7).
"""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binom

ORIGINAL = Path('C:/Data/Projects/Money Making/work/poker')
CV = ORIGINAL / 'Claude Version'
OUT = CV / 'experiments' / 'step248_hidden_family_eval_20260917'
SUB207 = CV / 'outputs_step207_20260917' / 'submission_step207_ens4_ci_dt_rolestrength_on_step169.csv'

STATS = ['contest_both', 'avoid_lr', 'avoid_z', 'avoid_sd',
         'shared_victim', 'sv_overlap', 'sv_conc',
         'chain_sym', 'chain_max', 'chain_z_sym', 'chain_z_max',
         'flow_z', 'direct', 'churn', 'net_flow', 'churn_ratio', 'churn_cancel',
         'copresence', 'seat_conc', 'adjacency', 'synchrony']
BLIND = {'chain_sym', 'chain_z_sym', 'copresence', 'seat_conc', 'adjacency', 'synchrony'}
QUANTILES = [0.99, 0.999]
NBOOT = 2000
MIN_SHARED = 38


def pair_key(a, b):
    return pl.when(pl.col(a) < pl.col(b)).then(pl.col(a)).otherwise(pl.col(b)).alias('p_lo'), \
        pl.when(pl.col(a) < pl.col(b)).then(pl.col(b)).otherwise(pl.col(a)).alias('p_hi')


def boot_threshold_indices(nn, k, nboot, rng):
    """Exact bootstrap distribution of the index, in the ORIGINAL sorted null array, of the
    k-th order statistic of a size-nn bootstrap resample.

    P(X*_(k) <= x_(j)) = P(Binomial(nn, j/nn) >= k), so the CDF over j is computed once and
    inverted with searchsorted.  This replaces nboot explicit resamples (OBS 16.5 requires the
    THRESHOLD to be the resampled quantity, not the expectation)."""
    j = np.arange(1, nn + 1)
    cdf = binom.sf(k - 1, nn, j / nn)          # P(Bin(nn, j/nn) >= k)
    cdf = np.maximum.accumulate(np.clip(cdf, 0, 1))
    cdf[-1] = 1.0
    u = rng.random(nboot)
    return np.searchsorted(cdf, u, side='left').clip(0, nn - 1)


def excess_table(evalv, nullv, strata_e, strata_n, n_strata, q, upper=True, rng=None,
                 nboot=NBOOT):
    """Observed eval count beyond the per-stratum dev-null quantile, and its bootstrap null."""
    obs = 0.0
    exp = 0.0
    boot = np.zeros(nboot)
    for s in range(n_strata):
        me = strata_e == s
        mn = strata_n == s
        ne, nn = int(me.sum()), int(mn.sum())
        if ne == 0 or nn < 200:
            continue
        ns = np.sort(nullv[mn])
        es = np.sort(evalv[me])
        k = int(np.clip(round((q if upper else 1 - q) * nn), 1, nn))
        thr = ns[k - 1]
        obs += float(ne - np.searchsorted(es, thr, side='right')) if upper \
            else float(np.searchsorted(es, thr, side='left'))
        exp += ne * (1 - q)
        idx = boot_threshold_indices(nn, k, nboot, rng)
        tb = ns[idx]
        if upper:
            boot += ne - np.searchsorted(es, tb, side='right')
        else:
            boot += np.searchsorted(es, tb, side='left')
    boot = boot - exp
    return obs, exp, obs - exp, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main():
    t0 = time.time()
    rng = np.random.default_rng(248)
    ev = pl.read_parquet(OUT / 'step248a-mechstats-evaluation.parquet')
    dv = pl.read_parquet(OUT / 'step248a-mechstats-development2000.parquet')
    print(f'eval pairs {ev.height}, dev2000 pairs {dv.height}  [{time.time()-t0:.0f}s]')

    # ---- eval analysis set: candidate pairs minus the STEP207 block -------------------
    ep = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv')
    sub = pl.read_csv(SUB207, columns=['pair_id', 'risk_score'])
    ep = ep.join(sub, on='pair_id', how='left')
    assert ep['risk_score'].null_count() == 0
    ep = ep.with_columns(*pair_key('player_1', 'player_2'))
    ev = ev.with_columns(*pair_key('player_a', 'player_b'))
    ev = ev.join(ep.select('p_lo', 'p_hi', 'pair_id', 'shared_hands', 'risk_score'),
                 on=['p_lo', 'p_hi'], how='inner')
    print(f'eval candidate pairs matched: {ev.height} (expect 112540)')
    assert ev.height == 112540
    # n_both from the battery must equal the organisers' shared_hands
    d = (ev['n_both'].cast(pl.Int64) - ev['shared_hands']).abs().max()
    print(f'max |n_both - shared_hands| = {d}')
    tail = ev.filter(pl.col('risk_score') < 0.5)
    block = ev.filter(pl.col('risk_score') >= 0.5)
    print(f'TAIL {tail.height}, BLOCK {block.height}')

    # ---- dev null ---------------------------------------------------------------------
    dl = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    pos = dl.filter(pl.col('label') == 1).with_columns(*pair_key('player_1', 'player_2'))
    dv = dv.with_columns(*pair_key('player_a', 'player_b'))
    dv = dv.join(pos.select('p_lo', 'p_hi').with_columns(pl.lit(True).alias('is_pos')),
                 on=['p_lo', 'p_hi'], how='left').with_columns(
        pl.col('is_pos').fill_null(False))
    dvn = dv.filter((pl.col('n_both') >= MIN_SHARED) & (~pl.col('is_pos')))
    print(f'dev null (n_both>=38, labelled positives removed): {dvn.height} '
          f'(removed {int(dv["is_pos"].sum())} positives present)')

    # sensitivity: additionally trim the top 2 % on the two known-family axes
    cb = dvn['contest_both'].to_numpy(); fz = dvn['flow_z'].to_numpy()
    keep = (cb <= np.quantile(cb, .98)) & (fz <= np.quantile(fz, .98))
    dvs = dvn.filter(pl.Series(keep))
    print(f'dev null sensitivity (known-family axes trimmed): {dvs.height}')

    # ---- strata on shared hands --------------------------------------------------------
    ne = tail['n_both'].to_numpy()
    edges = np.quantile(ne, np.linspace(0, 1, 11))
    edges[0] -= 1; edges[-1] += 1
    strata_e = np.clip(np.digitize(ne, edges[1:-1]), 0, 9)
    results = {}
    for arm, nulldf in (('primary', dvn), ('sensitivity', dvs)):
        nn = nulldf['n_both'].to_numpy()
        strata_n = np.clip(np.digitize(nn, edges[1:-1]), 0, 9)
        rows = []
        for st in STATS:
            e = tail[st].to_numpy().astype(float)
            n = nulldf[st].to_numpy().astype(float)
            for q in QUANTILES:
                for upper in (True, False):
                    obs, exp, exc, lo, hi = excess_table(e, n, strata_e, strata_n, 10, q,
                                                         upper, rng)
                    rows.append({'stat': st, 'q': q, 'tail': 'upper' if upper else 'lower',
                                 'obs': obs, 'exp': exp, 'excess': exc,
                                 'ci_lo': lo, 'ci_hi': hi,
                                 'blind_control': st in BLIND})
            print(f'  [{arm}] {st} done  [{time.time()-t0:.0f}s]', flush=True)
        results[arm] = rows
    # block positive control: does the battery see the STEP207 block at all?
    nb = block['n_both'].to_numpy()
    strata_b = np.clip(np.digitize(nb, edges[1:-1]), 0, 9)
    nn = dvn['n_both'].to_numpy()
    strata_n = np.clip(np.digitize(nn, edges[1:-1]), 0, 9)
    ctrl = []
    for st in STATS:
        obs, exp, exc, lo, hi = excess_table(block[st].to_numpy().astype(float),
                                             dvn[st].to_numpy().astype(float),
                                             strata_b, strata_n, 10, 0.99, True, rng, nboot=200)
        ctrl.append({'stat': st, 'obs': obs, 'exp': exp, 'excess': exc,
                     'ci_lo': lo, 'ci_hi': hi})
    results['block_positive_control_q99_upper'] = ctrl
    results['meta'] = {'eval_tail': tail.height, 'eval_block': block.height,
                       'dev_null_primary': dvn.height, 'dev_null_sensitivity': dvs.height,
                       'strata_edges': [float(x) for x in edges], 'nboot': NBOOT,
                       'max_abs_nboth_minus_sharedhands': int(d)}
    (OUT / 'step248a-results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')

    # flagged eval-tail pairs on the corrected laundering statistic, for the report
    for st in ['chain_max', 'chain_z_max', 'churn', 'sv_conc']:
        e = tail[st].to_numpy().astype(float)
        n = dvn[st].to_numpy().astype(float)
        thr = np.quantile(n, .999)
        sel = tail.filter(pl.Series(e > thr)).sort(st, descending=True).head(50)
        sel.select('pair_id', 'p_lo', 'p_hi', 'n_both', 'risk_score', st).write_csv(
            OUT / f'step248a-top-{st}.csv')
    print(f'done  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
