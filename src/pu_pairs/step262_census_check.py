"""STEP262 addendum: re-derive the coordinator's LABEL-FREE evaluation census from the STEP257
aggregates, so the CENSUS arm's stated motivation is checked rather than taken on trust.

Population statistics on evaluation only.  No evaluation label is used; no evaluation pair or player
is joined to any development record.  Writes census_check262.json.

Usage:  python step262_census_check.py
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
import sys, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl                                        # noqa: E402
from scipy.stats import binom                                           # noqa: E402

OUT = C + 'experiments/step262_mech_tail_view_20260918/'
EVALAGG = C + 'experiments/step257_eval_mech_scores_20260918/eval_pair_mech_aggregates.parquet'
BASE = C + 'outputs_step251_20260917/submission_step251_ens4_dtfp3_on_step222.csv'
Q999 = .44939


def main():
    t0 = time.time()
    dest = OUT + 'census_check262.json'
    assert not os.path.exists(dest), dest
    e = pl.read_parquet(EVALAGG, columns=['pair_id', 'n_hands', 'cnt_q999', 'cnt_q99', 'top3_logit'])
    base = pl.read_csv(BASE, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    d = e.join(base, on='pair_id', how='inner', validate='1:1')
    assert d.height == 112540
    tail = d.filter(pl.col('risk_score') < .5)
    rec = {'note': 'label-free evaluation census; population statistics only',
           'n_pairs': int(d.height), 'n_tail': int(tail.height),
           'n_head': int(d.height - tail.height), 'threshold_q999': Q999}
    n = tail['n_hands'].to_numpy().astype(float)
    k = tail['cnt_q999'].to_numpy().astype(float)
    r = tail['risk_score'].to_numpy()
    rec['tail_rate_all'] = float(k.sum() / n.sum())
    deep = r <= np.quantile(r, .9)
    rec['tail_rate_deep90'] = float(k[deep].sum() / n[deep].sum())
    deep99 = r <= np.quantile(r, .99)
    rec['tail_rate_deep99'] = float(k[deep99].sum() / n[deep99].sum())
    for rate_name, rate in (('coordinator_rate_0.00078', .00078), ('observed_deep90_rate', rec['tail_rate_deep90'])):
        p = binom.sf(k - 1, n.astype(int), rate)
        row = {}
        for a in (1e-3, 1e-4):
            sel = p < a
            row[f'p_lt_{a:g}'] = {'observed': int(sel.sum()), 'expected_uniform': round(float(tail.height * a), 1),
                                  'risk_median_of_selected': float(np.median(r[sel])) if sel.any() else None,
                                  'cnt_q999_min': float(k[sel].min()) if sel.any() else None}
        rec[rate_name] = row
    # variant A on evaluation, for the production promotion set size
    selA = k >= 3
    rec['variantA_cnt_q999_ge_3'] = {'n': int(selA.sum()), 'risk_median': float(np.median(r[selA])) if selA.any() else None,
                                     'risk_max': float(r[selA].max()) if selA.any() else None}
    rec['cnt_q999_histogram_tail'] = {str(int(v)): int((k == v).sum()) for v in np.unique(k)[:12]}
    # do the selected pairs share a player with a block pair?  (evaluation-internal, no labels)
    ep = pl.read_csv(ROOT + 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2'])
    dd = d.join(ep, on='pair_id', how='left', validate='1:1')
    assert dd['player_1'].null_count() == 0
    blk = dd.filter(pl.col('risk_score') >= .5)
    blk_players = set(blk['player_1'].to_list()) | set(blk['player_2'].to_list())
    blk_pairs = set(blk['pair_id'].to_list())
    t2 = dd.filter(pl.col('risk_score') < .5)
    p_deep = binom.sf(t2['cnt_q999'].to_numpy().astype(float) - 1, t2['n_hands'].to_numpy().astype(int), .00078)
    for nm, sel in (('p_lt_1e-3', p_deep < 1e-3), ('variantA', t2['cnt_q999'].to_numpy() >= 3)):
        s = t2.filter(pl.Series(sel))
        shares = [(a in blk_players) or (b in blk_players) for a, b in zip(s['player_1'].to_list(), s['player_2'].to_list())]
        rec[f'{nm}_block_player_overlap'] = {'n': int(s.height), 'shares_a_player_with_a_block_pair': int(sum(shares)),
                                             'shares_no_player_with_any_block_pair': int(len(shares) - sum(shares))}
    rec['n_block_pairs'] = len(blk_pairs)
    rec['seconds'] = round(time.time() - t0, 1)
    json.dump(rec, open(dest, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
