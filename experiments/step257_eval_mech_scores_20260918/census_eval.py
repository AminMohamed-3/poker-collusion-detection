"""Label-free census of the EVALUATION pairs with the sharper (MECHPLUS) event model -- repeat of STEP250 (raw model),
which found 1 tail pair at p < 1e-4 against 11 expected by chance and 70 % spill-over among tail pairs with >= 3 sharp hands.
Question: does the tail (risk < .5 in STEP251) hold pairs whose count of hands above the honest q99.9 (.44939, dev clean
pairs) is far above chance, and are they R1 spill-over pairs?  No labels; nothing is tuned.  Evaluation aggregates:
cnt_q999 per pair (STEP257).  Chance: Binomial(n_hands, .001) -- but the honest rate on evaluation may differ from
development (shorter histories, different mix), so the EMPIRICAL rate is read off the deep tail (risk < .01) first."""
import numpy as np, polars as pl
from scipy.stats import binom
CV = r'C:\Data\Projects\Money Making\work\poker\Claude Version'
a = pl.read_parquet(CV + r'\experiments\step257_eval_mech_scores_20260918\eval_pair_mech_aggregates.parquet')
sub = pl.read_csv(CV + r'\outputs_step251_20260917\submission_step251_ens4_dtfp3_on_step222.csv', columns=['pair_id', 'risk_score'])
c = pl.read_csv(CV + r'\outputs_spillover_20260911\submission_spillover_demotion_R1.csv', columns=['pair_id', 'risk_score']).rename({'risk_score': 'risk_C'})
b = pl.read_csv(CV + r'\outputs_chronology_20260911\submission_chronology_frac.csv', columns=['pair_id', 'risk_score']).rename({'risk_score': 'risk_base'})
d = a.join(sub, on='pair_id').join(c, on='pair_id').join(b, on='pair_id').with_columns(R1=((pl.col('risk_base') >= .5) & (pl.col('risk_C') < .5)))
print('pairs', d.height)
deep = d.filter(pl.col('risk_score') < .01)
rate = float(deep['cnt_q999'].sum() / deep['n_hands'].sum())
print(f"deep tail (risk < .01): pairs {deep.height}, hands {int(deep['n_hands'].sum())}, hands above dev q99.9: {int(deep['cnt_q999'].sum())} -> empirical rate {rate:.5f} (development honest rate .00100)")
for lo, hi in ((.99, 1.01), (.9, .99), (.5, .9), (.1, .5), (.01, .1), (0, .01)):
    g = d.filter((pl.col('risk_score') >= lo) & (pl.col('risk_score') < hi))
    n = g['n_hands'].to_numpy(); k = g['cnt_q999'].to_numpy()
    exp3 = float(np.sum(binom.sf(2, n, rate))); exp2 = float(np.sum(binom.sf(1, n, rate)))
    print(f"band [{lo},{hi}): pairs {g.height:6d} | mean cnt {k.mean():.3f} | >=2 sharp: {int((k >= 2).sum()):5d} (chance {exp2:.1f}) | >=3 sharp: {int((k >= 3).sum()):5d} (chance {exp3:.1f}) | R1 share among >=3: {g.filter(pl.col('cnt_q999') >= 3)['R1'].mean() if (k >= 3).any() else float('nan'):.2f}")
# the tail candidates: >= 3 sharp hands, risk < .5, not R1 -- how many, and how far below .5?
t = d.filter((pl.col('risk_score') < .5) & (pl.col('cnt_q999') >= 3) & ~pl.col('R1')).sort('cnt_q999', descending=True)
print('tail, not R1, >= 3 sharp hands:', t.height, '| risk quantiles', [round(float(t['risk_score'].quantile(q)), 3) for q in (.1, .5, .9)], '| cnt hist', dict(t.group_by('cnt_q999').len().sort('cnt_q999').iter_rows()))
# per-pair p-value of the count under the empirical rate; how many at p < 1e-4 in the tail (STEP250 found 1 vs 11 expected)
d = d.with_columns(pl.Series('pv', binom.sf(d['cnt_q999'].to_numpy() - 1, d['n_hands'].to_numpy(), rate)))
tail = d.filter(pl.col('risk_score') < .5)
print('tail pairs', tail.height, '| p < 1e-4:', tail.filter(pl.col('pv') < 1e-4).height, '(chance', round(tail.height * 1e-4, 1), ') of which R1:', tail.filter((pl.col('pv') < 1e-4) & pl.col('R1')).height,
      '| p < 1e-3:', tail.filter(pl.col('pv') < 1e-3).height, '(chance', round(tail.height * 1e-3, 1), ') R1:', tail.filter((pl.col('pv') < 1e-3) & pl.col('R1')).height)
print(tail.filter(pl.col('pv') < 1e-4).select('risk_score', 'n_hands', 'cnt_q999', 'cnt_q99', 'top3_logit', 'R1', 'pv').sort('pv').head(30))
