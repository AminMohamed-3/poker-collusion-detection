import polars as pl, numpy as np
from scipy.stats import poisson
ROOT = r'C:\Data\Projects\Money Making\work\poker'; CV = ROOT + r'\Claude Version'
c = pl.read_parquet(CV + r'\experiments\step250_event_count_tail_20260917\eval_pair_event_counts.parquet')
s = pl.read_csv(CV + r'\outputs_step207_20260917\submission_step207_ens4_ci_dt_rolestrength_on_step169.csv').select('pair_id', 'risk_score', 'predicted_behavior')
ep = pl.read_csv(ROOT + r'\data\evaluation_pairs.csv')
d = ep.join(c, on='pair_id').join(s, on='pair_id')
# player-level totals over all the player's evaluation pairs
long = pl.concat([d.select(pl.col('player_1').alias('pl'), 'pair_id', 'cnt999', 'n_hands', 'risk_score'), d.select(pl.col('player_2').alias('pl'), 'pair_id', 'cnt999', 'n_hands', 'risk_score')])
tot = long.group_by('pl').agg(pl.col('cnt999').sum().alias('c_tot'), pl.col('n_hands').sum().alias('n_tot'), pl.len().alias('n_pairs'), pl.col('risk_score').max().alias('max_risk_any'))
d = d.join(tot.rename({'pl': 'player_1', 'c_tot': 'c1', 'n_tot': 'n1', 'n_pairs': 'k1', 'max_risk_any': 'mr1'}), on='player_1').join(tot.rename({'pl': 'player_2', 'c_tot': 'c2', 'n_tot': 'n2', 'n_pairs': 'k2', 'max_risk_any': 'mr2'}), on='player_2')
glob = float(d['cnt999'].sum() / d['n_hands'].sum())
# leave-this-pair-out rate of each member with OTHER table-mates, shrunk to the global rate (prior weight 200 hands)
d = d.with_columns(r1=((pl.col('c1') - pl.col('cnt999')) + glob * 200) / ((pl.col('n1') - pl.col('n_hands')) + 200),
                   r2=((pl.col('c2') - pl.col('cnt999')) + glob * 200) / ((pl.col('n2') - pl.col('n_hands')) + 200))
d = d.with_columns(lam=pl.col('n_hands') * pl.max_horizontal('r1', 'r2'))
pv = poisson.sf(d['cnt999'].to_numpy().astype(np.int64) - 1, d['lam'].to_numpy())
d = d.with_columns(pl.Series('p_excess', pv), mlog=pl.Series('mlog', -np.log10(np.maximum(pv, 1e-300))))
print('global sharp rate per hand', round(glob, 6), '| pairs', d.height)
d = d.with_columns(band=pl.when(pl.col('risk_score') >= .99).then(pl.lit('a>=.99')).when(pl.col('risk_score') >= .9).then(pl.lit('b.9-.99')).when(pl.col('risk_score') >= .5).then(pl.lit('c.5-.9'))
                   .when(pl.col('risk_score') >= .1).then(pl.lit('d.1-.5')).when(pl.col('risk_score') >= .01).then(pl.lit('e.01-.1')).otherwise(pl.lit('f<.01')))
for thr in (3, 4, 5, 6):
    print('pairs with -log10 p >=', thr, d.filter(pl.col('mlog') >= thr).group_by('band').len().sort('band').to_dict(as_series=False))
d = d.with_columns(other_block_any=(pl.max_horizontal('mr1', 'mr2') >= .5)); print('tail base rate of sharing a player with a block pair:', round(float(d.filter(pl.col('risk_score') < .5)['other_block_any'].mean()), 4)); print('tail pairs by excess p (all cnt>=2):', {t: d.filter((pl.col('risk_score') < .5) & (pl.col('mlog') >= t)).height for t in (3, 3.5, 4, 5)}, 'expected under a calibrated null:', {t: round(111631 * 10 ** -t, 1) for t in (3, 3.5, 4, 5)})
T = d.filter((pl.col('risk_score') < .5) & (pl.col('cnt999') >= 3)).with_columns(other_block=(pl.max_horizontal('mr1', 'mr2') >= .5))
print('tail pairs cnt999>=3:', T.height, '| share a player with a block pair:', int(T['other_block'].sum()))
print(T.group_by('band').agg(pl.len(), pl.col('mlog').median().alias('med_mlog'), (pl.col('mlog') >= 4).sum().alias('mlog>=4'), pl.col('other_block').sum().alias('shares_block_player'), pl.col('lam').median().alias('med_lam')).sort('band'))
print(T.sort('mlog', descending=True).select('pair_id', 'risk_score', 'predicted_behavior', 'n_hands', 'cnt999', 'cnt99', 'lam', 'mlog', 'top3_logit', 'other_block').head(40))
d.write_parquet(CV + r'\experiments\step250_event_count_tail_20260917\eval_pair_excess.parquet')
