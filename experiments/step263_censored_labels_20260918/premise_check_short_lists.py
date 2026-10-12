"""Amin's idea (2026-09-18): lists have 3, 4 or 5 hands -- can a model fill in the missing 4th / 5th hand of the short
lists, and can we train on that?  First the FACT the idea rests on (development only, OOF scores, zero fits):
are short lists COMPLETE (the pair simply has 3-4 events) or TRUNCATED (there are more events that were not listed)?
Instrument: the MECHPLUS out-of-fold event probability (p_fmax) of the UNLISTED hands of positive pairs, against the honest
thresholds q99 = .05541 and q99.9 = .44939 (rates .01 / .001 on hands of confirmed non-target pairs by construction).
  * 5-hand lists: the cap binds, so unlisted hands AFTER the list's last hand should hold the censored events (large excess);
  * 3-4-hand lists: excess ~ 0 everywhere  => complete lists: nothing to fill in, but every unlisted hand of such a pair is a
    CERTIFIED within-pair negative (a colluding pair's ordinary hand);  excess > 0 => truncated: the fill-in idea has room.
"""
import numpy as np, polars as pl
ROOT = r'C:\Data\Projects\Money Making\work\poker'; CV = ROOT + r'\Claude Version'
Q99, Q999 = .05541, .44939
o = pl.read_parquet(CV + r'\experiments\step254_mechanical_tensor_20260917\hand_oof_step254.parquet',
                    columns=['pair_id', 'hand_id', 'listed', 'label', 'family', 'mechplus_p_fmax', 'mechplus_p_any'])
t = pl.read_parquet(ROOT + r'\data\hands.parquet', columns=['hand_id', 'started_at', 'phase'])
o = o.join(t, on='hand_id', how='left'); assert (o['phase'] == 'development').all()
pp = o.group_by('pair_id').agg(pl.col('listed').sum().alias('n_listed'), pl.len().alias('n_hands'),
                               pl.col('started_at').filter(pl.col('listed') == 1).max().alias('t_last'),
                               pl.col('started_at').filter(pl.col('listed') == 1).min().alias('t_first'))
o = o.join(pp, on='pair_id')
print('positive pairs by list length:', dict(pp.filter(pl.col('n_listed') > 0).group_by('n_listed').len().sort('n_listed').iter_rows()))
print('median shared hands by list length:', dict(pp.filter(pl.col('n_listed') > 0).group_by('n_listed').agg(pl.col('n_hands').median()).sort('n_listed').iter_rows()))
neg = o.filter(pl.col('label') == 0)
print(f"honest check (confirmed non-target pairs, {neg.height} hands): rate > q99 {(neg['mechplus_p_fmax'] > Q99).mean():.4f}, > q99.9 {(neg['mechplus_p_fmax'] > Q999).mean():.5f}")
lst = o.filter(pl.col('listed') == 1)
print(f"listed hands ({lst.height}): rate > q99 {(lst['mechplus_p_fmax'] > Q99).mean():.3f}, > q99.9 {(lst['mechplus_p_fmax'] > Q999).mean():.3f}")
u = o.filter((pl.col('label') == 1) & (pl.col('listed') == 0)).with_columns(
    seg=pl.when(pl.col('started_at') < pl.col('t_first')).then(pl.lit('before first')).when(pl.col('started_at') <= pl.col('t_last')).then(pl.lit('inside span')).otherwise(pl.lit('after last')))
print('\nUNLISTED hands of positive pairs: share above the honest thresholds, and EXCESS sharp hands per pair (observed - honest rate x hands)')
for nl in (3, 4, 5):
    g = u.filter(pl.col('n_listed') == nl); npairs = g['pair_id'].n_unique()
    for seg in ('before first', 'inside span', 'after last', 'ALL'):
        s = g if seg == 'ALL' else g.filter(pl.col('seg') == seg)
        if s.height == 0:
            continue
        a, b = (s['mechplus_p_fmax'] > Q99).sum(), (s['mechplus_p_fmax'] > Q999).sum()
        print(f"  listed {nl} ({npairs:3d} pairs) {seg:13s} hands {s.height:6d} | > q99: {a / s.height:.4f} (excess/pair {(a - .01 * s.height) / npairs:+.2f}) | > q99.9: {b / s.height:.5f} (excess/pair {(b - .001 * s.height) / npairs:+.2f})")
# by family for the short lists
for fam in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
    g = u.filter((pl.col('n_listed') < 5) & (pl.col('family') == fam)); npairs = g['pair_id'].n_unique()
    if npairs:
        b = (g['mechplus_p_fmax'] > Q999).sum(); a = (g['mechplus_p_fmax'] > Q99).sum()
        print(f"  short lists, {fam:22s} pairs {npairs:3d} hands {g.height:6d} | > q99 excess/pair {(a - .01 * g.height) / npairs:+.2f} | > q99.9 excess/pair {(b - .001 * g.height) / npairs:+.2f}")
# per-pair detail of the short lists: how many unlisted hands are sharp at q99.9
d = u.filter(pl.col('n_listed') < 5).group_by('pair_id').agg(pl.col('n_listed').first(), pl.col('n_hands').first(), pl.col('family').first(),
                                                           (pl.col('mechplus_p_fmax') > Q999).sum().alias('sharp999'), (pl.col('mechplus_p_fmax') > Q99).sum().alias('sharp99'))
print('\nshort-list pairs: distribution of unlisted hands above q99.9:', dict(d.group_by('sharp999').len().sort('sharp999').iter_rows()))
print('expected by chance per pair at the honest rate:', round(float((d['n_hands'] * .001).mean()), 3))
d5 = u.filter(pl.col('n_listed') == 5).group_by('pair_id').agg((pl.col('mechplus_p_fmax') > Q999).sum().alias('sharp999'))
print('5-hand-list pairs: unlisted hands above q99.9 per pair: mean', round(float(d5['sharp999'].mean()), 2), 'median', float(d5['sharp999'].median()), '| share of pairs with >= 2:', round(float((d5['sharp999'] >= 2).mean()), 3))
