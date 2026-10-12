"""STEP231 (exploration, development phase only, development labels used descriptively):
do the colluders' actions in LISTED hands carry an action-level fingerprint (sizing ratios, folds when a check is
free, all-in patterns) that the same players' actions in their UNLISTED shared hands do not?"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import polars as pl, json
ROOT = r'C:\Data\Projects\Money Making\work\poker'
hands = pl.scan_parquet(ROOT + r'\data\hands.parquet').filter(pl.col('phase') == 'development').select('hand_id', 'table_id', 'started_at', 'big_blind', 'players_dealt')
lab = pl.read_csv(ROOT + r'\data\development_labels.csv').filter(pl.col('label') == 1)
ev = pl.read_csv(ROOT + r'\data\development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
seats = pl.scan_parquet(ROOT + r'\data\seats.parquet').join(hands, on='hand_id')  # development-phase seats only
members = pl.concat([lab.select('pair_id', 'behavior_family', pl.col('player_1').alias('player_id')), lab.select('pair_id', 'behavior_family', pl.col('player_2').alias('player_id'))])
ms = seats.join(members.lazy(), on='player_id').select('pair_id', 'behavior_family', 'hand_id', 'player_id').collect()
shared = ms.group_by('pair_id', 'hand_id').agg(pl.len().alias('n')).filter(pl.col('n') == 2).select('pair_id', 'hand_id')
ms = ms.join(shared, on=['pair_id', 'hand_id']).join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('evidence_rank').is_not_null().alias('listed'))
assert ms.join(hands.select('hand_id').collect(), on='hand_id', how='anti').height == 0  # every hand is development phase
print('member-hand rows', ms.height, 'listed', ms['listed'].sum() // 2, 'pairs', ms['pair_id'].n_unique())
acts = pl.scan_parquet(ROOT + r'\data\actions.parquet').join(ms.lazy().select('hand_id', 'player_id', 'pair_id', 'behavior_family', 'listed'), on=['hand_id', 'player_id']).join(hands.select('hand_id', 'big_blind'), on='hand_id').collect()
print('member actions', acts.height, acts['action'].unique().to_list())
acts = acts.with_columns((pl.col('amount') / pl.col('pot_before')).alias('amt_pot'), (pl.col('amount_to') / pl.col('big_blind')).alias('to_bb'), (pl.col('amount') >= pl.col('stack_before')).alias('allin'), (pl.col('to_call') == 0).alias('free'), (pl.col('amount') / pl.col('to_call')).alias('amt_call'))
out = {}
for fam in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
    a = acts.filter(pl.col('behavior_family') == fam); res = {}
    for flag in (True, False):
        s = a.filter(pl.col('listed') == flag); n = s.height
        mix = s.group_by('action').agg(pl.len().alias('k')).with_columns((pl.col('k') / n).round(4).alias('share')).sort('action').select('action', 'share').to_dicts()
        ff = s.filter((pl.col('action') == 'fold') & pl.col('free')).height
        ai = s.filter(pl.col('allin') & pl.col('action').is_in(['bet', 'raise', 'call'])).height
        res['listed' if flag else 'unlisted'] = {'n_actions': n, 'mix': mix, 'fold_when_free': ff, 'allin_share': round(ai / max(n, 1), 4)}
    out[fam] = res
print(json.dumps(out, indent=1))
# sizing spikes: raise/bet amount over pot, rounded, listed vs unlisted (all families)
for act in ['bet', 'raise']:
    s = acts.filter(pl.col('action') == act).with_columns((pl.col('amt_pot') * 20).round(0).alias('bin') / 20)
    t = s.group_by('listed', 'bin').agg(pl.len().alias('k')).with_columns((pl.col('k') / pl.col('k').sum().over('listed')).alias('share'))
    w = t.pivot(on='listed', index='bin', values='share').fill_null(0).sort('bin')
    w = w.with_columns((pl.col('true') / (pl.col('false') + 1e-4)).alias('ratio')).filter((pl.col('true') > .01) | (pl.col('false') > .01))
    print(act, 'amount/pot bins (share listed vs unlisted)'); print(w.head(40))
acts.write_parquet(ROOT + r'\Claude Version\experiments\fingerprint_20260917\member_actions_dev.parquet')
