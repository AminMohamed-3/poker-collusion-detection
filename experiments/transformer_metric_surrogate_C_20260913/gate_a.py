"""STEP111-G (poker-a8): necessity of gameplay type predicates on listed hands and zero-parameter demotion of decode(S*) picks.
Training-only audit (162 queries); no held read."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
S = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/cf441075-1502-4bb5-9e98-d102782e30e6/scratchpad/s109/'
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
from audit_common import *
from evidence_aug.step104_C_typed_decoder_build import decode, chronology, align, OUT as O104, AUDIT_SCORES, KEYS
from evidence_aug import step106_features as F6
R106 = 'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/residual_reranker_C_20260913/'
DATA = 'C:/Data/Projects/Money Making/work/poker/data/'
t0 = time.time()
prot = check_protected()
d = pl.read_parquet(AUDIT / 'audit-oof.parquet', columns=KEYS + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(KEYS)
sc = pl.read_parquet(AUDIT_SCORES, columns=KEYS + ['table_id', 'B0', 'prefix_strength']).sort(KEYS)
fb = chronology(sc.select(KEYS + ['table_id', 'B0']).rename({'B0': 'score'}))
assert fb.select(KEYS).equals(d.select(KEYS)) and sc.select(KEYS).equals(d.select(KEYS))
fs = fb.with_columns(pl.Series('score', sc['prefix_strength'].to_numpy()))
t1 = align(fb, pl.read_parquet(O104 / 'audit-type-attributes.parquet'))
B0, decS = fb['score'].to_numpy(), decode(fs, t1)
x = F6.candidate_features(fs.with_columns(d['relevance']), pl.read_parquet(R106 + 'audit-reranker-attributes.parquet'), t1, decS)
top = x.select(KEYS).unique().with_columns(pl.lit(True).alias('top20'))
assert top.height == 162 * 20
lab = pl.read_csv(DATA + 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
fr = (d.select(KEYS + ['behavior_family', 'relevance', 'chrono_rank']).with_row_index('row')
      .join(top, on=KEYS, how='left').with_columns(pl.col('top20').fill_null(False))
      .join(lab, on='pair_id', how='left').sort('row').with_columns(pl.Series('dec', decS)))
assert fr.height == d.height and fr['player_1'].null_count() == 0
print('rows', fr.height, 'pairs', fr['pair_id'].n_unique(), 'listed', int(fr['relevance'].sum()), 'listed in top20', int((fr['relevance'].cast(pl.Boolean) & fr['top20']).sum()),
      'decS range', float(decS.min()), float(decS.max()), flush=True)
# ---- part B: predicates from raw seats/actions (members = development_labels player_1/player_2)
g_h, g_s = ['pair_id', 'hand_id'], ['pair_id', 'hand_id', 'street']
hids = fr['hand_id'].unique().implode()
seats = pl.scan_parquet(DATA + 'seats.parquet').select('hand_id', 'player_id', 'net_chips').filter(pl.col('hand_id').is_in(hids)).collect()
acts = (pl.scan_parquet(DATA + 'actions.parquet').select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call')
        .filter(pl.col('hand_id').is_in(hids)).collect())
pm = fr.select('pair_id', 'hand_id', 'player_1', 'player_2')
net = (pm.join(seats.rename({'player_id': 'player_1', 'net_chips': 'n1'}), on=['hand_id', 'player_1'], how='left')
         .join(seats.rename({'player_id': 'player_2', 'net_chips': 'n2'}), on=['hand_id', 'player_2'], how='left'))
assert net.height == pm.height
print('seats', seats.height, 'actions', acts.height, 'member seat missing', int(net['n1'].null_count()), int(net['n2'].null_count()), flush=True)
a = (pm.join(acts, on='hand_id', how='inner').sort('pair_id', 'hand_id', 'action_no')
     .with_columns(m1=(pl.col('player_id') == pl.col('player_1')), m2=(pl.col('player_id') == pl.col('player_2')),
                   aggr=(pl.col('action').is_in(['bet', 'raise']) | ((pl.col('action') == 'all_in') & (pl.col('amount') > pl.col('to_call'))))))
a = a.with_columns(mem=(pl.col('m1') | pl.col('m2')))
a = a.with_columns(nm_act=pl.when(~pl.col('mem')).then(pl.col('action')).otherwise(pl.lit(None, dtype=pl.String)),
                   pos=pl.int_range(pl.len()).over(g_s),
                   cnm=(~pl.col('mem')).cast(pl.Int32).cum_sum().over(g_s),
                   n_aggr_before=(pl.col('aggr').cast(pl.Int32).cum_sum().over(g_s) - pl.col('aggr').cast(pl.Int32)))
a = a.with_columns(next_act=pl.col('action').shift(-1).over(g_h), next_mem=pl.col('mem').shift(-1).over(g_h),
                   next_nm=pl.col('nm_act').shift(-1).fill_null(strategy='backward').over(g_s),
                   f1=pl.when(pl.col('m1') & pl.col('aggr')).then(pl.col('pos')).otherwise(None).min().over(g_s),
                   f2=pl.when(pl.col('m2') & pl.col('aggr')).then(pl.col('pos')).otherwise(None).min().over(g_s))
a = a.with_columns(c1=pl.when(pl.col('pos') == pl.col('f1')).then(pl.col('cnm')).otherwise(None).max().over(g_s),
                   c2=pl.when(pl.col('pos') == pl.col('f2')).then(pl.col('cnm')).otherwise(None).max().over(g_s))
ma = pl.col('mem') & pl.col('aggr')
a = a.with_columns(r_ci_strict=(ma & (pl.col('next_act') == 'fold') & ~pl.col('next_mem')).fill_null(False),
                   r_ci_relaxed=(ma & (pl.col('next_nm') == 'fold')).fill_null(False))
a = a.with_columns(r_ci_reraise=(pl.col('r_ci_relaxed') & (pl.col('n_aggr_before') > 0)),
                   r_ci_sandwich=((pl.col('m2') & pl.col('aggr') & (pl.col('f1') < pl.col('pos')) & ((pl.col('cnm') - pl.col('c1')) > 0))
                                  | (pl.col('m1') & pl.col('aggr') & (pl.col('f2') < pl.col('pos')) & ((pl.col('cnm') - pl.col('c2')) > 0))).fill_null(False),
                   r_riv1=(pl.col('m1') & (pl.col('street') == 'river') & (pl.col('action') == 'check')),
                   r_riv2=(pl.col('m2') & (pl.col('street') == 'river') & (pl.col('action') == 'check')))
st = (a.filter(pl.col('street') != 'preflop').group_by(g_s)
      .agg(allmem=pl.col('mem').all(), has1=pl.col('m1').any(), has2=pl.col('m2').any())
      .group_by(g_h).agg(sp_hu_street=(pl.col('allmem') & pl.col('has1') & pl.col('has2')).any()))
hand = (a.group_by(g_h).agg(ci_strict=pl.col('r_ci_strict').any(), ci_relaxed=pl.col('r_ci_relaxed').any(), ci_reraise=pl.col('r_ci_reraise').any(),
                            ci_sandwich=pl.col('r_ci_sandwich').any(), riv1=pl.col('r_riv1').any(), riv2=pl.col('r_riv2').any())
        .with_columns(sp_river_check_any=(pl.col('riv1') | pl.col('riv2')), sp_river_check_both=(pl.col('riv1') & pl.col('riv2'))).drop('riv1', 'riv2'))
PRED = ['opp_sign', 'ci_strict', 'ci_relaxed', 'ci_reraise', 'ci_sandwich', 'sp_hu_street', 'sp_river_check_any', 'sp_river_check_both']
fr = (fr.join(net.select(g_h + ['n1', 'n2']), on=g_h, how='left').with_columns(opp_sign=(pl.col('n1') * pl.col('n2') < 0).fill_null(False))
      .join(hand, on=g_h, how='left').join(st, on=g_h, how='left')
      .with_columns([pl.col(c).fill_null(False).alias(c) for c in PRED]).sort('row'))
assert fr.height == d.height and fr.select(KEYS).equals(d.select(KEYS))
print('hands with no actions', fr.height - hand.height, '| overall rates', {c: round(float(fr[c].mean()), 4) for c in PRED}, f'{time.time() - t0:.0f}s', flush=True)
del a, acts, seats
exec(open(S + 'gate_c1.py', encoding='utf-8').read())
exec(open(S + 'gate_c2.py', encoding='utf-8').read())
