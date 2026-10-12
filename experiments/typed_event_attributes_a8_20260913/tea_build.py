"""poker-a8 typed-event attributes for the STEP112 interface, audit split. Label-free: keys are development-phase hands where both pair members
hold a seat; members come from development_labels player_1/player_2 only (label, label_status, behavior_family never read)."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
E = 'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/typed_event_attributes_a8_20260913/'
CV = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
for p in (E, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src', CV + 'src'):
    sys.path.insert(0, p)
import polars as pl
from audit_common import check_protected
from tea_scan import load, scan, DATA
from tea_equity import equity_fields
from evidence_aug import step106_attrs as A6
from evidence_aug.step104_C_typed_decoder_build import type_attributes
KEYS = ['pair_id', 'hand_id']
T104 = CV + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'
G111 = CV + 'experiments/transformer_metric_surrogate_C_20260913/gate111G_predicates.parquet'
t0, prot0 = time.time(), check_protected()
d = pl.read_parquet(CV + 'experiments/label_process_20260911/audit-oof.parquet', columns=KEYS).sort(KEYS)
lab = pl.read_csv(DATA + 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2']).filter(pl.col('pair_id').is_in(d['pair_id'].unique().implode()))
pp = pl.concat([lab['player_1'], lab['player_2']]).unique().implode()
se = pl.scan_parquet(DATA + 'seats.parquet').select('hand_id', 'player_id').filter(pl.col('player_id').is_in(pp)).collect()
hh = pl.scan_parquet(DATA + 'hands.parquet').filter(pl.col('phase') == 'development').select('hand_id', 'table_id', 'started_at').collect()
keys = (lab.join(se.rename({'player_id': 'player_1'}), on='player_1').join(se.rename({'player_id': 'player_2'}), on=['player_2', 'hand_id'])
        .join(hh, on='hand_id').sort(KEYS))
assert keys.select(KEYS).equals(d) and keys.select(KEYS).is_unique().all() and lab.height == 162
keys = keys.with_columns(started_at_us=pl.col('started_at').dt.epoch('us'), t=(pl.col('started_at').rank('min').over('pair_id') - 1).cast(pl.Int64),
                         n_shared=pl.len().over('pair_id').cast(pl.Int64), t_tied=pl.col('started_at').is_duplicated().over('pair_id'))
keys = keys.with_columns(t_frac=pl.col('t') / pl.col('n_shared'))
members = {p: (u, v) for p, u, v in lab.iter_rows()}
top = pl.read_parquet(T104, columns=KEYS).sort(KEYS)
topset = set(top.iter_rows())
hands, seats, acts = load(keys['hand_id'].unique().sort().to_list())
print('keys', keys.height, 'top20', len(topset), 'ties', int(keys['t_tied'].sum()), f'load {time.time() - t0:.0f}s', flush=True)
rows, n = [], 0
for pid, hid in keys.select(KEYS).iter_rows():
    a, b = members[pid]
    h, st, ac = hands[hid], seats[hid], acts[hid]
    r, ev, w, l = scan(h, st, ac, a, b)
    r.update(A6.outsider_row(h, st, ac, a, b))
    r.update(pair_id=pid, hand_id=hid, top20=(pid, hid) in topset)
    if r['top20']:
        r.update(equity_fields(h, st, ac, a, b, w, l, ev))
        n += 1
        if n % 500 == 0:
            print('equity rows', n, f'{time.time() - t0:.0f}s', flush=True)
    rows.append(r)
att = pl.from_dicts(rows, infer_schema_length=None)
x = keys.drop('player_1', 'player_2', 'started_at').join(att, on=KEYS, how='left', validate='1:1')
x = x.sort('pair_id', 't').with_columns(tf=pl.when(pl.col('facing_partner_any')).then(pl.col('t')).otherwise(None))
x = x.with_columns(hands_since_prev_facing=pl.col('t') - pl.col('tf').shift(1).forward_fill().over('pair_id')).drop('tf')
g6 = type_attributes(top, members)
g6_exact = bool(g6.equals(pl.read_parquet(T104).sort(KEYS)))
g111 = pl.read_parquet(G111).drop('top20', 'union')
opp_same = bool(x.select(KEYS + ['opp_sign']).sort(KEYS).equals(g111.select(KEYS + ['opp_sign']).sort(KEYS)))
x = (x.join(g6.rename({'facing': 'g6_facing', 'fold_street': 'g6_fold_street', 'folder_equity': 'g6_folder_equity'}), on=KEYS, how='left', validate='1:1')
     .join(g111.drop('opp_sign'), on=KEYS, how='left', validate='1:1').sort(KEYS))
x.write_parquet(E + 'attrs-audit.parquet')
json.dump({'rows': x.height, 'cols': x.width, 'g6_exact_vs_step104': g6_exact, 'opp_sign_equals_111G': opp_same,
           'protected_unchanged': check_protected() == prot0, 'seconds': round(time.time() - t0, 1)}, open(E + 'build_run.json', 'w', encoding='utf-8'), indent=2)
print('rows', x.height, 'cols', x.width, 'g6 exact', g6_exact, 'opp same', opp_same, f'{time.time() - t0:.0f}s', flush=True)
