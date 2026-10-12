"""B4: build the STEP272 submission = the current best file with the 909-pair block re-ordered by the
M3xKM blend (M3x + the FINAL2 tail key + ARCH2_MLP2).  Built on STEP259b directly, so this ONE file carries
both the STEP270 and the STEP272 gains and supersedes the STEP270 file as the candidate.

Lineage gate (G4): re-deriving the CURRENT file's global order from
    M3x_order[:750] + the 23 STEP169-promoted pairs + M3x_order[750:909] + remaining tail
and handing out the base value multiset must reproduce the base file byte for byte.
Writes nothing outside the scratchpad unless WRITE=1.
"""
import os, sys, json, time, hashlib
os.environ['POLARS_MAX_THREADS'] = '4'
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/pair/')
import numpy as np, polars as pl

BASE = C + 'outputs_step259b_20260918/submission_step259b_geofusion_dt_sp_on_step251.csv'
KEYF = C + 'outputs_step163_20260915/eval_views_FINAL2_32.parquet'
NEW = pl.read_parquet(SC + 'block_key_M3xKM_16.parquet')          # pair_id, score_M3xKM, score_M3x
rec = {}
t0 = time.time()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


base_sha = sha(BASE)
rec['base'] = {'path': BASE, 'sha256': base_sha}
print('base sha', base_sha[:8], flush=True)
assert base_sha.startswith('5cd66739'), base_sha

cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
base = (pl.read_csv(BASE, infer_schema=False).with_row_index('row')
        .with_columns(pl.col('risk_score').cast(pl.Float64).alias('r')))
base = base.join(pl.read_parquet(KEYF, columns=['pair_id', 'k_FINAL2']), on='pair_id',
                 how='left', validate='1:1', maintain_order='left')
assert base['k_FINAL2'].null_count() == 0 and base.height == 112540
print('rows', base.height, flush=True)

blk909 = NEW['pair_id'].to_list()
assert len(blk909) == 909 and len(set(blk909)) == 909
cur_order = base.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list()
top932 = cur_order[:932]
promoted = [p for p in top932 if p not in set(blk909)]
rec['n_promoted_found'] = len(promoted)
print('promoted pairs found in top-932 but not in the M3x block:', len(promoted), flush=True)
assert len(promoted) == 23, promoted

sM3x = dict(zip(NEW['pair_id'].to_list(), NEW['score_M3x'].to_list()))
sNEW = dict(zip(NEW['pair_id'].to_list(), NEW['score_M3xKM'].to_list()))
promset = set(promoted)
rest_tail = [p for p in cur_order if p not in set(blk909) and p not in promset]
vals = base.sort('r', descending=True)['risk_score'].to_list()


def compose(scoremap):
    o = sorted(blk909, key=lambda p: (-scoremap[p], p))
    return o[:750] + promoted + o[750:] + rest_tail


def assign(order):
    assert len(order) == base.height and len(set(order)) == base.height
    new = pl.DataFrame({'pair_id': order, 'new_risk': vals})
    res = base.join(new, on='pair_id', how='left', validate='1:1', maintain_order='left').sort('row')
    return res.with_columns(pl.col('new_risk').alias('risk_score')).select(cols)


old_rebuilt = assign(compose(sM3x))
base_cols = base.sort('row').select(cols)
G4 = old_rebuilt.equals(base_cols)
rec['G4_lineage_rebuild_reproduces_base'] = bool(G4)
print('G4 lineage rebuild reproduces the base file:', G4, flush=True)
assert G4, 'lineage does not reproduce; do not build'

out = assign(compose(sNEW))
assert out.drop('risk_score').equals(base_cols.drop('risk_score'))
assert sorted(out['risk_score'].to_list()) == sorted(base_cols['risk_score'].to_list())
changed = [i for i, (a, b) in enumerate(zip(out['risk_score'].to_list(), base_cols['risk_score'].to_list())) if a != b]
rec['pairs_with_changed_risk'] = len(changed)
moved_ids = set(base_cols['pair_id'].to_list()[i] for i in changed)
rec['changed_all_inside_block'] = bool(moved_ids <= set(blk909))
print('risk cells changed:', len(changed), ' all inside the 909 block:', rec['changed_all_inside_block'], flush=True)

nb = pl.read_csv(BASE, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
after = out.select('pair_id', pl.col('risk_score').cast(pl.Float64).alias('r2'))
j = nb.join(after, on='pair_id', validate='1:1')
assert int((j['r2'] >= .5).sum()) == 909
rec['block_size_after'] = 909
ro = np.argsort(np.argsort(-j['risk_score'].to_numpy()))
rn = np.argsort(np.argsort(-j['r2'].to_numpy()))
mv = rn - ro
rec['global'] = {'spearman': float(np.corrcoef(ro, rn)[0, 1]),
                 'mean_abs_move_changed': float(np.abs(mv[np.abs(mv) > 0]).mean()),
                 'max_abs_move': int(np.abs(mv).max()), 'n_moved': int((mv != 0).sum())}
print('global order: spearman %.6f  moved %d  mean|move| %.1f  max %d'
      % (rec['global']['spearman'], rec['global']['n_moved'],
         rec['global']['mean_abs_move_changed'], rec['global']['max_abs_move']), flush=True)

if os.environ.get('WRITE') == '1':
    DESTD = C + 'outputs_step272_20260919/'
    os.makedirs(DESTD, exist_ok=True)
    DEST = DESTD + 'submission_step272_blockM3xKM_on_step259b.csv'
    assert not os.path.exists(DEST), DEST
    out.write_csv(DEST)
    A_, B_ = open(BASE, 'rb').read().split(b'\n'), open(DEST, 'rb').read().split(b'\n')
    assert len(A_) == len(B_)
    diff = [i for i in range(len(A_)) if A_[i] != B_[i]]
    for i in diff:
        x, y = A_[i].split(b','), B_[i].split(b',')
        assert x[0] == y[0] and x[2:] == y[2:]           # only the risk_score cell differs
    rec['rows_changed_vs_base'] = len(diff)
    rec['only_risk_cells_changed'] = True
    rec['output'] = DEST
    rec['output_sha256'] = sha(DEST)
    sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/src')
    from validation import validate_submission
    val = validate_submission(DEST,
                              'C:/Data/Projects/Money Making/work/poker/data/evaluation_pairs.csv',
                              'C:/Data/Projects/Money Making/work/poker/data/hands.parquet',
                              'C:/Data/Projects/Money Making/work/poker/data/seats.parquet',
                              'C:/Data/Projects/Money Making/work/poker/data/sample_submission.csv')
    rec['validation_errors'] = list(val.errors) if getattr(val, 'errors', None) else []
    print('WROTE', DEST, flush=True)
    print('sha', rec['output_sha256'][:8], 'rows changed', len(diff),
          'validator errors', rec['validation_errors'], flush=True)

rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'b4_build.json', 'w'), indent=1)
print('done', rec['seconds'], 's', flush=True)
