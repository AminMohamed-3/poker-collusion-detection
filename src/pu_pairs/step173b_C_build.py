"""Step 173b (Claude): production of outputs_step173_20260915/prediction_step173.json (dev primary interleave N10_M50 passed ER3).
On the .90821 file (3ee17909 = STEP167 LP evidence on the STEP169 merge): block order and tail order from the STEP163 file (aebbf71b, the
file STEP169 was built on); the union of the M = 159 lowest block pairs and the N = 23 highest tail pairs by the eval FINAL2_32 key is
ordered by that key; the sorted risk strings are handed out in the new order. Gate: the STEP169 merge order from the same inputs
reproduces the risk column of the .90821 file exactly. Only risk cells inside the union change; evidence and behaviour byte-identical.
Never uploads. Usage: REGSHA"""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, sha, check_protected
import polars as pl
sys.path.insert(0, str(ORIGINAL / 'src'))
from validation import validate_submission  # noqa: E402

OUTD = CV / 'outputs_step173_20260915'
REG = OUTD / 'prediction_step173.json'
PRE, PRE_SHA = CV / 'outputs_step163_20260915' / 'submission_step163.csv', 'aebbf71b1224ddaba7225eeefeaa11ef52ec60687c12ca955e01dd64f429cc0f'
BASE, BASE_SHA = CV / 'outputs_step167_20260915' / 'submission_step167_lp_on_step169.csv', '3ee17909242f0d7ae2aaa30c3f26db401a2b2011a683e00b13ed2805905fec40'
KEY, KEY_SHA = CV / 'outputs_step163_20260915' / 'eval_views_FINAL2_32.parquet', 'd07e605b2c83227deec3a17c9083188b4a706e4c096b6b67cec11646c1f959ff'
REG169B, REG169B_SHA = CV / 'outputs_step169_20260915' / 'prediction_step169_build.json', '725f8c4fd9d39c1d72968a3635a63c10fc0878ce8495e83b3d4ff1bdbe3d2432'
DEST = OUTD / 'submission_step173_interleave_on_90821.csv'


def main():
    t0 = time.time()
    reg_sha = sys.argv[1]
    assert sha(REG) == reg_sha and sha(PRE) == PRE_SHA and sha(BASE) == BASE_SHA and sha(KEY) == KEY_SHA and sha(REG169B) == REG169B_SHA
    assert not DEST.exists()
    dev = json.load(open(CV / 'experiments/boundary_merge_20260915/interleave173.json', encoding='utf-8'))
    assert dev['registration_sha256'] == reg_sha and dev['primary']['passes']
    r169 = json.loads(REG169B.read_text(encoding='utf-8')); N, M = r169['amendment']['N'], r169['amendment']['M']
    assert (N, M) == (23, 159)
    prot = check_protected()
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    pre = pl.read_csv(PRE, infer_schema=False, columns=['pair_id', 'risk_score']).with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    pre = pre.join(pl.read_parquet(KEY, columns=['pair_id', 'k_FINAL2']), on='pair_id', how='left', validate='1:1', maintain_order='left')
    assert pre['k_FINAL2'].null_count() == 0
    blk = pre.filter(pl.col('r') >= .5).sort(['r', 'pair_id'], descending=[True, False])
    tail = pre.filter(pl.col('r') < .5).sort(['r', 'pair_id'], descending=[True, False])
    B = blk.height; assert B == 909 and tail.height == 111631
    bo, cur = blk['pair_id'].to_list(), tail['pair_id'].to_list()
    top = tail.sort(['k_FINAL2', 'pair_id'], descending=[True, False])['pair_id'].to_list()[:N]
    topset = set(top)
    rest = [p for p in cur if p not in topset]
    vals = pre.sort('r', descending=True)['risk_score'].to_list()
    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row')
    # gate: STEP169 merge order reproduces the .90821 risk column
    merge_order = bo[:B - M] + top + bo[B - M:] + rest
    g = base.select('row', 'pair_id', 'risk_score').join(pl.DataFrame({'pair_id': merge_order, 'm': vals}), on='pair_id', how='left', validate='1:1')
    gate_merge = int((g['m'] != g['risk_score']).sum())
    assert gate_merge == 0, gate_merge
    U = pre.filter(pl.col('pair_id').is_in(bo[B - M:] + top)).sort(['k_FINAL2', 'pair_id'], descending=[True, False])
    assert U.height == M + N
    order = bo[:B - M] + U['pair_id'].to_list() + rest
    assert len(order) == pre.height and len(set(order)) == pre.height
    res = base.join(pl.DataFrame({'pair_id': order, 'new_risk': vals}), on='pair_id', how='left', validate='1:1', maintain_order='left').sort('row')
    moved = res.filter(pl.col('new_risk') != pl.col('risk_score'))
    out = res.with_columns(pl.col('new_risk').alias('risk_score')).select(cols)
    bc = base.sort('row').select(cols)
    assert out.drop('risk_score').equals(bc.drop('risk_score')) and sorted(out['risk_score'].to_list()) == sorted(bc['risk_score'].to_list())
    assert set(moved['pair_id'].to_list()) <= set(U['pair_id'].to_list())
    out.write_csv(DEST)
    A_, B_ = BASE.read_bytes().split(b'\n'), DEST.read_bytes().split(b'\n'); assert len(A_) == len(B_)
    diff = [i for i in range(len(A_)) if A_[i] != B_[i]]
    for i in diff:
        x, y = A_[i].split(b','), B_[i].split(b','); assert x[0] == y[0] and x[2:] == y[2:]
    back = pl.read_csv(DEST, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    assert back.filter(pl.col('risk_score') >= .5).height == 909
    above = set(back.filter(pl.col('risk_score') >= .5)['pair_id'].to_list()); above90821 = set(base.filter(pl.col('risk_score').cast(pl.Float64) >= .5)['pair_id'].to_list())
    val = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
    assert not val.errors, val
    assert check_protected() == prot
    ub = U.with_columns(pl.col('pair_id').is_in(bo[B - M:]).alias('from_block'))
    rec = {'registration_sha256': reg_sha, 'N': N, 'M': M, 'gate_step169_merge_replay_mismatches': gate_merge,
           'union_key_order_from_block_in_top23_positions': int(ub['from_block'][:N].sum()),
           'union_block_pairs_moving_below_half': len((set(bo[B - M:]) & above90821) - above), 'union_tail_pairs_now_at_or_above_half': len(topset & above),
           'set_at_or_above_half_changed_vs_90821': len(above ^ above90821) // 2,
           'pairs_with_changed_risk_vs_90821': moved.height, 'rows_changed_vs_90821': len(diff), 'only_risk_cells_changed': True, 'value_multiset_unchanged': True,
           'output': str(DEST), 'output_sha256': sha(DEST), 'validation': str(val)[:300], 'protected_unchanged': True,
           'build_script_sha256': sha(os.path.abspath(__file__)), 'seconds': round(time.time() - t0, 1)}
    (OUTD / 'receipt_step173.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
