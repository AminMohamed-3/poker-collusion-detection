"""Step 169c (Claude): production of outputs_step169_20260915/prediction_step169_build.json (STEP169 boundary merge, dev primary N10_M50
passed ER3). On the STEP163 file (aebbf71b, public .90563): the N highest tail pairs by the FINAL2_32 key are placed directly above the M
lowest block pairs; the base file's risk strings are handed out by value in the new order. N, M scaled to production size in the build
registration. Behaviour and evidence byte-identical. Never uploads. Usage: REGSHA"""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, sha, check_protected
import numpy as np, polars as pl
sys.path.insert(0, str(ORIGINAL / 'src'))
from validation import validate_submission  # noqa: E402

OUTD = CV / 'outputs_step169_20260915'
REG = OUTD / 'prediction_step169_build.json'
DEST = OUTD / 'submission_step169.csv'


def main():
    t0 = time.time()
    reg_sha = sys.argv[1]
    assert sha(REG) == reg_sha and not DEST.exists()
    reg = json.loads(REG.read_text(encoding='utf-8'))
    BASE, KEY = CV.parent / reg['base']['path'], CV.parent / reg['key']['path']
    assert sha(BASE) == reg['base']['sha256'] and sha(KEY) == reg['key']['sha256']
    N, M = reg['amendment']['N'], reg['amendment']['M']
    prot = check_protected()
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    base = base.join(pl.read_parquet(KEY, columns=['pair_id', 'k_FINAL2']), on='pair_id', how='left', validate='1:1', maintain_order='left')
    assert base['k_FINAL2'].null_count() == 0
    blk = base.filter(pl.col('r') >= .5).sort(['r', 'pair_id'], descending=[True, False])
    tail = base.filter(pl.col('r') < .5).sort(['r', 'pair_id'], descending=[True, False])
    B = blk.height; assert B == 909 and tail.height == 111631
    bykey = tail.sort(['k_FINAL2', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    cur = tail['pair_id'].to_list()
    eqt = int(tail.group_by('risk_score').len().filter(pl.col('len') > 1)['len'].sum())
    mism = sum(a != b for a, b in zip(cur, bykey))
    assert mism <= 2 * eqt, (mism, eqt)
    top = bykey[:N]
    top_eq_current = len(set(top) & set(cur[:N]))
    topset = set(top)
    order = blk['pair_id'].to_list()[:B - M] + top + blk['pair_id'].to_list()[B - M:] + [p for p in cur if p not in topset]
    assert len(order) == base.height and len(set(order)) == base.height
    vals = base.sort('r', descending=True)['risk_score'].to_list()
    new = pl.DataFrame({'pair_id': order, 'new_risk': vals})
    res = base.join(new, on='pair_id', how='left', validate='1:1', maintain_order='left').sort('row')
    moved = res.filter(pl.col('new_risk') != pl.col('risk_score'))
    out = res.with_columns(pl.col('new_risk').alias('risk_score')).select(cols)
    bc = base.sort('row').select(cols)
    assert out.drop('risk_score').equals(bc.drop('risk_score')) and sorted(out['risk_score'].to_list()) == sorted(bc['risk_score'].to_list())
    assert set(moved['pair_id'].to_list()) <= topset | set(blk['pair_id'].to_list()[B - M:])
    out.write_csv(DEST)
    A_, B_ = BASE.read_bytes().split(b'\n'), DEST.read_bytes().split(b'\n'); assert len(A_) == len(B_)
    diff = [i for i in range(len(A_)) if A_[i] != B_[i]]
    for i in diff:
        x, y = A_[i].split(b','), B_[i].split(b','); assert x[0] == y[0] and x[2:] == y[2:]
    back = pl.read_csv(DEST, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64}).join(base.select('pair_id', 'r', 'k_FINAL2'), on='pair_id')
    assert back.filter(pl.col('risk_score') >= .5).height == 909
    rk = back.sort(['risk_score', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    assert rk[B - M:B - M + N] == top or set(rk[B - M:B - M + N]) == topset
    now_tail = set(back.filter(pl.col('risk_score') < .5)['pair_id'].to_list())
    val = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
    assert not val.errors, val
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'base_sha256': reg['base']['sha256'], 'key_sha256': reg['key']['sha256'], 'N': N, 'M': M,
           'tail_lineage_mismatch': mism, 'tail_equal_value_rows': eqt, 'topN_by_key_equal_topN_by_base_tail_order': top_eq_current,
           'topN_key_range': [float(tail.filter(pl.col('pair_id').is_in(top))['k_FINAL2'].min()), float(tail.filter(pl.col('pair_id').is_in(top))['k_FINAL2'].max())],
           'topN_base_risk_range': [float(tail.filter(pl.col('pair_id').is_in(top))['r'].min()), float(tail.filter(pl.col('pair_id').is_in(top))['r'].max())],
           'lowest_M_block_risk_range': [float(blk['r'][B - M:].min()), float(blk['r'][B - M:].max())],
           'lowest_M_block_key_mean': float(blk['k_FINAL2'][B - M:].mean()), 'topN_key_mean': float(tail.filter(pl.col('pair_id').is_in(top))['k_FINAL2'].mean()),
           'block_pairs_now_below_half': len(set(blk['pair_id'].to_list()) & now_tail), 'tail_pairs_now_at_or_above_half': len(topset - now_tail),
           'pairs_with_changed_risk': moved.height, 'rows_changed_vs_base': len(diff), 'only_risk_cells_changed': True, 'value_multiset_unchanged': True,
           'output': str(DEST), 'output_sha256': sha(DEST), 'validation': str(val)[:300], 'protected_unchanged': True,
           'build_script_sha256': sha(os.path.abspath(__file__)), 'seconds': round(time.time() - t0, 1)}
    (OUTD / 'receipt_step169.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
