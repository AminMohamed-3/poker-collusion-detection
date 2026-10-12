"""Step 168c (Claude): production of outputs_step168_20260915/prediction_step168.json after the dev read kept types (arch3_dev_eval.json).
FINAL3_32 key = mean over the STEP163 FINAL2_32 type columns (eval_views_FINAL2_32.parquet) and the kept ARCH3 types' 32-draw mean lg(r).
Two files, both never uploaded:
  submission_step168.csv            the STEP163 file (aebbf71b, .90563) with its tail re-assigned in FINAL3_32 key order (step151b rule, as registered)
  submission_step168_merge169.csv   the STEP169 boundary merge recomputed with the FINAL3_32 key on that file (N and M of prediction_step169_build.json)
Block, behaviour and evidence byte-identical to the STEP163 file except the moved boundary pairs in the second file. Usage: REGSHA"""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, sha, check_protected
import polars as pl
from pu_pairs import step159d_C_build as S159D
sys.path.insert(0, str(ORIGINAL / 'src'))
from validation import validate_submission  # noqa: E402

OUTD = CV / 'outputs_step168_20260915'
REG = OUTD / 'prediction_step168.json'
O163 = CV / 'outputs_step163_20260915'
BASE, BASE_SHA = O163 / 'submission_step163.csv', 'aebbf71b1224ddaba7225eeefeaa11ef52ec60687c12ca955e01dd64f429cc0f'
V163, V163_SHA = O163 / 'eval_views_FINAL2_32.parquet', 'd07e605b2c83227deec3a17c9083188b4a706e4c096b6b67cec11646c1f959ff'
REG169B, REG169B_SHA = CV / 'outputs_step169_20260915' / 'prediction_step169_build.json', '725f8c4fd9d39c1d72968a3635a63c10fc0878ce8495e83b3d4ff1bdbe3d2432'
A3 = CV / 'experiments' / 'arch3_views_20260915'
S159D.EVAL_DRAWS['ARCH3'] = (A3, 'arch3_eval_k')
DEST, DEST_M = OUTD / 'submission_step168.csv', OUTD / 'submission_step168_merge169.csv'


def write_checked(res, base, cols, dest, allowed_ids):
    out = res.sort('row').select(cols)
    bc = base.sort('row').select(cols)
    assert out.drop('risk_score').equals(bc.drop('risk_score')) and sorted(out['risk_score'].to_list()) == sorted(bc['risk_score'].to_list())
    assert not dest.exists()
    out.write_csv(dest)
    A, B = BASE.read_bytes().split(b'\n'), dest.read_bytes().split(b'\n'); assert len(A) == len(B)
    diff = [i for i in range(len(A)) if A[i] != B[i]]
    for i in diff:
        x, y = A[i].split(b','), B[i].split(b','); assert x[0] == y[0] and x[2:] == y[2:] and x[0].decode() in allowed_ids
    back = pl.read_csv(dest, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    assert back.filter(pl.col('risk_score') >= .5).height == 909
    val = validate_submission(dest, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
    assert not val.errors, val
    return {'output': str(dest), 'output_sha256': sha(dest), 'rows_changed_vs_step163': len(diff), 'validation': str(val)[:300]}


def main():
    t0 = time.time()
    reg_sha = sys.argv[1]
    assert sha(REG) == reg_sha and sha(BASE) == BASE_SHA and sha(V163) == V163_SHA and sha(REG169B) == REG169B_SHA
    dev = json.load(open(A3 / 'arch3_dev_eval.json', encoding='utf-8'))
    assert dev['registration_sha256'] == reg_sha
    kept = dev['kept_types']; assert kept, 'no type kept: no STEP168 file'
    r169 = json.load(open(REG169B, encoding='utf-8')); N, M = r169['amendment']['N'], r169['amendment']['M']
    prot = check_protected()
    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    V = pl.read_parquet(V163)
    kc2 = [c for c in V.columns if c not in ('pair_id', 'k_FINAL', 'k_FINAL2')]
    assert len(kc2) == 10 and V.select((pl.mean_horizontal(kc2) - pl.col('k_FINAL2')).abs().max()).item() < 1e-12
    E3 = S159D.eval_means('ARCH3', kept)
    W = V.join(E3, on='pair_id', how='inner', validate='1:1'); assert W.height == 112540
    W = W.with_columns(pl.mean_horizontal(kc2 + [f'ARCH3_{v}' for v in kept]).alias('k_FINAL3')).sort('pair_id')
    W.write_parquet(OUTD / 'eval_views_FINAL3_32.parquet')
    rec = {'registration_sha256': reg_sha, 'kept_types': kept, 'dev_FINAL3_minus_FINAL2_16': dev['FINAL3_minus_FINAL2_16'],
           'prediction_vs_step163_file': dev['prediction_vs_step163_file'], 'eval_views_FINAL3_32_sha256': sha(OUTD / 'eval_views_FINAL3_32.parquet'),
           'build_script_sha256': sha(os.path.abspath(__file__))}
    base = base.join(W.select('pair_id', 'k_FINAL2', 'k_FINAL3'), on='pair_id', how='left', validate='1:1', maintain_order='left')
    assert base['k_FINAL3'].null_count() == 0
    tail = base.filter(pl.col('r') < .5); blk = base.filter(pl.col('r') >= .5)
    assert tail.height == 111631 and blk.height == 909
    eqt = int(tail.group_by('risk_score').len().filter(pl.col('len') > 1)['len'].sum())
    got = tail.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list(); want = tail.sort(['k_FINAL2', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    rec['tail_lineage_mismatch_vs_FINAL2'] = sum(a != b for a, b in zip(got, want)); rec['tail_equal_value_rows'] = eqt
    assert rec['tail_lineage_mismatch_vs_FINAL2'] <= 2 * eqt
    # file 1: registered tail re-order
    assign, w = S159D.reassign(base, tail, 'k_FINAL3')
    res = base.join(assign, on='pair_id', how='left', maintain_order='left').with_columns(pl.coalesce('new_risk', 'risk_score').alias('risk_score'))
    rec['file_tail_reorder'] = write_checked(res, base, cols, DEST, set(assign['pair_id'].to_list()))
    from scipy.stats import spearmanr
    rec['tail_spearman_FINAL2_vs_FINAL3'] = round(float(spearmanr(tail['k_FINAL2'], tail['k_FINAL3'])[0]), 4)
    rec['tail_topn_overlap_FINAL2_vs_FINAL3'] = {str(n): round(len(set(want[:n]) & set(w[:n])) / n, 4) for n in (23, 100, 1000)}
    # file 2: merge on top of file 1 (block by base order, top-N tail by FINAL3 key, rest of tail in FINAL3 order)
    b1 = res.select('row', 'pair_id', 'risk_score').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r1')).join(base.select('pair_id', 'k_FINAL3'), on='pair_id')
    bo = b1.filter(pl.col('r1') >= .5).sort(['r1', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    to = b1.filter(pl.col('r1') < .5).sort(['k_FINAL3', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    assert len(bo) == 909 and set(bo) == set(blk['pair_id'].to_list())
    order = bo[:909 - M] + to[:N] + bo[909 - M:] + to[N:]
    vals = b1.sort('r1', descending=True)['risk_score'].to_list()
    new = pl.DataFrame({'pair_id': order, 'new_risk': vals})
    res2 = base.join(new, on='pair_id', how='left', validate='1:1', maintain_order='left').with_columns(pl.col('new_risk').alias('risk_score'))
    rec['file_merge169'] = write_checked(res2, base, cols, DEST_M, set(assign['pair_id'].to_list()) | set(bo[909 - M:]))
    rec['merge'] = {'N': N, 'M': M, 'topN_overlap_with_step169_file': len(set(to[:N]) & set(want[:N]))}
    assert check_protected() == prot
    rec.update({'protected_unchanged': True, 'seconds': round(time.time() - t0, 1)})
    (OUTD / 'receipt_step168.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
