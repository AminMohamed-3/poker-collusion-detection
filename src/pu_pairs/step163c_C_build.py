"""Step 163c (Claude): production of outputs_step163_20260915/prediction_step163.json (sha 158a6e1b) after the dev read kept types
(arch2_dev_eval.json). FINAL2_32 key = mean over the STEP159 FINAL32 type columns (SXN3 five types + ARCH XGB3/ET/MLP, from
eval_views_FINAL32.parquet) and the kept ARCH2 types' 32-draw mean lg(r) from the step163a evaluation draws. The tail of the STEP159 file
(e4ca43de) gets its own risk strings in key order (ties by pair_id), the step151b rule. Block, behaviour and evidence byte-identical.
Never uploads."""
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

OUTD = CV / 'outputs_step163_20260915'
REG, REG_SHA = OUTD / 'prediction_step163.json', '158a6e1b705bcbd978e5f224a9d2e27b8a2b8be2f403cd9fd4f511b6ae08ebc0'
O159 = CV / 'outputs_step159_20260915'
BASE, BASE_SHA = O159 / 'submission_step159.csv', 'e4ca43de69ba1a9612d204bf24158e5c5f94f84b91eabfb5d52ccb3ff8d55a19'
V159, V159_SHA = O159 / 'eval_views_FINAL32.parquet', '4fecc2ba9c2aa2c152a4aea403f9f24caa4b1d1840413a71bc74010ac56b7c0f'
A2 = CV / 'experiments' / 'arch2_views_20260915'
S159D.EVAL_DRAWS['ARCH2'] = (A2, 'arch2_eval_k')
DEST = OUTD / 'submission_step163.csv'


def main():
    t0 = time.time()
    assert sha(REG) == REG_SHA and sha(BASE) == BASE_SHA and sha(V159) == V159_SHA and not DEST.exists()
    dev = json.load(open(A2 / 'arch2_dev_eval.json', encoding='utf-8'))
    assert dev['registration_sha256'] == REG_SHA
    kept = dev['kept_types']; assert kept, 'no type kept: no STEP163 file'
    r159 = json.load(open(O159 / 'receipt_step159.json', encoding='utf-8')); assert r159['output_sha256'] == BASE_SHA
    prot = check_protected()
    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    tail = base.filter(pl.col('r') < .5); assert tail.height == 111631 and base.height - tail.height == 909
    V = pl.read_parquet(V159)
    c159 = [f'SXN3_{v}' for v in S159D.VT] + [f'ARCH_{v}' for v in S159D.NEW_VT]
    assert set(c159 + ['pair_id', 'k_FINAL']) == set(V.columns), V.columns
    chk = V.select((pl.mean_horizontal(c159) - pl.col('k_FINAL')).abs().max()).item(); assert chk < 1e-12, chk
    rec = {'registration_sha256': REG_SHA, 'base_sha256': BASE_SHA, 'eval_views_FINAL32_sha256': V159_SHA, 'kept_types': kept,
           'dev_FINAL2_minus_FINAL16': dev['FINAL2_minus_FINAL16'], 'prediction_vs_step159_file': dev['prediction_vs_step159_file'],
           'build_script_sha256': sha(os.path.abspath(__file__))}
    # lineage: the STEP159 tail order equals the FINAL32 key (except inside equal-value groups)
    t = tail.join(V.select('pair_id', 'k_FINAL'), on='pair_id', how='left', validate='1:1', maintain_order='left'); assert t['k_FINAL'].null_count() == 0
    eqt = int(t.group_by('risk_score').len().filter(pl.col('len') > 1)['len'].sum())
    got = t.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list(); want = t.sort(['k_FINAL', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    rec['tail_lineage_mismatch'] = sum(a != b for a, b in zip(got, want)); rec['tail_equal_value_rows'] = eqt
    assert rec['tail_lineage_mismatch'] <= 2 * eqt, (rec['tail_lineage_mismatch'], eqt)
    E2 = S159D.eval_means('ARCH2', kept)
    W = V.join(E2, on='pair_id', how='inner', validate='1:1'); assert W.height == 112540
    kc = c159 + [f'ARCH2_{v}' for v in kept]
    W = W.with_columns(pl.mean_horizontal(kc).alias('k_FINAL2')).sort('pair_id')
    W.write_parquet(OUTD / 'eval_views_FINAL2_32.parquet'); rec['eval_views_FINAL2_32_sha256'] = sha(OUTD / 'eval_views_FINAL2_32.parquet')
    t = t.join(W.select('pair_id', 'k_FINAL2'), on='pair_id', how='left', validate='1:1', maintain_order='left'); assert t['k_FINAL2'].null_count() == 0
    assign, w = S159D.reassign(base, t, 'k_FINAL2')
    from scipy.stats import spearmanr
    rec['tail_spearman_FINAL32_vs_FINAL2_32'] = round(float(spearmanr(t['k_FINAL'], t['k_FINAL2'])[0]), 4)
    rec['tail_topn_overlap_FINAL32_vs_FINAL2_32'] = {str(n): round(len(set(want[:n]) & set(w[:n])) / n, 4) for n in (100, 1000, 5000)}
    res = base.join(assign, on='pair_id', how='left', maintain_order='left').with_columns(pl.coalesce('new_risk', 'risk_score').alias('risk_score')).sort('row').select(cols)
    bc = base.sort('row').select(cols)
    assert res.drop('risk_score').equals(bc.drop('risk_score')) and sorted(res['risk_score'].to_list()) == sorted(bc['risk_score'].to_list())
    res.write_csv(DEST)
    A, B = BASE.read_bytes().split(b'\n'), DEST.read_bytes().split(b'\n'); assert len(A) == len(B)
    changed_ids = set(assign['pair_id'].to_list()); diff = [i for i in range(len(A)) if A[i] != B[i]]
    for i in diff:
        x, y = A[i].split(b','), B[i].split(b','); assert x[0] == y[0] and x[2:] == y[2:] and x[0].decode() in changed_ids
    back = pl.read_csv(DEST, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    assert back.filter(pl.col('risk_score') >= .5).height == 909
    val = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
    assert check_protected() == prot
    rec.update({'output': str(DEST), 'output_sha256': sha(DEST), 'rows_changed_vs_base': len(diff), 'only_tail_risk_cells_changed': True, 'value_multiset_unchanged': True,
                'validation': json.loads(json.dumps(val, default=str)), 'protected_unchanged': True, 'seconds': round(time.time() - t0, 1)})
    (OUTD / 'receipt_step163.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k != 'validation'}, indent=1)); print('validation', str(val)[:300])


if __name__ == '__main__':
    main()
