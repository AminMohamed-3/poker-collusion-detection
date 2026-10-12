"""Step 156c (Claude): Part B production of outputs_step156_20260915/prediction_step156.json (sha fba91dde), run only if the registered
Part B rule passed (experiments/tail_keys_20260915_neg/sxn_dev_eval.json). (1) merge evaluation draws k = 0..31 into k_SXN;
(2) re-order the tail (risk < .5) of the Part A file (submission_step156_blockM3x_tailSX32.csv) by k_SXN with the step151b rule.
Block rows, behaviour and evidence byte-identical. Never uploads."""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, sha, check_protected
import numpy as np, polars as pl
sys.path.insert(0, str(ORIGINAL / 'src'))
from validation import validate_submission  # noqa: E402

OUTD = CV / 'outputs_step156_20260915'
REG, REG_SHA = OUTD / 'prediction_step156.json', 'fba91ddedfcb556ef2be83f0c6ea5a3edea57b703de150479292bcc374a7e43d'
TN = CV / 'experiments' / 'tail_keys_20260915_neg'
BASE = OUTD / 'submission_step156_blockM3x_tailSX32.csv'
SX32 = CV / 'experiments' / 'tail_keys_20260914' / 'eval_views_SX32.parquet'
DEST = OUTD / 'submission_step156_blockM3x_tailSXN32.csv'
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def merge(K=32):
    d = pl.concat([pl.read_parquet(TN / f) for f in sorted(os.listdir(TN)) if f.startswith('sxn_eval_k') and f.endswith('.parquet')]).filter(pl.col('k') < K)
    assert sorted(d['k'].unique().to_list()) == list(range(K))
    n = d['pair_id'].n_unique(); assert n == 112540 and d.height == n * K, (n, d.height)
    assert all(bool(d[f'r_{v}'].is_finite().all()) for v in VT)
    L = d.with_columns([pl.Series(f'L_{v}', lg(d[f'r_{v}'])) for v in VT]).group_by('pair_id').agg([pl.col(f'L_{v}').mean() for v in VT])
    out = L.with_columns(pl.mean_horizontal([f'L_{v}' for v in VT]).alias('k_SXN')).sort('pair_id')
    path = TN / f'eval_views_SXN{K}.parquet'; out.write_parquet(path)
    return out, sha(path)


def main():
    t0 = time.time()
    assert sha(REG) == REG_SHA
    rule = json.load(open(TN / 'sxn_dev_eval.json', encoding='utf-8'))['rule']
    assert rule['SXN16_REPLACES_SX16'], 'Part B rule did not pass: no SXN file'
    base_sha = json.load(open(OUTD / 'receipt_step156a.json', encoding='utf-8'))['output_sha256']
    assert sha(BASE) == base_sha and not DEST.exists()
    prot = check_protected()
    views, views_sha = merge(32)
    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    tail = base.filter(pl.col('r') < .5); assert tail.height == 111631
    t = tail.join(views.select('pair_id', 'k_SXN'), on='pair_id', how='left', validate='1:1', maintain_order='left')
    t = t.join(pl.read_parquet(SX32, columns=['pair_id', 'k_SX']), on='pair_id', how='left', validate='1:1', maintain_order='left')
    assert t['k_SXN'].null_count() == 0 and t['k_SX'].null_count() == 0
    got = t.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    want_sx = t.sort(['k_SX', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    eqg = int(t.group_by('risk_score').len().filter(pl.col('len') > 1)['len'].sum())
    lineage = sum(a != b for a, b in zip(got, want_sx)); assert lineage <= 2 * eqg, (lineage, eqg)
    values = t.sort(['r', 'pair_id'], descending=[True, False])['risk_score'].to_list()
    order = t.sort(['k_SXN', 'pair_id'], descending=[True, False]).select('pair_id', 'k_SXN')
    assign = order.with_columns(pl.Series('new_risk', values)).select('pair_id', 'new_risk')
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    res = base.join(assign, on='pair_id', how='left', maintain_order='left').with_columns(pl.coalesce('new_risk', 'risk_score').alias('risk_score')).sort('row').select(cols)
    bc = base.sort('row').select(cols)
    assert res.drop('risk_score').equals(bc.drop('risk_score')) and sorted(res['risk_score'].to_list()) == sorted(bc['risk_score'].to_list())
    res.write_csv(DEST)
    A, B = BASE.read_bytes().split(b'\n'), DEST.read_bytes().split(b'\n'); assert len(A) == len(B)
    tail_ids = set(tail['pair_id'].to_list()); diff = [i for i in range(len(A)) if A[i] != B[i]]
    for i in diff:
        x, y = A[i].split(b','), B[i].split(b','); assert x[0] == y[0] and x[2:] == y[2:] and x[0].decode() in tail_ids
    back = pl.read_csv(DEST, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    bt = back.join(tail.select('pair_id'), on='pair_id', how='semi'); assert float(bt['risk_score'].max()) < .5
    got2 = bt.sort(['risk_score', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    mismatch = sum(a != b for a, b in zip(got2, order['pair_id'].to_list()))
    from scipy.stats import spearmanr
    w = order['pair_id'].to_list()
    overlap = {str(n): round(len(set(want_sx[:n]) & set(w[:n])) / n, 4) for n in (100, 1000, 5000)}
    val = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
    assert check_protected() == prot
    rec = {'registration_sha256': REG_SHA, 'base': str(BASE), 'base_sha256': base_sha, 'views_SXN32_sha256': views_sha, 'output': str(DEST), 'output_sha256': sha(DEST),
           'tail_lineage_mismatch_vs_SX32': lineage, 'equal_value_group_rows': eqg, 'rows_changed_vs_base': len(diff), 'only_tail_risk_cells_changed': True,
           'order_mismatch_vs_key': mismatch, 'tail_spearman_SX32_vs_SXN32': round(float(spearmanr(t['k_SX'], t['k_SXN'])[0]), 4), 'tail_topn_overlap_SX32_vs_SXN32': overlap,
           'validation': json.loads(json.dumps(val, default=str)), 'protected_unchanged': True, 'seconds': round(time.time() - t0, 1)}
    (OUTD / 'receipt_step156c.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k != 'validation'}, indent=1)); print('validation', str(val)[:300])


if __name__ == '__main__':
    main()
