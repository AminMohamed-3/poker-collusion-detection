"""Step 156a (Claude): Part A of outputs_step156_20260915/prediction_step156.json (sha fba91dde). Re-order the 909-row evaluation block of the
SX32 file (6f1f967d) by the step 64f M3 blend refitted with 32-draw SHORT3/ALL5 (M3x). Gates: the unmodified refit reproduces
multiview_M3_fit.json and its evaluation order reproduces the base block order. Only block risk cells change. Never uploads.
Output: outputs_step156_20260915/{submission_step156_blockM3x_tailSX32.csv, receipt_step156a.json, block_key_M3x.parquet}."""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
import sys, json, time, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, sha, check_protected
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import testbed, final_fit, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import predict
from pu_pairs.step62_multiview_candidate import VD, eval_frame
sys.path.insert(0, str(ORIGINAL / 'src'))
from validation import validate_submission  # noqa: E402

OUTD = CV / 'outputs_step156_20260915'
REG, REG_SHA = OUTD / 'prediction_step156.json', 'fba91ddedfcb556ef2be83f0c6ea5a3edea57b703de150479292bcc374a7e43d'
BASE, BASE_SHA = CV / 'outputs_v1_followups_20260914' / 'submission_v1_ci_untyped_tail_SX32.csv', '6f1f967d5ee6f7ebe60e85efab6a1b9f3aa612bbf6575c4926f03b0c7e3e4d6e'
TK = CV / 'experiments' / 'tail_keys_20260914'
EVX, EVX_SHA = TK / 'eval_views_SX32.parquet', '2ea25cb790b9222145929d90e56394f0a80f8987c4cdc17e147b7566689d021e'
DEST = OUTD / 'submission_step156_blockM3x_tailSX32.csv'


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def dev_views32(d):
    """Fold-matched 32-draw logit means at prefix-60 for the testbed rows (fold chosen where draw 0 equals the stored single-draw views)."""
    SX = pl.concat([pl.read_parquet(TK / f) for f in sorted(os.listdir(TK)) if f.startswith('sx_dev_fold') and f.endswith('.parquet')]).filter(pl.col('surface') == 'prefix60')
    assert SX.select('fold', 'k').unique().height == 64
    k0 = SX.filter(pl.col('k') == 0).select('pair_id', 'fold', pl.col('r_SHORT3').alias('k0_S3'), pl.col('r_ALL5').alias('k0_A5'))
    j = d.select('pair_id', 'r_SHORT3', 'r_ALL5').join(k0, on='pair_id', how='left')
    j = j.with_columns(((pl.col('k0_S3') - pl.col('r_SHORT3')).abs() + (pl.col('k0_A5') - pl.col('r_ALL5')).abs()).alias('dif')).sort('pair_id', 'dif').unique(subset=['pair_id'], keep='first', maintain_order=True)
    gate = float(j['dif'].max())
    a = (SX.with_columns(pl.Series('L_S3', lg(SX['r_SHORT3'])), pl.Series('L_A5', lg(SX['r_ALL5'])))
         .group_by('pair_id', 'fold').agg(pl.col('L_S3').mean(), pl.col('L_A5').mean()))
    m = j.select('pair_id', 'fold').join(a, on=['pair_id', 'fold'], how='left', validate='1:1')
    assert m['L_S3'].null_count() == 0
    dx = d.join(m.select('pair_id', 'L_S3', 'L_A5'), on='pair_id', how='left', validate='1:1', maintain_order='left')
    return dx.with_columns(pl.Series('r_SHORT3', sig(dx['L_S3'])), pl.Series('r_ALL5', sig(dx['L_A5']))).drop('L_S3', 'L_A5'), gate


def main():
    t0 = time.time()
    assert sha(REG) == REG_SHA and sha(BASE) == BASE_SHA and sha(EVX) == EVX_SHA
    assert not DEST.exists()
    prot = check_protected()
    d = testbed()
    m3, info3 = final_fit(d, FS_M3)
    stored = json.load(open(VD / 'multiview_M3_fit.json', encoding='utf-8'))
    gate_fit = all(abs(info3['weights_standardised'][v] - stored['weights_standardised'][v]) < 1e-9 for v in FS_M3) and info3['bias'] == stored['bias']
    assert gate_fit, (info3['weights_standardised'], stored['weights_standardised'])
    dx, gate_fold = dev_views32(d)
    assert gate_fold == 0.0, gate_fold
    mx, infox = final_fit(dx, FS_M3)

    ev = eval_frame().with_columns(pl.col('R1').cast(pl.Int8).alias('R1i'))
    tve = pl.read_parquet(VD / 'tail_views_eval_frame.parquet'); ev = ev.join(tve, on='pair_id', validate='1:1'); assert ev.height == 909
    s3 = predict(m3, X_of(ev, FS_M3, 'risk_v13', 'R1i', 'risk_noT'))
    evv = pl.read_parquet(EVX, columns=['pair_id', 'L_SHORT3', 'L_ALL5'])
    evx = ev.join(evv, on='pair_id', how='left', validate='1:1', maintain_order='left')
    assert evx['L_SHORT3'].null_count() == 0
    evx = evx.with_columns(pl.Series('r_SHORT3', sig(evx['L_SHORT3'])), pl.Series('r_ALL5', sig(evx['L_ALL5'])))
    sx = predict(mx, X_of(evx, FS_M3, 'risk_v13', 'R1i', 'risk_noT'))

    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    blk = base.filter(pl.col('r') >= .5); assert blk.height == 909
    ids = ev['pair_id'].to_list(); assert set(ids) == set(blk['pair_id'].to_list())
    # lineage: base block order equals the M3 score order (ties only inside equal risk values)
    k3 = pl.DataFrame({'pair_id': ids, 's3': s3, 'sx': sx})
    b = blk.join(k3, on='pair_id', validate='1:1')
    got = b.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    want3 = b.sort(['s3', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    eq_groups = int(b.group_by('risk_score').len().filter(pl.col('len') > 1)['len'].sum())
    lineage_mismatch = sum(x != y for x, y in zip(got, want3))
    assert lineage_mismatch <= 2 * eq_groups, (lineage_mismatch, eq_groups)

    values = b.sort(['r', 'pair_id'], descending=[True, False])['risk_score'].to_list()
    order = b.sort(['sx', 'pair_id'], descending=[True, False]).select('pair_id')
    assign = order.with_columns(pl.Series('new_risk', values))
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    res = (base.join(assign, on='pair_id', how='left', maintain_order='left').with_columns(pl.coalesce('new_risk', 'risk_score').alias('risk_score'))
           .sort('row').select(cols))
    base_cols = base.sort('row').select(cols)
    assert res.drop('risk_score').equals(base_cols.drop('risk_score'))
    assert sorted(res['risk_score'].to_list()) == sorted(base_cols['risk_score'].to_list())
    res.write_csv(DEST)

    A, B = BASE.read_bytes().split(b'\n'), DEST.read_bytes().split(b'\n'); assert len(A) == len(B)
    diff = [i for i in range(len(A)) if A[i] != B[i]]
    blk_ids = set(ids)
    for i in diff:
        x, y = A[i].split(b','), B[i].split(b',')
        assert x[0] == y[0] and x[2:] == y[2:] and x[0].decode() in blk_ids
    back = pl.read_csv(DEST, columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    assert back.filter(pl.col('risk_score') >= .5).height == 909
    got2 = back.join(order, on='pair_id', how='semi').sort(['risk_score', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    order_mismatch = sum(x != y for x, y in zip(got2, order['pair_id'].to_list()))
    pl.DataFrame({'pair_id': ids, 'score_M3': s3, 'score_M3x': sx}).write_parquet(OUTD / 'block_key_M3x.parquet')

    from scipy.stats import spearmanr
    val = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
    assert check_protected() == prot
    top = {str(n): round(len(set(want3[:n]) & set(order['pair_id'].to_list()[:n])) / n, 4) for n in (50, 100, 250, 500)}
    rec = {'registration_sha256': REG_SHA, 'base_sha256': BASE_SHA, 'output': str(DEST), 'output_sha256': sha(DEST),
           'gate_M3_refit_reproduces_stored_fit': gate_fit, 'gate_fold_match_k0_maxdiff': gate_fold, 'block_lineage_mismatch_rows': lineage_mismatch, 'block_equal_value_group_rows': eq_groups,
           'M3x_fit': infox, 'M3_fit_stored': {k: stored[k] for k in ('weights_standardised', 'bias', 'AP_in_sample')},
           'rows_changed_vs_base': len(diff), 'only_block_risk_cells_changed': True, 'value_multiset_unchanged': True, 'order_mismatch_vs_key': order_mismatch,
           'block_spearman_M3_vs_M3x': round(float(spearmanr(s3, sx)[0]), 4), 'block_topn_overlap_M3_vs_M3x': top,
           'eval_views_sha256': EVX_SHA, 'block_key_sha256': sha(OUTD / 'block_key_M3x.parquet'),
           'validation': json.loads(json.dumps(val, default=str)), 'protected_unchanged': True, 'seconds': round(time.time() - t0, 1)}
    (OUTD / 'receipt_step156a.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k != 'validation'}, indent=1)); print('validation', str(val)[:400])


if __name__ == '__main__':
    main()
