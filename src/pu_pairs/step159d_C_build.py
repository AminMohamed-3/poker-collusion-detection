"""Step 159d (Claude): production of outputs_step159_20260915/prediction_step159.json (sha a77dcb64) from combination159.json.
Tail: FINAL32 key = mean over the FINAL view types of the 32-draw mean lg(r) from the evaluation draws (family F five types; XGB3/ET/MLP
if kept); the tail of the .90472 file (6fdfaae4) gets its own risk strings in key order (ties by pair_id). Block (only if part K passed):
M3x refitted on the def A testbed rows with F's fold-matched 16 development draws, 909 evaluation block rows scored with F's 32-draw
SHORT3/ALL5, block risk strings re-assigned in that order. Behaviour and evidence byte-identical. Never uploads."""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
import sys, json, time
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

OUTD = CV / 'outputs_step159_20260915'
REG, REG_SHA = OUTD / 'prediction_step159.json', 'a77dcb643b129e50b956d4d95a889eb1c5c7b4080ab6dc0bae38011ee53a2252'
BASE, BASE_SHA = CV / 'outputs_step156_20260915' / 'submission_step156_blockM3x_tailSXN32.csv', '6fdfaae403e97c074194b05e8e7d01aa506d74bcc369a92329477a70b4dcec94'
EXP = CV / 'experiments'
EVAL_DRAWS = {'SXN': (EXP / 'tail_keys_20260915_neg', 'sxn_eval_k'), 'SXN3': (EXP / 'tail_keys_20260915_neg3', 'sxn3_eval_k'),
              'SXB': (EXP / 'bagged_views_20260915', 'sxb_eval_k'), 'ARCH': (EXP / 'arch_views_20260915', 'arch_eval_neg1_k')}
DEV_DRAWS = {'SX': (EXP / 'tail_keys_20260914', 'sx_dev_fold'), 'SXN': (EXP / 'tail_keys_20260915_neg', 'sxn_dev_fold'),
             'SXN3': (EXP / 'tail_keys_20260915_neg3', 'sxn3_dev_fold'), 'SXB': (EXP / 'bagged_views_20260915', 'sxb_dev_fold')}
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']; NEW_VT = ['XGB3', 'ET', 'MLP']
DEST = OUTD / 'submission_step159.csv'


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def eval_means(name, vts, K=32):
    folder, prefix = EVAL_DRAWS[name]
    d = pl.concat([pl.read_parquet(folder / f) for f in sorted(os.listdir(folder)) if f.startswith(prefix) and f.endswith('.parquet')]).filter(pl.col('k') < K)
    assert sorted(d['k'].unique().to_list()) == list(range(K))
    n = d['pair_id'].n_unique(); assert n == 112540 and d.height == n * K, (name, n, d.height)
    assert all(bool(d[f'r_{v}'].is_finite().all()) for v in vts)
    return d.with_columns([pl.Series(f'L_{v}', lg(d[f'r_{v}'])) for v in vts]).group_by('pair_id').agg([pl.col(f'L_{v}').mean().alias(f'{name}_{v}') for v in vts])


def dev_block_views(d, F, K=16):
    """Fold-matched (SX draw 0 = stored views) 16-draw SHORT3/ALL5 logit means of family F at prefix 60 for the testbed rows."""
    def load(name):
        folder, prefix = DEV_DRAWS[name]
        return pl.concat([pl.read_parquet(folder / f) for f in sorted(os.listdir(folder)) if f.startswith(prefix) and f.endswith('.parquet')]).filter((pl.col('surface') == 'prefix60') & (pl.col('k') < K))
    SX = load('SX')
    k0 = SX.filter(pl.col('k') == 0).select('pair_id', 'fold', pl.col('r_SHORT3').alias('k0_S3'), pl.col('r_ALL5').alias('k0_A5'))
    j = d.select('pair_id', 'r_SHORT3', 'r_ALL5').join(k0, on='pair_id', how='left')
    j = j.with_columns(((pl.col('k0_S3') - pl.col('r_SHORT3')).abs() + (pl.col('k0_A5') - pl.col('r_ALL5')).abs()).alias('dif')).sort('pair_id', 'dif').unique(subset=['pair_id'], keep='first', maintain_order=True)
    gate = float(j['dif'].max())
    Fd = load(F); assert Fd.select('fold', 'k').unique().height == 2 * K
    a = Fd.with_columns(pl.Series('L_S3', lg(Fd['r_SHORT3'])), pl.Series('L_A5', lg(Fd['r_ALL5']))).group_by('pair_id', 'fold').agg(pl.col('L_S3').mean(), pl.col('L_A5').mean())
    m = j.select('pair_id', 'fold').join(a, on=['pair_id', 'fold'], how='left', validate='1:1'); assert m['L_S3'].null_count() == 0
    dx = d.join(m.select('pair_id', 'L_S3', 'L_A5'), on='pair_id', how='left', validate='1:1', maintain_order='left')
    return dx.with_columns(pl.Series('r_SHORT3', sig(dx['L_S3'])), pl.Series('r_ALL5', sig(dx['L_A5']))).drop('L_S3', 'L_A5'), gate


def reassign(base, region, key_col):
    """Region rows get the region's own risk strings (high to low) in key order (ties by pair_id)."""
    values = region.sort(['r', 'pair_id'], descending=[True, False])['risk_score'].to_list()
    order = region.sort([key_col, 'pair_id'], descending=[True, False]).select('pair_id')
    return order.with_columns(pl.Series('new_risk', values)), order['pair_id'].to_list()


def main():
    t0 = time.time()
    assert sha(REG) == REG_SHA and sha(BASE) == BASE_SHA and not DEST.exists()
    comb = json.load(open(OUTD / 'combination159.json', encoding='utf-8'))
    assert comb['registration_sha256'] == REG_SHA and comb['build_file'], 'nothing passed: no STEP159 file'
    F, new = comb['FINAL']['five_type_family'], comb['FINAL']['new_types']
    tail_changes, kpass = (F != 'SXN') or bool(new), comb['partK']['passes']
    prot = check_protected()
    base = pl.read_csv(BASE, infer_schema=False).with_row_index('row').with_columns(pl.col('risk_score').cast(pl.Float64).alias('r'))
    cols = pl.read_csv(BASE, n_rows=1, infer_schema=False).columns
    tail, blk = base.filter(pl.col('r') < .5), base.filter(pl.col('r') >= .5)
    assert tail.height == 111631 and blk.height == 909
    rec = {'registration_sha256': REG_SHA, 'base_sha256': BASE_SHA, 'combination159_sha256': sha(OUTD / 'combination159.json'), 'F': F, 'new_types': new,
           'tail_changes': tail_changes, 'block_changes': kpass}
    assigns = []
    # lineage: base tail order = SXN32 key, base block order = M3x(SX32) key (except inside equal-value groups)
    sxn = pl.read_parquet(EXP / 'tail_keys_20260915_neg' / 'eval_views_SXN32.parquet', columns=['pair_id', 'k_SXN'])
    t = tail.join(sxn, on='pair_id', how='left', validate='1:1', maintain_order='left'); assert t['k_SXN'].null_count() == 0
    eqt = int(t.group_by('risk_score').len().filter(pl.col('len') > 1)['len'].sum())
    got = t.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list(); want = t.sort(['k_SXN', 'pair_id'], descending=[True, False])['pair_id'].to_list()
    rec['tail_lineage_mismatch'] = sum(a != b for a, b in zip(got, want)); assert rec['tail_lineage_mismatch'] <= 2 * eqt, (rec['tail_lineage_mismatch'], eqt)
    if tail_changes:
        V = eval_means(F, VT)
        if new: V = V.join(eval_means('ARCH', NEW_VT), on='pair_id', how='inner', validate='1:1')
        kc = [f'{F}_{v}' for v in VT] + [f'ARCH_{v}' for v in new]
        V = V.with_columns(pl.mean_horizontal(kc).alias('k_FINAL')).sort('pair_id')
        V.write_parquet(OUTD / 'eval_views_FINAL32.parquet'); rec['eval_views_FINAL32_sha256'] = sha(OUTD / 'eval_views_FINAL32.parquet')
        t = t.join(V.select('pair_id', 'k_FINAL'), on='pair_id', how='left', validate='1:1', maintain_order='left'); assert t['k_FINAL'].null_count() == 0
        a, w = reassign(base, t, 'k_FINAL'); assigns.append(a)
        from scipy.stats import spearmanr
        rec['tail_spearman_SXN32_vs_FINAL32'] = round(float(spearmanr(t['k_SXN'], t['k_FINAL'])[0]), 4)
        rec['tail_topn_overlap_SXN32_vs_FINAL32'] = {str(n): round(len(set(want[:n]) & set(w[:n])) / n, 4) for n in (100, 1000, 5000)}
    if kpass:
        d = testbed()
        dx, gate = dev_block_views(d, F); assert gate == 0.0, gate
        mx, infox = final_fit(dx, FS_M3); rec['M3xF_fit'] = infox
        ev = eval_frame().with_columns(pl.col('R1').cast(pl.Int8).alias('R1i'))
        ev = ev.join(pl.read_parquet(VD / 'tail_views_eval_frame.parquet'), on='pair_id', validate='1:1'); assert ev.height == 909
        E = eval_means(F, ['SHORT3', 'ALL5'])
        evx = ev.join(E, on='pair_id', how='left', validate='1:1', maintain_order='left'); assert evx[f'{F}_SHORT3'].null_count() == 0
        evx = evx.with_columns(pl.Series('r_SHORT3', sig(evx[f'{F}_SHORT3'])), pl.Series('r_ALL5', sig(evx[f'{F}_ALL5'])))
        kF = pl.DataFrame({'pair_id': ev['pair_id'], 'score_M3xF': predict(mx, X_of(evx, FS_M3, 'risk_v13', 'R1i', 'risk_noT'))})
        bk = pl.read_parquet(CV / 'outputs_step156_20260915' / 'block_key_M3x.parquet', columns=['pair_id', 'score_M3x'])
        b = blk.join(kF, on='pair_id', validate='1:1').join(bk, on='pair_id', validate='1:1')
        eqb = int(b.group_by('risk_score').len().filter(pl.col('len') > 1)['len'].sum())
        gb = b.sort(['r', 'pair_id'], descending=[True, False])['pair_id'].to_list(); wb = b.sort(['score_M3x', 'pair_id'], descending=[True, False])['pair_id'].to_list()
        rec['block_lineage_mismatch'] = sum(x != y for x, y in zip(gb, wb)); assert rec['block_lineage_mismatch'] <= 2 * eqb, (rec['block_lineage_mismatch'], eqb)
        a, wbF = reassign(base, b, 'score_M3xF'); assigns.append(a)
        kF.write_parquet(OUTD / 'block_key_M3xF.parquet')
        rec['block_top100_overlap_M3x_vs_M3xF'] = round(len(set(wb[:100]) & set(wbF[:100])) / 100, 4)
    assign = pl.concat(assigns)
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
    rec.update({'output': str(DEST), 'output_sha256': sha(DEST), 'rows_changed_vs_base': len(diff), 'only_risk_cells_changed': True, 'value_multiset_unchanged': True,
                'validation': json.loads(json.dumps(val, default=str)), 'protected_unchanged': True, 'seconds': round(time.time() - t0, 1)})
    (OUTD / 'receipt_step159.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('validation', 'M3xF_fit')}, indent=1)); print('validation', str(val)[:300])


if __name__ == '__main__':
    main()
