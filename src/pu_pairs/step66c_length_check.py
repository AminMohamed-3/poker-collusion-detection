"""Step 66c: is the evidence-signature view (step 66, raw-only recipe) a proxy for history length?
E_top5 (mean of the five highest per-hand planted probabilities) grows with the number of hands under the null (an order
statistic), and the certified false positives of the prefix testbed are short (median 17 hands against 68 for the block
positives). Checks: (1) AUC positives vs certified false positives inside strata of the prefix hand count; (2) a
length-adjusted view E_adj = E_top5 minus the null quadratic in log(hands) (null = certified FP + mid + ambiguous +
labelled negatives + tail sample) in the same cross-fold blends as step 66; (3) the blend with E_top5 but without
log shared_hands. Output: experiments/viewdis_20260911/evidence_view_length_check.json"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import write_json
import numpy as np
import polars as pl
from pu_pairs.step60_multiview_blend import auc, ap
from pu_pairs.step64c_hand_view import VIEWS, DEFS, VD
from pu_pairs.step66_evidence_view import crossfold


def main():
    tb = pl.read_parquet(VD / 'evidence_view_prefix_rawonly.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first').select('pair_id', 'E_top5', 'E_n')
    mp = pl.read_parquet(VD / 'multiview_prefix.parquet'); tv = pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first').drop('fold_new')
    pops = pl.concat([pl.read_parquet(VD / f'prefix_fold{k}_ext_population.parquet').select('pair_id', 'why') for k in (1, 2)]).unique(subset=['pair_id'], keep='first')
    r1 = pl.read_parquet(VD / 'prefix_rescoring.parquet').select('pair_id', 'R1_prefix').unique(subset=['pair_id'], keep='first')
    d = mp.join(tv, on='pair_id', how='left').join(tb, on='pair_id', how='left').join(pops, on='pair_id', how='left').join(r1, on='pair_id', how='left').with_columns(pl.col('R1_prefix').fill_null(False))
    res = {}
    blk = d.filter((pl.col('risk_prefix') >= .5) & pl.col('E_top5').is_not_null()); strata = {}
    for lo, hi in ((0, 20), (20, 40), (40, 80), (80, 10000)):
        s = blk.filter((pl.col('E_n') >= lo) & (pl.col('E_n') < hi)); p = s.filter(pl.col('cls') == 'pos'); f = s.filter(pl.col('cls') == 'FP')
        strata[f'[{lo},{hi})'] = {'n_pos': p.height, 'n_FP': f.height, 'E_top5': round(auc(p['E_top5'].to_numpy(), f['E_top5'].to_numpy()), 4), 'FULL': round(auc(p['risk_prefix'].to_numpy(), f['risk_prefix'].to_numpy()), 4), 'ALL5': round(auc(p['r_ALL5'].to_numpy(), f['r_ALL5'].to_numpy()), 4)}
    res['stratified_auc_pos_vs_FP'] = strata
    null = d.filter(pl.col('E_top5').is_not_null() & (pl.col('cls').is_in(['FP', 'mid', 'ambiguous', 'neg']) | (pl.col('why') == 'tail_sample')) & (pl.col('label') != 1).fill_null(True))
    coef = np.polyfit(np.log(null['E_n'].to_numpy()), null['E_top5'].to_numpy(), 2)
    d = d.with_columns(pl.Series('E_adj', d['E_top5'].to_numpy() - np.polyval(coef, np.log(np.maximum(d['E_n'].fill_null(1).to_numpy(), 1)))))
    blk = d.filter((pl.col('risk_prefix') >= .5) & pl.col('E_top5').is_not_null()); p = blk.filter(pl.col('cls') == 'pos'); f = blk.filter(pl.col('cls') == 'FP')
    res['null_fit'] = {'coef_quadratic_in_log_n': [round(float(c), 4) for c in coef], 'n_null': null.height}
    res['block_auc_pos_vs_FP'] = {'E_top5': round(auc(p['E_top5'].to_numpy(), f['E_top5'].to_numpy()), 4), 'E_adj': round(auc(p['E_adj'].to_numpy(), f['E_adj'].to_numpy()), 4)}
    base_fs = VIEWS + ['sh', 'R1']; m3 = base_fs + ['SHORT3', 'ALL5']; blends = {}
    for floor in (.5, .1):
        for dn, (posset, negset) in DEFS.items():
            c = d.filter((pl.col('risk_prefix') >= floor) & pl.col('cls').is_in(posset + negset) & pl.col('E_top5').is_not_null() & pl.col('r_SHORT3').is_not_null()); y = c['cls'].is_in(posset).to_numpy().astype(float)
            out = {}
            for name, fs in (('M3', m3), ('M3+E_top5', m3 + ['E_top5']), ('M3+E_adj', m3 + ['E_adj']), ('M3+E_top5 without sh', [v for v in m3 if v != 'sh'] + ['E_top5'])):
                pred, (m1, m2), (in1, in2, insh) = crossfold(c, y, fs)
                out[name] = {'pooled': round(ap(y, pred), 4), 'f1': round(ap(y[in1], pred[in1]), 4), 'f2sh': round(ap(y[in2 | insh], pred[in2 | insh]), 4), 'wE': [round(float(w), 2) for v, w in zip(fs, m1[0]) if v.startswith('E_')] + [round(float(w), 2) for v, w in zip(fs, m2[0]) if v.startswith('E_')]}
            blends[f'floor {floor} def {dn}'] = out
    res['blends'] = blends
    write_json(VD / 'evidence_view_length_check.json', res)
    print(res)


if __name__ == '__main__':
    main()
