"""Does the testbed's M3-over-S gain survive when the testbed is restricted to pairs that could exist in evaluation (shared_hands >= 38)?"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
sys.path.insert(0, r'C:\Data\Projects\Money Making\work\poker\Claude Version\src')
import numpy as np, polars as pl, json
from pu_pairs.step60_multiview_blend import ap, lg
from pu_pairs.step64c_hand_view import VIEWS, DEFS, VD
from pu_pairs.step66_evidence_view import crossfold
CV = r'C:\Data\Projects\Money Making\work\poker\Claude Version'
rng = np.random.default_rng(1)
tb = pl.read_parquet(VD / 'evidence_view_prefix_rawonly.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first').select('pair_id', 'E_top5', 'E_n')
mp = pl.read_parquet(VD / 'multiview_prefix.parquet'); tv = pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first').drop('fold_new')
r1 = pl.read_parquet(VD / 'prefix_rescoring.parquet').select('pair_id', 'R1_prefix').unique(subset=['pair_id'], keep='first')
d = mp.join(tv, on='pair_id', how='left').join(tb, on='pair_id', how='left').join(r1, on='pair_id', how='left').with_columns(pl.col('R1_prefix').fill_null(False))
ev = pl.read_parquet(VD / 'multiview_eval.parquet')
out = {}
blk_ev = ev.filter(pl.col('risk') >= .5)
out['evaluation_block_shared_hands'] = {'n': blk_ev.height, 'min': int(blk_ev['shared_hands'].min()), 'q10': float(blk_ev['shared_hands'].quantile(.1)), 'median': float(blk_ev['shared_hands'].median()), 'q90': float(blk_ev['shared_hands'].quantile(.9))}
out['evaluation_all_shared_hands_min'] = int(ev['shared_hands'].min())
blk = d.filter(pl.col('risk_prefix') >= .5)
out['testbed_block_shared_hands_by_cls'] = {k[0]: {'n': g.height, 'min': int(g['shared_hands'].min()), 'median': float(g['shared_hands'].median()), 'share_lt_38': round(float((g['shared_hands'] < 38).mean()), 3)} for k, g in blk.group_by('cls', maintain_order=True)}
m3 = VIEWS + ['sh', 'R1', 'SHORT3', 'ALL5']
def run(c, tag):
    y = c['cls'].is_in(posset).to_numpy().astype(float)
    s_like = lg(c['r_FULL'].to_numpy()) + .5 * lg(c['r_RAW'].to_numpy()); s_like = np.where(c['R1_prefix'].to_numpy(), s_like - 100, s_like)
    frozen = c['risk_prefix'].to_numpy()
    p3, (m1, m2), _ = crossfold(c, y, m3); p4, _, _ = crossfold(c, y, m3 + ['E_top5'])
    w_sh = [round(float(w), 2) for v, w in zip(m3, m1[0]) if v == 'sh'] + [round(float(w), 2) for v, w in zip(m3, m2[0]) if v == 'sh']
    k = max(int(round(.3 * c.height)), 10); d3 = []
    for _ in range(2000):
        idx = rng.choice(c.height, k, replace=False)
        if y[idx].sum() < 3: continue
        d3.append(ap(y[idx], p3[idx]) - ap(y[idx], s_like[idx]))
    return {'n': int(c.height), 'n_pos': int(y.sum()), 'AP_frozen': round(ap(y, frozen), 4), 'AP_S_like': round(ap(y, s_like), 4), 'AP_M3': round(ap(y, p3), 4), 'AP_M4': round(ap(y, p4), 4), 'M3_minus_S': round(ap(y, p3) - ap(y, s_like), 4), 'w_sh_fold_models': w_sh, 'sub30_M3_minus_S_sd': round(float(np.std(d3)), 4), 'sub30_share_negative': round(float(np.mean(np.array(d3) < 0)), 3)}
for dn, (posset, negset) in DEFS.items():
    c0 = d.filter((pl.col('risk_prefix') >= .5) & pl.col('cls').is_in(posset + negset) & pl.col('E_top5').is_not_null() & pl.col('r_SHORT3').is_not_null())
    out[f'def {dn} all'] = run(c0, 'all')
    out[f'def {dn} shared>=38'] = run(c0.filter(pl.col('shared_hands') >= 38), 'ge38')
    out[f'def {dn} shared>=50'] = run(c0.filter(pl.col('shared_hands') >= 50), 'ge50')
    # blend fitted on all rows, evaluated on the >= 38 subset only
    y0 = c0['cls'].is_in(posset).to_numpy().astype(float); p3, _, _ = crossfold(c0, y0, m3)
    s_like = lg(c0['r_FULL'].to_numpy()) + .5 * lg(c0['r_RAW'].to_numpy()); s_like = np.where(c0['R1_prefix'].to_numpy(), s_like - 100, s_like)
    m = (c0['shared_hands'] >= 38).to_numpy()
    out[f'def {dn} fit-all eval>=38'] = {'AP_S_like': round(ap(y0[m], s_like[m]), 4), 'AP_M3': round(ap(y0[m], p3[m]), 4), 'M3_minus_S': round(ap(y0[m], p3[m]) - ap(y0[m], s_like[m]), 4)}
json.dump(out, open(f'{CV}/experiments/viewdis_20260911/length_transfer_check.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(out, indent=1))
