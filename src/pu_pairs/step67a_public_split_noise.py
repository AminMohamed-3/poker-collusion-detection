"""How much can a 30 % public split move the deltas we measured? Subsample 30 % of the held/testbed rows repeatedly."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
sys.path.insert(0, r'C:\Data\Projects\Money Making\work\poker\Claude Version\src')
import numpy as np, polars as pl, json
from pu_pairs.step60_multiview_blend import ap, lg
from pu_pairs.step64c_hand_view import VIEWS, DEFS, VD
from pu_pairs.step66_evidence_view import crossfold
rng = np.random.default_rng(0)
CV = r'C:\Data\Projects\Money Making\work\poker\Claude Version'
out = {}
# ---------------- P side: testbed def A block rows, cross-fold M3 / M4 vs the fixed S-like blend
tb = pl.read_parquet(VD / 'evidence_view_prefix_rawonly.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first').select('pair_id', 'E_top5', 'E_n')
mp = pl.read_parquet(VD / 'multiview_prefix.parquet'); tv = pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first').drop('fold_new')
r1 = pl.read_parquet(VD / 'prefix_rescoring.parquet').select('pair_id', 'R1_prefix').unique(subset=['pair_id'], keep='first')
d = mp.join(tv, on='pair_id', how='left').join(tb, on='pair_id', how='left').join(r1, on='pair_id', how='left').with_columns(pl.col('R1_prefix').fill_null(False))
m3 = VIEWS + ['sh', 'R1', 'SHORT3', 'ALL5']
for dn, (posset, negset) in DEFS.items():
    c = d.filter((pl.col('risk_prefix') >= .5) & pl.col('cls').is_in(posset + negset) & pl.col('E_top5').is_not_null() & pl.col('r_SHORT3').is_not_null())
    y = c['cls'].is_in(posset).to_numpy().astype(float)
    s_like = lg(c['r_FULL'].to_numpy()) + .5 * lg(c['r_RAW'].to_numpy()); s_like = np.where(c['R1_prefix'].to_numpy(), s_like - 100, s_like)  # C's hard R1 demotion
    frozen = c['risk_prefix'].to_numpy()
    p3, _, _ = crossfold(c, y, m3); p4, _, _ = crossfold(c, y, m3 + ['E_top5'])
    full = {'n': int(c.height), 'n_pos': int(y.sum()), 'AP_frozen': ap(y, frozen), 'AP_S_like': ap(y, s_like), 'AP_M3': ap(y, p3), 'AP_M4': ap(y, p4)}
    k = max(int(round(.3 * c.height)), 10); dS, d3, d4 = [], [], []
    for _ in range(3000):
        idx = rng.choice(c.height, k, replace=False)
        if y[idx].sum() < 3: continue
        dS.append(ap(y[idx], s_like[idx]) - ap(y[idx], frozen[idx])); d3.append(ap(y[idx], p3[idx]) - ap(y[idx], s_like[idx])); d4.append(ap(y[idx], p4[idx]) - ap(y[idx], p3[idx]))
    q = lambda v: {'mean': round(float(np.mean(v)), 4), 'sd': round(float(np.std(v)), 4), 'q05': round(float(np.quantile(v, .05)), 4), 'q95': round(float(np.quantile(v, .95)), 4), 'share_negative': round(float(np.mean(np.array(v) < 0)), 3)}
    out[f'P def {dn}'] = {'full_sample': {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in full.items()}, 'subsample_30pct_rows': k, 'dAP_S_minus_frozen': q(dS), 'dAP_M3_minus_S': q(d3), 'dAP_M4_minus_M3': q(d4)}
# ---------------- E side: held per-pair AP (folds 0-2), struct vs chrono1 and r382 primary vs struct
def deltas(path, a, b):
    p = pl.read_csv(path).filter(pl.col('candidate').is_in([a, b])).pivot(on='candidate', index=['fold', 'pair_id'], values='map5').drop_nulls()
    p = p.sort('pair_id', 'fold').unique(subset=['pair_id'], keep='first')
    return (p[b] - p[a]).to_numpy(), p
for name, path, a, b in (('E struct minus chrono1', f'{CV}/experiments/family_stage2_20260911/pairs.csv', 'chrono1', 'allfam_top20_struct'), ('E r382(all fam) minus struct', f'{CV}/experiments/family_stage2_r382_20260911/pairs.csv', 'allfam_top20_struct', 'allfam_top20_r382')):
    try:
        dl, p = deltas(path, a, b)
    except Exception as e:
        cands = pl.read_csv(path)['candidate'].unique().to_list(); out[name] = {'error': str(e)[:200], 'candidates': cands}; continue
    k = max(int(round(.3 * len(dl))), 10); sub = [float(np.mean(dl[rng.choice(len(dl), k, replace=False)])) for _ in range(3000)]
    out[name] = {'n_pairs': int(len(dl)), 'full_mean': round(float(dl.mean()), 4), 'subsample_30pct_pairs': k, 'sub_mean': round(float(np.mean(sub)), 4), 'sub_sd': round(float(np.std(sub)), 4), 'sub_q05': round(float(np.quantile(sub, .05)), 4), 'sub_q95': round(float(np.quantile(sub, .95)), 4), 'share_negative': round(float(np.mean(np.array(sub) < 0)), 3)}
json.dump(out, open(f'{CV}/experiments/viewdis_20260911/public_split_noise.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(out, indent=1))
