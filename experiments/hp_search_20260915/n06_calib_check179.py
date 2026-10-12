"""Descriptive (disclosed): does H2's N06 (lam_list2 0) gain survive the STEP175 calibrated decoder (b .5)? No selection; two arms only."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '4'
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src'); sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
import numpy as np, polars as pl
from pathlib import Path
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug import step172b_C_hp_search as S172B
from evidence_aug.step175_C_decoder_calibration import calibrate
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step5_evaluate import half
EXP = Path('C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/hp_search_20260915')
PARTS = ['P0', 'P1651', 'P1652']
D = S165.load(); d, Q = D['d'], D['Q']
meta = json.loads((EXP / 'cache/meta172a.json').read_text(encoding='utf-8'))
V = np.asarray(np.load(EXP / 'cache/views41.npy', mmap_mode='r')); idx = {f: np.array(x) for f, x in meta['family_column_index'].items()}
y = Q['relevance'].to_numpy().astype(int); fam = Q['behavior_family'].to_numpy()
truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows()); pools = sorted(set(ptab.values()))
key = d.select('pair_id', 'hand_id', 'table_id'); qids = sorted(truths)
M175 = np.load(EXP / 'calib175/maps175.npy'); M174 = np.load(EXP / 'net174/maps174.npy')
ids175 = [f'a{a}_b{b}' for a in [.6, .8, 1., 1.25, 1.5, 2.] for b in [-1., -.5, 0., .5, 1.]]
arms174 = ['N00_incumbent', 'F_medium_features', 'N01_wide_128_32_64', 'N02_narrow_32_8_16', 'N03_wd_1e-1', 'N04_wd_1e-3', 'N05_lr_2e-3', 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3', 'N09_two_stage', 'N10_patience80_epochs800', 'N11_seeds20']
out = {}
for arm, src in (('N00', EXP / 'cache/nn_{}.npz'), ('N06', EXP / 'net174/nn_N06_lam_list2_0_{}.npz')):
    Mm = np.zeros((2, 3, len(qids)))
    for j, n in enumerate(PARTS):
        s = S172B.v1_full(dict(S172B.INCUMBENT), V, idx, dict(np.load(str(src).format(n))), y, fam, D['SS'], D['QIX'])
        for i, b in enumerate((0., .5)):
            _, m = summarize(key.with_columns(pl.Series('score', D['dec'](calibrate(s, D['QIX'], 1., b)))), truths, fams)
            Mm[i, j] = [m[q] for q in qids]
    out[arm] = Mm
    print(arm, 'done', flush=True)
g1 = float(np.abs(out['N00'][0] - M175[ids175.index('a1.0_b0.0')]).max()); g2 = float(np.abs(out['N00'][1] - M175[ids175.index('a1.0_b0.5')]).max())
g3 = float(np.abs(out['N06'][0] - M174[arms174.index('N06_lam_list2_0')]).max())
print('gates N00 b0 vs 175', g1, 'N00 b.5 vs 175', g2, 'N06 b0 vs 174', g3)
hv = np.array([half(ptab[q]) for q in qids])
for i, b in enumerate((0., .5)):
    A0, A6 = out['N00'][i].mean(0), out['N06'][i].mean(0)
    c = boot({q: A0[t] for t, q in enumerate(qids)}, {q: A6[t] for t, q in enumerate(qids)}, ptab, pools)
    fd = {f[:2]: round(float(np.mean([A6[t] - A0[t] for t, q in enumerate(qids) if fams[q] == f])), 4) for f in ('directed_transfer', 'soft_play', 'coordinated_isolation')}
    print('decoder b', b, 'N06 - N00 mean', round(c['mean_delta'], 5), 'ci', [round(x, 4) for x in c['ci95']], 'bw', c['pairs_better'], c['pairs_worse'],
          'halves', round(float((A6 - A0)[hv == 0].mean()), 5), round(float((A6 - A0)[hv == 1].mean()), 5),
          'partitions', [round(float((out['N06'][i][j] - out['N00'][i][j]).mean()), 5) for j in range(3)], 'fam', fd, 'MAP N00', round(float(A0.mean()), 4), 'N06', round(float(A6.mean()), 4))
