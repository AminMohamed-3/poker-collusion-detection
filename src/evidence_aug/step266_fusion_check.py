"""STEP266 stage 3: zero-fit fusion composition on the production stack v1_DTfp3 with each STEP266 arm's
DT and SP heads in place of the registered MECHPLUS heads.

Fusion (STEP259 production form): p' = p_stage2^(1-w) * prior_shift(p_evt, F=10)^w, applied to the rows of
one family at a time, w = .25 on DT and on SP.  Reference (arm REF = the registered oof_mechplus.npy
recomputed here): COMP_DT25_SP25 +.011147, halves +.011887 / +.010289; DT_w025 +.007521; SP_w025 +.003627.
That reproduction is the second gate of STEP266.

Arms read from the STEP266 folder as oof_<arm>.npy (4 x 223,749, rows any/DT/SP/CI, STEP254 parquet row
order; the 'any' and 'CI' rows are the registered ones in every arm, so only the fused heads differ):
REF, S3, A4, G4, M4, SINGLE_j0..SINGLE_j3.

Reported per arm: DT_w025, SP_w025, COMP_DT25_SP25 - each as delta (mean over the 162 audit queries of
MAP@5 against the unfused base), halves h0 / h1, per-family means, up/down counts.  Paired bootstrap over
the queries (2,000 draws, seed 0) of A4 - REF, G4 - A4, G4 - REF, M4 - A4, S3 - REF and
A4 - mean(SINGLE_j0..3), for COMP_DT25_SP25.

The adoption rule is in registration266.json and is applied in REPORT.txt, not here.  Development only.
"""
import os, sys, json
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
from pathlib import Path
import numpy as np, polars as pl
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
CV = POKER / 'Claude Version'
OUT = CV / 'experiments/step266_fold_average_calibration_20260918'
K, FAMS = S165.K, S165.FAMS
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}
SINGLES = [f'SINGLE_j{i}' for i in range(4)]
ARMS = [a for a in ('REF', 'S3', 'A4', 'G4', 'M4', *SINGLES) if (OUT / f'oof_{a}.npy').exists()]
assert 'REF' in ARMS, 'build the arms first'
D = S165.load(); d, Q, QIX, dec = D['d'], D['Q'], D['QIX'], D['dec']
sc = pl.read_parquet(CV / 'outputs_step251_20260917/audit_scores_step251.parquet'); assert sc.select(K).equals(d.select(K))
BASE = sc['v1_DTfp3'].to_numpy().copy()
oof = pl.read_parquet(CV / 'experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet')
reg = np.load(CV / 'experiments/step254_mechanical_tensor_20260917/oof_mechplus.npy')
for i, nm in enumerate(('any', 'DT', 'SP', 'CI')):
    assert np.allclose(oof[f'mechplus_p_{nm}'].to_numpy(), reg[i], atol=1e-7), nm   # parquet row order == npy row order
add = []
for a in ARMS:
    arr = np.load(OUT / f'oof_{a}.npy'); assert arr.shape == reg.shape and np.isfinite(arr).all()
    assert np.array_equal(arr[0], reg[0]) and np.array_equal(arr[3], reg[3]), 'any/CI rows must be the registered ones'
    if a == 'REF':
        assert np.array_equal(arr, reg), 'REF must be the registered file'
    add += [pl.Series(f'{a}_p_{nm}', arr[i]) for i, nm in enumerate(('any', 'DT', 'SP', 'CI'))]
oof = oof.with_columns(add)
cols = [f'{s}_p_{nm}' for s in ARMS for nm in ('DT', 'SP')]
j = Q.select(K + ['behavior_family']).join(oof.select(K + cols), on=K, how='left', validate='1:1', maintain_order='left')
assert j.select(K).equals(Q.select(K)) and sum(j[c].null_count() for c in cols) == 0
isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
key = d.select('pair_id', 'hand_id', 'table_id'); qids = sorted(truths)
hv = np.array([half(ptabq[q]) for q in qids]); fq = np.array([fams[q] for q in qids])


def mapof(s):
    _, mm = summarize(key.with_columns(pl.Series('score', dec(s))), truths, fams)
    return np.array([mm[q] for q in qids])


def shift(p, F=10.0):
    q = np.clip(p, 1e-9, 1 - 1e-9); o = q / (1 - q) * F; return o / (1 + o)


def fuse(vec, col, f, w):
    p_top = vec[QIX] - 10.0; pe = shift(j[col].to_numpy().astype(np.float64)); m = isf[f]; newp = p_top.copy()
    newp[m] = np.clip(p_top[m], 1e-9, 1.0) ** (1 - w) * np.clip(pe[m], 1e-9, 1 - 1e-9) ** w
    s = vec.copy(); s[QIX] = 10.0 + newp; return s


M0 = mapof(BASE); out = {}; PQ = {}
for src in ARMS:
    dt25 = fuse(BASE, f'{src}_p_DT', 'directed_transfer', .25)
    sp25 = fuse(BASE, f'{src}_p_SP', 'soft_play', .25)
    comp = fuse(dt25, f'{src}_p_SP', 'soft_play', .25)
    for nm, s in ((f'{src}:DT_w025', dt25), (f'{src}:SP_w025', sp25), (f'{src}:COMP_DT25_SP25', comp)):
        dl = mapof(s) - M0
        PQ[nm] = dl
        out[nm] = {'delta': round(float(dl.mean()), 6), 'h0': round(float(dl[hv == 0].mean()), 6), 'h1': round(float(dl[hv == 1].mean()), 6),
                   **{SHORT[f]: round(float(dl[fq == f].mean()), 6) for f in FAMS}, 'up_down': [int((dl > 0).sum()), int((dl < 0).sum())]}
        print(nm, json.dumps(out[nm]), flush=True)

# the single-model reference level: the average over the four choices of j of the single-model fusion value
MS = np.mean([PQ[f'{s}:COMP_DT25_SP25'] for s in SINGLES], axis=0)
out['MEAN_SINGLE:COMP_DT25_SP25'] = {'delta': round(float(MS.mean()), 6), 'h0': round(float(MS[hv == 0].mean()), 6),
                                     'h1': round(float(MS[hv == 1].mean()), 6),
                                     **{SHORT[f]: round(float(MS[fq == f].mean()), 6) for f in FAMS},
                                     'up_down': [int((MS > 0).sum()), int((MS < 0).sum())],
                                     'note': 'mean of the four SINGLE_j* per-query deltas (the A4_S3AVG diagnostic)'}
print('MEAN_SINGLE:COMP_DT25_SP25', json.dumps(out['MEAN_SINGLE:COMP_DT25_SP25']), flush=True)
for nm in ('DT_w025', 'SP_w025'):
    v = np.mean([PQ[f'{s}:{nm}'] for s in SINGLES], axis=0)
    out[f'MEAN_SINGLE:{nm}'] = {'delta': round(float(v.mean()), 6), 'h0': round(float(v[hv == 0].mean()), 6),
                                'h1': round(float(v[hv == 1].mean()), 6)}

# paired bootstrap over queries (2,000 draws, seed 0)
nq = len(qids); rng = np.random.default_rng(0); IDX = rng.integers(0, nq, size=(2000, nq))


def boot(v):
    b = v[IDX].mean(1)
    return {'point': round(float(v.mean()), 6), 'P_gt_0': round(float((b > 0).mean()), 4),
            'P_lt_0': round(float((b < 0).mean()), 4),
            'p05': round(float(np.quantile(b, .05)), 6), 'p50': round(float(np.quantile(b, .50)), 6),
            'p95': round(float(np.quantile(b, .95)), 6)}


C = {a: PQ[f'{a}:COMP_DT25_SP25'] for a in ARMS}
bt = {'n_queries': nq, 'draws': 2000, 'seed': 0}
pairs = [('A4_minus_REF', C['A4'] - C['REF']), ('G4_minus_A4', C['G4'] - C['A4']),
         ('G4_minus_REF', C['G4'] - C['REF']), ('M4_minus_A4', C['M4'] - C['A4']),
         ('M4_minus_G4', C['M4'] - C['G4']), ('S3_minus_REF', C['S3'] - C['REF']),
         ('A4_minus_MEAN_SINGLE', C['A4'] - MS), ('G4_minus_MEAN_SINGLE', C['G4'] - MS),
         ('MEAN_SINGLE_minus_REF', MS - C['REF'])]
for nm, v in pairs:
    bt[nm] = boot(v)
for a in ARMS:
    bt[f'{a}_vs_zero'] = boot(C[a])
bt['MEAN_SINGLE_vs_zero'] = boot(MS)
# halves of the decisive contrasts
for nm, v in pairs:
    bt[nm]['h0'] = round(float(v[hv == 0].mean()), 6)
    bt[nm]['h1'] = round(float(v[hv == 1].mean()), 6)
for k_, v in bt.items():
    if isinstance(v, dict):
        print('BOOT', k_, json.dumps(v), flush=True)
out['_bootstrap'] = bt
out['_arms'] = ARMS
json.dump(out, open(OUT / 'fusion_check266.json', 'w', encoding='utf-8'), indent=1)
print('wrote', OUT / 'fusion_check266.json')
