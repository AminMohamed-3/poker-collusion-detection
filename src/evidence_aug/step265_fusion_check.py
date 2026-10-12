"""STEP265 readout (c): zero-fit fusion composition on the production stack v1_DTfp3 with the per-family heads of
each STEP265 stack arm in place of the registered MECHPLUS heads.

Fusion (STEP259 production form): p' = p_stage2^(1-w) * prior_shift(p_evt, F=10)^w, applied to the rows of one
family at a time. Reference (the 'mechplus' rows recomputed here): COMP_DT25_SP25 +.011147, halves
+.011887 / +.010289; DT_w025 +.007521; SP_w025 +.003627; CI_w025 -.005988; CI_w010 -.002798.

Sources: mechplus (STEP254 oof_mechplus.npy) + the STEP265 arms A1 / A2 / C1 / C0 (oof_<arm>.npy in the STEP265
folder, 4 x 223,749, rows any/DT/SP/CI, STEP254 parquet row order).

Reported per source: DT_w025, SP_w025, CI_w025, CI_w010, COMP_DT25_SP25, COMP_DT25_SP25_CI10 - each as delta
(mean over the 162 audit queries of MAP@5 against the unfused base), halves h0 / h1, per-family means, up/down
counts. Plus a paired bootstrap over the queries (2,000 draws, seed 0) of (arm - mechplus) per query for
COMP_DT25_SP25, and of the CI_w025 delta itself for the CI question: P(>0), p05, p50, p95.

Adoption rule is in registration265.json and is applied in REPORT.txt, not here. Development only.
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
OUT = CV / 'experiments/step265_seqstack_fusion_20260918'
K, FAMS = S165.K, S165.FAMS
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}
ARMS = [a for a in ('A1', 'A2', 'C1', 'C0') if (OUT / f'oof_{a}.npy').exists()]
D = S165.load(); d, Q, QIX, dec = D['d'], D['Q'], D['QIX'], D['dec']
sc = pl.read_parquet(CV / 'outputs_step251_20260917/audit_scores_step251.parquet'); assert sc.select(K).equals(d.select(K))
BASE = sc['v1_DTfp3'].to_numpy().copy()
oof = pl.read_parquet(CV / 'experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet')
ref = np.load(CV / 'experiments/step254_mechanical_tensor_20260917/oof_mechplus.npy')
for i, nm in enumerate(('any', 'DT', 'SP', 'CI')):
    assert np.allclose(oof[f'mechplus_p_{nm}'].to_numpy(), ref[i], atol=1e-7), nm      # parquet row order == npy row order
add = []
for a in ARMS:
    arr = np.load(OUT / f'oof_{a}.npy'); assert arr.shape == ref.shape and np.isfinite(arr).all()
    add += [pl.Series(f'{a}_p_{nm}', arr[i]) for i, nm in enumerate(('any', 'DT', 'SP', 'CI'))]
oof = oof.with_columns(add)
srcs = ['mechplus'] + ARMS
cols = [f'{s}_p_{nm}' for s in srcs for nm in ('DT', 'SP', 'CI')]
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
for src in srcs:
    dt25 = fuse(BASE, f'{src}_p_DT', 'directed_transfer', .25)
    sp25 = fuse(BASE, f'{src}_p_SP', 'soft_play', .25)
    comp = fuse(dt25, f'{src}_p_SP', 'soft_play', .25)
    for nm, s in ((f'{src}:DT_w025', dt25), (f'{src}:SP_w025', sp25),
                  (f'{src}:CI_w025', fuse(BASE, f'{src}_p_CI', 'coordinated_isolation', .25)),
                  (f'{src}:CI_w010', fuse(BASE, f'{src}_p_CI', 'coordinated_isolation', .10)),
                  (f'{src}:COMP_DT25_SP25', comp),
                  (f'{src}:COMP_DT25_SP25_CI10', fuse(comp, f'{src}_p_CI', 'coordinated_isolation', .10))):
        dl = mapof(s) - M0
        PQ[nm] = dl
        out[nm] = {'delta': round(float(dl.mean()), 6), 'h0': round(float(dl[hv == 0].mean()), 6), 'h1': round(float(dl[hv == 1].mean()), 6),
                   **{SHORT[f]: round(float(dl[fq == f].mean()), 6) for f in FAMS}, 'up_down': [int((dl > 0).sum()), int((dl < 0).sum())]}
        print(nm, json.dumps(out[nm]), flush=True)

# paired bootstrap over queries (2,000 draws, seed 0)
nq = len(qids); rng = np.random.default_rng(0); IDX = rng.integers(0, nq, size=(2000, nq))


def boot(v):
    b = v[IDX].mean(1)
    return {'point': round(float(v.mean()), 6), 'P_gt_0': round(float((b > 0).mean()), 4),
            'p05': round(float(np.quantile(b, .05)), 6), 'p50': round(float(np.quantile(b, .50)), 6),
            'p95': round(float(np.quantile(b, .95)), 6)}


bt = {'n_queries': nq, 'draws': 2000, 'seed': 0}
for a in ARMS:
    bt[f'{a}_vs_mechplus:COMP_DT25_SP25'] = boot(PQ[f'{a}:COMP_DT25_SP25'] - PQ['mechplus:COMP_DT25_SP25'])
    bt[f'{a}_vs_mechplus:COMP_DT25_SP25_CI10'] = boot(PQ[f'{a}:COMP_DT25_SP25_CI10'] - PQ['mechplus:COMP_DT25_SP25'])
for s in srcs:
    bt[f'{s}:CI_w025_vs_zero'] = boot(PQ[f'{s}:CI_w025'])
    bt[f'{s}:CI_w010_vs_zero'] = boot(PQ[f'{s}:CI_w010'])
for k, v in bt.items():
    if isinstance(v, dict):
        print('BOOT', k, json.dumps(v), flush=True)
out['_bootstrap'] = bt
json.dump(out, open(OUT / 'fusion_check265.json', 'w', encoding='utf-8'), indent=1)
print('wrote', OUT / 'fusion_check265.json')
