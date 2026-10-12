"""STEP263 readout (c): zero-fit fusion composition DT_w025 + SP_w025 on the production stack v1_DTfp3 with the heads of
each STEP263 arm in place of the registered MECHPLUS heads (reference +.011147, halves +.011887 / +.010289).
Rule (registration263.json): an arm replaces the registered event model in a future build iff composition >= reference +
.0015, each half >= reference half - .001, and no family head AP more than .005 below the reference."""
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
OUT = CV / 'experiments/step263_censored_labels_20260918'
K, FAMS = S165.K, S165.FAMS
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}
ARMS = [a for a in ('HN', 'PP', 'HNPP') if (OUT / f'oof_{a}.npy').exists()]
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


M0 = mapof(BASE); out = {}
for src in srcs:
    for nm, s in ((f'{src}:DT_w025', fuse(BASE, f'{src}_p_DT', 'directed_transfer', .25)), (f'{src}:SP_w025', fuse(BASE, f'{src}_p_SP', 'soft_play', .25)),
                  (f'{src}:CI_w025', fuse(BASE, f'{src}_p_CI', 'coordinated_isolation', .25)),
                  (f'{src}:COMP_DT25_SP25', fuse(fuse(BASE, f'{src}_p_DT', 'directed_transfer', .25), f'{src}_p_SP', 'soft_play', .25))):
        dl = mapof(s) - M0
        out[nm] = {'delta': round(float(dl.mean()), 6), 'h0': round(float(dl[hv == 0].mean()), 6), 'h1': round(float(dl[hv == 1].mean()), 6),
                   **{SHORT[f]: round(float(dl[fq == f].mean()), 6) for f in FAMS}, 'up_down': [int((dl > 0).sum()), int((dl < 0).sum())]}
        print(nm, json.dumps(out[nm]), flush=True)
json.dump(out, open(OUT / 'fusion_check263.json', 'w', encoding='utf-8'), indent=1)
