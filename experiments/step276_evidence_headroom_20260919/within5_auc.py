"""STEP276-AM: within-the-selected-five discrimination.  PURE MEASUREMENT, NO FITTING.

STEP276-AL: re-ordering production's own five -- set untouched -- is worth +.055046 decoded, and production
captures only ~49% of the random->oracle ordering span.  124 of 162 lists are partially correct and carry 174
misses.  So the question is whether ANY signal separates cited from not-cited INSIDE that hard five.

No model is fitted here, so no fold noise and no selection to pay for: for every list, over all (cited,
not-cited) pairs among the five, the fraction where the signal puts the cited hand higher.  Production's own
p is the baseline to beat, because production orders by p*q.  Everything is reported with a per-list bootstrap.
"""
import os, sys, json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, polars as pl
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R/'Claude Version'
OUT = CV/'experiments/step276_evidence_headroom_20260919'
sys.path.insert(0, str(CV/'src')); sys.path.insert(0, str(R/'Codex Resume/src')); sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology, align, less_than_five
K = S165.K; CI = 'coordinated_isolation'
D = S165.load(); d, Q, QIX, dec = D['d'], D['Q'], D['QIX'], D['dec']
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id','behavior_family').unique().iter_rows())
famall = np.array([fams[p] for p in pid]); top = D['top']
v = pl.read_parquet(CV/'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
fr = chronology(d.select(K+['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
t1all = align(fr, pl.read_parquet(CV/'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
cr = fr['chrono_rank'].to_numpy()
PRI = (~(t1all & (famall != CI))).astype(np.int64)
P = np.load(S/'inc_p_PRODbare.npy'); pf = np.zeros(len(pid)); pf[QIX] = P
qids = sorted(set(pid[top]))
GT = {p: np.flatnonzero((pid==p)&top) for p in qids}
sc = np.zeros(len(pid)); qv = np.zeros(len(pid))
for q, g in GT.items():
    p = np.clip(pf[g], 0., 1.)
    o = np.lexsort((cr[g], PRI[g])); w = np.empty(len(g)); w[o] = less_than_five(p[o])
    qv[g] = w; sc[g] = 10.0 + p*w + 1e-9*p
SEL = {q: GT[q][np.argsort(-sc[GT[q]], kind='stable')[:5]] for q in qids}
# lists where discrimination is even defined
ACT = [q for q in qids if 0 < rel[SEL[q]].sum() < len(SEL[q])]
print('lists with a live within-five contrast: %d of %d' % (len(ACT), len(qids)), flush=True)
IDXQ = {q: i for i, q in enumerate(ACT)}
fq = np.array([fams[q] for q in ACT])

def w5auc(sig):
    """per-list concordance among (cited, not-cited) pairs inside the five; returns per-list array + weights"""
    a = np.empty(len(ACT)); wt = np.empty(len(ACT))
    for i, q in enumerate(ACT):
        s = SEL[q]; x = sig[s].astype(float); y = rel[s]
        pos = x[y==1]; neg = x[y==0]
        c = (pos[:,None] > neg[None,:]).sum() + .5*(pos[:,None] == neg[None,:]).sum()
        a[i] = c/(len(pos)*len(neg)); wt[i] = len(pos)*len(neg)
    return a, wt

MP = CV/'experiments/step254_mechanical_tensor_20260917'
mech = d.select(K).join(pl.read_parquet(MP/'step254_features.parquet'), on=K, how='left')
mcols = [c for c in mech.columns if c not in K and mech[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Boolean)]
print('mechanical columns available: %d' % len(mcols), flush=True)
base = {'p_stage2': pf, 'prod_score(p*q)': sc, 'q_censor': qv,
        'oof_stage1': d['oof'].to_numpy().astype(float), '-oof_rank': -d['oof_rank'].to_numpy().astype(float),
        'chrono_rank': cr.astype(float), 'T1_flag': t1all.astype(float)}
rows = []
for nm, sig in base.items():
    a, wt = w5auc(sig); rows.append((nm, float(np.average(a, weights=wt)), float(a.mean()), len(ACT)))
print('', flush=True)
print('%-22s %10s %10s' % ('signal', 'wAUC5', 'meanAUC5'), flush=True)
for nm, wa, ma, n in rows:
    print('%-22s %10.4f %10.4f' % (nm, wa, ma), flush=True)
pb, wt = w5auc(pf)
print('', flush=True)
print('=== every mechanical column, ranked by |wAUC5 - .5| ===', flush=True)
out = []
for c in mcols:
    x = np.nan_to_num(mech[c].to_numpy().astype(float), nan=0.)
    if np.unique(x).size < 2: continue
    a, w = w5auc(x)
    out.append((c, float(np.average(a, weights=w))))
out.sort(key=lambda t: -abs(t[1]-.5))
rng = np.random.default_rng(276)
BS = [rng.choice(len(ACT), len(ACT), replace=True) for _ in range(2000)]
print('%-44s %9s %9s %8s' % ('column', 'wAUC5', 'P(!=.5)', 'signed'), flush=True)
top20 = []
for c, wa in out[:20]:
    x = np.nan_to_num(mech[c].to_numpy().astype(float), nan=0.)
    a, w = w5auc(x)
    bs = np.array([np.average(a[ix], weights=w[ix]) for ix in BS])
    pv = float((bs > .5).mean() if wa > .5 else (bs < .5).mean())
    top20.append({'col': c, 'wAUC5': round(wa,4), 'P': round(pv,4)})
    print('%-44s %9.4f %9.4f %8s' % (c[:44], wa, pv, '+' if wa > .5 else '-'), flush=True)
bp = np.array([np.average(pb[ix], weights=wt[ix]) for ix in BS])
json.dump({'n_live_lists': len(ACT), 'baselines': {nm: round(wa,4) for nm, wa, _, _ in rows},
           'p_stage2_wAUC5_ci': [round(float(np.percentile(bp,2.5)),4), round(float(np.percentile(bp,97.5)),4)],
           'top_mechanical': top20, 'n_mech_cols': len(mcols)},
          open(OUT/'within5_auc.json','w'), indent=2)
print('', flush=True)
print('p_stage2 wAUC5 95%% CI [%.4f, %.4f]' % (np.percentile(bp,2.5), np.percentile(bp,97.5)), flush=True)
print('wrote within5_auc.json', flush=True)
