"""STEP276-AN: does ANY signal add ordering information to p*q inside the selected five, paid for honestly?

AL measured the prize: +.055046 decoded from re-ordering alone, set untouched.  AM measured the baseline:
production's p*q has within-five AUC .7217, and the best single mechanical column reaches only .094 from chance
while several of the apparent leaders (t4_shared_idx, t4_cumabsflow_before, t1_bbs_aggressor_*) sit at EXACTLY
chrono_rank's .4385, i.e. they are chronology proxies, not new information.

Here the ordering is blended in within-five rank space, score = rank(p*q) + lam * rank(signal), and the pair
(column, lam) is SELECTED ON HALF THE TABLE POOLS AND SCORED ON THE OTHER HALF, 400 times.  No LightGBM is
fitted anywhere, so the +-.004 fold-assignment noise that closed the refit lanes today cannot enter: the only
randomness is the split, and selection over 669 x 15 cells is paid for in full.

lam = 0 is in the grid as the control -- if the honest held-out delta against lam = 0 is <= 0 the lane is closed,
and that is the answer regardless of how good the in-sample cell looks.
"""
import os, sys, json, time
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, polars as pl
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R / 'Claude Version'
OUT = CV / 'experiments/step276_evidence_headroom_20260919'
sys.path.insert(0, str(CV / 'src')); sys.path.insert(0, str(R / 'Codex Resume/src')); sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology, align, less_than_five
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K = S165.K; CI = 'coordinated_isolation'
D = S165.load(); d, Q, QIX, dec = D['d'], D['Q'], D['QIX'], D['dec']
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
famall = np.array([fams[p] for p in pid]); top = D['top']
key = d.select('pair_id', 'hand_id', 'table_id')
v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
fr = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
t1all = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
cr = fr['chrono_rank'].to_numpy()
PRI = (~(t1all & (famall != CI))).astype(np.int64)
P = np.load(S / 'inc_p_PRODbare.npy'); pf = np.zeros(len(pid)); pf[QIX] = P
qall = sorted(set(pid[top]))
GT = {p: np.flatnonzero((pid == p) & top) for p in qall}
sc = np.zeros(len(pid)); qv = np.zeros(len(pid))
for q, g in GT.items():
    p = np.clip(pf[g], 0., 1.)
    o = np.lexsort((cr[g], PRI[g])); w = np.empty(len(g)); w[o] = less_than_five(p[o])
    qv[g] = w; sc[g] = 10.0 + p * w + 1e-9 * p
SEL = {q: GT[q][np.argsort(-sc[GT[q]], kind='stable')[:5]] for q in qall}
NREL = {q: len(truths[q]) for q in qall}
ACT = [q for q in qall if 0 < rel[SEL[q]].sum() < len(SEL[q])]
SIX = np.array([SEL[q] for q in ACT])
Y = rel[SIX]
tabq = np.array([ptab[q] for q in ACT])
PAIRW = (Y[:, :, None] * (1 - Y)[:, None, :]).astype(float)
WT = PAIRW.sum((1, 2))
print('live lists %d   contrast pairs %d' % (len(ACT), int(WT.sum())), flush=True)


def wr(vals):
    """within-list average rank scaled to [0,1], shape (n,5)"""
    out = np.empty(vals.shape, dtype=float)
    for i in range(vals.shape[0]):
        _, inv, cnt = np.unique(vals[i], return_inverse=True, return_counts=True)
        cum = np.cumsum(cnt); start = cum - cnt
        out[i] = (start[inv] + cum[inv] - 1) / 2.0
    return out / max(vals.shape[1] - 1, 1)


def auc(Sm):
    gt = (Sm[:, :, None] > Sm[:, None, :]).astype(float) + .5 * (Sm[:, :, None] == Sm[:, None, :])
    return (gt * PAIRW).sum((1, 2)) / WT


RP = wr(sc[SIX])
MPTH = CV / 'experiments/step254_mechanical_tensor_20260917'
mech = d.select(K).join(pl.read_parquet(MPTH / 'step254_features.parquet'), on=K, how='left')
mcols = [c for c in mech.columns if c not in K and mech[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Boolean)]
CAND = {'chrono_rank': cr.astype(float), 'q_censor': qv, 'oof_stage1': d['oof'].to_numpy().astype(float),
        'T1_flag': t1all.astype(float), 'p_stage2': pf}
for c in mcols:
    CAND[c] = np.nan_to_num(mech[c].to_numpy().astype(float), nan=0.)
names = ['chrono_rank', 'q_censor', 'oof_stage1', 'T1_flag']  # PRE-SPECIFIED on mechanism, named before the 664-column sweep
LAM = np.array([0., .05, .1, .2, .35, .5, .75, 1.0, -.05, -.1, -.2, -.35, -.5, -.75, -1.0])
print('candidates %d   lambda cells %d   total %d' % (len(names), len(LAM), len(names) * len(LAM)), flush=True)
A = np.empty((len(names), len(LAM), len(ACT)))
t0 = time.perf_counter()
for i, c in enumerate(names):
    rc = wr(CAND[c][SIX])
    for j, lam in enumerate(LAM):
        A[i, j] = auc(RP + lam * rc)
    if i and i % 200 == 0:
        print('   %d/%d  (%.0fs)' % (i, len(names), time.perf_counter() - t0), flush=True)
A0 = A[0, 0]
assert np.allclose(A[:, 0], A0[None, :]), 'lam=0 must be column-independent'
BASE = float(np.average(A0, weights=WT))
print('', flush=True)
print('baseline within-five wAUC of p*q: %.4f   (%d cells, %.0fs)' % (BASE, len(names) * len(LAM), time.perf_counter() - t0), flush=True)
F = A.reshape(-1, len(ACT)); CELL = [(c, float(l)) for c in names for l in LAM]
full = (F * WT).sum(1) / WT.sum()
kb = int(np.argmax(full))
print('in-sample best: %-40s lam %+0.2f   wAUC %.4f  (%+.4f vs p*q)' % (
    CELL[kb][0][:40], CELL[kb][1], full[kb], full[kb] - BASE), flush=True)
pa = np.array(sorted(set(tabq))); rr = np.random.default_rng(276)
held = []; picks = []
for _ in range(200):
    pr = rr.permutation(len(pa))
    Am = np.isin(tabq, pa[pr[:len(pa) // 2]]); Bm = ~Am
    for tr, te in ((Am, Bm), (Bm, Am)):
        tw = WT[tr]; ew = WT[te]
        kk = int(np.argmax((F[:, tr] * tw).sum(1) / tw.sum()))
        held.append(float((F[kk, te] * ew).sum() / ew.sum()) - float((A0[te] * ew).sum() / ew.sum()))
        picks.append(CELL[kk])
held = np.array(held)
from collections import Counter
cnt = Counter(picks); modal = cnt.most_common(3)
print('', flush=True)
print('HONEST split-half over all %d cells: held-out d_wAUC %+.5f   positive %d/400' % (
    len(CELL), held.mean(), int((held > 0).sum())), flush=True)
print('   modal picks: %s' % [(c[:34], l, n) for (c, l), n in modal], flush=True)
res = {'baseline_wAUC5': round(BASE, 4), 'n_cells': len(CELL), 'n_live_lists': len(ACT),
       'insample_best': {'col': CELL[kb][0], 'lam': CELL[kb][1], 'wAUC': round(float(full[kb]), 4),
                         'd_vs_pq': round(float(full[kb] - BASE), 4)},
       'honest_heldout_d_wAUC': round(float(held.mean()), 6), 'honest_pos_400': int((held > 0).sum()),
       'modal_picks': [{'col': c, 'lam': l, 'n': n} for (c, l), n in modal]}


def ap5(hits, q):
    h = 0; s = 0.
    for i, r in enumerate(hits, 1):
        if r:
            h += 1; s += h / i
    return s / min(5, NREL[q])


_, mprod = summarize(key.with_columns(pl.Series('score', sc)), truths, fams)
D0 = np.array([mprod[q] for q in qall])
fqa = np.array([fams[q] for q in qall]); hva = np.array([half(ptab[q]) for q in qall])
rngb = np.random.default_rng(272); pools = sorted(set(ptab.values()))
qbp = {pp: [i for i, q in enumerate(qall) if ptab[q] == pp] for pp in pools}
IDX = [np.concatenate([qbp[pools[i]] for i in rngb.choice(len(pools), len(pools), replace=True)]) for _ in range(4000)]
print('', flush=True)
print('%-44s %9s %11s %8s   %9s %9s %9s' % ('reorder rule (SET HELD FIXED)', 'MAP@5', 'd_dec', 'P(>0)', 'DT', 'SP', 'CI'), flush=True)
print('%-44s %9.6f %+11.6f' % ('production order (p*q)', D0.mean(), 0.), flush=True)
dec_res = {}
seen = set()
for tag, (c, lam) in [('honest modal pick', modal[0][0]), ('in-sample best', CELL[kb])]:
    if (c, lam) in seen:
        continue
    seen.add((c, lam))
    if lam == 0.:
        dec_res[tag] = {'col': c, 'lam': lam, 'd_dec': 0.0, 'note': 'lam=0 control won -- no change'}
        print('%-44s %9s %11s %8s' % (tag + ': lam=0 CONTROL WON', '--', '--', '--'), flush=True)
        continue
    m = []
    for q in qall:
        s5 = SEL[q]
        rp = wr(sc[s5][None, :])[0]; rc2 = wr(CAND[c][s5][None, :])[0]
        o = np.argsort(-(rp + lam * rc2), kind='stable')
        m.append(ap5(rel[s5][o], q))
    m = np.array(m); dl = m - D0
    bs = np.array([dl[ix].mean() for ix in IDX])
    dec_res[tag] = {'col': c, 'lam': lam, 'map5': round(float(m.mean()), 6), 'd_dec': round(float(dl.mean()), 6),
                    'P_gt_0': round(float((bs > 0).mean()), 4),
                    'DT': round(float(dl[fqa == 'directed_transfer'].mean()), 6),
                    'SP': round(float(dl[fqa == 'soft_play'].mean()), 6), 'CI': round(float(dl[fqa == CI].mean()), 6),
                    'h0': round(float(dl[hva == 0].mean()), 6), 'h1': round(float(dl[hva == 1].mean()), 6)}
    print('%-44s %9.6f %+11.6f %8.4f   %+9.5f %+9.5f %+9.5f' % (
        ('%s: %s lam%+.2f' % (tag, c[:20], lam))[:44], m.mean(), dl.mean(), (bs > 0).mean(),
        dl[fqa == 'directed_transfer'].mean(), dl[fqa == 'soft_play'].mean(), dl[fqa == CI].mean()), flush=True)
res['decoded'] = dec_res
json.dump(res, open(OUT / 'within5_prespec.json', 'w'), indent=2)
print('', flush=True)
print('wrote within5_prespec.json  (%.0fs)' % (time.perf_counter() - t0), flush=True)
