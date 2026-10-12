"""STEP277-I: the DEPLOYABLE twin of the soft-play finding -- already computed on both sides.

STEP277-F/G/H found that an out-of-sample-induced per-hand citation rule, blended into production's p for SOFT
PLAY only at w = .30, gains +.024 on soft play (+.0081 whole audit, honest split-half +.003039 with 307/400
positive), and that the gain is the LABELS: a within-pair label permutation reads -.054 and training on the other
families reads -.023, while five hyperparameter settings and ten 70% training subsets are all positive.

My own rule cannot ship: the STEP254 mechanical tensor is development-only by construction (keys off step232's
1,860-pair hand_oof and development_labels.csv, with hard asserts), so porting it to the evaluation phase is a
rewrite, not a flag.

But STEP257 already did exactly that work and kept the model outputs:
  development, OUT OF FOLD   hand_oof_step254.parquet -> mech_p_SP, mechplus_p_SP, and perm_p_SP (a permutation
                             control that was computed at the time, so the null costs nothing to run)
  evaluation                 eval_hand_scores_batch_*.parquet -> p_SP for all 112,540 pairs, produced by the SAME
                             mechplus_SP_fold* models, with a reproduction gate reading max_abs_diff 0.0,
                             spearman 1.0, PASS_EXACT
So if mechplus_p_SP reproduces the gain on the audit, the build needs no new features and no port.

Caveat stated up front, because it is the one real difference: mechplus_p_SP is cross-fitted over 5 folds on all
1,860 development pairs, which INCLUDE the audit's 162.  It is out of fold per hand -- the same construction
production's own oof uses -- but it is weaker than my rule's strictly disjoint 210/162 split.  That is precisely
why both are reported side by side: agreement between a 5-fold OOF score and a disjoint-pair score is the
evidence, not either one alone.
"""
import os, sys, json
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, polars as pl
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R / 'Claude Version'
OUT = CV / 'experiments/step277_generator_rule_20260919'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CV / 'src')); sys.path.insert(0, str(R / 'Codex Resume/src')); sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology, align, less_than_five
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K = S165.K; CI = 'coordinated_isolation'; SP = 'soft_play'; DT = 'directed_transfer'
D = S165.load(); d = D['d']; QIX = D['QIX']; top = D['top']
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
famall = np.array([fams[p] for p in pid]); key = d.select('pair_id', 'hand_id', 'table_id')
OOFP = CV / 'experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet'
SCOLS = ['mech_p_SP', 'mechplus_p_SP', 'perm_p_SP', 'mech_p_any', 'mechplus_p_any',
         'mech_p_DT', 'mechplus_p_DT', 'mech_p_CI', 'mechplus_p_CI']
oof = d.select(K).join(pl.read_parquet(OOFP, columns=K + SCOLS), on=K, how='left')
for c in SCOLS:
    print('%-18s non-null on the audit: %d of %d' % (c, oof[c].drop_nulls().len(), oof.height), flush=True)
SIG = {c: np.nan_to_num(oof[c].to_numpy().astype(float), nan=0.) for c in SCOLS}
v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
fr = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
t1all = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
cr = fr['chrono_rank'].to_numpy()
PRI = (~(t1all & (famall != CI))).astype(np.int64)
P = np.load(S / 'inc_p_PRODbare.npy'); pf = np.zeros(len(pid)); pf[QIX] = P
qall = sorted(set(pid[top]))
GT = {p: np.flatnonzero((pid == p) & top) for p in qall}
NREL = {q: len(truths[q]) for q in qall}
fqa = np.array([fams[q] for q in qall]); hva = np.array([half(ptab[q]) for q in qall])


def ap5(hits, q):
    h = 0; s = 0.
    for i, r in enumerate(hits, 1):
        if r:
            h += 1; s += h / i
    return s / min(5, NREL[q])


def run(pu):
    sc = np.zeros(len(pid)); m = []; sel = {}
    for q in qall:
        g = GT[q]; p = np.clip(pu[g], 0., 1.)
        o = np.lexsort((cr[g], PRI[g])); wv = np.empty(len(g)); wv[o] = less_than_five(p[o])
        sc[g] = 10.0 + p * wv + 1e-9 * p
        s5 = g[np.argsort(-sc[g], kind='stable')[:5]]
        sel[q] = tuple(s5); m.append(ap5(rel[s5], q))
    return np.array(m), sel, sc


D0, SEL0, sc0 = run(pf)
_, mchk = summarize(key.with_columns(pl.Series('score', sc0)), truths, fams)
assert np.allclose(D0, np.array([mchk[q] for q in qall])), 'GATE'
print('', flush=True)
print('GATE ok.  production decoded %.6f   soft-play subset %.6f' % (D0.mean(), D0[fqa == SP].mean()), flush=True)
rngb = np.random.default_rng(272); pools = sorted(set(ptab.values()))
qbp = {pp: [i for i, q in enumerate(qall) if ptab[q] == pp] for pp in pools}
IDX = [np.concatenate([qbp[pools[i]] for i in rngb.choice(len(pools), len(pools), replace=True)]) for _ in range(4000)]
WS = (.05, .1, .15, .2, .3, .4, .5)


def blend(sig, scope, w):
    pu = pf.copy(); sel = np.ones(len(pid), bool) if scope == 'ALL' else (famall == scope)
    pu[sel] = np.clip(pf[sel], 1e-9, 1.) ** (1 - w) * np.clip(sig[sel], 1e-9, 1.) ** w
    return pu


print('', flush=True)
print('%-30s %6s %9s %11s %8s %9s %9s %6s %6s' % (
    'signal x scope', 'w', 'MAP@5', 'd_dec', 'P(>0)', 'h0', 'h1', 'chg', 'set'), flush=True)
res = {}
for cname, scope in (('mechplus_p_SP', SP), ('mech_p_SP', SP), ('perm_p_SP', SP),
                     ('mechplus_p_DT', DT), ('mechplus_p_CI', CI), ('mechplus_p_any', 'ALL')):
    for w in WS:
        m, sel, _ = run(blend(SIG[cname], scope, w)); dl = m - D0
        bs = np.array([dl[ix].mean() for ix in IDX])
        fd = dl[fqa == scope].mean() if scope != 'ALL' else dl.mean()
        chg = sum(1 for q in qall if sel[q] != SEL0[q])
        cst = sum(1 for q in qall if set(sel[q]) != set(SEL0[q]))
        res['%s|%s|w=%.2f' % (cname, scope, w)] = {
            'map5': round(float(m.mean()), 6), 'd_dec': round(float(dl.mean()), 6),
            'P_gt_0': round(float((bs > 0).mean()), 4), 'famdelta': round(float(fd), 6),
            'h0': round(float(dl[hva == 0].mean()), 6), 'h1': round(float(dl[hva == 1].mean()), 6),
            'lists_changed': chg, 'lists_set_changed': cst}
        r = res['%s|%s|w=%.2f' % (cname, scope, w)]
        print('%-30s %6.2f %9.6f %+11.6f %8.4f %+9.5f %+9.5f %6d %6d' % (
            '%s x %s' % (cname[:16], scope[:8]), w, r['map5'], r['d_dec'], r['P_gt_0'], r['h0'], r['h1'], chg, cst), flush=True)
    print('', flush=True)
# honest split-half over the SP grid for the deployable signal, control w=0 included
CELL = [('mechplus_p_SP', w) for w in (0.,) + WS] + [('mech_p_SP', w) for w in (0.,) + WS]
A = np.empty((len(CELL), len(qall)))
for k2, (cn, w) in enumerate(CELL):
    A[k2] = D0 if w == 0 else run(blend(SIG[cn], SP, w))[0]
base = D0
tabq = np.array([ptab[q] for q in qall]); pa = np.array(sorted(set(tabq))); rr = np.random.default_rng(277)
held = []; picks = []
for _ in range(200):
    pr = rr.permutation(len(pa))
    Am = np.isin(tabq, pa[pr[:len(pa) // 2]]); Bm = ~Am
    for trm, tem in ((Am, Bm), (Bm, Am)):
        kk = int(np.argmax(A[:, trm].mean(1)))
        held.append(float(A[kk, tem].mean() - base[tem].mean())); picks.append(CELL[kk])
held = np.array(held)
from collections import Counter
print('HONEST split-half over %d cells (both signals x w, control included): d_dec %+.6f  positive %d/400' % (
    len(CELL), held.mean(), int((held > 0).sum())), flush=True)
print('   modal picks: %s' % [(c, w, n) for (c, w), n in Counter(picks).most_common(3)], flush=True)
res['honest'] = {'heldout_d_dec': round(float(held.mean()), 6), 'pos_400': int((held > 0).sum()),
                 'modal': [[c, w, n] for (c, w), n in Counter(picks).most_common(3)], 'n_cells': len(CELL)}
json.dump(res, open(OUT / 'sp_deploy.json', 'w'), indent=2)
print('', flush=True)
print('wrote sp_deploy.json', flush=True)
