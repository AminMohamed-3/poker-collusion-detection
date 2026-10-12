"""STEP280: do the two independently-validated SOFT-PLAY arms carry the SAME information or different information?

Today produced two SP-scoped positives, built from different objects and validated by different nulls:

  STEP277  the induced CITATION rule -- LightGBM on the 662 frozen mechanical columns, fitted on the 85 SP
           development pairs OUTSIDE the audit over their full candidate universes, 8 seeds, blended
           p^(1-w) * rule^w at w = .30.  SP subgroup +.028090, whole audit +.008148, honest split-half over 28
           cells +.003039 at 307/400, within-pair label permutation -.054178.

  STEP272-U/V  MECH_fuse | SP | F1 | w.20 -- the mechanical EVENT probability prior-shifted and fused,
           +.004378 whole audit, maximum over 384 grid cells, and its within-pair permutation null gives
           null grid max q95 +.004083 and p(null max >= arm) = .0417 with the chosen cell's null at -.018242.

If they are the same information, the composition reads like the better of the two and the lane is one win, not
two.  If they are different, it reads additive and Amin's "chain the small wins" instruction has an actual chain.
Either answer is worth having before the last slot is spent.

  RANK        Spearman between the two scores WITHIN each SP pair's candidate set -- redundancy, model-free
  MARGINALS   each arm alone, at its own validated cell, on the identical decode path
  JOINT       log-linear pooling p^(1-wr-wm) * rule^wr * shift(mech,F)^wm over a grid with (0,0) as a cell
  CHAIN       the sequential form a real build would produce by patching twice, since it is not identical
  HONEST      pick the cell on half the audit's table pools, score on the other half, 400 times, control in grid

Nothing here is fitted on the audit: both scores are frozen inputs.  What is paid for is the choice of cell.
"""
import os, sys, json, time
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from collections import Counter
import numpy as np, polars as pl
R = Path('C:/Data/Projects/Money Making/work/poker'); S = Path(os.environ['SCR']); CV = R / 'Claude Version'
OUT = CV / 'experiments/step277_generator_rule_20260919'
sys.path.insert(0, str(CV / 'src')); sys.path.insert(0, str(R / 'Codex Resume/src')); sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology, align, less_than_five
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K = S165.K; CI = 'coordinated_isolation'; SP = 'soft_play'

D = S165.load(); d = D['d']; QIX = D['QIX']; top = D['top']; DEC = D['dec']
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
famall = np.array([fams[p] for p in pid]); key = d.select('pair_id', 'hand_id', 'table_id')
v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
fr = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
t1all = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
cr = fr['chrono_rank'].to_numpy(); PRI = (~(t1all & (famall != CI))).astype(np.int64)

pf = np.zeros(len(pid)); pf[QIX] = np.load(S / 'inc_p_PRODbare.npy')
mf = np.zeros(len(pid)); mf[QIX] = np.load(S / 'inc_p_MECH_fuse.npy')
rule = np.load(OUT / 'induced_rule_audit.npy')
qall = sorted(set(pid[top])); GT = {p: np.flatnonzero((pid == p) & top) for p in qall}
NREL = {q: len(truths[q]) for q in qall}
spq = [q for q in qall if fams[q] == SP]
print('audit queries %d   soft-play %d   candidate rows %d' % (len(qall), len(spq), int(top.sum())), flush=True)


def ap5(hits, q):
    h = 0; s = 0.
    for i, r in enumerate(hits, 1):
        if r:
            h += 1; s += h / i
    return s / min(5, NREL[q])


def shift(p, F):
    q = np.clip(p, 1e-9, 1 - 1e-9); o = q / (1 - q) * F
    return o / (1 + o)


def run(pu):
    """the gate-verified decode: typed traversal, Poisson-binomial censoring, top five."""
    sc = np.zeros(len(pid)); m = []; sel5 = {}
    for q in qall:
        g = GT[q]; p = np.clip(pu[g], 0., 1.)
        o = np.lexsort((cr[g], PRI[g])); wv = np.empty(len(g)); wv[o] = less_than_five(p[o])
        sc[g] = 10.0 + p * wv + 1e-9 * p
        s5 = g[np.argsort(-sc[g], kind='stable')[:5]]
        sel5[q] = tuple(s5); m.append(ap5(rel[s5], q))
    return np.array(m), sel5


def pool(wr, wm, F):
    """log-linear pooling on SOFT-PLAY rows only; weights on production, rule and shifted mech sum to 1."""
    pu = pf.copy()
    if wr == 0 and wm == 0:
        return pu
    sel = famall == SP
    lp = (1 - wr - wm) * np.log(np.clip(pf[sel], 1e-9, 1.))
    if wr > 0:
        lp = lp + wr * np.log(np.clip(rule[sel], 1e-9, 1.))
    if wm > 0:
        lp = lp + wm * np.log(np.clip(shift(mf, F)[sel], 1e-9, 1.))
    pu[sel] = np.exp(lp)
    return pu


def chain(wr, wm, F):
    """the sequential form: patch the rule in, then patch the mech fusion on top of the result."""
    pu = pf.copy(); sel = famall == SP
    if wr > 0:
        pu[sel] = np.clip(pf[sel], 1e-9, 1.) ** (1 - wr) * np.clip(rule[sel], 1e-9, 1.) ** wr
    if wm > 0:
        pu[sel] = np.clip(pu[sel], 1e-9, 1.) ** (1 - wm) * shift(mf, F)[sel] ** wm
    return pu


# ---- GATE: the decode path must reproduce the official summarize on production
D0, SEL0 = run(pf)


def official(pu):
    """the official path: the production decoder applied to 10 + p, scored by summarize."""
    _, mm = summarize(key.with_columns(
        pl.Series('score', DEC(np.where(top, 10.0 + pu, 0.0)))), truths, fams)
    return np.array([mm[q] for q in qall])


OFF = official(pf)
gate = abs(D0.mean() - OFF.mean())
print('GATE decode  mine %.6f   official decoder + summarize %.6f   |diff| %.2e' % (
    D0.mean(), OFF.mean(), gate), flush=True)
assert gate < 1e-9, 'decode path does not match the official metric'
gp = abs(run(pool(.3, 0., 1.))[0].mean() - official(pool(.3, 0., 1.)).mean())
print('GATE decode  same identity on the rule-blended score: |diff| %.2e' % gp, flush=True)
assert gp < 1e-9, 'decode path diverges from the official metric once the score changes'

# ---- RANK: are the two scores the same information within an SP pair?
rs = []
for q in spq:
    g = GT[q]
    a = rule[g]; b = mf[g]
    if a.std() > 0 and b.std() > 0:
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        rs.append(np.corrcoef(ra, rb)[0, 1])
rs = np.array(rs)
rp = []
for q in spq:
    g = GT[q]
    a = rule[g]; b = pf[g]
    if a.std() > 0 and b.std() > 0:
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        rp.append(np.corrcoef(ra, rb)[0, 1])
rm = []
for q in spq:
    g = GT[q]
    a = mf[g]; b = pf[g]
    if a.std() > 0 and b.std() > 0:
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        rm.append(np.corrcoef(ra, rb)[0, 1])
print('', flush=True)
print('RANK within SP pairs   rule vs mech %+.4f (sd %.3f)   rule vs prod %+.4f   mech vs prod %+.4f' % (
    rs.mean(), rs.std(), np.mean(rp), np.mean(rm)), flush=True)

# ---- the grid
WR = (0., .1, .2, .3, .4, .5)
WM = (0., .1, .2, .3)
FS = (1., 10.)
CELL = [(wr, wm, F) for wr in WR for wm in WM for F in FS if wr + wm <= .6 + 1e-9]
CELL = [c for c in CELL if not (c[0] == 0 and c[1] == 0 and c[2] != FS[0])]     # one control, not two
A = np.empty((len(CELL), len(qall))); SELS = {}
t0 = time.perf_counter()
for k, c in enumerate(CELL):
    m, sl = run(pool(*c)); A[k] = m; SELS[c] = sl
print('', flush=True)
print('grid %d cells in %.0fs' % (len(CELL), time.perf_counter() - t0), flush=True)
base = A[CELL.index((0., 0., FS[0]))]
fqa = np.array([fams[q] for q in qall]); tabq = np.array([ptab[q] for q in qall])
hva = np.array([half(ptab[q]) for q in qall]); spm = fqa == SP

rngb = np.random.default_rng(272); pools = sorted(set(ptab.values()))
qbp = {pp: [i for i, q in enumerate(qall) if ptab[q] == pp] for pp in pools}
IDX = [np.concatenate([qbp[pools[i]] for i in rngb.choice(len(pools), len(pools), replace=True)])
       for _ in range(4000)]


def report(nm, m, sl):
    dl = m - base
    bs = np.array([dl[ix].mean() for ix in IDX])
    chg = sum(1 for q in qall if sl[q] != SEL0[q])
    cst = sum(1 for q in qall if set(sl[q]) != set(SEL0[q]))
    print('%-28s %9.6f %+10.6f %7.4f %7d %7d   %+9.5f %+9.5f %+9.5f' % (
        nm, m.mean(), dl.mean(), (bs > 0).mean(), chg, cst, dl[spm].mean(),
        dl[hva == 0].mean(), dl[hva == 1].mean()), flush=True)
    return {'map5': round(float(m.mean()), 6), 'd_dec': round(float(dl.mean()), 6),
            'P_gt_0': round(float((bs > 0).mean()), 4), 'lists_changed': chg, 'lists_set_changed': cst,
            'd_SP': round(float(dl[spm].mean()), 6),
            'h0': round(float(dl[hva == 0].mean()), 6), 'h1': round(float(dl[hva == 1].mean()), 6)}


print('', flush=True)
print('%-28s %9s %10s %7s %7s %7s   %9s %9s %9s' % (
    'arm', 'MAP@5', 'd_dec', 'P(>0)', 'chg', 'chgSet', 'd_SP', 'h0', 'h1'), flush=True)
res = {'rank_rule_vs_mech': round(float(rs.mean()), 4), 'rank_rule_vs_prod': round(float(np.mean(rp)), 4),
       'rank_mech_vs_prod': round(float(np.mean(rm)), 4), 'arms': {}}
RULEONLY = (.3, 0., FS[0]); MECHONLY = (0., .2, 1.)
for nm, c in (('rule only w=.30', RULEONLY), ('mech only w=.20 F1', MECHONLY)):
    res['arms'][nm] = report(nm, A[CELL.index(c)], SELS[c])
JOINTS = [(.3, .1, 1.), (.3, .2, 1.), (.2, .2, 1.), (.2, .1, 1.), (.3, .2, 10.), (.4, .2, 1.)]
for c in JOINTS:
    nm = 'joint wr=%.1f wm=%.1f F%g' % c
    res['arms'][nm] = report(nm, A[CELL.index(c)], SELS[c])
dr = res['arms']['rule only w=.30']['d_dec']; dm = res['arms']['mech only w=.20 F1']['d_dec']
bestj = max(JOINTS, key=lambda c: A[CELL.index(c)].mean())
dj = float((A[CELL.index(bestj)] - base).mean())
print('', flush=True)
print('ADDITIVITY   rule %+.6f + mech %+.6f = %+.6f   best joint %+.6f   ratio %.2f   (1.0 = additive, %.2f = redundant)'
      % (dr, dm, dr + dm, dj, dj / max(dr + dm, 1e-9), max(dr, dm) / max(dr + dm, 1e-9)), flush=True)
res['additivity'] = {'d_rule': dr, 'd_mech': dm, 'sum': round(dr + dm, 6), 'best_joint_cell': list(bestj),
                     'd_best_joint': round(dj, 6), 'ratio_to_sum': round(dj / max(dr + dm, 1e-9), 3),
                     'ratio_if_redundant': round(max(dr, dm) / max(dr + dm, 1e-9), 3)}

# ---- CHAIN: the form a real build produces by patching twice
print('', flush=True)
for c in [RULEONLY, (.3, .1, 1.), (.3, .2, 1.)]:
    m, sl = run(chain(*c))
    res['arms']['CHAIN wr=%.1f wm=%.1f F%g' % c] = report('CHAIN wr=%.1f wm=%.1f F%g' % c, m, sl)

# ---- HONEST: pay for choosing the cell
pa = np.array(sorted(set(tabq))); rr = np.random.default_rng(280)
held = []; picks = []
for _ in range(200):
    pr = rr.permutation(len(pa))
    Am = np.isin(tabq, pa[pr[:len(pa) // 2]]); Bm = ~Am
    for trm, tem in ((Am, Bm), (Bm, Am)):
        kk = int(np.argmax(A[:, trm].mean(1)))
        held.append(float(A[kk, tem].mean() - base[tem].mean())); picks.append(CELL[kk])
held = np.array(held); cnt = Counter(picks)
print('', flush=True)
print('HONEST split-half over %d cells (control included): held-out d_dec %+.6f   positive %d/400' % (
    len(CELL), held.mean(), int((held > 0).sum())), flush=True)
for c, n in cnt.most_common(4):
    print('   modal pick wr=%.1f wm=%.1f F%g : %d of 400' % (c[0], c[1], c[2], n), flush=True)
res['honest'] = {'cells': len(CELL), 'heldout_d_dec': round(float(held.mean()), 6),
                 'positive_400': int((held > 0).sum()),
                 'modal': [[c[0], c[1], c[2], n] for c, n in cnt.most_common(4)]}
# honest restricted to the rule axis alone, for comparison with the 28-cell read
mask = [i for i, c in enumerate(CELL) if c[1] == 0]
h2 = []
for _ in range(200):
    pr = rr.permutation(len(pa))
    Am = np.isin(tabq, pa[pr[:len(pa) // 2]]); Bm = ~Am
    for trm, tem in ((Am, Bm), (Bm, Am)):
        kk = mask[int(np.argmax(A[np.ix_(mask, np.flatnonzero(trm))].mean(1)))]
        h2.append(float(A[kk, tem].mean() - base[tem].mean()))
h2 = np.array(h2)
print('   rule axis only (%d cells): %+.6f   %d/400' % (len(mask), h2.mean(), int((h2 > 0).sum())), flush=True)
res['honest_rule_axis'] = {'cells': len(mask), 'heldout_d_dec': round(float(h2.mean()), 6),
                           'positive_400': int((h2 > 0).sum())}

# ---- the rule-weight axis in full: is the shipped w = .30 the right cell?
print('', flush=True)
print('RULE AXIS (wm = 0), the shipped arm is w = .30', flush=True)
print('%-28s %9s %10s %7s %7s %7s   %9s %9s %9s' % (
    'arm', 'MAP@5', 'd_dec', 'P(>0)', 'chg', 'chgSet', 'd_SP', 'h0', 'h1'), flush=True)
res['rule_axis'] = {}
for wr in WR:
    c = (wr, 0., FS[0])
    if c in CELL:
        res['rule_axis']['w=%.2f' % wr] = report('rule w=%.2f' % wr, A[CELL.index(c)], SELS[c])
json.dump(res, open(OUT / 'compose.json', 'w'), indent=2)
print('', flush=True)
print('wrote compose.json', flush=True)
