"""STEP277-D: do production's listed hands VIOLATE the generator's hard constraints?  Honest split by design.

STEP277-C found predicates the generator never violates over 1,817 cited hands of 372 pairs -- above all
`t5_fold_to_nobody`, satisfied by ZERO cited hands in all three families while covering 46-49% of each pair's
candidates, plus per-family ones (DT: every cited hand has t5_transfer_bb > 0, true of only 22% of hands;
DT exclusion t1_ap_fold_folded; CI exclusion t1_fsA_raise_mn).  Zero of 1,817 against ~850 expected under
randomness is not a small-sample accident.

But a constraint production already respects is worth exactly nothing -- that is what killed the block
composition lane in 19.234(j).  So the only question that matters: does production's own output violate it?

DESIGN, and the split is the point.  The 372 labelled pairs INCLUDE the audit's 162, so deriving a constraint on
all 372 and then testing it on the audit is in-sample.  Here constraints are DERIVED on the 210 pairs OUTSIDE the
audit and TESTED on the 162, which makes the violation counts and the decoded delta honest without any refit.

  1  derive: every column with recall 1.0000 (inclusion) or 0.0000 (exclusion) on the 210 held-out pairs
  2  confirm: how often the AUDIT's truth violates it -- the honesty check on the derivation
  3  measure: how often PRODUCTION's 810 listed slots violate it -- the value check
  4  apply:  demote violating candidates below all compliant ones inside the top-20 and decode
"""
import os, sys, json, time
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
K = S165.K; CI = 'coordinated_isolation'
MP = CV / 'experiments/step254_mechanical_tensor_20260917/step254_features.parquet'
ev = pl.read_csv(R / 'data/development_evidence.csv')
fam372 = dict(ev.select('pair_id', 'behavior_family').unique().iter_rows())
D = S165.load(); d = D['d']; QIX = D['QIX']; top = D['top']
apairs = set(d['pair_id'].unique().to_list())
hpairs = [p for p in fam372 if p not in apairs]
print('labelled pairs %d   audit %d   held-out for derivation %d' % (len(fam372), len(apairs & set(fam372)), len(hpairs)), flush=True)
hands = pl.read_parquet(R / 'data/hands.parquet', columns=['hand_id', 'started_at'])
mech = pl.scan_parquet(MP).filter(pl.col('pair_id').is_in(hpairs)).collect().join(hands, on='hand_id', how='left')
mech = mech.join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'], how='left')
mech = mech.with_columns(pl.col('pair_id').replace_strict(fam372, default=None).alias('family')).sort(['pair_id', 'started_at'])
cols = [c for c in mech.columns if c.startswith(('t1_', 't2_', 't3_', 't4_', 't5_'))]
Mh = np.nan_to_num(mech.select(cols).to_numpy().astype(float), nan=0., posinf=0., neginf=0.)
ch = np.array([v is not None for v in mech['evidence_rank'].to_list()])
fh = mech['family'].to_numpy()
FAMS = sorted(set(fam372.values()))
print('derivation set: %d rows, %d cited' % (len(ch), int(ch.sum())), flush=True)
# ---- 1. derive
RULES = {}
for f in FAMS:
    ix = np.flatnonzero(fh == f); cf = ch[ix]
    inc, exc = [], []
    for j, cn in enumerate(cols):
        pos = Mh[ix, j] > 0
        sup = float(pos.mean())
        if sup <= .02 or sup >= .98:
            continue
        r = float(pos[cf].mean())
        if r >= 0.9999:
            inc.append((cn, sup))
        elif r <= 1e-9:
            exc.append((cn, sup))
    inc.sort(key=lambda t: t[1]); exc.sort(key=lambda t: -t[1])
    RULES[f] = {'include': inc[:6], 'exclude': exc[:6]}
    print('', flush=True)
    print('%s  derived on %d held-out pairs: %d never-violated inclusions, %d zero-cited exclusions' % (
        f.upper(), int(len(set(mech['pair_id'].to_numpy()[ix]))), len(inc), len(exc)), flush=True)
    for cn, sup in inc[:4]:
        print('   INCLUDE  %-38s support %.4f  (lift %.2fx)' % (cn[:38], sup, 1 / max(sup, 1e-9)), flush=True)
    for cn, sup in exc[:4]:
        print('   EXCLUDE  %-38s removes %.4f' % (cn[:38], sup), flush=True)
# ---- audit side
pid = d['pair_id'].to_numpy(); rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
truths = {kk[0]: set(gg['hand_id']) for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
famall = np.array([fams[p] for p in pid]); key = d.select('pair_id', 'hand_id', 'table_id')
am = d.select(K).join(pl.scan_parquet(MP).collect(), on=K, how='left')
MA = np.nan_to_num(am.select(cols).to_numpy().astype(float), nan=0., posinf=0., neginf=0.)
CIX = {c: j for j, c in enumerate(cols)}
v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
fr = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
t1all = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
cr = fr['chrono_rank'].to_numpy()
PRI = (~(t1all & (famall != CI))).astype(np.int64)
P = np.load(S / 'inc_p_PRODbare.npy'); pf = np.zeros(len(pid)); pf[QIX] = P
qall = sorted(set(pid[top]))
GT = {p: np.flatnonzero((pid == p) & top) for p in qall}
sc = np.zeros(len(pid))
for q, g in GT.items():
    p = np.clip(pf[g], 0., 1.)
    o = np.lexsort((cr[g], PRI[g])); w = np.empty(len(g)); w[o] = less_than_five(p[o])
    sc[g] = 10.0 + p * w + 1e-9 * p
SEL = {q: GT[q][np.argsort(-sc[GT[q]], kind='stable')[:5]] for q in qall}
NREL = {q: len(truths[q]) for q in qall}


def viol(rows, f, cn, kind):
    x = MA[rows, CIX[cn]] > 0
    return ~x if kind == 'include' else x


print('', flush=True)
print('%-40s %-9s %-6s %11s %11s %11s' % ('constraint', 'kind', 'fam', 'truth viol', 'prod viol', 'top20 viol'), flush=True)
rep = []
for f in FAMS:
    qf = [q for q in qall if fams[q] == f]
    for kind in ('include', 'exclude'):
        for cn, sup in RULES[f][kind]:
            tv = pv = c20 = n20 = ntr = npr = 0
            for q in qf:
                g = GT[q]; tr = g[rel[g] == 1]; se = SEL[q]
                tv += int(viol(tr, f, cn, kind).sum()); ntr += len(tr)
                pv += int(viol(se, f, cn, kind).sum()); npr += 5
                c20 += int(viol(g, f, cn, kind).sum()); n20 += len(g)
            rep.append({'family': f, 'col': cn, 'kind': kind, 'support': sup,
                        'truth_viol': tv, 'n_truth': ntr, 'prod_viol': pv, 'n_prod': npr,
                        'top20_viol': c20, 'n_top20': n20})
            print('%-40s %-9s %-6s %5d/%-5d %5d/%-5d %5d/%-5d' % (
                cn[:40], kind, f[:5], tv, ntr, pv, npr, c20, n20), flush=True)
json.dump({'rules': {f: RULES[f] for f in FAMS}, 'audit_check': rep,
           'n_derivation_pairs': len(hpairs)}, open(OUT / 'gen_apply.json', 'w'), indent=2)
# ---- 4. apply the constraints that production actually violates
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
live = [r for r in rep if r['prod_viol'] > 0 and r['truth_viol'] == 0]
print('', flush=True)
print('constraints that production VIOLATES and the audit truth does NOT: %d' % len(live), flush=True)
if not live:
    print('LANE VERDICT: production already respects every derived constraint -- nothing to gain.', flush=True)
    sys.exit(0)
print('', flush=True)
print('%-46s %9s %11s %8s   %9s %9s' % ('arm (demote violators inside the top-20)', 'MAP@5', 'd_dec', 'P(>0)', 'slots', 'lists'), flush=True)
arms = {}
for r in sorted(live, key=lambda r: -r['prod_viol'])[:8]:
    f, cn, kind = r['family'], r['col'], r['kind']
    sc2 = sc.copy(); nl = 0
    for q in qall:
        if fams[q] != f:
            continue
        g = GT[q]; bad = viol(g, f, cn, kind)
        if not bad.any() or bad.all():
            continue
        vals = np.sort(sc[g])[::-1]
        ordr = np.argsort(-sc[g], kind='stable')
        newo = [i for i in ordr if not bad[i]] + [i for i in ordr if bad[i]]
        tmp = np.empty(len(g))
        for rk, i in enumerate(newo):
            tmp[i] = vals[rk]
        if not np.array_equal(np.argsort(-tmp, kind='stable')[:5], ordr[:5]):
            nl += 1
        sc2[g] = tmp
    m = []
    for q in qall:
        s5 = GT[q][np.argsort(-sc2[GT[q]], kind='stable')[:5]]
        m.append(ap5(rel[s5], q))
    m = np.array(m); dl = m - D0
    bs = np.array([dl[ix].mean() for ix in IDX])
    arms['%s|%s|%s' % (f, cn, kind)] = {'map5': round(float(m.mean()), 6), 'd_dec': round(float(dl.mean()), 6),
                                        'P_gt_0': round(float((bs > 0).mean()), 4), 'lists_changed': nl,
                                        'prod_viol_slots': r['prod_viol'],
                                        'h0': round(float(dl[hva == 0].mean()), 6), 'h1': round(float(dl[hva == 1].mean()), 6),
                                        'DT': round(float(dl[fqa == 'directed_transfer'].mean()), 6),
                                        'SP': round(float(dl[fqa == 'soft_play'].mean()), 6),
                                        'CI': round(float(dl[fqa == CI].mean()), 6)}
    print('%-46s %9.6f %+11.6f %8.4f   %9d %9d' % (
        ('%s %s %s' % (f[:3], cn[:28], kind[:3]))[:46], m.mean(), dl.mean(), (bs > 0).mean(), r['prod_viol'], nl), flush=True)
json.dump({'rules': {f: RULES[f] for f in FAMS}, 'audit_check': rep, 'arms': arms,
           'prod_dec': float(D0.mean()), 'n_derivation_pairs': len(hpairs)},
          open(OUT / 'gen_apply.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_apply.json', flush=True)
