"""STEP277-E: provably-uncitable hands are consuming the decoder's censoring mass.

STEP277-D established two things.  The generator's hard constraints are REAL -- derived on the 210 pairs outside
the audit they are violated by 0 of 195 CI, 0 of 353 DT and 0 of 223 SP cited hands.  And production already
respects them in its OUTPUT: at most 2 of 360 listed DT slots violate any of them, and demoting violators moves
decoded MAP@5 by exactly +0.000000.

But they are violated all over the CANDIDATE POOL -- DT's t5_transfer_bb inclusion by 367 of 1,440 top-20 rows,
t1_ap_fold_folded by 225, CI's t3_both_blinds by 38 of 860.  That matters for a mechanism nothing has tested,
because the typed decoder's censoring term is

    q_i = P(fewer than five of the hands ahead of i in the traversal are listed)

accumulated over EVERY candidate in the pool.  A hand the generator provably cannot cite still contributes its p
to that accumulation, so it eats censoring mass that belongs to citable hands and depresses q for everything
behind it.  Removing ineligible hands from the accumulation is not a re-parameterisation of the decoder -- the
form, the cap, the exponent and the traversal are all untouched -- it is a correction to the SET the sum runs
over, justified by a constraint measured on the generator's own labels and derived out of sample.

Arms, all on production's frozen p, no fit anywhere:
  QMASS    ineligible hands excluded from the q accumulation, still selectable (conservative)
  QMASS+X  ineligible hands excluded from the accumulation AND from selection
  CTRL_RND a random mask of the SAME per-pair size as the real one, 20 draws -- the control that says whether any
           gain is the constraint or merely perturbing the accumulation
The control is the point: an arm that cannot beat its own size-matched random mask has learned nothing.
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
hands = pl.read_parquet(R / 'data/hands.parquet', columns=['hand_id', 'started_at'])
mech = pl.scan_parquet(MP).filter(pl.col('pair_id').is_in(hpairs)).collect().join(hands, on='hand_id', how='left')
mech = mech.join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'], how='left')
mech = mech.with_columns(pl.col('pair_id').replace_strict(fam372, default=None).alias('family'))
cols = [c for c in mech.columns if c.startswith(('t1_', 't2_', 't3_', 't4_', 't5_'))]
Mh = np.nan_to_num(mech.select(cols).to_numpy().astype(float), nan=0., posinf=0., neginf=0.)
ch = np.array([v is not None for v in mech['evidence_rank'].to_list()]); fh = mech['family'].to_numpy()
FAMS = sorted(set(fam372.values()))
RULES = {}
for f in FAMS:
    ix = np.flatnonzero(fh == f); cf = ch[ix]
    inc, exc = [], []
    for j, cn in enumerate(cols):
        pos = Mh[ix, j] > 0; sup = float(pos.mean())
        if sup <= .02 or sup >= .98:
            continue
        r = float(pos[cf].mean())
        if r >= 0.9999:
            inc.append(cn)
        elif r <= 1e-9:
            exc.append(cn)
    RULES[f] = {'include': inc, 'exclude': exc}
    print('%-24s derived: %d inclusions, %d exclusions' % (f, len(inc), len(exc)), flush=True)
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
NREL = {q: len(truths[q]) for q in qall}
# eligibility mask: conjunction of every derived constraint for the pair's family
ELIG = {}
for q in qall:
    g = GT[q]; f = fams[q]; ok = np.ones(len(g), bool)
    for cn in RULES[f]['include']:
        ok &= (MA[g, CIX[cn]] > 0)
    for cn in RULES[f]['exclude']:
        ok &= ~(MA[g, CIX[cn]] > 0)
    ELIG[q] = ok
ne = np.array([int(ELIG[q].sum()) for q in qall])
cov = []; ncov = 0
for q in qall:
    g = GT[q]; tr = rel[g] == 1
    cov.append(float(ELIG[q][tr].mean()) if tr.any() else 1.0)
    ncov += int(ELIG[q][tr].all())
print('', flush=True)
print('eligible candidates per pair (of 20): mean %.2f  min %d  p10 %.0f  p50 %.0f' % (
    ne.mean(), ne.min(), np.percentile(ne, 10), np.percentile(ne, 50)), flush=True)
print('truth hands inside the eligible set: %.4f of hands   %d of %d lists fully covered' % (
    float(np.mean(cov)), ncov, len(qall)), flush=True)
print('pairs with fewer than 5 eligible candidates: %d' % int((ne < 5).sum()), flush=True)


def decode(mask_q, mask_sel=None):
    out = np.zeros(len(pid))
    for q in qall:
        g = GT[q]; p = np.clip(pf[g], 0., 1.)
        o = np.lexsort((cr[g], PRI[g]))
        mq = mask_q[q][o]
        pq = np.where(mq, p[o], 0.0)
        w = np.empty(len(g)); w[o] = less_than_five(pq)
        s = 10.0 + p * w + 1e-9 * p
        if mask_sel is not None:
            s = np.where(mask_sel[q], s, s - 5.0)
        out[g] = s
    return out


def ap5(hits, q):
    h = 0; s = 0.
    for i, r in enumerate(hits, 1):
        if r:
            h += 1; s += h / i
    return s / min(5, NREL[q])


def score_of(sc):
    m = []
    for q in qall:
        g = GT[q]; s5 = g[np.argsort(-sc[g], kind='stable')[:5]]
        m.append(ap5(rel[s5], q))
    return np.array(m)


ALL = {q: np.ones(len(GT[q]), bool) for q in qall}
sc0 = decode(ALL)
_, mprod = summarize(key.with_columns(pl.Series('score', sc0)), truths, fams)
D0 = np.array([mprod[q] for q in qall])
assert np.allclose(D0, score_of(sc0)), 'GATE: AP@5 must match summarize'
print('', flush=True)
print('GATE ok. production decoded %.6f' % D0.mean(), flush=True)
fqa = np.array([fams[q] for q in qall]); hva = np.array([half(ptab[q]) for q in qall])
rngb = np.random.default_rng(272); pools = sorted(set(ptab.values()))
qbp = {pp: [i for i, q in enumerate(qall) if ptab[q] == pp] for pp in pools}
IDX = [np.concatenate([qbp[pools[i]] for i in rngb.choice(len(pools), len(pools), replace=True)]) for _ in range(4000)]
print('', flush=True)
print('%-32s %9s %11s %8s   %9s %9s   %9s %9s %9s' % (
    'arm', 'MAP@5', 'd_dec', 'P(>0)', 'h0', 'h1', 'DT', 'SP', 'CI'), flush=True)
res = {}


def report(nm, sc):
    m = score_of(sc); dl = m - D0
    bs = np.array([dl[ix].mean() for ix in IDX])
    res[nm] = {'map5': round(float(m.mean()), 6), 'd_dec': round(float(dl.mean()), 6),
               'P_gt_0': round(float((bs > 0).mean()), 4), 'h0': round(float(dl[hva == 0].mean()), 6),
               'h1': round(float(dl[hva == 1].mean()), 6), 'DT': round(float(dl[fqa == 'directed_transfer'].mean()), 6),
               'SP': round(float(dl[fqa == 'soft_play'].mean()), 6), 'CI': round(float(dl[fqa == CI].mean()), 6)}
    r = res[nm]
    print('%-32s %9.6f %+11.6f %8.4f   %+9.5f %+9.5f   %+9.5f %+9.5f %+9.5f' % (
        nm, r['map5'], r['d_dec'], r['P_gt_0'], r['h0'], r['h1'], r['DT'], r['SP'], r['CI']), flush=True)
    return res[nm]['d_dec']


report('production (control)', sc0)
report('QMASS', decode(ELIG))
report('QMASS+X', decode(ELIG, ELIG))
rng = np.random.default_rng(277); rnd_d = []
for it in range(20):
    RM = {}
    for q in qall:
        n = len(GT[q]); k = int(ELIG[q].sum())
        m = np.zeros(n, bool); m[rng.choice(n, k, replace=False)] = True
        RM[q] = m
    mm = score_of(decode(RM)); rnd_d.append(float((mm - D0).mean()))
rnd_d = np.array(rnd_d)
print('%-32s %9s %+11.6f %8s   size-matched random mask, 20 draws (sd %.6f, max %+.6f)' % (
    'CTRL_RND', '--', rnd_d.mean(), '--', rnd_d.std(), rnd_d.max()), flush=True)
res['CTRL_RND'] = {'d_dec_mean': round(float(rnd_d.mean()), 6), 'sd': round(float(rnd_d.std()), 6),
                   'max': round(float(rnd_d.max()), 6), 'draws': 20}
json.dump({'prod_dec': float(D0.mean()), 'eligible_mean': float(ne.mean()),
           'truth_coverage': float(np.mean(cov)), 'lists_fully_covered': ncov,
           'rules_n': {f: {k: len(v) for k, v in RULES[f].items()} for f in FAMS}, 'arms': res},
          open(OUT / 'gen_qmass.json', 'w'), indent=2)
print('', flush=True)
print('wrote gen_qmass.json', flush=True)
