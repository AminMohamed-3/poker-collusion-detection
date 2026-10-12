"""STEP276-AO: cross-pair structure at a shared table -- the one NEW-INFORMATION lever left locally.

Every model in the evidence stack scores a (pair, hand) row using that pair's own history.  Nothing in stage 1,
stage 2, the event fusion or the decoder knows that OTHER pairs sit at the same table and that the generator
also cited hands for them.  A hand is a table-level object, so the same hand_id appears in several pairs'
candidate sets.  If citation clusters on hands -- the table's genuinely anomalous hands get cited for whoever
was involved -- then "how many other pairs at this table cite this hand" is information the stack has never had.

Order of business, cheapest decisive test first:
  0  does the structure even exist?  tables with >= 2 labelled pairs, and shared candidate hands between them
  1  ORACLE feature: n_other_pairs_citing(h), using the OTHER pairs' TRUE labels.  If the oracle carries no
     within-five and no within-top-20 signal, the lane is dead and no deployable version can rescue it.
  2  only if the oracle bites: the DEPLOYABLE version, built from production's own predicted scores for the
     other pairs, which is what would exist at test time.

The oracle is deliberately unfair -- it is an upper bound.  A dead upper bound closes the lane in one run.
"""
import os, sys, json
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
K = S165.K; CI = 'coordinated_isolation'
D = S165.load(); d = D['d']; QIX = D['QIX']; top = D['top']
pid = d['pair_id'].to_numpy(); hid = d['hand_id'].to_numpy(); tid = d['table_id'].to_numpy()
rel = d['relevance'].to_numpy().astype(int)
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
famall = np.array([fams[p] for p in pid])

# ---- 0. does the structure exist at all?
pt = d.select('pair_id', 'table_id').unique()
per_tab = pt.group_by('table_id').len().sort('len', descending=True)
print('labelled pairs %d over %d tables' % (pt.height, per_tab.height), flush=True)
print('pairs per table: %s' % dict(zip(*np.unique(per_tab['len'].to_numpy(), return_counts=True)[::1])), flush=True)
multi = set(per_tab.filter(pl.col('len') >= 2)['table_id'].to_list())
print('tables carrying >= 2 labelled pairs: %d   (pairs on them: %d)' % (
    len(multi), int(pt.filter(pl.col('table_id').is_in(list(multi))).height)), flush=True)
if not multi:
    print('', flush=True); print('LANE VERDICT: no table carries two labelled pairs -- cross-pair structure is unobservable. CLOSED.', flush=True)
    json.dump({'tables_with_2plus_pairs': 0, 'verdict': 'CLOSED - no shared tables'}, open(OUT / 'crosspair.json', 'w'), indent=2)
    sys.exit(0)

# cited hands per pair, and the shared-hand structure
cited = {}
for kk, gg in d.filter(pl.col('relevance') == 1).group_by('pair_id'):
    cited[kk[0]] = set(gg['hand_id'].to_list())
cands = {}
for kk, gg in d.group_by('pair_id'):
    cands[kk[0]] = set(gg['hand_id'].to_list())
tab_pairs = {}
for p, t in pt.iter_rows():
    tab_pairs.setdefault(t, []).append(p)
ov_obs = []; ov_exp = []; nshare = []
for t in multi:
    ps = tab_pairs[t]
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, b = ps[i], ps[j]
            sh = cands[a] & cands[b]
            if not sh:
                continue
            nshare.append(len(sh))
            ca = cited.get(a, set()) & sh; cb = cited.get(b, set()) & sh
            if not ca or not cb:
                continue
            ov_obs.append(len(ca & cb))
            ov_exp.append(len(ca) * len(cb) / len(sh))
print('', flush=True)
print('pair-pairs sharing >= 1 candidate hand: %d   median shared hands %.0f' % (
    len(nshare), np.median(nshare) if nshare else 0), flush=True)
if ov_obs:
    print('cited-hand overlap on shared hands: observed %.4f per pair-pair, chance %.4f  (ratio %.2fx, n=%d)' % (
        np.mean(ov_obs), np.mean(ov_exp), np.mean(ov_obs) / max(np.mean(ov_exp), 1e-9), len(ov_obs)), flush=True)
else:
    print('no pair-pair has cited hands on both sides of a shared candidate set', flush=True)

# ---- 1. ORACLE feature and its discrimination
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
# n other LABELLED pairs at this table citing this hand (excluding the row's own pair)
cited_by_tab_hand = {}
for p, hs in cited.items():
    t = d.filter(pl.col('pair_id') == p)['table_id'][0]
    for h in hs:
        cited_by_tab_hand.setdefault((t, h), set()).add(p)
ORC = np.array([len(cited_by_tab_hand.get((tid[i], hid[i]), set()) - {pid[i]}) for i in range(len(pid))], dtype=float)
print('', flush=True)
print('ORACLE feature n_other_pairs_citing: nonzero on %d of %d rows (%.2f%%), max %d' % (
    int((ORC > 0).sum()), len(ORC), 100 * (ORC > 0).mean(), int(ORC.max())), flush=True)
print('   on the 3,240 top-20 candidate rows: nonzero %d   on the 810 selected rows: nonzero %d' % (
    int((ORC[top] > 0).sum()), int(sum((ORC[SEL[q]] > 0).sum() for q in qall))), flush=True)
res = {'tables_with_2plus_pairs': len(multi), 'pair_pairs_sharing_candidates': len(nshare),
       'overlap_observed': round(float(np.mean(ov_obs)), 4) if ov_obs else None,
       'overlap_chance': round(float(np.mean(ov_exp)), 4) if ov_obs else None,
       'oracle_nonzero_rows': int((ORC > 0).sum()), 'oracle_nonzero_top20': int((ORC[top] > 0).sum())}
if (ORC[top] > 0).sum() == 0:
    print('', flush=True)
    print('LANE VERDICT: the oracle cross-pair feature is IDENTICALLY ZERO on every candidate row.', flush=True)
    print('No hand in any pair\'s top-20 is cited for another labelled pair at the same table. CLOSED.', flush=True)
    res['verdict'] = 'CLOSED - oracle feature identically zero on candidates'
    json.dump(res, open(OUT / 'crosspair.json', 'w'), indent=2)
    sys.exit(0)
# within-top-20 and within-five AUC of the oracle feature
def auc_groups(sig, groups):
    num = 0.; den = 0.
    for g in groups:
        x = sig[g].astype(float); y = rel[g]
        pos = x[y == 1]; neg = x[y == 0]
        if not len(pos) or not len(neg):
            continue
        num += (pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()
        den += len(pos) * len(neg)
    return num / den if den else float('nan'), den
a20, n20 = auc_groups(ORC, [GT[q] for q in qall])
a5, n5 = auc_groups(ORC, [SEL[q] for q in qall])
p20, _ = auc_groups(pf, [GT[q] for q in qall])
print('', flush=True)
print('   within-top-20 AUC:  oracle cross-pair %.4f   (production p %.4f)   contrast pairs %d' % (a20, p20, n20), flush=True)
print('   within-five   AUC:  oracle cross-pair %.4f                          contrast pairs %d' % (a5, n5), flush=True)
res.update({'oracle_auc_top20': round(float(a20), 4), 'prod_p_auc_top20': round(float(p20), 4),
            'oracle_auc_within5': round(float(a5), 4)})
json.dump(res, open(OUT / 'crosspair.json', 'w'), indent=2)
print('', flush=True)
print('wrote crosspair.json', flush=True)
