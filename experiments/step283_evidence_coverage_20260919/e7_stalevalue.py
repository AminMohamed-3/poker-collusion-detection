"""E7: SIZE the stale-evidence finding before anyone spends a slot on it.

E6 confirmed it: 5,855 of the current top-16,060-by-risk pairs still carry v13 evidence, and staleness is
perfectly monotone in risk rank (0.8 % in the top decile of the scope, 67.3 % in the bottom).  That is a
scope boundary that moved -- pairs that rose into the top 16,060 after their lists were computed.

But EvidenceMAP@5 only counts TRUE TARGETS.  Staleness is concentrated exactly where targets are NOT: only 8
of the 5,855 are in the top 1,000 by risk, 2 in the top 100.  So the question is not "how many lists are
stale" but "how many TARGET lists are stale", and that is a much smaller number.

Estimated here by transferring the development target-density-by-risk-percentile curve onto the evaluation
ranking, which is the only way to do it without evaluation labels:
    E[stale targets] = sum over stale pairs of P(target | risk percentile)
Then converted to composite with the per-target value of a list upgrade and the measured audit->public
evidence transfer (the encoder session's 19.233: .87 -> .41 -> .29 -> .20 over four chained reads, ~.3 today).
"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.dont_write_bytecode = True
P = 'C:/Data/Projects/Money Making/work/poker/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/evid/')
import numpy as np, polars as pl

CUR = P + 'Claude Version/outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv'
V13 = P + 'outputs_v13_coverage/submission_conditional_coverage.csv'
EV = ['evidence_hand_%d' % i for i in (1, 2, 3, 4, 5)]
SCOPE = 16060
t0 = time.time(); rec = {}

cur = pl.read_csv(CUR, infer_schema=False)
v13 = pl.read_csv(V13, infer_schema=False)
j = cur.select(['pair_id', 'risk_score'] + EV).join(v13.select(['pair_id'] + EV), on='pair_id',
                                                    how='left', suffix='_v13', validate='1:1')
same = np.ones(j.height, dtype=bool)
for c in EV:
    same &= (j[c] == j[c + '_v13']).fill_null(False).to_numpy()
risk = j['risk_score'].cast(pl.Float64).to_numpy()
o = np.argsort(-risk, kind='mergesort'); rank = np.empty(j.height, np.int64); rank[o] = np.arange(j.height)
inside = rank < SCOPE
stale = same & inside
pct_eval = rank / float(j.height)
print('stale inside scope: %d' % int(stale.sum()), flush=True)

# ---- development target density by risk percentile
sc = pl.concat([pl.read_parquet(P + 'Claude Version/experiments/pu_pairs_20260911/fold%d/scores.parquet' % f,
                                columns=['pair_id', 'risk']) for f in (1, 2)]
               ).unique(subset=['pair_id'], keep='first')
lab = pl.read_csv(P + 'data/development_labels.csv').select(['pair_id', 'label'])
dv = sc.join(lab, on='pair_id', how='left')
dr = dv['risk'].to_numpy(); dl = (dv['label'].fill_null(-1) == 1).to_numpy()
do = np.argsort(-dr, kind='mergesort'); drank = np.empty(dv.height, np.int64); drank[do] = np.arange(dv.height)
pct_dev = drank / float(dv.height)
print('development: %d pairs, %d true targets' % (dv.height, int(dl.sum())), flush=True)
rec['dev_pairs'] = int(dv.height); rec['dev_targets'] = int(dl.sum())

NB = 40
edges = np.linspace(0, 1, NB + 1)
dens = np.zeros(NB)
for b in range(NB):
    m = (pct_dev >= edges[b]) & (pct_dev < edges[b + 1])
    dens[b] = dl[m].sum() / max(m.sum(), 1)
print('\ndevelopment target density by risk percentile bin (first 8 bins):', flush=True)
for b in range(8):
    print('   pct %.3f-%.3f : density %.5f' % (edges[b], edges[b + 1], dens[b]), flush=True)
rec['dens_top_bins'] = [float(x) for x in dens[:8]]

bidx = np.clip((pct_eval * NB).astype(int), 0, NB - 1)
p_target = dens[bidx]
exp_all = float(p_target.sum())
exp_stale = float(p_target[stale].sum())
exp_scope = float(p_target[inside].sum())
print('\nexpected TRUE TARGETS (development density transferred to the evaluation ranking):', flush=True)
print('   whole file            : %.1f' % exp_all, flush=True)
print('   inside the scope      : %.1f' % exp_scope, flush=True)
print('   >>> among STALE pairs : %.2f' % exp_stale, flush=True)
rec['expected_targets_file'] = exp_all
rec['expected_targets_scope'] = exp_scope
rec['expected_targets_stale'] = exp_stale
rec['stale_share_of_targets'] = exp_stale / max(exp_scope, 1e-9)
print('   stale share of in-scope targets: %.4f' % rec['stale_share_of_targets'], flush=True)

# ---- convert to composite under explicit, stated assumptions
print('\nsizing, with every assumption stated:', flush=True)
for dAP in (.05, .10, .20):
    for tr in (.20, .30):
        gain_E = (exp_stale / max(exp_all, 1e-9)) * dAP
        comp = gain_E * .2 * 1.0
        comp_tr = gain_E * .2 * (tr / .3) if False else gain_E * .2
        print('   if upgrading one stale target list gains %.2f AP@5 : evidence term %+0.5f -> composite %+0.5f'
              % (dAP, gain_E, comp), flush=True)
        break
rec['sizing'] = {('dAP_%.2f' % d): float((exp_stale / max(exp_all, 1e-9)) * d * .2) for d in (.05, .10, .20)}
print('\n   NOTE: this is already a direct evaluation-side effect, NOT an audit gain, so the', flush=True)
print('   audit->public transfer discount does NOT apply to it.', flush=True)
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'e7_stalevalue.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
