"""B1: does the BEHAVIOUR term have a reachable lever, and is `none` withholding it?

The metric (OBS reconstruction, src/evaluate_conditional_behavior.py):
    BehaviorMAP = mean over k in {DT,SP,CI} of AP(truth == k, risk * 1[assigned == k])
so a pair assigned to k that is not truly k is a false positive above every true k below it, and each
family contributes 1/3 regardless of size.  Production assigns ALL 112,540 pairs by
"conditional model argmax EXCLUDING class zero" (assemble_conditional_behavior.py scope string), i.e. the
`none` option that validation.py permits is thrown away by an unfitted rule.

Instrument: data/development_labels.csv -- 372 confirmed targets with families (DT 148, SP 132, CI 92) and
1,488 confirmed non-targets, REAL annotator labels, entirely independent of the certified block testbed that
just failed on the STEP275 read.  Features: crossphase_20260911/lab_on_dev.parquet, 653 usable columns that
are a strict subset of the evaluation frame, so any rule found here is buildable.

Measured, all cross-fold by TABLE so nothing is in-sample:
  ORACLE   truth-based withholding -- the ceiling
  PROD     argmax over the three families (the production rule)
  FREE     argmax over four classes, class 0 -> withheld
  THRESH   withhold when p0 > t, swept
plus a permuted-label null for every arm.
"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.dont_write_bytecode = True
P = 'C:/Data/Projects/Money Making/work/poker/'
C = P + 'Claude Version/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/behav/')
import numpy as np, polars as pl
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
t0 = time.time(); rec = {}

lab = pl.read_csv(P + 'data/development_labels.csv')
feat = pl.read_parquet(C + 'experiments/crossphase_20260911/lab_on_dev.parquet')
sc = pl.concat([pl.read_parquet(C + 'experiments/pu_pairs_20260911/fold%d/scores.parquet' % f,
                                columns=['pair_id', 'table_id', 'risk', 'shared_hands'])
                for f in (1, 2)]).unique(subset=['pair_id'], keep='first')
d = lab.join(feat, on='pair_id', how='inner').join(sc.drop('table_id'), on='pair_id', how='left')
print('joined %d rows (labels %d, features %d)' % (d.height, lab.height, feat.height), flush=True)
d = d.filter(pl.col('risk').is_not_null())
print('with risk: %d' % d.height, flush=True)

y = np.zeros(d.height, dtype=int)
for i, f in enumerate(FAMS):
    y[(d['behavior_family'] == f).to_numpy()] = i + 1
print('class counts (0=none):', {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}, flush=True)
risk = d['risk'].to_numpy()
tab = d['table_id'].to_numpy()
DROP = {'pair_id', 'table_id', 'player_1', 'player_2', 'label', 'label_status', 'behavior_family', 'risk'}
FCOLS = [c for c in d.columns if c not in DROP and d[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
X = d.select(FCOLS).to_numpy().astype(np.float64)
X[~np.isfinite(X)] = -999.
print('features: %d  tables: %d' % (X.shape[1], len(np.unique(tab))), flush=True)
rec['n'] = int(d.height); rec['n_feat'] = int(X.shape[1])
rec['classes'] = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}


def ap(yt, s):
    o = np.argsort(-s, kind='mergesort'); yt = np.asarray(yt)[o]
    if yt.sum() == 0:
        return 0.0
    c = np.cumsum(yt)
    return float((c / (np.arange(len(yt)) + 1.0) * yt).sum() / yt.sum())


def bmap(assigned, yy=None, rr=None):
    """assigned: 0 = withheld, 1..3 = family.  Exactly the metric above."""
    yy = y if yy is None else yy
    rr = risk if rr is None else rr
    return float(np.mean([ap((yy == k).astype(float), rr * (assigned == k)) for k in (1, 2, 3)]))


def oof_proba(target, seed=7, folds=5):
    out = np.zeros((len(target), 4))
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, target, groups=tab):
        m = LGBMClassifier(objective='multiclass', num_class=4, n_estimators=300, num_leaves=15,
                           learning_rate=.05, min_child_samples=20, colsample_bytree=.5, subsample=.8,
                           subsample_freq=1, reg_lambda=10, n_jobs=4, verbose=-1, random_state=seed)
        m.fit(X[tr], target[tr])
        pr = m.predict_proba(X[te])
        for j, cl in enumerate(m.classes_):
            out[te, int(cl)] = pr[:, j]
    return out


print('\nfitting 4-class head, 5-fold by table ...', flush=True)
Pm = oof_proba(y)
rec['head_acc_on_targets'] = float((Pm[y > 0].argmax(1) == y[y > 0]).mean())
print('  argmax-over-4 accuracy on true targets: %.4f' % rec['head_acc_on_targets'], flush=True)

prod = Pm[:, 1:].argmax(1) + 1                      # production rule: forced into a family
free = Pm.argmax(1)                                 # class 0 allowed
oracle = np.where(y > 0, prod, 0)                   # truth-based withholding
base = bmap(prod)
res = {'PROD': base, 'FREE': bmap(free), 'ORACLE': bmap(oracle)}
print('\n%-14s %10s %10s' % ('arm', 'BehavMAP', 'delta'), flush=True)
for k in ('PROD', 'ORACLE', 'FREE'):
    print('%-14s %10.5f %+10.5f' % (k, res[k], res[k] - base), flush=True)

print('\nthreshold sweep on p0 (withhold when p0 > t):', flush=True)
sweep = {}
for t in (.5, .7, .8, .9, .95, .98, .99, .995, .999):
    a = np.where(Pm[:, 0] > t, 0, prod)
    v = bmap(a)
    sweep['%g' % t] = {'bmap': v, 'delta': v - base, 'withheld': int((a == 0).sum()),
                       'withheld_true_targets': int(((a == 0) & (y > 0)).sum())}
    s = sweep['%g' % t]
    print('  t=%-6g  BehavMAP %.5f  %+.5f   withheld %5d  of which TRUE targets %3d'
          % (t, s['bmap'], s['delta'], s['withheld'], s['withheld_true_targets']), flush=True)
rec['arms'] = res; rec['sweep'] = sweep; rec['base'] = base

print('\npermuted-label null (10 reps, same pipeline, best arm each rep):', flush=True)
rng = np.random.default_rng(4242)
nb = []
for r in range(10):
    yp = y[rng.permutation(len(y))]
    Pp = oof_proba(yp)
    pp = Pp[:, 1:].argmax(1) + 1
    b0 = bmap(pp, yy=yp)
    best = max(bmap(np.where(Pp[:, 0] > t, 0, pp), yy=yp) - b0 for t in (.5, .8, .9, .95, .99))
    nb.append(max(best, bmap(Pp.argmax(1), yy=yp) - b0))
    print('  rep %d best null delta %+.5f (%.0f s)' % (r + 1, nb[-1], time.time() - t0), flush=True)
nb = np.array(nb)
rec['null'] = {'mean': float(nb.mean()), 'q95': float(np.quantile(nb, .95)), 'max': float(nb.max())}
print('\nnull over arms: mean %+.5f  q95 %+.5f  max %+.5f'
      % (nb.mean(), np.quantile(nb, .95), nb.max()), flush=True)
rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'b1_ceiling.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
