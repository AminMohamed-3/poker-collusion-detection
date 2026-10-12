"""STEP233 prototype (development phase only; training lists of the 372 positives, cross-fitted by table; no held-fold
evidence read beyond what the audit protocol already uses; raw 163 per-hand features only).
Evidence as: P(listed) = P(event | hand) x P(fewer than five events precede it in (type, time) order), with P(event)
from the STEP232 absolute classifier (listed hands vs all hands of confirmed non-target pairs: clean negatives).
Baseline on the same features: within-pair listed-vs-unlisted classifier (+ chronology), sorted and decoded."""
import os, json
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np, polars as pl, lightgbm as lgb
ROOT = r'C:\Data\Projects\Money Making\work\poker'
EXP = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917'
oof = pl.read_parquet(EXP + r'\hand_oof.parquet').filter(pl.col('label') == 1)
h = pl.read_parquet(ROOT + r'\outputs_v2\hands_development\batch_*.parquet')
t = pl.read_parquet(ROOT + r'\data\hands.parquet', columns=['hand_id', 'started_at', 'phase']).filter(pl.col('phase') == 'development')
d = oof.join(h.drop('table_id'), on=['pair_id', 'hand_id']).join(t.select('hand_id', 'started_at'), on='hand_id')
ev = pl.read_csv(ROOT + r'\data\development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
d = d.join(ev, on=['pair_id', 'hand_id'], how='left').sort('pair_id', 'started_at')
d = d.with_columns((pl.col('started_at').rank('ordinal').over('pair_id') - 1).alias('chrono_rank'), pl.len().over('pair_id').alias('n_hands')).with_columns((pl.col('chrono_rank') / pl.col('n_hands')).alias('chrono_frac'))
feats = [c for c in h.columns if c not in ('pair_id', 'hand_id', 'table_id')]
X = d.select(feats).to_numpy().astype(np.float32); Xc = np.column_stack([X, d['chrono_rank'].to_numpy(), d['chrono_frac'].to_numpy(), d['n_hands'].to_numpy()]).astype(np.float32)
EC = ['p_any', 'p_directed_transfer', 'p_soft_play', 'p_coordinated_isolation', 'p_fmax']
d = d.with_columns([pl.col(c).rank('ordinal', descending=True).over('pair_id').alias(c + '_rk') for c in EC] + [(pl.col(c) / pl.col(c).max().over('pair_id')).alias(c + '_rel') for c in EC] + [(pl.col(c) > .4662).sum().over('pair_id').alias(c + '_nsharp') for c in ('p_fmax',)])
ECX = d.select(EC + [c + '_rk' for c in EC] + [c + '_rel' for c in EC] + ['p_fmax_nsharp']).to_numpy().astype(np.float32)
Xe = np.column_stack([Xc, ECX]).astype(np.float32)
y = d['listed'].to_numpy(); fold = d['fold'].to_numpy(); fam = d['behavior_family'].to_numpy(); pid = d['pair_id'].to_numpy()
# type labels from one-descent lists: listed hands before the time descent are block 1
lst = d.filter(pl.col('listed') == 1).sort('pair_id', 'evidence_rank').with_columns((pl.col('started_at') < pl.col('started_at').shift(1).over('pair_id')).fill_null(False).alias('desc')).with_columns(pl.col('desc').cum_sum().over('pair_id').alias('block'), pl.col('desc').sum().over('pair_id').alias('n_desc'))
tl = lst.filter(pl.col('n_desc') == 1).select('pair_id', 'hand_id', (pl.col('block') == 0).cast(pl.Int8).alias('is_b1'))
d = d.join(tl, on=['pair_id', 'hand_id'], how='left'); isb1 = d['is_b1'].to_numpy()
P = dict(n_estimators=300, num_leaves=15, learning_rate=.03, min_child_samples=20, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=5.0, n_jobs=2, verbosity=-1, random_state=233)
p_within = np.zeros(len(y)); p_withe = np.zeros(len(y)); p_type = np.full(len(y), .5)
for f_ in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
    for k in range(5):
        tr = (fold != k) & (fam == f_); te = (fold == k) & (fam == f_)
        if te.sum() == 0: continue
        p_within[te] = lgb.LGBMClassifier(**P).fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]
        p_withe[te] = lgb.LGBMClassifier(**P).fit(Xe[tr], y[tr]).predict_proba(Xe[te])[:, 1]
        trt = tr & ~np.isnan(isb1.astype(float))
        if f_ != 'coordinated_isolation' and trt.sum() > 50: p_type[te] = lgb.LGBMClassifier(**{**P, 'n_estimators': 150}).fit(X[trt], isb1[trt].astype(int)).predict_proba(X[te])[:, 1]
d = d.with_columns(pl.Series('p_within', p_within), pl.Series('p_withe', p_withe), pl.Series('p_type1', p_type))
def ap5(order_scores, rel):
    o = np.argsort(-order_scores, kind='stable')[:5]; hits = 0; s = 0.0
    for i, j in enumerate(o):
        if rel[j]: hits += 1; s += hits / (i + 1)
    return s / min(int(rel.sum()), 5)
def decode(p, t1, time, a=1.0):
    # priority: type-1 block first, then time; P(listed) = p * P(fewer than 5 events before)
    pr = np.lexsort((time, -(t1 > .5).astype(int))); q = np.clip(p[pr] * a, 0, 1); dist = np.zeros(6); dist[0] = 1.0; out = np.zeros(len(p))
    for idx, qi in zip(pr, q):
        out[idx] = qi * dist[:5].sum(); new = np.zeros(6); new[0] = dist[0] * (1 - qi)
        for c in range(1, 6): new[c] = dist[c] * (1 - qi) + dist[c - 1] * qi
        new[5] += dist[5] * qi; dist = new
    return out
res = {}
groups = d.partition_by('pair_id', maintain_order=True)
def run(name, fn):
    v = {'all': [], 'directed_transfer': [], 'soft_play': [], 'coordinated_isolation': []}
    for g in groups:
        a = fn(g); rel = g['listed'].to_numpy().astype(bool); s = ap5(a, rel); v['all'].append(s); v[g['behavior_family'][0]].append(s)
    res[name] = {k: round(float(np.mean(x)), 4) for k, x in v.items()}; print(name, res[name])
def prior(p, f): o = p / (1 - p) * f; return o / (1 + o)
famcol = {'directed_transfer': 'p_directed_transfer', 'soft_play': 'p_soft_play', 'coordinated_isolation': 'p_coordinated_isolation'}
run('within_sorted', lambda g: g['p_within'].to_numpy())
run('within_decoded', lambda g: decode(g['p_within'].to_numpy(), g['p_type1'].to_numpy(), g['started_at'].to_numpy().astype('int64')))
run('within+eventcols_sorted', lambda g: g['p_withe'].to_numpy())
run('within+eventcols_decoded', lambda g: decode(g['p_withe'].to_numpy(), g['p_type1'].to_numpy(), g['started_at'].to_numpy().astype('int64')))
run('event_sorted (no time info)', lambda g: g[famcol[g['behavior_family'][0]]].to_numpy())
for f in (10,):
    run(f'event_decoded prior x{f}', lambda g, f=f: decode(prior(np.clip(g[famcol[g['behavior_family'][0]]].to_numpy(), 1e-6, 1 - 1e-6), f), g['p_type1'].to_numpy(), g['started_at'].to_numpy().astype('int64')))
for f in ():
    run(f'event x within geometric, decoded prior x{f}', lambda g, f=f: decode(np.sqrt(prior(np.clip(g[famcol[g['behavior_family'][0]]].to_numpy(), 1e-6, 1 - 1e-6), f) * g['p_within'].to_numpy()), g['p_type1'].to_numpy(), g['started_at'].to_numpy().astype('int64')))
a_ = []; b_ = []
for g in groups:
    rel = g['listed'].to_numpy().astype(bool); tt = g['started_at'].to_numpy().astype('int64'); t1 = g['p_type1'].to_numpy()
    a_.append(ap5(decode(g['p_within'].to_numpy(), t1, tt), rel)); b_.append(ap5(decode(g['p_withe'].to_numpy(), t1, tt), rel))
a_ = np.array(a_); b_ = np.array(b_); rng = np.random.default_rng(7); dd = b_ - a_; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
famv = np.array([g['behavior_family'][0] for g in groups])
res['paired decoded delta (eventcols - within)'] = {'mean': round(float(dd.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)], 'up': int((dd > 0).sum()), 'down': int((dd < 0).sum()), 'by_family': {f: round(float(dd[famv == f].mean()), 4) for f in np.unique(famv)}}
print(res['paired decoded delta (eventcols - within)'])
json.dump(res, open(EXP + r'\step233b_proto.json', 'w', encoding='utf-8'), indent=1); print('DONE')
