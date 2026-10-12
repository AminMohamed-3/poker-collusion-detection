"""STEP245 prototype -- Amin's idea (2026-09-17): train evidence like masked / autoregressive prediction.
"Given this pair and some of its evidence hands, predict the others."  Development phase only, all 372 positives,
raw 163 per-hand features, 5 folds by table (STEP232 folds), LightGBM per family.  A PROTOTYPE on weak features: only
contrasts count (baseline = STEP233 within-pair classifier).

Arms (second-stage models; every arm sees the same own-features + the base score):
  STACK     control: [own features, chrono, base p]                                  (what stacking alone gives)
  ORACLE_C  + CONTENT context from the pair's TRUE other listed hands (bound; no time relations)
  ORACLE_CT + content and TIME relations to the true other listed hands (bound incl. the ordering rule)
  SELF2     + content context from the base model's own top-2 picks (excluding the candidate)    <- buildable
  SELF2_T   + content and time relations to those picks                                          <- buildable
Masking: for the ORACLE arms each training row is drawn 3 times with a random subset of the other listed hands
(sizes 1..n-1) as context; at test the context is the other listed hands with ONE removed at random for unlisted
candidates, so the context size never reveals the label.
Output: experiments/step245_masked_context_20260917/proto245.json
"""
import json
import os
os.environ['POLARS_MAX_THREADS'] = '1'
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
OUT = CV / 'experiments' / 'step245_masked_context_20260917'
OUT.mkdir(parents=True, exist_ok=True)
oof = pl.read_parquet(CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet').filter(pl.col('label') == 1)
h = pl.read_parquet(str(ROOT / 'outputs_v2' / 'hands_development' / 'batch_*.parquet'))
t = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'started_at', 'phase'])
d = oof.join(h.drop('table_id'), on=['pair_id', 'hand_id']).join(t, on='hand_id', how='left')
assert set(d['phase'].unique().to_list()) == {'development'}
d = d.with_columns(ts=pl.col('started_at').dt.epoch('s')).drop('started_at', 'phase').sort('pair_id', 'ts')
d = d.with_columns((pl.col('ts').rank('ordinal').over('pair_id') - 1).alias('chrono_rank'), pl.len().over('pair_id').alias('n_hands')).with_columns(
    (pl.col('chrono_rank') / pl.col('n_hands')).alias('chrono_frac'))
feats = [c for c in h.columns if c not in ('pair_id', 'hand_id', 'table_id')]
X = d.select(feats).to_numpy().astype(np.float32)
Xc = np.column_stack([X, d['chrono_rank'].to_numpy(), d['chrono_frac'].to_numpy(), d['n_hands'].to_numpy()]).astype(np.float32)
y = d['listed'].to_numpy().astype(int); fold = d['fold'].to_numpy(); fam = d['behavior_family'].to_numpy()
pid = d['pair_id'].to_numpy(); cr = d['chrono_rank'].to_numpy().astype(float); nh = d['n_hands'].to_numpy().astype(float)
mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
Z = np.clip(np.nan_to_num((X - mu) / sd), -5, 5).astype(np.float32)
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
P = dict(n_estimators=300, num_leaves=15, learning_rate=.03, min_child_samples=20, subsample=.8, subsample_freq=1,
         colsample_bytree=.5, reg_lambda=5.0, n_jobs=1, verbosity=-1, random_state=245)
# ---- base model (as STEP233) + per (family, fold) top-25 features by gain
base = np.zeros(len(y)); TOP = {}
for f in FAMS:
    for k in range(5):
        tr = (fold != k) & (fam == f); te = (fold == k) & (fam == f)
        m = lgb.LGBMClassifier(**P).fit(Xc[tr], y[tr])
        base[te] = m.predict_proba(Xc[te])[:, 1]
        g = m.booster_.feature_importance('gain')[:len(feats)]
        TOP[(f, k)] = np.argsort(-g)[:25]
# base score for TRAINING rows must also be out-of-fold: it is (every row's base comes from the fold that excludes its table)
start = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1]; end = np.r_[start[1:], len(pid)]
rng = np.random.default_rng(245)


def ctx_features(i, S, top, with_time):
    """content (and optionally time) relations of candidate row i to context rows S (global row indices)."""
    if len(S) == 0:
        c = [0., 0.] + [0.] * (2 * len(top)) + [0.]
        return c + ([0., 0., 0.] if with_time else [])
    m = Z[S].mean(0)
    zi = Z[i]
    cos = float(zi @ m / (np.linalg.norm(zi) * np.linalg.norm(m) + 1e-9))
    dist = float(np.sqrt(((zi - m) ** 2).mean()))
    c = [cos, dist] + (zi[top] - m[top]).tolist() + m[top].tolist() + [float(len(S))]
    if with_time:
        c += [float((cr[i] - cr[S].min()) / nh[i]), float((cr[i] - cr[S].max()) / nh[i]), float((cr[S] < cr[i]).sum())]
    return c


def build(arm, rows_mask, train, top):
    """returns (row index array, context feature matrix) for the rows in rows_mask."""
    idx, C = [], []
    with_time = arm.endswith('T')
    for a, b in zip(start, end):
        if not rows_mask[a]:
            continue
        rows = np.arange(a, b)
        L = rows[y[rows] == 1]
        if arm.startswith('ORACLE'):
            draws = 3 if train else 1
            for i in rows:
                others = L[L != i]
                for _ in range(draws):
                    if train:
                        msz = rng.integers(1, max(len(L) - 1, 1) + 1)
                        S = rng.choice(others, size=min(msz, len(others)), replace=False)
                    else:
                        S = others if y[i] == 1 else rng.choice(L, size=max(len(L) - 1, 1), replace=False)
                    idx.append(i); C.append(ctx_features(i, S, top, with_time))
        else:
            order = rows[np.argsort(-base[rows], kind='stable')]
            for i in rows:
                S = order[order != i][:2]
                idx.append(i); C.append(ctx_features(i, S, top, with_time))
    return np.array(idx), np.array(C, dtype=np.float32)


def ap5(s, rel):
    o = np.argsort(-s, kind='stable')[:5]; hits = 0; v = 0.0
    for r, j in enumerate(o):
        if rel[j]:
            hits += 1; v += hits / (r + 1)
    return v / min(int(rel.sum()), 5)


def decode_time(p, time):
    pr = np.argsort(time, kind='stable'); dist = np.zeros(6); dist[0] = 1.0; out = np.zeros(len(p))
    for i in pr:
        out[i] = p[i] * dist[:5].sum(); new = dist * (1 - p[i]); new[1:] += dist[:-1] * p[i]; new[5] += dist[5] * p[i]; dist = new
    return out


def score(pred):
    v = {k: [] for k in ['all'] + FAMS}; w = {k: [] for k in ['all'] + FAMS}
    for a, b in zip(start, end):
        rel = y[a:b].astype(bool)
        s = ap5(pred[a:b], rel); s2 = ap5(decode_time(np.clip(pred[a:b], 0, 1), cr[a:b]), rel)
        v['all'].append(s); v[fam[a]].append(s); w['all'].append(s2); w[fam[a]].append(s2)
    return {'sorted': {k: round(float(np.mean(x)), 4) for k, x in v.items()}, 'time_decoded': {k: round(float(np.mean(x)), 4) for k, x in w.items()},
            '_per_pair_sorted': v['all']}


res = {'BASE': score(base)}
print('BASE', res['BASE']['sorted'], res['BASE']['time_decoded'], flush=True)
own = np.column_stack([Xc, base]).astype(np.float32)
for arm in ['STACK', 'ORACLE_C', 'ORACLE_CT', 'SELF2', 'SELF2_T']:
    pred = np.zeros(len(y))
    for f in FAMS:
        for k in range(5):
            trm = (fold != k) & (fam == f); tem = (fold == k) & (fam == f)
            if arm == 'STACK':
                m = lgb.LGBMClassifier(**P).fit(own[trm], y[trm]); pred[tem] = m.predict_proba(own[tem])[:, 1]
                continue
            top = TOP[(f, k)]   # one column layout per second-stage model; from a base model that never saw fold k
            itr, Ctr = build(arm, trm, True, top); ite, Cte = build(arm, tem, False, top)
            m = lgb.LGBMClassifier(**P).fit(np.column_stack([own[itr], Ctr]), y[itr])
            pred[ite] = m.predict_proba(np.column_stack([own[ite], Cte]))[:, 1]
    res[arm] = score(pred)
    print(arm, res[arm]['sorted'], res[arm]['time_decoded'], flush=True)
# paired bootstrap over pairs against STACK (sorted MAP@5)
bs = {}
ref = np.array(res['STACK']['_per_pair_sorted'])
for arm in ['ORACLE_C', 'ORACLE_CT', 'SELF2', 'SELF2_T']:
    dl = np.array(res[arm]['_per_pair_sorted']) - ref
    b = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(4000)]
    bs[arm] = {'delta_vs_STACK': round(float(dl.mean()), 4), 'boot90': [round(float(np.quantile(b, .05)), 4), round(float(np.quantile(b, .95)), 4)]}
for a in res:
    res[a].pop('_per_pair_sorted')
res['paired_vs_STACK_sorted'] = bs
(OUT / 'proto245.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print(json.dumps(bs, indent=1))
