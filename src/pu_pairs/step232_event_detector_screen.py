"""STEP232 screen (development phase, development labels; no held evidence truth beyond the training lists, no evaluation rows).
An ABSOLUTE injected-event classifier: listed hands of positive pairs (1) against every hand of the confirmed
non-target pairs (0, clean negatives: an honest pair has no injected event). Unlisted hands of positive pairs are
left out of training (they mix natural hands with cap-censored events). 5-fold cross-fit by table. Pair pooling of
the out-of-fold per-hand log-odds gives a dose-aware pair score. Questions: (1) how sharp is one event against
honest look-alikes, (2) where do the low-dose positives (3-4 listed hands) land among honest pairs."""
import os, json, glob
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = r'C:\Data\Projects\Money Making\work\poker'
OUT = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917'
h = pl.read_parquet(ROOT + r'\outputs_v2\hands_development\batch_*.parquet')
lab = pl.read_csv(ROOT + r'\data\development_labels.csv').select('pair_id', 'label', 'behavior_family')
ev = pl.read_csv(ROOT + r'\data\development_evidence.csv').select('pair_id', 'hand_id').with_columns(pl.lit(1).alias('listed'))
phase = pl.read_parquet(ROOT + r'\data\hands.parquet', columns=['hand_id', 'phase'])
h = h.join(phase, on='hand_id', how='left'); assert (h['phase'] == 'development').all(); h = h.drop('phase')
h = h.join(lab, on='pair_id').join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('listed').fill_null(0))
feats = [c for c in h.columns if c not in ('pair_id', 'hand_id', 'table_id', 'label', 'behavior_family', 'listed')]
print('rows', h.height, 'features', len(feats), 'listed', int(h['listed'].sum()))
tables = h['table_id'].unique().sort().to_list(); rng = np.random.default_rng(232); perm = rng.permutation(len(tables)); fold_of = {t: int(i % 5) for t, i in zip(tables, np.argsort(perm))}
h = h.with_columns(pl.col('table_id').replace_strict(fold_of, return_dtype=pl.Int64).alias('fold'))
X = h.select(feats).to_numpy().astype(np.float32); y_listed = h['listed'].to_numpy(); lbl = h['label'].to_numpy(); fold = h['fold'].to_numpy(); fam = h['behavior_family'].to_numpy()
P = dict(n_estimators=400, num_leaves=15, learning_rate=.03, min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=5.0, n_jobs=2, verbosity=-1, random_state=232)
oof = {}
for name, posmask in (('any', y_listed == 1), ('directed_transfer', (y_listed == 1) & (fam == 'directed_transfer')), ('soft_play', (y_listed == 1) & (fam == 'soft_play')), ('coordinated_isolation', (y_listed == 1) & (fam == 'coordinated_isolation'))):
    pr = np.full(h.height, np.nan)
    for k in range(5):
        tr = (fold != k) & (posmask | (lbl == 0)); te = fold == k
        m = lgb.LGBMClassifier(**P).fit(X[tr], posmask[tr].astype(int)); pr[te] = m.predict_proba(X[te])[:, 1]
    oof[name] = pr
    ev_mask = posmask | (lbl == 0); yy = posmask[ev_mask].astype(int); ss = pr[ev_mask]
    neg = np.sort(ss[yy == 0])[::-1]; out = {'auc': round(roc_auc_score(yy, ss), 5), 'ap': round(average_precision_score(yy, ss), 4), 'n_pos': int(yy.sum()), 'n_neg': int((yy == 0).sum())}
    for fpr in (1e-4, 1e-3, 1e-2): thr = neg[int(fpr * len(neg))]; out[f'tpr@fpr{fpr}'] = round(float((ss[yy == 1] > thr).mean()), 4)
    print(name, out)
h = h.with_columns([pl.Series('p_' + k, v) for k, v in oof.items()])
h = h.with_columns(pl.max_horizontal('p_directed_transfer', 'p_soft_play', 'p_coordinated_isolation').alias('p_fmax'))
def lo(c): return (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
agg = []
for c in ('p_any', 'p_fmax'):
    for k in (1, 2, 3, 5): agg.append(lo(c).top_k(k).mean().alias(f'{c}_top{k}'))
    agg.append((pl.col(c) > .5).sum().alias(f'{c}_cnt50')); agg.append((pl.col(c) > .2).sum().alias(f'{c}_cnt20')); agg.append((1 - pl.col(c)).log().sum().neg().alias(f'{c}_nor'))
pp = h.group_by('pair_id').agg(pl.col('label').first(), pl.col('behavior_family').first(), pl.len().alias('n_hands'), pl.col('listed').sum().alias('n_listed'), *agg)
ypair = pp['label'].to_numpy(); res = {}
for c in [x for x in pp.columns if x.startswith('p_')]:
    s = pp[c].to_numpy(); negs = np.sort(s[ypair == 0]); row = {'pair_AP': round(average_precision_score(ypair, s), 4), 'pair_AUC': round(roc_auc_score(ypair, s), 4)}
    for nl in (3, 4, 5):
        q = pp.filter((pl.col('label') == 1) & (pl.col('n_listed') == nl))[c].to_numpy(); pct = np.searchsorted(negs, q) / len(negs)
        row[f'listed{nl}_above_all_negs'] = round(float((pct >= 1.0).mean()), 3); row[f'listed{nl}_above_p99.5'] = round(float((pct >= .995).mean()), 3)
    res[c] = row; print(c, row)
pp.write_parquet(OUT + r'\pair_pooled_oof.parquet'); h.select('pair_id', 'hand_id', 'table_id', 'label', 'behavior_family', 'listed', 'fold', *[c for c in h.columns if c.startswith('p_')]).write_parquet(OUT + r'\hand_oof.parquet')
json.dump(res, open(OUT + r'\screen.json', 'w', encoding='utf-8'), indent=1)
print('DONE')
