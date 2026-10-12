"""STEP250 part B -- score EVERY evaluation pair-hand with the clean-negative injected-event classifier (STEP232 recipe,
fitted on ALL labelled development hands: listed hands of positive pairs = 1, every hand of confirmed non-target pairs = 0;
unlisted hands of positive pairs left out).  One model per disclosed family.  Output: one row per evaluation pair with
event COUNTS at thresholds fixed from development out-of-fold honest hands (p_fmax q99 = .0852, q99.9 = .4662, 19.184)
and top-k log-odds summaries.  No evaluation label exists; nothing here reads the leaderboard.  Single phase per
artefact: the training frame asserts development, the scored frame asserts evaluation (hand_id -> phase).
This script only COMPUTES; the promotion rule is registered (registration250.json) before the aggregates are inspected.
"""
import os, json, glob, time
os.environ['POLARS_MAX_THREADS'] = '2'
import numpy as np, polars as pl, lightgbm as lgb

ROOT = r'C:\Data\Projects\Money Making\work\poker'
OUT = ROOT + r'\Claude Version\experiments\step250_event_count_tail_20260917'
os.makedirs(OUT, exist_ok=True)
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
P = dict(n_estimators=400, num_leaves=15, learning_rate=.03, min_child_samples=30, subsample=.8, subsample_freq=1,
         colsample_bytree=.5, reg_lambda=5.0, n_jobs=6, verbosity=-1, random_state=232)
THR99, THR999 = .0852, .4662          # development OOF honest-hand quantiles of p_fmax (19.184)
t0 = time.time()
phase = pl.read_parquet(ROOT + r'\data\hands.parquet', columns=['hand_id', 'phase'])
h = pl.read_parquet(ROOT + r'\outputs_v2\hands_development\batch_*.parquet')
hp = h.select('hand_id').unique().join(phase, on='hand_id', how='left')
assert (hp['phase'] == 'development').all()
lab = pl.read_csv(ROOT + r'\data\development_labels.csv').select('pair_id', 'label', 'behavior_family')
ev = pl.read_csv(ROOT + r'\data\development_evidence.csv').select('pair_id', 'hand_id').with_columns(pl.lit(1).alias('listed'))
h = h.join(lab, on='pair_id').join(ev, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('listed').fill_null(0))
feats = [c for c in h.columns if c not in ('pair_id', 'hand_id', 'table_id', 'label', 'behavior_family', 'listed')]
X = h.select(feats).to_numpy().astype(np.float32)
listed = h['listed'].to_numpy() == 1; lbl = h['label'].to_numpy(); fam = h['behavior_family'].to_numpy()
models = {}
for f in FAMS:
    pos = listed & (fam == f); tr = pos | (lbl == 0)
    models[f] = lgb.LGBMClassifier(**P).fit(X[tr], pos[tr].astype(int))
    print('fitted', f, int(pos.sum()), 'positives', int((lbl == 0).sum()), 'honest hands', round(time.time() - t0), 's', flush=True)
del X, h
evp = phase.filter(pl.col('phase') == 'evaluation').select('hand_id')
files = sorted(glob.glob(ROOT + r'\outputs_v2\hands_evaluation\batch_*.parquet'))
parts = []
for fp in files:
    g = pl.read_parquet(fp)
    assert [c for c in g.columns if c not in ('pair_id', 'hand_id', 'table_id')] == feats, 'column layout differs'
    assert g.select('hand_id').unique().join(evp, on='hand_id', how='anti').height == 0, 'non-evaluation hand in ' + fp
    Xe = g.select(feats).to_numpy().astype(np.float32)
    sc = {f: models[f].predict_proba(Xe)[:, 1] for f in FAMS}
    del Xe
    g = g.select('pair_id', 'hand_id').with_columns([pl.Series('p_' + f, sc[f].astype(np.float32)) for f in FAMS])
    g = g.with_columns(pl.max_horizontal(['p_' + f for f in FAMS]).alias('p_fmax'))
    lo = (pl.col('p_fmax').clip(1e-6, 1 - 1e-6) / (1 - pl.col('p_fmax').clip(1e-6, 1 - 1e-6))).log()
    agg = g.group_by('pair_id').agg(
        pl.len().alias('n_hands'),
        (pl.col('p_fmax') > THR999).sum().alias('cnt999'), (pl.col('p_fmax') > THR99).sum().alias('cnt99'),
        (pl.col('p_fmax') > .9).sum().alias('cnt_p90'),
        *[(pl.col('p_' + f) > THR999).sum().alias('cnt999_' + f) for f in FAMS],
        lo.top_k(3).mean().alias('top3_logit'), lo.top_k(5).mean().alias('top5_logit'), lo.max().alias('top1_logit'),
        (1 - pl.col('p_fmax').clip(0, 1 - 1e-6)).log().sum().neg().alias('noisy_or'))
    parts.append(agg)
    # keep the sharp hands themselves (for evidence lists and for reading), small
    g.filter(pl.col('p_fmax') > THR99).write_parquet(OUT + '\\sharp_hands_' + os.path.basename(fp))
    print(os.path.basename(fp), g.height, 'rows', round(time.time() - t0), 's', flush=True)
allp = pl.concat(parts)
assert allp['pair_id'].n_unique() == allp.height, 'a pair is split across batches'
allp.write_parquet(OUT + r'\eval_pair_event_counts.parquet')
json.dump({'pairs': allp.height, 'seconds': round(time.time() - t0), 'thresholds': {'q99': THR99, 'q999': THR999}, 'features': len(feats)},
          open(OUT + r'\score_receipt.json', 'w', encoding='utf-8'), indent=1)
print('DONE', allp.height)
