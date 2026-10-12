"""Step 8: training-only diagnosis of why synthetic intervention hands fail as evidence labels.

Uses only training-pool original data (297 pools) and the 78 synthetic pairs. Three pool-grouped
source classifiers on the 379 non-family ranker features (within-pair contexts included):
  real_listed_vs_synthetic_intervention   -- are synthetic positives a different kind of hand than listed evidence?
  real_listed_vs_real_unlisted            -- how separable is the ranker's real task (soft-play pairs only)?
  synthetic_intervention_vs_synthetic_other -- how separable are interventions inside synthetic pairs?
Plus the top discriminating features (gain) and their mean values in both groups.
Output: Claude Version/experiments/evidence_augmentation_20260910/diagnosis-synthetic-evidence.json
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import EXP, write_json, read_json
from evidence_aug.frame import load_original, held_pools, training_relevance, attach_relevance, matrix, family_columns
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

PARAMS = dict(n_estimators=100, num_leaves=7, max_depth=3, min_child_samples=30, class_weight='balanced', learning_rate=.05, n_jobs=2, verbosity=-1, random_state=19083)


def source_auc(x, y, groups):
    pred = np.full(len(y), np.nan); gains = np.zeros(x.shape[1])
    for k, (tr, te) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=19083).split(x, y, groups)):
        assert not set(groups[tr]) & set(groups[te])
        model = lgb.LGBMClassifier(**{**PARAMS, 'random_state': 19083 + k}).fit(x[tr], y[tr])
        pred[te] = model.predict_proba(x[te])[:, 1]; gains += model.booster_.feature_importance('gain')
    return float(roc_auc_score(y, pred)), gains


def main():
    data, features, caf = load_original()
    pools = held_pools()
    train = data.filter(~pl.col('table_id').is_in(pools))
    train = attach_relevance(train, training_relevance(set(train['table_id'])))
    real = train.filter(pl.col('behavior_family') == 'soft_play')
    synthetic = pl.concat([pl.read_parquet(EXP / 'synthetic' / f'batch_{b}.parquet') for b in '012'])
    traces = pl.read_parquet(EXP / 'synthetic' / 'traces.parquet').select('pair_id', 'hand_id').with_columns(pl.lit(1, dtype=pl.Int64).alias('relevance'))
    synthetic = attach_relevance(synthetic, traces)
    cols = [c for c in features if not c.startswith('family_')]
    assert len(cols) == 379
    contrasts = {
        'real_listed_vs_synthetic_intervention': (real.filter(pl.col('relevance') == 1), synthetic.filter(pl.col('relevance') == 1)),
        'real_listed_vs_real_unlisted': (real.filter(pl.col('relevance') == 1), real.filter(pl.col('relevance') == 0)),
        'synthetic_intervention_vs_synthetic_other': (synthetic.filter(pl.col('relevance') == 1), synthetic.filter(pl.col('relevance') == 0)),
        'real_unlisted_vs_synthetic_other': (real.filter(pl.col('relevance') == 0), synthetic.filter(pl.col('relevance') == 0)),
    }
    out = {'training_only': True, 'features': len(cols), 'contrasts': {}}
    for name, (a, b) in contrasts.items():
        x = np.r_[matrix(a, cols), matrix(b, cols)]; y = np.r_[np.zeros(a.height), np.ones(b.height)]
        groups = np.r_[a['table_id'].to_numpy(), b['table_id'].to_numpy()]
        auc, gains = source_auc(x, y, groups)
        top = np.argsort(gains)[::-1][:12]
        out['contrasts'][name] = {'rows': [a.height, b.height], 'auc_second_group_positive': auc,
                                  'top_features': [{'feature': cols[i], 'gain_share': float(gains[i] / gains.sum()), 'mean_first': float(a[cols[i]].mean()), 'mean_second': float(b[cols[i]].mean())} for i in top]}
        print(json.dumps({name: {'auc': auc, 'rows': [a.height, b.height], 'top3': [cols[i] for i in top[:3]]}}), flush=True)
    write_json(EXP / 'diagnosis-synthetic-evidence.json', out)


if __name__ == '__main__':
    main()
