"""Step 1: rebuild the 382-feature frame without held truth, reproduce saved predictions, refit original-only.

Output: Claude Version/experiments/evidence_augmentation_20260910/step1-baseline-parity.json
"""
import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import EXP, BUNDLE, BUNDLE_SHA, SAVED_SCORES, sha, write_json, check_protected
from evidence_aug.frame import (load_original, held_pools, roles, training_relevance, attach_relevance, score,
                                cache_identity, family_columns, PARAMS, matrix)
import joblib
import numpy as np
import polars as pl
import lightgbm as lgb


def fit_ranker(training, features, params=PARAMS):
    training = family_columns(training.sort('pair_id', 'hand_id'), 'behavior_family')
    groups = training.group_by('pair_id', maintain_order=True).agg(pl.len().alias('n'), pl.col('relevance').sum())
    assert groups['relevance'].min() >= 1
    model = lgb.LGBMRanker(**params)
    model.fit(matrix(training, features), training['relevance'].to_numpy(), group=groups['n'].to_numpy())
    return model


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    protected = check_protected()
    assert sha(BUNDLE) == BUNDLE_SHA
    bundle = joblib.load(BUNDLE)
    data, features, caf = load_original()
    assert bundle['features'] == features, 'feature order differs from the saved bundle'
    assert dict(bundle['model'].get_params()) == dict(lgb.LGBMRanker(**PARAMS).get_params())
    pools = held_pools()
    held = data.filter(pl.col('table_id').is_in(pools))
    train = data.filter(~pl.col('table_id').is_in(pools))
    assert not set(held['table_id']) & set(train['table_id'])
    # Only 181 of the 297 training pools and 64 of the 100 held pools contain confirmed target pairs.
    assert held['pair_id'].n_unique() == 98 and train['pair_id'].n_unique() == 274
    assert train['table_id'].n_unique() == 181 and held['table_id'].n_unique() == 64
    report = {'seconds': None, 'bundle_sha256': BUNDLE_SHA, 'cache_identity': cache_identity(), 'protected': protected,
              'held_pools': len(pools), 'held_target_pairs': 98, 'held_rows': held.height, 'training_target_pairs': 274,
              'training_rows': train.height, 'parity': {}}
    # 1) saved-model predictions under both fixed role assignments
    for kind, saved_path in SAVED_SCORES.items():
        frame = held.join(roles(kind), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        predicted = score(bundle['model'], family_columns(frame, 'predicted_behavior'), features)
        saved = pl.read_parquet(saved_path)
        assert frame.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
        diff = np.abs(predicted - saved['score'].to_numpy())
        report['parity'][kind] = {'rows': int(saved.height), 'exact': bool(np.array_equal(predicted, saved['score'].to_numpy())),
                                  'max_abs_diff': float(diff.max()), 'saved_sha256': sha(saved_path)}
        print(json.dumps({kind: report['parity'][kind]}), flush=True)
    # 2) refit original-only with training-pool relevance only; compare to the saved bundle
    evidence = training_relevance(set(train['table_id']))
    assert not set(evidence['pair_id']) & set(held['pair_id'])
    training = attach_relevance(train, evidence)
    t1 = time.perf_counter()
    model = fit_ranker(training, features)
    fit_seconds = time.perf_counter() - t1
    same_trees = model.booster_.model_to_string() == bundle['model'].booster_.model_to_string()
    frame = held.join(roles('conditional_roles'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
    refit = score(model, family_columns(frame, 'predicted_behavior'), features)
    saved = pl.read_parquet(SAVED_SCORES['conditional_roles'])['score'].to_numpy()
    report['parity']['refit_original_only'] = {
        'training_listed_hands': int(evidence.height), 'training_queries': int(training['pair_id'].n_unique()),
        'identical_model_string': bool(same_trees), 'exact_scores': bool(np.array_equal(refit, saved)),
        'max_abs_diff': float(np.abs(refit - saved).max()), 'fit_seconds': fit_seconds}
    print(json.dumps({'refit_original_only': report['parity']['refit_original_only']}), flush=True)
    report['seconds'] = time.perf_counter() - t0
    report['held_evidence_truth_read'] = False
    write_json(EXP / 'step1-baseline-parity.json', report)
    print(json.dumps({'complete': True, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
