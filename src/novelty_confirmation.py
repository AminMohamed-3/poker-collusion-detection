"""Confirm the frozen novelty recipe on existing pool folds 1 and 2 only."""
from __future__ import annotations

import os

for key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'

import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit

from novelty_experiment import PARAMS, score_report, training_percentile
from robust_pair_experiment import FAMILIES, LABELS, MODEL_PARAMS, assert_disjoint, peak_memory_bytes, sha

ROOT = Path(__file__).resolve().parents[1]
DATA, SOURCE, REPORT = ROOT / 'data', ROOT / 'outputs_v2', ROOT / 'reports'


def run():
    started = time.perf_counter()
    inputs = [DATA / 'development_labels.csv', SOURCE / 'pairs_development.parquet',
              SOURCE / 'validation_pairs.csv', REPORT / 'open-set-results.json',
              REPORT / 'novelty-results.json', ROOT / 'src' / 'novelty_experiment.py',
              ROOT / 'src' / 'open_set_experiment.py', ROOT / 'src' / 'robust_pair_experiment.py']
    protected = [SOURCE / 'pair_model_robust.joblib', SOURCE / 'pair_model.joblib',
                 SOURCE / 'hand_model.joblib', ROOT / 'outputs' / 'submission_baseline.csv',
                 SOURCE / 'submission_advanced.csv', SOURCE / 'submission_advanced_robust.csv']
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protected_before = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns}
                        for p in protected}
    previous = json.loads((REPORT / 'open-set-results.json').read_text(encoding='utf-8'))
    fold0 = json.loads((REPORT / 'novelty-results.json').read_text(encoding='utf-8'))
    if PARAMS != fold0['model_parameters'] or MODEL_PARAMS != previous['model_parameters']:
        raise ValueError('The confirmation must preserve both model recipes exactly')
    labels = pl.read_csv(DATA / 'development_labels.csv')
    confirmed = (((pl.col('label') == 0) & (pl.col('label_status') == 'confirmed_non_target')) |
                 ((pl.col('label') == 1) & (pl.col('label_status') == 'confirmed_target')))
    if (labels.height != 1860 or labels['pair_id'].n_unique() != 1860 or
            labels.filter(~confirmed).height or set(labels['behavior_family']) != set(FAMILIES)):
        raise ValueError('Expected exactly 1860 unique explicit, consistent development labels')
    full = pl.read_parquet(SOURCE / 'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    if full.height != 1860 or full['pair_id'].n_unique() != 1860:
        raise ValueError('Explicit labels and pair summaries must match once each')
    excluded = {'pair_id', 'hand_id', 'table_id', 'player_1', 'player_2',
                'label', 'label_status', 'behavior_family'}
    features = [c for c in full.columns if c not in excluded]
    if (len(features) != 653 or features != previous['feature_names'] or
            features != fold0['feature_names'] or any(not full.schema[c].is_numeric() for c in features)):
        raise ValueError('Feature columns must preserve all 653 original numeric features')
    truth = np.array([LABELS[name] for name in full['behavior_family']])
    if not np.array_equal(truth > 0, full['label'].to_numpy() == 1):
        raise ValueError('Behavior and binary labels disagree')
    splits = list(GroupShuffleSplit(n_splits=3, test_size=.25, random_state=2901).split(
        full, truth, full['table_id']))
    if set(full[splits[0][1].tolist()]['pair_id']) != set(pl.read_csv(SOURCE / 'validation_pairs.csv')['pair_id']):
        raise ValueError('Split sequence does not reproduce saved fold 0')
    results, predictions, references = [], [], []
    supervised_fits = novelty_fits = 0
    for fold in [1, 2]:
        train_index, held_index = splits[fold]
        training = full[train_index.tolist()].sort('pair_id')
        held = full[held_index.tolist()].sort('pair_id')
        assert_disjoint(training, held)
        negatives = training.filter((pl.col('behavior_family') == 'none') & (pl.col('label') == 0) &
                                    (pl.col('label_status') == 'confirmed_non_target'))
        if negatives.height != training.filter(pl.col('label') == 0).height:
            raise ValueError('Novelty training set must contain exactly the confirmed-none training pairs')
        X_normal = negatives.select(features).to_numpy().astype(np.float32)
        X_held = held.select(features).to_numpy().astype(np.float32)
        if not np.isfinite(X_normal).all() or not np.isfinite(X_held).all():
            raise ValueError('Raw numeric feature matrices must be finite')
        novelty_model = IsolationForest(**PARAMS)
        novelty_model.fit(X_normal)
        novelty_fits += 1
        if len(novelty_model.estimators_) != 200 or novelty_model.max_samples_ != 256:
            raise ValueError('Unexpected novelty fit size')
        normal_scores = -novelty_model.score_samples(X_normal)
        novelty = -novelty_model.score_samples(X_held)
        percentile = training_percentile(normal_scores, novelty)
        floor = .5 * np.clip((percentile - .95) / .05, 0, 1)
        held_truth = np.array([LABELS[name] for name in held['behavior_family']])
        reference_frame = negatives.select('pair_id', 'table_id').with_columns(
            pl.lit(fold).alias('fold'), pl.Series('novelty_score', normal_scores))
        references.append(reference_frame)
        result = {'fold': fold, 'training_pairs': training.height,
                  'training_confirmed_none_pairs': negatives.height, 'heldout_pairs': held.height,
                  'training_pools': training['table_id'].n_unique(), 'heldout_pools': held['table_id'].n_unique(),
                  'heldout_counts': {family: int((held_truth == LABELS[family]).sum()) for family in FAMILIES},
                  'pair_table_player_overlap': False, 'standalone': score_report(held_truth, novelty),
                  'heldout_none_above_95pct_training_reference': int(((held_truth == 0) & (percentile > .95)).sum()),
                  'comparisons': []}
        for omitted in ['all_families', *FAMILIES[1:]]:
            subset = training if omitted == 'all_families' else training.filter(pl.col('behavior_family') != omitted)
            assert_disjoint(subset, held)
            if (subset.filter(pl.col('label') == 0).height != negatives.height or
                    (omitted != 'all_families' and omitted in set(subset['behavior_family']))):
                raise ValueError('Omission must remove only the named positive family')
            X_train = subset.select(features).to_numpy().astype(np.float32)
            if not np.isfinite(X_train).all():
                raise ValueError('Raw supervised input matrix must be finite')
            y = np.array([LABELS[name] for name in subset['behavior_family']])
            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            model.fit(X_train, y)
            supervised_fits += 1
            classes = model.classes_.tolist()
            if (classes.count(0) != 1 or model.booster_.current_iteration() != 226 or
                    (omitted != 'all_families' and LABELS[omitted] in classes)):
                raise ValueError('Unexpected fitted class mapping or boosting count')
            risk = 1 - model.predict_proba(X_held)[:, classes.index(0)]
            blended = np.maximum(risk, floor)
            comparison = {'training_omission': omitted, 'training_pairs': subset.height,
                          'removed_training_pairs': training.height - subset.height, 'model_classes': classes,
                          'boosting_rounds': model.booster_.current_iteration(),
                          'multiclass': score_report(held_truth, risk),
                          'fixed_blend': score_report(held_truth, blended)}
            result['comparisons'].append(comparison)
            predictions.append(held.select('pair_id', 'table_id', 'label', 'behavior_family').with_columns(
                pl.lit(fold).alias('fold'), pl.lit(omitted).alias('training_omission'),
                pl.Series('multiclass_risk', risk), pl.Series('novelty_score', novelty),
                pl.Series('normal_train_percentile', percentile), pl.Series('novelty_risk_floor', floor),
                pl.Series('blended_risk', blended)))
            metric_name = 'overall_targets_vs_none' if omitted == 'all_families' else 'each_family_vs_none'
            scores = {name: (comparison[name][metric_name] if omitted == 'all_families'
                             else comparison[name][metric_name][omitted]) for name in ['multiclass', 'fixed_blend']}
            print(json.dumps({'fold': fold, 'training_omission': omitted, 'metric_population':
                              'all_targets_vs_none' if omitted == 'all_families' else 'omitted_family_vs_none',
                              **{name: {'ap': values['ap'], 'ap_1pct': values['ap_at_hypothetical_1pct_prevalence']}
                                 for name, values in scores.items()}}), flush=True)
            del model, X_train
            gc.collect()
        results.append(result)
        del novelty_model, X_normal, X_held
        gc.collect()
    if supervised_fits != 8 or novelty_fits != 2:
        raise ValueError('Confirmation must remain bounded to eight multiclass and two novelty fits')
    for relative, fingerprint in fingerprints.items():
        if sha(ROOT / relative) != fingerprint:
            raise ValueError(f'Input changed during confirmation: {relative}')
    for relative, before in protected_before.items():
        path = ROOT / relative
        if sha(path) != before['sha256'] or path.stat().st_mtime_ns != before['mtime_ns']:
            raise ValueError(f'Protected artifact changed: {relative}')
    output = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
              'input_sha256': fingerprints, 'input_hashes_unchanged': True,
              'protected_artifacts_before_and_after': protected_before, 'protected_artifacts_unchanged': True,
              'split': {'n_splits': 3, 'test_size': .25, 'random_state': 2901, 'executed_folds': [1, 2]},
              'feature_count': len(features), 'feature_names': features,
              'isolation_forest_parameters': PARAMS, 'multiclass_parameters': MODEL_PARAMS,
              'fitted_novelty_models': novelty_fits, 'fitted_supervised_models': supervised_fits,
              'blend': fold0['blend'], 'reference_definition': fold0['reference']['definition'].replace(' / 1129', ' / training_negative_count'),
              'results': results, 'elapsed_seconds': time.perf_counter() - started,
              'peak_resident_memory_bytes': peak_memory_bytes(),
              'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                           'lightgbm': lgb.__version__, 'numpy': np.__version__, 'polars': pl.__version__},
              'limitations': [
                  'The three GroupShuffleSplit holdouts can overlap; these are not three independent data sets.',
                  'Known-family omission only proxies the undisclosed fourth mechanism.',
                  'Training-normal empirical percentiles are not probabilities; reference scores are in-sample.',
                  'One-percent prevalence results are reweighted diagnostics, not estimated test prevalence.',
                  'Full-history multiclass controls omit production history augmentation.',
                  'No parameter, threshold, feature, or ensemble-weight search; no evaluation predictions or saved models.']}
    (REPORT / 'novelty-confirmation-results.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    pl.concat(predictions).write_csv(REPORT / 'novelty-confirmation-predictions.csv')
    pl.concat(references).write_csv(REPORT / 'novelty-confirmation-training-reference.csv')
    print(json.dumps({'supervised_fits': supervised_fits, 'novelty_fits': novelty_fits,
                      'elapsed_seconds': output['elapsed_seconds'], 'protected_artifacts_unchanged': True}), flush=True)


if __name__ == '__main__':
    run()
