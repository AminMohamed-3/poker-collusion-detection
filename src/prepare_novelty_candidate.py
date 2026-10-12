"""Private fixed novelty alternative; never assembles or uploads a submission."""
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

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import IsolationForest

from novelty_experiment import PARAMS, training_percentile
from robust_pair_experiment import peak_memory_bytes, sha

ROOT = Path(__file__).resolve().parents[1]
DATA, SOURCE = ROOT / 'data', ROOT / 'outputs_v2'
OUT, CANDIDATE = ROOT / 'outputs_novelty', ROOT / 'outputs_v3' / 'novelty_candidate.parquet'
CHUNK_ROWS = 4096


def novelty_score(model, values):
    with joblib.parallel_backend('threading', n_jobs=2):
        return -model.score_samples(values)


def snapshot(paths):
    return {str(path.relative_to(ROOT)): {'sha256': sha(path), 'mtime_ns': path.stat().st_mtime_ns}
            for path in paths}


def run():
    started = time.perf_counter()
    if CANDIDATE.exists() or OUT.exists():
        raise ValueError('Private novelty output already exists; review it rather than overwrite it')
    inputs = [DATA / 'development_labels.csv', DATA / 'evaluation_pairs.csv', DATA / 'sample_submission.csv',
              SOURCE / 'pairs_development.parquet', SOURCE / 'pairs_evaluation.parquet',
              SOURCE / 'submission_advanced_robust.csv', ROOT / 'reports' / 'novelty-results.json',
              ROOT / 'src' / 'novelty_experiment.py', ROOT / 'src' / 'robust_pair_experiment.py']
    protected = [SOURCE / 'pair_model_robust.joblib', SOURCE / 'pair_model.joblib', SOURCE / 'hand_model.joblib',
                 SOURCE / 'submission_advanced.csv', ROOT / 'outputs' / 'submission_baseline.csv',
                 ROOT / 'outputs_v3' / 'evidence_ranker.joblib', ROOT / 'outputs_v3' / 'fit.json']
    before = snapshot([*inputs, *protected])
    fixed = json.loads((ROOT / 'reports' / 'novelty-results.json').read_text(encoding='utf-8'))
    features = fixed['feature_names']
    if len(features) != 653 or PARAMS != fixed['model_parameters']:
        raise ValueError('The fixed model recipe or feature list changed')
    forbidden = {'pair_id', 'hand_id', 'table_id', 'player_1', 'player_2', 'label', 'label_status', 'behavior_family'}
    if forbidden & set(features):
        raise ValueError('Identifiers, groups, and labels cannot be features')
    labels = pl.read_csv(DATA / 'development_labels.csv')
    if labels.height != 1860 or labels['pair_id'].n_unique() != 1860:
        raise ValueError('Expected exactly 1860 unique explicit development labels')
    negative_labels = labels.filter((pl.col('label') == 0) & (pl.col('label_status') == 'confirmed_non_target') &
                                    (pl.col('behavior_family') == 'none'))
    if negative_labels.height != 1488 or labels.filter(pl.col('label') == 0).height != 1488:
        raise ValueError('Expected exactly 1488 explicitly confirmed-none pairs')
    development = pl.read_parquet(SOURCE / 'pairs_development.parquet')
    if development['pair_id'].n_unique() != development.height:
        raise ValueError('Development pair summaries contain duplicate IDs')
    training = development.join(negative_labels.select('pair_id'), on='pair_id', how='inner').sort('pair_id')
    if training.height != 1488 or set(training['pair_id']) != set(negative_labels['pair_id']):
        raise ValueError('Training coverage does not exactly match confirmed-none labels')
    if any(not training.schema[c].is_numeric() for c in features):
        raise ValueError('All 653 features must be numerical')
    X_train = training.select(features).to_numpy().astype(np.float32)
    if not np.isfinite(X_train).all():
        raise ValueError('Raw training features must be finite')
    model = IsolationForest(**PARAMS)
    model.fit(X_train)
    if model.n_features_in_ != 653 or len(model.estimators_) != 200 or model.max_samples_ != 256:
        raise ValueError('The fitted model does not match the declared recipe')
    reference = novelty_score(model, X_train)
    template = pl.read_csv(DATA / 'sample_submission.csv', columns=['pair_id'])
    evaluation = pl.read_csv(DATA / 'evaluation_pairs.csv', columns=['pair_id'])
    original = pl.read_csv(SOURCE / 'submission_advanced_robust.csv',
                           columns=['pair_id', 'risk_score', 'predicted_behavior']).rename({'risk_score': 'original_robust_risk'})
    expected_ids = set(template['pair_id'])
    for frame in [template, evaluation, original]:
        if frame.height != 112540 or frame['pair_id'].n_unique() != 112540 or set(frame['pair_id']) != expected_ids:
            raise ValueError('Template, evaluation pairs, and original robust risks must match exactly')
    if expected_ids & set(labels['pair_id']):
        raise ValueError('Development and evaluation pairs must be disjoint')
    raw_risk = original['original_robust_risk'].to_numpy()
    if not np.isfinite(raw_risk).all() or not ((raw_risk >= 0).all() and (raw_risk <= 1).all()):
        raise ValueError('Original robust risks must be finite probabilities in [0, 1]')
    parts = []
    parquet = pq.ParquetFile(SOURCE / 'pairs_evaluation.parquet')
    if parquet.metadata.num_rows != 112540:
        raise ValueError('Evaluation summaries must contain exactly 112540 rows')
    for index, batch in enumerate(parquet.iter_batches(batch_size=CHUNK_ROWS, columns=['pair_id', *features], use_threads=False)):
        frame = pl.from_arrow(batch)
        values = frame.select(features).to_numpy().astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'Evaluation chunk {index} contains non-finite raw features')
        novelty = novelty_score(model, values)
        percentile = training_percentile(reference, novelty)
        floor = .5 * np.clip((percentile - .95) / .05, 0, 1)
        parts.append(frame.select('pair_id').with_columns(pl.Series('novelty_score', novelty),
            pl.Series('reference_percentile', percentile), pl.Series('risk_floor', floor)))
        if index % 8 == 0:
            print(json.dumps({'chunk': index, 'scored_pairs': min((index + 1) * CHUNK_ROWS, 112540)}), flush=True)
        del frame, values
    novelty_frame = pl.concat(parts)
    if novelty_frame.height != 112540 or novelty_frame['pair_id'].n_unique() != 112540 or set(novelty_frame['pair_id']) != expected_ids:
        raise ValueError('Scored pair coverage differs from the template')
    combined = template.join(novelty_frame, on='pair_id', how='left', maintain_order='left').join(
        original, on='pair_id', how='left', maintain_order='left').with_columns(
            pl.max_horizontal('original_robust_risk', 'risk_floor').alias('blended_risk'))
    numeric = combined.select('novelty_score', 'reference_percentile', 'risk_floor', 'original_robust_risk', 'blended_risk').to_numpy()
    if not np.isfinite(numeric).all() or combined['pair_id'].to_list() != template['pair_id'].to_list():
        raise ValueError('Candidate must be finite and preserve exact template ordering')
    if (combined.filter(~pl.col('reference_percentile').is_between(0, 1)).height or
            combined.filter(~pl.col('risk_floor').is_between(0, .5)).height or
            combined.filter(~pl.col('blended_risk').is_between(0, 1)).height):
        raise ValueError('Candidate scores exceed their declared bounds')
    np.testing.assert_array_equal(combined['blended_risk'].to_numpy(),
                                   np.maximum(combined['original_robust_risk'].to_numpy(), combined['risk_floor'].to_numpy()))
    candidate = combined.select('pair_id', 'novelty_score', 'reference_percentile', 'risk_floor', 'blended_risk')
    descriptive = combined.with_columns((pl.col('blended_risk') > pl.col('original_robust_risk')).alias('risk_raised'))
    family_summary = descriptive.group_by('predicted_behavior').agg(pl.len().alias('pairs'),
        pl.col('risk_raised').sum().alias('risk_raised')).sort('predicted_behavior').to_dicts()
    bins = [(0, .01), (.01, .1), (.1, .5), (.5, 1.0000000001)]
    risk_summary = []
    for low, high in bins:
        subset = descriptive.filter((pl.col('original_robust_risk') >= low) & (pl.col('original_robust_risk') < high))
        risk_summary.append({'original_risk_range': f'[{low}, {min(high, 1)}' + (']' if high > 1 else ')'),
                             'pairs': subset.height, 'risk_raised': int(subset['risk_raised'].sum())})
    if snapshot([*inputs, *protected]) != before:
        raise ValueError('An input or protected production artifact changed during preparation')
    OUT.mkdir()
    original_ordered = template.join(original, on='pair_id', how='left', maintain_order='left')
    original_ordered.write_parquet(OUT / 'original_robust_risk.parquet')
    training.select('pair_id', 'table_id').with_columns(pl.Series('novelty_score', reference)).write_parquet(
        OUT / 'training_reference.parquet')
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
        'input_and_protected_artifacts_before_and_after': before, 'input_and_protected_artifacts_unchanged': True,
        'model_parameters': PARAMS, 'feature_count': len(features), 'features': features,
        'fitted_models': 1, 'training_pairs': training.height, 'training_confirmed_none_only': True,
        'training_target_pairs': 0, 'training_unlisted_pairs': 0, 'training_augmentation': False,
        'preprocessing': 'float32 only; all raw features finite; no scaling or imputation',
        'reference': 'right-inclusive empirical CDF of this model scores on all 1488 fitted confirmed-none training pairs',
        'blend': fixed['blend'], 'evaluation_pairs': candidate.height, 'chunk_rows': CHUNK_ROWS,
        'score_jobs': 2, 'exact_pair_coverage_and_template_order': True, 'finite_scores_and_bounds_verified': True,
        'risk_raised_pairs': int(descriptive['risk_raised'].sum()), 'predicted_family_summary': family_summary,
        'original_risk_summary': risk_summary,
        'evaluation_score_quantiles': {name: {str(q): float(np.quantile(combined[name].to_numpy(), q))
            for q in [0, .5, .95, .99, 1]} for name in ['novelty_score', 'reference_percentile', 'risk_floor', 'original_robust_risk', 'blended_risk']},
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'numpy': np.__version__, 'polars': pl.__version__, 'joblib': joblib.__version__},
        'limitations': ['Private alternative only; no submission assembled or uploaded.',
            'Predicted families are existing model outputs, not evaluation labels.',
            'Empirical percentiles and risk floor are not calibrated probabilities.',
            'Validation showed mixed tradeoffs, not dominance or demonstrated fourth-mechanism performance.']}
    bundle_path = OUT / 'isolation_forest.joblib'
    joblib.dump({'model': model, 'features': features, 'fit_provenance': provenance}, bundle_path)
    restored = joblib.load(bundle_path)
    np.testing.assert_allclose(novelty_score(restored['model'], X_train[:32]), reference[:32], rtol=0, atol=1e-12)
    candidate.write_parquet(CANDIDATE)
    if not pl.read_parquet(CANDIDATE).equals(candidate):
        raise ValueError('Candidate did not round-trip exactly')
    if snapshot([*inputs, *protected]) != before:
        raise ValueError('An input or protected production artifact changed before completion')
    provenance.update({'model_reload_verified': True, 'candidate_roundtrip_verified': True,
        'output_sha256': {str(path.relative_to(ROOT)): sha(path) for path in [bundle_path, CANDIDATE,
            OUT / 'original_robust_risk.parquet', OUT / 'training_reference.parquet']},
        'elapsed_seconds': time.perf_counter() - started, 'peak_resident_memory_bytes': peak_memory_bytes()})
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: provenance[key] for key in ['training_pairs', 'evaluation_pairs', 'risk_raised_pairs',
        'predicted_family_summary', 'original_risk_summary', 'elapsed_seconds', 'input_and_protected_artifacts_unchanged']}), flush=True)
    gc.collect()


if __name__ == '__main__':
    run()
