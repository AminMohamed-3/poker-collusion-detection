"""Fit the prescribed repeated-event pair model and prepare a private risk-only entry.

Keeps the current v3 behavior and evidence strings unchanged. No public action.
"""
from __future__ import annotations

import os
for variable in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '2'

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import (ROOT, SOURCE, DATA, KEYS, LABELS, FAMILIES,
    TRAIN_VIEWS, MODEL_PARAMS, thin, aggregate, matrix, sha)
from tail_pair_experiment import tails

OUT = ROOT / 'outputs_v5'
MODEL = OUT / 'pair_model_event_tails.joblib'
SUBMISSION = OUT / 'submission_event_tails.csv'
REFERENCE = ROOT / 'outputs_v3' / 'submission_evidence_ranker.csv'


def identity(phase):
    paths = [Path(__file__), ROOT/'src/tail_pair_experiment.py',
             ROOT/'src/robust_pair_experiment.py', SOURCE/f'pairs_{phase}.parquet',
             SOURCE/f'hands_{phase}/manifest.json']
    paths += sorted((SOURCE / f'hands_{phase}').glob('batch_*.parquet'))
    paths += ([DATA/'development_labels.csv'] if phase == 'development' else
              [DATA/'evaluation_pairs.csv', REFERENCE, MODEL, OUT/'fit.json'])
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def fit():
    OUT.mkdir(exist_ok=True)
    if MODEL.exists() or (OUT/'fit.json').exists():
        raise FileExistsError('Refusing to overwrite a fitted model')
    hashes = identity('development')
    labels = pl.read_csv(DATA/'development_labels.csv')
    assert labels.height == labels['pair_id'].n_unique() == 1860
    assert not labels.filter(~pl.col('label_status').is_in(
        ['confirmed_target', 'confirmed_non_target'])).height
    full = pl.read_parquet(SOURCE/'pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    base_features = joblib.load(SOURCE/'pair_model_robust.joblib')['features']
    raw = pl.concat([pl.read_parquet(path) for path in
                    sorted((SOURCE/'hands_development').glob('batch_*.parquet'))])
    assert raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == raw.height
    assert not raw.select('pair_id', 'table_id').unique().join(full.select('pair_id', 'table_id'),
        on=['pair_id', 'table_id'], how='anti').height
    counts = raw.group_by('pair_id').len().join(full.select('pair_id', 'shared_hands'), on='pair_id')
    assert counts.height == full.height == labels.height
    assert not counts.filter(pl.col('len') != pl.col('shared_hands')).height
    raw_names = [c for c in raw.columns if c not in KEYS]
    events = [c for c in raw_names if c.startswith('v2_')]
    assert len(base_features) == 653 and len(events) == 56
    extra = tails(raw, events)
    extra_names = [c for c in extra.columns if c not in KEYS]
    full = full.join(extra, on=['pair_id', 'table_id']).sort('pair_id')
    features = base_features + extra_names
    views = [full]
    for fraction, seed in TRAIN_VIEWS:
        sample = thin(raw, fraction, seed)
        frame = aggregate(sample, raw_names).join(tails(sample, events), on=['pair_id', 'table_id'])
        frame = frame.join(labels, on='pair_id').select(full.columns).sort('pair_id')
        assert frame['pair_id'].to_list() == full['pair_id'].to_list()
        np.testing.assert_array_equal(frame['shared_hands'],
            np.maximum(1, np.floor(full['shared_hands'].to_numpy()*fraction)))
        views.append(frame)
    training = pl.concat(views)
    assert training.height == 5580
    weights = np.full(training.height, 1/3)
    weight_check = training.select('pair_id').with_columns(pl.Series('weight', weights))
    totals = weight_check.group_by('pair_id').agg(pl.col('weight').sum())
    np.testing.assert_allclose(totals['weight'], 1, rtol=0, atol=1e-12)
    y = np.array([LABELS[name] for name in training['behavior_family']])
    assert np.array_equal(y > 0, training['label'].to_numpy() == 1)
    X = matrix(training, features)
    model = lgb.LGBMClassifier(**MODEL_PARAMS)
    model.fit(X, y, sample_weight=weights)
    assert np.array_equal(model.classes_, np.arange(4)) and model.n_features_in_ == 765
    bundle = {'model': model, 'features': features, 'events': events,
              'base_features': base_features, 'extra_features': extra_names,
              'class_names': FAMILIES, 'classes': model.classes_.tolist()}
    temporary = MODEL.with_suffix('.joblib.tmp')
    joblib.dump(bundle, temporary)
    reloaded = joblib.load(temporary)
    np.testing.assert_allclose(model.predict_proba(X[:64]), reloaded['model'].predict_proba(X[:64]),
                               rtol=0, atol=1e-12)
    assert identity('development') == hashes, 'Training inputs changed'
    temporary.replace(MODEL)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': hashes, 'model_sha256': sha(MODEL), 'model_params': MODEL_PARAMS,
        'python': platform.python_version(), 'polars': pl.__version__,
        'numpy': np.__version__, 'lightgbm': lgb.__version__,
        'source_pairs': labels.height, 'training_rows': training.height,
        'source_pair_hands': raw.height, 'feature_count': len(features),
        'class_pairs': {name: full.filter(pl.col('behavior_family') == name).height for name in FAMILIES},
        'views': [[1., None], *TRAIN_VIEWS], 'weight_per_view': 1/3,
        'reload_predictions_match': True, 'unknown_pair_training_rows': 0,
        'validation': 'reports/tail-pair-experiment/metrics.json',
        'limitations': ['Final all-label fit has no validation score of its own.',
            'The retained v3 behavior labels and evidence are not changed by this risk model.']}
    (OUT/'fit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['model_sha256', 'source_pairs', 'training_rows',
                                          'feature_count', 'reload_predictions_match']}), flush=True)


def predict():
    if SUBMISSION.exists():
        raise FileExistsError('Refusing to overwrite a prepared entry')
    report = json.loads((OUT/'fit.json').read_text(encoding='utf-8'))
    assert sha(MODEL) == report['model_sha256']
    for path in ['src/tail_pair_candidate.py', 'src/tail_pair_experiment.py',
                 'src/robust_pair_experiment.py']:
        assert sha(ROOT/path) == report['input_sha256'][path], 'Training source changed'
    hashes = identity('evaluation')
    bundle = joblib.load(MODEL)
    reference = pl.read_csv(REFERENCE, infer_schema=False)
    pairs = pl.read_parquet(SOURCE/'pairs_evaluation.parquet')
    assert pairs.height == pairs['pair_id'].n_unique() == reference.height == 112540
    assert set(pairs['pair_id']) == set(reference['pair_id'])
    predictions, observed = [], []
    for path in sorted((SOURCE/'hands_evaluation').glob('batch_*.parquet')):
        raw = pl.read_parquet(path, columns=KEYS + bundle['events'])
        extra = tails(raw, bundle['events'])
        frame = pairs.join(extra, on=['pair_id', 'table_id']).sort('pair_id')
        assert frame.height == extra.height == extra['pair_id'].n_unique()
        counts = raw.group_by('pair_id').len().join(frame.select('pair_id', 'shared_hands'), on='pair_id')
        assert counts.height == frame.height
        assert not counts.filter(pl.col('len') != pl.col('shared_hands')).height
        risk = 1-bundle['model'].predict_proba(matrix(frame, bundle['features']))[:, 0]
        assert np.isfinite(risk).all() and np.all((risk >= 0) & (risk <= 1))
        predictions.append(frame.select('pair_id').with_columns(pl.Series('_risk', risk)))
        observed.append({'batch': path.name, 'pairs': frame.height, 'pair_hands': raw.height})
        print(f'{path.name}: {frame.height} pairs', flush=True)
    prediction = pl.concat(predictions)
    assert prediction.height == prediction['pair_id'].n_unique() == reference.height
    assert set(prediction['pair_id']) == set(reference['pair_id'])
    assembled = reference.join(prediction, on='pair_id', maintain_order='left').with_columns(
        pl.col('_risk').cast(pl.String).alias('risk_score')).select(reference.columns)
    fixed = [c for c in reference.columns if c != 'risk_score']
    assert assembled.select(fixed).equals(reference.select(fixed))
    temporary = SUBMISSION.with_suffix('.csv.tmp')
    assembled.write_csv(temporary)
    reloaded = pl.read_csv(temporary, infer_schema=False)
    assert reloaded.equals(assembled)
    assert identity('evaluation') == hashes, 'Prediction inputs changed'
    temporary.replace(SUBMISSION)
    out = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'submission_sha256': sha(SUBMISSION), 'input_sha256': hashes,
        'rows': assembled.height, 'changed_column': 'risk_score',
        'unchanged_columns': fixed, 'exact_string_preservation_verified': True,
        'changed_risk_rows': assembled.filter(pl.col('risk_score') != reference['risk_score']).height,
        'batches': observed, 'full_membership_validation': 'Must run src/validation.py separately.'}
    (OUT/'submission_event_tails.provenance.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['submission_sha256', 'rows', 'changed_risk_rows']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fit', 'predict'])
    args = parser.parse_args()
    fit() if args.command == 'fit' else predict()
