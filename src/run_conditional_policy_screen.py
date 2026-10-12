"""Nested pool-separated normal-action references and partner-card residuals."""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '4'
from pathlib import Path
import hashlib
import json
import time
import importlib.metadata
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, mean_squared_error


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hand_features(frame, probability, strength_prediction):
    y = frame['action_class'].to_numpy()
    residual = (np.eye(4)[y] - probability) / np.sqrt(probability * (1 - probability) + .05)
    residual = np.clip(residual, -4, 4)
    partner = np.clip((frame['partner_strength'].to_numpy() - strength_prediction) / .25, -3, 3)
    info = residual * partner[:, None]
    values = {f'ca_residual_{i}': residual[:, i] for i in range(4)}
    values.update({f'ca_partner_cov_{i}': info[:, i] for i in range(4)})
    values['ca_surprise'] = -np.log(np.maximum(probability[np.arange(len(y)), y], 1e-5))
    values['ca_fold_benefit'] = residual[:, 0] * frame['facing_partner'].to_numpy() * frame['own_strength'].to_numpy() * (1 - frame['price'].to_numpy())
    values['ca_call_cost'] = residual[:, 2] * frame['facing_partner'].to_numpy() * (1-frame['own_strength'].to_numpy()) * frame['observed_log_amount_bb'].to_numpy()
    values['ca_weak_pressure'] = residual[:, 3] * (frame['players_active'].to_numpy() >= 3) * (1-frame['own_strength'].to_numpy())
    unseen = (frame['street_index'].to_numpy() == 0) & (frame['partner_prior_actions'].to_numpy() == 0)
    values.update({f'ca_preflop_info_{i}': info[:, i] * unseen for i in range(4)})
    assert all(np.isfinite(v).all() for v in values.values())
    data = frame.select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(n, v.astype(np.float32)) for n, v in values.items()])
    result = data.group_by('pair_id', 'hand_id', 'table_id').agg(pl.len().cast(pl.Float32).alias('ca_actions'),
        *[pl.col(n).mean().alias(n + '_mean') for n in values],
        *[pl.col(n).abs().max().alias(n + '_absmax') for n in values])
    return result


def main():
    root = Path(__file__).resolve().parent
    planpath = root / 'policy-plan.json'; plan = json.loads(planpath.read_text())
    assert all(sha(root / n) == h for n, h in plan['files'].items())
    out = root / 'policy-screen'; out.mkdir(exist_ok=True)
    labels = pl.read_csv(root / 'development_labels.csv').select('pair_id', 'label', 'label_status')
    assert set(labels.filter(pl.col('label') == 0)['label_status']) == {'confirmed_non_target'}
    data = pl.read_parquet(sorted((root / 'action-cache').glob('*.parquet'))).join(labels, on='pair_id', validate='m:1').sort('pair_id', 'hand_id', 'direction', 'action_no')
    pools = sorted(data['table_id'].unique())
    held = set(pl.read_csv(root / 'held-pairs.csv')['table_id'])
    training = np.array(sorted(set(pools)-held)); np.random.default_rng(39117).shuffle(training)
    inner = [set(s.tolist()) for s in np.array_split(training, 3)]
    specs = [(f'inner{i}', set(training)-part, part) for i, part in enumerate(inner)] + [('held', set(training), held)]
    completed = []
    for job, trainpools, scorepools in specs:
        destination = out / (job + '.parquet'); receipt = out / (job + '.json')
        if receipt.exists():
            saved = json.loads(receipt.read_text()); assert saved['plan_sha256'] == sha(planpath) and saved['feature_sha256'] == sha(destination)
            completed.append(saved); continue
        assert not destination.exists()
        start = time.perf_counter()
        train = data.filter((pl.col('label') == 0) & pl.col('table_id').is_in(sorted(trainpools)))
        rng = np.random.default_rng(55193)
        train = train.with_columns(pl.Series('sample_order', rng.random(train.height))).sort('pair_id', 'sample_order').group_by('pair_id', maintain_order=True).head(128)
        if train.height > 100000: train = train.sample(n=100000, seed=55193)
        scored = data.filter(pl.col('table_id').is_in(sorted(scorepools)))
        assert not set(train['table_id']) & set(scored['table_id']) and not set(train['table_id']) & held
        X = train.select(plan['inputs']).to_numpy().astype(np.float32)
        score_X = scored.select(plan['inputs']).to_numpy().astype(np.float32)
        classifier = lgb.LGBMClassifier(**plan['parameters'])
        regressor = lgb.LGBMRegressor(**plan['parameters'])
        classifier.fit(X, train['action_class'].to_numpy())
        regressor.fit(X, train['partner_strength'].to_numpy())
        assert np.array_equal(classifier.classes_, np.arange(4))
        probability = classifier.predict_proba(score_X)
        predicted_strength = regressor.predict(score_X)
        features = hand_features(scored, probability, predicted_strength)
        features.write_parquet(destination)
        modelpath = out / (job + '-models.joblib')
        joblib.dump({'classifier': classifier, 'regressor': regressor, 'inputs': plan['inputs']}, modelpath)
        mask = scored['label'].to_numpy() == 0
        saved = {'plan_sha256': sha(planpath), 'feature_sha256': sha(destination), 'model_sha256': sha(modelpath),
            'training_pools': sorted(trainpools), 'scored_pools': sorted(scorepools), 'training_actions': train.height,
            'scored_actions': scored.height, 'pair_hands': features.height, 'seconds': time.perf_counter()-start,
            'normal_action_log_loss': float(log_loss(scored['action_class'].to_numpy()[mask], probability[mask], labels=[0,1,2,3])),
            'normal_partner_strength_mse': float(mean_squared_error(scored['partner_strength'].to_numpy()[mask], predicted_strength[mask]))}
        receipt.write_text(json.dumps(saved, indent=2) + '\n'); completed.append(saved)
        print(json.dumps({k:v for k,v in saved.items() if k not in ['training_pools','scored_pools']}), flush=True)
    combined = pl.concat([pl.read_parquet(out / (j + '.parquet')) for j, _, _ in specs]).sort('pair_id', 'hand_id')
    assert combined.select(pl.struct('pair_id','hand_id').n_unique()).item() == combined.height
    assert combined['pair_id'].n_unique() == 1860
    combined.write_parquet(out / 'hand-features.parquet')
    report = {'completed': completed, 'rows': combined.height, 'features': [c for c in combined.columns if c.startswith('ca_')],
        'hand_features_sha256': sha(out / 'hand-features.parquet'), 'packages': {p: importlib.metadata.version(p) for p in ['numpy','polars','lightgbm']}}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'rows': combined.height, 'features': len(report['features'])}), flush=True)


if __name__ == '__main__': main()
