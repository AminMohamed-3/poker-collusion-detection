"""Cache evidence for each behavior, then assemble a validated candidate entry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from train import FAMILIES, matrix

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA = ROOT/'outputs_v2', ROOT/'data'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    cache = OUT/'evidence_rankings'
    cache.mkdir(exist_ok=True)
    manifest = cache/'manifest.json'
    identity = {'code': digest(__file__), 'hand_model': digest(OUT/'hand_model.joblib'),
                'features': digest(OUT/'hands_evaluation'/'manifest.json'), 'tie_seed': 843177}
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Evidence cache mismatch; move old cache aside before recomputing')
    else:
        if list(cache.glob('*.parquet')):
            raise ValueError('Untracked evidence cache')
        manifest.write_text(json.dumps(identity, indent=2)+'\n')
    bundle = joblib.load(OUT/'hand_model.joblib')
    paths = sorted((OUT/'hands_evaluation').glob('batch_*.parquet'))
    for batch, path in enumerate(paths):
        dest = cache/path.name
        if dest.exists():
            print(f'evidence {batch}: cached', flush=True)
            continue
        frame = pl.read_parquet(path)
        prob = bundle['model'].predict_proba(matrix(frame, bundle['features']))
        # Uniform independent tie-breaking avoids using identifier values to rank.
        rng = np.random.default_rng(843177+batch)
        ranking = frame.select('pair_id', 'hand_id').with_columns(
            pl.Series('tie', rng.random(frame.height)),
            *[pl.Series(f'prob_{family}', prob[:, index])
              for index, family in enumerate(FAMILIES) if index])
        result = ranking.group_by('pair_id').agg(
            *[pl.col('hand_id').sort_by([f'prob_{family}', 'tie'], descending=[True, False])
              .head(5).alias(family) for family in FAMILIES[1:]])
        result.write_parquet(dest)
        print(f'evidence {batch}: ranked {frame.height} pair-hands for all three behaviors', flush=True)
        del frame, prob, ranking, result


def assemble(pair_model, output_name):
    bundle = joblib.load(pair_model)
    pairs = pl.read_parquet(OUT/'pairs_evaluation.parquet')
    prob = bundle['model'].predict_proba(matrix(pairs, bundle['features']))
    family = np.argmax(prob[:, 1:], axis=1)+1
    predictions = pairs.select('pair_id').with_columns(
        pl.Series('risk_score', 1-prob[:, 0]),
        pl.Series('predicted_behavior', [FAMILIES[k] for k in family]))
    paths = sorted((OUT/'hands_evaluation').glob('batch_*.parquet'))
    required = [OUT/'evidence_rankings'/p.name for p in paths]
    if not all(p.exists() for p in required):
        raise ValueError('Evidence ranking is incomplete')
    ev = pl.read_parquet(required)
    if ev['pair_id'].n_unique() != ev.height:
        raise ValueError('Evidence pairs occur in multiple batches')
    predictions = predictions.join(ev, on='pair_id', how='left').with_columns(
        pl.when(pl.col('predicted_behavior') == FAMILIES[1]).then(pl.col(FAMILIES[1]))
        .when(pl.col('predicted_behavior') == FAMILIES[2]).then(pl.col(FAMILIES[2]))
        .otherwise(pl.col(FAMILIES[3])).alias('chosen'))
    predictions = predictions.with_columns(
        *[pl.col('chosen').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE')
          .alias(f'evidence_hand_{i+1}') for i in range(5)])
    template = pl.read_csv(DATA/'sample_submission.csv')
    result = template.select('pair_id').join(predictions, on='pair_id', how='left').select(template.columns)
    if result.height != template.height or result['risk_score'].null_count():
        raise ValueError('Prediction pair coverage mismatch')
    if Path(output_name).name != output_name:
        raise ValueError('Output name must be a filename inside outputs_v2')
    dest = OUT/output_name
    result.write_csv(dest)
    (OUT/(output_name+'.provenance.json')).write_text(json.dumps({
        'sha256': digest(dest), 'rows': result.height, 'pair_model': str(pair_model),
        'pair_model_sha256': digest(pair_model), 'hand_model_sha256': digest(OUT/'hand_model.joblib'),
        'evidence_manifest_sha256': digest(OUT/'evidence_rankings'/'manifest.json'),
    }, indent=2)+'\n')
    print(f'wrote {result.height} rows to {dest}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evidence-only', action='store_true')
    p.add_argument('--pair-model', type=Path, default=OUT/'pair_model.joblib')
    p.add_argument('--output-name', default='submission_advanced.csv')
    args = p.parse_args()
    evidence()
    if not args.evidence_only:
        assemble(args.pair_model, args.output_name)
