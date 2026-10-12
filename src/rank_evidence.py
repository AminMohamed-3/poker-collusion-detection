"""Production evidence ranker in outputs_v3; preserves v2 pair-risk/class cells."""
from __future__ import annotations
import os
os.environ['POLARS_MAX_THREADS'] = '4'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'

import argparse
import gc
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from features import file_sha256 as sha, prepare_cache, write_manifest
from validation import EVIDENCE_COLUMNS, SUBMISSION_COLUMNS, NO_EVIDENCE

ROOT = Path(__file__).resolve().parents[1]
DATA, BASE, OUT = ROOT/'data', ROOT/'outputs_v2', ROOT/'outputs_v3'
SOURCE = BASE/'submission_advanced_robust.csv'
FAMILIES = ['directed_transfer', 'soft_play', 'coordinated_isolation']
KEYS = ['pair_id', 'hand_id', 'table_id']
PARAMS = dict(objective='lambdarank', n_estimators=238, learning_rate=.03,
              num_leaves=15, min_child_samples=20, colsample_bytree=.8, reg_lambda=5,
              n_jobs=2, verbosity=-1, random_state=7329, deterministic=True,
              force_col_wise=True, lambdarank_truncation_level=10, label_gain=[0, 1])


def paths(phase):
    found = sorted((BASE/f'hands_{phase}').glob('*.parquet'))
    if [p.name for p in found] != [f'batch_{i:03}.parquet' for i in range(20)]:
        raise ValueError(f'{phase}: expected exactly the 20 complete advanced batches')
    return found


def fingerprint(files):
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def code_identity():
    return fingerprint([Path(__file__).resolve(), ROOT/'src/features.py'])


def context(frame, events):
    # Identical ddof=1, denominator floor, clipping, and Float32 transform to experiment.
    return frame.with_columns(*[
        ((pl.col(c)-pl.col(c).mean().over('pair_id')) /
         pl.col(c).std().over('pair_id').fill_null(0).clip(lower_bound=.001))
        .clip(-20, 20).fill_null(0).cast(pl.Float32).alias('context_'+c) for c in events])


def family_columns(frame, column):
    return frame.with_columns(*[(pl.col(column) == family).cast(pl.Float32)
                               .alias('family_'+family) for family in FAMILIES])


def matrix(frame, columns):
    values = frame.select(columns).to_numpy(order='c').astype(np.float32, copy=False)
    return np.nan_to_num(values, copy=False, nan=0, posinf=0, neginf=0)


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def train():
    raw_paths = paths('development')
    inputs = fingerprint([DATA/'development_labels.csv', DATA/'development_evidence.csv', *raw_paths])
    identity = {'code': code_identity(), 'inputs': inputs, 'parameters': PARAMS,
                'polars': pl.__version__, 'numpy': np.__version__, 'lightgbm': lgb.__version__}
    OUT.mkdir(exist_ok=True)
    artifact, meta_path = OUT/'evidence_ranker.joblib', OUT/'fit.json'
    if artifact.exists() or meta_path.exists():
        if not artifact.exists() or not meta_path.exists():
            raise ValueError('Untracked/incomplete ranker fit; preserve it for review and use a fresh output directory')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['identity'] != identity or meta['model_sha256'] != sha(artifact):
            raise ValueError('Existing ranker identity mismatch; refusing replacement')
        print('Existing ranker fit verified; no training or overwrite performed.', flush=True)
        return
    labels = pl.read_csv(DATA/'development_labels.csv').filter(pl.col('label') == 1)
    if (labels.height != 372 or labels['pair_id'].n_unique() != 372 or
            set(labels['label_status']) != {'confirmed_target'} or
            not set(labels['behavior_family']).issubset(FAMILIES)):
        raise ValueError('Expected exactly 372 explicit target pairs')
    evidence = pl.read_csv(DATA/'development_evidence.csv').select('pair_id', 'hand_id')
    if (evidence.unique().height != evidence.height or
            evidence.join(labels.select('pair_id'), on='pair_id', how='anti').height):
        raise ValueError('Duplicate evidence or evidence outside explicit target pairs')
    chunks = [pl.read_parquet(p).join(labels.select('pair_id', 'behavior_family'), on='pair_id')
              for p in raw_paths]
    raw = pl.concat(chunks)
    if raw['pair_id'].n_unique() != 372 or raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() != raw.height:
        raise ValueError('Incomplete target coverage or duplicate pair-hand rows')
    base = [c for c in raw.columns if c not in KEYS+['behavior_family']]
    events = [c for c in base if c.startswith('v2_')]
    features = base+['context_'+c for c in events]+['family_'+f for f in FAMILIES]
    if (len(base), len(events), len(features)) != (163, 56, 222):
        raise ValueError('Expected 163 raw + 56 context + 3 family features')
    data = family_columns(context(raw, events), 'behavior_family').join(
        evidence.with_columns(pl.lit(1).alias('relevance')), on=['pair_id', 'hand_id'], how='left')
    data = data.with_columns(pl.col('relevance').fill_null(0)).sort('pair_id', 'hand_id')
    grouped = data.group_by('pair_id', maintain_order=True).agg(pl.len().alias('n'), pl.col('relevance').sum().alias('listed'))
    if grouped['listed'].min() < 1 or int(data['relevance'].sum()) != evidence.height:
        raise ValueError('Every target needs matched listed evidence; annotations may not be lost')
    X = matrix(data, features)
    model = lgb.LGBMRanker(**PARAMS)
    model.fit(X, data['relevance'].to_numpy(), group=grouped['n'].to_numpy())
    if model.n_features_in_ != 222 or model.booster_.current_iteration() != 238:
        raise ValueError('Unexpected fitted ranker dimensions')
    if fingerprint([ROOT/p for p in inputs]) != inputs or code_identity() != identity['code']:
        raise ValueError('Inputs changed during final fit')
    fit = {'identity': identity, 'target_pairs': grouped.height, 'hands': data.height,
           'listed_relevance_one': evidence.height, 'unannotated_relevance_zero': data.height-evidence.height,
           'tables': data['table_id'].n_unique(), 'features': len(features),
           'relevance_semantics': '1=listed evidence; 0=unannotated/noisy within-pair preference, not a confirmed negative',
           'all_explicit_target_pairs_included': True, 'unlisted_pairs': 0}
    bundle = {'model': model, 'features': features, 'raw_features': base, 'events': events, 'fit': fit}
    temp = artifact.with_suffix('.joblib.tmp')
    joblib.dump(bundle, temp)
    reloaded = joblib.load(temp)
    np.testing.assert_allclose(model.predict(X[:64]), reloaded['model'].predict(X[:64]), rtol=0, atol=1e-12)
    temp.replace(artifact)
    write_json(meta_path, {**fit, 'model_sha256': sha(artifact), 'reload_predictions_identical': True})
    print(json.dumps({k: fit[k] for k in ['target_pairs', 'hands', 'listed_relevance_one', 'unannotated_relevance_zero', 'features']}), flush=True)


def assemble(source, ranking, expected):
    if ranking['pair_id'].n_unique() != ranking.height:
        raise ValueError('Pair occurs in multiple cached batches')
    if ranking.join(source.select('pair_id'), on='pair_id', how='anti').height:
        raise ValueError('Unknown pair in rankings')
    counts = expected.select('pair_id', 'shared_hands').join(
        ranking.select('pair_id', pl.col('shared_hands').alias('observed')), on='pair_id', how='left')
    if counts.filter(pl.col('shared_hands') != pl.col('observed').fill_null(0)).height:
        raise ValueError('Incomplete pair-history coverage; refusing assembly')
    for hands in ranking['hands']:
        if len(hands) > 5 or len(set(hands)) != len(hands) or any(not h or h == NO_EVIDENCE for h in hands):
            raise ValueError('Invalid or duplicate actual evidence IDs')
    expanded = ranking.select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True)
        .fill_null(NO_EVIDENCE).alias(name) for i, name in enumerate(EVIDENCE_COLUMNS)])
    fixed = [c for c in source.columns if c not in EVIDENCE_COLUMNS]
    result = source.select(fixed).join(expanded, on='pair_id', how='left', maintain_order='left')
    result = result.with_columns(pl.col(EVIDENCE_COLUMNS).fill_null(NO_EVIDENCE)).select(source.columns)
    if result.height != source.height or not result.select(fixed).equals(source.select(fixed)):
        raise ValueError('Non-evidence cells changed')
    return result


def predict():
    raw_paths = paths('evaluation')
    artifact, meta_path = OUT/'evidence_ranker.joblib', OUT/'fit.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['model_sha256'] != sha(artifact) or meta['identity']['code'] != code_identity():
        raise ValueError('Ranker/model source identity mismatch')
    bundle = joblib.load(artifact)
    if bundle['fit']['identity'] != meta['identity']:
        raise ValueError('Model provenance differs from fit record')
    source = pl.read_csv(SOURCE, infer_schema=False)
    expected = pl.read_csv(DATA/'evaluation_pairs.csv', schema_overrides={'pair_id': pl.String})
    if (source.columns != SUBMISSION_COLUMNS or source['pair_id'].n_unique() != source.height or
            source.height != expected.height or set(source['pair_id']) != set(expected['pair_id'])):
        raise ValueError('Source submission pair coverage/schema is invalid')
    if not set(source['predicted_behavior']).issubset(FAMILIES):
        raise ValueError('Source behavior labels must be one of the three trained families')
    files = [SOURCE, DATA/'evaluation_pairs.csv', artifact, meta_path, *raw_paths]
    identity = {'code': code_identity(), 'inputs': fingerprint(files), 'tie_seed_base': 931207,
                'polars': pl.__version__, 'numpy': np.__version__, 'lightgbm': lgb.__version__,
                'batches': {p.name: sha(p) for p in raw_paths}}
    cache = OUT/'rank_batches'
    cache.mkdir(exist_ok=True)
    manifest = prepare_cache(cache, identity, None)
    for index, path in enumerate(raw_paths):
        dest = cache/path.name
        if dest.exists():
            print(f'{path.name}: verified cached rankings', flush=True)
            continue
        raw = pl.read_parquet(path)
        if [c for c in raw.columns if c not in KEYS] != bundle['raw_features']:
            raise ValueError('Evaluation feature schema differs from fitted model')
        data = raw.join(source.select('pair_id', 'predicted_behavior'), on='pair_id').sort('pair_id', 'hand_id')
        if data.height != raw.height:
            raise ValueError('Raw evaluation pair missing from source submission')
        data = family_columns(context(data, bundle['events']), 'predicted_behavior')
        score = bundle['model'].predict(matrix(data, bundle['features']), num_threads=2)
        if not np.isfinite(score).all():
            raise ValueError('Ranker returned non-finite scores')
        scored = data.select('pair_id', 'hand_id').with_columns(pl.Series('score', score),
            pl.Series('tie', np.random.default_rng(931207+index).random(data.height)))
        top = scored.sort(['pair_id', 'score', 'tie'], descending=[False, True, False])
        top = top.group_by('pair_id', maintain_order=True).agg(pl.col('hand_id').head(5).alias('hands'),
                                                             pl.len().alias('shared_hands'))
        temp = dest.with_suffix('.parquet.tmp')
        top.write_parquet(temp)
        temp.replace(dest)
        manifest['files'][dest.name] = sha(dest)
        write_manifest(cache, manifest)
        print(f'{path.name}: ranked {data.height} hands for {top.height} pairs', flush=True)
        del raw, data, score, scored, top
        gc.collect()
    prepare_cache(cache, identity, None, check=True)  # incomplete caches cannot assemble
    if fingerprint(files) != identity['inputs']:
        raise ValueError('Prediction inputs changed during ranking')
    ranking = pl.read_parquet([cache/p.name for p in raw_paths])
    result = assemble(source, ranking, expected)
    dest, provenance_path = OUT/'submission_evidence_ranker.csv', OUT/'submission_evidence_ranker.provenance.json'
    if dest.exists() or provenance_path.exists():
        if not dest.exists() or not provenance_path.exists():
            raise ValueError('Untracked/incomplete submission output')
        old = json.loads(provenance_path.read_text(encoding='utf-8'))
        if old['identity'] != identity or old['submission_sha256'] != sha(dest):
            raise ValueError('Existing submission differs; refusing overwrite')
        print('Existing ranked submission verified; no overwrite performed.', flush=True)
        return
    temp = dest.with_suffix('.csv.tmp')
    result.write_csv(temp)
    reread = pl.read_csv(temp, infer_schema=False)
    fixed = [c for c in source.columns if c not in EVIDENCE_COLUMNS]
    if not reread.select(fixed).equals(source.select(fixed)) or reread.null_count().sum_horizontal().sum():
        raise ValueError('CSV round trip altered fixed cells or introduced empty values')
    temp.replace(dest)
    write_json(provenance_path, {'identity': identity, 'submission_sha256': sha(dest),
        'pairs': result.height, 'preserved_columns': fixed, 'non_evidence_cells_exact': True,
        'all_batches_complete': True, 'full_membership_validation_required': True})
    print(f'Wrote {dest}; risk/class cells preserved exactly. Run the full submission validator before submission.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    train() if args.mode == 'train' else predict()
