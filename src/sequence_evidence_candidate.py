"""Fit and prepare the exploratory v6 sequence-evidence entry; preserve v5."""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from rank_evidence import PARAMS, KEYS, FAMILIES, matrix, context, family_columns, assemble
from relative_evidence_experiment import load, fit, sha, ROOT
from sequence_evidence_fast import extract
from sequence_evidence_experiment import FEATURES

OUT = ROOT/'outputs_v6'
SOURCE = ROOT/'outputs_v5/submission_event_tails.csv'
RESEARCH = ROOT/'reports/sequence-evidence-experiment'
CODE = [Path(__file__), ROOT/'src/sequence_evidence_features.py', ROOT/'src/sequence_evidence_fast.py',
        ROOT/'src/rank_evidence.py', ROOT/'src/relative_evidence_experiment.py', ROOT/'src/sequence_evidence_experiment.py']


def identity(paths):
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def verify_source_cache(phase):
    if phase == 'development':
        reference = json.loads((ROOT/'outputs_v3/fit.json').read_text())['identity']['inputs']
    else:
        reference = json.loads((ROOT/'outputs_v3/rank_batches/manifest.json').read_text())['identity']['inputs']
    expected = {p: digest for p, digest in reference.items() if f'hands_{phase}' in p}
    assert len(expected) == 20
    for p, digest in expected.items():
        assert sha(ROOT/p) == digest, f'Existing validated raw cache changed: {p}'


def train():
    OUT.mkdir(exist_ok=True)
    if (OUT/'sequence_ranker.joblib').exists() or (OUT/'fit.json').exists():
        raise FileExistsError('Final fit already exists; preserve it')
    verify_source_cache('development')
    verification = json.loads((RESEARCH/'fast-verification.json').read_text())
    assert verification['passed'] and verification['different_float32_cells'] == 0
    for p, digest in verification['sha256'].items():
        assert sha(p) == digest
    inputs = CODE + [RESEARCH/'development.parquet', RESEARCH/'confirmation-plan.md',
        RESEARCH/'confirmation-v2/metrics.json', RESEARCH/'fast-verification.json', SOURCE,
        ROOT/'data/development_labels.csv', ROOT/'data/development_evidence.csv']
    inputs += sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet'))
    before = identity(inputs)
    data, current, _, _, _, _ = load()
    features = current+FEATURES+['context_'+f for f in FEATURES]
    data = context(data.join(pl.read_parquet(RESEARCH/'development.parquet'),on=['pair_id','hand_id'],validate='1:1'), FEATURES)
    model = fit(data, features)
    bundle = {'model': model, 'features': features, 'parameters': PARAMS, 'features_sequence': FEATURES,
              'raw_features': [c for c in current if not c.startswith(('context_','family_'))], 'identity': before}
    destination = OUT/'sequence_ranker.joblib'
    temp = destination.with_suffix('.joblib.tmp')
    joblib.dump(bundle, temp)
    check = family_columns(data.head(64), 'behavior_family')
    np.testing.assert_array_equal(model.predict(matrix(check, features)), joblib.load(temp)['model'].predict(matrix(check,features)))
    assert identity(inputs) == before
    temp.replace(destination)
    result = {'identity': before, 'model_sha256': sha(destination), 'features': len(features),
              'target_pairs': data['pair_id'].n_unique(), 'hands': data.height, 'parameters': PARAMS,
              'status': 'Exploratory entry: strict statistical confirmation failed; adaptive decision documented in confirmation-plan.md',
              'reload_predictions_identical': True, 'protected_v5_sha256': sha(SOURCE)}
    write_json(OUT/'fit.json', result)
    print(json.dumps(result), flush=True)


def predict():
    started = time.perf_counter()
    verify_source_cache('evaluation')
    meta = json.loads((OUT/'fit.json').read_text())
    model_path = OUT/'sequence_ranker.joblib'
    assert sha(model_path) == meta['model_sha256'] and sha(SOURCE) == meta['protected_v5_sha256']
    for p in CODE:
        assert sha(p) == meta['identity'][p.relative_to(ROOT).as_posix()]
    bundle = joblib.load(model_path)
    source = pl.read_csv(SOURCE, infer_schema=False)
    pairs = pl.read_csv(ROOT/'data/evaluation_pairs.csv')
    raw_paths = sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet'))
    paths = CODE+raw_paths+[model_path, OUT/'fit.json', SOURCE, ROOT/'data/evaluation_pairs.csv']
    paths += [ROOT/'data'/f for f in ['hands.parquet','seats.parquet','actions.parquet']]
    fingerprints = identity(paths)
    cache = OUT/'rank_batches'
    cache.mkdir(exist_ok=True)
    manifest_path = cache/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['identity'] == fingerprints
        assert set(manifest['files']) == {p.name for p in cache.glob('*.parquet')}
        for name, digest in manifest['files'].items():
            assert sha(cache/name) == digest
    else:
        assert not list(cache.glob('*.parquet'))
        manifest = {'identity': fingerprints, 'files': {}}
        write_json(manifest_path, manifest)
    for index, path in enumerate(raw_paths):
        destination = cache/path.name
        if destination.exists():
            print(f'{path.name}: verified cached ranking', flush=True)
            continue
        raw = pl.read_parquet(path)
        assert [c for c in raw.columns if c not in KEYS] == bundle['raw_features']
        keys = raw.select(KEYS).join(pairs.select('pair_id','player_1','player_2'),on='pair_id',validate='m:1')
        ids = keys.select('hand_id').unique().lazy()
        hands = pl.scan_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='evaluation').join(ids,on='hand_id').collect()
        assert hands.height == keys['hand_id'].n_unique()
        seats = pl.scan_parquet(ROOT/'data/seats.parquet').join(ids,on='hand_id').collect()
        actions = pl.scan_parquet(ROOT/'data/actions.parquet').join(ids,on='hand_id').collect()
        pools = sorted(raw['table_id'].unique().to_list())
        pieces = []
        for offset in range(0,len(pools),5):
            chosen = pools[offset:offset+5]
            part = raw.filter(pl.col('table_id').is_in(chosen))
            pk = keys.filter(pl.col('table_id').is_in(chosen))
            ph = hands.filter(pl.col('table_id').is_in(chosen))
            phids = ph.select('hand_id')
            seq = extract(pk, ph, seats.join(phids,on='hand_id'), actions.join(phids,on='hand_id'))
            data = part.join(seq,on=['pair_id','hand_id'],validate='1:1').join(source.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id')
            assert data.height == part.height
            data = family_columns(context(data, [c for c in bundle['raw_features'] if c.startswith('v2_')]+FEATURES), 'predicted_behavior')
            scores = bundle['model'].predict(matrix(data,bundle['features']),num_threads=2)
            assert np.isfinite(scores).all()
            scored = data.select('pair_id','hand_id').with_columns(pl.Series('score',scores),
                pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(data.height)))
            ranking = scored.sort(['pair_id','score','tie'],descending=[False,True,False]).group_by('pair_id',maintain_order=True).agg(
                pl.col('hand_id').head(5).alias('hands'),pl.len().alias('shared_hands'))
            pieces.append(ranking)
            print(f'{path.name}: pools {offset+len(chosen)}/{len(pools)}, {data.height} hands ranked',flush=True)
            del part, pk, ph, phids, seq, data, scores, scored, ranking
            gc.collect()
        result = pl.concat(pieces)
        expected = raw.group_by('pair_id').len().rename({'len':'expected_hands'})
        assert not result.join(expected,on='pair_id').filter(pl.col('shared_hands')!=pl.col('expected_hands')).height
        temp = destination.with_suffix('.parquet.tmp')
        result.write_parquet(temp); temp.replace(destination)
        manifest['files'][destination.name] = sha(destination)
        write_json(manifest_path,manifest)
        print(f'Completed {index+1}/20 evaluation batches in {time.perf_counter()-started:.1f}s',flush=True)
        del raw, keys, hands, seats, actions, pieces, result, expected
        gc.collect()
    assert len(manifest['files']) == 20 and identity(paths) == fingerprints
    ranked = pl.read_parquet([cache/p.name for p in raw_paths])
    result = assemble(source,ranked,pairs)
    destination = OUT/'submission_sequence_evidence.csv'
    if destination.exists():
        raise FileExistsError('Preserve completed candidate CSV')
    result.write_csv(destination)
    fixed = ['pair_id','risk_score','predicted_behavior']
    assert pl.read_csv(destination,infer_schema=False).select(fixed).equals(source.select(fixed))
    write_json(OUT/'submission_sequence_evidence.provenance.json',{'identity': fingerprints,'submission_sha256':sha(destination),
        'rows':result.height,'preserved_exactly':fixed,'risk_and_behavior_unchanged':True,'all_batches_complete':True,
        'membership_validation_required':True,'seconds':time.perf_counter()-started})
    print(f'Wrote {destination}; full membership validation remains required.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','predict'])
    args=parser.parse_args()
    train() if args.mode=='train' else predict()
