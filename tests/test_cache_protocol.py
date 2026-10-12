"""Regression checks for stale cache reuse and explicit legacy adoption."""
from copy import deepcopy
import polars as pl
import pytest

from src import features


@pytest.fixture
def legacy(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    pairs = pl.DataFrame({'pair_id': ['p1', 'p2'], 'table_id': ['t1', 't2']})
    for i in range(2):
        pl.DataFrame({'pair_id': [f'p{i + 1}'], 'hand_id': [f'h{i + 1}'],
                      'table_id': [f't{i + 1}'], 'feature': [1.0]}).write_parquet(
            cache / f'batch_{i:03}.parquet')
    identity = {'phase': 'development', 'batch_size': 1, 'code_sha256': 'code-v1',
                'inputs': {'hands.parquet': 'data-v1'}, 'polars_version': pl.__version__,
                'batches': {'batch_000.parquet': ['t1'], 'batch_001.parquet': ['t2']}}
    return cache, identity, pairs


def adopt(legacy):
    cache, identity, pairs = legacy
    return features.prepare_cache(cache, identity, pairs, adopt=True,
                                  note='Independent feature review completed')


def test_legacy_cache_requires_explicit_adoption(legacy):
    cache, identity, pairs = legacy
    with pytest.raises(ValueError, match='no manifest'):
        features.prepare_cache(cache, identity, pairs)
    with pytest.raises(ValueError, match='nonempty'):
        features.prepare_cache(cache, identity, pairs, adopt=True)
    assert not (cache / 'manifest.json').exists()


def test_adoption_preserves_batches_and_records_review(legacy):
    cache, identity, pairs = legacy
    before = {p.name: (features.file_sha256(p), p.stat().st_mtime_ns)
              for p in cache.glob('*.parquet')}
    manifest = adopt(legacy)
    assert manifest['provenance']['mode'] == 'adopted_existing'
    assert manifest['provenance']['verification']['pair_hands'] == 2
    assert manifest['provenance']['note'] == 'Independent feature review completed'
    assert features.prepare_cache(cache, identity, pairs, check=True) == manifest
    after = {p.name: (features.file_sha256(p), p.stat().st_mtime_ns)
             for p in cache.glob('*.parquet')}
    assert before == after


@pytest.mark.parametrize('field,value', [
    ('batch_size', 20), ('code_sha256', 'code-v2'),
    ('inputs', {'hands.parquet': 'data-v2'}), ('polars_version', 'different-version'),
])
def test_incompatible_cache_refused_without_manifest_overwrite(legacy, field, value):
    cache, identity, pairs = legacy
    adopt(legacy)
    before = (cache / 'manifest.json').read_bytes()
    changed = deepcopy(identity)
    changed[field] = value
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        features.prepare_cache(cache, changed, pairs)
    with pytest.raises(ValueError, match='cannot override'):
        features.prepare_cache(cache, changed, pairs, adopt=True, note='Bypass attempt')
    assert (cache / 'manifest.json').read_bytes() == before


def test_unexpected_stale_batch_refused(legacy):
    cache, identity, pairs = legacy
    adopt(legacy)
    (cache / 'batch_099.parquet').write_bytes((cache / 'batch_000.parquet').read_bytes())
    with pytest.raises(ValueError, match='Unexpected Parquet'):
        features.prepare_cache(cache, identity, pairs)


def test_altered_cache_refused(legacy):
    cache, identity, pairs = legacy
    adopt(legacy)
    batch = cache / 'batch_000.parquet'
    pl.read_parquet(batch).with_columns(pl.lit(99.0).alias('feature')).write_parquet(batch)
    with pytest.raises(ValueError, match='content hash mismatch'):
        features.prepare_cache(cache, identity, pairs)


@pytest.mark.parametrize('problem,match', [
    ('layout', 'table membership'), ('duplicate', 'duplicate pair-hand'),
    ('coverage', 'pair/table coverage'),
])
def test_invalid_legacy_cache_cannot_be_adopted(legacy, problem, match):
    cache, identity, pairs = legacy
    batch = cache / 'batch_000.parquet'
    frame = pl.read_parquet(batch)
    if problem == 'layout':
        frame = frame.with_columns(pl.lit('wrong-table').alias('table_id'))
    elif problem == 'duplicate':
        frame = pl.concat([frame, frame])
    else:
        frame = frame.with_columns(pl.lit('unknown-pair').alias('pair_id'))
    frame.write_parquet(batch)
    with pytest.raises(ValueError, match=match):
        adopt(legacy)
    assert not (cache / 'manifest.json').exists()


def test_evaluation_adoption_checks_supplied_shared_hand_counts(legacy, tmp_path, monkeypatch):
    cache, identity, pairs = legacy
    identity['phase'] = 'evaluation'
    monkeypatch.setattr(features, 'DATA', tmp_path)
    pairs.with_columns(pl.lit(2).alias('shared_hands')).write_csv(tmp_path / 'evaluation_pairs.csv')
    with pytest.raises(ValueError, match='shared-hand counts differ'):
        adopt(legacy)
    assert not (cache / 'manifest.json').exists()


def test_empty_cache_can_start_but_cannot_pass_complete_check(tmp_path):
    identity = {'batches': {'batch_000.parquet': ['t1']}}
    pairs = pl.DataFrame({'pair_id': ['p1'], 'table_id': ['t1']})
    manifest = features.prepare_cache(tmp_path, identity, pairs)
    assert manifest['provenance']['mode'] == 'generated'
    assert manifest['files'] == {}
    with pytest.raises(ValueError, match='incomplete'):
        features.prepare_cache(tmp_path, identity, pairs, check=True)


def test_identity_changes_when_source_data_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(features, 'DATA', tmp_path)
    for name in ['hands.parquet', 'seats.parquet', 'actions.parquet', 'development_labels.csv']:
        (tmp_path / name).write_bytes(b'initial content')
    before = features.cache_identity('development', 20, ['t1'])
    (tmp_path / 'actions.parquet').write_bytes(b'changed action content')
    after = features.cache_identity('development', 20, ['t1'])
    assert before['inputs']['actions.parquet'] != after['inputs']['actions.parquet']
    assert before['code_sha256'] == after['code_sha256']
