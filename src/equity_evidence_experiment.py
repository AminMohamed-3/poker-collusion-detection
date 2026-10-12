"""Frozen 24 equity proxies plus 24 pair contexts, versus the unchanged v6 ranker."""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import polars as pl

from evidence_capacity_experiment import load_sequence, fit_variant, compare
from equity_evidence_features import CARDS, FEATURES, PREFLOP_SAMPLES, SAMPLE_SEED, extract_pair_hand, _canonical_equity
from rank_evidence import PARAMS, context, family_columns, matrix
from relative_evidence_experiment import ROOT, sha, records, summarize

REPORT = ROOT / 'reports/equity-evidence-experiment'


def build(keys, destination):
    pairs = pl.read_csv(ROOT / 'data/development_labels.csv').select('pair_id', 'player_1', 'player_2')
    keys = keys.join(pairs, on='pair_id', validate='m:1').sort('hand_id', 'pair_id')
    ids = keys.select('hand_id').unique()
    hands = pl.scan_parquet(ROOT / 'data/hands.parquet').filter(pl.col('phase') == 'development').join(ids.lazy(), on='hand_id').select('hand_id', 'big_blind', 'board_cards').collect()
    hand_lookup = {h: (bb, [CARDS[c] for c in (board or '').split()]) for h, bb, board in hands.iter_rows()}
    assert hands.height == ids.height
    seats = pl.scan_parquet(ROOT / 'data/seats.parquet').join(ids.lazy(), on='hand_id').select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').collect()
    seat_lookup = {}
    for h, p, c1, c2 in seats.iter_rows():
        seat_lookup.setdefault(h, {})[p] = (CARDS[c1], CARDS[c2])
    actions = pl.scan_parquet(ROOT / 'data/actions.parquet').join(ids.lazy(), on='hand_id').select(
        'hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call', 'pot_before', 'stack_before').collect()
    action_lookup = {}
    for row in actions.iter_rows(named=True):
        h = row.pop('hand_id')
        action_lookup.setdefault(h, []).append(row)
    rows = []
    start = time.perf_counter()
    for i, (pair, hand, first, second) in enumerate(keys.select('pair_id', 'hand_id', 'player_1', 'player_2').iter_rows()):
        blind, board = hand_lookup[hand]
        values = extract_pair_hand(first, second, blind, seat_lookup[hand], board, action_lookup.get(hand, []))
        rows.append((pair, hand, *values))
        if (i + 1) % 5000 == 0:
            print(json.dumps({'extracted': i + 1, 'total': keys.height, 'seconds': time.perf_counter() - start}), flush=True)
    result = pl.DataFrame(rows, schema=['pair_id', 'hand_id', *FEATURES], orient='row').with_columns(pl.col(FEATURES).cast(pl.Float32))
    assert result.height == keys.height
    result.write_parquet(destination)
    print(json.dumps({'extraction_seconds': time.perf_counter() - start, 'equity_cache': _canonical_equity.cache_info()._asdict()}), flush=True)
    return result


def main():
    REPORT.mkdir(parents=True, exist_ok=False)
    protected = ['outputs_v6/submission_sequence_evidence.csv', 'outputs_v6/sequence_ranker.joblib',
                 'outputs_v5/submission_event_tails.csv', 'outputs_v5/pair_model_event_tails.joblib']
    fingerprints = {p: sha(ROOT / p) for p in protected}
    inputs = ['data/hands.parquet', 'data/seats.parquet', 'data/actions.parquet',
              'data/development_labels.csv', 'data/development_evidence.csv',
              'src/equity_evidence_features.py', 'src/equity_evidence_experiment.py']
    input_hashes = {p: sha(ROOT / p) for p in inputs}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': fingerprints, 'inputs': input_hashes,
            'hypothesis': 'True relative hand equity and pot price may distinguish harmful transfers from legitimate draws and folds.',
            'features': FEATURES, 'context': 'Add within-pair z-score for all 24 equity features to the v6 268 features.',
            'equity': {'preflop_runouts': PREFLOP_SAMPLES, 'seed': SAMPLE_SEED, 'postflop': 'exact over all legal future boards'},
            'parameters': PARAMS, 'scope': 'Reused fold0, two fits; no independent confirmation claims.',
            'selection': 'Advance unchanged candidate if held MAP improves by at least 0.005 and no family declines by more than 0.025.',
            'limitations': ['Uniform HU runouts against actual partner hole cards; not a player belief or range model.',
                           'Multiway and side-pot expected value are not modeled.', 'Unannotated hands remain unknown.']}
    (REPORT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    started = time.perf_counter()
    data, current, truths, pairpred, held_tables = load_sequence()
    extra = build(data.select('pair_id', 'hand_id'), REPORT / 'development.parquet')
    data = context(data.join(extra, on=['pair_id', 'hand_id'], validate='1:1'), FEATURES)
    candidate = current + FEATURES + ['context_' + c for c in FEATURES]
    train = data.filter(~pl.col('table_id').is_in(held_tables))
    held = data.filter(pl.col('table_id').is_in(held_tables)).join(pairpred.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
    assert not set(train['table_id']) & set(held['table_id'])
    results, rows = [], []
    for name, features in [('v6_control', current), ('equity', candidate)]:
        t0 = time.perf_counter()
        model = fit_variant(train, features, PARAMS)
        scores = model.predict(matrix(family_columns(held, 'predicted_behavior'), features), num_threads=2)
        selected = records(held, scores, truths, 'reused_fold0', name)
        rows.extend(selected)
        result = {'candidate': name, 'features': len(features), 'seconds': time.perf_counter() - t0, **summarize(selected)}
        results.append(result)
        held.select('pair_id', 'hand_id').with_columns(pl.Series('score', scores)).write_parquet(REPORT / f'{name}-scores.parquet')
        joblib.dump({'model': model, 'features': features}, REPORT / f'{name}.joblib')
        print(json.dumps(result), flush=True)
    comparison = compare(rows, 'equity')
    selected = comparison['delta'] >= .005 and min(comparison['by_family_delta'].values()) >= -.025
    assert fingerprints == {p: sha(ROOT / p) for p in protected}
    assert input_hashes == {p: sha(ROOT / p) for p in inputs}
    payload = {'results': results, 'comparison': comparison, 'advance': bool(selected), 'protected_unchanged': True,
               'feature_cache_sha256': sha(REPORT / 'development.parquet'), 'seconds': time.perf_counter() - started}
    pl.DataFrame(rows).write_csv(REPORT / 'pairs.csv')
    (REPORT / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
