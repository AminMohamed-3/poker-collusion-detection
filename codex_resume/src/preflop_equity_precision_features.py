"""Write-once precision reconstruction for the frozen 24 public equity inputs.

Run only after review. The controller uses isolated worker processes so the
original equity module is never edited or reused across the 256/4096 regimes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CV = ROOT / 'Claude Version'
ROOT_ORIGINAL = ROOT / 'src' / 'equity_evidence_features.py'
CLAUDE_ORIGINAL = CV / 'src' / 'equity_evidence_features.py'
# The root source is authoritative.  The frozen Claude overlay must be byte-identical.
ORIGINAL = ROOT_ORIGINAL
OUT = HERE / 'research' / 'preflop_equity_precision_20260912'
PREV = HERE / 'experiments' / 'symmetry_censor_20260912'
DATA = ROOT / 'data'
SAMPLE_SEED = 408331
BASE_SAMPLES, HIGH_SAMPLES = 256, 4096

NAMES = [
    'partner_call_deficit_bb', 'partner_call_surplus_bb', 'partner_call_equity_cost_bb',
    'partner_fold_surplus_pot_bb', 'partner_fold_equity', 'partner_fold_underdog',
    'late_check_equity', 'heads_up_check_equity', 'partner_raise_equity_cost_bb',
    'multiway_raise_low_equity_logcost', 'partner_call_equity_drop_bb', 'partner_fold_equity_gain',
]
FEATURES = ['eq_' + name + suffix for suffix in ['_sum', '_max'] for name in NAMES]
assert len(FEATURES) == 24


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sample_indices(count):
    return np.argsort(np.random.default_rng(SAMPLE_SEED).random((count, 48)), axis=1)[:, :5]


def current_keys():
    saved = pl.read_parquet(PREV / 'scores.parquet').sort('pair_id', 'hand_id')
    ranks = saved.with_columns(pl.col('stage1_only').rank('ordinal', descending=True).over('pair_id').alias('_rank'))
    keys = ranks.filter(pl.col('_rank') <= 20).select('pair_id', 'hand_id').sort('pair_id', 'hand_id')
    assert keys.height == 3240 and keys['pair_id'].n_unique() == 162
    return keys


def caller_inputs(keys):
    # Only participant IDs are read from this file; no behavior/relevance field is selected.
    pairs = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    result = keys.join(pairs, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
    assert result.height == keys.height and result.select('pair_id', 'hand_id').equals(keys)
    return result


def cached_baseline(keys, features):
    sys.path.insert(0, str(CV / 'src'))
    from evidence_aug.fold import load_fold

    data, available, _, _ = load_fold(0)
    expected = [name for name in available if name in features]
    if list(features) != expected:
        raise AssertionError('cached equity feature manifest/order differs from expected 24 columns')
    baseline = (data.join(keys, on=['pair_id', 'hand_id'], how='semi')
                .select(['pair_id', 'hand_id', *features]).sort('pair_id', 'hand_id'))
    assert baseline.select('pair_id', 'hand_id').equals(keys)
    return baseline


def worker_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location('original_equity_precision', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_samples(module, count):
    indices = sample_indices(count)
    module.PREFLOP_SAMPLES = count
    module._SAMPLE_INDICES = indices
    module._canonical_equity.cache_clear()
    return indices


def public_inputs(keys, module):
    hand_ids = keys['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(hand_ids))
             .select('hand_id', 'big_blind', 'board_cards').collect())
    if hands.height != len(hand_ids):
        raise RuntimeError('missing requested hand')
    hand_lookup = {hand: (blind, [module.CARDS[card] for card in (board or '').split()])
                   for hand, blind, board in hands.iter_rows()}
    seats = (pl.scan_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(hand_ids))
             .select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').collect())
    seat_lookup = {}
    for hand, player, c1, c2 in seats.iter_rows():
        seat_lookup.setdefault(hand, {})[player] = (module.CARDS[c1], module.CARDS[c2])
    actions = (pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(hand_ids))
               .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call',
                       'pot_before', 'stack_before').sort('hand_id', 'action_no').collect())
    action_lookup = {}
    for row in actions.iter_rows(named=True):
        hand = row.pop('hand_id')
        action_lookup.setdefault(hand, []).append(row)
    return hand_lookup, seat_lookup, action_lookup, actions.height


def swap_test(module):
    cards = module.CARDS
    seats = {'a': (cards['As'], cards['Ad']), 'b': (cards['Kc'], cards['Kd']), 'c': (cards['4c'], cards['5d'])}
    preflop_left = module.heads_up_equity(seats['a'], seats['b'], [])
    preflop_right = module.heads_up_equity(seats['b'], seats['a'], [])
    assert preflop_left + preflop_right == 1.0, 'preflop paired-swap complement failed'
    preflop_actions = [
        {'action_no': 0, 'player_id': 'a', 'action': 'raise', 'amount': 6, 'to_call': 2, 'pot_before': 3, 'stack_before': 100, 'street': 'preflop'},
        {'action_no': 1, 'player_id': 'b', 'action': 'call', 'amount': 4, 'to_call': 4, 'pot_before': 9, 'stack_before': 100, 'street': 'preflop'},
    ]
    preflop_features = module.extract_pair_hand('a', 'b', 2, seats, [], preflop_actions)
    preflop_swapped = module.extract_pair_hand('b', 'a', 2, seats, [], list(reversed(preflop_actions)))
    assert preflop_features == preflop_swapped, 'preflop paired-swap feature invariance failed'
    board = [cards[c] for c in ('2c', '3h', '7s', '9d', 'Jc')]
    river_actions = [
        {'action_no': 2, 'player_id': 'a', 'action': 'bet', 'amount': 10, 'to_call': 0, 'pot_before': 20, 'stack_before': 100, 'street': 'river'},
        {'action_no': 3, 'player_id': 'b', 'action': 'call', 'amount': 10, 'to_call': 10, 'pot_before': 30, 'stack_before': 100, 'street': 'river'},
    ]
    river_left = module.extract_pair_hand('a', 'b', 2, seats, board, river_actions)
    river_right = module.extract_pair_hand('b', 'a', 2, seats, board, list(reversed(river_actions)))
    assert river_left == river_right, 'postflop paired-swap feature invariance failed'


def worker(input_path, count, output_path, receipt_path):
    keys = pl.read_parquet(input_path).sort('pair_id', 'hand_id')
    module = worker_module()
    if tuple(module.FEATURES) != tuple(FEATURES):
        raise AssertionError('root equity source feature manifest differs from expected 24 columns')
    # Verify the original default 256 first block before replacing any runtime globals.
    assert module.PREFLOP_SAMPLES == BASE_SAMPLES
    assert np.array_equal(module._SAMPLE_INDICES, sample_indices(BASE_SAMPLES)), 'original 256 generator mismatch'
    common_prefix_verified = np.array_equal(sample_indices(HIGH_SAMPLES)[:BASE_SAMPLES], sample_indices(BASE_SAMPLES))
    assert common_prefix_verified, '4096 generator does not retain the 256 common prefix'
    indices = configure_samples(module, count)
    assert np.array_equal(module._SAMPLE_INDICES, sample_indices(count)), 'configured sample matrix mismatch'
    swap_test(module)
    hand_lookup, seat_lookup, action_lookup, action_rows = public_inputs(keys, module)
    real_heads_up = module.heads_up_equity
    postflop_high = {}

    def tracked(first, second, board):
        value = real_heads_up(first, second, board)
        if board:
            postflop_high[(tuple(first), tuple(second), tuple(board))] = value
        return value

    module.heads_up_equity = tracked
    rows = []
    for pair_id, hand_id, first, second in keys.select('pair_id', 'hand_id', 'player_1', 'player_2').iter_rows():
        blind, board = hand_lookup[hand_id]
        values = module.extract_pair_hand(first, second, blind, seat_lookup[hand_id], board, action_lookup.get(hand_id, []))
        rows.append((pair_id, hand_id, *values))
    frame = (pl.DataFrame(rows, schema=['pair_id', 'hand_id', *module.FEATURES], orient='row')
             .with_columns(pl.col(module.FEATURES).cast(pl.Float32)).sort('pair_id', 'hand_id'))
    assert frame.select('pair_id', 'hand_id').equals(keys.select('pair_id', 'hand_id'))
    postflop_exact = 0
    if count == HIGH_SAMPLES:
        # One 256 cache reset is sufficient: all recorded states must use the exact branch.
        configure_samples(module, BASE_SAMPLES)
        for state, high_value in postflop_high.items():
            low_value = real_heads_up(*state)
            if low_value != high_value:
                raise AssertionError('postflop equity changed with preflop sample count')
            postflop_exact += 1
    frame.write_parquet(output_path)
    write(receipt_path, {
        'samples': count, 'feature_count': len(module.FEATURES), 'rows': frame.height,
        'action_rows': action_rows, 'original_default_256_generation_verified': True,
        'common_prefix_verified': common_prefix_verified,
        'paired_swap_preflop_complement_and_feature_invariance_verified': True,
        'postflop_exact_states_verified': postflop_exact,
        'root_equity_source_sha256': sha(ROOT_ORIGINAL),
        'claude_overlay_source_sha256': sha(CLAUDE_ORIGINAL),
    })


def run_worker(count, tag):
    subprocess.run([
        sys.executable, str(Path(__file__)), '--worker', '--input', str(OUT / 'inputs-id-only.parquet'),
        '--samples', str(count), '--output', str(OUT / f'features{tag}-recomputed.parquet'),
        '--receipt', str(OUT / f'worker-{tag}.json'),
    ], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'freeze.json').exists(), 'write-once extraction already started'
    assert sha(ROOT_ORIGINAL) == sha(CLAUDE_ORIGINAL), 'root and Claude equity sources differ; refusing to run overlay cache parity'
    # The controller intentionally does not import the original module. Its implementation
    # is imported only by fresh worker processes below.
    keys = current_keys()
    inputs = caller_inputs(keys)
    baseline = cached_baseline(keys, FEATURES)
    sources = [
        Path(__file__), ROOT_ORIGINAL, CLAUDE_ORIGINAL, PREV / 'scores.parquet', DATA / 'hands.parquet',
        DATA / 'seats.parquet', DATA / 'actions.parquet', DATA / 'development_labels.csv',
    ]
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_before_recomputation',
        'keys': {'pair_hands': keys.height, 'pairs': keys['pair_id'].n_unique(), 'topk': 20},
        'features': FEATURES, 'sample_seed': SAMPLE_SEED, 'samples': [BASE_SAMPLES, HIGH_SAMPLES],
        'preflop_generation': 'np.argsort(default_rng(SAMPLE_SEED).random((N,48)), axis=1)[:,:5]; 4096 retains the first 256 draws.',
        'postflop': 'Original exact combinations branch unchanged; the high-sample worker compares every observed postflop state after one 256 cache reset.',
        'scope': 'No fit, metric, outer truth, evaluation data, parameter sweep, seed sweep, or permutation sweep.',
        'source_hashes': {str(path): sha(path) for path in sources},
    }
    write(OUT / 'freeze.json', freeze)
    inputs.write_parquet(OUT / 'inputs-id-only.parquet')
    baseline.write_parquet(OUT / 'baseline256-cached.parquet')

    # Hard gate: no high-sample worker starts until the current float32 cache is exactly rebuilt.
    run_worker(BASE_SAMPLES, '256')
    reconstructed = pl.read_parquet(OUT / 'features256-recomputed.parquet').sort('pair_id', 'hand_id')
    assert reconstructed.select('pair_id', 'hand_id').equals(keys)
    assert np.array_equal(reconstructed.select(FEATURES).to_numpy(), baseline.select(FEATURES).to_numpy()), \
        '256 reconstruction mismatch with current raw cache; refusing 4096 recomputation'
    low_meta = json.loads((OUT / 'worker-256.json').read_text(encoding='utf-8'))
    write(OUT / 'parity256.json', {
        'exact_float32_cache_parity': True, 'rows': reconstructed.height,
        'baseline_cached_sha256': sha(OUT / 'baseline256-cached.parquet'),
        'reconstructed256_sha256': sha(OUT / 'features256-recomputed.parquet'),
        'worker_receipt': low_meta,
    })

    run_worker(HIGH_SAMPLES, '4096')
    high = pl.read_parquet(OUT / 'features4096-recomputed.parquet').sort('pair_id', 'hand_id')
    assert high.select('pair_id', 'hand_id').equals(keys)
    # The only delivered feature file is the aligned replacement block with unchanged names.
    high.select(['pair_id', 'hand_id', *FEATURES]).write_parquet(OUT / 'features4096.parquet')
    high_meta = json.loads((OUT / 'worker-4096.json').read_text(encoding='utf-8'))
    write(OUT / 'receipt.json', {
        'status': 'complete_exact256_parity_and_4096_replacement', 'freeze_sha256': sha(OUT / 'freeze.json'),
        'baseline_cached_sha256': sha(OUT / 'baseline256-cached.parquet'),
        'reconstructed256_sha256': sha(OUT / 'features256-recomputed.parquet'),
        'features4096_sha256': sha(OUT / 'features4096.parquet'),
        'exact_float32_256_parity': True,
        'paired_swap_verified': True,
        'postflop_exact_states_verified': high_meta['postflop_exact_states_verified'],
        'worker_receipts': {'256': low_meta, '4096': high_meta},
        'limitations': [
            '4096 is a lower-noise preflop approximation, not exact preflop truth.',
            'All postflop values use the original exact engine unchanged.',
            'No fit or metric is performed by this extractor.',
        ],
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--output')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.input), args.samples, Path(args.output), Path(args.receipt))
    else:
        main()
