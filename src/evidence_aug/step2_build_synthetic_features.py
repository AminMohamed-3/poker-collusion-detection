"""Step 2: 382-feature frames for the 78 selected synthetic soft-play pairs, plus trace labels and parity checks.

Outputs under Claude Version/experiments/evidence_augmentation_20260910/synthetic/:
  batch_<b>.parquet          -- assembled 382-feature frame (all shared hands of the 26 selected pairs of that batch)
  traces.parquet             -- (pair_id, hand_id, batch, changes) for every verified intervention hand of a selected pair
  parity.json                -- original-cache parity of the re-implemented builders and per-batch receipts
"""
import os
import sys
import time
import json
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import EXP, ORIGINAL, UTILITY, SCREEN, sha, write_json, read_json
from evidence_aug.frame import CACHES, assemble, held_pools, load_original, family_columns
from evidence_aug.synthetic import gameplay, sequence_features, equity_features, style_counts, conditional_features, clear_caches, KEYS3
import joblib
import numpy as np
import polars as pl

OUT = EXP / 'synthetic'
SOURCES = {'0': 'bet_size_realism_20260910/sized', '1': 'bet_size_confirmation_20260910/b1_sized', '2': 'bet_size_confirmation_20260910/fresh_sized'}


def original_parity(policy, n_pairs=3):
    """Re-run the three builders on a few ORIGINAL pairs and compare with the cached features exactly.

    Sequence/equity use held and training target pairs; conditional uses held pairs only, because the
    cached training rows came from inner cross-fit references rather than the held reference.
    """
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    pools = held_pools()
    keys_all = pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet')), columns=KEYS3)
    targets = labels.filter(pl.col('label') == 1)
    held_ids = sorted(set(targets['pair_id']) & set(keys_all.filter(pl.col('table_id').is_in(pools))['pair_id']))[:n_pairs]
    train_ids = sorted(set(targets['pair_id']) - set(keys_all.filter(pl.col('table_id').is_in(pools))['pair_id']))[:n_pairs]
    sample = held_ids + train_ids
    keys = keys_all.filter(pl.col('pair_id').is_in(sample)).join(labels.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
    hands, seats, actions = gameplay(ORIGINAL / 'data', keys.select('hand_id').unique())
    result = {'sample_pairs': sample, 'pair_hands': keys.height}
    seq = sequence_features(keys, hands, seats, actions).sort('pair_id', 'hand_id')
    ref = pl.read_parquet(CACHES['sequence']).filter(pl.col('pair_id').is_in(sample)).sort('pair_id', 'hand_id')
    result['sequence_exact'] = bool(seq.select(ref.columns).equals(ref))
    eq = equity_features(keys, hands, seats, actions).sort('pair_id', 'hand_id')
    ref = pl.read_parquet(CACHES['equity']).filter(pl.col('pair_id').is_in(sample)).sort('pair_id', 'hand_id')
    result['equity_exact'] = bool(eq.select(ref.columns).equals(ref))
    # Public style totals: recomputed from data/actions.parquet and compared with the saved counts file.
    counts = style_counts(ORIGINAL / 'data')
    saved = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(SCREEN / 'public-action-style-counts.parquet').iter_rows()}
    result['style_counts_exact'] = bool(set(counts) == set(saved) and all(np.array_equal(counts[k], saved[k]) for k in counts))
    hk = keys.filter(pl.col('pair_id').is_in(held_ids))
    ca, caf, info = conditional_features(hk, hands, seats, actions, counts, policy)
    ref = pl.read_parquet(CACHES['conditional']).filter(pl.col('pair_id').is_in(held_ids)).sort('pair_id', 'hand_id')
    diff = np.abs(ca.select(caf).to_numpy() - ref.select(caf).to_numpy())
    result['conditional_held_exact'] = bool(ca.select(ref.columns).equals(ref))
    result['conditional_held_max_abs_diff'] = float(diff.max())
    result['conditional_info'] = info
    clear_caches()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    freeze = read_json(UTILITY / 'trial' / 'freeze.json')
    plan = read_json(UTILITY / 'study-plan.json')
    sources = {e['arm']: e['directory'].replace('\\', '/') for e in plan['generation_sources']}
    for b, rel in SOURCES.items():
        assert sources['b%s_sized' % b].endswith(rel), sources
    policy = joblib.load(SCREEN / 'policy-screen' / 'held-models.joblib')
    held_receipt = read_json(SCREEN / 'policy-screen' / 'held.json')
    assert held_receipt['model_sha256'] == sha(SCREEN / 'policy-screen' / 'held-models.joblib')
    assert len(held_receipt['training_pools']) == 297 and not set(held_receipt['training_pools']) & set(held_pools())
    original_columns = pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet'))[0], n_rows=1).columns
    _, original_features, _ = load_original()
    report = {'parity': original_parity(policy), 'batches': {}, 'reference_model_sha256': held_receipt['model_sha256']}
    print(json.dumps({'original_parity': report['parity']}), flush=True)
    trace_rows = []
    for b, rel in SOURCES.items():
        t1 = time.perf_counter()
        src = ORIGINAL / 'experiments' / rel
        selected = list(freeze['selected_target_pairs'][b])
        assert len(selected) == 26
        labels = pl.read_csv(src / 'features' / 'data' / 'development_labels.csv')
        chosen = labels.filter(pl.col('pair_id').is_in(selected))
        assert chosen.height == 26 and set(chosen['behavior_family']) == {'soft_play'} and set(chosen['label_status']) == {'synthetic_intervention'}
        raw = pl.read_parquet(sorted((src / 'features' / 'advanced' / 'hands_development').glob('batch_*.parquet')))
        assert raw.columns == original_columns, 'synthetic raw schema differs'
        raw = raw.filter(pl.col('pair_id').is_in(selected))
        assert raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == raw.height and raw['pair_id'].n_unique() == 26
        keys = raw.select(KEYS3).join(chosen.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        hands, seats, actions = gameplay(src / 'features' / 'data', keys.select('hand_id').unique())
        assert set(hands['table_id']) == set(keys['table_id'])
        seq = sequence_features(keys, hands, seats, actions)
        eq = equity_features(keys, hands, seats, actions)
        counts = style_counts(src / 'features' / 'data')
        ca, caf, info = conditional_features(keys, hands, seats, actions, counts, policy)
        data, features, caf2 = assemble(raw, seq, eq, ca, chosen.select('pair_id', 'behavior_family'))
        assert features == original_features and caf2 == caf
        values = family_columns(data, 'behavior_family').select(features).to_numpy()
        assert np.isfinite(values).all(), 'non-finite synthetic feature'
        # trace labels: every intervention hand of a selected pair, each with at least one base!=chosen change
        hand_set = set(zip(data['pair_id'], data['hand_id']))
        n_hands = 0
        for p in selected:
            pool = f'pool-{int(p[3:7]):04d}'
            for e in read_json(src / 'pilot' / pool / 'interventions.json'):
                assert e['pair_id'] == p
                changes = [x for x in e['interventions'] if x['base_intent'][0] != x['chosen_intent'][0]]
                assert len(changes) == len(e['interventions']) >= 1
                assert (p, e['hand_id']) in hand_set, 'intervention hand missing from feature frame'
                trace_rows.append({'pair_id': p, 'hand_id': e['hand_id'], 'batch': int(b), 'changes': len(changes)})
                n_hands += 1
        dest = OUT / f'batch_{b}.parquet'
        data.write_parquet(dest)
        report['batches'][b] = {'source': rel, 'pairs': 26, 'rows': data.height, 'hands_per_pair_min': int(data.group_by('pair_id').len()['len'].min()),
                               'hands_per_pair_max': int(data.group_by('pair_id').len()['len'].max()), 'intervention_hands': n_hands,
                               'conditional_info': info, 'seconds': time.perf_counter() - t1, 'sha256': sha(dest),
                               'inputs': {n: sha(src / 'features' / 'data' / n) for n in ['hands.parquet', 'seats.parquet', 'actions.parquet', 'development_labels.csv']}}
        print(json.dumps({'batch': b, **report['batches'][b]}), flush=True)
        clear_caches()
    traces = pl.DataFrame(trace_rows).sort('pair_id', 'hand_id')
    assert traces.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == traces.height
    assert traces.height == 3172 and traces['changes'].sum() == 5715 and traces['pair_id'].n_unique() == 78
    traces.write_parquet(OUT / 'traces.parquet')
    report['traces'] = {'hands': traces.height, 'changes': int(traces['changes'].sum()), 'pairs': 78, 'sha256': sha(OUT / 'traces.parquet'),
                        'hands_per_pair': traces.group_by('pair_id').len()['len'].describe().to_dict(as_series=False)}
    report['seconds'] = time.perf_counter() - t0
    report['held_evidence_truth_read'] = False
    write_json(OUT / 'parity.json', report)
    print(json.dumps({'complete': True, 'seconds': report['seconds'], 'traces': report['traces']}), flush=True)


if __name__ == '__main__':
    main()
