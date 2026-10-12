"""Step 3+4: freeze the six-arm design, run the six fixed CPU fits, freeze predictions before any truth is read.

Arms (one fixed recipe, PARAMS from rank_evidence, 382 features):
  original_only          274 original training target queries (listed evidence = 1, other hands = 0/unknown)
  control_88721          original + 78 duplicated real soft-play queries (plain permutation draw; defines histogram H)
  control_99643          original + 78 duplicated real soft-play queries drawn within annotation-count strata to match H
  control_110017         same with seed 110017
  synthetic_capped       original + 78 synthetic queries, per-query positives capped to H by fixed hash selection
  synthetic_all_causal   original + 78 synthetic queries, every verified intervention hand positive

Outputs under Claude Version/experiments/evidence_augmentation_20260910/:
  freeze.json, models/<arm>.joblib, scores/<arm>-<roles>.parquet, predictions-frozen.json
"""
import os
import sys
import time
import json
import hashlib
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import EXP, SRC, UTILITY, BUNDLE, BUNDLE_SHA, SAVED_SCORES, sha, write_json, read_json, check_protected
from evidence_aug.frame import load_original, held_pools, roles, training_relevance, attach_relevance, score, cache_identity, family_columns, PARAMS
from evidence_aug.step1_reproduce_baseline import fit_ranker
import numpy as np
import polars as pl
import joblib

SYN = EXP / 'synthetic'
CONTROL_SEEDS = [88721, 99643, 110017]
HASH_PREFIX = 'evidence-aug-v1:'
N_ADDED = 78


def h256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def control_draws(soft_counts):
    """soft_counts: {pair_id: annotation count} for the 92 original training soft-play pairs."""
    ids = sorted(soft_counts)
    first = [str(p) for p in np.random.default_rng(CONTROL_SEEDS[0]).permutation(ids)[:N_ADDED]]
    histogram = {}
    for p in first:
        histogram[soft_counts[p]] = histogram.get(soft_counts[p], 0) + 1
    draws = {CONTROL_SEEDS[0]: first}
    for seed in CONTROL_SEEDS[1:]:
        rng = np.random.default_rng(seed)
        chosen = []
        for k in sorted(histogram):
            stratum = sorted(p for p in ids if soft_counts[p] == k)
            assert len(stratum) >= histogram[k]
            chosen += [str(p) for p in rng.permutation(stratum)[:histogram[k]]]
        draws[seed] = chosen
    for seed, chosen in draws.items():
        assert len(chosen) == N_ADDED == len(set(chosen))
        got = {}
        for p in chosen:
            got[soft_counts[p]] = got.get(soft_counts[p], 0) + 1
        assert got == histogram
    return draws, histogram


def capped_selection(traces, histogram):
    """Assign H's counts to synthetic pairs in hash order; pick that many trace hands per pair by hash."""
    pairs = sorted(set(traces['pair_id']), key=lambda p: h256(HASH_PREFIX + p))
    counts = sorted(k for k, n in histogram.items() for _ in range(n))
    assert len(pairs) == len(counts) == N_ADDED
    selection = {}
    for p, k in zip(pairs, counts):
        hands = sorted(traces.filter(pl.col('pair_id') == p)['hand_id'], key=lambda hnd: h256(HASH_PREFIX + p + ':' + hnd))
        assert len(hands) >= k
        selection[p] = hands[:k]
    return selection


def main():
    t0 = time.perf_counter()
    assert not (EXP / 'freeze.json').exists(), 'freeze already exists; preserve it'
    protected = check_protected()
    assert sha(BUNDLE) == BUNDLE_SHA
    parity = read_json(EXP / 'step1-baseline-parity.json')
    assert all(v['exact'] for k, v in parity['parity'].items() if k != 'refit_original_only') and parity['parity']['refit_original_only']['exact_scores']
    syn_parity = read_json(SYN / 'parity.json')
    assert all(syn_parity['parity'][k] for k in ['sequence_exact', 'equity_exact', 'style_counts_exact', 'conditional_held_exact'])
    data, features, caf = load_original()
    pools = held_pools()
    held = data.filter(pl.col('table_id').is_in(pools))
    train = data.filter(~pl.col('table_id').is_in(pools))
    evidence = training_relevance(set(train['table_id']))
    assert not set(evidence['pair_id']) & set(held['pair_id'])
    train = attach_relevance(train, evidence)
    soft = train.filter(pl.col('behavior_family') == 'soft_play').group_by('pair_id').agg(pl.col('relevance').sum().alias('k'))
    soft_counts = {p: int(k) for p, k in soft.iter_rows()}
    assert len(soft_counts) == 92 and sum(soft_counts.values()) == 445 and set(soft_counts.values()) <= {3, 4, 5}
    draws, histogram = control_draws(soft_counts)
    traces = pl.read_parquet(SYN / 'traces.parquet')
    assert sha(SYN / 'traces.parquet') == syn_parity['traces']['sha256'] and traces.height == 3172
    synthetic = pl.concat([pl.read_parquet(SYN / f'batch_{b}.parquet') for b in '012']).sort('pair_id', 'hand_id')
    assert synthetic['pair_id'].n_unique() == 78 and set(synthetic['behavior_family']) == {'soft_play'}
    assert synthetic.columns == data.columns
    freeze_selected = read_json(UTILITY / 'trial' / 'freeze.json')['selected_target_pairs']
    assert set(synthetic['pair_id']) == {p for b in '012' for p in freeze_selected[b]} == set(traces['pair_id'])
    capped = capped_selection(traces, histogram)
    all_causal = traces.select('pair_id', 'hand_id').with_columns(pl.lit(1, dtype=pl.Int64).alias('relevance'))
    capped_frame = pl.DataFrame([(p, hnd) for p, hands in capped.items() for hnd in hands], schema=['pair_id', 'hand_id'], orient='row').with_columns(pl.lit(1, dtype=pl.Int64).alias('relevance'))
    assert capped_frame.height == sum(k * n for k, n in histogram.items())
    assert capped_frame.join(traces.select('pair_id', 'hand_id'), on=['pair_id', 'hand_id'], how='anti').height == 0
    arms = {}
    arms['original_only'] = train
    for seed, chosen in draws.items():
        dup = train.filter(pl.col('pair_id').is_in(chosen)).with_columns((pl.col('pair_id') + '#dup').alias('pair_id'))
        assert dup['pair_id'].n_unique() == N_ADDED
        arms[f'control_{seed}'] = pl.concat([train, dup])
    arms['synthetic_capped'] = pl.concat([train, attach_relevance(synthetic, capped_frame)])
    arms['synthetic_all_causal'] = pl.concat([train, attach_relevance(synthetic, all_causal)])
    summary = {}
    for name, frame in arms.items():
        q = frame.group_by('pair_id').agg(pl.len().alias('n'), pl.col('relevance').sum().alias('k'))
        assert q['k'].min() >= 1
        summary[name] = {'queries': q.height, 'rows': frame.height, 'positives': int(q['k'].sum()),
                         'added_queries': q.height - 274, 'added_positives': int(q['k'].sum()) - 1344,
                         'mean_positives_per_added_query': float((q['k'].sum() - 1344) / max(q.height - 274, 1))}
    code = {p.name: sha(p) for p in sorted((SRC / 'evidence_aug').glob('*.py'))}
    code.update({'src/' + n: sha(SRC / n) for n in ['rank_evidence.py', 'sequence_evidence_features.py', 'equity_evidence_features.py', 'conditional_action_features.py',
                                                      'conditional_inference_worker.py', 'run_conditional_policy_screen.py', 'advanced_features.py']})
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'bundle_sha256': BUNDLE_SHA, 'protected': protected,
        'recipe': PARAMS, 'features': 382, 'code_sha256': code, 'original_cache_identity': cache_identity(),
        'synthetic_inputs': {b: syn_parity['batches'][b]['sha256'] for b in '012'}, 'traces_sha256': syn_parity['traces']['sha256'],
        'training_pools': 297, 'training_target_queries': 274, 'training_listed_hands': 1344, 'original_soft_queries': 92, 'original_soft_listed_hands': 445,
        'annotation_histogram': {str(k): v for k, v in sorted(histogram.items())},
        'control_draws': {str(seed): chosen for seed, chosen in draws.items()},
        'control_rule': 'seed 88721: rng.permutation(sorted soft ids)[:78] defines H; seeds 99643/110017: within each annotation-count stratum, rng.permutation(sorted stratum)[:H[k]]. Duplicates get group id pair_id+"#dup".',
        'capped_rule': f'Synthetic pairs sorted by sha256("{HASH_PREFIX}"+pair_id); counts = sorted expansion of H assigned in that order; hands = first k of trace hands sorted by sha256("{HASH_PREFIX}"+pair_id+":"+hand_id). Unselected hands (including other intervention hands) are relevance 0 = unknown.',
        'capped_selection': capped, 'all_causal_rule': 'every verified intervention hand of the 78 pairs is relevance 1',
        'arms': summary, 'roles': {'primary': 'conditional_roles (pair-predictions.csv conditional_class)', 'secondary': 'original_roles (validation_pairs.csv predicted_behavior)'},
        'routing': 'experimental ranker scores only held pairs whose fixed predicted family is soft_play; every other pair keeps the saved baseline scores',
        'metric': 'MAP@5 per held target pair, top-5 by score desc with seeded tie breaker (931207) as in relative_evidence_experiment.records; ap5 = sum(hits/rank)/min(5,|truth|)',
        'gate': {'overall_gain_vs_original_min': .005, 'overall_gain_vs_control_mean_min': .005, 'true_soft_gain_min': .01, 'true_family_loss_max': .01, 'pool_halves': 'both nonnegative'},
        'pool_halves': "sha256('softplay-fast-v1:'+table_id) % 2", 'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool'},
        'held_evidence_truth_read_before_freeze': False,
    }
    write_json(EXP / 'freeze.json', freeze)
    print(json.dumps({'frozen': True, 'histogram': freeze['annotation_histogram'], 'arms': summary}), flush=True)
    (EXP / 'models').mkdir(exist_ok=True); (EXP / 'scores').mkdir(exist_ok=True)
    role_frames = {k: roles(k) for k in ['conditional_roles', 'original_roles']}
    timings = {}
    for name, frame in arms.items():
        t1 = time.perf_counter()
        model = fit_ranker(frame, features)
        timings[name] = time.perf_counter() - t1
        joblib.dump({'model': model, 'features': features, 'arm': name}, EXP / 'models' / f'{name}.joblib')
        for kind, rf in role_frames.items():
            hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
            s = score(model, family_columns(hf, 'predicted_behavior'), features)
            assert np.isfinite(s).all()
            hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(EXP / 'scores' / f'{name}-{kind}.parquet')
        # oracle-family diagnostic inputs (true family one-hots), never used for routing
        hf = held.sort('pair_id', 'hand_id')
        s = score(model, family_columns(hf, 'behavior_family'), features)
        hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(EXP / 'scores' / f'{name}-oracle_family.parquet')
        print(json.dumps({'fit': name, 'seconds': timings[name]}), flush=True)
    # original_only must reproduce the saved baseline exactly
    base = pl.read_parquet(EXP / 'scores' / 'original_only-conditional_roles.parquet')['score'].to_numpy()
    saved = pl.read_parquet(SAVED_SCORES['conditional_roles'])['score'].to_numpy()
    assert np.array_equal(base, saved), 'original-only arm does not reproduce the saved baseline'
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(EXP / 'freeze.json'),
              'scores': {p.name: sha(p) for p in sorted((EXP / 'scores').glob('*.parquet'))},
              'models': {p.name: sha(p) for p in sorted((EXP / 'models').glob('*.joblib'))},
              'fit_seconds': timings, 'original_only_reproduces_saved_baseline': True, 'held_evidence_truth_read': False,
              'seconds': time.perf_counter() - t0}
    write_json(EXP / 'predictions-frozen.json', frozen)
    print(json.dumps({'predictions_frozen': True, 'fit_seconds': timings, 'seconds': frozen['seconds']}), flush=True)


if __name__ == '__main__':
    main()
