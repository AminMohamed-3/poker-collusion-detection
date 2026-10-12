"""Two frozen evidence-only entries for calibration against the public score.

Both replace evidence for the same 10,000 highest-risk pairs; all other evidence,
and every risk/behavior cell, remain exactly as in v6. No test labels are used.
"""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import gc
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import polars as pl

from evidence_capacity_experiment import ROOT, VARIANTS, load_sequence, fit_variant
from equity_evidence_features import CARDS, FEATURES as EQ, extract_pair_hand
from rank_evidence import PARAMS, KEYS, EVIDENCE_COLUMNS, context, family_columns, matrix
from relative_evidence_experiment import sha
from sequence_evidence_experiment import FEATURES as SEQ
from sequence_evidence_fast import extract
from sequence_evidence_candidate import verify_source_cache, write_json

REPORT = ROOT / 'reports/evidence-public-calibration'
OUTPUTS = {'capacity': ROOT / 'outputs_v7_capacity', 'equity': ROOT / 'outputs_v8_equity'}
SOURCE = ROOT / 'outputs_v6/submission_sequence_evidence.csv'
TOP_PAIRS = 10000


def equity_frame(keys, hands, seats, actions):
    hand_lookup = {h: (bb, [CARDS[c] for c in (board or '').split()])
                   for h, bb, board in hands.select('hand_id', 'big_blind', 'board_cards').iter_rows()}
    seat_lookup, action_lookup = {}, {}
    for h, p, a, b in seats.select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').iter_rows():
        seat_lookup.setdefault(h, {})[p] = (CARDS[a], CARDS[b])
    for row in actions.select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call', 'pot_before', 'stack_before').iter_rows(named=True):
        hand = row.pop('hand_id')
        action_lookup.setdefault(hand, []).append(row)
    result = []
    for pair, hand, first, second in keys.select('pair_id', 'hand_id', 'player_1', 'player_2').iter_rows():
        blind, board = hand_lookup[hand]
        values = extract_pair_hand(first, second, blind, seat_lookup[hand], board, action_lookup.get(hand, []))
        result.append((pair, hand, *values))
    return pl.DataFrame(result, schema=['pair_id', 'hand_id', *EQ], orient='row').with_columns(pl.col(EQ).cast(pl.Float32))


def rank(frame, scores):
    return frame.select('pair_id', 'hand_id', 'tie').with_columns(pl.Series('score', scores)).sort(
        ['pair_id', 'score', 'tie'], descending=[False, True, False]).group_by('pair_id', maintain_order=True).agg(
            pl.col('hand_id').head(5).alias('hands'), pl.len().alias('shared_hands'))


def main():
    REPORT.mkdir(exist_ok=True)
    source = pl.read_csv(SOURCE, infer_schema=False)
    selected = source.with_columns(pl.col('risk_score').cast(pl.Float64)).sort(
        ['risk_score', 'pair_id'], descending=[True, False]).head(TOP_PAIRS).select('pair_id')
    protected = {str(p.relative_to(ROOT)): sha(p) for p in [SOURCE, ROOT / 'outputs_v6/sequence_ranker.joblib',
                 ROOT / 'outputs_v5/submission_event_tails.csv', ROOT / 'outputs_v5/pair_model_event_tails.joblib']}
    code = ['src/evidence_public_calibration.py', 'src/evidence_capacity_experiment.py', 'src/equity_evidence_features.py',
            'src/rank_evidence.py', 'src/sequence_evidence_fast.py', 'src/sequence_evidence_features.py']
    inputs = {p: sha(ROOT / p) for p in code + ['reports/evidence-capacity-experiment/metrics.json',
              'reports/equity-evidence-experiment/metrics.json', 'reports/evidence-probe-confirmation/metrics.json']}
    plan_path = REPORT / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan['protected'] == protected and plan['inputs'] == inputs
        assert pl.read_parquet(REPORT / 'selected-pairs.parquet').equals(selected)
    else:
        assert not list(REPORT.glob('*.parquet'))
        plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'inputs': inputs,
                'top_pairs': TOP_PAIRS, 'selection': 'Highest v6 risk, full histories; all remaining evidence unchanged.',
                'capacity': VARIANTS['larger_regularized'], 'equity': PARAMS,
                'purpose': 'User-requested controlled model submissions to measure local/public agreement.',
                'decision': 'Adaptive experiments, not statistically confirmed improvements. Equity failed the first screening gate but improved on the second split; capacity improved slightly on both.',
                'rules': 'Maximum five per day; two final entries. No row-level label probes, artifact features, or generator internals.'}
        write_json(plan_path, plan)
        selected.write_parquet(REPORT / 'selected-pairs.parquet')
    verify_source_cache('development')
    verify_source_cache('evaluation')
    data, current, _, _, _ = load_sequence()
    extra = pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/development.parquet')
    data = context(data.join(extra, on=['pair_id', 'hand_id'], validate='1:1'), EQ)
    bundles = {}
    for name, features, params in [('capacity', current, VARIANTS['larger_regularized']),
                                   ('equity', current + EQ + ['context_' + c for c in EQ], PARAMS)]:
        out = OUTPUTS[name]
        out.mkdir(exist_ok=True)
        artifact, receipt = out / 'ranker.joblib', out / 'fit.json'
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert meta['plan_sha256'] == sha(plan_path) and meta['model_sha256'] == sha(artifact)
            bundle = joblib.load(artifact)
        else:
            assert not artifact.exists()
            model = fit_variant(data, features, params)
            bundle = {'model': model, 'features': features, 'parameters': params}
            joblib.dump(bundle, artifact)
            check = matrix(family_columns(data.head(64), 'behavior_family'), features)
            np.testing.assert_array_equal(model.predict(check), joblib.load(artifact)['model'].predict(check))
            write_json(receipt, {'plan_sha256': sha(plan_path), 'model_sha256': sha(artifact),
                                'features': len(features), 'training_pairs': data['pair_id'].n_unique(), 'training_hands': data.height,
                                'reload_predictions_identical': True, 'protected_source_sha256': sha(SOURCE)})
        bundles[name] = bundle
        (out / 'rank_batches').mkdir(exist_ok=True)
        print(f'Full {name} model ready', flush=True)
    del data, extra
    gc.collect()
    control = joblib.load(ROOT / 'outputs_v6/sequence_ranker.joblib')
    pairs = pl.read_csv(ROOT / 'data/evaluation_pairs.csv')
    started = time.perf_counter()
    for index, path in enumerate(sorted((ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet'))):
        batch_receipt = REPORT / (path.stem + '.json')
        if batch_receipt.exists():
            verified = json.loads(batch_receipt.read_text())
            assert verified['plan_sha256'] == sha(plan_path)
            for p, digest in verified['files'].items():
                assert sha(ROOT / p) == digest
            print(f'Batch {index + 1}/20 cached and verified', flush=True)
            continue
        raw = pl.read_parquet(path)
        all_pools = sorted(raw['table_id'].unique().to_list())
        selected_raw = raw.join(selected, on='pair_id', how='semi')
        keys = selected_raw.select(KEYS).join(pairs.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        ids = keys.select('hand_id').unique().lazy()
        hands = pl.scan_parquet(ROOT / 'data/hands.parquet').filter(pl.col('phase') == 'evaluation').join(ids, on='hand_id').collect()
        seats = pl.scan_parquet(ROOT / 'data/seats.parquet').join(ids, on='hand_id').collect()
        actions = pl.scan_parquet(ROOT / 'data/actions.parquet').join(ids, on='hand_id').collect()
        pieces = {name: [] for name in OUTPUTS}
        features_saved = []
        for offset in range(0, len(all_pools), 5):
            pools = all_pools[offset:offset + 5]
            original = raw.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
            original = original.with_columns(pl.Series('tie', np.random.default_rng(931207 + index * 100 + offset).random(original.height)))
            part = original.join(selected, on='pair_id', how='semi')
            if not part.height:
                continue
            pk = keys.filter(pl.col('table_id').is_in(pools))
            ph = hands.filter(pl.col('table_id').is_in(pools))
            phids = ph.select('hand_id')
            ps, pa = seats.join(phids, on='hand_id'), actions.join(phids, on='hand_id')
            seq = extract(pk, ph, ps, pa)
            eq = equity_frame(pk, ph, ps, pa)
            joined = seq.join(eq, on=['pair_id', 'hand_id'], validate='1:1')
            features_saved.append(joined)
            frame = part.join(joined, on=['pair_id', 'hand_id'], validate='1:1').join(
                source.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
            frame = family_columns(context(frame, [c for c in control['raw_features'] if c.startswith('v2_')] + SEQ + EQ), 'predicted_behavior')
            control_rank = rank(frame, control['model'].predict(matrix(frame, control['features']), num_threads=2))
            wanted = source.join(control_rank.select('pair_id'), on='pair_id', how='semi').select(
                'pair_id', pl.concat_list(EVIDENCE_COLUMNS).alias('expected'))
            comparison = control_rank.join(wanted, on='pair_id')
            assert all([h for h in expected if h != 'NO_EVIDENCE'] == actual
                       for actual, expected in comparison.select('hands', 'expected').iter_rows()), 'v6 control ranking differs'
            for name, bundle in bundles.items():
                scores = bundle['model'].predict(matrix(frame, bundle['features']), num_threads=2)
                assert np.isfinite(scores).all()
                pieces[name].append(rank(frame, scores))
            print(f'Batch {index + 1}/20 pools {offset + len(pools)}/{len(all_pools)}: {frame.height} candidate hands; v6 control exact', flush=True)
            del original, part, pk, ph, ps, pa, seq, eq, joined, frame, control_rank, comparison
            gc.collect()
        saved = []
        feature_path = REPORT / (path.stem + '-features.parquet')
        pl.concat(features_saved).write_parquet(feature_path)
        saved.append(feature_path)
        for name, chunks in pieces.items():
            destination = OUTPUTS[name] / 'rank_batches' / path.name
            ranked = pl.concat(chunks)
            expected = selected_raw.group_by('pair_id').len()
            assert not ranked.join(expected, on='pair_id').filter(pl.col('shared_hands') != pl.col('len')).height
            ranked.write_parquet(destination)
            saved.append(destination)
        write_json(batch_receipt, {'plan_sha256': sha(plan_path), 'control_exact': True,
                                  'files': {str(p.relative_to(ROOT)): sha(p) for p in saved}, 'elapsed_seconds': time.perf_counter() - started})
        del raw, selected_raw, keys, hands, seats, actions, pieces, features_saved
        gc.collect()
    for name, out in OUTPUTS.items():
        ranked = pl.read_parquet(sorted((out / 'rank_batches').glob('batch_*.parquet')))
        assert ranked.height == ranked['pair_id'].n_unique() == TOP_PAIRS
        expanded = ranked.select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(c + '_new')
                                            for i, c in enumerate(EVIDENCE_COLUMNS)])
        result = source.join(expanded, on='pair_id', how='left', maintain_order='left').with_columns(
            *[pl.coalesce(c + '_new', c).alias(c) for c in EVIDENCE_COLUMNS]).select(source.columns)
        fixed = ['pair_id', 'risk_score', 'predicted_behavior']
        assert result.select(fixed).equals(source.select(fixed))
        assert result.join(selected, on='pair_id', how='anti').equals(source.join(selected, on='pair_id', how='anti'))
        destination = out / f'submission_{name}.csv'
        if destination.exists():
            assert pl.read_csv(destination, infer_schema=False).equals(result)
        else:
            result.write_csv(destination)
        changed = sum(a != b for a, b in zip(result.select(EVIDENCE_COLUMNS).rows(), source.select(EVIDENCE_COLUMNS).rows()))
        write_json(out / 'provenance.json', {'plan_sha256': sha(plan_path), 'submission_sha256': sha(destination),
                                           'rows': result.height, 'reranked_pairs': TOP_PAIRS, 'changed_evidence_rows': changed,
                                           'preserved_exactly': fixed, 'other_evidence_unchanged': True, 'control_rankings_exact': True,
                                           'validation_required': True})
        print(json.dumps({'candidate': name, 'file': str(destination), 'changed_rows': changed}), flush=True)
    assert protected == {p: sha(ROOT / p) for p in protected}
    assert inputs == {p: sha(ROOT / p) for p in inputs}


if __name__ == '__main__':
    main()
