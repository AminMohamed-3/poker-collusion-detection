"""Build and screen one gameplay action-sequence evidence candidate."""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import polars as pl
from relative_evidence_experiment import load, fit, sha, records, summarize, ROOT
from rank_evidence import family_columns, matrix, context, FAMILIES
from sequence_evidence_features import NAMES, extract_pair_hand

REPORT = ROOT/'reports'/'sequence-evidence-experiment'
FEATURES = ['seq_'+n for n in NAMES]


def build_features(keys, pairs, data, phase, destination):
    """Batch by observed hand; collect only required phase and pair-hands."""
    if destination.exists():
        raise FileExistsError('Feature output already exists')
    sources = [data/'hands.parquet', data/'seats.parquet', data/'actions.parquet', Path(__file__).with_name('sequence_evidence_features.py')]
    identity = {str(p): sha(p) for p in sources}
    keys = keys.join(pairs.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1').sort('hand_id', 'pair_id')
    assert keys.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == keys.height
    ids = keys.select('hand_id').unique()
    hands = pl.scan_parquet(data/'hands.parquet').filter(pl.col('phase') == phase).join(ids.lazy(), on='hand_id').select('hand_id', 'big_blind').collect()
    assert hands.height == ids.height
    blind = dict(hands.iter_rows())
    seats = pl.scan_parquet(data/'seats.parquet').join(ids.lazy(), on='hand_id').select('hand_id', 'player_id', 'total_contribution', 'net_chips').collect()
    seat_lookup = {}
    for h, p, contribution, net in seats.iter_rows():
        seat_lookup.setdefault(h, {})[p] = (contribution, net)
    actions = pl.scan_parquet(data/'actions.parquet').join(ids.lazy(), on='hand_id').select(
        'hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call', 'pot_before').collect()
    action_lookup = {}
    for row in actions.iter_rows(named=True):
        h = row.pop('hand_id')
        action_lookup.setdefault(h, []).append(row)
    output = []
    for i, row in enumerate(keys.iter_rows(named=True)):
        h = row['hand_id']
        values = extract_pair_hand(row['player_1'], row['player_2'], blind[h], seat_lookup[h], action_lookup.get(h, []))
        output.append((row['pair_id'], h, *values))
        if (i+1) % 10000 == 0:
            print(f'Extracted sequence features: {i+1}/{keys.height} pair-hands', flush=True)
    frame = pl.DataFrame(output, schema=['pair_id', 'hand_id', *FEATURES], orient='row').with_columns(pl.col(FEATURES).cast(pl.Float32))
    assert frame.height == keys.height
    assert identity == {str(p): sha(p) for p in sources}
    frame.write_parquet(destination)
    destination.with_suffix('.json').write_text(json.dumps({'phase': phase, 'rows': frame.height, 'hands': hands.height,
        'features': FEATURES, 'inputs': identity, 'output_sha256': sha(destination)}, indent=2)+'\n')
    return frame


def screen():
    REPORT.mkdir(exist_ok=True)
    if (REPORT/'start.json').exists():
        raise FileExistsError('Preserve prior fixed screening run')
    protected = [ROOT/'outputs_v5/submission_event_tails.csv', ROOT/'outputs_v5/pair_model_event_tails.joblib', ROOT/'outputs_v3/evidence_ranker.joblib']
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'candidate': 'Existing 222 + 23 action-sequence values + 23 within-pair z-scores',
            'feature_code_sha256': sha(Path(__file__).with_name('sequence_evidence_features.py')), 'source_sha256': sha(__file__),
            'known_gate': {'map_gain_min': .015, 'family_regression_max': .02},
            'omitted_gate': {'mean_gain_min': 0, 'family_regression_max': .025},
            'fit_budget': '2 known-family fits; 6 omitted-family fits only if known gate passes', 'protected': fingerprints,
            'scope': 'Existing fold0; less-reused confirmation pools still unselected/unscored'}
    (REPORT/'start.json').write_text(json.dumps(plan, indent=2)+'\n')
    started = time.perf_counter()
    data, current, _, truths, pairpred, held_tables = load()
    extra = build_features(data.select('pair_id', 'hand_id'), pl.read_csv(ROOT/'data/development_labels.csv'),
                           ROOT/'data', 'development', REPORT/'development.parquet')
    data = context(data.join(extra, on=['pair_id', 'hand_id'], validate='1:1'), FEATURES)
    candidate = current + FEATURES + ['context_'+c for c in FEATURES]
    train = data.filter(~pl.col('table_id').is_in(held_tables))
    held = data.filter(pl.col('table_id').is_in(held_tables)).join(pairpred.select('pair_id', 'predicted_behavior'), on='pair_id').sort('pair_id', 'hand_id')
    assert not set(train['table_id']) & set(held['table_id'])
    results, allrows = [], []
    eligible = False
    for omitted in [None, *FAMILIES]:
        if omitted is not None and not eligible:
            break
        tag = 'known' if omitted is None else 'omit_'+omitted
        training = train if omitted is None else train.filter(pl.col('behavior_family') != omitted)
        validation = held if omitted is None else held.filter(pl.col('behavior_family') == omitted)
        for name, features in [('current', current), ('sequence', candidate)]:
            t0 = time.perf_counter()
            model = fit(training, features)
            if omitted is None:
                scores = model.predict(matrix(family_columns(validation, 'predicted_behavior'), features), num_threads=2)
            else:
                scores = np.mean([model.predict(matrix(family_columns(validation.with_columns(pl.lit(f).alias('hypothesis')),
                    'hypothesis'), features), num_threads=2) for f in FAMILIES if f != omitted], axis=0)
            rows = records(validation, scores, truths, tag, name)
            result = {'experiment': tag, 'candidate': name, 'seconds': time.perf_counter()-t0, **summarize(rows)}
            results.append(result); allrows.extend(rows)
            print(json.dumps(result), flush=True)
            if omitted is None:
                validation.select('pair_id', 'hand_id').with_columns(pl.Series('score', scores)).write_parquet(REPORT/f'known-{name}-scores.parquet')
        if omitted is None:
            base, new = results
            eligible = new['map5']-base['map5'] >= .015 and min(new['by_family'][f]-base['by_family'][f] for f in FAMILIES) >= -.02
    lookup = {(r['experiment'], r['candidate']): r for r in results}
    if eligible:
        gains = [lookup['omit_'+f, 'sequence']['map5']-lookup['omit_'+f, 'current']['map5'] for f in FAMILIES]
        eligible = np.mean(gains) >= 0 and min(gains) >= -.025
    wide = pl.DataFrame([r for r in allrows if r['experiment']=='known']).pivot(on='candidate', index=['pair_id', 'table_id'], values='map5')
    pooled = wide.group_by('table_id').agg((pl.col('sequence')-pl.col('current')).sum().alias('d'), pl.len().alias('n'))
    ix = np.random.default_rng(849217).integers(0, pooled.height, (2000, pooled.height))
    bootstrap = pooled['d'].to_numpy()[ix].sum(axis=1)/pooled['n'].to_numpy()[ix].sum(axis=1)
    assert fingerprints == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    payload = {'plan': plan, 'results': results, 'screen_passed': bool(eligible),
               'known_pool_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
               'protected_unchanged': True, 'seconds': time.perf_counter()-started,
               'limitations': ['Reused exploratory fold; no independent confirmation.',
                   'Sequence values describe observed behavior and outcomes, not proof of intention.',
                   'Investment-before-fold counts preceding action amounts; blind postings absent from actions are not imputed.',
                   'Unlisted target-pair hands are noisy ranking preferences, not confirmed negative labels.']}
    pl.DataFrame(allrows).write_csv(REPORT/'screen-pairs.csv')
    (REPORT/'screen.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'screen_passed': bool(eligible), 'seconds': payload['seconds']}), flush=True)


if __name__ == '__main__':
    screen()
