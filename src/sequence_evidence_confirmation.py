"""One frozen confirmation of the sequence ranker on less-reused pools."""
from __future__ import annotations
import os
for name in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from relative_evidence_experiment import load, fit, sha, records, summarize, ROOT
from rank_evidence import family_columns, context, matrix, FAMILIES
from robust_pair_experiment import TRAIN_VIEWS, MODEL_PARAMS, thin, aggregate, assert_disjoint, stable_ap, LABELS
from tail_pair_experiment import tails
from sequence_evidence_experiment import FEATURES
sys.path.insert(0, str(ROOT.parents[1]/'research/coherent-history-thinning'))
from extractor import load_pool_inputs, sample_hands, replace_residuals

PARENT = ROOT/'reports/sequence-evidence-experiment'
REPORT = PARENT/'confirmation-v2'


def pair_metrics(frame, risk, predicted):
    y = np.array([LABELS[f] for f in frame['behavior_family']])
    return {'pair_ap': stable_ap(y > 0, risk),
            'behavior_map': float(np.mean([stable_ap(y == i, np.where(predicted == i, risk, 0)) for i in [1, 2, 3]]))}


def prepare():
    REPORT.mkdir(exist_ok=True)
    if (REPORT/'start.json').exists():
        raise FileExistsError('Confirmation already started; never retune or silently restart')
    raw = pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    labels = pl.read_csv(ROOT/'data/development_labels.csv')
    full = pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(labels, on='pair_id').sort('pair_id')
    old = pl.read_csv(ROOT/'reports/tail-pair-experiment/predictions.csv')['table_id'].unique().to_list()
    all_pools = sorted(full['table_id'].unique().to_list())
    eligible = sorted(set(all_pools)-set(old))
    # Three of the 400 gameplay pools have no explicitly labeled pair. The
    # labeled-pair cache therefore contains 397 pools; 169 remain eligible.
    assert len(all_pools) == 397 and len(old) == 228 and len(eligible) == 169
    held_pools = sorted(np.random.default_rng(20260909).choice(eligible, size=100, replace=False).tolist())
    prior = PARENT/'confirmation'
    if (prior/'start.json').exists():
        prior_identity = json.loads((prior/'start.json').read_text())
        assert held_pools == prior_identity['held_pools']
        assert not list(prior.glob('*.joblib')) and not (prior/'metrics.json').exists()
    train_pools = sorted(set(all_pools)-set(held_pools))
    membership = full.select('pair_id', 'table_id', 'player_1', 'player_2').with_columns(
        pl.col('table_id').is_in(held_pools).alias('confirmation'))
    assert_disjoint(membership.filter(~pl.col('confirmation')), membership.filter('confirmation'))
    inputs = [PARENT/'confirmation-plan.md', PARENT/'development.parquet', PARENT/'development.json',
              ROOT/'data/development_labels.csv', ROOT/'data/development_evidence.csv',
              ROOT/'reports/tail-pair-experiment/predictions.csv', Path(__file__),
              ROOT/'src/sequence_evidence_features.py', ROOT/'src/rank_evidence.py',
              ROOT/'src/robust_pair_experiment.py', ROOT/'src/tail_pair_experiment.py',
              ROOT.parents[1]/'research/coherent-history-thinning/extractor.py']
    inputs += sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet'))
    protected = [ROOT/'outputs_v5/submission_event_tails.csv', ROOT/'outputs_v5/pair_model_event_tails.joblib', ROOT/'outputs_v3/evidence_ranker.joblib']
    identity = {'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': {str(p): sha(p) for p in inputs},
                'protected': {str(p): sha(p) for p in protected}, 'held_pools': held_pools, 'train_pools': train_pools,
                'eligible_pool_count': len(eligible), 'seed': 20260909, 'model_fits': 4,
                'parameters_pair': MODEL_PARAMS, 'training_views': TRAIN_VIEWS,
                'short': {'fraction': .65, 'seed': 7365, 'coherent_pool_hands': True, 'recompute_player_baselines': True}}
    (REPORT/'start.json').write_text(json.dumps(identity, indent=2)+'\n')
    membership.write_parquet(REPORT/'membership.parquet')
    print(json.dumps({'confirmation_pools': len(held_pools), 'eligible_pools': len(eligible),
                      'held_pairs': membership.filter('confirmation').height}), flush=True)
    raw_names = [c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]
    events = [c for c in raw_names if c.startswith('v2_')]
    base_features = joblib.load(ROOT/'outputs_v2/pair_model_robust.joblib')['features']
    risk_features = joblib.load(ROOT/'outputs_v5/pair_model_event_tails.joblib')['features']
    full = full.join(tails(raw, events), on=['pair_id', 'table_id']).sort('pair_id')
    views = [full.filter(pl.col('table_id').is_in(train_pools))]
    for fraction, seed in TRAIN_VIEWS:
        sample = thin(raw.filter(pl.col('table_id').is_in(train_pools)), fraction, seed)
        view = aggregate(sample, raw_names).join(tails(sample, events), on=['pair_id','table_id']).join(labels, on='pair_id')
        views.append(view.select(full.columns).sort('pair_id'))
    train = pl.concat(views)
    held_full = full.filter(pl.col('table_id').is_in(held_pools)).sort('pair_id')
    assert_disjoint(train, held_full)
    assert train.group_by('pair_id').len()['len'].min() == train.group_by('pair_id').len()['len'].max() == 3
    shortened = []
    for offset in range(0, len(held_pools), 20):
        pools = held_pools[offset:offset+20]
        hands, seats, actions = load_pool_inputs(ROOT/'data', 'development', pools)
        sampled = sample_hands(hands, .65, 7365)
        original = raw.filter(pl.col('table_id').is_in(pools))
        reduced, _ = replace_residuals(original, labels, sampled, seats, actions)
        assert reduced['pair_id'].n_unique() == original['pair_id'].n_unique()
        shortened.append(reduced)
        print(f'Coherent short histories rebuilt: {offset+len(pools)}/100 pools', flush=True)
    short_raw = pl.concat(shortened)
    held_short = aggregate(short_raw, raw_names).join(tails(short_raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').select(full.columns).sort('pair_id')
    assert held_short['pair_id'].to_list() == held_full['pair_id'].to_list()
    short_raw.write_parquet(REPORT/'short-hands.parquet')
    return identity, train, {'full': held_full, 'short_conditional': held_short}, base_features, risk_features, short_raw, held_pools


def run():
    started = time.perf_counter()
    identity, train, held, base_features, risk_features, short_raw, held_pools = prepare()
    y = np.array([LABELS[f] for f in train['behavior_family']])
    weights = np.full(train.height, 1/3)
    pair_predictions = {key: frame.select('pair_id', 'table_id', 'behavior_family') for key, frame in held.items()}
    for name, features in [('behavior', base_features), ('risk', risk_features)]:
        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(matrix(train, features), y, sample_weight=weights)
        assert np.array_equal(model.classes_, np.arange(4))
        for history, frame in held.items():
            scores = model.predict_proba(matrix(frame, features))
            if name == 'behavior':
                predicted = np.argmax(scores[:, 1:], axis=1)+1
                pair_predictions[history] = pair_predictions[history].with_columns(
                    pl.Series('predicted_class', predicted), pl.Series('predicted_behavior', [FAMILIES[i-1] for i in predicted]))
            else:
                pair_predictions[history] = pair_predictions[history].with_columns(pl.Series('risk', 1-scores[:, 0]))
        joblib.dump({'model': model, 'features': features}, REPORT/f'{name}.joblib')
        print(f'Fitted confirmation {name} model on training pools only', flush=True)
    data, current, _, _, _, _ = load()
    extra = pl.read_parquet(PARENT/'development.parquet')
    assert sha(PARENT/'development.parquet') == json.loads((PARENT/'development.json').read_text())['output_sha256']
    data = context(data.join(extra, on=['pair_id','hand_id'], validate='1:1'), FEATURES)
    candidate = current+FEATURES+['context_'+c for c in FEATURES]
    training = data.filter(~pl.col('table_id').is_in(held_pools))
    raw_names = [c for c in short_raw.columns if c not in ['pair_id','hand_id','table_id']]
    events = [c for c in raw_names if c.startswith('v2_')]
    labels = pl.read_csv(ROOT/'data/development_labels.csv').filter(pl.col('label') == 1)
    short_targets = short_raw.join(labels.select('pair_id', 'behavior_family'), on='pair_id').join(extra, on=['pair_id','hand_id'], validate='1:1')
    short_targets = context(context(short_targets, events), FEATURES)
    frames = {'full': data.filter(pl.col('table_id').is_in(held_pools)), 'short_conditional': short_targets}
    evidence = pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id', 'hand_id')
    truths, coverage = {}, {}
    for history, frame in frames.items():
        matched = evidence.join(frame.select('pair_id','hand_id'), on=['pair_id','hand_id'], how='semi')
        truths[history] = {k[0]: set(g['hand_id']) for k, g in matched.group_by('pair_id')}
        coverage[history] = {'all_target_pairs': frame['pair_id'].n_unique(), 'pairs_with_retained_annotation': len(truths[history]), 'retained_annotations': matched.height}
        frames[history] = frame.filter(pl.col('pair_id').is_in(list(truths[history]))).join(
            pair_predictions[history].select('pair_id', 'predicted_behavior'), on='pair_id').sort('pair_id','hand_id')
    results, allrows = [], []
    for name, features in [('current', current), ('sequence', candidate)]:
        model = fit(training, features)
        for history, frame in frames.items():
            scores = model.predict(matrix(family_columns(frame, 'predicted_behavior'), features), num_threads=2)
            rows = records(frame, scores, truths[history], history, name)
            prediction = pair_predictions[history]
            metrics = pair_metrics(held[history], prediction['risk'].to_numpy(), prediction['predicted_class'].to_numpy())
            result = {'history': history, 'candidate': name, **summarize(rows), **metrics}
            if history == 'full':
                result['combined_score'] = .7*metrics['pair_ap']+.1*metrics['behavior_map']+.2*result['map5']
            results.append(result); allrows.extend(rows)
            print(json.dumps(result), flush=True)
        joblib.dump({'model': model, 'features': features}, REPORT/f'{name}-ranker.joblib')
    finish(identity, pair_predictions, results, allrows, coverage, started)


def finish(identity, pair_predictions, results, allrows, coverage, started):
    lookup = {(r['history'], r['candidate']): r for r in results}
    deltas = {h: lookup[h, 'sequence']['map5']-lookup[h, 'current']['map5'] for h in pair_predictions}
    family_deltas = {h: {f: lookup[h, 'sequence']['by_family'][f]-lookup[h, 'current']['by_family'][f] for f in FAMILIES} for h in pair_predictions}
    rows = pl.DataFrame(allrows)
    full = rows.filter(pl.col('experiment') == 'full').pivot(on='candidate', index=['pair_id', 'table_id'], values='map5')
    pools = full.group_by('table_id').agg((pl.col('sequence')-pl.col('current')).sum().alias('d'), pl.len().alias('n'))
    ix = np.random.default_rng(849217).integers(0, pools.height, (2000, pools.height))
    interval = np.quantile(pools['d'].to_numpy()[ix].sum(axis=1)/pools['n'].to_numpy()[ix].sum(axis=1), [.025,.975]).tolist()
    passed = deltas['full'] >= .015 and interval[0] > 0 and min(family_deltas['full'].values()) >= -.02 and deltas['short_conditional'] >= 0 and min(family_deltas['short_conditional'].values()) >= -.03
    for section in ['inputs','protected']:
        assert identity[section] == {p: sha(p) for p in identity[section]}
    payload = {'identity': identity, 'results': results, 'evidence_deltas': deltas, 'family_deltas': family_deltas,
               'full_paired_pool_bootstrap_95': interval, 'annotation_coverage': coverage, 'confirmation_passed': bool(passed),
               'protected_unchanged': True, 'seconds': time.perf_counter()-started,
               'limitations': ['Less-reused internal holdout; not independent external evidence or a public score.',
                  'An explicit adaptive exception allowed this confirmation despite failure of the original unseen-family screen.',
                  'Hidden-mechanism retrieval remains uncertain and worsened on average in omitted-family proxies.',
                  'Short evidence is conditional on retained annotations; full pair labels are retained.']}
    rows.write_csv(REPORT/'evidence-pairs.csv')
    for history, frame in pair_predictions.items():
        frame.write_csv(REPORT/f'{history}-pair-predictions.csv')
    (REPORT/'metrics.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'confirmation_passed': bool(passed), 'evidence_deltas': deltas, 'full_ci95': interval,
                      'seconds': payload['seconds']}), flush=True)


if __name__ == '__main__':
    run()
