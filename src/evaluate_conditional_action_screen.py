"""Fixed pair-risk and evidence tests of nested, cross-fitted action residuals."""
import os
for key in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '2'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from tail_pair_experiment import tails
from robust_pair_experiment import ROOT, LABELS, TRAIN_VIEWS, HELDOUT_VIEW, MODEL_PARAMS, thin, aggregate, matrix
from evidence_capacity_experiment import load_sequence, fit_variant, compare
from rank_evidence import context, family_columns, PARAMS
from equity_evidence_features import FEATURES as EQ
from relative_evidence_experiment import records, summarize

OUT = ROOT / 'reports/conditional-action-experiment'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extra_aggregate(frame, features):
    return frame.group_by('pair_id', 'table_id').agg(
        *[pl.col(c).mean().alias(c + '_mean') for c in features],
        *[pl.col(c).std().fill_null(0).alias(c + '_std') for c in features],
        *[pl.col(c).max().alias(c + '_max') for c in features],
        *[pl.col(c).quantile(.95).alias(c + '_q95') for c in features])


def metrics(y, risk):
    y = y > 0
    weights = np.where(y, .01/y.mean(), .99/(1-y.mean()))
    return {'pair_ap': float(average_precision_score(y, risk)),
            'sparse_ap': float(average_precision_score(y, risk, sample_weight=weights))}


def main():
    report = OUT / 'screen-results'; report.mkdir(exist_ok=False)
    features_path = OUT / 'policy-screen/hand-features.parquet'
    policy = json.loads((OUT / 'policy-screen/summary.json').read_text())
    assert sha(features_path) == policy['hand_features_sha256']
    ca = pl.read_parquet(features_path)
    ca_features = [c for c in ca.columns if c.startswith('ca_')]
    held_pools = sorted(pl.read_csv(ROOT / 'reports/tabicl-screen/held-pairs.csv')['table_id'].unique())
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'policy_features_sha256': sha(features_path), 'held_pools': held_pools, 'new_hand_features': ca_features,
        'pair_parameters': MODEL_PARAMS, 'evidence_parameters': PARAMS,
        'pair_gate': 'Relative to v5: full and shortened AP nondecreasing, average gain >=.001, and neither hypothetical 1% AP decreases. Also require no ordinary or sparse AP decrease against v10 when blending expanded-tree risk equally with the original TabICL risk.',
        'evidence_gate': 'MAP gain >=.005 relative to equity, and no family loss beyond .025.',
        'scope': 'Reused screening fold. Any passing candidate needs further training-pool/family confirmation. Risk candidates keep original behavior and evidence; evidence candidates keep original risk and behavior.'}
    (report / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    labels = pl.read_csv(ROOT / 'data/development_labels.csv')
    raw = pl.read_parquet(sorted((ROOT / 'outputs_v2/hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    original = json.loads((ROOT / 'reports/tail-pair-experiment/metrics.json').read_text())
    base_features = original['base_features'] + original['extra_features']
    combined = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left').with_columns(pl.col(ca_features).fill_null(0))
    assert combined.height == raw.height
    ext_features = [c + suffix for suffix in ['_mean', '_std', '_max', '_q95'] for c in ca_features]
    baseline_full = pl.read_parquet(ROOT / 'outputs_v2/pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    full = baseline_full.join(extra_aggregate(combined, ca_features), on=['pair_id','table_id']).sort('pair_id')
    views = {}
    for fraction, seed in TRAIN_VIEWS + [HELDOUT_VIEW]:
        raw_sample = thin(raw, fraction, seed)
        added_sample = thin(combined, fraction, seed)
        assert raw_sample.select('pair_id','hand_id').equals(added_sample.select('pair_id','hand_id'))
        view = aggregate(raw_sample, names).join(tails(raw_sample, events), on=['pair_id','table_id']).join(labels, on='pair_id')
        views[(fraction, seed)] = view.join(extra_aggregate(added_sample, ca_features), on=['pair_id','table_id']).select(full.columns).sort('pair_id')
    train = pl.concat([full.filter(~pl.col('table_id').is_in(held_pools))] + [views[k].filter(~pl.col('table_id').is_in(held_pools)) for k in TRAIN_VIEWS])
    held_frames = {'full': full.filter(pl.col('table_id').is_in(held_pools)),
                   'shortened': views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(held_pools))}
    old = pl.read_csv(ROOT / 'reports/tail-pair-experiment/predictions.csv').filter((pl.col('candidate') == 'augmented_with_event_tails') & (pl.col('fold') == 0))
    tab = np.load(ROOT / 'reports/tabicl-screen/predictions.npz', allow_pickle=False)
    known_held = pl.read_csv(ROOT / 'reports/tabicl-screen/held-pairs.csv')
    pair_rows, pair_predictions = [], []
    for name, features in [('v5_control', base_features), ('conditional', base_features + ext_features)]:
        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(matrix(train, features), np.array([LABELS[f] for f in train['behavior_family']]), sample_weight=np.full(train.height, 1/3))
        joblib.dump({'model': model, 'features': features}, report / (name + '-pair.joblib'))
        for history, frame in held_frames.items():
            assert frame['pair_id'].to_list() == known_held['pair_id'].to_list()
            risk = 1-model.predict_proba(matrix(frame, features))[:, 0]
            if name == 'v5_control': np.testing.assert_array_equal(risk, old.filter(pl.col('history') == history).sort('pair_id')['risk'].to_numpy())
            y = frame['label'].to_numpy()
            for candidate, values in [(name, risk), ('v10_control' if name == 'v5_control' else 'conditional_hybrid', .5*(risk+tab[history]))]:
                pair_rows.append({'history': history, 'candidate': candidate, **metrics(y, values)})
                pair_predictions.extend({'pair_id': p, 'history': history, 'candidate': candidate, 'risk': float(v)} for p,v in zip(frame['pair_id'],values))
    pl.DataFrame(pair_predictions).write_csv(report / 'pair-predictions.csv')
    data, current, truths, pairpred, evidence_pools = load_sequence()
    assert set(evidence_pools) == set(held_pools)
    data = context(data.join(pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/development.parquet'), on=['pair_id','hand_id'], validate='1:1'), EQ)
    base_evidence = current + EQ + ['context_'+c for c in EQ]
    data = data.join(ca.drop('table_id'), on=['pair_id','hand_id'], how='left', validate='1:1').with_columns(pl.col(ca_features).fill_null(0))
    data = context(data, ca_features)
    train_hands = data.filter(~pl.col('table_id').is_in(held_pools))
    held_hands = data.filter(pl.col('table_id').is_in(held_pools)).join(pairpred.select('pair_id','predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id','hand_id')
    held_input = family_columns(held_hands, 'predicted_behavior')
    evidence_rows, evidence_metrics = [], []
    for name, features in [('equity', base_evidence), ('conditional', base_evidence + ca_features + ['context_'+c for c in ca_features])]:
        if name == 'equity':
            bundle = joblib.load(ROOT / 'reports/equity-evidence-experiment/equity.joblib'); assert bundle['features'] == features; model = bundle['model']
        else:
            model = fit_variant(train_hands, features, PARAMS)
            joblib.dump({'model': model, 'features': features}, report / 'conditional-evidence.joblib')
        score = model.predict(matrix(held_input, features), num_threads=2)
        if name == 'equity':
            prior = pl.read_parquet(ROOT / 'reports/equity-evidence-experiment/equity-scores.parquet')
            assert prior['pair_id'].to_list() == held_hands['pair_id'].to_list()
            np.testing.assert_array_equal(prior['score'].to_numpy(), score)
        rows = records(held_hands, score, truths, 'screen', name); evidence_rows.extend(rows)
        evidence_metrics.append({'candidate': name, **summarize(rows)})
        held_hands.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(report / (name+'-evidence-scores.parquet'))
    evidence_comparison = compare(evidence_rows, 'conditional', baseline='equity')
    lookup = {(r['history'],r['candidate']):r for r in pair_rows}
    ordinary = [lookup[h,'conditional']['pair_ap']-lookup[h,'v5_control']['pair_ap'] for h in held_frames]
    sparse = [lookup[h,'conditional']['sparse_ap']-lookup[h,'v5_control']['sparse_ap'] for h in held_frames]
    hybrid = [lookup[h,'conditional_hybrid'][m]-lookup[h,'v10_control'][m] for h in held_frames for m in ['pair_ap','sparse_ap']]
    advance_pair = min(ordinary) >= 0 and np.mean(ordinary) >= .001 and min(sparse) >= 0 and min(hybrid) >= 0
    advance_evidence = evidence_comparison['delta'] >= .005 and min(evidence_comparison['by_family_delta'].values()) >= -.025
    pl.DataFrame(evidence_rows).write_csv(report / 'evidence-pairs.csv')
    result = {'pair_results':pair_rows,'evidence_results':evidence_metrics,'evidence_comparison':evidence_comparison,
        'advance_pair':bool(advance_pair),'advance_evidence':bool(advance_evidence),'v5_and_equity_controls_reproduced_exactly':True}
    (report / 'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
