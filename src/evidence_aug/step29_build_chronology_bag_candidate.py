"""Step 29: build the seed-bagged chronology evidence candidate on top of the public-best v13 coverage submission. NO UPLOAD.

Runs only if experiments/chronology_bag_20260911/results.json shows the step-27 gate passed. Same path as step 13:
  * production chrono_frac ranker refit with seed 7329 on all 372 annotated development targets must reproduce
    outputs_chronology_20260911/evidence-chronology_frac.joblib predictions bit-for-bit (control);
  * bag5 = mean of the five seed models' raw scores (seeds frozen in step 27);
  * per evaluation batch the incumbent ranker's ranks must reproduce the v13 evidence lists exactly for the
    16,060 selected pairs before the bagged lists are written; risk, behavior and evidence outside the selection are
    byte-identical to v13; full membership validation.
Outputs: Claude Version/outputs_chronology_bag_20260911/{plan.json, fit.json, batches/, submission_chronology_bag5.csv, *-validation.json, ready-audit.json}
"""
import os
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import sys
import gc
import json
import time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.step13_build_chronology_candidates import add_chronology, started_at
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import ROOT, matrix
from evidence_capacity_experiment import load_sequence, fit_variant
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from rank_evidence import context, family_columns, PARAMS, EVIDENCE_COLUMNS
from evidence_public_calibration import rank
from validation import validate_submission
from sequence_evidence_candidate import verify_source_cache
import joblib
import numpy as np
import polars as pl

assert ROOT.resolve() == ORIGINAL.resolve(), ROOT
OUT = CV / 'outputs_chronology_bag_20260911'
PRIOR = CV / 'outputs_chronology_20260911'
BAG = CV / 'experiments' / 'chronology_bag_20260911'
BASE = ROOT / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv'
CONDITIONAL = ROOT / 'outputs_v11_conditional'
PACKAGE = CONDITIONAL / 'inference-package'
EXTRA = ['chrono_frac']
CAND = 'chronology_bag5'


def main():
    t0 = time.perf_counter()
    gate = read_json(BAG / 'results.json')['gate']
    assert gate['passed'], f'step-27 gate did not pass: {gate}'
    seeds = read_json(BAG / 'freeze.json')['seeds']
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'batches').mkdir(exist_ok=True)
    protected = check_protected()
    fitted = joblib.load(CONDITIONAL / 'evidence-model.joblib'); caf = fitted['conditional_features']; features = fitted['features']
    prior = joblib.load(PRIOR / 'evidence-chronology_frac.joblib'); assert prior['features'] == features + EXTRA
    selection = ROOT / 'reports' / 'replay-v13' / 'selected-pairs.parquet'
    identity = {'purpose': 'Seed-bagged chronology evidence candidate for the user to submit or not; nothing uploaded. Risk/behavior and evidence outside the 16,060 selected pairs identical to v13.',
                'source_csv_sha256': sha(BASE), 'baseline_model_sha256': sha(CONDITIONAL / 'evidence-model.joblib'), 'prior_chrono_frac_model_sha256': sha(PRIOR / 'evidence-chronology_frac.joblib'),
                'selection_sha256': sha(selection), 'hands_parquet_sha256': sha(ROOT / 'data' / 'hands.parquet'), 'code_sha256': sha(os.path.abspath(__file__)),
                'confirmation': {'bag_freeze_sha256': sha(BAG / 'freeze.json'), 'bag_results_sha256': sha(BAG / 'results.json'), 'gate': gate},
                'seeds': seeds, 'features': '382 + chrono_frac', 'aggregation': 'mean of raw scores', 'recipe': PARAMS, 'protected': protected}
    plan = OUT / 'plan.json'
    if plan.exists():
        assert read_json(plan) == identity, 'plan changed; preserve the existing outputs'
    else:
        write_json(plan, identity)
    verify_source_cache('development'); verify_source_cache('evaluation')
    fitpath = OUT / 'fit.json'
    if not fitpath.exists():
        data, current, _, _, _ = load_sequence()
        data = context(data.join(pl.read_parquet(ROOT / 'reports' / 'equity-evidence-experiment' / 'development.parquet'), on=['pair_id', 'hand_id'], validate='1:1'), EQ)
        data = context(data.join(pl.read_parquet(CONDITIONAL / 'policy' / 'development-hand-features.parquet').drop('table_id'), on=['pair_id', 'hand_id'], validate='1:1'), caf)
        assert features == current + EQ + ['context_' + c for c in EQ] + caf + ['context_' + c for c in caf]
        data = add_chronology(data, started_at('development')).sort('pair_id', 'hand_id')
        assert data['pair_id'].n_unique() == 372
        check = family_columns(data, 'behavior_family'); xe = matrix(check, features + EXTRA)
        shas = {}; gains = {}
        for seed in seeds:
            model = fit_variant(data, features + EXTRA, dict(PARAMS, random_state=seed))
            if seed == 7329:
                np.testing.assert_array_equal(model.predict(xe, num_threads=2), prior['model'].predict(xe, num_threads=2))
                print('seed 7329 refit reproduces outputs_chronology_20260911/evidence-chronology_frac.joblib exactly', flush=True)
            path = OUT / f'evidence-chrono_frac-seed{seed}.joblib'
            joblib.dump({'model': model, 'features': features + EXTRA, 'conditional_features': caf, 'seed': seed}, path)
            np.testing.assert_array_equal(model.predict(xe, num_threads=2), joblib.load(path)['model'].predict(xe, num_threads=2))
            shas[path.name] = sha(path)
            g = model.booster_.feature_importance('gain'); gains[str(seed)] = float(g[(features + EXTRA).index('chrono_frac')] / g.sum())
            print(json.dumps({'seed': seed, 'chrono_frac_gain_share': gains[str(seed)]}), flush=True)
        write_json(fitpath, {'plan_sha256': sha(plan), 'training_pairs': data['pair_id'].n_unique(), 'training_hands': data.height, 'annotations': int(data['relevance'].sum()), 'seed_7329_reproduces_prior_exact': True,
                             'models': shas, 'reload_predictions_exact': True, 'chrono_frac_gain_share': gains})
        del data, check, xe; gc.collect()
    fit = read_json(fitpath); assert fit['plan_sha256'] == sha(plan) and all(sha(OUT / p) == h for p, h in fit['models'].items())
    models = [joblib.load(OUT / f'evidence-chrono_frac-seed{seed}.joblib')['model'] for seed in seeds]
    source = pl.read_csv(BASE, infer_schema=False); selected = pl.read_parquet(selection)
    old = pl.read_parquet(ROOT / 'reports' / 'evidence-public-calibration' / 'selected-pairs.parquet').select('pair_id'); added = selected.join(old, on='pair_id', how='anti')
    assert selected.height == 16060 and added.height == 6060
    eval_started = started_at('evaluation')
    for index, path in enumerate(sorted((ROOT / 'outputs_v2' / 'hands_evaluation').glob('batch_*.parquet'))):
        name = path.stem; dest = OUT / 'batches'; receipt = dest / (name + '.json')
        if receipt.exists():
            saved = read_json(receipt); assert saved['plan_sha256'] == sha(plan) and all(sha(dest / p) == h for p, h in saved['files'].items()); continue
        start = time.perf_counter(); raw = pl.read_parquet(path); events = [c for c in raw.columns if c.startswith('v2_')]
        eq = pl.read_parquet(ROOT / 'reports' / 'evidence-public-calibration' / (name + '-features.parquet'))
        ca = pl.read_parquet(PACKAGE / 'results' / (name + '-evidence.parquet')).drop('table_id')
        feature_receipt = read_json(PACKAGE / 'results' / (name + '.json')); assert all(sha(PACKAGE / 'results' / p) == h for p, h in feature_receipt['files'].items())
        extra_receipt = read_json(ROOT / 'outputs_v13_coverage' / 'batches' / (name + '.json')); assert all(sha(ROOT / 'outputs_v13_coverage' / 'batches' / p) == h for p, h in extra_receipt['files'].items())
        extra = pl.concat([eq.join(ca, on=['pair_id', 'hand_id'], validate='1:1').select(['pair_id', 'hand_id'] + SEQ + EQ + caf),
                           pl.read_parquet(ROOT / 'outputs_v13_coverage' / 'batches' / (name + '-features.parquet')).join(added, on='pair_id', how='semi').select(['pair_id', 'hand_id'] + SEQ + EQ + caf)])
        outputs = {'baseline': [], 'prior_chrono_frac': [], CAND: []}; pools = sorted(raw['table_id'].unique()); count = 0
        for offset in range(0, len(pools), 5):
            part = raw.filter(pl.col('table_id').is_in(pools[offset:offset + 5])).sort('pair_id', 'hand_id')
            part = part.with_columns(pl.Series('tie', np.random.default_rng(931207 + index * 100 + offset).random(part.height))).join(selected, on='pair_id', how='semi')
            if not part.height:
                continue
            frame = part.join(extra, on=['pair_id', 'hand_id'], validate='1:1').join(source.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
            assert frame.height == part.height
            frame = add_chronology(family_columns(context(frame, events + SEQ + EQ + caf), 'predicted_behavior'), eval_started).sort('pair_id', 'hand_id')
            x = matrix(frame, features); count += frame.height
            outputs['baseline'].append(rank(frame, fitted['model'].predict(x, num_threads=2)))
            xe = matrix(frame, features + EXTRA)
            preds = np.stack([m.predict(xe, num_threads=2) for m in models])
            outputs['prior_chrono_frac'].append(rank(frame, preds[0]))
            outputs[CAND].append(rank(frame, preds.mean(axis=0)))
        files = {}
        prior_lists = pl.read_parquet(PRIOR / 'batches' / (name + '-chronology_frac.parquet'))
        for cand, pieces in outputs.items():
            ranked = pl.concat(pieces).sort('pair_id'); expanded = ranked.select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i, c in enumerate(EVIDENCE_COLUMNS)])
            if cand == 'baseline':
                assert expanded.equals(source.join(expanded.select('pair_id'), on='pair_id', how='semi').select(expanded.columns).sort('pair_id')), 'Incumbent ranks differ'
            if cand == 'prior_chrono_frac':
                assert expanded.equals(prior_lists.select(expanded.columns).sort('pair_id')), 'Prior chronology_frac lists differ'
            output = dest / (name + '-' + cand + '.parquet'); expanded.write_parquet(output); files[output.name] = sha(output)
        write_json(receipt, {'plan_sha256': sha(plan), 'files': files, 'selected_hands': count, 'incumbent_evidence_exact': True, 'prior_chrono_frac_exact': True, 'seconds': time.perf_counter() - start})
        print(json.dumps({'batch': index, 'selected_hands': count, 'seconds': round(time.perf_counter() - start, 1)}), flush=True)
        del raw, part, extra, frame, x, xe, outputs; gc.collect()
    replacement = pl.read_parquet(sorted((OUT / 'batches').glob('batch_*-' + CAND + '.parquet'))).sort('pair_id'); assert replacement.height == 16060
    result = pl.concat([source.join(selected, on='pair_id', how='anti'), source.join(selected, on='pair_id', how='semi').drop(EVIDENCE_COLUMNS).join(replacement, on='pair_id', validate='1:1')]).select(source.columns)
    result = source.select('pair_id').join(result, on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(source.select('pair_id', 'risk_score', 'predicted_behavior'))
    assert result.join(selected, on='pair_id', how='anti').equals(source.join(selected, on='pair_id', how='anti'))
    output = OUT / ('submission_' + CAND + '.csv')
    if output.exists():
        assert pl.read_csv(output, infer_schema=False).equals(result)
    else:
        result.write_csv(output)
    validation = validate_submission(output, ROOT / 'data' / 'evaluation_pairs.csv', ROOT / 'data' / 'hands.parquet', ROOT / 'data' / 'seats.parquet', ROOT / 'data' / 'sample_submission.csv').to_dict()
    write_json(OUT / (CAND + '-validation.json'), validation); assert validation['valid'], validation
    prior_csv = pl.read_csv(PRIOR / 'submission_chronology_frac.csv', infer_schema=False)
    changed_vs_v13 = int(result.select(pl.any_horizontal(*[pl.col(c) != source[c] for c in EVIDENCE_COLUMNS]).sum()).item())
    changed_vs_prior = int(result.select(pl.any_horizontal(*[pl.col(c) != prior_csv[c] for c in EVIDENCE_COLUMNS]).sum()).item())
    overlap = result.join(selected, on='pair_id', how='semi').join(prior_csv.select('pair_id', *[pl.col(c).alias('p_' + c) for c in EVIDENCE_COLUMNS]), on='pair_id')
    shared = overlap.select(pl.sum_horizontal(*[pl.col(c).is_in(pl.concat_list([pl.col('p_' + d) for d in EVIDENCE_COLUMNS])) for c in EVIDENCE_COLUMNS]).alias('k'))['k'].mean() if hasattr(pl, 'concat_list') else None
    entry = {'candidate': CAND, 'file': output.name, 'sha256': sha(output), 'valid': validation['valid'], 'changed_evidence_lists_vs_v13': changed_vs_v13, 'changed_evidence_lists_vs_chronology_frac': changed_vs_prior,
             'mean_shared_hands_with_chronology_frac_list': float(shared) if shared is not None else None, 'risk_behavior_exact': True, 'outside_selection_exact': True, 'submitted': False}
    write_json(OUT / 'ready-audit.json', {'completed_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(plan), 'fit_sha256': sha(fitpath), 'entry': entry,
                                          'twenty_incumbent_batch_controls_exact': True, 'twenty_prior_chrono_frac_controls_exact': True, 'no_submissions_attempted': True, 'protected_after': check_protected(), 'seconds': time.perf_counter() - t0})
    print(json.dumps(entry), flush=True)
    print(json.dumps({'ready': True, 'seconds': round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
