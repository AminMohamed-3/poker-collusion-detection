"""Step 13: build two chronology evidence candidates on top of the public-best v13 coverage submission. NO UPLOAD.

Mirrors the previous agent's v14 portfolio path (src/build_diverse_evidence_candidates.py) exactly:
  * production evidence ranker refit on all 372 annotated development targets must reproduce
    outputs_v11_conditional/evidence-model.joblib predictions bit-for-bit before anything else is fitted;
  * chronology_frac  = 382 features + chrono_frac                       (primary: scale-free under shorter histories)
    chronology_full  = 382 features + chrono_rank, chrono_frac, n_hands (secondary)
  * per evaluation batch the incumbent ranker's ranks must reproduce the v13 evidence lists exactly for the
    16,060 selected pairs before the alternative lists are written; risk, behavior and all evidence outside the
    selection are byte-identical to v13; full membership validation with the phase-bearing hands/seats tables.
Outputs: Claude Version/outputs_chronology_20260911/{plan.json, fit.json, batches/, submission_chronology_*.csv, *-validation.json, ready-audit.json}
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
sys.path.insert(0, str(ORIGINAL / 'src'))  # production modules from the ORIGINAL tree, never the copy
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
OUT = CV / 'outputs_chronology_20260911'
BASE = ROOT / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv'
CONDITIONAL = ROOT / 'outputs_v11_conditional'
PACKAGE = CONDITIONAL / 'inference-package'
CANDIDATES = {'chronology_frac': ['chrono_frac'], 'chronology_full': ['chrono_rank', 'chrono_frac', 'n_hands']}
CHRONO = ['chrono_rank', 'chrono_frac', 'n_hands']


def started_at(phase):
    return pl.scan_parquet(ROOT / 'data' / 'hands.parquet').filter(pl.col('phase') == phase).select('hand_id', 'started_at').collect()


def add_chronology(frame, started):
    out = frame.join(started, on='hand_id', how='left', validate='m:1')
    assert out.height == frame.height and out['started_at'].null_count() == 0
    out = out.with_columns(pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'), pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'))
    out = out.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac')).drop('started_at')
    assert out.select(pl.col(CHRONO).is_finite().all()).row(0) == (True, True, True)
    return out


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'batches').mkdir(exist_ok=True)
    protected = check_protected()
    fitted = joblib.load(CONDITIONAL / 'evidence-model.joblib'); caf = fitted['conditional_features']; features = fitted['features']
    assert len(features) == 382
    selection = ROOT / 'reports' / 'replay-v13' / 'selected-pairs.parquet'
    identity = {'purpose': 'Chronology evidence candidates for the user to submit or not; nothing uploaded by this script. Risk/behavior and all evidence outside the 16,060 selected pairs are identical to v13.',
                'source_csv_sha256': sha(BASE), 'baseline_model_sha256': sha(CONDITIONAL / 'evidence-model.joblib'), 'final_reference_features_sha256': sha(CONDITIONAL / 'policy' / 'development-hand-features.parquet'),
                'selection_sha256': sha(selection), 'hands_parquet_sha256': sha(ROOT / 'data' / 'hands.parquet'), 'code_sha256': sha(os.path.abspath(__file__)),
                'confirmation': {'freeze_sha256': sha(CV / 'experiments' / 'chronology_20260910' / 'freeze.json'), 'results_sha256': sha(CV / 'experiments' / 'chronology_20260910' / 'results.json')},
                'candidates': CANDIDATES, 'recipe': PARAMS, 'protected': protected}
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
        check = family_columns(data, 'behavior_family'); x = matrix(check, features)
        refit = fit_variant(data, features, PARAMS)
        np.testing.assert_array_equal(refit.predict(x, num_threads=2), fitted['model'].predict(x, num_threads=2))
        print('production baseline refit reproduces evidence-model.joblib exactly', flush=True)
        models = {}
        for name, extra in CANDIDATES.items():
            model = fit_variant(data, features + extra, PARAMS); path = OUT / f'evidence-{name}.joblib'
            joblib.dump({'model': model, 'features': features + extra, 'conditional_features': caf}, path)
            xe = matrix(check, features + extra)
            np.testing.assert_array_equal(model.predict(xe, num_threads=2), joblib.load(path)['model'].predict(xe, num_threads=2))
            gain = model.booster_.feature_importance('gain'); top = np.argsort(gain)[::-1][:8]
            models[name] = {'top_gain_features': [((features + extra)[i], float(gain[i] / gain.sum())) for i in top], 'chrono_gain_share': {c: float(gain[(features + extra).index(c)] / gain.sum()) for c in extra}}
            print(json.dumps({name: models[name]['chrono_gain_share']}), flush=True)
        write_json(fitpath, {'plan_sha256': sha(plan), 'training_pairs': data['pair_id'].n_unique(), 'training_hands': data.height, 'annotations': int(data['relevance'].sum()), 'baseline_refit_exact': True,
                             'models': {p.name: sha(p) for p in sorted(OUT.glob('*.joblib'))}, 'reload_predictions_exact': True, 'importance': models})
        del data, check, x; gc.collect()
    fit = read_json(fitpath); assert fit['plan_sha256'] == sha(plan) and all(sha(OUT / p) == h for p, h in fit['models'].items())
    models = {name: joblib.load(OUT / f'evidence-{name}.joblib') for name in CANDIDATES}
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
        outputs = {k: [] for k in ['baseline'] + list(CANDIDATES)}; pools = sorted(raw['table_id'].unique()); count = 0; lengths = []
        for offset in range(0, len(pools), 5):
            part = raw.filter(pl.col('table_id').is_in(pools[offset:offset + 5])).sort('pair_id', 'hand_id')
            part = part.with_columns(pl.Series('tie', np.random.default_rng(931207 + index * 100 + offset).random(part.height))).join(selected, on='pair_id', how='semi')
            if not part.height:
                continue
            frame = part.join(extra, on=['pair_id', 'hand_id'], validate='1:1').join(source.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
            assert frame.height == part.height
            frame = add_chronology(family_columns(context(frame, events + SEQ + EQ + caf), 'predicted_behavior'), eval_started).sort('pair_id', 'hand_id')
            x = matrix(frame, features); count += frame.height; lengths.append(frame.group_by('pair_id').len()['len'])
            outputs['baseline'].append(rank(frame, fitted['model'].predict(x, num_threads=2)))
            for cand, bundle in models.items():
                outputs[cand].append(rank(frame, bundle['model'].predict(matrix(frame, bundle['features']), num_threads=2)))
        files = {}
        for cand, pieces in outputs.items():
            ranked = pl.concat(pieces).sort('pair_id'); expanded = ranked.select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i, c in enumerate(EVIDENCE_COLUMNS)])
            if cand == 'baseline':
                assert expanded.equals(source.join(expanded.select('pair_id'), on='pair_id', how='semi').select(expanded.columns).sort('pair_id')), 'Incumbent ranks differ'
            output = dest / (name + '-' + cand + '.parquet'); expanded.write_parquet(output); files[output.name] = sha(output)
        nh = pl.concat(lengths)
        write_json(receipt, {'plan_sha256': sha(plan), 'files': files, 'selected_hands': count, 'selected_pairs': int(nh.len()), 'hands_per_pair_quantiles': [float(nh.quantile(q)) for q in [.05, .5, .95]],
                             'incumbent_evidence_exact': True, 'seconds': time.perf_counter() - start})
        print(json.dumps({'batch': index, 'selected_hands': count, 'seconds': round(time.perf_counter() - start, 1)}), flush=True)
        del raw, part, extra, frame, x, outputs; gc.collect()
    entries = []
    for cand in CANDIDATES:
        replacement = pl.read_parquet(sorted((OUT / 'batches').glob('batch_*-' + cand + '.parquet'))).sort('pair_id'); assert replacement.height == 16060
        result = pl.concat([source.join(selected, on='pair_id', how='anti'), source.join(selected, on='pair_id', how='semi').drop(EVIDENCE_COLUMNS).join(replacement, on='pair_id', validate='1:1')]).select(source.columns)
        result = source.select('pair_id').join(result, on='pair_id', validate='1:1', maintain_order='left')
        assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(source.select('pair_id', 'risk_score', 'predicted_behavior'))
        assert result.join(selected, on='pair_id', how='anti').equals(source.join(selected, on='pair_id', how='anti'))
        output = OUT / ('submission_' + cand + '.csv')
        if output.exists():
            assert pl.read_csv(output, infer_schema=False).equals(result)
        else:
            result.write_csv(output)
        validation = validate_submission(output, ROOT / 'data' / 'evaluation_pairs.csv', ROOT / 'data' / 'hands.parquet', ROOT / 'data' / 'seats.parquet', ROOT / 'data' / 'sample_submission.csv').to_dict()
        write_json(OUT / (cand + '-validation.json'), validation); assert validation['valid'], validation
        changed = int(result.select(pl.any_horizontal(*[pl.col(c) != source[c] for c in EVIDENCE_COLUMNS]).sum()).item())
        entries.append({'candidate': cand, 'file': output.name, 'sha256': sha(output), 'valid': validation['valid'], 'changed_evidence_lists': changed, 'risk_behavior_exact': True, 'outside_selection_exact': True, 'submitted': False})
        print(json.dumps({'candidate': cand, 'changed_evidence_lists': changed, 'valid': validation['valid']}), flush=True)
    assert len({e['sha256'] for e in entries}) == len(CANDIDATES)
    write_json(OUT / 'ready-audit.json', {'completed_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(plan), 'fit_sha256': sha(fitpath), 'entries': entries,
                                          'twenty_incumbent_batch_controls_exact': True, 'no_submissions_attempted': True, 'protected_after': check_protected(), 'seconds': time.perf_counter() - t0})
    print(json.dumps({'ready': True, 'seconds': round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
