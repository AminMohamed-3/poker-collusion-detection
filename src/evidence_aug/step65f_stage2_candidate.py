"""Step 65f: production candidates for the per-family second-stage evidence re-ranker. NO UPLOAD.

Held confirmation (steps 65c/65d, folds 0-2, evaluated once): the variant 'allfam_top20_struct' (every pair's top-20
stage-1 candidates re-ranked by its predicted family's stage-2 model; structural relational-action features + chronology
+ stage-1 score/rank) passed every pre-registered criterion: pooled MAP@5 +.0411 vs chrono1, pool bootstrap
[+.0266, +.0563], gains in all three true families in all three folds, both halves positive. The CI-only primary passed
the same criteria except the per-fold family-loss cap (one true-transfer pair predicted isolation in fold 1).

Production, mirroring step 13 exactly for stage 1:
  * stage 1 = outputs_chronology_20260911/evidence-chronology_frac.joblib (382 + chrono_frac); a refit on all 372
    development targets must reproduce its predictions bit-for-bit; its 5-way pool-hash out-of-fold scores
    (audit-trunc-v1 splits, the recipe of steps 14/65c) on the same 372 targets feed stage-2 training;
  * stage 2 = one LGBMClassifier (step-14 CLF, 3 seeds averaged) per TRUE family on the top-20 out-of-fold candidates
    of the training pairs; features = structural (step 65) + chronology + s1/s1_rank. Two feature variants:
    chrono3 (chrono_rank, chrono_frac, n_hands: the held-confirmed design) and chrono1 (chrono_frac only: a
    length-robust variant with audit support only, step 65g);
  * per evaluation batch the stage-1 ranks must reproduce the incumbent evidence lists (chronology_frac = M3 = S lists)
    for the 16,060 selected pairs before anything is written; each selected pair is routed by its submission
    predicted_behavior; its top-20 is re-ranked by that family's stage-2, the tail keeps the stage-1 order;
  * candidates: M3 risk/behaviour + stage-2 evidence (primary: chrono3; secondary: chrono1) and S risk/behaviour +
    stage-2 chrono3 evidence (attribution against the measured .89055). Evidence outside the selection unchanged.
Prediction JSON is written before any candidate; receipts carry hashes; protected files re-checked.
Outputs: Claude Version/outputs_stage2_20260911/{prediction.json, fit.json, stage2-models.joblib, batches/, submission_*.csv,
*-validation.json, receipt.json}."""
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
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step13_build_chronology_candidates import started_at, add_chronology, OUT as CHRONO_OUT, CONDITIONAL, PACKAGE
from evidence_aug.step14_audit_truncation import split_of, SPLITS, CLF, TOP
from evidence_aug.step65_structural_evidence import SEEDS
from evidence_aug.step65c_family_stage2_fit import OUT as FS2, CHRONO, CHRONO1, S1, stage2_fit, stage2_predict, with_rank
import joblib
import numpy as np
import polars as pl

assert ROOT.resolve() == ORIGINAL.resolve(), ROOT
OUT = CV / 'outputs_stage2_20260911'
BASE = CHRONO_OUT / 'submission_chronology_frac.csv'
M3 = CV / 'outputs_multiview_20260911' / 'submission_multiview_M3_lengthviews.csv'
S = CV / 'outputs_blend_20260911' / 'submission_blend_w0.5_plus_R1.csv'
STAGE1 = CHRONO_OUT / 'evidence-chronology_frac.joblib'
VARIANTS = {'chrono3': CHRONO, 'chrono1': CHRONO1}
CANDIDATES = {'M3_stage2_chrono3': ('M3', 'chrono3'), 'M3_stage2_chrono1': ('M3', 'chrono1'), 'S_stage2_chrono3': ('S', 'chrono3')}
KEYS = ['pair_id', 'hand_id']


def log(*a):
    print(*a, flush=True)


def fit_stage2(caf, features, chrono, held_results):
    """Stage-1 refit check, 5-way out-of-fold stage-1 scores on the 372 targets, per-family stage-2 models."""
    data, current, _, _, _ = load_sequence()
    data = context(data.join(pl.read_parquet(ROOT / 'reports' / 'equity-evidence-experiment' / 'development.parquet'), on=KEYS, validate='1:1'), EQ)
    data = context(data.join(pl.read_parquet(CONDITIONAL / 'policy' / 'development-hand-features.parquet').drop('table_id'), on=KEYS, validate='1:1'), caf)
    assert features == current + EQ + ['context_' + c for c in EQ] + caf + ['context_' + c for c in caf]
    data = add_chronology(data, started_at('development')).sort('pair_id', 'hand_id')
    assert data['pair_id'].n_unique() == 372 and 'table_id' in data.columns and 'relevance' in data.columns
    f1 = chrono['features']; assert f1 == features + CHRONO1
    check = family_columns(data, 'behavior_family'); x = matrix(check, f1)
    refit = fit_variant(data, f1, PARAMS)
    np.testing.assert_array_equal(refit.predict(x, num_threads=2), chrono['model'].predict(x, num_threads=2))
    log('production chronology_frac refit reproduces evidence-chronology_frac.joblib exactly')
    data = data.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
    oof = np.full(data.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(data['split'].to_numpy() == s)[0]
        m = fit_variant(data.filter(pl.col('split') != s), f1, PARAMS)
        oof[idx] = m.predict(matrix(family_columns(data[idx], 'behavior_family'), f1), num_threads=2)
    assert np.isfinite(oof).all()
    data = with_rank(data.with_columns(pl.Series('s1', oof)), 's1', 's1_rank')
    truths = {k[0]: set(g['hand_id']) for k, g in data.filter(pl.col('relevance') == 1).group_by('pair_id')}
    oof_map = per_pair_map(data.select('pair_id', 'hand_id', 'table_id', 's1').rename({'s1': 'score'}), truths)
    labels = dict(data.select('pair_id', 'behavior_family').unique().iter_rows())
    recall20 = data.filter(pl.col('relevance') == 1).select((pl.col('s1_rank') <= TOP).mean()).item()
    feat = pl.read_parquet(FS2 / 'dev-structural-features.parquet'); struct = [c for c in feat.columns if c not in KEYS]
    overlap = sorted(set(struct) & set(data.columns))
    data = data.join(feat, on=KEYS, how='left', validate='1:1')  # as in step 65c: overlapping names keep the frame's column
    assert data.height == 45129 and data.select(pl.sum_horizontal(pl.col(struct).null_count())).item() == 0
    models = {}; cols = {}; info = {'stage1_oof_map5_372': float(np.mean(list(oof_map.values()))), 'stage1_oof_map5_by_family': {f: float(np.mean([v for p, v in oof_map.items() if labels[p] == f])) for f in FAMILIES},
                                    'stage1_oof_recall_at_20': float(recall20), 'struct_frame_name_overlap_kept_frame_column': overlap, 'stage2': {}}
    for variant, chron in VARIANTS.items():
        cols[variant] = struct + chron + S1
        for fam in FAMILIES:
            ms, nq, npos = stage2_fit(data, cols[variant], fam, True); models[f'{variant}/{fam}'] = ms
            g = np.mean([m.booster_.feature_importance('gain') for m in ms], axis=0); order = np.argsort(g)[::-1][:8]
            info['stage2'][f'{variant}/{fam}'] = {'training_pairs': int(nq), 'training_rows': int(data.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP)).height), 'training_listed': int(npos),
                                                  'top_gain': [(cols[variant][i], round(float(g[i] / g.sum()), 3)) for i in order]}
            log(json.dumps({f'{variant}/{fam}': info['stage2'][f'{variant}/{fam}']['top_gain'][:5]}))
    bundle = {'models': models, 'cols': cols, 'struct': struct, 'clf': CLF, 'seeds': list(SEEDS), 'top': TOP, 'stage1_sha256': sha(STAGE1), 'held_results_sha256': sha(held_results)}
    joblib.dump(bundle, OUT / 'stage2-models.joblib'); reloaded = joblib.load(OUT / 'stage2-models.joblib')
    top = data.filter(pl.col('s1_rank') <= TOP)
    for key, ms in models.items():
        v = key.split('/')[0]
        np.testing.assert_array_equal(stage2_predict(ms, top, cols[v]), stage2_predict(reloaded['models'][key], top, cols[v]))
    info['reload_predictions_exact'] = True; info['models_sha256'] = sha(OUT / 'stage2-models.joblib')
    del data, check, x, top; gc.collect()
    return info


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'batches').mkdir(exist_ok=True)
    protected = check_protected()
    held_results = FS2 / 'results.json'; held = read_json(held_results); gate = held['gates']['allfam_top20_struct']
    assert gate['passed'] and held['gates']['hybridCI_top20_struct']['passed'] is False
    fitted = joblib.load(CONDITIONAL / 'evidence-model.joblib'); caf = fitted['conditional_features']; features = fitted['features']; assert len(features) == 382
    chrono = joblib.load(STAGE1)
    selection = ROOT / 'reports' / 'replay-v13' / 'selected-pairs.parquet'
    struct_receipt = read_json(FS2 / 'eval_structural' / 'receipt.json'); assert struct_receipt['selected_pairs'] == 16060
    identity = {'purpose': 'Per-family second-stage evidence candidates for the user to submit or not; nothing uploaded by this script. Risk/behavior and all evidence outside the 16,060 selected pairs are identical to the source (M3 or S).',
                'sources': {'M3': sha(M3), 'S': sha(S), 'base_chronology_frac': sha(BASE)}, 'stage1_sha256': sha(STAGE1), 'baseline_model_sha256': sha(CONDITIONAL / 'evidence-model.joblib'),
                'final_reference_features_sha256': sha(CONDITIONAL / 'policy' / 'development-hand-features.parquet'), 'selection_sha256': sha(selection), 'hands_parquet_sha256': sha(ROOT / 'data' / 'hands.parquet'),
                'dev_structural_sha256': sha(FS2 / 'dev-structural-features.parquet'), 'eval_structural_receipt_sha256': sha(FS2 / 'eval_structural' / 'receipt.json'), 'code_sha256': sha(os.path.abspath(__file__)),
                'confirmation': {'freeze_sha256': sha(FS2 / 'freeze.json'), 'predictions_frozen_sha256': sha(FS2 / 'predictions-frozen.json'), 'results_sha256': sha(held_results)},
                'variants': VARIANTS, 'candidates': CANDIDATES, 'stage2_recipe': {'clf': CLF, 'seeds': list(SEEDS), 'top': TOP, 'rows': 'top-20 stage-1 OOF candidates per TRUE family, 372 targets'}, 'protected': protected}
    plan = OUT / 'plan.json'
    if plan.exists():
        assert read_json(plan) == identity, 'plan changed; preserve the existing outputs'
    else:
        write_json(plan, identity)
    # pre-registered prediction, written before any fit or candidate
    pred_path = OUT / 'prediction.json'
    if not pred_path.exists():
        m3pred = read_json(CV / 'outputs_multiview_20260911' / 'prediction_M3.json')['prediction']
        e_delta = [0.020, 0.055]
        write_json(pred_path, {
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'plan_sha256': sha(plan),
            'held_confirmation': {'variant': 'allfam_top20_struct', 'pooled_delta_map5_vs_chrono1': gate['pooled_delta'], 'pool_bootstrap_95': gate['pooled_pool_bootstrap_95'], 'by_true_family': gate['pooled_delta_by_true_family'], 'halves': gate['pooled_halves'],
                                 'per_fold_delta': {k: v['candidates']['allfam_top20_struct']['delta'] for k, v in held['folds'].items()}, 'gate_passed': True,
                                 'selection_note': 'The pre-registered primary was the CI-only hybrid (+.0259 pooled, [+.0152, +.0381]); it failed only the per-fold family-loss cap through one true-transfer pair predicted isolation in fold 1. allfam_top20_struct was a pre-registered variant of the same freeze and passed every criterion; it is chosen over the hybrid on the held result, which is one selection among four pre-registered variants.'},
            'rule': 'Each of the 16,060 selected pairs keeps its stage-1 (chronology_frac) order except that its top-20 stage-1 candidates are re-ranked by the stage-2 model of its predicted family (submission behaviour column); risk and behaviour are those of the source (M3 or S).',
            'evidence_map_delta_interval': e_delta, 'evidence_interval_note': 'Held [+.027, +.056] under near-perfect routing and full development histories; widened downward for routing error (behaviour accuracy ~.976 on the public split) and the shorter evaluation histories (median selected pair ~78 hands vs ~121 in development); n_hands/chrono_rank shift is the reason for the chrono1 variant.',
            'composite_intervals': {'S_stage2_chrono3 minus S (.89055)': [round(.2 * e_delta[0], 4), round(.2 * e_delta[1], 4)], 'M3_stage2_chrono3 minus M3': [round(.2 * e_delta[0], 4), round(.2 * e_delta[1], 4)],
                                    'M3_stage2_chrono3 minus S (.89055)': [round(m3pred['interval_M3_minus_S'][0] + .2 * e_delta[0], 4), round(m3pred['interval_M3_minus_S'][1] + .2 * e_delta[1], 4)],
                                    'M3_stage2_chrono1 minus M3_stage2_chrono3': [-0.004, 0.004]},
            'point_estimate_note': 'E lever point ~+.008 composite (held .041 x .2); M3 minus S point ~+.012 (step 64); together ~+.020 over .89055 -> ~.910 if both transfer at their point estimates.',
            'decision_rule': 'Measure the primary (M3_stage2_chrono3) against S or M3 (whichever is known). A gain below +.002 says the stage-2 does not transfer (routing or length shift): keep the previous evidence and try chrono1 only if the length shift is the suspected cause. No re-tuning on outcomes.',
            'label_free': 'Evaluation side uses the frozen chronology_frac ranker, label-trained stage-2 models, the submission behaviour column and label-free structural hand features. No leaderboard feedback used to fit anything.',
            'not_uploaded_by_claude': True})
    verify_source_cache('development'); verify_source_cache('evaluation')
    fitpath = OUT / 'fit.json'
    if not fitpath.exists():
        info = fit_stage2(caf, features, chrono, held_results)
        write_json(fitpath, {'plan_sha256': sha(plan), 'prediction_sha256': sha(pred_path), **info})
    fit = read_json(fitpath); assert fit['plan_sha256'] == sha(plan) and fit['models_sha256'] == sha(OUT / 'stage2-models.joblib')
    bundle = joblib.load(OUT / 'stage2-models.joblib'); struct = bundle['struct']
    sources = {'M3': pl.read_csv(M3, infer_schema=False), 'S': pl.read_csv(S, infer_schema=False)}; base = pl.read_csv(BASE, infer_schema=False)
    for name, src in sources.items():
        assert src.select('pair_id', 'predicted_behavior', *EVIDENCE_COLUMNS).equals(base.select('pair_id', 'predicted_behavior', *EVIDENCE_COLUMNS)), name
    source = base; selected = pl.read_parquet(selection)
    old = pl.read_parquet(ROOT / 'reports' / 'evidence-public-calibration' / 'selected-pairs.parquet').select('pair_id'); added = selected.join(old, on='pair_id', how='anti')
    assert selected.height == 16060 and added.height == 6060
    eval_started = started_at('evaluation'); f1 = chrono['features']
    for index, path in enumerate(sorted((ROOT / 'outputs_v2' / 'hands_evaluation').glob('batch_*.parquet'))):
        name = path.stem; dest = OUT / 'batches'; receipt = dest / (name + '.json')
        if receipt.exists():
            saved = read_json(receipt); assert saved['plan_sha256'] == sha(plan) and all(sha(dest / p) == h for p, h in saved['files'].items()); continue
        start = time.perf_counter(); raw = pl.read_parquet(path); events = [c for c in raw.columns if c.startswith('v2_')]
        eq = pl.read_parquet(ROOT / 'reports' / 'evidence-public-calibration' / (name + '-features.parquet'))
        ca = pl.read_parquet(PACKAGE / 'results' / (name + '-evidence.parquet')).drop('table_id')
        feature_receipt = read_json(PACKAGE / 'results' / (name + '.json')); assert all(sha(PACKAGE / 'results' / p) == h for p, h in feature_receipt['files'].items())
        extra_receipt = read_json(ROOT / 'outputs_v13_coverage' / 'batches' / (name + '.json')); assert all(sha(ROOT / 'outputs_v13_coverage' / 'batches' / p) == h for p, h in extra_receipt['files'].items())
        extra = pl.concat([eq.join(ca, on=KEYS, validate='1:1').select(KEYS + SEQ + EQ + caf),
                           pl.read_parquet(ROOT / 'outputs_v13_coverage' / 'batches' / (name + '-features.parquet')).join(added, on='pair_id', how='semi').select(KEYS + SEQ + EQ + caf)])
        sfile = FS2 / 'eval_structural' / (name + '.parquet'); assert sha(sfile) == struct_receipt['files'][sfile.name]; sfeat = pl.read_parquet(sfile)
        outputs = {k: [] for k in ['incumbent'] + list(VARIANTS)}; pools = sorted(raw['table_id'].unique()); count = 0; diag = []
        for offset in range(0, len(pools), 5):
            part = raw.filter(pl.col('table_id').is_in(pools[offset:offset + 5])).sort('pair_id', 'hand_id')
            part = part.with_columns(pl.Series('tie', np.random.default_rng(931207 + index * 100 + offset).random(part.height))).join(selected, on='pair_id', how='semi')
            if not part.height:
                continue
            frame = part.join(extra, on=KEYS, validate='1:1').join(source.select('pair_id', 'predicted_behavior'), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
            assert frame.height == part.height
            frame = add_chronology(family_columns(context(frame, events + SEQ + EQ + caf), 'predicted_behavior'), eval_started).sort('pair_id', 'hand_id')
            s1 = chrono['model'].predict(matrix(frame, f1), num_threads=2); count += frame.height
            outputs['incumbent'].append(rank(frame, s1))
            frame = with_rank(frame.with_columns(pl.Series('s1', s1)), 's1', 's1_rank').join(sfeat, on=KEYS, how='left', validate='1:1').sort('pair_id', 'hand_id')
            assert frame.height == part.height and frame.select(pl.sum_horizontal(pl.col(struct).null_count())).item() == 0
            base_s = frame['s1'].to_numpy(); in_top = (frame['s1_rank'] <= TOP).to_numpy(); fam_col = frame['predicted_behavior'].to_numpy()
            for variant in VARIANTS:
                pred = np.full(frame.height, np.nan)
                for fam in FAMILIES:
                    mask = in_top & (fam_col == fam)
                    if mask.any():
                        pred[mask] = stage2_predict(bundle['models'][f'{variant}/{fam}'], frame.filter(pl.Series(mask)), bundle['cols'][variant])
                final = np.where(in_top, pred + 10.0, base_s - 1e6); assert np.isfinite(final).all()
                outputs[variant].append(rank(frame, final))
            diag.append(frame.select('pair_id', 'predicted_behavior').unique())
        files = {}; lists = {}
        for cand, pieces in outputs.items():
            ranked = pl.concat(pieces).sort('pair_id'); expanded = ranked.select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i, c in enumerate(EVIDENCE_COLUMNS)])
            if cand == 'incumbent':
                assert expanded.equals(source.join(expanded.select('pair_id'), on='pair_id', how='semi').select(expanded.columns).sort('pair_id')), 'Incumbent ranks differ'
            lists[cand] = expanded
            output = dest / (name + '-' + cand + '.parquet'); expanded.write_parquet(output); files[output.name] = sha(output)
        fams = pl.concat(diag)
        changed = {}
        for variant in VARIANTS:
            j = lists['incumbent'].join(lists[variant], on='pair_id', suffix='_v').join(fams, on='pair_id')
            ch = j.select(pl.any_horizontal(*[pl.col(c) != pl.col(c + '_v') for c in EVIDENCE_COLUMNS]).alias('changed'), 'predicted_behavior')
            changed[variant] = {'pairs_with_changed_top5': int(ch['changed'].sum()), 'by_family': {k: int(v) for k, v in ch.group_by('predicted_behavior').agg(pl.col('changed').sum()).iter_rows()}}
        j = lists['chrono3'].join(lists['chrono1'], on='pair_id', suffix='_v')
        agree = int(j.select(pl.all_horizontal(*[pl.col(c) == pl.col(c + '_v') for c in EVIDENCE_COLUMNS]).sum()).item())
        write_json(receipt, {'plan_sha256': sha(plan), 'files': files, 'selected_hands': count, 'selected_pairs': int(lists['incumbent'].height), 'incumbent_evidence_exact': True, 'changed_vs_incumbent': changed,
                             'chrono3_chrono1_identical_top5_pairs': agree, 'seconds': time.perf_counter() - start})
        log(json.dumps({'batch': index, 'selected_hands': count, 'changed': {v: c['pairs_with_changed_top5'] for v, c in changed.items()}, 'variants_agree': agree, 'seconds': round(time.perf_counter() - start, 1)}))
        del raw, part, extra, frame, outputs, sfeat; gc.collect()
    entries = []
    for cand, (src_name, variant) in CANDIDATES.items():
        src = sources[src_name]
        replacement = pl.read_parquet(sorted((OUT / 'batches').glob('batch_*-' + variant + '.parquet'))).sort('pair_id'); assert replacement.height == 16060
        result = pl.concat([src.join(selected, on='pair_id', how='anti'), src.join(selected, on='pair_id', how='semi').drop(EVIDENCE_COLUMNS).join(replacement, on='pair_id', validate='1:1')]).select(src.columns)
        result = src.select('pair_id').join(result, on='pair_id', validate='1:1', maintain_order='left')
        assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(src.select('pair_id', 'risk_score', 'predicted_behavior'))
        assert result.join(selected, on='pair_id', how='anti').equals(src.join(selected, on='pair_id', how='anti'))
        output = OUT / ('submission_' + cand + '.csv')
        if output.exists():
            assert pl.read_csv(output, infer_schema=False).equals(result)
        else:
            result.write_csv(output)
        validation = validate_submission(output, ROOT / 'data' / 'evaluation_pairs.csv', ROOT / 'data' / 'hands.parquet', ROOT / 'data' / 'seats.parquet', ROOT / 'data' / 'sample_submission.csv').to_dict()
        write_json(OUT / (cand + '-validation.json'), validation); assert validation['valid'], validation
        changed = int(result.select(pl.any_horizontal(*[pl.col(c) != src[c] for c in EVIDENCE_COLUMNS]).sum()).item())
        entries.append({'candidate': cand, 'source': src_name, 'variant': variant, 'file': output.name, 'sha256': sha(output), 'valid': validation['valid'], 'changed_evidence_lists_vs_source': changed, 'risk_behavior_exact': True, 'outside_selection_exact': True, 'submitted': False})
        log(json.dumps({'candidate': cand, 'changed_evidence_lists': changed, 'valid': validation['valid'], 'sha256': entries[-1]['sha256']}))
    assert len({e['sha256'] for e in entries}) == len(CANDIDATES)
    write_json(OUT / 'receipt.json', {'completed_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(plan), 'prediction_sha256': sha(pred_path), 'fit_sha256': sha(fitpath), 'entries': entries,
                                      'twenty_incumbent_batch_controls_exact': True, 'prediction_written_first': True, 'no_submissions_attempted': True, 'protected_after': check_protected(), 'seconds': time.perf_counter() - t0})
    log(json.dumps({'ready': True, 'seconds': round(time.perf_counter() - t0, 1)}))


if __name__ == '__main__':
    main()
