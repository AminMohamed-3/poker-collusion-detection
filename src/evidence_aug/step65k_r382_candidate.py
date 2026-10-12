"""Step 65k: production candidates for the per-family second stage WITH the 382 stage-1 features (r382). NO UPLOAD.
Held outcome (step 65j, one read): the pre-registered primary (r382 for every family) FAILED its gate through the
coordinated-isolation family (pooled -.018, fold 2 -.043) while transfer / soft play gained (+.022 / +.024; pooled all
families +.013 [+.0015, +.026]). The POST-HOC hybrid 'r382 for predicted transfer / soft play, the step-65c/65f struct
model for predicted isolation' (step 65j2, derived from the same read, no new truth) is +.016 [+.005, +.028], per fold
+.020 / +.017 / +.012, both halves positive, true-family minimum -.007. It is one selection among the family subsets made
after seeing the per-family results and is labelled post-hoc everywhere. Everything else mirrors step 65f exactly:
stage 1 = the production chronology_frac ranker (refit must reproduce it bit-for-bit; 5-way pool-hash OOF on the 372
targets feeds stage 2); stage 2 = one LGBMClassifier per TRUE family on the top-20 OOF candidates with
structural + chrono3 + s1/s1_rank + the ranker's own features; per evaluation batch the stage-1 ranks must reproduce
the incumbent lists AND the step-65f chrono3 stage-2 lists must be reproduced from the step-65f bundle before anything
is written (reproduction control of the previous stage); each selected pair is routed by its submission behaviour.
Candidates: M3 risk/behaviour + r382 evidence (primary) and S risk/behaviour + r382 evidence (attribution).
Prediction JSON is written before any candidate; receipts carry hashes; protected files re-checked.
Outputs: Claude Version/outputs_stage2_r382_20260911/{plan.json, prediction.json, fit.json, stage2-models.joblib, batches/,
submission_*.csv, *-validation.json, receipt.json}."""
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
from evidence_aug.step65c_family_stage2_fit import OUT as FS2, CHRONO, CHRONO1, S1, stage2_predict, with_rank
from evidence_aug.step65f_stage2_candidate import OUT as PREV_OUT, BASE, M3, S, STAGE1, KEYS, log
from evidence_aug.step65i_r382_stage2_fit import OUT as FS2R, FF50, stage2_fit
import joblib
import numpy as np
import polars as pl

assert ROOT.resolve() == ORIGINAL.resolve(), ROOT
OUT = CV / 'outputs_stage2_r382_20260911'
PARAMS_BY_VARIANT = {'allfam_top20_r382': CLF, 'allfam_top20_r382ff50': FF50, 'hybrid_DTSP_r382': CLF}
CI = 'coordinated_isolation'


def choose_variant(held):
    """Pre-registered decision rule of the step-65i freeze: the primary if it passed; else a passing secondary (a recorded selection);
    else the post-hoc hybrid of step 65j2 if its derived numbers are clean (pooled CI above zero, every fold nonnegative, no true family below -.01)."""
    gates = held['gates']
    if gates['allfam_top20_r382']['passed']:
        return 'allfam_top20_r382', 'primary passed', gates['allfam_top20_r382']
    if gates.get('allfam_top20_r382ff50', {}).get('passed'):
        return 'allfam_top20_r382ff50', 'primary FAILED; secondary ff50 passed (one selection among three variants; interval widened)', gates['allfam_top20_r382ff50']
    post = read_json(FS2R / 'posthoc_hybrid.json')['variants']['hybrid_DTSP_r382']
    clean = post['pool_bootstrap_95'][0] > 0 and min(post['per_fold'].values()) >= 0 and min(post['by_true_family'].values()) >= -.01
    if clean:
        g = {'pooled_delta': post['pooled_delta_vs_allfam_top20_struct'], 'pooled_pool_bootstrap_95': post['pool_bootstrap_95'], 'pooled_delta_by_true_family': post['by_true_family'], 'pooled_halves': post['halves'], 'per_fold_delta': post['per_fold'], 'post_hoc': True}
        return 'hybrid_DTSP_r382', 'primary and secondaries FAILED the pre-registered gate (isolation family loss); POST-HOC hybrid: r382 models for predicted transfer / soft play, step-65f chrono3 struct model for predicted isolation; derived from the same read (step 65j2), selection made after seeing per-family results', g
    raise SystemExit('no variant passed and the post-hoc hybrid is not clean; keep M3_stage2_chrono3')


def fit_stage2(caf, features, chrono, held_results, variant):
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
    feat = pl.read_parquet(FS2 / 'dev-structural-features.parquet'); struct = [c for c in feat.columns if c not in KEYS]
    overlap = sorted(set(struct) & set(data.columns))
    data = data.join(feat, on=KEYS, how='left', validate='1:1')  # as in steps 65c/65f: overlapping names keep the frame's column
    assert data.height == 45129 and data.select(pl.sum_horizontal(pl.col(struct).null_count())).item() == 0
    r382 = [f for f in features if not f.startswith('family_') and f not in struct + CHRONO + S1]
    assert all(f in data.columns for f in r382)
    # the step-65f chrono3 stage-2 must be reproduced from this frame (same rows, same features, same recipe)
    prev = joblib.load(PREV_OUT / 'stage2-models.joblib'); top = data.filter(pl.col('s1_rank') <= TOP)
    for fam in FAMILIES:
        ms, _, _ = stage2_fit(data, struct + CHRONO + S1, fam, CLF)
        np.testing.assert_array_equal(stage2_predict(ms, top, prev['cols']['chrono3']), stage2_predict(prev['models'][f'chrono3/{fam}'], top, prev['cols']['chrono3']))
    log('step-65f chrono3 stage-2 models reproduced exactly from the production frame')
    params = PARAMS_BY_VARIANT[variant]
    cols = struct + CHRONO + S1 + r382; models = {}
    info = {'stage1_oof_map5_372': float(np.mean(list(oof_map.values()))), 'stage1_oof_map5_by_family': {f: float(np.mean([v for p, v in oof_map.items() if labels[p] == f])) for f in FAMILIES},
            'struct_frame_name_overlap_kept_frame_column': overlap, 'n_r382': len(r382), 'variant': variant, 'params': params, 'step65f_chrono3_reproduced': True, 'stage2': {}}
    for fam in FAMILIES:
        ms, nq, npos = stage2_fit(data, cols, fam, params); models[f'r382/{fam}'] = ms
        g = np.mean([m.booster_.feature_importance('gain') for m in ms], axis=0); order = np.argsort(g)[::-1][:12]
        info['stage2'][f'r382/{fam}'] = {'training_pairs': int(nq), 'training_rows': int(data.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP)).height), 'training_listed': int(npos),
                                         'top_gain': [(cols[i], round(float(g[i] / g.sum()), 3)) for i in order]}
        log(json.dumps({f'r382/{fam}': info['stage2'][f'r382/{fam}']['top_gain'][:6]}))
    bundle = {'models': models, 'cols': {'r382': cols}, 'struct': struct, 'r382': r382, 'params': params, 'variant': variant, 'seeds': list(SEEDS), 'top': TOP, 'stage1_sha256': sha(STAGE1), 'held_results_sha256': sha(held_results)}
    joblib.dump(bundle, OUT / 'stage2-models.joblib'); reloaded = joblib.load(OUT / 'stage2-models.joblib')
    for key, ms in models.items():
        np.testing.assert_array_equal(stage2_predict(ms, top, cols), stage2_predict(reloaded['models'][key], top, cols))
    info['reload_predictions_exact'] = True; info['models_sha256'] = sha(OUT / 'stage2-models.joblib')
    del data, check, x, top; gc.collect()
    return info


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'batches').mkdir(exist_ok=True)
    protected = check_protected()
    held_results = FS2R / 'results.json'; held = read_json(held_results); variant, why, gate = choose_variant(held)
    prev_receipt = read_json(PREV_OUT / 'receipt.json'); assert prev_receipt['twenty_incumbent_batch_controls_exact']
    fitted = joblib.load(CONDITIONAL / 'evidence-model.joblib'); caf = fitted['conditional_features']; features = fitted['features']; assert len(features) == 382
    chrono = joblib.load(STAGE1)
    selection = ROOT / 'reports' / 'replay-v13' / 'selected-pairs.parquet'
    struct_receipt = read_json(FS2 / 'eval_structural' / 'receipt.json'); assert struct_receipt['selected_pairs'] == 16060
    tag = 'r382DTSP' if variant == 'hybrid_DTSP_r382' else 'r382'
    candidates = {f'M3_stage2_{tag}': ('M3', 'r382'), f'S_stage2_{tag}': ('S', 'r382')}
    identity = {'purpose': 'Per-family second-stage evidence candidates with the 382 stage-1 features (r382) for the user to submit or not; nothing uploaded by this script. Risk/behaviour and all evidence outside the 16,060 selected pairs are identical to the source (M3 or S).',
                'sources': {'M3': sha(M3), 'S': sha(S), 'base_chronology_frac': sha(BASE)}, 'stage1_sha256': sha(STAGE1), 'baseline_model_sha256': sha(CONDITIONAL / 'evidence-model.joblib'),
                'final_reference_features_sha256': sha(CONDITIONAL / 'policy' / 'development-hand-features.parquet'), 'selection_sha256': sha(selection), 'hands_parquet_sha256': sha(ROOT / 'data' / 'hands.parquet'),
                'dev_structural_sha256': sha(FS2 / 'dev-structural-features.parquet'), 'eval_structural_receipt_sha256': sha(FS2 / 'eval_structural' / 'receipt.json'), 'code_sha256': sha(os.path.abspath(__file__)),
                'previous_stage': {'dir': str(PREV_OUT), 'receipt_sha256': sha(PREV_OUT / 'receipt.json'), 'models_sha256': sha(PREV_OUT / 'stage2-models.joblib')},
                'confirmation': {'freeze_sha256': sha(FS2R / 'freeze.json'), 'predictions_frozen_sha256': sha(FS2R / 'predictions-frozen.json'), 'results_sha256': sha(held_results), 'posthoc_sha256': sha(FS2R / 'posthoc_hybrid.json') if (FS2R / 'posthoc_hybrid.json').exists() else None, 'variant': variant, 'why': why, 'primary_gate_passed': held['gates']['allfam_top20_r382']['passed']},
                'candidates': candidates, 'stage2_recipe': {'params': PARAMS_BY_VARIANT[variant], 'seeds': list(SEEDS), 'top': TOP, 'rows': 'top-20 stage-1 OOF candidates per TRUE family, 372 targets', 'features': 'struct + chrono3 + s1/s1_rank + 382 ranker features (minus family_* and pot_bb)'}, 'protected': protected}
    plan = OUT / 'plan.json'
    if plan.exists():
        assert read_json(plan) == identity, 'plan changed; preserve the existing outputs'
    else:
        write_json(plan, identity)
    pred_path = OUT / 'prediction.json'
    if not pred_path.exists():
        prevpred = read_json(PREV_OUT / 'prediction.json')
        lo, hi = gate['pooled_pool_bootstrap_95']; e_delta = [round(min(lo, gate['pooled_delta'] * .5) - .005, 4), round(hi + .005, 4)]
        write_json(pred_path, {
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'plan_sha256': sha(plan),
            'held_confirmation': {'variant': variant, 'why': why, 'pooled_delta_map5_vs_allfam_top20_struct': gate['pooled_delta'], 'pool_bootstrap_95': gate['pooled_pool_bootstrap_95'], 'by_true_family': gate['pooled_delta_by_true_family'],
                                  'halves': gate['pooled_halves'], 'per_fold_delta': gate['per_fold_delta'], 'pre_registered_gate_passed': not gate.get('post_hoc', False), 'post_hoc': gate.get('post_hoc', False),
                                  'primary_all_families_result': {'pooled_delta': held['gates']['allfam_top20_r382']['pooled_delta'], 'ci95': held['gates']['allfam_top20_r382']['pooled_pool_bootstrap_95'], 'by_true_family': held['gates']['allfam_top20_r382']['pooled_delta_by_true_family'], 'passed': False},
                                  'held_truth_reads': 'third read of folds 0-2 for evidence (steps 11, 65d, 65j)'},
            'rule': 'As M3_stage2_chrono3 (step 65f) with the transfer / soft-play stage-2 models refitted on struct + chrono3 + s1 + the 382 ranker features (isolation keeps the step-65f model when the variant is the hybrid); risk and behaviour are those of the source (M3 or S).',
            'evidence_map_delta_interval_vs_M3_stage2_chrono3': e_delta,
            'evidence_interval_note': 'Held pooled interval vs the step-65c design, widened by .005 on both sides for routing error and the shorter evaluation histories; the lower end also capped at half the point estimate. For the post-hoc hybrid the interval is a derived, selection-biased quantity: expect the realised gain nearer the lower end.',
            'composite_intervals': {'M3_stage2_r382 minus M3_stage2_chrono3': [round(.2 * e_delta[0], 4), round(.2 * e_delta[1], 4)], 'S_stage2_r382 minus S_stage2_chrono3': [round(.2 * e_delta[0], 4), round(.2 * e_delta[1], 4)],
                                    'M3_stage2_r382 minus S (.89055)': [round(prevpred['composite_intervals']['M3_stage2_chrono3 minus S (.89055)'][0] + .2 * e_delta[0], 4), round(prevpred['composite_intervals']['M3_stage2_chrono3 minus S (.89055)'][1] + .2 * e_delta[1], 4)]},
            'decision_rule': 'Upload M3_stage2_chrono3 first (cleanly confirmed). This candidate second, only if a submission is spare: its increment (~+.003 composite) is below leaderboard resolution, so the public score cannot rank the two; the final choice rests on the held evidence. No re-tuning on outcomes.',
            'label_free': 'Evaluation side uses the frozen chronology_frac ranker, label-trained stage-2 models, the submission behaviour column and label-free structural hand features. No leaderboard feedback used to fit anything.',
            'not_uploaded_by_claude': True})
    verify_source_cache('development'); verify_source_cache('evaluation')
    fitpath = OUT / 'fit.json'
    if not fitpath.exists():
        info = fit_stage2(caf, features, chrono, held_results, variant)
        write_json(fitpath, {'plan_sha256': sha(plan), 'prediction_sha256': sha(pred_path), **info})
    fit = read_json(fitpath); assert fit['plan_sha256'] == sha(plan) and fit['models_sha256'] == sha(OUT / 'stage2-models.joblib')
    bundle = joblib.load(OUT / 'stage2-models.joblib'); struct = bundle['struct']; cols = bundle['cols']['r382']
    prev_bundle = joblib.load(PREV_OUT / 'stage2-models.joblib')
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
        prev_chrono3 = pl.read_parquet(PREV_OUT / 'batches' / (name + '-chrono3.parquet'))
        outputs = {'incumbent': [], 'chrono3_control': [], 'r382': []}; pools = sorted(raw['table_id'].unique()); count = 0; diag = []
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
            for out_name in ('chrono3_control', 'r382'):
                pred = np.full(frame.height, np.nan)
                for fam in FAMILIES:
                    mask = in_top & (fam_col == fam)
                    if mask.any():
                        if out_name == 'chrono3_control' or (variant == 'hybrid_DTSP_r382' and fam == CI):
                            pred[mask] = stage2_predict(prev_bundle['models'][f'chrono3/{fam}'], frame.filter(pl.Series(mask)), prev_bundle['cols']['chrono3'])
                        else:
                            pred[mask] = stage2_predict(bundle['models'][f'r382/{fam}'], frame.filter(pl.Series(mask)), cols)
                final = np.where(in_top, pred + 10.0, base_s - 1e6); assert np.isfinite(final).all()
                outputs[out_name].append(rank(frame, final))
            diag.append(frame.select('pair_id', 'predicted_behavior').unique())
        files = {}; lists = {}
        for cand, pieces in outputs.items():
            ranked = pl.concat(pieces).sort('pair_id'); expanded = ranked.select('pair_id', *[pl.col('hands').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i, c in enumerate(EVIDENCE_COLUMNS)])
            if cand == 'incumbent':
                assert expanded.equals(source.join(expanded.select('pair_id'), on='pair_id', how='semi').select(expanded.columns).sort('pair_id')), 'Incumbent ranks differ'
            if cand == 'chrono3_control':
                assert expanded.equals(prev_chrono3.select(expanded.columns).sort('pair_id')), 'step-65f chrono3 lists not reproduced'
                continue
            lists[cand] = expanded
            output = dest / (name + '-' + cand + '.parquet'); expanded.write_parquet(output); files[output.name] = sha(output)
        fams = pl.concat(diag)
        j = lists['incumbent'].join(lists['r382'], on='pair_id', suffix='_v').join(fams, on='pair_id')
        ch = j.select(pl.any_horizontal(*[pl.col(c) != pl.col(c + '_v') for c in EVIDENCE_COLUMNS]).alias('changed'), 'predicted_behavior')
        changed = {'pairs_with_changed_top5': int(ch['changed'].sum()), 'by_family': {k: int(v) for k, v in ch.group_by('predicted_behavior').agg(pl.col('changed').sum()).iter_rows()}}
        j = prev_chrono3.select(lists['r382'].columns).join(lists['r382'], on='pair_id', suffix='_v')
        agree = int(j.select(pl.all_horizontal(*[pl.col(c) == pl.col(c + '_v') for c in EVIDENCE_COLUMNS]).sum()).item())
        write_json(receipt, {'plan_sha256': sha(plan), 'files': files, 'selected_hands': count, 'selected_pairs': int(lists['incumbent'].height), 'incumbent_evidence_exact': True, 'step65f_chrono3_lists_exact': True,
                             'changed_vs_incumbent': changed, 'r382_chrono3_identical_top5_pairs': agree, 'seconds': time.perf_counter() - start})
        log(json.dumps({'batch': index, 'selected_hands': count, 'changed_vs_incumbent': changed['pairs_with_changed_top5'], 'same_as_chrono3': agree, 'seconds': round(time.perf_counter() - start, 1)}))
        del raw, part, extra, frame, outputs, sfeat, prev_chrono3; gc.collect()
    entries = []
    for cand, (src_name, vname) in candidates.items():
        src = sources[src_name]
        replacement = pl.read_parquet(sorted((OUT / 'batches').glob('batch_*-' + vname + '.parquet'))).sort('pair_id'); assert replacement.height == 16060
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
        prev_cand = pl.read_csv(PREV_OUT / ('submission_' + cand.replace(tag, 'chrono3') + '.csv'), infer_schema=False)
        changed_prev = int(result.select(pl.any_horizontal(*[pl.col(c) != prev_cand[c] for c in EVIDENCE_COLUMNS]).sum()).item())
        entries.append({'candidate': cand, 'source': src_name, 'variant': variant, 'file': output.name, 'sha256': sha(output), 'valid': validation['valid'], 'changed_evidence_lists_vs_source': changed,
                        'changed_evidence_lists_vs_step65f_chrono3': changed_prev, 'risk_behavior_exact': True, 'outside_selection_exact': True, 'submitted': False})
        log(json.dumps({'candidate': cand, 'changed_vs_source': changed, 'changed_vs_chrono3': changed_prev, 'valid': validation['valid'], 'sha256': entries[-1]['sha256']}))
    assert len({e['sha256'] for e in entries}) == len(candidates)
    write_json(OUT / 'receipt.json', {'completed_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(plan), 'prediction_sha256': sha(pred_path), 'fit_sha256': sha(fitpath), 'entries': entries,
                                      'twenty_incumbent_batch_controls_exact': True, 'twenty_step65f_chrono3_controls_exact': True, 'prediction_written_first': True, 'no_submissions_attempted': True, 'protected_after': check_protected(), 'seconds': time.perf_counter() - t0})
    log(json.dumps({'ready': True, 'seconds': round(time.perf_counter() - t0, 1)}))


if __name__ == '__main__':
    main()
