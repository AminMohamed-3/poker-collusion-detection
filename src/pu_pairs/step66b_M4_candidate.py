"""Step 66b: candidate M4 = the M3 construction (step 64f) with the EVIDENCE-SIGNATURE view of step 66 added to the blend.
Step 66 (OBS 14.42): a per-hand 'planted vs ordinary' LightGBM classifier trained on the listed hands of the development
positives (the evidence truth of the training pools only; 4 % of their hands are listed), scored on every hand of a pair
and pooled as the mean of the five highest probabilities (E_top5), separates positives from certified false positives at
AUC .94-.95 on its own (FULL .968, ALL5 .986) and adds to the learned blend cross-fold on every certified cell with the
same sign on both folds (raw-only recipe, floor .5: def A .9690 vs .9679, def C .9784 vs .9760; floor .1: def A .9379 vs
.9367, def C .9381 vs .9353). The structural relational features of step 65 and the policy ca features add nothing
reliable on top (ablations in evidence_view_eval{,_nostruct,_rawonly}.json), so the production recipe is RAW-ONLY:
163 per-hand raw features + chrono_frac, which exist for every evaluation hand (outputs_v2/hands_evaluation).
Construction identical to M3: all 909 evaluation block pairs (base risk >= .5) re-ordered by the blend of {FULL, RAW, CA,
BASE, OPP, V2, MEAN, TAILS, SHORT3, ALL5, E_top5, log shared_hands, R1 (soft)}, the sorted multiset of the 909 base values
re-assigned in that order, every other row byte-identical to the base. The planted-hand model for production is the
all-labelled fit (all 372 development positives' hands), as the SHORT3 / ALL5 views of step 64e. Pre-registered before
any evaluation truth is read; never uploaded by Claude. A second file, M4_stage2_chrono3, carries M4's risk with the
step-65f stage-2 evidence lists of M3_stage2_chrono3 (evidence and behaviour columns byte-identical to that file).
Outputs: experiments/viewdis_20260911/{evidence_view_eval_frame.parquet, plant_model_full.joblib, plant_fit_full.json,
         construction_testbed_M4.json, multiview_M4_fit.json},
         outputs_multiview_20260911/{prediction_M4.json, submission_multiview_M4_evidenceview.csv, receipt_M4.json,
         submission_M4_stage2_chrono3.csv, receipt_M4_stage2_chrono3.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
import numpy as np
import polars as pl
import lightgbm as lgb
import joblib
from pu_pairs.build_allpairs_fold import sha, fold_pools
from pu_pairs.step60_multiview_blend import ap, logreg, predict, lg
from pu_pairs.step62_multiview_candidate import VD, OUT, SFILE, VIEWS, L2, DEFS, eval_frame
from pu_pairs.step63_softR1_candidate import X_of as X_of63, MFILE
from pu_pairs.step59_blend_candidate import BASE, CFILE
from pu_pairs.step64f_M3_candidate import testbed as testbed_M3, FS_M2, FS_M3, M2FILE, DEST as M3FILE
from pu_pairs.step66_evidence_view import chrono, pool, PLANT_PARAMS

DEST = OUT / 'submission_multiview_M4_evidenceview.csv'
DEST2 = OUT / 'submission_M4_stage2_chrono3.csv'
M3S2 = CV / 'outputs_stage2_20260911' / 'submission_M3_stage2_chrono3.csv'
FS_M4 = VIEWS + ['SHORT3', 'ALL5', 'E_top5', 'sh', 'R1']
EVIEW = 'E_top5'


def log(*a):
    print(*a, flush=True)


def X4(c, fs, full_col, r1_col, noT_col):
    cols = []
    for v in fs:
        if v == EVIEW: cols.append(c[EVIEW].to_numpy().astype(float))
        else: cols.append(X_of63(c, [v], full_col, r1_col, noT_col)[:, 0])
    return np.column_stack(cols)


def fit_plant(t0):
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    listed = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id').with_columns(pl.lit(1.0).alias('relevance'))
    d = raw.join(labels, on='pair_id').filter(pl.col('label') == 1).join(listed, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('relevance').fill_null(0.0))
    d = chrono(d); feats = names + ['chrono_frac']
    X = np.nan_to_num(d.select(feats).to_numpy().astype(np.float32)); y = d['relevance'].to_numpy().astype(int)
    m = lgb.LGBMClassifier(**PLANT_PARAMS).fit(X, y); prior = float(y.mean())
    gain = sorted(zip(feats, m.booster_.feature_importance('gain')), key=lambda kv: -kv[1]); tot = sum(v for _, v in gain) or 1.0
    info = {'recipe': 'raw-only: 163 per-hand raw features + chrono_frac; label = listed in development_evidence; all 372 development positives (all-labelled fit)', 'params': {k: v for k, v in PLANT_PARAMS.items()},
            'train_hands': int(len(y)), 'train_listed': int(y.sum()), 'train_pairs': int(d['pair_id'].n_unique()), 'prior_listed': round(prior, 4), 'n_features': len(feats),
            'top_gain': [(k, round(float(v) / tot, 4)) for k, v in gain[:15]], 'in_sample_auc_not_reported': 'no held truth is read here; the cross-fold evidence is step 66'}
    joblib.dump({'model': m, 'features': feats, 'prior': prior}, VD / 'plant_model_full.joblib'); write_json(VD / 'plant_fit_full.json', info)
    log('plant model fitted', json.dumps({k: info[k] for k in ('train_hands', 'train_listed', 'train_pairs', 'prior_listed', 'n_features')}), round(time.time() - t0))
    return m, feats, prior


def score_eval_block(m, feats, prior, ids, t0):
    parts = []
    for f in sorted((ORIGINAL / 'outputs_v2' / 'hands_evaluation').glob('batch_*.parquet')):
        d = pl.read_parquet(f).filter(pl.col('pair_id').is_in(sorted(ids)))
        if d.height: parts.append(d)
    pre = pl.concat(parts); assert pre.select('pair_id', 'hand_id').n_unique() == pre.height and pre['pair_id'].n_unique() == len(ids) == 909, (pre.height, pre['pair_id'].n_unique())
    pre = chrono(pre)
    Xp = np.nan_to_num(pre.select(feats).to_numpy().astype(np.float32))
    sc = pre.select('pair_id', 'hand_id').with_columns(pl.Series('p', m.predict_proba(Xp)[:, 1]))
    g = pool(sc, prior); g.write_parquet(VD / 'evidence_view_eval_frame.parquet')
    log('evaluation block scored', pre.height, 'hands', g.height, 'pairs', round(time.time() - t0))
    return g


def testbed():
    d = testbed_M3()
    ev = pl.read_parquet(VD / 'evidence_view_prefix_rawonly.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).select('pair_id', EVIEW)
    d = d.join(ev, on='pair_id', how='left')
    used = d.filter(pl.col('cls').is_in(sorted(set(sum([list(p) + list(n) for p, n in DEFS.values()], [])))))
    cov = {'rows_in_defs': used.height, 'rows_missing_E': int(used[EVIEW].null_count()), 'fill': 'population minimum'}
    d = d.with_columns(pl.col(EVIEW).fill_null(float(d[EVIEW].min())))
    return d, cov


def constructions(d):
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    out = {}
    for dn, (posset, negset) in DEFS.items():
        c = d.filter(pl.col('cls').is_in(posset + negset)); y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy()
        in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
        r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum())}
        for name, fs in (('M2', FS_M2), ('M3', FS_M3), ('M4', FS_M4)):
            X = X4(c, fs, 'risk_prefix', 'R1_prefix', 'risk_noT'); pred = np.full(len(y), np.nan)
            m1 = logreg(X[in1], y[in1], L2); pred[in2 | insh] = predict(m1, X[in2 | insh]); m2 = logreg(X[in2], y[in2], L2); pred[in1] = predict(m2, X[in1])
            r[name] = {'pooled': round(ap(y, pred), 4), 'test_f2sh': round(ap(y[in2 | insh], pred[in2 | insh]), 4), 'test_f1': round(ap(y[in1], pred[in1]), 4),
                       'w1_new': {v: round(float(w), 2) for v, w in zip(fs, m1[0]) if v in ('SHORT3', 'ALL5', EVIEW, 'R1')}, 'w2_new': {v: round(float(w), 2) for v, w in zip(fs, m2[0]) if v in ('SHORT3', 'ALL5', EVIEW, 'R1')}}
        r['frozen'] = {'pooled': round(ap(y, c['risk_prefix'].to_numpy()), 4), 'test_f2sh': round(ap(y[in2 | insh], c['risk_prefix'].to_numpy()[in2 | insh]), 4), 'test_f1': round(ap(y[in1], c['risk_prefix'].to_numpy()[in1]), 4)}
        out[f'def {dn}'] = r
    return out


def final_fit(d, fs):
    posset, negset = DEFS['A']
    c = d.filter(pl.col('cls').is_in(posset + negset)); y = c['cls'].is_in(posset).to_numpy().astype(float)
    X = X4(c, fs, 'risk_prefix', 'R1_prefix', 'risk_noT'); m = logreg(X, y, L2)
    info = {'n_rows': int(len(y)), 'n_pos': int(y.sum()), 'l2': L2, 'features': fs, 'weights_standardised': {v: round(float(w), 3) for v, w in zip(fs, m[0])}, 'mu': {v: round(float(u), 3) for v, u in zip(fs, m[2])},
            'sd': {v: round(float(s), 3) for v, s in zip(fs, m[3])}, 'bias': round(float(m[1]), 3), 'AP_in_sample': round(ap(y, predict(m, X)), 4), 'AP_frozen_same_rows': round(ap(y, c['risk_prefix'].to_numpy()), 4)}
    return m, info


def build(ev, score, interval, fit_info, cons, cov, plant_info):
    base = pl.read_csv(BASE); bmap = dict(zip(base['pair_id'].to_list(), base['risk_score'].to_list()))
    ids = ev['pair_id'].to_list(); base_risk = np.array([bmap[i] for i in ids]); assert np.allclose(base_risk, ev['risk_v13'].to_numpy(), atol=1e-9)
    order = np.argsort(-score, kind='stable'); values = np.sort(base_risk)[::-1]
    mapping = {ids[i]: float(v) for i, v in zip(order, values)}; assert len(mapping) == ev.height == 909
    R1 = ev['R1'].to_numpy(); pct = np.empty(len(score)); pct[order] = np.arange(len(score)) / (len(score) - 1)
    S = pl.read_csv(SFILE); M3 = pl.read_csv(M3FILE)
    pred = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'candidate': str(DEST.relative_to(CV)),
            'rule': f'M4 = rank permutation of ALL {ev.height} evaluation block pairs (risk >= .5 in the base) by a logistic blend of {FS_M4} (as M3 of step 64f plus E_top5 = mean of the five highest per-hand probabilities of a planted-hand classifier trained on the listed hands of the development positives, raw-only recipe, all-labelled fit); the sorted multiset of the 909 base risks is re-assigned in blend order; every other row byte-identical to the base.',
            'relation_to_M3': 'Identical construction and fit rows; the only change is the evidence-signature view in the blend.',
            'hypothesis': 'OBS 14.42 (step 66): a classifier of what a PLANTED hand looks like (listed vs unlisted hands inside positive pairs, a 4 % target) pooled per pair is a length-robust view of a pair: cross-fold on the prefix testbed it adds +.001 (def A) / +.002 to +.004 (def C) block AP over M3 with the same sign on both folds and positive weights .4-.5 in both fold fits. The tail positives are untouched (median percentile .47 among nulls), so this is a block re-ordering only.',
            'evidence_view_coverage_testbed': cov, 'plant_fit': plant_info, 'construction_testbed': cons, 'final_fit': fit_info,
            'eval_R1_placement': {'n_R1': int(R1.sum()), 'R1_percentile_median': round(float(np.median(pct[R1])), 3), 'R1_share_in_top_half': round(float((pct[R1] < .5).mean()), 3)},
            'prediction': interval,
            'decision_rule': 'Upload after M3 (or M3_stage2_chrono3). M4 - M3 within [-.001, +.005] is the expected range and confirms the evidence-signature view as a small stacking lever; below -.003 says the all-labelled planted-hand fit does not transfer (keep M3). The increment is at leaderboard resolution: one measurement cannot rank M3 and M4, the final choice rests on the testbed evidence. No re-tuning on outcomes.',
            'label_free': 'Evaluation side uses frozen risks, label-trained views (including the planted-hand model, trained on development evidence truth only), shared_hands, dominance and risk_noT. No leaderboard feedback used to fit anything.', 'not_uploaded_by_claude': True}
    write_json(OUT / 'prediction_M4.json', pred)
    cand = base.with_columns(pl.col('pair_id').replace_strict(mapping, default=None, return_dtype=pl.Float64).alias('new')).with_columns(pl.coalesce([pl.col('new'), pl.col('risk_score')]).alias('risk_score')).drop('new')
    cand.write_csv(DEST); chk = pl.read_csv(DEST)
    same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score')
    blk_after = chk.filter(pl.col('risk_score') >= .5); multiset_ok = bool(np.allclose(np.sort(blk_after['risk_score'].to_numpy()), np.sort(base_risk)))
    outside_same = bool((chk.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy() == base.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy()).all())
    from scipy.stats import spearmanr
    j = chk.join(S.select('pair_id', pl.col('risk_score').alias('rS')), on='pair_id').join(M3.select('pair_id', pl.col('risk_score').alias('rM3')), on='pair_id').filter(pl.col('pair_id').is_in(ids))
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    prot = check_protected()
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'S_sha256': sha(SFILE), 'M_sha256': sha(MFILE), 'M2_sha256': sha(M2FILE), 'M3_sha256': sha(M3FILE), 'C_sha256': sha(CFILE), 'rule': pred['rule'],
               'n_block': ev.height, 'rows_with_changed_risk_vs_base': int((chk['risk_score'] != base['risk_score']).sum()), 'rows_with_changed_risk_vs_S': int((chk['risk_score'] != S['risk_score']).sum()), 'rows_with_changed_risk_vs_M3': int((chk['risk_score'] != M3['risk_score']).sum()),
               'block_spearman_vs_S': round(float(spearmanr(j['risk_score'].to_numpy(), j['rS'].to_numpy()).correlation), 4), 'block_spearman_vs_M3': round(float(spearmanr(j['risk_score'].to_numpy(), j['rM3'].to_numpy()).correlation), 4),
               'block_value_multiset_unchanged': multiset_ok, 'block_ge_.5_after': int(blk_after.height), 'rows_outside_block_identical_to_base': outside_same, 'non_risk_columns_identical_to_base': bool(same_cols),
               'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
    write_json(OUT / 'receipt_M4.json', receipt)
    log(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1)); log('validation:', json.dumps(vrep)[:300])
    # M4 risk + step-65f stage-2 evidence lists (M3_stage2_chrono3): evidence and behaviour columns byte-identical to that file
    m3s2 = pl.read_csv(M3S2, infer_schema=False); m3 = pl.read_csv(M3FILE, infer_schema=False); m4 = pl.read_csv(DEST, infer_schema=False)
    assert m3s2['pair_id'].to_list() == m3['pair_id'].to_list() == m4['pair_id'].to_list()
    assert (m3s2['risk_score'] == m3['risk_score']).all() and (m3s2['predicted_behavior'] == m3['predicted_behavior']).all(), 'M3_stage2_chrono3 must carry M3 risk and behaviour'
    ecols = [c for c in m3s2.columns if c.startswith('evidence_hand_')]
    out2 = m3s2.with_columns(m4['risk_score'].alias('risk_score')); out2.write_csv(DEST2); chk2 = pl.read_csv(DEST2, infer_schema=False)
    assert (chk2['risk_score'] == m4['risk_score']).all() and all((chk2[c] == m3s2[c]).all() for c in ecols + ['pair_id', 'predicted_behavior'])
    vrep2 = validate_submission(DEST2, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    rec2 = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST2), 'sha256': sha(DEST2), 'M4_sha256': sha(DEST), 'M3_stage2_chrono3_sha256': sha(M3S2), 'M3_sha256': sha(M3FILE),
            'rule': 'risk_score of M4; pair_id, predicted_behavior and the five evidence columns byte-identical to outputs_stage2_20260911/submission_M3_stage2_chrono3.csv (step 65f, held-confirmed stage-2 evidence).',
            'rows_with_changed_risk_vs_M3_stage2_chrono3': int((chk2['risk_score'] != m3s2['risk_score']).sum()), 'evidence_columns_identical': True, 'behaviour_identical': True,
            'prediction': {'vs_M3_stage2_chrono3': interval['interval_M4_minus_M3'], 'note': 'P-side increment only; the evidence and behaviour parts are those of M3_stage2_chrono3.'},
            'validation': vrep2, 'protected_check': check_protected(), 'do_not_upload_by_claude': True}
    write_json(OUT / 'receipt_M4_stage2_chrono3.json', rec2)
    log('M4_stage2_chrono3', json.dumps({k: v for k, v in rec2.items() if k not in ('validation', 'protected_check')}, indent=1)); log('validation2:', json.dumps(vrep2)[:300])


def main():
    t0 = time.time()
    d, cov = testbed(); log('testbed coverage', cov)
    cons = constructions(d); write_json(VD / 'construction_testbed_M4.json', cons); log(json.dumps(cons, indent=1))
    m, info = final_fit(d, FS_M4); write_json(VD / 'multiview_M4_fit.json', info); log(json.dumps(info, indent=1))
    ev = eval_frame().with_columns(pl.col('R1').cast(pl.Int8).alias('R1i'))
    tve = pl.read_parquet(VD / 'tail_views_eval_frame.parquet'); ev = ev.join(tve, on='pair_id', validate='1:1'); assert ev.height == 909
    pm, feats, prior = fit_plant(t0)
    plant_info = json.load(open(VD / 'plant_fit_full.json', encoding='utf-8'))
    g = score_eval_block(pm, feats, prior, set(ev['pair_id'].to_list()), t0)
    ev = ev.join(g.select('pair_id', EVIEW, 'E_n', 'E_max'), on='pair_id', validate='1:1'); assert ev.height == 909 and ev[EVIEW].null_count() == 0
    score = predict(m, X4(ev, FS_M4, 'risk_v13', 'R1i', 'risk_noT'))
    tb = [cons[dn]['M4'][t] - cons[dn]['M3'][t] for dn in ('def A', 'def C') for t in ('test_f2sh', 'test_f1')]
    m3pred = json.load(open(OUT / 'prediction_M3.json', encoding='utf-8'))['prediction']
    interval = {'testbed_M4_minus_M3_crossfold': [round(x, 4) for x in tb], 'interval_M4_minus_M3': [round(min(tb) - .002, 3), round(max(tb) + .002, 3)],
                'interval_M4_minus_S': [round(m3pred['interval_M3_minus_S'][0] + min(tb) - .001, 3), round(m3pred['interval_M3_minus_S'][1] + max(tb) + .001, 3)],
                'interval_M4_vs_base': [round(m3pred['interval_M3_vs_base'][0] + min(tb) - .001, 3), round(m3pred['interval_M3_vs_base'][1] + max(tb) + .001, 3)],
                'M3_intervals_inherited': m3pred, 'note': 'Same convention as steps 63-64f (testbed block-AP deltas carried to the public composite one-to-one, +-.002 for the transfer error of the new view); scenario model not used.',
                'eval_block_E_summary': {'E_top5_median': round(float(np.median(ev[EVIEW].to_numpy())), 4), 'E_top5_q10_q90': [round(float(q), 4) for q in np.quantile(ev[EVIEW].to_numpy(), [.1, .9])], 'hands_per_pair_median': float(np.median(ev['E_n'].to_numpy()))}}
    log('interval', json.dumps(interval))
    build(ev, score, interval, info, cons, cov, plant_info)
    log('seconds', round(time.time() - t0))


if __name__ == '__main__':
    main()
