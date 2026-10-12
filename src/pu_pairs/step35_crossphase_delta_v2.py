"""Step 35: drift-corrected cross-phase delta model (OBSERVATIONS 14.12 -> 14.13).

Step 32's delta model learned the phase offset (stack depth, action counts) as part of "development" and therefore
scored clean evaluation pairs high 3x too often (step 34). Corrections, all label-free with respect to evaluation:
  1. per-table centring: for every delta feature the median over the ~280 evaluation-list pairs of the same table,
     computed in the application direction (eval - dev). For the training direction (dev - eval) the same median
     enters with the opposite sign (the median of the negated values is the negated median).
  2. Welch standardisation: mean deltas divided by sqrt(s_t^2 / n_t + s_b^2 / n_b); std deltas divided by
     sqrt(s_t^2 / (2 n_t) + s_b^2 / (2 n_b)), with n the shared-hand count of each phase. Training pairs have more
     shared hands (median 109) than application pairs (76); raw deltas are noisier at application time.
Gates (written to freeze-v2.json before any evaluation score):
  V1  pooled-table 5-fold CV on labelled pairs: delta-only AP >= .90; combined >= single-phase - .002.
  V3  reversed-negative check: OOF-scoring held-out labelled negatives in the APPLICATION direction must give
      rates >= .5 and >= .9 within 1.5x of the forward-direction rates (both directions on the same held-out pairs).
  V2  evaluation pseudo-labels: AUC of the delta score, v13 >= .9 vs v13 < .1, >= .85.
  C   shared-hands skew: median evaluation shared hands of tail pairs with delta >= .9 must be >= .85 x the tail median.
Candidate (tail only, built regardless, receipt records gates): pairs with v13 risk >= .5 keep risk and order; below,
risk' = .5 * scaled(logit(risk) + logit(delta)), evidence/behaviour from submission_chronology_frac.csv unchanged.
Secondary, exploratory, not used for the candidate: family-agnostic change magnitude = Ledoit-Wolf Mahalanobis
distance of the standardised delta vector, covariance from training-fold negatives (reported AP on labelled folds).
Outputs: experiments/crossphase_20260911/{freeze-v2,cv-v2,results-v2}.json, eval-scores-v2.parquet,
         outputs_crossphase_20260911/submission_crossphase_tail_v2.csv (+ validation, receipt-v2.json)
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time, hashlib
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
from pu_pairs.step32_crossphase_delta_model import EXP, OUTD, TAU_BLOCK, PARAMS, sha, X
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.covariance import LedoitWolf

ZCLIP = 20.0


def raw_names(frame):
    return [c[:-5] for c in frame.columns if c.endswith('_mean')]


def zdelta(target, baseline, raws, centre=None, centre_sign=1.0):
    """Standardised, optionally table-centred deltas. target/baseline: merged step-31 frames (pair_id, table_id,
    shared_hands, r_mean, r_std, ...). centre: frame (table_id, c_<r>_mean, c_<r>_std) of application-direction
    medians; centre_sign = +1 for application (eval - dev), -1 for training (dev - eval)."""
    cols_t = ['pair_id', 'table_id', pl.col('shared_hands').alias('n_t')] + [pl.col(f'{r}_mean').alias(f'{r}_mean_t') for r in raws] + [pl.col(f'{r}_std').alias(f'{r}_std_t') for r in raws]
    cols_b = ['pair_id', pl.col('shared_hands').alias('n_b')] + [pl.col(f'{r}_mean').alias(f'{r}_mean_b') for r in raws] + [pl.col(f'{r}_std').alias(f'{r}_std_b') for r in raws]
    j = target.select(cols_t).join(baseline.select(cols_b), on='pair_id', how='inner')
    raw_d = j.select('pair_id', 'table_id', 'n_t', 'n_b',
                     *[(pl.col(f'{r}_mean_t') - pl.col(f'{r}_mean_b')).alias(f'd_{r}_mean') for r in raws],
                     *[(pl.col(f'{r}_std_t') - pl.col(f'{r}_std_b')).alias(f'd_{r}_std') for r in raws],
                     *[(pl.col(f'{r}_std_t') ** 2 / pl.col('n_t') + pl.col(f'{r}_std_b') ** 2 / pl.col('n_b')).sqrt().alias(f'se_{r}_mean') for r in raws],
                     *[(pl.col(f'{r}_std_t') ** 2 / (2 * pl.col('n_t')) + pl.col(f'{r}_std_b') ** 2 / (2 * pl.col('n_b'))).sqrt().alias(f'se_{r}_std') for r in raws])
    if centre is not None:
        raw_d = raw_d.join(centre, on='table_id', how='left')
        raw_d = raw_d.with_columns(*[(pl.col(f'd_{r}_{k}') - centre_sign * pl.col(f'c_{r}_{k}').fill_null(0.0)).alias(f'd_{r}_{k}') for r in raws for k in ('mean', 'std')])
    z = raw_d.select('pair_id', 'table_id', 'n_t', 'n_b',
                     *[(pl.col(f'd_{r}_{k}') / (pl.col(f'se_{r}_{k}') + 1e-9)).clip(-ZCLIP, ZCLIP).alias(f'z_{r}_{k}') for r in raws for k in ('mean', 'std')])
    return raw_d, z


def table_centres(raw_d_app, raws):
    """Per-table medians of the application-direction raw deltas (label-free, evaluation-list pairs)."""
    return raw_d_app.group_by('table_id').agg(*[pl.col(f'd_{r}_{k}').median().alias(f'c_{r}_{k}') for r in raws for k in ('mean', 'std')])


def cv_eval(frame, cols, y, groups, tag, rev=None):
    oof = np.zeros(len(y)); oof_r = np.zeros(len(y)) if rev is not None else None
    for tr, te in GroupKFold(n_splits=5).split(frame, y, groups):
        m = lgb.LGBMClassifier(**PARAMS).fit(X(frame[tr], cols), y[tr])
        oof[te] = m.predict_proba(X(frame[te], cols))[:, 1]
        if rev is not None:
            oof_r[te] = m.predict_proba(X(rev[te], cols))[:, 1]
    res = {'ap': float(average_precision_score(y, oof)), 'auc': float(roc_auc_score(y, oof)), 'n_features': len(cols),
           'neg_score_q': [float(q) for q in np.quantile(oof[y == 0], [.5, .9, .99, .999])], 'pos_score_q': [float(q) for q in np.quantile(oof[y == 1], [.01, .1, .5])]}
    print(json.dumps({'cv': tag, **res}), flush=True)
    return res, oof, oof_r


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    t0 = time.perf_counter()
    OUTD.mkdir(exist_ok=True)
    protected = check_protected()
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    lab_dev = pl.read_parquet(EXP / 'lab_on_dev.parquet'); lab_ev = pl.read_parquet(EXP / 'lab_on_eval.parquet')
    ev_t = pl.read_parquet(EXP / 'eval_on_eval.parquet'); ev_b = pl.read_parquet(EXP / 'eval_on_dev.parquet')
    raws = raw_names(lab_dev)
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'params': PARAMS, 'tau_block': TAU_BLOCK, 'n_raw_features': len(raws), 'z_clip': ZCLIP,
              'features': 'per-table centred (evaluation-list medians, sign-flipped for training direction), Welch-standardised deltas of means and stds',
              'gates': {'V1_delta_only_ap_min': .90, 'V1_combined_vs_single_max_loss': .002, 'V3_reversed_over_forward_rate_max': 1.5, 'V2_pseudo_auc_min': .85, 'C_shared_hands_median_ratio_min': .85},
              'candidate_rule': 'risk >= .5 unchanged; below: risk = .5 * minmax(logit(risk) + logit(delta)) within the tail; evidence/behaviour from submission_chronology_frac.csv',
              'inputs': {k: sha(EXP / f'{k}.parquet') for k in ['lab_on_dev', 'lab_on_eval', 'eval_on_eval', 'eval_on_dev']}, 'evaluation_scores_computed_before_freeze': False,
              'prediction_if_uploaded': 'public P within [-.005, +.03] of .938 (composite [-.0035, +.021] vs .876); E, B unchanged'}
    write_json(EXP / 'freeze-v2.json', freeze)
    # ---- table centres from the evaluation-list pairs (application direction) ----
    raw_app, _ = zdelta(ev_t, ev_b, raws)
    centre = table_centres(raw_app, raws)
    # ---- training frames: forward (dev - eval) and reversed (eval - dev), both centred and standardised ----
    _, z_f = zdelta(lab_dev, lab_ev, raws, centre, centre_sign=-1.0)
    _, z_r = zdelta(lab_ev, lab_dev, raws, centre, centre_sign=+1.0)
    zcols = [c for c in z_f.columns if c.startswith('z_')]
    single = [f'{r}_mean' for r in raws] + [f'{r}_std' for r in raws]
    tr = lab_dev.select('pair_id', 'table_id', *single).join(z_f.drop('table_id'), on='pair_id', how='inner').join(lab, on='pair_id', how='inner').sort('pair_id')
    rv = tr.select('pair_id').join(z_r.drop('table_id', 'n_t', 'n_b'), on='pair_id', how='left')
    assert rv['pair_id'].equals(tr['pair_id'])
    y = (tr['label'] == 1).to_numpy().astype(int); groups = tr['table_id'].to_numpy()
    cv = {'n_train': tr.height, 'n_pos': int(y.sum())}
    cv['single_phase'], _, _ = cv_eval(tr, single, y, groups, 'single_phase')
    cv['delta_only'], oof_d, oof_r = cv_eval(tr, zcols, y, groups, 'delta_only_z', rev=rv)
    cv['combined'], _, _ = cv_eval(tr, single + zcols, y, groups, 'combined')
    neg = y == 0; pos = y == 1
    cv['by_family_delta_only_ap'] = {fam: float(average_precision_score(y[(tr['behavior_family'] == fam).to_numpy() | neg], oof_d[(tr['behavior_family'] == fam).to_numpy() | neg])) for fam in ['directed_transfer', 'soft_play', 'coordinated_isolation']}
    cv['reversed'] = {'forward_neg_rate_ge_0.5': float((oof_d[neg] >= .5).mean()), 'reversed_neg_rate_ge_0.5': float((oof_r[neg] >= .5).mean()),
                      'forward_neg_rate_ge_0.9': float((oof_d[neg] >= .9).mean()), 'reversed_neg_rate_ge_0.9': float((oof_r[neg] >= .9).mean()),
                      'forward_neg_q': [float(q) for q in np.quantile(oof_d[neg], [.5, .9, .99, .999])], 'reversed_neg_q': [float(q) for q in np.quantile(oof_r[neg], [.5, .9, .99, .999])],
                      'reversed_pos_q': [float(q) for q in np.quantile(oof_r[pos], [.5, .9, .99])]}
    r5 = cv['reversed']['reversed_neg_rate_ge_0.5'] / max(cv['reversed']['forward_neg_rate_ge_0.5'], 1e-9)
    r9 = cv['reversed']['reversed_neg_rate_ge_0.9'] / max(cv['reversed']['forward_neg_rate_ge_0.9'], 1e-9)
    cv['gate_V1'] = {'delta_only_ap_ok': cv['delta_only']['ap'] >= .90, 'combined_not_worse': cv['combined']['ap'] >= cv['single_phase']['ap'] - .002}
    cv['gate_V3'] = {'ratio_ge_0.5': r5, 'ratio_ge_0.9': r9, 'passed': bool(r5 <= 1.5 and r9 <= 1.5)}
    # exploratory: family-agnostic change magnitude (Mahalanobis, covariance from training-fold negatives)
    Z = X(tr, zcols); mah = np.zeros(len(y)); mah_r = np.zeros(len(y)); Zr = X(rv, zcols)
    for a, b in GroupKFold(n_splits=5).split(tr, y, groups):
        lw = LedoitWolf().fit(Z[a][y[a] == 0])
        mah[b] = lw.mahalanobis(Z[b]); mah_r[b] = lw.mahalanobis(Zr[b])
    cv['change_magnitude'] = {'ap': float(average_precision_score(y, mah)), 'auc': float(roc_auc_score(y, mah)),
                              'by_family_ap': {fam: float(average_precision_score(y[(tr['behavior_family'] == fam).to_numpy() | neg], mah[(tr['behavior_family'] == fam).to_numpy() | neg])) for fam in ['directed_transfer', 'soft_play', 'coordinated_isolation']},
                              'neg_q': [float(q) for q in np.quantile(mah[neg], [.5, .9, .99])], 'pos_q': [float(q) for q in np.quantile(mah[pos], [.1, .5, .9])],
                              'reversed_neg_q': [float(q) for q in np.quantile(mah_r[neg], [.5, .9, .99])], 'reversed_pos_q': [float(q) for q in np.quantile(mah_r[pos], [.1, .5, .9])]}
    write_json(EXP / 'cv-v2.json', cv)
    print(json.dumps({'gate_V1': cv['gate_V1'], 'gate_V3': cv['gate_V3'], 'by_family': cv['by_family_delta_only_ap'], 'change_magnitude': {k: cv['change_magnitude'][k] for k in ['ap', 'auc', 'by_family_ap']}}), flush=True)
    # ---- fit on all labelled pairs; score evaluation pairs (application direction, centred) ----
    model = lgb.LGBMClassifier(**PARAMS).fit(Z, y)
    _, z_app = zdelta(ev_t, ev_b, raws, centre, centre_sign=+1.0)
    lw = LedoitWolf().fit(Z[neg])
    v13 = pl.read_csv(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv').select('pair_id', pl.col('risk_score').alias('risk'))
    ev = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'shared_hands').join(v13, on='pair_id').join(z_app.select('pair_id', 'n_t', 'n_b', *zcols), on='pair_id', how='left')
    m_has = ev[zcols[0]].is_not_null().to_numpy()
    has = ev.filter(pl.col(zcols[0]).is_not_null())
    score = np.full(ev.height, np.nan); score[m_has] = model.predict_proba(X(has, zcols))[:, 1]
    cm = np.full(ev.height, np.nan); cm[m_has] = lw.mahalanobis(X(has, zcols))
    ev = ev.select('pair_id', 'shared_hands', 'risk', 'n_t', 'n_b').with_columns(pl.Series('delta_score', score).fill_nan(None), pl.Series('change_mag', cm).fill_nan(None))
    ev.write_parquet(EXP / 'eval-scores-v2.parquet')
    hi = ev.filter(pl.col('risk') >= .9)['delta_score'].drop_nulls().to_numpy(); lo_f = ev.filter(pl.col('risk') < .1)
    lo = lo_f['delta_score'].drop_nulls().to_numpy(); mid = ev.filter((pl.col('risk') >= .1) & (pl.col('risk') < .9))['delta_score'].drop_nulls().to_numpy()
    tail_sh = float(lo_f['shared_hands'].median()); hi_sh = float(lo_f.filter(pl.col('delta_score') >= .9)['shared_hands'].median()) if (lo >= .9).any() else float('nan')
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'eval_pairs': ev.height, 'eval_pairs_with_delta': int(m_has.sum()),
           'pseudo': {'n_hi': len(hi), 'n_lo': len(lo), 'n_mid': len(mid), 'auc_hi_vs_lo': float(roc_auc_score(np.r_[np.ones(len(hi)), np.zeros(len(lo))], np.r_[hi, lo])),
                      'auc_mid_vs_lo': float(roc_auc_score(np.r_[np.ones(len(mid)), np.zeros(len(lo))], np.r_[mid, lo])),
                      'hi_q': [float(q) for q in np.quantile(hi, [.1, .5, .9])], 'mid_q': [float(q) for q in np.quantile(mid, [.1, .5, .9])], 'lo_q': [float(q) for q in np.quantile(lo, [.5, .9, .99, .999])],
                      'lo_rate_ge_0.5': float((lo >= .5).mean()), 'lo_rate_ge_0.9': float((lo >= .9).mean()), 'lo_count_ge_0.5': int((lo >= .5).sum()), 'lo_count_ge_0.9': int((lo >= .9).sum()), 'lo_count_ge_0.99': int((lo >= .99).sum())},
           'change_magnitude_eval': {'hi_q': [float(q) for q in np.quantile(ev.filter(pl.col('risk') >= .9)['change_mag'].drop_nulls().to_numpy(), [.1, .5, .9])],
                                     'lo_q': [float(q) for q in np.quantile(lo_f['change_mag'].drop_nulls().to_numpy(), [.5, .9, .99, .999])],
                                     'auc_hi_vs_lo': float(roc_auc_score(np.r_[np.ones(len(hi)), np.zeros(len(lo))], np.r_[ev.filter(pl.col('risk') >= .9)['change_mag'].drop_nulls().to_numpy(), lo_f['change_mag'].drop_nulls().to_numpy()]))},
           'shared_hands': {'tail_median': tail_sh, 'hi_delta_median': hi_sh, 'ratio': hi_sh / tail_sh if tail_sh else None},
           'labelled_negative_oof': cv['reversed']}
    res['gate_V2'] = res['pseudo']['auc_hi_vs_lo'] >= .85
    res['gate_C'] = bool(res['shared_hands']['ratio'] >= .85) if hi_sh == hi_sh else True
    print(json.dumps({k: res[k] for k in ['eval_pairs_with_delta', 'pseudo', 'change_magnitude_eval', 'shared_hands', 'gate_V2', 'gate_C']}, indent=1), flush=True)
    # ---- tail-only candidate, logit-sum ----
    base = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'); cols = base.columns
    j = base.join(ev.select('pair_id', 'delta_score', 'risk'), on='pair_id', how='left')
    assert j.height == base.height
    assert float(np.abs(j['risk_score'].to_numpy() - j['risk'].to_numpy()).max()) < 1e-9
    j = j.drop('risk')
    tail = j.filter(pl.col('risk_score') < TAU_BLOCK)
    d = tail['delta_score'].to_numpy().astype(float); r = tail['risk_score'].to_numpy().astype(float)
    d = np.where(np.isnan(d), np.nanmedian(d), d)  # pairs without a development baseline keep their v13 order at the delta median
    s = logit(r) + logit(d)
    new = TAU_BLOCK * (s - s.min()) / (s.max() - s.min() + 1e-12) * (1 - 1e-6)
    tail = tail.select('pair_id').with_columns(pl.Series('new_risk', new))
    out = j.join(tail, on='pair_id', how='left').with_columns(pl.coalesce([pl.col('new_risk'), pl.col('risk_score')]).alias('risk_score')).select(cols)
    assert out.height == base.height and out['pair_id'].equals(base['pair_id'])
    blk_old = base.filter(pl.col('risk_score') >= TAU_BLOCK); blk_new = out.filter(pl.col('risk_score') >= TAU_BLOCK)
    assert blk_old.height == blk_new.height and (blk_old['risk_score'].to_numpy() == blk_new['risk_score'].to_numpy()).all()
    for c in cols:
        if c != 'risk_score':
            assert out[c].equals(base[c]), c
    dest = OUTD / 'submission_crossphase_tail_v2.csv'
    out.write_csv(dest)
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(dest, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    write_json(OUTD / 'submission_crossphase_tail_v2-validation.json', vrep)
    assert vrep['valid'], vrep
    # how far does the tail move? (descriptive)
    old_rank = tail.select('pair_id').join(base.select('pair_id', 'risk_score'), on='pair_id')['risk_score'].rank(descending=True).to_numpy()
    new_rank = pl.Series(new).rank(descending=True).to_numpy()
    res['candidate'] = {'file': str(dest.relative_to(CV)), 'sha256': sha(dest), 'base_sha256': sha(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'), 'tail_pairs_reordered': int(tail.height),
                        'block_pairs_unchanged': blk_old.height, 'validation_valid': bool(vrep['valid']), 'max_tail_risk': float(new.max()),
                        'tail_top3000_old_rank_q': [float(q) for q in np.quantile(old_rank[np.argsort(new_rank)[:3000]], [.1, .5, .9])],
                        'risk_ge_0.1_tail_pairs': int((r >= .1).sum()), 'risk_ge_0.1_in_new_top3000': int((r[np.argsort(new_rank)[:3000]] >= .1).sum()),
                        'gates_passed': bool(cv['gate_V1']['delta_only_ap_ok'] and cv['gate_V1']['combined_not_worse'] and cv['gate_V3']['passed'] and res['gate_V2'] and res['gate_C']),
                        'prediction': freeze['prediction_if_uploaded'], 'submitted': False}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    write_json(EXP / 'results-v2.json', res)
    write_json(OUTD / 'receipt-v2.json', {'freeze': freeze, 'cv': cv, 'results': res})
    print(json.dumps(res['candidate'], indent=1), flush=True)


if __name__ == '__main__':
    main()
