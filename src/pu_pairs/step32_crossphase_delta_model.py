"""Step 32: change-versus-own-baseline ("delta") pair model, trained on labelled development pairs, applied to
evaluation pairs; pre-registered validation and a tail-only candidate rule (OBSERVATIONS 14.10).

Training rows: the 1,860 labelled development pairs. Features: for each base/v2 aggregate (mean and std only,
scale-free across the 3,000- vs 2,000-hand phases), delta = target phase minus baseline phase, where for training
target = development hands (where the positives collude) and baseline = evaluation hands (where, by the exclusion
rule and step 22's cross-phase check, they are presumed clean). For application target = evaluation, baseline =
development. Comparison models with the same recipe: single-phase (target means+stds) and combined.

Pre-registered (written to freeze.json before any evaluation score is computed):
  V1 pooled-table 5-fold CV on labelled pairs: AP/AUC of single-phase, delta-only, combined. Delta-only AP >= .90
     is required for the signal to count as present; combined must not be below single-phase by more than .002.
  V2 evaluation pseudo-labels: AUC of the delta score separating v13 risk >= .9 from v13 risk < .1 pairs.
     Reported whatever it is; >= .85 is required to build a candidate.
  Candidate rule (tail only): pairs with v13 risk >= TAU_BLOCK = .5 keep their risk and order. Pairs below it get
     risk' = TAU_BLOCK * fused, fused = mean of the within-tail percentile ranks of v13 risk and of the delta score
     (equal weights, fixed a priori), scaled into [0, TAU_BLOCK). Evidence lists and behaviour labels are those of
     submission_chronology_frac.csv, unchanged. Prediction recorded before upload.
Outputs: Claude Version/experiments/crossphase_20260911/{freeze.json, cv.json, eval-scores.parquet, results.json}
         Claude Version/outputs_crossphase_20260911/submission_crossphase_tail.csv (+ validation, receipt)
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time, hashlib
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

EXP = CV / 'experiments' / 'crossphase_20260911'
OUTD = CV / 'outputs_crossphase_20260911'
TAU_BLOCK = 0.5
PARAMS = dict(objective='binary', n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
              reg_lambda=5.0, random_state=7329, deterministic=True, force_col_wise=True, n_jobs=2, verbosity=-1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_names(frame):
    return [c for c in frame.columns if c.endswith('_mean') or c.endswith('_std')]


def delta(target, baseline, feats):
    t = target.select('pair_id', *feats)
    b = baseline.select('pair_id', *[pl.col(c).alias(c + '__b') for c in feats])
    j = t.join(b, on='pair_id', how='inner')
    d = j.select('pair_id', *[(pl.col(c) - pl.col(c + '__b')).alias('d_' + c) for c in feats])
    return d


def X(frame, cols):
    return np.nan_to_num(frame.select(cols).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0)


def cv_eval(frame, cols, y, groups, tag):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(frame, y, groups):
        m = lgb.LGBMClassifier(**PARAMS).fit(X(frame[tr], cols), y[tr])
        oof[te] = m.predict_proba(X(frame[te], cols))[:, 1]
    res = {'ap': float(average_precision_score(y, oof)), 'auc': float(roc_auc_score(y, oof)), 'n_features': len(cols),
           'neg_score_q': [float(q) for q in np.quantile(oof[y == 0], [.5, .9, .99, .999])], 'pos_score_q': [float(q) for q in np.quantile(oof[y == 1], [.01, .1, .5])]}
    print(json.dumps({'cv': tag, **res}), flush=True)
    return res, oof


def main():
    t0 = time.perf_counter()
    OUTD.mkdir(exist_ok=True)
    protected = check_protected()
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    lab_dev = pl.read_parquet(EXP / 'lab_on_dev.parquet')
    lab_ev = pl.read_parquet(EXP / 'lab_on_eval.parquet')
    feats = feature_names(lab_dev)
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'params': PARAMS, 'tau_block': TAU_BLOCK, 'n_base_features': len(feats),
              'features': 'delta = target - baseline over *_mean and *_std aggregates (base + v2); single-phase = target means+stds',
              'gates': {'V1_delta_only_ap_min': .90, 'V1_combined_vs_single_max_loss': .002, 'V2_pseudo_auc_min': .85},
              'candidate_rule': 'risk >= .5 unchanged; below: risk = .5 * mean(pct_rank(risk), pct_rank(delta_score)) within the tail; evidence/behaviour from submission_chronology_frac.csv',
              'inputs': {k: sha(EXP / f'{k}.parquet') for k in ['lab_on_dev', 'lab_on_eval', 'eval_on_eval', 'eval_on_dev']}, 'evaluation_scores_computed_before_freeze': False}
    write_json(EXP / 'freeze.json', freeze)
    # ---- training frame ----
    d_train = delta(lab_dev, lab_ev, feats)
    tr = lab_dev.select('pair_id', 'table_id', *feats).join(d_train, on='pair_id', how='inner').join(lab, on='pair_id', how='inner').sort('pair_id')
    y = (tr['label'] == 1).to_numpy().astype(int)
    groups = tr['table_id'].to_numpy()
    dcols = ['d_' + c for c in feats]
    cv = {'n_train': tr.height, 'n_pos': int(y.sum()), 'labelled_without_eval_rows': lab_dev.height - tr.height}
    cv['single_phase'], _ = cv_eval(tr, feats, y, groups, 'single_phase')
    cv['delta_only'], oof_delta = cv_eval(tr, dcols, y, groups, 'delta_only')
    cv['combined'], _ = cv_eval(tr, feats + dcols, y, groups, 'combined')
    cv['by_family_delta_only_ap'] = {}
    for fam in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
        mask = (tr['behavior_family'] == fam).to_numpy() | (y == 0)
        cv['by_family_delta_only_ap'][fam] = float(average_precision_score(y[mask], oof_delta[mask]))
    cv['gate_V1'] = {'delta_only_ap_ok': cv['delta_only']['ap'] >= .90, 'combined_not_worse': cv['combined']['ap'] >= cv['single_phase']['ap'] - .002}
    write_json(EXP / 'cv.json', cv)
    print(json.dumps({'gate_V1': cv['gate_V1'], 'by_family': cv['by_family_delta_only_ap']}), flush=True)
    # ---- fit on all labelled pairs, score evaluation pairs ----
    model = lgb.LGBMClassifier(**PARAMS).fit(X(tr, dcols), y)
    ev_t = pl.read_parquet(EXP / 'eval_on_eval.parquet')
    ev_b = pl.read_parquet(EXP / 'eval_on_dev.parquet')
    d_eval = delta(ev_t, ev_b, feats)
    v13 = pl.read_csv(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv').select('pair_id', pl.col('risk_score').alias('risk'))
    ev = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'shared_hands').join(v13, on='pair_id').join(d_eval, on='pair_id', how='left')
    has = ev.filter(pl.col(dcols[0]).is_not_null())
    score = np.full(ev.height, np.nan)
    score[ev[dcols[0]].is_not_null().to_numpy()] = model.predict_proba(X(has, dcols))[:, 1]
    ev = ev.select('pair_id', 'shared_hands', 'risk').with_columns(pl.Series('delta_score', score).fill_nan(None))
    ev.write_parquet(EXP / 'eval-scores.parquet')
    hi = ev.filter(pl.col('risk') >= .9)['delta_score'].drop_nulls().to_numpy()
    lo = ev.filter(pl.col('risk') < .1)['delta_score'].drop_nulls().to_numpy()
    mid = ev.filter((pl.col('risk') >= .1) & (pl.col('risk') < .9))['delta_score'].drop_nulls().to_numpy()
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'eval_pairs': ev.height, 'eval_pairs_with_delta': int(ev['delta_score'].is_not_null().sum()),
           'pseudo': {'n_hi': len(hi), 'n_lo': len(lo), 'n_mid': len(mid), 'auc_hi_vs_lo': float(roc_auc_score(np.r_[np.ones(len(hi)), np.zeros(len(lo))], np.r_[hi, lo])),
                      'auc_mid_vs_lo': float(roc_auc_score(np.r_[np.ones(len(mid)), np.zeros(len(lo))], np.r_[mid, lo])),
                      'hi_q': [float(q) for q in np.quantile(hi, [.1, .5, .9])], 'mid_q': [float(q) for q in np.quantile(mid, [.1, .5, .9])], 'lo_q': [float(q) for q in np.quantile(lo, [.5, .9, .99, .999])],
                      'lo_count_above_0.5': int((lo >= .5).sum()), 'lo_count_above_0.9': int((lo >= .9).sum()), 'lo_count_above_0.99': int((lo >= .99).sum())},
           'labelled_negative_oof_q': cv['delta_only']['neg_score_q']}
    res['gate_V2'] = res['pseudo']['auc_hi_vs_lo'] >= .85
    print(json.dumps(res, indent=1), flush=True)
    # ---- tail-only candidate (built regardless; receipt records the gates) ----
    base = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv')
    cols = base.columns
    j = base.join(ev.select('pair_id', 'delta_score', 'risk'), on='pair_id', how='left')
    assert j.height == base.height and j['delta_score'].is_null().sum() == ev['delta_score'].is_null().sum()
    risk_gap = float(np.abs(j['risk_score'].to_numpy() - j['risk'].to_numpy()).max())
    assert risk_gap < 1e-9, f'base risk differs from v13 risk by {risk_gap}'
    j = j.drop('risk')
    tail = j.filter(pl.col('risk_score') < TAU_BLOCK).with_columns(pl.col('delta_score').fill_null(0.0))
    n = tail.height
    r_rank = tail['risk_score'].rank(method='average').to_numpy() / n
    d_rank = tail['delta_score'].rank(method='average').to_numpy() / n
    fused = 0.5 * r_rank + 0.5 * d_rank
    new = TAU_BLOCK * (fused - fused.min()) / (fused.max() - fused.min() + 1e-12) * (1 - 1e-6)
    tail = tail.select('pair_id').with_columns(pl.Series('new_risk', new))
    out = j.join(tail, on='pair_id', how='left').with_columns(pl.coalesce([pl.col('new_risk'), pl.col('risk_score')]).alias('risk_score')).select(cols)
    assert out.height == base.height and out['pair_id'].equals(base['pair_id'])
    blk_old = base.filter(pl.col('risk_score') >= TAU_BLOCK)
    blk_new = out.filter(pl.col('risk_score') >= TAU_BLOCK)
    assert blk_old.height == blk_new.height and (blk_old['risk_score'].to_numpy() == blk_new['risk_score'].to_numpy()).all()
    for c in cols:
        if c != 'risk_score':
            assert out[c].equals(base[c]), c
    dest = OUTD / 'submission_crossphase_tail.csv'
    out.write_csv(dest)
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(dest, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet',
                               ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    write_json(OUTD / 'submission_crossphase_tail-validation.json', vrep)
    assert vrep['valid'], vrep
    res['candidate'] = {'file': str(dest.relative_to(CV)), 'sha256': sha(dest), 'base_sha256': sha(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'), 'tail_pairs_reordered': n,
                        'validation_valid': bool(vrep['valid']),
                        'block_pairs_unchanged': blk_old.height, 'max_tail_risk': float(new.max()), 'risk_range_ok': bool((out['risk_score'] >= 0).all() and (out['risk_score'] <= 1).all()),
                        'gates_passed': bool(cv['gate_V1']['delta_only_ap_ok'] and cv['gate_V1']['combined_not_worse'] and res['gate_V2']),
                        'prediction': 'public P within [-.005, +.03] of .938 (composite [-.0035, +.021] vs .876); E and B unchanged except via the 16,060 evidence set, which is unchanged here', 'submitted': False}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    write_json(EXP / 'results.json', res)
    write_json(OUTD / 'receipt.json', {'freeze': freeze, 'cv': cv, 'results': res})
    print(json.dumps(res['candidate'], indent=1), flush=True)


if __name__ == '__main__':
    main()
