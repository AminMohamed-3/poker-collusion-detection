"""Step 36: cross-phase delta model v3 plus two label-free side checks (OBSERVATIONS 14.13).

v2 (step 35) fixed the direction drift at the >= .5 level (reversed/forward labelled-negative rate .86) but its
pre-registered >= .9 criterion failed on counts of 5 vs 3 out of 1,481 negatives, and Welch standardisation alone
cost labelled AP (.978 -> .947): the likelihood ratio of collusion depends on the effect size as well as on its
significance. v3 therefore gives the model both views and the sample sizes:
  features = per-table centred raw deltas (326) + Welch z of the same (326) + log shared hands of both phases (2).
Gates, fixed before running (freeze-v3.json):
  V1  delta-only AP >= .90; combined >= single-phase - .002 (pooled-table 5-fold CV, 1,849 labelled pairs).
  V3' direction symmetry on held-out labelled negatives: AUC(forward OOF vs reversed OOF) <= .56 AND
      reversed/forward rate at >= .5 within [1/1.5, 1.5]. (The >= .9 count ratio of v2 is dropped as underpowered:
      3 vs 5 events. This replacement was decided before v3 was run and is recorded here.)
  V2  evaluation pseudo-labels AUC (v13 >= .9 vs < .1) >= .85.
  C   median evaluation shared hands of tail pairs with delta >= .9 at least .85 x the tail median.
Side checks (label-free on evaluation):
  P   persistence: a single-phase model (326 target-phase aggregates, trained on all labelled development pairs)
      applied to the evaluation pairs on DEVELOPMENT hands. If the v13-certain evaluation pairs also score high on
      their development hands, evaluation colluders are not fresh plants and a delta signal cannot exist for them.
  X   family-agnostic extreme count: number of centred z beyond |4| per pair; AP on labelled pairs, reversed-negative
      null, evaluation tail distribution. Exploratory only.
Candidate (tail only, built regardless): logit(v13 risk) + logit(delta) ordering below .5, block unchanged.
Outputs: experiments/crossphase_20260911/{freeze-v3,cv-v3,results-v3}.json, eval-scores-v3.parquet,
         outputs_crossphase_20260911/submission_crossphase_tail_v3.csv (+ validation, receipt-v3.json)
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
from pu_pairs.step32_crossphase_delta_model import EXP, OUTD, TAU_BLOCK, PARAMS, sha, X
from pu_pairs.step35_crossphase_delta_v2 import raw_names, zdelta, table_centres, logit
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']


def frame_v3(raw_d, z, raws):
    dcols = [f'd_{r}_{k}' for r in raws for k in ('mean', 'std')]
    f = raw_d.select('pair_id', 'table_id', *dcols).join(z.drop('table_id'), on='pair_id')
    return f.with_columns(pl.col('n_t').log().alias('log_n_t'), pl.col('n_b').log().alias('log_n_b'))


def cv_scores(frame, cols, y, groups, rev=None):
    oof = np.zeros(len(y)); oof_r = np.zeros(len(y)) if rev is not None else None
    for tr, te in GroupKFold(n_splits=5).split(frame, y, groups):
        m = lgb.LGBMClassifier(**PARAMS).fit(X(frame[tr], cols), y[tr])
        oof[te] = m.predict_proba(X(frame[te], cols))[:, 1]
        if rev is not None:
            oof_r[te] = m.predict_proba(X(rev[te], cols))[:, 1]
    return oof, oof_r


def summary(y, oof, tag):
    res = {'ap': float(average_precision_score(y, oof)), 'auc': float(roc_auc_score(y, oof)),
           'neg_score_q': [float(q) for q in np.quantile(oof[y == 0], [.5, .9, .99, .999])], 'pos_score_q': [float(q) for q in np.quantile(oof[y == 1], [.01, .1, .5])]}
    print(json.dumps({'cv': tag, **res}), flush=True)
    return res


def main():
    t0 = time.perf_counter()
    protected = check_protected()
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    lab_dev = pl.read_parquet(EXP / 'lab_on_dev.parquet'); lab_ev = pl.read_parquet(EXP / 'lab_on_eval.parquet')
    ev_t = pl.read_parquet(EXP / 'eval_on_eval.parquet'); ev_b = pl.read_parquet(EXP / 'eval_on_dev.parquet')
    raws = raw_names(lab_dev)
    single = [f'{r}_mean' for r in raws] + [f'{r}_std' for r in raws]
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'params': PARAMS, 'tau_block': TAU_BLOCK,
              'features': 'centred raw deltas + Welch z + log shared hands (both phases)',
              'gates': {'V1_delta_only_ap_min': .90, 'V1_combined_vs_single_max_loss': .002, 'V3_direction_auc_max': .56, 'V3_rate_ratio_ge_0.5_range': [1 / 1.5, 1.5],
                        'V2_pseudo_auc_min': .85, 'C_shared_hands_median_ratio_min': .85},
              'v2_gate_change_note': 'v2 >= .9 count-ratio criterion (5 vs 3 events) replaced by an AUC criterion; decided before v3 was run',
              'candidate_rule': 'risk >= .5 unchanged; below: .5 * minmax(logit(risk) + logit(delta)); evidence/behaviour from submission_chronology_frac.csv',
              'inputs': {k: sha(EXP / f'{k}.parquet') for k in ['lab_on_dev', 'lab_on_eval', 'eval_on_eval', 'eval_on_dev']},
              'prediction_if_uploaded': 'public P within [-.005, +.03] of .938 (composite [-.0035, +.021] vs .876); E, B unchanged'}
    write_json(EXP / 'freeze-v3.json', freeze)
    raw_app, z_app = zdelta(ev_t, ev_b, raws)
    centre = table_centres(raw_app, raws)
    raw_app, z_app = zdelta(ev_t, ev_b, raws, centre, +1.0)
    raw_f, z_f = zdelta(lab_dev, lab_ev, raws, centre, -1.0)
    raw_r, z_r = zdelta(lab_ev, lab_dev, raws, centre, +1.0)
    F = frame_v3(raw_f, z_f, raws); R = frame_v3(raw_r, z_r, raws); A = frame_v3(raw_app, z_app, raws)
    v3cols = [c for c in F.columns if c.startswith('d_') or c.startswith('z_') or c.startswith('log_n')]
    tr = lab_dev.select('pair_id', 'table_id', *single).join(F.drop('table_id'), on='pair_id', how='inner').join(lab, on='pair_id', how='inner').sort('pair_id')
    rv = tr.select('pair_id').join(R.drop('table_id'), on='pair_id', how='left')
    assert rv['pair_id'].equals(tr['pair_id'])
    y = (tr['label'] == 1).to_numpy().astype(int); groups = tr['table_id'].to_numpy(); neg = y == 0; pos = y == 1
    cv = {'n_train': tr.height, 'n_pos': int(y.sum()), 'n_features_delta': len(v3cols)}
    oof_s, _ = cv_scores(tr, single, y, groups); cv['single_phase'] = summary(y, oof_s, 'single_phase')
    oof_d, oof_r = cv_scores(tr, v3cols, y, groups, rev=rv); cv['delta_only'] = summary(y, oof_d, 'delta_v3')
    oof_c, _ = cv_scores(tr, single + v3cols, y, groups); cv['combined'] = summary(y, oof_c, 'combined')
    cv['by_family_delta_only_ap'] = {fam: float(average_precision_score(y[(tr['behavior_family'] == fam).to_numpy() | neg], oof_d[(tr['behavior_family'] == fam).to_numpy() | neg])) for fam in FAMS}
    dir_auc = float(roc_auc_score(np.r_[np.zeros(neg.sum()), np.ones(neg.sum())], np.r_[oof_d[neg], oof_r[neg]]))
    f5 = float((oof_d[neg] >= .5).mean()); r5 = float((oof_r[neg] >= .5).mean())
    cv['reversed'] = {'direction_auc_reversed_vs_forward_negatives': dir_auc, 'forward_neg_rate_ge_0.5': f5, 'reversed_neg_rate_ge_0.5': r5, 'forward_neg_count_ge_0.5': int((oof_d[neg] >= .5).sum()),
                      'reversed_neg_count_ge_0.5': int((oof_r[neg] >= .5).sum()), 'forward_neg_count_ge_0.9': int((oof_d[neg] >= .9).sum()), 'reversed_neg_count_ge_0.9': int((oof_r[neg] >= .9).sum()),
                      'forward_neg_q': [float(q) for q in np.quantile(oof_d[neg], [.5, .9, .99, .999])], 'reversed_neg_q': [float(q) for q in np.quantile(oof_r[neg], [.5, .9, .99, .999])],
                      'reversed_pos_q': [float(q) for q in np.quantile(oof_r[pos], [.5, .9, .99])]}
    ratio = r5 / max(f5, 1e-9)
    cv['gate_V1'] = {'delta_only_ap_ok': cv['delta_only']['ap'] >= .90, 'combined_not_worse': cv['combined']['ap'] >= cv['single_phase']['ap'] - .002}
    cv['gate_V3'] = {'direction_auc': dir_auc, 'rate_ratio_ge_0.5': ratio, 'passed': bool(dir_auc <= .56 and 1 / 1.5 <= ratio <= 1.5)}
    # X: family-agnostic extreme count on centred z
    zcols = [c for c in v3cols if c.startswith('z_')]
    xc = (np.abs(X(tr, zcols)) > 4).sum(axis=1).astype(float); xr = (np.abs(X(rv, zcols)) > 4).sum(axis=1).astype(float)
    cv['extreme_count'] = {'ap': float(average_precision_score(y, xc)), 'auc': float(roc_auc_score(y, xc)), 'neg_q': [float(q) for q in np.quantile(xc[neg], [.5, .9, .99])],
                           'pos_q': [float(q) for q in np.quantile(xc[pos], [.1, .5, .9])], 'reversed_neg_q': [float(q) for q in np.quantile(xr[neg], [.5, .9, .99])],
                           'by_family_ap': {fam: float(average_precision_score(y[(tr['behavior_family'] == fam).to_numpy() | neg], xc[(tr['behavior_family'] == fam).to_numpy() | neg])) for fam in FAMS}}
    write_json(EXP / 'cv-v3.json', cv)
    print(json.dumps({'gate_V1': cv['gate_V1'], 'gate_V3': cv['gate_V3'], 'by_family': cv['by_family_delta_only_ap'], 'extreme_count': {k: cv['extreme_count'][k] for k in ['ap', 'auc', 'by_family_ap']}}), flush=True)
    # ---- fit on all labelled; score evaluation ----
    model = lgb.LGBMClassifier(**PARAMS).fit(X(tr, v3cols), y)
    single_model = lgb.LGBMClassifier(**PARAMS).fit(X(tr, single), y)
    v13 = pl.read_csv(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv').select('pair_id', pl.col('risk_score').alias('risk'))
    ev = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'shared_hands').join(v13, on='pair_id')
    ev = ev.join(A.select('pair_id', 'n_t', 'n_b', *v3cols), on='pair_id', how='left')
    m_has = ev[v3cols[0]].is_not_null().to_numpy(); has = ev.filter(pl.col(v3cols[0]).is_not_null())
    score = np.full(ev.height, np.nan); score[m_has] = model.predict_proba(X(has, v3cols))[:, 1]
    xcnt = np.full(ev.height, np.nan); xcnt[m_has] = (np.abs(X(has, zcols)) > 4).sum(axis=1)
    # persistence: single-phase model on evaluation hands and on development hands
    s_ev = ev.select('pair_id').join(ev_t.select('pair_id', *single), on='pair_id', how='left')
    s_dv = ev.select('pair_id').join(ev_b.select('pair_id', *single), on='pair_id', how='left')
    m_ev = s_ev[single[0]].is_not_null().to_numpy(); m_dv = s_dv[single[0]].is_not_null().to_numpy()
    sc_ev = np.full(ev.height, np.nan); sc_ev[m_ev] = single_model.predict_proba(X(s_ev.filter(pl.col(single[0]).is_not_null()), single))[:, 1]
    sc_dv = np.full(ev.height, np.nan); sc_dv[m_dv] = single_model.predict_proba(X(s_dv.filter(pl.col(single[0]).is_not_null()), single))[:, 1]
    ev = ev.select('pair_id', 'shared_hands', 'risk', 'n_t', 'n_b').with_columns(pl.Series('delta_score', score).fill_nan(None), pl.Series('extreme_count', xcnt).fill_nan(None),
                                                                                 pl.Series('single_on_eval', sc_ev).fill_nan(None), pl.Series('single_on_dev', sc_dv).fill_nan(None))
    ev.write_parquet(EXP / 'eval-scores-v3.parquet')
    hi_f = ev.filter(pl.col('risk') >= .9); lo_f = ev.filter(pl.col('risk') < .1); mid_f = ev.filter((pl.col('risk') >= .1) & (pl.col('risk') < .9))
    hi = hi_f['delta_score'].drop_nulls().to_numpy(); lo = lo_f['delta_score'].drop_nulls().to_numpy(); mid = mid_f['delta_score'].drop_nulls().to_numpy()
    tail_sh = float(lo_f['shared_hands'].median()); hd = lo_f.filter(pl.col('delta_score') >= .9); hi_sh = float(hd['shared_hands'].median()) if hd.height else float('nan')
    def qq(a, qs):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return [float(q) for q in np.quantile(a, qs)] if len(a) else []
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'eval_pairs': ev.height, 'eval_pairs_with_delta': int(m_has.sum()),
           'pseudo': {'n_hi': len(hi), 'n_lo': len(lo), 'n_mid': len(mid), 'auc_hi_vs_lo': float(roc_auc_score(np.r_[np.ones(len(hi)), np.zeros(len(lo))], np.r_[hi, lo])),
                      'auc_mid_vs_lo': float(roc_auc_score(np.r_[np.ones(len(mid)), np.zeros(len(lo))], np.r_[mid, lo])), 'hi_q': qq(hi, [.1, .5, .9]), 'mid_q': qq(mid, [.1, .5, .9]), 'lo_q': qq(lo, [.5, .9, .99, .999]),
                      'lo_rate_ge_0.5': float((lo >= .5).mean()), 'lo_rate_ge_0.9': float((lo >= .9).mean()), 'lo_count_ge_0.5': int((lo >= .5).sum()), 'lo_count_ge_0.9': int((lo >= .9).sum()), 'lo_count_ge_0.99': int((lo >= .99).sum())},
           'persistence': {'single_on_eval_vs_v13_auc_hi_lo': float(roc_auc_score(np.r_[np.ones(hi_f.height), np.zeros(lo_f.height)], np.r_[hi_f['single_on_eval'].fill_null(0).to_numpy(), lo_f['single_on_eval'].fill_null(0).to_numpy()])),
                           'hi_single_on_eval_q': qq(hi_f['single_on_eval'].to_numpy(), [.1, .5, .9]), 'hi_single_on_dev_q': qq(hi_f['single_on_dev'].to_numpy(), [.5, .9, .99]),
                           'hi_frac_single_on_dev_ge_0.5': float(np.nanmean(hi_f['single_on_dev'].to_numpy().astype(float) >= .5)),
                           'lo_single_on_dev_q': qq(lo_f['single_on_dev'].to_numpy(), [.5, .9, .99, .999]), 'lo_frac_single_on_dev_ge_0.5': float(np.nanmean(lo_f['single_on_dev'].to_numpy().astype(float) >= .5)),
                           'lo_count_single_on_dev_ge_0.9': int(np.nansum(lo_f['single_on_dev'].to_numpy().astype(float) >= .9)),
                           'hi_delta_lt_0.05_count': int((hi < .05).sum()), 'hi_delta_lt_0.05_single_on_dev_q': qq(hi_f.filter(pl.col('delta_score') < .05)['single_on_dev'].to_numpy(), [.1, .5, .9])},
           'extreme_count_eval': {'hi_q': qq(hi_f['extreme_count'].to_numpy(), [.1, .5, .9]), 'lo_q': qq(lo_f['extreme_count'].to_numpy(), [.5, .9, .99, .999]),
                                  'auc_hi_vs_lo': float(roc_auc_score(np.r_[np.ones(hi_f.height), np.zeros(lo_f.height)], np.r_[hi_f['extreme_count'].fill_null(0).to_numpy(), lo_f['extreme_count'].fill_null(0).to_numpy()]))},
           'shared_hands': {'tail_median': tail_sh, 'hi_delta_median': hi_sh, 'ratio': hi_sh / tail_sh if tail_sh and hi_sh == hi_sh else None}, 'labelled_negative_oof': cv['reversed']}
    res['gate_V2'] = res['pseudo']['auc_hi_vs_lo'] >= .85
    res['gate_C'] = bool(res['shared_hands']['ratio'] >= .85) if res['shared_hands']['ratio'] is not None else True
    print(json.dumps({k: res[k] for k in ['pseudo', 'persistence', 'extreme_count_eval', 'shared_hands', 'gate_V2', 'gate_C']}, indent=1), flush=True)
    # ---- candidate ----
    base = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'); cols = base.columns
    j = base.join(ev.select('pair_id', 'delta_score', 'risk'), on='pair_id', how='left')
    assert j.height == base.height and float(np.abs(j['risk_score'].to_numpy() - j['risk'].to_numpy()).max()) < 1e-9
    j = j.drop('risk'); tail = j.filter(pl.col('risk_score') < TAU_BLOCK)
    d = tail['delta_score'].to_numpy().astype(float); r = tail['risk_score'].to_numpy().astype(float); d = np.where(np.isnan(d), np.nanmedian(d), d)
    s = logit(r) + logit(d); new = TAU_BLOCK * (s - s.min()) / (s.max() - s.min() + 1e-12) * (1 - 1e-6)
    tail = tail.select('pair_id').with_columns(pl.Series('new_risk', new))
    out = j.join(tail, on='pair_id', how='left').with_columns(pl.coalesce([pl.col('new_risk'), pl.col('risk_score')]).alias('risk_score')).select(cols)
    assert out.height == base.height and out['pair_id'].equals(base['pair_id'])
    blk_old = base.filter(pl.col('risk_score') >= TAU_BLOCK); blk_new = out.filter(pl.col('risk_score') >= TAU_BLOCK)
    assert blk_old.height == blk_new.height and (blk_old['risk_score'].to_numpy() == blk_new['risk_score'].to_numpy()).all()
    for c in cols:
        if c != 'risk_score':
            assert out[c].equals(base[c]), c
    dest = OUTD / 'submission_crossphase_tail_v3.csv'; out.write_csv(dest)
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(dest, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    write_json(OUTD / 'submission_crossphase_tail_v3-validation.json', vrep); assert vrep['valid'], vrep
    new_order = np.argsort(-new)
    res['candidate'] = {'file': str(dest.relative_to(CV)), 'sha256': sha(dest), 'base_sha256': sha(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'), 'tail_pairs_reordered': int(tail.height),
                        'block_pairs_unchanged': blk_old.height, 'validation_valid': bool(vrep['valid']), 'risk_ge_0.1_tail_pairs': int((r >= .1).sum()), 'risk_ge_0.1_in_new_top3000': int((r[new_order[:3000]] >= .1).sum()),
                        'new_top3000_v13_risk_q': qq(r[new_order[:3000]], [.1, .5, .9]), 'new_top3000_delta_q': qq(d[new_order[:3000]], [.1, .5, .9]),
                        'gates_passed': bool(cv['gate_V1']['delta_only_ap_ok'] and cv['gate_V1']['combined_not_worse'] and cv['gate_V3']['passed'] and res['gate_V2'] and res['gate_C']),
                        'prediction': freeze['prediction_if_uploaded'], 'submitted': False}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    write_json(EXP / 'results-v3.json', res); write_json(OUTD / 'receipt-v3.json', {'freeze': freeze, 'cv': cv, 'results': res})
    print(json.dumps(res['candidate'], indent=1), flush=True)


if __name__ == '__main__':
    main()
