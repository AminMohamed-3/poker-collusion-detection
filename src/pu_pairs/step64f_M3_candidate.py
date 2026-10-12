"""Step 64f: candidate M3 = the M2 construction (step 63) with the two LENGTH-MATCHED views of step 64a added to the blend.
Step 64 (OBS 14.39): training the production recipe on thinned views only (SHORT3: 40 / 25 / 15 % of each training pair's
hands) or on full + four thinned views (ALL5) gives the most length-robust views measured so far (within-block AUC
positives vs certified false positives .983 / .986 against .968 for the frozen model and .967 for RAW), and adding both
to the learned blend gains cross-fold on every certified cell (floor .5: def A .9679 vs .9660, def C .9760 vs .9717;
floor .1: def A .9367 vs .9337, def C .9353 vs .9318). The hand-level pooled view and the family one-vs-rest views were
null and are not used. Construction identical to M2: all 909 evaluation block pairs (base risk >= .5) re-ordered by the
blend of {FULL, RAW, CA, BASE, OPP, V2, MEAN, TAILS, SHORT3, ALL5, log shared_hands, R1 (soft)}, the sorted multiset of
the 909 base values re-assigned in that order, every other row byte-identical to the base. Pre-registered before any
evaluation truth is read; never uploaded by Claude.
Outputs: experiments/viewdis_20260911/{construction_testbed_M3.json, multiview_M3_fit.json},
         outputs_multiview_20260911/{prediction_M3.json, submission_multiview_M3_lengthviews.csv, receipt_M3.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
import numpy as np
import polars as pl
from pu_pairs.build_allpairs_fold import sha, fold_pools
from pu_pairs.step60_multiview_blend import ap, logreg, predict
from pu_pairs.step62_multiview_candidate import VD, OUT, SFILE, VIEWS, L2, DEFS, eval_frame
from pu_pairs.step63_softR1_candidate import X_of, load_testbed, MFILE
from pu_pairs.step59_blend_candidate import BASE, CFILE

M2FILE = OUT / 'submission_multiview_softR1.csv'
DEST = OUT / 'submission_multiview_M3_lengthviews.csv'
FS_M2 = VIEWS + ['sh', 'R1']
FS_M3 = VIEWS + ['SHORT3', 'ALL5', 'sh', 'R1']


def log(*a):
    print(*a, flush=True)


def testbed():
    d = load_testbed()
    tv = pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).select('pair_id', 'r_SHORT3', 'r_ALL5')
    d = d.join(tv, on='pair_id', how='left'); assert d['r_SHORT3'].null_count() == 0
    return d


def constructions(d):
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    out = {}
    for dn, (posset, negset) in DEFS.items():
        c = d.filter(pl.col('cls').is_in(posset + negset)); y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy()
        in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
        r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum())}
        for name, fs in (('M2: learned views+sh+R1 (soft)', FS_M2), ('M3: learned views+SHORT3+ALL5+sh+R1 (soft)', FS_M3)):
            X = X_of(c, fs, 'risk_prefix', 'R1_prefix', 'risk_noT'); pred = np.full(len(y), np.nan)
            m1 = logreg(X[in1], y[in1], L2); pred[in2 | insh] = predict(m1, X[in2 | insh]); m2 = logreg(X[in2], y[in2], L2); pred[in1] = predict(m2, X[in1])
            r[name] = {'pooled': round(ap(y, pred), 4), 'test_f2sh': round(ap(y[in2 | insh], pred[in2 | insh]), 4), 'test_f1': round(ap(y[in1], pred[in1]), 4)}
        r['frozen'] = {'pooled': round(ap(y, c['risk_prefix'].to_numpy()), 4), 'test_f2sh': round(ap(y[in2 | insh], c['risk_prefix'].to_numpy()[in2 | insh]), 4), 'test_f1': round(ap(y[in1], c['risk_prefix'].to_numpy()[in1]), 4)}
        out[f'def {dn}'] = r
    return out


def final_fit(d, fs):
    posset, negset = DEFS['A']
    c = d.filter(pl.col('cls').is_in(posset + negset)); y = c['cls'].is_in(posset).to_numpy().astype(float)
    X = X_of(c, fs, 'risk_prefix', 'R1_prefix', 'risk_noT'); m = logreg(X, y, L2)
    info = {'n_rows': int(len(y)), 'n_pos': int(y.sum()), 'l2': L2, 'features': fs, 'weights_standardised': {v: round(float(w), 3) for v, w in zip(fs, m[0])}, 'mu': {v: round(float(u), 3) for v, u in zip(fs, m[2])},
            'sd': {v: round(float(s), 3) for v, s in zip(fs, m[3])}, 'bias': round(float(m[1]), 3), 'AP_in_sample': round(ap(y, predict(m, X)), 4), 'AP_frozen_same_rows': round(ap(y, c['risk_prefix'].to_numpy()), 4)}
    return m, info


def build(ev, score, interval, fit_info, cons):
    base = pl.read_csv(BASE); bmap = dict(zip(base['pair_id'].to_list(), base['risk_score'].to_list()))
    ids = ev['pair_id'].to_list(); base_risk = np.array([bmap[i] for i in ids]); assert np.allclose(base_risk, ev['risk_v13'].to_numpy(), atol=1e-9)
    order = np.argsort(-score, kind='stable'); values = np.sort(base_risk)[::-1]
    mapping = {ids[i]: float(v) for i, v in zip(order, values)}; assert len(mapping) == ev.height == 909
    R1 = ev['R1'].to_numpy(); pct = np.empty(len(score)); pct[order] = np.arange(len(score)) / (len(score) - 1)
    S = pl.read_csv(SFILE); M2 = pl.read_csv(M2FILE)
    pred = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'candidate': str(DEST.relative_to(CV)),
            'rule': f'M3 = rank permutation of ALL {ev.height} evaluation block pairs (risk >= .5 in the base) by a logistic blend of {FS_M3} (FULL = frozen v13 risk; SHORT3 / ALL5 = production recipe refitted on thinned training views 40/25/15 % and full+70/40/25/15 %, all-labelled fits of step 64e; R1 = dominated & risk_noT < .1 as a 0/1 feature), l2 = 10, fitted on the certified 60 %-prefix block of folds 1-2 (def A). The sorted multiset of the 909 base values is re-assigned in that order; no hard demotion; every other row byte-identical to the base.',
            'relation_to_M2': 'Identical construction and fit rows; the only change is the two length-matched views in the blend.',
            'hypothesis': 'OBS 14.39 (step 64): a training length distribution matched to the evaluation length separates length-induced false positives from positives better than any existing view; cross-fold the blend gains +.002 (def A) / +.004 (def C) at floor .5 over M2 with the same sign on both folds. Hand-level pooled evidence and per-family detectors are null and excluded. The recall tail (positives below .5 at evaluation length) is untouched: no view found lifts more than 3 of the 18 testbed tail positives into the top 1 %.',
            'construction_testbed': cons, 'final_fit': fit_info,
            'eval_R1_placement': {'n_R1': int(R1.sum()), 'R1_percentile_median': round(float(np.median(pct[R1])), 3), 'R1_share_in_top_half': round(float((pct[R1] < .5).mean()), 3)},
            'prediction': interval,
            'decision_rule': 'If M2 is measured first: M3 - M2 within [-.002, +.006] is the expected range and confirms the length-matched views as a small stacking lever; a drop below -.003 says the SHORT3/ALL5 final fits do not transfer to evaluation (keep M2). If M3 is measured first: judge against S (.89055) with interval_M3_minus_S. No re-tuning on outcomes.',
            'label_free': 'Evaluation side uses frozen risks, label-trained views, shared_hands, dominance and risk_noT. No leaderboard feedback used to fit anything.', 'not_uploaded_by_claude': True}
    write_json(OUT / 'prediction_M3.json', pred)
    cand = base.with_columns(pl.col('pair_id').replace_strict(mapping, default=None, return_dtype=pl.Float64).alias('new')).with_columns(pl.coalesce([pl.col('new'), pl.col('risk_score')]).alias('risk_score')).drop('new')
    cand.write_csv(DEST); chk = pl.read_csv(DEST)
    same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score')
    blk_after = chk.filter(pl.col('risk_score') >= .5); multiset_ok = bool(np.allclose(np.sort(blk_after['risk_score'].to_numpy()), np.sort(base_risk)))
    outside_same = bool((chk.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy() == base.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy()).all())
    from scipy.stats import spearmanr
    j = chk.join(S.select('pair_id', pl.col('risk_score').alias('rS')), on='pair_id').join(M2.select('pair_id', pl.col('risk_score').alias('rM2')), on='pair_id').filter(pl.col('pair_id').is_in(ids))
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    prot = check_protected()
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'S_sha256': sha(SFILE), 'M_sha256': sha(MFILE), 'M2_sha256': sha(M2FILE), 'C_sha256': sha(CFILE), 'rule': pred['rule'],
               'n_block': ev.height, 'rows_with_changed_risk_vs_base': int((chk['risk_score'] != base['risk_score']).sum()), 'rows_with_changed_risk_vs_S': int((chk['risk_score'] != S['risk_score']).sum()), 'rows_with_changed_risk_vs_M2': int((chk['risk_score'] != M2['risk_score']).sum()),
               'block_spearman_vs_S': round(float(spearmanr(j['risk_score'].to_numpy(), j['rS'].to_numpy()).correlation), 4), 'block_spearman_vs_M2': round(float(spearmanr(j['risk_score'].to_numpy(), j['rM2'].to_numpy()).correlation), 4),
               'block_value_multiset_unchanged': multiset_ok, 'block_ge_.5_after': int(blk_after.height), 'rows_outside_block_identical_to_base': outside_same, 'non_risk_columns_identical_to_base': bool(same_cols),
               'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
    write_json(OUT / 'receipt_M3.json', receipt)
    log(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1)); log('validation:', json.dumps(vrep)[:300])


def main():
    t0 = time.time()
    d = testbed()
    cons = constructions(d); write_json(VD / 'construction_testbed_M3.json', cons); log(json.dumps(cons, indent=1))
    m, info = final_fit(d, FS_M3); write_json(VD / 'multiview_M3_fit.json', info); log(json.dumps(info, indent=1))
    ev = eval_frame().with_columns(pl.col('R1').cast(pl.Int8).alias('R1i'))
    tve = pl.read_parquet(VD / 'tail_views_eval_frame.parquet'); ev = ev.join(tve, on='pair_id', validate='1:1'); assert ev.height == 909
    score = predict(m, X_of(ev, FS_M3, 'risk_v13', 'R1i', 'risk_noT'))
    tb = [cons[dn]['M3: learned views+SHORT3+ALL5+sh+R1 (soft)'][t] - cons[dn]['M2: learned views+sh+R1 (soft)'][t] for dn in ('def A', 'def C') for t in ('test_f2sh', 'test_f1')]
    m2pred = json.load(open(OUT / 'prediction_softR1.json', encoding='utf-8'))['prediction']
    interval = {'testbed_M3_minus_M2_crossfold': [round(x, 4) for x in tb], 'interval_M3_minus_M2': [round(min(tb) - .002, 3), round(max(tb) + .002, 3)],
                'interval_M3_minus_S': [round(m2pred['interval_M2_minus_S'][0] + min(tb) - .001, 3), round(m2pred['interval_M2_minus_S'][1] + max(tb) + .001, 3)],
                'interval_M3_vs_base': [round(m2pred['interval_M2_vs_base'][0] + min(tb) - .001, 3), round(m2pred['interval_M2_vs_base'][1] + max(tb) + .001, 3)],
                'M2_intervals_inherited': m2pred, 'note': 'Testbed transfer (S: testbed +.0138 -> public +.0143), +-.002 for the extra transfer error of the new views; scenario model not used (OBS 14.37).'}
    log('interval', json.dumps(interval))
    build(ev, score, interval, info, cons)
    log('seconds', round(time.time() - t0))


if __name__ == '__main__':
    main()
