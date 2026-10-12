"""Step 63: hard vs soft R1 on the certified prefix block, and candidate M2 = learned multi-view blend WITH the R1 flag as a feature
(soft demotion) over all block pairs, pre-registered.
Step 61 (OBS 14.38): at the 60 % prefix the R1 rule (dominated & risk_noT < .1, computed from the prefix scores) flags 67 certified
false positives + 22 ambiguous + 19 full-mid + 8 undisclosed_R1 against 1 labelled positive + 7 undisclosed (full >= .9, not R1 at
full length). A HARD demotion of the R1 pairs to the bottom (candidates C, D, S, M) costs -.0056 block AP on def A relative to the
fixed blend alone (.9513 vs .9569) because the 8 positives it drags down outweigh the false positives that the blend already ranks
low; R1 as a FEATURE of the learned blend (weight -.4 standardised) gains +.003 on both definitions (def A .9656 vs .9627, def C
.9716 vs .9685). This step (1) scores the candidate constructions cross-fold on the same rows - fixed / learned, hard R1 / soft R1 /
no R1 - so that the choice is made on certified data, (2) fits the final blend (views + log sh + R1, l2 = 10, union of both folds'
certified block rows, def A), (3) builds M2: all 909 evaluation block pairs re-assigned the sorted multiset of their base values in
learned order (no pair leaves the block; R1 pairs are demoted only as far as their views and the R1 weight say), (4) pre-registers.
Outputs: experiments/viewdis_20260911/{construction_testbed.json, multiview_softR1_fit.json},
         outputs_multiview_20260911/{prediction_softR1.json, submission_multiview_softR1.csv, receipt_softR1.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
import numpy as np
import polars as pl
from pu_pairs.build_allpairs_fold import sha, fold_pools
from pu_pairs.step60_multiview_blend import lg, ap, logreg, predict, NEW_VIEWS
from pu_pairs.step62_multiview_candidate import VD, SP, OUT, SFILE, VIEWS, L2, DEFS, eval_frame
from pu_pairs.step59_blend_candidate import BASE, CFILE

MFILE = OUT / 'submission_multiview_learned_plus_R1.csv'
SETS = {'views+sh': VIEWS + ['sh'], 'views+sh+R1': VIEWS + ['sh', 'R1'], 'views+sh+R1+noT': VIEWS + ['sh', 'R1', 'noT'], 'views+sh+noT': VIEWS + ['sh', 'noT']}


def log(*a):
    print(*a, flush=True)


def X_of(c, fs, full_col, r1_col, noT_col):
    cols = []
    for v in fs:
        if v == 'sh': cols.append(np.log(c['shared_hands'].to_numpy().astype(float)))
        elif v == 'FULL': cols.append(lg(c[full_col].to_numpy()))
        elif v == 'R1': cols.append(c[r1_col].to_numpy().astype(float))
        elif v == 'noT': cols.append(lg(c[noT_col].fill_nan(0).fill_null(0).to_numpy()))
        else: cols.append(lg(c[f'r_{v}'].to_numpy()))
    return np.column_stack(cols)


def load_testbed():
    d = pl.read_parquet(VD / 'prefix_rescoring.parquet')
    assert 'R1_prefix' in d.columns and 'cls' in d.columns
    return d


def constructions(d):
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    res = {}
    for dn, (posset, negset) in DEFS.items():
        c = d.filter(pl.col('cls').is_in(posset + negset))
        y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy(); risk = c['risk_prefix'].to_numpy(); raw = c['r_RAW'].to_numpy(); R1 = c['R1_prefix'].to_numpy()
        in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
        def crossfold(fs):
            X = X_of(c, fs, 'risk_prefix', 'R1_prefix', 'risk_noT'); pred = np.full(len(y), np.nan)
            m1 = logreg(X[in1], y[in1], L2); pred[in2 | insh] = predict(m1, X[in2 | insh]); m2 = logreg(X[in2], y[in2], L2); pred[in1] = predict(m2, X[in1])
            return pred
        def hard(score):
            return np.where(R1, -1e6 + lg(c['risk_noT'].fill_nan(0).fill_null(0).to_numpy()), score)
        fixed = lg(risk) + .5 * lg(raw)
        orders = {'frozen': risk, 'C-like: frozen + hard R1': hard(risk), 'S-like: fixed blend (no R1)': fixed, 'S: fixed blend + hard R1': hard(fixed)}
        preds = {k: crossfold(fs) for k, fs in SETS.items()}
        orders['M: learned views+sh + hard R1'] = hard(preds['views+sh']); orders['learned views+sh (no R1)'] = preds['views+sh']
        orders['M2: learned views+sh+R1 (soft)'] = preds['views+sh+R1']; orders['learned views+sh+R1+noT (soft)'] = preds['views+sh+R1+noT']; orders['learned views+sh+noT (soft)'] = preds['views+sh+noT']
        orders['learned views+sh+R1 (soft) + hard R1'] = hard(preds['views+sh+R1'])
        r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'n_R1': int(R1.sum()), 'R1_pos': int(y[R1].sum()), 'R1_neg': int((1 - y[R1]).sum())}
        for k, s in orders.items():
            r[k] = {'pooled': round(ap(y, s), 4), 'test_f2sh': round(ap(y[in2 | insh], s[in2 | insh]), 4), 'test_f1': round(ap(y[in1], s[in1]), 4)}
        # where do the R1 positives / negatives land under the soft order (percentile rank within the block)
        s = preds['views+sh+R1']; pct = np.empty(len(s)); pct[np.argsort(-s, kind='stable')] = np.arange(len(s)) / (len(s) - 1)
        r['soft_R1_percentile'] = {'R1_pos_median': round(float(np.median(pct[R1 & (y == 1)])), 3) if (R1 & (y == 1)).any() else None, 'R1_neg_median': round(float(np.median(pct[R1 & (y == 0)])), 3),
                                   'R1_neg_share_in_top_half': round(float((pct[R1 & (y == 0)] < .5).mean()), 3), 'R1_pos_share_in_top_half': round(float((pct[R1 & (y == 1)] < .5).mean()), 3) if (R1 & (y == 1)).any() else None}
        res[f'def {dn}'] = r
    return res


def final_fit(d, fs):
    posset, negset = DEFS['A']
    c = d.filter(pl.col('cls').is_in(posset + negset)); y = c['cls'].is_in(posset).to_numpy().astype(float)
    X = X_of(c, fs, 'risk_prefix', 'R1_prefix', 'risk_noT'); m = logreg(X, y, L2)
    info = {'n_rows': int(len(y)), 'n_pos': int(y.sum()), 'l2': L2, 'features': fs, 'weights_standardised': {v: round(float(w), 3) for v, w in zip(fs, m[0])}, 'mu': {v: round(float(u), 3) for v, u in zip(fs, m[2])},
            'sd': {v: round(float(s), 3) for v, s in zip(fs, m[3])}, 'bias': round(float(m[1]), 3), 'AP_in_sample': round(ap(y, predict(m, X)), 4), 'AP_frozen_same_rows': round(ap(y, c['risk_prefix'].to_numpy()), 4)}
    return m, info


def build(ev, score, fs, interval, fit_info, cons):
    base = pl.read_csv(BASE); bmap = dict(zip(base['pair_id'].to_list(), base['risk_score'].to_list()))
    ids = ev['pair_id'].to_list(); base_risk = np.array([bmap[i] for i in ids]); assert np.allclose(base_risk, ev['risk_v13'].to_numpy(), atol=1e-9)
    order = np.argsort(-score, kind='stable'); values = np.sort(base_risk)[::-1]
    mapping = {ids[i]: float(v) for i, v in zip(order, values)}; assert len(mapping) == ev.height == 909
    R1 = ev['R1'].to_numpy(); pct = np.empty(len(score)); pct[order] = np.arange(len(score)) / (len(score) - 1)
    S = pl.read_csv(SFILE); M = pl.read_csv(MFILE)
    DEST = OUT / 'submission_multiview_softR1.csv'
    pred = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'candidate': str(DEST.relative_to(CV)),
            'rule': f'M2 = rank permutation of ALL {ev.height} evaluation block pairs (risk >= .5 in the base) by a logistic blend of {fs} (FULL = frozen v13 risk; R1 = dominated & risk_noT < .1 as a 0/1 feature), l2 = 10, fitted on the certified 60 %-prefix block of folds 1-2 (def A, R1 computed at the prefix). The sorted multiset of the 909 base values is re-assigned in that order; no hard demotion; every other row byte-identical to the base.',
            'relation_to_S_and_M': 'S and M demote the 205 R1 pairs to [.49,.4949] regardless of their other views; M2 demotes them only as far as the learned R1 weight says, so R1 pairs with strong views can stay high.',
            'hypothesis': 'OBS 14.38 (step 61/63): at evaluation-like length the R1 rule flags ~10 % positives among the pairs it demotes; on the certified block the hard demotion costs block AP relative to the blend alone under def A while the soft R1 feature gains on both definitions. See construction_testbed.json.',
            'construction_testbed': cons, 'final_fit': fit_info,
            'eval_R1_placement': {'n_R1': int(R1.sum()), 'R1_percentile_median': round(float(np.median(pct[R1])), 3), 'R1_share_in_top_half': round(float((pct[R1] < .5).mean()), 3), 'R1_share_in_bottom_quarter': round(float((pct[R1] >= .75).mean()), 3)},
            'prediction': interval,
            'decision_rule': 'Compare with the measured S (.89055) and, if uploaded, M. M2 - S > +.003: the soft demotion is right, the hard R1 demotion of S/M was over-correcting; stack the tail next. Within +-.003: hard and soft are equivalent on evaluation. < -.003: the R1 pairs on evaluation are more often false positives than on the prefix (hard demotion was right); keep M/S. No re-tuning on outcomes.',
            'label_free': 'Evaluation side uses frozen risks, label-trained views, shared_hands, dominance and risk_noT (label-free re-scoring). No leaderboard feedback used to fit anything.', 'not_uploaded_by_claude': True}
    write_json(OUT / 'prediction_softR1.json', pred)
    cand = base.with_columns(pl.col('pair_id').replace_strict(mapping, default=None, return_dtype=pl.Float64).alias('new')).with_columns(pl.coalesce([pl.col('new'), pl.col('risk_score')]).alias('risk_score')).drop('new')
    cand.write_csv(DEST); chk = pl.read_csv(DEST)
    same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score')
    blk_after = chk.filter(pl.col('risk_score') >= .5); multiset_ok = bool(np.allclose(np.sort(blk_after['risk_score'].to_numpy()), np.sort(base_risk)))
    outside_same = bool((chk.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy() == base.filter(~pl.col('pair_id').is_in(ids))['risk_score'].to_numpy()).all())
    from scipy.stats import spearmanr
    j = chk.join(S.select('pair_id', pl.col('risk_score').alias('rS')), on='pair_id').join(M.select('pair_id', pl.col('risk_score').alias('rM')), on='pair_id').filter(pl.col('pair_id').is_in(ids))
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    prot = check_protected()
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'S_sha256': sha(SFILE), 'M_sha256': sha(MFILE), 'C_sha256': sha(CFILE), 'rule': pred['rule'],
               'n_block': ev.height, 'rows_with_changed_risk_vs_base': int((chk['risk_score'] != base['risk_score']).sum()), 'rows_with_changed_risk_vs_S': int((chk['risk_score'] != S['risk_score']).sum()), 'rows_with_changed_risk_vs_M': int((chk['risk_score'] != M['risk_score']).sum()),
               'block_spearman_vs_S': round(float(spearmanr(j['risk_score'].to_numpy(), j['rS'].to_numpy()).correlation), 4), 'block_spearman_vs_M': round(float(spearmanr(j['risk_score'].to_numpy(), j['rM'].to_numpy()).correlation), 4),
               'block_value_multiset_unchanged': multiset_ok, 'block_ge_.5_after': int(blk_after.height), 'rows_outside_block_identical_to_base': outside_same, 'non_risk_columns_identical_to_base': bool(same_cols),
               'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
    write_json(OUT / 'receipt_softR1.json', receipt)
    log(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1)); log('validation:', json.dumps(vrep)[:300])


def main():
    t0 = time.time()
    d = load_testbed()
    cons = constructions(d); write_json(VD / 'construction_testbed.json', cons); log(json.dumps(cons, indent=1))
    fs = SETS['views+sh+R1']
    m, info = final_fit(d, fs); write_json(VD / 'multiview_softR1_fit.json', info); log(json.dumps(info, indent=1))
    ev = eval_frame().with_columns(pl.col('R1').cast(pl.Int8).alias('R1i'))
    score = predict(m, X_of(ev, fs, 'risk_v13', 'R1i', 'risk_noT'))
    # pre-registration from testbed transfer (S: testbed +.0138 -> public +.0143): M2 minus S (soft vs hard) cross-fold deltas on both definitions
    tb = [cons[dn]['M2: learned views+sh+R1 (soft)'][t] - cons[dn]['S: fixed blend + hard R1'][t] for dn in ('def A', 'def C') for t in ('test_f2sh', 'test_f1')]
    tbM = [cons[dn]['M2: learned views+sh+R1 (soft)'][t] - cons[dn]['M: learned views+sh + hard R1'][t] for dn in ('def A', 'def C') for t in ('test_f2sh', 'test_f1')]
    interval = {'testbed_M2_minus_S_crossfold': [round(x, 4) for x in tb], 'testbed_M2_minus_M_crossfold': [round(x, 4) for x in tbM],
                'interval_M2_minus_S': [round(min(tb) - .003, 3), round(max(tb) + .003, 3)], 'interval_M2_minus_M': [round(min(tbM) - .003, 3), round(max(tbM) + .003, 3)],
                'interval_M2_vs_base': [round(.0143 + min(tb) - .004, 3), round(.0143 + max(tb) + .004, 3)],
                'note': 'Testbed transfer with +-.003 / .004 for transfer error; the step-59 scenario model is not used (structurally blind beyond RAW, OBS 14.37). The soft-vs-hard part depends on the share of true positives among the evaluation R1 pairs, which the prefix estimates at ~10 %.'}
    log('interval', json.dumps(interval))
    build(ev, score, fs, interval, info, cons)
    log('seconds', round(time.time() - t0))


if __name__ == '__main__':
    main()
