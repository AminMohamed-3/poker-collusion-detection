"""Step 22: are the high-scoring UNLABELLED development pairs undisclosed positives or false positives?

Training-only, label-free for the unlabelled pairs. Four checks on the fold scores from step 21, restricted to
pairs with >= 38 shared hands (the evaluation minimum):
 1. Per-table structure: labelled positives vs unlabelled high scorers per table. A fixed planting budget per
    table shows up as a NEGATIVE correlation (tables with 2 disclosed positives have fewer undisclosed ones);
    model false positives would be independent of the disclosed count.
 2. Cross-phase consistency: the same two players also form a pair in the evaluation phase (players are unique
    to their table, both phases). If a dev unlabelled-high pair is a real colluding pair, its evaluation
    counterpart should score high with the frozen v13 model far more often than a random dev pair does.
    Labelled positives cannot be checked (their players are excluded from evaluation), so the comparison is
    unlabelled-high vs unlabelled-low at the same shared-hands range.
 3. Separability: labelled positives vs unlabelled-high (risk >= .9) with the 897 pair features, pool-grouped
    5-fold AUC. AUC ~.5 means the two groups are one population; a high AUC means the high scorers differ
    systematically from real positives (a false-positive population). Selection bias caveat noted in output.
 4. Family signature: mean of a few interpretable features (net chip flow, showdown, partner folds) for
    labelled positives by family vs unlabelled-high by predicted family vs labelled negatives.
Output: Claude Version/experiments/pu_pairs_20260911/fold{F}/unlabelled-audit.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
sys.path.insert(0, str(ORIGINAL / 'src'))
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from robust_pair_experiment import matrix  # noqa: E402
from pu_pairs.build_allpairs_fold import EXP, CONF  # noqa: E402

MIN_SHARED = 38
FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}
SIG = ['net_bb_gap_mean', 'net_bb_sum_mean', 'opposite_outcomes_mean', 'went_to_showdown_sum_mean', 'both_fold_mean', 'joint_win_share_mean',
       'v2_partner_fold_count_sum_mean', 'v2_partner_call_count_sum_mean', 'v2_soft_heads_up_check_sum_mean', 'v2_isolation_reraise_sum_mean',
       'v2_weak_partner_call_sum_mean', 'v2_soft_favorable_fold_sum_mean', 'v2_partner_commit_behind_sum_mean', 'strong_fold_to_partner_mean']


def fmean(x):
    return float(np.mean(np.asarray(x, dtype=float))) if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=1)
    args = ap.parse_args()
    out = EXP / f'fold{args.fold}'
    scores = pl.read_parquet(out / 'scores.parquet')
    feats = pl.concat([pl.read_parquet(b) for b in sorted(out.glob('batch_*-pairs.parquet'))])
    s = scores.filter(pl.col('shared_hands') >= MIN_SHARED)
    res = {'fold': args.fold, 'min_shared_hands': MIN_SHARED, 'pairs_all': scores.height, 'pairs_eligible': s.height,
           'eligible_counts': {'positives': int((s['label'] == 1).sum()), 'labelled_negatives': int((s['label'] == 0).sum()), 'unlabelled': int((s['label'] < 0).sum())}}
    unl = s.filter(pl.col('label') < 0)
    pos = s.filter(pl.col('label') == 1)
    neg = s.filter(pl.col('label') == 0)
    for t in (.99, .9, .5, .1):
        res[f'eligible_ge_{t}'] = {'positives': int((pos['risk'] >= t).sum()), 'labelled_negatives': int((neg['risk'] >= t).sum()), 'unlabelled': int((unl['risk'] >= t).sum()),
                                   'unlabelled_per_table': float((unl['risk'] >= t).sum() / s['table_id'].n_unique())}
    res['labelled_negative_rate_ge'] = {str(t): float((neg['risk'] >= t).mean()) for t in (.99, .9, .5, .1)}
    res['unlabelled_rate_ge'] = {str(t): float((unl['risk'] >= t).mean()) for t in (.99, .9, .5, .1)}
    # 1. per-table structure
    per = s.group_by('table_id').agg(((pl.col('label') == 1)).sum().alias('disclosed_pos'), ((pl.col('label') < 0) & (pl.col('risk') >= .9)).sum().alias('unl_hi'),
                                     ((pl.col('label') < 0) & (pl.col('risk') >= .5)).sum().alias('unl_mid'))
    res['per_table'] = {'tables': per.height, 'disclosed_pos_mean': fmean(per['disclosed_pos']), 'unl_hi_mean': fmean(per['unl_hi']), 'total_hi_mean': fmean(per['disclosed_pos'] + per['unl_hi']),
                        'corr_disclosed_vs_unl_hi': float(np.corrcoef(per['disclosed_pos'].to_numpy(), per['unl_hi'].to_numpy())[0, 1]),
                        'total_hi_distribution': {str(k): int(v) for k, v in (per['disclosed_pos'] + per['unl_hi']).value_counts().sort('disclosed_pos').iter_rows()},
                        'unl_hi_by_disclosed': {str(k): fmean(g['unl_hi']) for k, g in sorted(per.group_by('disclosed_pos'), key=lambda kv: kv[0][0])},
                        'unl_hi_var_over_mean': float(per['unl_hi'].var() / max(1e-9, per['unl_hi'].mean()))}
    # 2. cross-phase consistency with the frozen v13 evaluation scores
    ev = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').join(pl.read_csv(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv').select('pair_id', 'risk_score', 'predicted_behavior'), on='pair_id')
    x = unl.join(ev.select('player_1', 'player_2', pl.col('risk_score').alias('eval_risk'), pl.col('shared_hands').alias('eval_shared'), pl.col('predicted_behavior').alias('eval_behavior')), on=['player_1', 'player_2'], how='left')
    res['cross_phase'] = {'unlabelled_eligible_with_eval_counterpart': int(x['eval_risk'].is_not_null().sum())}
    x = x.filter(pl.col('eval_risk').is_not_null())
    for name, mask in [('dev_ge_0.99', pl.col('risk') >= .99), ('dev_ge_0.9', pl.col('risk') >= .9), ('dev_0.5_0.9', (pl.col('risk') >= .5) & (pl.col('risk') < .9)), ('dev_0.1_0.5', (pl.col('risk') >= .1) & (pl.col('risk') < .5)), ('dev_lt_0.1', pl.col('risk') < .1)]:
        g = x.filter(mask)
        if g.height:
            res['cross_phase'][name] = {'n': g.height, 'eval_risk_mean': fmean(g['eval_risk']), 'eval_ge_0.5': float((g['eval_risk'] >= .5).mean()), 'eval_ge_0.9': float((g['eval_risk'] >= .9).mean()),
                                       'same_family_when_both_ge_0.5': fmean([FAM.get(int(c)) == b for c, b, r, e in zip(g['predicted_class'], g['eval_behavior'], g['risk'], g['eval_risk']) if r >= .5 and e >= .5])}
    # 3. separability of labelled positives vs unlabelled-high
    bundle = joblib.load(CONF / 'results' / f'fold{args.fold}' / 'conditional-pair.joblib')
    features = bundle['features']
    imp = np.asarray(bundle['model'].feature_importances_, dtype=float)
    top = set(np.array(features)[np.argsort(-imp)[:100]].tolist())
    rest = [f for f in features if f not in top]
    hi = unl.filter(pl.col('risk') >= .9)
    frame = pl.concat([pos.select('pair_id', 'table_id').with_columns(pl.lit(1).alias('y')), hi.select('pair_id', 'table_id').with_columns(pl.lit(0).alias('y'))]).join(feats.drop('table_id'), on='pair_id')
    res['separability'] = {'positives': pos.height, 'unlabelled_ge_0.9': hi.height, 'caveat': 'unlabelled-high were selected by the same model family; AUC near .5 supports one population, high AUC flags a different population'}
    if pos.height >= 20 and hi.height >= 20:
        y = frame['y'].to_numpy(); groups = frame['table_id'].to_numpy()
        for name, cols in [('all_897', features), ('outside_top100_importance', rest)]:
            X = matrix(frame, cols)
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
                clf = lgb.LGBMClassifier(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=20, learning_rate=.05, colsample_bytree=.5, n_jobs=2, verbosity=-1, random_state=3)
                clf.fit(X[tr], y[tr])
                pred[te] = clf.predict_proba(X[te])[:, 1]
            res['separability'][name] = {'auc': float(roc_auc_score(y, pred))}
        # same test for labelled positives vs labelled negatives as a reference (should be ~1)
        ref = pl.concat([pos.select('pair_id', 'table_id').with_columns(pl.lit(1).alias('y')), neg.select('pair_id', 'table_id').with_columns(pl.lit(0).alias('y'))]).join(feats.drop('table_id'), on='pair_id')
        y = ref['y'].to_numpy(); groups = ref['table_id'].to_numpy(); X = matrix(ref, rest); pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = lgb.LGBMClassifier(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=20, learning_rate=.05, colsample_bytree=.5, n_jobs=2, verbosity=-1, random_state=3)
            clf.fit(X[tr], y[tr]); pred[te] = clf.predict_proba(X[te])[:, 1]
        res['separability']['reference_pos_vs_labelled_neg_outside_top100'] = {'auc': float(roc_auc_score(y, pred))}
    # 4. family signatures
    sig = {}
    fs = feats.select('pair_id', *[c for c in SIG if c in feats.columns])
    for fam_id, fam in FAM.items():
        a = pos.filter(pl.col('behavior_family') == fam).join(fs, on='pair_id')
        b = hi.filter(pl.col('predicted_class') == fam_id).join(fs, on='pair_id')
        sig[fam] = {'labelled_pos_n': a.height, 'unlabelled_hi_n': b.height, 'features': {c: {'labelled_pos': fmean(a[c]), 'unlabelled_hi': fmean(b[c])} for c in fs.columns if c != 'pair_id'}}
    n = neg.join(fs, on='pair_id')
    sig['labelled_negatives'] = {'n': n.height, 'features': {c: fmean(n[c]) for c in fs.columns if c != 'pair_id'}}
    res['family_signature'] = sig
    res['unlabelled_hi_predicted_family'] = {FAM[int(k)]: int(v) for k, v in hi.group_by('predicted_class').agg(pl.len()).iter_rows()}
    res['unlabelled_hi_family_confidence_quantiles_10_50'] = [float(v) for v in np.quantile(hi.select(pl.max_horizontal('p1', 'p2', 'p3')).to_numpy().ravel(), [.1, .5])] if hi.height else None
    res['labelled_pos_family_confidence_quantiles_10_50'] = [float(v) for v in np.quantile(pos.select(pl.max_horizontal('p1', 'p2', 'p3')).to_numpy().ravel(), [.1, .5])] if pos.height else None
    write_json(out / 'unlabelled-audit.json', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'family_signature'}, indent=1, default=str))
    for fam, d in sig.items():
        print(fam, json.dumps(d, default=lambda v: round(v, 3) if isinstance(v, float) else v)[:1500])


if __name__ == '__main__':
    main()
