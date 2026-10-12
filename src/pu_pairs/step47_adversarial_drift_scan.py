"""Step 47 (label-free): per-feature adversarial validation, development vs evaluation, over the 897 production pair features.
Development side = fold-1 and fold-2 pool frames (full histories, shared_hands >= 38, mostly null pairs, labels NOT used
except to report labelled negatives separately); evaluation side = the step-46 frame (112,540 pairs). A feature whose
rank-AUC for 'is evaluation' is far from .5 is distributed differently across phases: the model learned it on one phase
and applies it on the other. Also fits a global adversarial LightGBM (GroupKFold by table) to measure total separability.
Output: experiments/eval_frame_20260911/adversarial_scan.json."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pu_pairs.build_allpairs_fold import EXP, CONF

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'eval_frame_20260911'
MIN_SH = 38


def rank_auc(x, is_eval):
    """AUC of x for predicting is_eval via Mann-Whitney on ranks (ties averaged)."""
    r = rankdata(x)
    n1 = is_eval.sum(); n0 = len(x) - n1
    return float((r[is_eval].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    bundle = joblib.load(CONF / 'results' / 'fold1' / 'conditional-pair.joblib'); features = list(bundle['features'])
    imp = dict(zip(features, bundle['model'].feature_importances_.tolist())) if 'model' in bundle and hasattr(bundle['model'], 'feature_importances_') else {}
    ev = pl.read_parquet(OUT / 'eval_frame.parquet')
    dev = pl.concat([pl.concat([pl.read_parquet(b) for b in sorted((EXP / f'fold{k}').glob('batch_*-pairs.parquet'))]).join(
        pl.read_csv(EXP / f'fold{k}' / 'pairs.csv').select('pair_id', 'label'), on='pair_id', how='left') for k in (1, 2)]).unique(subset=['pair_id'])
    dev = dev.filter(pl.col('shared_hands') >= MIN_SH)
    ev = ev.filter(pl.col('shared_hands') >= MIN_SH)
    cols = ['pair_id', 'table_id'] + features
    both = pl.concat([dev.select(cols).with_columns(pl.lit(0).alias('is_eval'), pl.col('pair_id')), ev.select(cols).with_columns(pl.lit(1).alias('is_eval'))])
    is_eval = both['is_eval'].to_numpy().astype(bool)
    out = {'dev_pairs': int(dev.height), 'dev_tables': int(dev['table_id'].n_unique()), 'eval_pairs': int(ev.height), 'eval_tables': int(ev['table_id'].n_unique()), 'features': len(features)}
    per = {}
    for f in features:
        x = np.nan_to_num(both[f].to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        per[f] = rank_auc(x, is_eval)
    # labelled negatives only vs eval (clean pairs both sides)
    negs = dev.filter(pl.col('label') == 0)
    both_n = pl.concat([negs.select(cols).with_columns(pl.lit(0).alias('is_eval')), ev.select(cols).with_columns(pl.lit(1).alias('is_eval'))])
    is_eval_n = both_n['is_eval'].to_numpy().astype(bool)
    per_n = {f: rank_auc(np.nan_to_num(both_n[f].to_numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0), is_eval_n) for f in features}
    dev_dist = sorted(((abs(a - .5), f) for f, a in per.items()), reverse=True)
    out['labelled_negatives'] = int(negs.height)
    out['n_features_abs_auc_minus_half_ge'] = {str(t): int(sum(1 for d, _ in dev_dist if d >= t)) for t in (.05, .1, .15, .2, .25, .3)}
    out['top40_drifting_features'] = [{'feature': f, 'auc_eval_vs_dev': round(per[f], 4), 'auc_eval_vs_labelled_neg': round(per_n[f], 4), 'importance_fold1': imp.get(f)} for _, f in dev_dist[:40]]
    # group by feature prefix (family of features)
    def group(f):
        for p in ('ca_', 'v2_', 'seq_', 'eq_'):
            if f.startswith(p): return p
        return f.split('_')[0]
    groups = {}
    for f, a in per.items():
        groups.setdefault(group(f), []).append(abs(a - .5))
    out['mean_abs_drift_by_group'] = {g: {'n': len(v), 'mean_abs_auc_minus_half': round(float(np.mean(v)), 4), 'max': round(float(np.max(v)), 4)} for g, v in sorted(groups.items(), key=lambda kv: -np.mean(kv[1]))}
    # importance-weighted drift: do the features the model uses most drift?
    if imp:
        tot = sum(imp.values()) or 1
        out['importance_weighted_mean_abs_drift'] = round(float(sum(imp[f] * abs(per[f] - .5) for f in features) / tot), 4)
        out['unweighted_mean_abs_drift'] = round(float(np.mean([abs(a - .5) for a in per.values()])), 4)
        top_imp = sorted(features, key=lambda f: -imp[f])[:30]
        out['top30_importance_features_drift'] = [{'feature': f, 'importance': imp[f], 'auc_eval_vs_dev': round(per[f], 4)} for f in top_imp]
    # global adversarial model (subsample dev/eval to balance, GroupKFold by table)
    rng = np.random.default_rng(7329)
    X = np.nan_to_num(both.select(features).to_numpy().astype(np.float32), nan=0, posinf=0, neginf=0); y = is_eval.astype(int); g = both['table_id'].to_numpy()
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=50, colsample_bytree=.5, n_jobs=2, verbosity=-1, random_state=7329)
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict_proba(X[te])[:, 1]
    out['global_adversarial_auc'] = round(float(roc_auc_score(y, pred)), 4)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=50, colsample_bytree=.5, n_jobs=2, verbosity=-1, random_state=7329).fit(X, y)
    gi = sorted(zip(features, clf.booster_.feature_importance('gain')), key=lambda kv: -kv[1])[:30]
    out['global_adversarial_top30_gain'] = [{'feature': f, 'gain': round(float(v), 1), 'auc_eval_vs_dev': round(per[f], 4)} for f, v in gi]
    # drift of shared_hands itself (eligibility / length)
    out['shared_hands'] = {'dev_q': [float(x) for x in np.quantile(dev['shared_hands'].to_numpy(), [.1, .25, .5, .75, .9])], 'eval_q': [float(x) for x in np.quantile(ev['shared_hands'].to_numpy(), [.1, .25, .5, .75, .9])]}
    out['seconds'] = round(time.time() - t0)
    write_json(OUT / 'adversarial_scan.json', out)
    write_json(OUT / 'adversarial_per_feature.json', {'auc_eval_vs_dev': per, 'auc_eval_vs_labelled_neg': per_n})
    print(json.dumps({k: out[k] for k in ['dev_pairs', 'eval_pairs', 'n_features_abs_auc_minus_half_ge', 'global_adversarial_auc', 'importance_weighted_mean_abs_drift', 'unweighted_mean_abs_drift', 'shared_hands'] if k in out}, indent=1))
    print('top drifting:', json.dumps(out['top40_drifting_features'][:15], indent=0))
    print('by group:', json.dumps(out['mean_abs_drift_by_group'], indent=0))
    print('global top gain:', json.dumps(out['global_adversarial_top30_gain'][:12], indent=0))


if __name__ == '__main__':
    main()
