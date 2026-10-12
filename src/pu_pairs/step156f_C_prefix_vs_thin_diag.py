"""Step 156f (Claude): descriptive diagnostic registered in prediction_step156.json (sha fba91dde). Can a LightGBM tell prefix40 rows
(fold-2 all-pairs frame, fold-2 held CA policy) from fraction-.4 thinned rows (fold-1 training frame, draw 0, fold-1 CA policy) of the
SAME labelled pairs (fold-2 tables outside fold-1 pools)? 5-fold GroupKFold by table; AUC on all 897 features and on the 765 non-ca_
features. Collusion labels are not used. Output experiments/tail_keys_20260915_neg/prefix_vs_thin_diag.json."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pu_pairs.build_allpairs_fold import CONF, fold_pools
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded
from pu_pairs.step156b_C_SXN_views import candidates, read_rows
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import matrix  # noqa: E402


def main():
    t0 = time.time()
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    held = set(fold_pools(1))
    fr = view_frames_seeded(CONF / 'conditional-fold1-v1' / 'policy-screen' / 'hand-features.parquet', held, features, 0)
    cand = candidates(2, held).filter(pl.col('label') >= 0)
    ids = sorted(set(cand['pair_id'].to_list()) & set(fr[0.4]['pair_id'].to_list()))
    thin = fr[0.4].filter(pl.col('pair_id').is_in(ids)).select(['pair_id', 'table_id'] + features).sort('pair_id')
    pref = read_rows(2, 'prefix40', ids, features).join(thin.select('pair_id', 'table_id'), on='pair_id').select(['pair_id', 'table_id'] + features).sort('pair_id')
    both = pl.concat([thin.with_columns(pl.lit(0).alias('y')), pref.with_columns(pl.lit(1).alias('y'))], how='vertical_relaxed')
    y, grp = both['y'].to_numpy(), both['table_id'].to_numpy()
    res = {'pairs': len(ids), 'rows': both.height}
    for name, fs in (('all_features', features), ('non_ca_features', [f for f in features if not f.startswith('ca_')])):
        X = matrix(both, fs); p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            p[te] = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=20, colsample_bytree=.8, verbosity=-1, random_state=1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=20, colsample_bytree=.8, verbosity=-1, random_state=1).fit(X, y)
        imp = sorted(zip(m.booster_.feature_importance('gain'), fs), reverse=True)[:10]
        res[name] = {'auc': round(float(roc_auc_score(y, p)), 4), 'n_features': len(fs), 'top_gain': [[f, round(float(g), 1)] for g, f in imp]}
    res['seconds'] = round(time.time() - t0)
    out = CV / 'experiments' / 'tail_keys_20260915_neg' / 'prefix_vs_thin_diag.json'
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
