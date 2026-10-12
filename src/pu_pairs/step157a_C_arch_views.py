"""Step 157a (Claude): new-architecture tail views of outputs_step156_20260915/prediction_step157.json (sha 1a5208f4). For draw k, rebuild
exactly the incumbent frames (step151c seeds; with --neg also the step156b added rows and seeds) and fit XGB3 (xgboost multiclass, ALL5
frames), ET (ExtraTrees binary, ALL5 frames) and MLP (binary, PROD frames, LIN standardisation). The five incumbent view types are not refitted.
Development: step157a_C_arch_views.py dev NEG FOLD K0 K1 -> experiments/arch_views_20260915/arch_dev_neg{NEG}_fold{F}_k{K0}-{K1}.parquet
Evaluation:  step157a_C_arch_views.py eval NEG K0 K1     -> experiments/arch_views_20260915/arch_eval_neg{NEG}_k{K0}-{K1}.parquet"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time, hashlib, warnings
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs import step64a_tail_aggregate_views as S64
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded
from pu_pairs.step156b_C_SXN_views import candidates, added_rows, ADD
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, matrix  # noqa: E402

REG = CV / 'outputs_step156_20260915' / 'prediction_step157.json'
REG_SHA = '1a5208f4aa5e9be7eab9b9106a7c317c87816f8bf1d95dcd0db71005b477aee5'
OUT = CV / 'experiments' / 'arch_views_20260915'


def fit_arch(fr, add, features, k):
    cols = ['pair_id', 'label', 'behavior_family'] + features
    def ts(name):
        parts = [fr[f].select(cols) for f in S64.TRAIN_SETS[name]] + ([add[L].select(cols) for L in ADD[name]] if add else [])
        return pl.concat(parts, how='vertical_relaxed')
    M = {}
    all5 = ts('ALL5'); X5 = matrix(all5, features)
    M['XGB3'] = ('mc', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.5, min_child_weight=2, reg_lambda=5,
                                         tree_method='hist', random_state=7329 + k, n_jobs=2, verbosity=0).fit(X5, np.array([LABELS[f] for f in all5['behavior_family']])))
    M['ET'] = ('bin', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features='sqrt', random_state=7329 + k, n_jobs=2).fit(X5, (all5['label'] == 1).to_numpy().astype(int)))
    del X5, all5
    prod = ts('PROD'); Xp = matrix(prod, features).astype(float); mu = Xp.mean(0); sd = Xp.std(0) + 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=.01, early_stopping=True, max_iter=200, random_state=7329 + k).fit(S64.lin_transform(Xp, mu, sd), (prod['label'] == 1).to_numpy().astype(int))
    M['MLP'] = ('mlp', (mlp, mu, sd))
    return M


def score_arch(M, frame, features):
    X = matrix(frame, features); out = {}
    for name, (kind, m) in M.items():
        if kind == 'mc': out[f'r_{name}'] = 1 - m.predict_proba(X)[:, 0]
        elif kind == 'bin': out[f'r_{name}'] = m.predict_proba(X)[:, 1]
        else:
            mlp, mu, sd = m; out[f'r_{name}'] = mlp.predict_proba(S64.lin_transform(X.astype(float), mu, sd))[:, 1]
    return out


def main():
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == REG_SHA
    mode, neg = sys.argv[1], int(sys.argv[2]); t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    parts, infos = [], []
    if mode == 'dev':
        fold, k0, k1 = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); g = 3 - fold
        held = set(fold_pools(fold)); cand = candidates(g, held) if neg else None
        prefs = {p: pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
                 for p in ('prefix40', 'prefix60', 'prefix80')}
        dest = OUT / f'arch_dev_neg{neg}_fold{fold}_k{k0}-{k1}.parquet'
    else:
        k0, k1 = int(sys.argv[3]), int(sys.argv[4]); held = set()
        if neg:
            c1, c2 = candidates(1, set()), candidates(2, set()); cand = pl.concat([c1, c2.join(c1, on='pair_id', how='anti')])
        ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
        dest = OUT / f'arch_eval_neg{neg}_k{k0}-{k1}.parquet'
    for k in range(k0, k1 + 1):
        if mode == 'dev':
            fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features, k); seed = 156000 + 10 * fold + k
        else:
            fr = view_frames_seeded(FINAL_CA, held, features, k); seed = 156900 + k
        add, info = (added_rows(cand, fr[1.0].height, seed, features) if neg else (None, {}))
        info = {**info, 'k': k}; infos.append(info)
        M = fit_arch(fr, add, features, k); del fr, add
        if mode == 'dev':
            for p, pref in prefs.items():
                sc = score_arch(M, pref, features)
                parts.append(pref.select('pair_id').with_columns(pl.lit(fold).alias('fold'), pl.lit(p).alias('surface'), pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        else:
            sc = score_arch(M, ev, features)
            parts.append(ev.select('pair_id').with_columns(pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        print(json.dumps({**info, 'seconds': round(time.time() - t0)}), flush=True)
    pl.concat(parts).write_parquet(dest)
    json.dump(infos, open(str(dest).replace('.parquet', '.json'), 'w'), indent=1)
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
