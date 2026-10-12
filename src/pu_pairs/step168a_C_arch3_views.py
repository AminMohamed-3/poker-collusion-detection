"""Step 168a (Claude): learner-diversity round 3 tail views of outputs_step168_20260915/prediction_step168.json. For draw k, rebuild exactly
the SXN3 frames of step159a / step163a (step151c thinning seeds + 1000k; added rows at M = round(8.7 x labelled full-frame pairs) with seed
159000 + 10 fold + k on development and 159900 + k on evaluation) and fit three new view types; the ten FINAL2 types are not refitted.
  RF5     sklearn RandomForest multiclass (ALL5 frames + rows), 300 trees, min_samples_leaf 3, sqrt features, risk = 1 - P(none)
  DART5   LightGBM dart multiclass (ALL5 frames + rows), params(k) with boosting dart, drop_rate .1, risk = 1 - P(none)
  MLPS3   sklearn MLP (64,), alpha .03, binary label==1, SHORT3 frames + rows, LIN standardisation
Development: step168a_C_arch3_views.py dev FOLD K0 K1 REGSHA -> experiments/arch3_views_20260915/arch3_dev_fold{F}_k{K0}-{K1}.parquet
Evaluation:  step168a_C_arch3_views.py eval K0 K1 REGSHA     -> experiments/arch3_views_20260915/arch3_eval_k{K0}-{K1}.parquet
No labels of evaluation pairs; development labels only in training frames."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time, hashlib, warnings
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs import step64a_tail_aggregate_views as S64
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded, params
from pu_pairs import step156b_C_SXN_views as N
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, matrix  # noqa: E402

REG = CV / 'outputs_step168_20260915' / 'prediction_step168.json'
OUT = CV / 'experiments' / 'arch3_views_20260915'
RATIO3 = 8.7


def fit_arch3(fr, add, features, k):
    cols = ['pair_id', 'label', 'behavior_family'] + features

    def ts(name):
        return pl.concat([fr[f].select(cols) for f in S64.TRAIN_SETS[name]] + [add[L].select(cols) for L in N.ADD[name]], how='vertical_relaxed')
    M = {}
    all5 = ts('ALL5'); X5 = matrix(all5, features); y5 = np.array([LABELS[f] for f in all5['behavior_family']])
    M['RF5'] = ('mc', RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features='sqrt', random_state=7329 + k, n_jobs=2).fit(X5, y5))
    M['DART5'] = ('mc', lgb.LGBMClassifier(**{**params(k), 'boosting_type': 'dart', 'drop_rate': .1}).fit(X5, y5))
    del X5, all5
    s3 = ts('SHORT3'); X3 = matrix(s3, features).astype(float); mu, sd = X3.mean(0), X3.std(0) + 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=.03, early_stopping=True, max_iter=200, random_state=7329 + k).fit(
            S64.lin_transform(X3, mu, sd), (s3['label'] == 1).to_numpy().astype(int))
    M['MLPS3'] = ('mlp', (mlp, mu, sd))
    return M


def score_arch3(M, frame, features):
    X = matrix(frame, features); out = {}
    for name, (kind, m) in M.items():
        if kind == 'mc': out[f'r_{name}'] = 1 - m.predict_proba(X)[:, 0]
        else:
            mlp, mu, sd = m; out[f'r_{name}'] = mlp.predict_proba(S64.lin_transform(X.astype(float), mu, sd))[:, 1]
    return out


def main():
    reg_sha = sys.argv[-1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'pass the registration sha as the last argument'
    N.RATIO = RATIO3
    mode = sys.argv[1]; t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    parts, infos = [], []
    if mode == 'dev':
        fold, k0, k1 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); g = 3 - fold
        held = set(fold_pools(fold)); cand = N.candidates(g, held)
        prefs = {p: pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
                 for p in ('prefix40', 'prefix60', 'prefix80')}
        dest = OUT / f'arch3_dev_fold{fold}_k{k0}-{k1}.parquet'
    else:
        k0, k1 = int(sys.argv[2]), int(sys.argv[3]); held = set()
        c1, c2 = N.candidates(1, set()), N.candidates(2, set())
        cand = pl.concat([c1, c2.join(c1, on='pair_id', how='anti')])
        ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
        dest = OUT / f'arch3_eval_k{k0}-{k1}.parquet'
    assert not dest.exists()
    for k in range(k0, k1 + 1):
        if mode == 'dev':
            fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features, k); seed = 159000 + 10 * fold + k
        else:
            fr = view_frames_seeded(FINAL_CA, held, features, k); seed = 159900 + k
        add, info = N.added_rows(cand, fr[1.0].height, seed, features)
        info.update({'k': k, 'n_lab_full_frame': fr[1.0].height, 'ratio': RATIO3}); infos.append(info)
        M = fit_arch3(fr, add, features, k); del fr, add
        if mode == 'dev':
            for p, pref in prefs.items():
                sc = score_arch3(M, pref, features)
                parts.append(pref.select('pair_id').with_columns(pl.lit(fold).alias('fold'), pl.lit(p).alias('surface'), pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        else:
            sc = score_arch3(M, ev, features)
            parts.append(ev.select('pair_id').with_columns(pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        del M
        print(json.dumps({**info, 'seconds': round(time.time() - t0)}), flush=True)
    pl.concat(parts).write_parquet(dest)
    json.dump(infos, open(str(dest).replace('.parquet', '.json'), 'w'), indent=1)
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
