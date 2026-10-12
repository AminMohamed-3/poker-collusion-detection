"""Step 164a (Claude): length-weighted (LW) tail views of outputs_step164_20260915/prediction_step164.json (idea 3 of the route list).
For draw k, rebuild exactly the SXN3 frames of step159a (step151c thinning seeds + 1000k; added rows at M = round(8.7 x labelled full-frame
pairs), seed 159000 + 10 fold + k on development and 159900 + k on evaluation; LightGBM params(k)) and refit four SXN3 types with per-row
training weights; SHORT3 and every other type are not refitted.
  weight of a row in training set V = clip(p_eval(bin) / p_V(bin), 0, 4), then scaled to mean 1 over V, where bin = shared_hands binned at
  EDGES, p_eval = the shared_hands histogram of the 112,540 evaluation pairs (a label-free covariate) and p_V = the histogram of V in this draw.
  ALL5    LightGBM multiclass on ALL5 frames + rows, sample_weight
  LIN30 / LIN300   L2 logistic (label==1) on PROD frames + rows, S64 standardisation, weighted Newton loss (wlogreg)
  DT / SP / CI     LightGBM one-vs-rest on PROD frames + rows, sample_weight; FAMMAX = their max
Development: step164a_C_lw_views.py dev FOLD K0 K1 REGSHA -> experiments/length_weights_20260915/lw_dev_fold{F}_k{K0}-{K1}.parquet
Evaluation:  step164a_C_lw_views.py eval K0 K1 REGSHA     -> experiments/length_weights_20260915/lw_eval_k{K0}-{K1}.parquet
Unit gate:   step164a_C_lw_views.py unit                  -> weights all ones, fold 1, k = 0; must reproduce the stored SXN3 draw and logreg
No labels of evaluation pairs; development labels only in training frames."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib, lightgbm as lgb
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs import step64a_tail_aggregate_views as S64
from pu_pairs.step60_multiview_blend import logreg, predict
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded, params
from pu_pairs import step156b_C_SXN_views as N
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, matrix  # noqa: E402

REG = CV / 'outputs_step164_20260915' / 'prediction_step164.json'
OUT = CV / 'experiments' / 'length_weights_20260915'
RATIO3 = 8.7
EDGES = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 170, 200, 250, 10 ** 6]
WMAX = 4.0


def eval_hist():
    ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=['pair_id', 'shared_hands'])
    assert ev.height == 112540 and ev['pair_id'].n_unique() == 112540
    h, _ = np.histogram(ev['shared_hands'].to_numpy().astype(float), bins=EDGES); return h / h.sum()


def length_weights(sh, pe, ones=False):
    sh = np.asarray(sh, dtype=float)
    if ones: return np.ones(len(sh))
    h, _ = np.histogram(sh, bins=EDGES); pv = h / h.sum(); b = np.digitize(sh, EDGES[1:-1])
    w = np.clip(np.where(pv[b] > 0, pe[b] / np.maximum(pv[b], 1e-12), 0.0), 0.0, WMAX)
    return w / w.mean()


def wlogreg(X, y, sw, l2=1.0, iters=200):
    """step60 logreg with per-row loss weights sw (sw = 1 gives the same Newton steps)."""
    mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd
    n, k = Z.shape; w = np.zeros(k); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b))); r = sw * (p - y); g = Z.T @ r + l2 * w; gb = r.sum()
        W = sw * p * (1 - p); H = (Z * W[:, None]).T @ Z + l2 * np.eye(k); Hb = W.sum()
        step = np.linalg.solve(H, g); w -= step; b -= gb / max(Hb, 1e-9)
        if np.abs(step).max() < 1e-8: break
    return w, b, mu, sd


def fit_linear_w(train, features, l2, sw):
    X = matrix(train, features).astype(float); mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = S64.lin_transform(X, mu, sd); y = (train['label'] == 1).to_numpy().astype(float)
    w, b, m2, s2 = wlogreg(Z, y, sw, l2=l2, iters=60)
    return (w, b, m2, s2, mu, sd)


def fit_lw(fr, add, features, k, pe, ones=False):
    cols = ['pair_id', 'label', 'behavior_family', 'shared_hands'] + [f for f in features if f != 'shared_hands']

    def ts(name):
        return pl.concat([fr[f].select(cols) for f in S64.TRAIN_SETS[name]] + [add[L].select(cols) for L in N.ADD[name]], how='vertical_relaxed')
    M, winfo = {}, {}
    all5 = ts('ALL5'); w5 = length_weights(all5['shared_hands'], pe, ones)
    M['ALL5'] = ('mc', lgb.LGBMClassifier(**params(k)).fit(matrix(all5, features), np.array([LABELS[f] for f in all5['behavior_family']]), sample_weight=w5))
    winfo['ALL5'] = {'n': len(w5), 'ess_ratio': float(w5.sum() ** 2 / (w5 ** 2).sum() / len(w5)), 'zero_share': float((w5 == 0).mean())}
    del all5
    prod = ts('PROD'); wp = length_weights(prod['shared_hands'], pe, ones)
    winfo['PROD'] = {'n': len(wp), 'ess_ratio': float(wp.sum() ** 2 / (wp ** 2).sum() / len(wp)), 'zero_share': float((wp == 0).mean())}
    for l2 in (30.0, 300.0):
        M[f'LIN{int(l2)}'] = ('lin', fit_linear_w(prod, features, l2, wp))
    for key, fam in S64.FAMS.items():
        M[key] = ('ovr', lgb.LGBMClassifier(**params(k)).fit(matrix(prod, features), (prod['behavior_family'] == fam).to_numpy().astype(int), sample_weight=wp))
    return M, winfo


def score_lw(M, frame, features):
    X = matrix(frame, features); out = {}
    for name, (kind, m) in M.items():
        if kind == 'mc': out[f'r_{name}'] = 1 - m.predict_proba(X)[:, 0]
        elif kind == 'ovr': out[f'r_{name}'] = m.predict_proba(X)[:, 1]
        else: out[f'r_{name}'] = S64.score_linear(m, frame, features)
    out['r_FAMMAX'] = np.max([out['r_DT'], out['r_SP'], out['r_CI']], axis=0)
    return out


def unit():
    """Weights all ones must reproduce the stored step159a SXN3 fold-1 draw k = 0 (scores only; no surface labels read)."""
    t0 = time.time(); N.RATIO = RATIO3; fold, k = 1, 0
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    held = set(fold_pools(fold)); cand = N.candidates(3 - fold, held)
    fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features, k)
    add, info = N.added_rows(cand, fr[1.0].height, 159000 + 10 * fold + k, features)
    pe = eval_hist()
    M, winfo = fit_lw(fr, add, features, k, pe, ones=True)
    prod = pl.concat([fr[f].select(['label'] + features) for f in S64.TRAIN_SETS['PROD']] + [add[L].select(['label'] + features) for L in N.ADD['PROD']], how='vertical_relaxed')
    X = matrix(prod, features).astype(float); mu = X.mean(0); sd = X.std(0) + 1e-9; Z = S64.lin_transform(X, mu, sd); y = (prod['label'] == 1).to_numpy().astype(float)
    a, b = logreg(Z, y, l2=30.0, iters=60), wlogreg(Z, y, np.ones(len(y)), l2=30.0, iters=60)
    lin_same = bool(all(np.array_equal(u, v) for u, v in zip(a, b)))
    lin_maxdiff = float(max(np.abs(np.asarray(u) - np.asarray(v)).max() for u, v in zip(a, b)))
    w_real = length_weights(prod['shared_hands'], pe)
    ref = pl.read_parquet(CV / 'experiments' / 'tail_keys_20260915_neg3' / 'sxn3_dev_fold1_k0-7.parquet').filter((pl.col('k') == 0) & (pl.col('surface') == 'prefix60'))
    pref = pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-prefix60').glob('batch_*-pairs.parquet'))])
    sc = score_lw(M, pref, features)
    got = pref.select('pair_id').with_columns(*[pl.Series(c, v) for c, v in sc.items()]).join(ref.select('pair_id', 'r_ALL5', 'r_LIN30', 'r_LIN300', 'r_FAMMAX'), on='pair_id', suffix='_ref', validate='1:1')
    diffs = {c: float((got[c] - got[f'{c}_ref']).abs().max()) for c in ('r_ALL5', 'r_LIN30', 'r_LIN300', 'r_FAMMAX')}
    res = {'rows': got.height, 'max_abs_diff_vs_stored_sxn3': diffs, 'wlogreg_ones_equals_logreg': lin_same, 'wlogreg_ones_maxdiff': lin_maxdiff, 'winfo_ones': winfo,
           'real_weights_PROD': {'ess_ratio': float(w_real.sum() ** 2 / (w_real ** 2).sum() / len(w_real)), 'zero_share': float((w_real == 0).mean()),
                                 'q10_50_90': [float(np.percentile(w_real, p)) for p in (10, 50, 90)]}, 'seconds': round(time.time() - t0)}
    print(json.dumps(res, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / 'lw_unit_gate.json', 'w'), indent=1)


def main():
    if sys.argv[1] == 'unit':
        return unit()
    reg_sha = sys.argv[-1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'pass the registration sha as the last argument'
    N.RATIO = RATIO3  # added_rows reads the module-level ratio (as step159a)
    mode = sys.argv[1]; t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    pe = eval_hist()
    parts, infos = [], []
    if mode == 'dev':
        fold, k0, k1 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); g = 3 - fold
        held = set(fold_pools(fold)); cand = N.candidates(g, held)
        prefs = {p: pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
                 for p in ('prefix40', 'prefix60', 'prefix80')}
        dest = OUT / f'lw_dev_fold{fold}_k{k0}-{k1}.parquet'
    else:
        k0, k1 = int(sys.argv[2]), int(sys.argv[3]); held = set()
        c1, c2 = N.candidates(1, set()), N.candidates(2, set())
        cand = pl.concat([c1, c2.join(c1, on='pair_id', how='anti')])
        ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
        dest = OUT / f'lw_eval_k{k0}-{k1}.parquet'
    assert not dest.exists()
    for k in range(k0, k1 + 1):
        if mode == 'dev':
            fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features, k); seed = 159000 + 10 * fold + k
        else:
            fr = view_frames_seeded(FINAL_CA, held, features, k); seed = 159900 + k
        add, info = N.added_rows(cand, fr[1.0].height, seed, features)
        M, winfo = fit_lw(fr, add, features, k, pe); del fr, add
        info.update({'k': k, 'ratio': RATIO3, 'weights': winfo}); infos.append(info)
        if mode == 'dev':
            for p, pref in prefs.items():
                sc = score_lw(M, pref, features)
                parts.append(pref.select('pair_id').with_columns(pl.lit(fold).alias('fold'), pl.lit(p).alias('surface'), pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        else:
            sc = score_lw(M, ev, features)
            parts.append(ev.select('pair_id').with_columns(pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        del M
        print(json.dumps({**info, 'seconds': round(time.time() - t0)}), flush=True)
    pl.concat(parts).write_parquet(dest)
    json.dump(infos, open(str(dest).replace('.parquet', '.json'), 'w'), indent=1)
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
