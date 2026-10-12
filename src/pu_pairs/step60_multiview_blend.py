"""Step 60: MULTI-VIEW decomposition of the pair model and a cross-fold LEARNED blend on the prefix testbed.
Candidate S (step 59, measured +.0143 public) showed that the RAW view (765 non-ca features) separates length-induced
false positives from positives where the frozen model does not. This step splits the 897 features further and refits the
production recipe on each subset per fold (held pools excluded), scores the 60 %-prefix frames of folds 1-2, and asks:
(1) which view is the most length-robust (AUC positives vs certified false positives within the prefix block);
(2) does a blend learned CROSS-FOLD (logistic regression on the view logits + log shared hands, trained on one fold's
    non-shared tables, tested on the other's) beat the fixed logit(FULL) + .5 logit(RAW) of candidate S;
(3) the same below the block (prefix risk >= .1), where the 3-6 % recall deficit lives.
Views: FULL (897, frozen prefix risk), RAW (765 non-ca), CA (132 ca + shared_hands) reused from step 58; new: BASE (non-ca,
non-v2, non-opp aggregates), OPP (opp_*), V2 (v2_* event tails), MEAN (all *_mean / *_fraction), TAILS (all *_std/_max/_q95).
Also fits every new view on all labelled pairs (final policy ca features) and scores the evaluation frame, so a candidate
can be built from the same views without refitting. Development labels: validation + blend fitting (cross-fold) only.
Outputs: experiments/viewdis_20260911/{multiview_prefix.parquet, multiview_eval.parquet, multiview_models_full.joblib, multiview_blend.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs.step53_view_disagreement import train_frame, FINAL_CA
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, MODEL_PARAMS, matrix  # noqa: E402

VD = CV / 'experiments' / 'viewdis_20260911'
NEW_VIEWS = ['BASE', 'OPP', 'V2', 'MEAN', 'TAILS']


def log(*a):
    print(*a, flush=True)


def view_sets(features):
    f = list(features)
    base = [c for c in f if not c.startswith(('ca_', 'v2_', 'opp_'))]
    return {'BASE': base,
            'OPP': [c for c in f if c.startswith('opp_')] + ['shared_hands'],
            'V2': [c for c in f if c.startswith('v2_')] + ['shared_hands'],
            'MEAN': [c for c in f if c.endswith(('_mean', '_fraction'))] + ['shared_hands'],
            'TAILS': [c for c in f if c.endswith(('_std', '_max', '_q95'))] + ['shared_hands']}


def fit(train, features):
    y = np.array([LABELS[f] for f in train['behavior_family']])
    return {name: lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(train, cols), y) for name, cols in view_sets(features).items()}


def score(models, frame, features):
    out = frame.select('pair_id')
    for name, cols in view_sets(features).items():
        p = models[name].predict_proba(matrix(frame, cols)); out = out.with_columns(pl.Series(f'r_{name}', 1 - p[:, 0]))
    return out


def lg(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def ap(y, s):
    o = np.argsort(-s, kind='stable'); y = y[o]; tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / max(1, y.sum()))


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def logreg(X, y, l2=1.0, iters=200):
    """L2-regularised logistic regression by Newton steps on standardised inputs; returns (w, b, mu, sd)."""
    mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd
    n, k = Z.shape; w = np.zeros(k); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = Z.T @ (p - y) + l2 * w; gb = (p - y).sum()
        W = p * (1 - p); H = (Z * W[:, None]).T @ Z + l2 * np.eye(k); Hb = W.sum()
        step = np.linalg.solve(H, g); w -= step; b -= gb / max(Hb, 1e-9)
        if np.abs(step).max() < 1e-8: break
    return w, b, mu, sd


def predict(model, X):
    w, b, mu, sd = model; return ((X - mu) / sd) @ w + b


def classes(d):
    return d.with_columns(pl.when(pl.col('label') == 1).then(pl.lit('pos')).when(pl.col('label') == 0).then(pl.lit('neg'))
                          .when((pl.col('risk_full') >= .9) & ~pl.col('R1_full').fill_null(False)).then(pl.lit('undisclosed'))
                          .when(pl.col('risk_full') >= .9).then(pl.lit('undisclosed_R1'))
                          .when(pl.col('risk_full') >= .5).then(pl.lit('ambiguous')).when(pl.col('risk_full') >= .1).then(pl.lit('mid')).otherwise(pl.lit('FP')).alias('cls'))


def main():
    t0 = time.time()
    bundle = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib'); features = list(bundle['features'])
    res = {'n_features': {k: len(v) for k, v in view_sets(features).items()}}
    log(res['n_features'])
    parts = []
    for fold in (1, 2):
        train = train_frame(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', set(fold_pools(fold)), features)
        models = fit(train, features); log(f'fold{fold} views fitted', round(time.time() - t0))
        pref = pl.concat([pl.read_parquet(p, columns=list(dict.fromkeys(['pair_id'] + features))) for p in sorted((EXP / f'fold{fold}-prefix60').glob('batch_*-pairs.parquet'))])
        parts.append(score(models, pref, features).with_columns(pl.lit(fold).alias('fold_new')))
        log(f'fold{fold} prefix scored', pref.height, round(time.time() - t0))
        del train, models, pref
    new = pl.concat(parts).unique(subset=['pair_id'], keep='first', maintain_order=True)
    base = pl.read_parquet(VD / 'prefix_testbed_views.parquet').drop('cls')
    vd = pl.read_parquet(VD / 'view_dev.parquet').select('pair_id', pl.col('R1').alias('R1_full'))
    d = classes(base.join(new.drop('fold_new'), on='pair_id', how='left').join(vd, on='pair_id', how='left'))
    d.write_parquet(VD / 'multiview_prefix.parquet')
    views = ['FULL', 'RAW', 'CA'] + NEW_VIEWS
    # (1) per-view separation within the prefix block
    blk = d.filter(pl.col('risk_prefix') >= .5)
    sep = {}
    for v in views:
        col = 'risk_prefix' if v == 'FULL' else f'r_{v}'
        g = {c: blk.filter(pl.col('cls') == c)[col].to_numpy() for c in ('pos', 'undisclosed', 'undisclosed_R1', 'ambiguous', 'mid', 'FP')}
        sep[v] = {'median': {c: round(float(np.median(x)), 3) for c, x in g.items() if len(x)},
                  'AUC_pos_vs_FP': round(auc(g['pos'], g['FP']), 4), 'AUC_undisclosed_vs_FP': round(auc(g['undisclosed'], g['FP']), 4),
                  'AUC_pos_vs_mid': round(auc(g['pos'], g['mid']), 4), 'share_lt_.1': {c: round(float((x < .1).mean()), 3) for c, x in g.items() if len(x)}}
    res['block_separation'] = sep
    # (2) blends: fixed and cross-fold learned
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    defs = {'A': (['pos', 'undisclosed'], ['FP', 'undisclosed_R1']), 'C': (['pos'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg'])}
    feature_sets = {'FULL+RAW': ['FULL', 'RAW'], 'FULL+RAW+CA': ['FULL', 'RAW', 'CA'], 'all views': views, 'all views + log sh': views + ['sh'],
                    'FULL+MEAN': ['FULL', 'MEAN'], 'FULL+BASE+V2+OPP+CA': ['FULL', 'BASE', 'V2', 'OPP', 'CA']}
    def X_of(c, fs):
        cols = []
        for v in fs:
            if v == 'sh': cols.append(np.log(c['shared_hands'].to_numpy().astype(float)))
            else: cols.append(lg(c['risk_prefix'].to_numpy() if v == 'FULL' else c[f'r_{v}'].to_numpy()))
        return np.column_stack(cols)
    out = {}
    for floor in (.5, .1):
        c0 = d.filter(pl.col('risk_prefix') >= floor)
        for dn, (posset, negset) in defs.items():
            c = c0.filter(pl.col('cls').is_in(posset + negset))
            y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy()
            risk = c['risk_prefix'].to_numpy(); raw = c['r_RAW'].to_numpy()
            r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'frozen': round(ap(y, risk), 4), 'fixed w=.5': round(ap(y, lg(risk) + .5 * lg(raw)), 4)}
            for v in NEW_VIEWS:
                r[f'fixed FULL+.5 {v}'] = round(ap(y, lg(risk) + .5 * lg(c[f'r_{v}'].to_numpy())), 4)
            in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
            for fsn, fs in feature_sets.items():
                X = X_of(c, fs)
                for l2 in (1.0, 10.0):
                    pred = np.full(len(y), np.nan)
                    m1 = logreg(X[in1], y[in1], l2); pred[in2 | insh] = predict(m1, X[in2 | insh])
                    m2 = logreg(X[in2], y[in2], l2); pred[in1] = predict(m2, X[in1])
                    key = f'learned {fsn} l2={l2}'
                    r[key] = {'pooled_AP': round(ap(y, pred), 4), 'AP_test_fold2+shared': round(ap(y[in2 | insh], pred[in2 | insh]), 4), 'AP_test_fold1': round(ap(y[in1], pred[in1]), 4),
                              'frozen_same_rows': [round(ap(y[in2 | insh], risk[in2 | insh]), 4), round(ap(y[in1], risk[in1]), 4)],
                              'fixed_same_rows': [round(ap(y[in2 | insh], (lg(risk) + .5 * lg(raw))[in2 | insh]), 4), round(ap(y[in1], (lg(risk) + .5 * lg(raw))[in1]), 4)],
                              'weights_fold1_model': {v: round(float(w), 3) for v, w in zip(fs, m1[0])}, 'weights_fold2_model': {v: round(float(w), 3) for v, w in zip(fs, m2[0])}}
            out[f'floor {floor} def {dn}'] = r
    res['blends'] = out
    # (3) all-labelled fits for the evaluation frame
    train = train_frame(FINAL_CA, set(), features); models = fit(train, features); log('full train fitted', train.height, round(time.time() - t0))
    ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id', 'table_id', 'shared_hands'] + features)))
    sc = score(models, ev, features)
    ve = pl.read_parquet(VD / 'view_eval.parquet').select('pair_id', 'risk', 'r_FULL', 'r_RAW', 'r_CA', 'R1')
    evd = ev.select('pair_id', 'table_id', 'shared_hands').join(sc, on='pair_id').join(ve, on='pair_id', how='left')
    evd.write_parquet(VD / 'multiview_eval.parquet'); joblib.dump(models, VD / 'multiview_models_full.joblib')
    eb = evd.filter(pl.col('risk') >= .5)
    res['eval_block'] = {v: {'median': round(float(eb[f'r_{v}'].median()), 3), 'share_lt_.1': round(float((eb[f'r_{v}'] < .1).mean()), 3),
                             'median_R1': round(float(eb.filter(pl.col('R1'))[f'r_{v}'].median()), 3), 'median_notR1_ge.99': round(float(eb.filter(~pl.col('R1') & (pl.col('risk') >= .99))[f'r_{v}'].median()), 3)} for v in ['RAW', 'CA'] + NEW_VIEWS}
    res['seconds'] = round(time.time() - t0)
    write_json(VD / 'multiview_blend.json', res)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
