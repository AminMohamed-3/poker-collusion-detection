"""Step 64a: new AGGREGATE-level views aimed at the recall tail of the prefix testbed (folds 1-2, 60 % prefix).
Step 61-63 closed most of the block-ORDER headroom (candidate M2). What remains on the pair side is the recall tail:
at evaluation-like length ~11 % of the held positives (20 of 186) sit below risk .5 and ~half of those below .1, and every
existing view (FULL / RAW / CA / BASE / OPP / V2 / MEAN / TAILS) scores them low. They are SHORT pairs (11-82 shared hands
at the prefix). Hypotheses tested here, all as extra views scored on every testbed pair:
  SHORT3  production recipe trained on thinned views only (40 % / 25 % / 15 % of each training pair's hands), i.e. a model
          whose training length distribution matches the evaluation length, instead of full + 70 % + 40 %;
  ALL5    full + 70 % + 40 % + 25 % + 15 % (length-augmented);
  LIN     L2 logistic regression (positive vs negative pair) on the standardised 897 aggregates, trained on the production
          frame: a smooth model may extrapolate to short pairs where the trees fall back to the majority leaf;
  DT/SP/CI one-vs-rest LightGBM per family (family vs everything else) on the production frame, FAMMAX = their max: a
          family-specific detector may fire on a weak positive that the shared 4-class model dilutes.
Per fold the held pools are excluded from training (train_frame protocol of step 53). Development labels: training and
validation only. Outputs: experiments/viewdis_20260911/tail_views_prefix.parquet (+ .json with sizes/timings)."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools, extra_aggregate
from pu_pairs.step60_multiview_blend import logreg, predict
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, MODEL_PARAMS, thin, aggregate, matrix  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

VD = CV / 'experiments' / 'viewdis_20260911'
FRACTIONS = {1.0: None, 0.7: 7307, 0.4: 7304, 0.25: 7325, 0.15: 7315}
TRAIN_SETS = {'SHORT3': [0.4, 0.25, 0.15], 'ALL5': [1.0, 0.7, 0.4, 0.25, 0.15], 'PROD': [1.0, 0.7, 0.4]}
FAMS = {'DT': 'directed_transfer', 'SP': 'soft_play', 'CI': 'coordinated_isolation'}


def log(*a):
    print(*a, flush=True)


def view_frames(ca_path, held, features):
    """Per-fraction training frames (labelled pairs outside `held`): {fraction: frame}; step 53 recipe with extra fractions."""
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    ca = pl.read_parquet(ca_path)
    ca_features = [c for c in ca.columns if c.startswith('ca_')]
    combined = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left').with_columns(pl.col(ca_features).fill_null(0))
    assert combined.height == raw.height
    baseline_full = pl.read_parquet(ORIGINAL / 'outputs_v2' / 'pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    full = baseline_full.join(extra_aggregate(combined, ca_features), on=['pair_id', 'table_id']).sort('pair_id')
    held = sorted(held)
    out = {1.0: full.filter(~pl.col('table_id').is_in(held))}
    for fraction, seed in FRACTIONS.items():
        if seed is None: continue
        raw_sample = thin(raw, fraction, seed); added_sample = thin(combined, fraction, seed)
        view = aggregate(raw_sample, names).join(tails(raw_sample, events), on=['pair_id', 'table_id']).join(labels, on='pair_id')
        view = view.join(extra_aggregate(added_sample, ca_features), on=['pair_id', 'table_id']).select(full.columns).sort('pair_id')
        out[fraction] = view.filter(~pl.col('table_id').is_in(held))
    missing = [f for f in features if f not in full.columns]
    assert not missing, missing[:5]
    return out


def fit_multiclass(train, features):
    y = np.array([LABELS[f] for f in train['behavior_family']])
    return lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(train, features), y)


def fit_ovr(train, features, family):
    y = (train['behavior_family'] == family).to_numpy().astype(int)
    return lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(train, features), y)


def lin_transform(X, mu, sd):
    return np.clip((X - mu) / sd, -5, 5)


def fit_linear(train, features, l2):
    X = matrix(train, features).astype(float); mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = lin_transform(X, mu, sd); y = (train['label'] == 1).to_numpy().astype(float)
    w, b, m2, s2 = logreg(Z, y, l2=l2, iters=60)
    return (w, b, m2, s2, mu, sd)


def score_linear(model, frame, features):
    w, b, m2, s2, mu, sd = model
    Z = lin_transform(matrix(frame, features).astype(float), mu, sd)
    z = predict((w, b, m2, s2), Z); return 1 / (1 + np.exp(-z))


def main():
    t0 = time.time()
    bundle = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib'); features = list(bundle['features'])
    res = {'fractions': list(FRACTIONS), 'train_sets': TRAIN_SETS, 'folds': {}}
    parts = []
    for fold in (1, 2):
        held = set(fold_pools(fold))
        frames = view_frames(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features)
        info = {'rows_per_fraction': {str(k): v.height for k, v in frames.items()}, 'median_shared_hands_pos': {str(k): float(v.filter(pl.col('label') == 1)['shared_hands'].median()) for k, v in frames.items()}}
        log(f'fold{fold} frames', json.dumps(info), round(time.time() - t0))
        pref = pl.concat([pl.read_parquet(p, columns=list(dict.fromkeys(['pair_id'] + features))) for p in sorted((EXP / f'fold{fold}-prefix60').glob('batch_*-pairs.parquet'))])
        out = pref.select('pair_id').with_columns(pl.lit(fold).alias('fold_new'))
        for name, fracs in TRAIN_SETS.items():
            train = pl.concat([frames[f] for f in fracs]); m = fit_multiclass(train, features)
            p = m.predict_proba(matrix(pref, features)); out = out.with_columns(pl.Series(f'r_{name}', 1 - p[:, 0]))
            log(f'fold{fold} {name} fitted on {train.height} rows', round(time.time() - t0))
        prod = pl.concat([frames[f] for f in TRAIN_SETS['PROD']])
        for l2 in (30.0, 300.0):
            lm = fit_linear(prod, features, l2); out = out.with_columns(pl.Series(f'r_LIN{int(l2)}', score_linear(lm, pref, features)))
            log(f'fold{fold} LIN l2={l2} fitted', round(time.time() - t0))
        for key, fam in FAMS.items():
            m = fit_ovr(prod, features, fam); out = out.with_columns(pl.Series(f'r_{key}', m.predict_proba(matrix(pref, features))[:, 1]))
        out = out.with_columns(pl.max_horizontal('r_DT', 'r_SP', 'r_CI').alias('r_FAMMAX'))
        log(f'fold{fold} OVR fitted', round(time.time() - t0))
        res['folds'][fold] = info; parts.append(out)
        del frames, prod, pref
    new = pl.concat(parts)
    new.write_parquet(VD / 'tail_views_prefix.parquet')
    res['rows'] = new.height; res['unique_pairs'] = new['pair_id'].n_unique(); res['seconds'] = round(time.time() - t0)
    write_json(VD / 'tail_views_prefix.json', res)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
