"""Step 151c (Claude): part B of prediction_step151.json. Multi-draw length-robust tail views on development folds 1-2.
Draw k: thinning seeds FRACTIONS[f] + 1000k, LightGBM random_state 7329 + k; fits per the step 64a recipe (held pools excluded,
fold policy ca features). Views SHORT3, ALL5 (multiclass), LIN30, LIN300 (PROD frames), DT/SP/CI one-vs-rest (PROD frames).
Scores the fold's prefix-40/60/80 pair frames. Development labels: training frames only; no arm is read here.
Usage: step151c_C_SX_views_dev FOLD K0 K1  -> experiments/tail_keys_20260914/sx_dev_fold{FOLD}_k{K0}-{K1}.parquet"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib, lightgbm as lgb
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools, extra_aggregate
from pu_pairs import step64a_tail_aggregate_views as S64
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, MODEL_PARAMS, thin, aggregate, matrix  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

OUT = CV / 'experiments' / 'tail_keys_20260914'


def view_frames_seeded(ca_path, held, features, k):
    """S64.view_frames with thinning seeds shifted by 1000k (k = 0 is the stored recipe)."""
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    ca = pl.read_parquet(ca_path)
    ca_features = [c for c in ca.columns if c.startswith('ca_')]
    combined = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left').with_columns(pl.col(ca_features).fill_null(0))
    baseline_full = pl.read_parquet(ORIGINAL / 'outputs_v2' / 'pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    full = baseline_full.join(extra_aggregate(combined, ca_features), on=['pair_id', 'table_id']).sort('pair_id')
    held = sorted(held)
    out = {1.0: full.filter(~pl.col('table_id').is_in(held))}
    for fraction, seed in S64.FRACTIONS.items():
        if seed is None: continue
        s = seed + 1000 * k
        raw_sample = thin(raw, fraction, s); added_sample = thin(combined, fraction, s)
        view = aggregate(raw_sample, names).join(tails(raw_sample, events), on=['pair_id', 'table_id']).join(labels, on='pair_id')
        view = view.join(extra_aggregate(added_sample, ca_features), on=['pair_id', 'table_id']).select(full.columns).sort('pair_id')
        out[fraction] = view.filter(~pl.col('table_id').is_in(held))
    return out


def params(k):
    return {**MODEL_PARAMS, 'random_state': MODEL_PARAMS['random_state'] + k}


def fit_views(fr, features, k):
    M = {}
    for name in ('SHORT3', 'ALL5'):
        tr = pl.concat([fr[f] for f in S64.TRAIN_SETS[name]])
        M[name] = ('mc', lgb.LGBMClassifier(**params(k)).fit(matrix(tr, features), np.array([LABELS[f] for f in tr['behavior_family']])))
    prod = pl.concat([fr[f] for f in S64.TRAIN_SETS['PROD']])
    for l2 in (30.0, 300.0):
        M[f'LIN{int(l2)}'] = ('lin', S64.fit_linear(prod, features, l2))
    for key, fam in S64.FAMS.items():
        M[key] = ('ovr', lgb.LGBMClassifier(**params(k)).fit(matrix(prod, features), (prod['behavior_family'] == fam).to_numpy().astype(int)))
    return M


def score(M, frame, features):
    X = matrix(frame, features); out = {}
    for name, (kind, m) in M.items():
        if kind == 'mc': out[f'r_{name}'] = 1 - m.predict_proba(X)[:, 0]
        elif kind == 'ovr': out[f'r_{name}'] = m.predict_proba(X)[:, 1]
        else: out[f'r_{name}'] = S64.score_linear(m, frame, features)
    out['r_FAMMAX'] = np.max([out['r_DT'], out['r_SP'], out['r_CI']], axis=0)
    return out


def main():
    fold, k0, k1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    prefs = {p: pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
             for p in ('prefix40', 'prefix60', 'prefix80')}
    parts = []
    for k in range(k0, k1 + 1):
        fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', set(fold_pools(fold)), features, k)
        M = fit_views(fr, features, k); del fr
        for p, pref in prefs.items():
            sc = score(M, pref, features)
            parts.append(pref.select('pair_id').with_columns(pl.lit(fold).alias('fold'), pl.lit(p).alias('surface'), pl.lit(k).alias('k'),
                                                             *[pl.Series(c, v) for c, v in sc.items()]))
        print(json.dumps({'fold': fold, 'k': k, 'seconds': round(time.time() - t0)}), flush=True)
    pl.concat(parts).write_parquet(OUT / f'sx_dev_fold{fold}_k{k0}-{k1}.parquet')
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
