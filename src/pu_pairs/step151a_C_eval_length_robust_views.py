"""Step 151a (Claude, 2026-09-14): all-labelled fits of the length-robust tail views LIN30, LIN300 and the one-vs-rest DT/SP/CI
LightGBMs (FAMMAX = their max), with the step 64a recipe on the step 64e training frame (all 1,860 labelled pairs, FINAL_CA,
PROD fractions full/70/40), scored on the evaluation frame. Label-free on evaluation; no submission is written here.
Lineage check first: SHORT3 and ALL5 refitted on the same frames must reproduce the saved step-64e evaluation scores.
Output: experiments/tail_keys_20260914/{eval_views_S5.parquet, eval_views_S5.json}."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, sha
import numpy as np
import polars as pl
import joblib
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs.step64a_tail_aggregate_views import view_frames, fit_multiclass, fit_linear, score_linear, fit_ovr, TRAIN_SETS, FAMS, VD
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import matrix  # noqa: E402

OUT = CV / 'experiments' / 'tail_keys_20260914'


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    frames = view_frames(FINAL_CA, set(), features)
    log('frames', {str(k): v.height for k, v in frames.items()}, round(time.time() - t0))
    ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
    X = matrix(ev, features)
    out = ev.select('pair_id')
    saved = pl.read_parquet(VD / 'tail_views_eval_frame.parquet')
    lineage = {}
    for name in ('SHORT3', 'ALL5'):
        m = fit_multiclass(pl.concat([frames[f] for f in TRAIN_SETS[name]]), features)
        out = out.with_columns(pl.Series(f'r_{name}', 1 - m.predict_proba(X)[:, 0]))
        j = out.select('pair_id', f'r_{name}').join(saved.select('pair_id', f'r_{name}'), on='pair_id', suffix='_saved', validate='1:1')
        lineage[name] = float((j[f'r_{name}'] - j[f'r_{name}_saved']).abs().max())
        log(name, 'refit max abs diff vs saved step-64e', lineage[name], round(time.time() - t0))
    assert max(lineage.values()) <= 1e-9, lineage
    prod = pl.concat([frames[f] for f in TRAIN_SETS['PROD']])
    models = {}
    for l2 in (30.0, 300.0):
        lm = fit_linear(prod, features, l2); models[f'LIN{int(l2)}'] = lm
        out = out.with_columns(pl.Series(f'r_LIN{int(l2)}', score_linear(lm, ev, features)))
        log(f'LIN{int(l2)} fitted and scored', round(time.time() - t0))
    for key, fam in FAMS.items():
        m = fit_ovr(prod, features, fam); models[key] = m
        out = out.with_columns(pl.Series(f'r_{key}', m.predict_proba(X)[:, 1]))
    out = out.with_columns(pl.max_horizontal('r_DT', 'r_SP', 'r_CI').alias('r_FAMMAX'))
    log('OVR fitted and scored', round(time.time() - t0))
    cols = ['r_SHORT3', 'r_ALL5', 'r_LIN30', 'r_LIN300', 'r_DT', 'r_SP', 'r_CI', 'r_FAMMAX']
    assert all(out[c].null_count() == 0 and bool(out[c].is_finite().all()) for c in cols)
    out.write_parquet(OUT / 'eval_views_S5.parquet'); joblib.dump(models, OUT / 'eval_views_S5_models.joblib')
    res = {'rows': out.height, 'prod_rows': prod.height, 'lineage_refit_max_abs_diff': lineage,
           'medians': {c: round(float(out[c].median()), 5) for c in cols},
           'output_sha256': sha(OUT / 'eval_views_S5.parquet'), 'models_sha256': sha(OUT / 'eval_views_S5_models.joblib'),
           'code_sha256': sha(os.path.abspath(__file__)), 'seconds': round(time.time() - t0)}
    write_json(OUT / 'eval_views_S5.json', res); log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
