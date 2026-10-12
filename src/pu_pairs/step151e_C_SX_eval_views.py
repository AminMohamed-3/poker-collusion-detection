"""Step 151e (Claude): production of part B only if the registered rule passed (prediction_step151.json). All-labelled fits of draws
k = K0..K1 (thinning seeds + 1000k, random_state 7329 + k; step 151a recipe on FINAL_CA), scored on the evaluation frame.
Label-free on evaluation. k = 0 must equal eval_views_S5.parquet exactly. Usage: step151e_C_SX_eval_views K0 K1."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import polars as pl, joblib
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded, fit_views, score, OUT


def main():
    k0, k1 = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
    parts = []
    for k in range(k0, k1 + 1):
        M = fit_views(view_frames_seeded(FINAL_CA, set(), features, k), features, k)
        sc = score(M, ev, features)
        parts.append(ev.select('pair_id').with_columns(pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        print(json.dumps({'k': k, 'seconds': round(time.time() - t0)}), flush=True)
    pl.concat(parts).write_parquet(OUT / f'sx_eval_k{k0}-{k1}.parquet')
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
