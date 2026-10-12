"""Step 159a (Claude): part N of outputs_step159_20260915/prediction_step159.json (sha a77dcb64). SXN3 draws = step156b SXN draw k (same
thinning seeds, random_state, length mapping, candidate pools, all labelled candidates) with M = round(8.7 x labelled full-frame pairs)
unlabelled pairs (fold v13 risk < .5) as 'none', sampled with seed 159000 + 10 fold + k (development) or 159900 + k (evaluation).
Development: step159a_C_SXN3_views.py dev FOLD K0 K1 -> experiments/tail_keys_20260915_neg3/sxn3_dev_fold{F}_k{K0}-{K1}.parquet
Evaluation:  step159a_C_SXN3_views.py eval K0 K1      -> experiments/tail_keys_20260915_neg3/sxn3_eval_k{K0}-{K1}.parquet
No labels of evaluation pairs; development labels only in training frames."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded, score
from pu_pairs import step156b_C_SXN_views as N

REG = CV / 'outputs_step159_20260915' / 'prediction_step159.json'
REG_SHA = 'a77dcb643b129e50b956d4d95a889eb1c5c7b4080ab6dc0bae38011ee53a2252'
OUT = CV / 'experiments' / 'tail_keys_20260915_neg3'
RATIO3 = 8.7


def main():
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == REG_SHA
    N.RATIO = RATIO3  # added_rows reads the module-level ratio
    mode = sys.argv[1]; t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    parts, infos = [], []
    if mode == 'dev':
        fold, k0, k1 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); g = 3 - fold
        held = set(fold_pools(fold)); cand = N.candidates(g, held)
        prefs = {p: pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
                 for p in ('prefix40', 'prefix60', 'prefix80')}
        dest = OUT / f'sxn3_dev_fold{fold}_k{k0}-{k1}.parquet'
    else:
        k0, k1 = int(sys.argv[2]), int(sys.argv[3]); held = set()
        c1, c2 = N.candidates(1, set()), N.candidates(2, set())
        cand = pl.concat([c1, c2.join(c1, on='pair_id', how='anti')])
        ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
        dest = OUT / f'sxn3_eval_k{k0}-{k1}.parquet'
    assert not dest.exists()
    for k in range(k0, k1 + 1):
        if mode == 'dev':
            fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features, k); seed = 159000 + 10 * fold + k
        else:
            fr = view_frames_seeded(FINAL_CA, held, features, k); seed = 159900 + k
        add, info = N.added_rows(cand, fr[1.0].height, seed, features)
        info.update({'k': k, 'n_lab_full_frame': fr[1.0].height, 'ratio': RATIO3}); infos.append(info)
        M = N.fit_views_neg(fr, add, features, k); del fr, add
        if mode == 'dev':
            for p, pref in prefs.items():
                sc = score(M, pref, features)
                parts.append(pref.select('pair_id').with_columns(pl.lit(fold).alias('fold'), pl.lit(p).alias('surface'), pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        else:
            sc = score(M, ev, features)
            parts.append(ev.select('pair_id').with_columns(pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        del M
        print(json.dumps({**info, 'seconds': round(time.time() - t0)}), flush=True)
    pl.concat(parts).write_parquet(dest)
    json.dump(infos, open(str(dest).replace('.parquet', '.json'), 'w'), indent=1)
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
