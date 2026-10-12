"""Step 158a (Claude): table-bagged tail views of outputs_step156_20260915/prediction_step158.json. Draw k = STEP156 SXN draw k with training
rows restricted to floor(.8 n) sampled tables (seed 158000 + 10 fold + k dev, 158900 + k eval). Five incumbent view types refitted.
Development: step158a_C_bagged_views.py dev FOLD K0 K1 -> experiments/bagged_views_20260915/sxb_dev_fold{F}_k{K0}-{K1}.parquet
Evaluation:  step158a_C_bagged_views.py eval K0 K1      -> experiments/bagged_views_20260915/sxb_eval_k{K0}-{K1}.parquet"""
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
from pu_pairs.step156b_C_SXN_views import candidates, added_rows, fit_views_neg

REG = CV / 'outputs_step156_20260915' / 'prediction_step158.json'
REG_SHA = 'b46f88038938d7302c2c6e092ceaabcffef6697dede602732676ea0ace23a9e3'
OUT = CV / 'experiments' / 'bagged_views_20260915'


def bag(fr, add, cand, seed):
    tabs = sorted(set(fr[1.0]['table_id'].to_list()) | set(cand['table_id'].to_list()))
    keep = set(np.random.default_rng(seed).choice(tabs, size=int(np.floor(.8 * len(tabs))), replace=False).tolist())
    fr2 = {f: v.filter(pl.col('table_id').is_in(sorted(keep))) for f, v in fr.items()}
    tmap = cand.select('pair_id', 'table_id')
    add2 = {L: v.join(tmap, on='pair_id', how='left').filter(pl.col('table_id').is_in(sorted(keep))).drop('table_id') for L, v in add.items()}
    return fr2, add2, {'tables': len(tabs), 'kept': len(keep), 'full_frame_rows_kept': fr2[1.0].height, 'added_full_rows_kept': add2['full'].height}


def main():
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == REG_SHA
    mode = sys.argv[1]; t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    parts, infos = [], []
    if mode == 'dev':
        fold, k0, k1 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); g = 3 - fold
        held = set(fold_pools(fold)); cand = candidates(g, held)
        prefs = {p: pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
                 for p in ('prefix40', 'prefix60', 'prefix80')}
        dest = OUT / f'sxb_dev_fold{fold}_k{k0}-{k1}.parquet'
    else:
        k0, k1 = int(sys.argv[2]), int(sys.argv[3]); held = set()
        c1, c2 = candidates(1, set()), candidates(2, set()); cand = pl.concat([c1, c2.join(c1, on='pair_id', how='anti')])
        ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
        dest = OUT / f'sxb_eval_k{k0}-{k1}.parquet'
    for k in range(k0, k1 + 1):
        if mode == 'dev':
            fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features, k)
            seed, bseed = 156000 + 10 * fold + k, 158000 + 10 * fold + k
        else:
            fr = view_frames_seeded(FINAL_CA, held, features, k); seed, bseed = 156900 + k, 158900 + k
        add, info = added_rows(cand, fr[1.0].height, seed, features)
        fr, add, binfo = bag(fr, add, cand, bseed)
        info = {**info, **binfo, 'k': k}; infos.append(info)
        M = fit_views_neg(fr, add, features, k); del fr, add
        if mode == 'dev':
            for p, pref in prefs.items():
                sc = score(M, pref, features)
                parts.append(pref.select('pair_id').with_columns(pl.lit(fold).alias('fold'), pl.lit(p).alias('surface'), pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        else:
            sc = score(M, ev, features)
            parts.append(ev.select('pair_id').with_columns(pl.lit(k).alias('k'), *[pl.Series(c, v) for c, v in sc.items()]))
        print(json.dumps({**info, 'seconds': round(time.time() - t0)}), flush=True)
    pl.concat(parts).write_parquet(dest)
    json.dump(infos, open(str(dest).replace('.parquet', '.json'), 'w'), indent=1)
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
