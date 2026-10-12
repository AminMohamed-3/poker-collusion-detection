"""STEP131 side output for poker-09 (pair TTA lane): refit and save the per-fold PROD multiclass pair models exactly as tv131_views.py
(step 64a recipe, held pools excluded, MODEL_PARAMS fixed seed). Checks: prefix-60 scores equal the stored r_PROD; full-length scores
against the fold's v13 out-of-fold risk (scores.parquet). Training frames only; nothing is evaluated."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
from evidence_aug.paths import ORIGINAL, CV
import polars as pl, joblib
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs import step64a_tail_aggregate_views as S64
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import MODEL_PARAMS, matrix  # noqa: E402
OUT = CV / 'experiments' / 'tail_view_confirmation_a8_20260913'
MD = OUT / 'prod_models'
MD.mkdir(exist_ok=True)
t0 = time.time()
log = lambda *a: print(*a, f'{time.time() - t0:.0f}s', flush=True)
features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
views = pl.read_parquet(OUT / 'views_prefix40_60_80.parquet', columns=['pair_id', 'fold', 'surface', 'r_PROD'])
res = {}
for fold in (1, 2):
    pools = sorted(fold_pools(fold))
    fr = S64.view_frames(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', set(pools), features)
    prod = pl.concat([fr[f] for f in S64.TRAIN_SETS['PROD']])
    m = S64.fit_multiclass(prod, features)
    del fr
    fp = MD / f'prod_model_fold{fold}.joblib'
    joblib.dump({'model': m, 'features': features, 'fold': fold, 'held_pools': pools, 'train_fractions': [str(x) for x in S64.TRAIN_SETS['PROD']],
                 'train_rows': prod.height, 'model_params': MODEL_PARAMS, 'score': "1 - model.predict_proba(robust_pair_experiment.matrix(frame, features))[:, 0]"}, fp)
    r = {'train_rows': prod.height, 'file': fp.name, 'sha256': hashlib.sha256(fp.read_bytes()).hexdigest()}
    for p in ('prefix60', 'full'):
        src = EXP / (f'fold{fold}-{p}' if p != 'full' else f'fold{fold}')
        pref = pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted(src.glob('batch_*-pairs.parquet'))])
        sc = pref.select('pair_id').with_columns(pl.Series('r', 1 - m.predict_proba(matrix(pref, features))[:, 0]))
        del pref
        if p == 'prefix60':
            j = sc.join(views.filter((pl.col('surface') == p) & (pl.col('fold') == fold)), on='pair_id', how='inner')
            r['prefix60_vs_views_r_PROD'] = {'rows': j.height, 'of': sc.height, 'max_abs': float((j['r'] - j['r_PROD']).abs().max())}
        else:
            j = sc.join(pl.read_parquet(src / 'scores.parquet', columns=['pair_id', 'risk']), on='pair_id', how='inner')
            r['full_vs_v13_fold_risk'] = {'rows': j.height, 'of': sc.height, 'max_abs': float((j['r'] - j['risk']).abs().max())}
    res[f'fold{fold}'] = r
    log(f'fold{fold}', json.dumps(r))
json.dump(res, open(MD / 'prod_models_run.json', 'w', encoding='utf-8'), indent=1)
log('done')
