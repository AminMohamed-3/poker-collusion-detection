"""STEP131 (poker-a8): rebuild the fitted tail views of the pair decomposition (RAW, CA, BASE, OPP, V2, MEAN, TAILS, SHORT3, ALL5, PROD, LIN30, LIN300,
DT, SP, CI, FAMMAX) per fold with the step 53/60/64a recipes (held pools excluded; MODEL_PARAMS fixed seed) and score the fold's prefix-40/60/80
pair frames. prefix-60 is scored only to prove the rebuild reproduces the stored views. Development labels: training frames only; no arm is evaluated."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib, lightgbm as lgb
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs import step53_view_disagreement as S53, step60_multiview_blend as S60, step64a_tail_aggregate_views as S64
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, MODEL_PARAMS, TRAIN_VIEWS, matrix  # noqa: E402
OUT, VD = CV / 'experiments' / 'tail_view_confirmation_a8_20260913', CV / 'experiments' / 'viewdis_20260911'
t0 = time.time()
log = lambda *a: print(*a, f'{time.time() - t0:.0f}s', flush=True)
features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
tv = [(float(a), int(b)) for a, b in TRAIN_VIEWS]
assert tv == [(0.7, S64.FRACTIONS[0.7]), (0.4, S64.FRACTIONS[0.4])], tv
parts = []
for fold in (1, 2):
    fr = S64.view_frames(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', set(fold_pools(fold)), features)
    prod = pl.concat([fr[f] for f in S64.TRAIN_SETS['PROD']])
    y = np.array([LABELS[f] for f in prod['behavior_family']])
    sets = {k: v for k, v in S53.view_sets(features).items() if k in ('RAW', 'CA')}
    sets.update(S60.view_sets(features))
    M = {k: lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(prod, cols), y) for k, cols in sets.items()}
    for name in ('SHORT3', 'ALL5', 'PROD'):
        M[name] = S64.fit_multiclass(pl.concat([fr[f] for f in S64.TRAIN_SETS[name]]), features); sets[name] = features
    lin = {l2: S64.fit_linear(prod, features, l2) for l2 in (30.0, 300.0)}
    ovr = {k: S64.fit_ovr(prod, features, fam) for k, fam in S64.FAMS.items()}
    log(f'fold{fold} fitted', {str(k): v.height for k, v in fr.items()})
    del fr, prod
    for p in ('prefix40', 'prefix60', 'prefix80'):
        pref = pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features))) for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
        out = pref.select('pair_id').with_columns(pl.lit(fold).alias('fold'), pl.lit(p).alias('surface'))
        for k, m in M.items():
            out = out.with_columns(pl.Series(f'r_{k}', 1 - m.predict_proba(matrix(pref, sets[k]))[:, 0]))
        for l2, lm in lin.items():
            out = out.with_columns(pl.Series(f'r_LIN{int(l2)}', S64.score_linear(lm, pref, features)))
        for k, m in ovr.items():
            out = out.with_columns(pl.Series(f'r_{k}', m.predict_proba(matrix(pref, features))[:, 1]))
        parts.append(out.with_columns(pl.max_horizontal('r_DT', 'r_SP', 'r_CI').alias('r_FAMMAX')))
        log(f'fold{fold} {p} scored', pref.height)
        del pref
new = pl.concat(parts)
new.write_parquet(OUT / 'views_prefix40_60_80.parquet')
p60 = new.filter(pl.col('surface') == 'prefix60')
chk = {}
j = p60.join(pl.read_parquet(VD / 'tail_views_prefix.parquet').rename({'fold_new': 'fold'}), on=['pair_id', 'fold'], how='inner', suffix='_old')
chk['tail_views_rows'] = j.height
for c in ['r_SHORT3', 'r_ALL5', 'r_PROD', 'r_LIN30', 'r_LIN300', 'r_DT', 'r_SP', 'r_CI', 'r_FAMMAX']:
    chk[c] = float((j[c] - j[c + '_old']).abs().max())
V6 = ['r_RAW', 'r_CA', 'r_BASE', 'r_OPP', 'r_V2', 'r_MEAN', 'r_TAILS']
j = p60.join(pl.read_parquet(VD / 'multiview_prefix.parquet', columns=['pair_id', 'fold'] + V6), on=['pair_id', 'fold'], how='inner', suffix='_old')
chk['multiview_rows'] = j.height
for c in V6:
    chk[c] = float((j[c] - j[c + '_old']).abs().max())
res = {'rows': new.height, 'by_surface': dict(new.group_by('surface').len().iter_rows()), 'replication_prefix60_max_abs_diff': chk, 'seconds': round(time.time() - t0)}
json.dump(res, open(OUT / 'views_run.json', 'w', encoding='utf-8'), indent=1)
log(json.dumps(res))
