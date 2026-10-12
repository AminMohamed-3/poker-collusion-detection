"""Step 156b (Claude): Part B of outputs_step156_20260915/prediction_step156.json (sha fba91dde). Multi-draw tail views (step151c recipe)
plus added training rows from development all-pairs frames: every labelled candidate pair with its label and M reliable unlabelled pairs
(v13 fold risk < .5) as 'none'. Lengths: SHORT3 <- prefix40; ALL5 <- full, prefix80, prefix60, prefix40; PROD <- full, prefix60, prefix40.
Development: python step156b_C_SXN_views.py dev FOLD K0 K1 -> experiments/tail_keys_20260915_neg/sxn_dev_fold{F}_k{K0}-{K1}.parquet
Evaluation:  python step156b_C_SXN_views.py eval K0 K1      -> experiments/tail_keys_20260915_neg/sxn_eval_k{K0}-{K1}.parquet
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
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded, params, score
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, matrix  # noqa: E402

REG = CV / 'outputs_step156_20260915' / 'prediction_step156.json'
REG_SHA = 'fba91ddedfcb556ef2be83f0c6ea5a3edea57b703de150479292bcc374a7e43d'
OUT = CV / 'experiments' / 'tail_keys_20260915_neg'
LENGTHS = {'full': '', 'prefix80': '-prefix80', 'prefix60': '-prefix60', 'prefix40': '-prefix40'}
ADD = {'SHORT3': ['prefix40'], 'ALL5': ['full', 'prefix80', 'prefix60', 'prefix40'], 'PROD': ['full', 'prefix60', 'prefix40']}
RATIO = 2.9


def candidates(g, exclude_tables):
    s = pl.read_parquet(EXP / f'fold{g}' / 'scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'shared_hands', 'risk'])
    s = s.filter((pl.col('shared_hands') >= 38) & ~pl.col('table_id').is_in(sorted(exclude_tables)))
    for L, suf in LENGTHS.items():
        ids = pl.scan_parquet(sorted((EXP / f'fold{g}{suf}').glob('batch_*-pairs.parquet'))).select('pair_id').collect()
        s = s.join(ids, on='pair_id', how='semi')
    return s.with_columns(pl.lit(g).alias('src_fold'))


def read_rows(g, L, ids, features):
    return (pl.scan_parquet(sorted((EXP / f'fold{g}{LENGTHS[L]}').glob('batch_*-pairs.parquet'))).select(['pair_id'] + features)
            .filter(pl.col('pair_id').is_in(ids)).collect())


def added_rows(cand, n_lab_full, seed, features):
    lab = cand.filter(pl.col('label') >= 0)
    unl = cand.filter((pl.col('label') == -1) & (pl.col('risk') < .5)).sort('pair_id')
    M = int(round(RATIO * n_lab_full))
    pick = np.random.default_rng(seed).choice(unl.height, size=M, replace=False)
    unl = unl[np.sort(pick)].with_columns(pl.lit(0, dtype=pl.Int64).alias('label'), pl.lit('none').alias('behavior_family'))
    meta = pl.concat([lab.select('pair_id', 'src_fold', 'label', 'behavior_family'), unl.select('pair_id', 'src_fold', 'label', 'behavior_family')])
    out = {}
    for L in LENGTHS:
        parts = []
        for g in sorted(meta['src_fold'].unique().to_list()):
            ids = meta.filter(pl.col('src_fold') == g)['pair_id'].to_list()
            parts.append(read_rows(g, L, ids, features).join(meta.filter(pl.col('src_fold') == g).select('pair_id', 'label', 'behavior_family'), on='pair_id', validate='1:1'))
        out[L] = pl.concat(parts)
    info = {'labelled_added': lab.height, 'labelled_pos_added': int((lab['label'] == 1).sum()), 'unlabelled_pool': int(cand.filter((pl.col('label') == -1) & (pl.col('risk') < .5)).height), 'M': M,
            'rows': {L: v.height for L, v in out.items()}}
    return out, info


def fit_views_neg(fr, add, features, k):
    cols = ['pair_id', 'label', 'behavior_family'] + features
    def ts(name):
        return pl.concat([fr[f].select(cols) for f in S64.TRAIN_SETS[name]] + [add[L].select(cols) for L in ADD[name]], how='vertical_relaxed')
    M = {}
    for name in ('SHORT3', 'ALL5'):
        tr = ts(name)
        M[name] = ('mc', lgb.LGBMClassifier(**params(k)).fit(matrix(tr, features), np.array([LABELS[f] for f in tr['behavior_family']])))
    prod = ts('PROD')
    for l2 in (30.0, 300.0):
        M[f'LIN{int(l2)}'] = ('lin', S64.fit_linear(prod, features, l2))
    for key, fam in S64.FAMS.items():
        M[key] = ('ovr', lgb.LGBMClassifier(**params(k)).fit(matrix(prod, features), (prod['behavior_family'] == fam).to_numpy().astype(int)))
    return M


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
        dest = OUT / f'sxn_dev_fold{fold}_k{k0}-{k1}.parquet'
    else:
        k0, k1 = int(sys.argv[2]), int(sys.argv[3]); held = set()
        c1, c2 = candidates(1, set()), candidates(2, set())
        cand = pl.concat([c1, c2.join(c1, on='pair_id', how='anti')])
        ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id'] + features)))
        dest = OUT / f'sxn_eval_k{k0}-{k1}.parquet'
    for k in range(k0, k1 + 1):
        if mode == 'dev':
            fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', held, features, k); seed = 156000 + 10 * fold + k
        else:
            fr = view_frames_seeded(FINAL_CA, held, features, k); seed = 156900 + k
        add, info = added_rows(cand, fr[1.0].height, seed, features)
        info.update({'k': k, 'n_lab_full_frame': fr[1.0].height}); infos.append(info)
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
