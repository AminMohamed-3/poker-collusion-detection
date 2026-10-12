"""STEP258 step 3: event probabilities for the prefix-60 testbed pair-hands (and the parity universe).

  arm 'mechplus'  uses the STEP257 fold models
                  (experiments/step257_eval_mech_scores_20260918/models/, whose fit gate reports
                  max_abs_diff_vs_step254_oof = 0.0 on all four heads, i.e. they reproduce the
                  STEP254 out-of-fold probabilities exactly).  No training here.
  arm 'plus'      the 163 production + 25 STEP237 columns only (old information control).
  arm 'perm'      the MECHPLUS columns with the mechanical block shuffled ACROSS THE HANDS OF THE
                  SAME PAIR, in training and in the scored frames alike (step254_eval.py 'perm'
                  construction, rng seed 1254, stable argsort by pair_id, block-wise shuffle).
  'plus' and 'perm' are trained here with the step254_eval.py recipe verbatim and their out-of-fold
  probabilities are checked against the stored plus_p_* / perm_p_* columns of hand_oof_step254.parquet.

Every pair is scored ONLY by the fold model whose held-out fold contains the pair's table (step232
folds).  Development phase only.  Writes models/ (plus, perm), scores_<arm>_{prefix,full149}.parquet
and score_receipt_<arm>.json under experiments/step258_mech_block_view_20260918/.

Usage:  python step258_train_score.py {mechplus|plus|perm}
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, gc, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl, lightgbm as lgb

OUT = C + 'experiments/step258_mech_block_view_20260918/'
S254 = C + 'experiments/step254_mechanical_tensor_20260917/'
S257 = C + 'experiments/step257_eval_mech_scores_20260918/'
S237 = C + 'experiments/step237_invisible_events_20260917/'
FAMS = (('DT', 'directed_transfer'), ('SP', 'soft_play'), ('CI', 'coordinated_isolation'))
NJOBS = 5
P = dict(n_estimators=400, num_leaves=15, learning_rate=.03, min_child_samples=30, subsample=.8,
         subsample_freq=1, colsample_bytree=.5, reg_lambda=5.0, n_jobs=NJOBS, verbosity=-1,
         random_state=232)
PERM_SEED = 1254


def log(*a):
    print(*a, flush=True)


def within_pair_shuffle(pair_ids, seed=PERM_SEED):
    """step254_eval.py 'perm' construction: a within-pair row permutation, stable argsort by pair_id."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pair_ids))
    order = np.argsort(pair_ids, kind='stable')
    starts = np.flatnonzero(np.r_[True, pair_ids[order][1:] != pair_ids[order][:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        blk = order[a:b].copy()
        rng.shuffle(blk)
        idx[order[a:b]] = blk
    return idx


def load_training_frame(arm):
    """step254_eval.load() verbatim + the joins that arm needs. Returns (frame, column order)."""
    h = pl.read_parquet(ROOT + 'outputs_v2/hands_development/batch_*.parquet')
    lab = pl.read_csv(DATA + 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    ev = (pl.read_csv(DATA + 'development_evidence.csv').select('pair_id', 'hand_id')
          .with_columns(pl.lit(1).alias('listed')))
    phase = pl.read_parquet(DATA + 'hands.parquet', columns=['hand_id', 'phase'])
    h = h.join(phase, on='hand_id', how='left')
    assert (h['phase'] == 'development').all(), 'non-development hand in the feature cache'
    h = h.drop('phase')
    h = (h.join(lab, on='pair_id').join(ev, on=['pair_id', 'hand_id'], how='left')
         .with_columns(pl.col('listed').fill_null(0)))
    feats = [c for c in h.columns
             if c not in ('pair_id', 'hand_id', 'table_id', 'label', 'behavior_family', 'listed')]
    tables = h['table_id'].unique().sort().to_list()
    rng = np.random.default_rng(232)
    perm = rng.permutation(len(tables))
    fold_of = {t: int(i % 5) for t, i in zip(tables, np.argsort(perm))}
    h = h.with_columns(pl.col('table_id').replace_strict(fold_of, return_dtype=pl.Int64).alias('fold'))
    ref = pl.read_parquet(C + 'experiments/step232_event_detector_20260917/hand_oof.parquet',
                          columns=['pair_id', 'hand_id', 'fold'])
    chk = h.select('pair_id', 'hand_id', 'fold').join(ref, on=['pair_id', 'hand_id'], suffix='_ref')
    assert chk.height == h.height and (chk['fold'] == chk['fold_ref']).all(), 'fold mismatch vs step232'
    f2 = pl.read_parquet(S237 + 'step237_features.parquet')
    F237 = [c for c in f2.columns if c not in ('pair_id', 'hand_id')]
    h = h.join(f2, on=['pair_id', 'hand_id'], how='left')
    assert h[F237[0]].null_count() == 0
    MECH = []
    if arm in ('mechplus', 'perm'):
        fx = pl.read_parquet(S254 + 'step254_features.parquet')
        MECH = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
        h = h.join(fx, on=['pair_id', 'hand_id'], how='left')
        del fx
        gc.collect()
        assert h[MECH[0]].null_count() == 0
    return h, feats + F237 + MECH, feats, F237, MECH, fold_of


def matrix_of(frame, cols, mech, perm_rows=None):
    """column-by-column fill (never a float64 copy); optionally permute the mech block within pairs."""
    X = np.empty((frame.height, len(cols)), np.float32)
    nold = len(cols) - len(mech)
    for i, c in enumerate(cols[:nold]):
        X[:, i] = frame[c].to_numpy()
    if mech:
        M = np.empty((frame.height, len(mech)), np.float32)
        for i, c in enumerate(mech):
            M[:, i] = frame[c].to_numpy()
        X[:, nold:] = M[perm_rows] if perm_rows is not None else M
        del M
        gc.collect()
    return X


def score_universes(arm, boosters, cols, mech):
    """score the prefix testbed and the full-history parity universe, fold-matched by table."""
    ft = json.load(open(S257 + 'models/fold_tables.json', encoding='utf-8'))['table_to_fold']
    out = {}
    for mode in ('prefix', 'full149'):
        f = pl.read_parquet(OUT + f'mechplus_{mode}.parquet')
        assert all(c in f.columns for c in cols)
        fold = np.array([ft[t] for t in f['table_id'].to_list()], np.int64)
        pr = within_pair_shuffle(f['pair_id'].to_numpy()) if arm == 'perm' else None
        X = matrix_of(f, cols, mech, pr)
        p = {}
        for tag, _ in FAMS:
            v = np.full(f.height, np.nan)
            for k in range(5):
                te = fold == k
                if te.any():
                    v[te] = boosters[(tag, k)].predict(X[te])
            assert np.isfinite(v).all()
            p[tag] = v
        del X
        gc.collect()
        fm = np.max(np.stack([p[t] for t, _ in FAMS]), axis=0)
        d = f.select('pair_id', 'hand_id', 'table_id').with_columns(
            [pl.Series(f'p_{t}', p[t]) for t, _ in FAMS] + [pl.Series('p_fmax', fm),
                                                            pl.Series('fold', fold)])
        d.write_parquet(OUT + f'scores_{arm}_{mode}.parquet')
        out[mode] = {'rows': int(d.height), 'pairs': int(d['pair_id'].n_unique()),
                     'p_fmax_mean': float(fm.mean()), 'p_fmax_max': float(fm.max())}
        log('scored', arm, mode, out[mode])
    return out


def main():
    arm = sys.argv[1]
    assert arm in ('mechplus', 'plus', 'perm')
    t0 = time.time()
    rec = {'arm': arm, 'n_jobs': NJOBS, 'params': P}
    h, cols, feats, F237, MECH, fold_of = load_training_frame(arm)
    log('training frame', h.height, 'columns', len(cols), round(time.time() - t0, 1), 's')
    if arm == 'mechplus':
        fo = json.load(open(S257 + 'models/feature_order.json', encoding='utf-8'))['all']
        assert fo == cols, 'STEP257 feature order differs from the step254_eval order'
        boosters = {}
        for tag, _ in FAMS:
            for k in range(5):
                boosters[(tag, k)] = lgb.Booster(model_file=S257 + f'models/mechplus_{tag}_fold{k}.txt')
        rec['models'] = 'STEP257 experiments/step257_eval_mech_scores_20260918/models (fit gate: exact vs STEP254 OOF)'
        rec['oof_parity'] = json.load(open(S257 + 'models/fit_gate257.json', encoding='utf-8'))['heads']
        del h
        gc.collect()
    else:
        mdir = OUT + 'models/'
        os.makedirs(mdir, exist_ok=True)
        y_listed = h['listed'].to_numpy()
        lbl = h['label'].to_numpy()
        fold = h['fold'].to_numpy()
        fam = h['behavior_family'].to_numpy()
        pid = h['pair_id'].to_numpy()
        pr = within_pair_shuffle(pid) if arm == 'perm' else None
        X = matrix_of(h, cols, MECH, pr)
        del h
        gc.collect()
        log('matrix', X.shape, round(time.time() - t0, 1), 's')
        boosters, oofp = {}, {}
        for tag, family in FAMS:
            posmask = (y_listed == 1) & (fam == family)
            v = np.full(len(lbl), np.nan)
            for k in range(5):
                tr = (fold != k) & (posmask | (lbl == 0))
                m = lgb.LGBMClassifier(**P).fit(X[tr], posmask[tr].astype(int))
                b = m.booster_
                b.save_model(mdir + f'{arm}_{tag}_fold{k}.txt')
                boosters[(tag, k)] = lgb.Booster(model_file=mdir + f'{arm}_{tag}_fold{k}.txt')
                te = fold == k
                v[te] = b.predict(X[te])
                del m, b
                gc.collect()
                log(f'  {arm} {tag} fold {k} n_train {int(tr.sum())} {round(time.time() - t0, 1)}s')
            oofp[tag] = v
        del X
        gc.collect()
        ref = pl.read_parquet(S254 + 'hand_oof_step254.parquet',
                              columns=['pair_id', 'hand_id'] + [f'{arm}_p_{t}' for t, _ in FAMS])
        key = {k: i for i, k in enumerate(zip(ref['pair_id'].to_list(), ref['hand_id'].to_list()))}
        # the training frame row order equals the OOF frame row order only up to the join; map explicitly
        tf = pl.read_parquet(ROOT + 'outputs_v2/hands_development/batch_*.parquet',
                             columns=['pair_id', 'hand_id'])
        gi = np.fromiter((key[k] for k in zip(tf['pair_id'].to_list(), tf['hand_id'].to_list())),
                         np.int64, tf.height)
        par = {}
        for tag, _ in FAMS:
            a = oofp[tag]
            b = ref[f'{arm}_p_{tag}'].to_numpy()[gi]
            par[tag] = {'max_abs_diff_vs_step254_oof': float(np.abs(a - b).max()),
                        'exact': bool(np.array_equal(a, b))}
            log(arm, tag, par[tag])
        rec['models'] = mdir
        rec['oof_parity'] = par
        json.dump({'feature_order': cols, 'n': len(cols)},
                  open(mdir + f'feature_order_{arm}.json', 'w', encoding='utf-8'), indent=1)
    rec['scored'] = score_universes(arm, boosters, cols, MECH)
    res = json.load(open(S254 + f'results_{arm}.json', encoding='utf-8'))[arm]
    rec['honest_thresholds_stored'] = res['_thresholds']
    rec['seconds'] = round(time.time() - t0, 1)
    json.dump(rec, open(OUT + f'score_receipt_{arm}.json', 'w', encoding='utf-8'), indent=1)
    log('DONE', arm, rec['seconds'], 's')


if __name__ == '__main__':
    main()
