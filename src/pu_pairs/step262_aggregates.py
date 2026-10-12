"""STEP262 step 3: the six pooled per-pair aggregates, for the harness rows and for the training rows.

registration262.json 'inputs.aggregates' / 'inputs.training_aggregates'.

  dev    experiments/step262_mech_tail_view_20260918/aggregates262.parquet
         one row per (pair_id, surface) of the harness D, from the STEP262 shard scores.
  train  experiments/step262_mech_tail_view_20260918/train_aggs262.parquet
         one row per (pair_id, draw k, thinning fraction) of the LABELLED development pairs, from the
         STORED STEP254 out-of-fold per-hand probabilities over the pair's FULL development history,
         thinned with robust_pair_experiment.thin at the PRODUCTION fractions 1.0 / 0.7 / 0.4 and the
         PRODUCTION seeds step64a.FRACTIONS[frac] + 1000*k (the step151c / step159a / step163a rule).

Aggregates = evidence_aug/step254_D2_pair.pair_aggs verbatim (step258_aggregates.pair_aggs), each arm
with its own stored honest thresholds.  Development phase only.

Usage:  python step262_aggregates.py {dev|train|both}
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
import sys, glob, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl                                        # noqa: E402
from pu_pairs.step258_aggregates import pair_aggs, AGG                  # noqa: E402
from pu_pairs import step64a_tail_aggregate_views as S64                # noqa: E402
sys.path.insert(0, str(ROOT + 'src'))
from robust_pair_experiment import thin                                 # noqa: E402

OUT = C + 'experiments/step262_mech_tail_view_20260918/'
S254 = C + 'experiments/step254_mechanical_tensor_20260917/'
S258 = C + 'experiments/step258_mech_block_view_20260918/'
SURF = ('prefix40', 'prefix60', 'prefix80')
ARMS = ('mechplus', 'plus')
NSHARDS = 24
DRAWS = 8
FRACS = (1.0, 0.7, 0.4)                       # step64a TRAIN_SETS['PROD']


def log(*a):
    print(*a, flush=True)


def thresholds():
    return {a: json.load(open(S258 + f'score_receipt_{a}.json', encoding='utf-8'))['honest_thresholds_stored']
            for a in ARMS}


def aggs_of(frame, pcols, thr, idcols):
    """ONE group_by for every arm, so the per-pair lists share one order.
    Returns (ids, n_hands, {arm: 6-column array})."""
    g = frame.group_by(idcols).agg([pl.col(c) for c in pcols.values()])
    ids = {c: g[c].to_list() for c in idcols}
    n = None
    A = {}
    for arm, pcol in pcols.items():
        M = np.zeros((g.height, 6), np.float32)
        nn = np.zeros(g.height, np.int64)
        q99, q999 = thr[arm]['honest_q99'], thr[arm]['honest_q999']
        for i, v in enumerate(g[pcol].to_list()):
            a = np.asarray(v, dtype=np.float64)
            nn[i] = len(a)
            M[i] = pair_aggs(a, q99, q999)
        A[arm] = M
        if n is None:
            n = nn
        else:
            assert np.array_equal(n, nn)
    return ids, n, A


def build_dev():
    t0 = time.time()
    thr = thresholds()
    u = pl.read_parquet(OUT + 'universe262.parquet').select('pair_id', 'surface', 'fold', 'table_id', 'label', 'risk', 'shared_hands')
    parts = []
    for p in SURF:
        files = sorted(glob.glob(OUT + f'scores/scores_{p}_s{NSHARDS:03d}_*.parquet'))
        assert len(files) == NSHARDS, (p, len(files))
        sc = pl.read_parquet(files)
        want = u.filter(pl.col('surface') == p)
        assert sc['pair_id'].n_unique() == want.height, (p, sc['pair_id'].n_unique(), want.height)
        ids, n, A = aggs_of(sc, {a: f'{a}_p_fmax' for a in ARMS}, thr, ['pair_id'])
        cols = {f'{arm}_{nm}': A[arm][:, j] for arm in ARMS for j, nm in enumerate(AGG)}
        d = pl.DataFrame({'pair_id': ids['pair_id'], 'n_hands': n, **cols}).with_columns(pl.lit(p).alias('surface'))
        d = d.join(want.select('pair_id', 'fold', 'table_id', 'label', 'risk', 'shared_hands'), on='pair_id', how='inner', validate='1:1')
        assert d.height == want.height
        assert int((d['n_hands'] != d['shared_hands']).sum()) == 0, 'n_hands != stored shared_hands'
        parts.append(d)
        log('dev aggregates', p, d.shape, round(time.time() - t0, 1), 's')
        del sc
    out = pl.concat(parts)
    out.write_parquet(OUT + 'aggregates262.parquet')
    log('wrote aggregates262.parquet', out.shape)


def build_train():
    t0 = time.time()
    thr = thresholds()
    keys = pl.read_parquet(sorted(glob.glob(ROOT + 'outputs_v2/hands_development/batch_*.parquet')),
                           columns=['pair_id', 'hand_id', 'table_id'])
    ph = pl.read_parquet(DATA + 'hands.parquet', columns=['hand_id', 'phase'])
    chk = keys.select('hand_id').unique().join(ph, on='hand_id', how='left')
    assert (chk['phase'] == 'development').all(), 'non-development hand in the STEP262 training universe'
    oof = pl.read_parquet(S254 + 'hand_oof_step254.parquet',
                          columns=['pair_id', 'hand_id', 'mechplus_p_fmax', 'plus_p_fmax'])
    h = keys.join(oof, on=['pair_id', 'hand_id'], how='inner', validate='1:1')
    assert h.height == keys.height, (h.height, keys.height)
    lab = pl.read_csv(DATA + 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    log('training hand frame', h.shape, 'pairs', h['pair_id'].n_unique(), round(time.time() - t0, 1), 's')
    parts = []
    for k in range(DRAWS):
        for frac in FRACS:
            seed = S64.FRACTIONS[frac]
            f = h if frac == 1.0 else thin(h, frac, seed + 1000 * k)
            ids, n, A = aggs_of(f, {a: f'{a}_p_fmax' for a in ARMS}, thr, ['pair_id'])
            cols = {f'{arm}_{nm}': A[arm][:, j] for arm in ARMS for j, nm in enumerate(AGG)}
            d = pl.DataFrame({'pair_id': ids['pair_id'], 'n_hands': n, **cols}).with_columns(pl.lit(k).alias('k'), pl.lit(frac).alias('frac'))
            parts.append(d.join(lab, on='pair_id', how='inner', validate='1:1'))
        log('train aggregates draw', k, round(time.time() - t0, 1), 's')
    out = pl.concat(parts)
    tab = h.select('pair_id', 'table_id').unique(subset=['pair_id'])
    out = out.join(tab, on='pair_id', how='left', validate='m:1')
    assert out['table_id'].null_count() == 0
    out.write_parquet(OUT + 'train_aggs262.parquet')
    rec = {'rows': int(out.height), 'pairs': int(out['pair_id'].n_unique()), 'draws': DRAWS, 'fracs': list(FRACS),
           'thinning_seeds': {str(f): (None if S64.FRACTIONS[f] is None else [S64.FRACTIONS[f] + 1000 * k for k in range(DRAWS)]) for f in FRACS},
           'n_pos': int((out.filter(pl.col('frac') == 1.0)['label'] == 1).sum()),
           'n_neg': int((out.filter(pl.col('frac') == 1.0)['label'] == 0).sum()),
           'seconds': round(time.time() - t0, 1)}
    json.dump(rec, open(OUT + 'train_aggs262.json', 'w', encoding='utf-8'), indent=1)
    log('wrote train_aggs262.parquet', out.shape, json.dumps({k2: v for k2, v in rec.items() if k2 != 'thinning_seeds'}))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if what in ('train', 'both'):
        build_train()
    if what in ('dev', 'both'):
        build_dev()


if __name__ == '__main__':
    main()
