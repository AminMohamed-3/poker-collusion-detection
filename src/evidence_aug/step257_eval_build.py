"""STEP257 part 1, stage C: the 850 MECHPLUS columns for every evaluation pair-hand, scored in batches.

For each batch of evaluation hands this assembles

    163 production per-hand features   outputs_v2/hands_evaluation/batch_*.parquet
                                       (the same cache STEP250's evaluation scorer used)
  +  25 STEP237 invisible features     evidence_aug.step246_invisible_features.build(..., 'evaluation')
                                       - STEP251's own evaluation route, which drives STEP237's
                                         functions read-only
  + 662 STEP254 mechanical columns     step257_mech_eval.build_chunk (= step254_features unchanged)
                                       + step257_prep_eval's chronology block

applies the five saved step232-fold models per head, averages their probabilities, and writes ONLY

    pair_id, hand_id, p_any, p_DT, p_SP, p_CI, p_fmax   (Float32)

The 850-column matrix never reaches disk.  Evaluation phase is asserted on every hand touched.

Usage: python step257_eval_build.py <first_batch> <last_batch_exclusive>
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')

import gc
import glob
import json
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

import lightgbm as lgb                              # noqa: E402
import step257_mech_eval as M                       # noqa: E402
import step257_prep_eval as PREP                    # noqa: E402

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step257_eval_mech_scores_20260918'
MODELS = OUT + r'\models'
SCRATCH = PREP.SCRATCH
NBATCH = 20
SUBHANDS = int(os.environ.get('STEP257_SUBHANDS', '2500'))
WORKERS = int(os.environ.get('STEP257_WORKERS', '3'))
HEADS = ('any', 'DT', 'SP', 'CI')
KEYS = ['pair_id', 'hand_id']


def hand_chunks(phase: str = 'evaluation') -> pl.DataFrame:
    """deterministic partition of the phase's pair-hands into NBATCH batches of hands."""
    path = SCRATCH + f'\\hand_chunk_{phase}.parquet'
    if os.path.exists(path):
        return pl.read_parquet(path)
    os.makedirs(SCRATCH, exist_ok=True)
    hs = pl.concat([pl.read_parquet(f, columns=['hand_id'])
                    for f in PREP.batch_files(phase)])['hand_id'].unique().sort()
    b = np.linspace(0, len(hs), NBATCH + 1).astype(int)
    ch = np.zeros(len(hs), np.int32)
    for c in range(NBATCH):
        ch[b[c]:b[c + 1]] = c
    df = pl.DataFrame({'hand_id': hs, 'chunk': ch})
    df.write_parquet(path)
    return df


def load_boosters():
    bs = {}
    for h in HEADS:
        bs[h] = [lgb.Booster(model_file=MODELS + f'\\mechplus_{h}_fold{k}.txt') for k in range(5)]
    return bs


def read_filtered(files, hands: pl.DataFrame, columns=None) -> pl.DataFrame:
    parts = []
    for f in files:
        d = pl.read_parquet(f, columns=columns)
        d = d.join(hands, on='hand_id', how='semi')
        if d.height:
            parts.append(d)
    return pl.concat(parts)


def score_batch(c: int, order: dict, bs, PCT169, EQ169, hc: pl.DataFrame) -> dict:
    t0 = time.time()
    phase = 'evaluation'
    hands = hc.filter(pl.col('chunk') == c).select('hand_id')
    lab = PREP.pair_list(phase)
    chrono = read_filtered(sorted(glob.glob(SCRATCH + f'\\chrono_{phase}\\chrono_b*.parquet')), hands)
    f163 = read_filtered(PREP.batch_files(phase), hands)
    f163 = f163.drop('table_id')
    feats163 = [x for x in f163.columns if x not in KEYS]
    assert feats163 == order['block_163_production'], 'the 163-column layout differs'
    assert chrono.height == f163.height, (chrono.height, f163.height)
    keys = chrono.select(KEYS + ['p1_is_A']).join(lab, on='pair_id', how='left')
    assert keys['player_1'].null_count() == 0
    t_read = time.time() - t0

    # ---- the 25 STEP237 features, through STEP251's own evaluation route
    from evidence_aug import step246_invisible_features as S246
    pmap = {r[0]: (r[1], r[2]) for r in lab.iter_rows()}
    tmp = SCRATCH + f'\\inv_{phase}_b{c:03d}.parquet'
    if os.path.exists(tmp):
        f25 = pl.read_parquet(tmp)
    else:
        f25 = S246.build(keys.select(KEYS), pmap, phase, tmp, workers=WORKERS)
    nm25 = S246.names()
    assert nm25 == order['block_25_step237'], 'the 25-column layout differs'
    t_25 = time.time() - t0 - t_read

    allc, inhand, chron = M.kept_columns()
    assert order['block_662_mechanical'] == allc, 'the 662-column layout differs'
    pos = {nm: j for j, nm in enumerate(order['all'])}
    assert [pos[x] for x in feats163] == list(range(163))
    assert [pos[x] for x in nm25] == list(range(163, 188))
    hl = hands['hand_id'].to_list()
    lut163 = f163.select(KEYS).with_row_index('r163')
    lut25 = f25.select(KEYS).with_row_index('r25')
    X163 = f163.select(feats163).to_numpy()
    X25 = f25.select(nm25).to_numpy()
    del f163, f25
    gc.collect()
    chrono_np = {x: chrono[x].to_numpy() for x in chron}
    lutc = chrono.select(KEYS).with_row_index('rc')
    del chrono
    gc.collect()

    parts = []
    nrow = 0
    ncol = len(order['all'])
    cmin = np.full(ncol, np.inf, np.float64)
    cmax = np.full(ncol, -np.inf, np.float64)
    keep_rows = []          # deterministic 1-in-STRIDE subsample, for the drift table only
    STRIDE = 32
    for a0 in range(0, len(hl), SUBHANDS):
        hsub = pl.DataFrame({'hand_id': hl[a0:a0 + SUBHANDS]})
        ks = keys.join(hsub, on='hand_id', how='semi')
        if not ks.height:
            continue
        mech = M.build_chunk(ks, phase, PCT169, EQ169).select(KEYS + inhand)
        mech = mech.join(lut163, on=KEYS, how='left').join(lut25, on=KEYS, how='left') \
                   .join(lutc, on=KEYS, how='left')
        assert mech['r163'].null_count() == 0 and mech['r25'].null_count() == 0 \
            and mech['rc'].null_count() == 0, 'a pair-hand lost its feature row'
        i163 = mech['r163'].to_numpy()
        i25 = mech['r25'].to_numpy()
        ic = mech['rc'].to_numpy()
        n = mech.height
        X = np.empty((n, len(order['all'])), np.float32)
        X[:, :163] = X163[i163]
        X[:, 163:188] = X25[i25]
        for x in inhand:
            X[:, pos[x]] = mech[x].to_numpy()
        for x in chron:
            X[:, pos[x]] = chrono_np[x][ic]
        kdf = mech.select(KEYS)
        del mech
        for j0 in range(0, X.shape[1], 64):      # chunked: never allocates an (n, 850) bool array
            assert np.isfinite(X[:, j0:j0 + 64]).all(), 'non-finite value in the evaluation matrix'
        np.minimum(cmin, X.min(axis=0), out=cmin)
        np.maximum(cmax, X.max(axis=0), out=cmax)
        sel = np.arange((-nrow) % STRIDE, n, STRIDE)
        if len(sel):
            keep_rows.append(X[sel].copy())
        p = {}
        for h in HEADS:
            acc = np.zeros(n, np.float64)
            for bst in bs[h]:
                acc += bst.predict(X, num_threads=WORKERS)
            p[h] = (acc / 5.0).astype(np.float32)
        fmax = np.maximum(np.maximum(p['DT'], p['SP']), p['CI'])
        parts.append(kdf.with_columns([
            pl.Series('p_any', p['any']), pl.Series('p_DT', p['DT']),
            pl.Series('p_SP', p['SP']), pl.Series('p_CI', p['CI']),
            pl.Series('p_fmax', fmax)]))
        nrow += n
        del X, kdf
        gc.collect()
        print(f'   b{c:03d} hands {a0 + len(hsub):6d}/{len(hl)}  rows {nrow}  '
              f'{time.time() - t0:.0f}s', flush=True)
    out = pl.concat(parts)
    assert out.height == nrow == len(X163) and out.select(KEYS).n_unique() == nrow
    fn = OUT + f'\\eval_hand_scores_batch_{c:03d}.parquet'
    out.write_parquet(fn)
    samp = np.concatenate(keep_rows, axis=0)
    np.save(SCRATCH + f'\\driftsample_b{c:03d}.npy', samp)
    np.savez(SCRATCH + f'\\colstats_b{c:03d}.npz', cmin=cmin, cmax=cmax)
    os.remove(tmp)
    r = {'batch': c, 'hands': len(hl), 'rows': out.height, 'seconds': round(time.time() - t0, 1),
         'read_s': round(t_read, 1), 'invisible_s': round(t_25, 1),
         'drift_sample_rows': int(samp.shape[0])}
    json.dump(r, open(OUT + f'\\log_batch_{c:03d}.json', 'w', encoding='utf-8'), indent=1)
    print('BATCH', json.dumps(r), flush=True)
    return r


def main() -> None:
    c0, c1 = int(sys.argv[1]), int(sys.argv[2])
    order = json.load(open(MODELS + r'\feature_order.json', encoding='utf-8'))
    assert order['n'] == 850
    bs = load_boosters()
    PCT169, EQ169 = M.load_pf169()
    hc = hand_chunks('evaluation')
    for c in range(c0, c1):
        if os.path.exists(OUT + f'\\eval_hand_scores_batch_{c:03d}.parquet'):
            print('batch', c, 'already done', flush=True)
            continue
        score_batch(c, order, bs, PCT169, EQ169, hc)


if __name__ == '__main__':
    main()
