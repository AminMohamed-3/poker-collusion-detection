"""STEP257 gate G_parity: the evaluation code path, run unchanged on DEVELOPMENT pair-hands.

The only thing that changes between this run and the production evaluation run is the phase string
and the pair list.  Everything else - step257_prep_eval.chrono_for, step257_mech_eval.build_chunk,
step246_invisible_features.build, the 163-column cache lookup and the row-alignment machinery of
step257_eval_build - is the same code.

Compared, column by column, on a sample of complete development pairs:
  662  vs experiments/step254_mechanical_tensor_20260917/step254_features.parquet
   25  vs experiments/step237_invisible_events_20260917/step237_features.parquet
  163  vs the frame step254_eval.load() builds from outputs_v2/hands_development/batch_*.parquet
Required: max abs diff exactly 0 (< 1e-5 tolerated on Float32 equity / strength columns).

Also re-checks that Booster.predict (what the evaluation scorer calls) equals the saved STEP254
out-of-fold probabilities on the sampled fold rows.

Writes experiments/step257_eval_mech_scores_20260918/gate_parity257.json
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import glob
import json
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True

import lightgbm as lgb                       # noqa: E402
import step257_mech_eval as M                # noqa: E402
import step257_prep_eval as PREP             # noqa: E402

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step257_eval_mech_scores_20260918'
MODELS = OUT + r'\models'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'
S237 = CV + r'\experiments\step237_invisible_events_20260917'
SCRATCH = PREP.SCRATCH
KEYS = ['pair_id', 'hand_id']
PHASE = 'development'
NPAIR = int(os.environ.get('STEP257_GATE_PAIRS', '60'))


def maxabs(j: pl.DataFrame, cols, suffix='_ref') -> dict:
    return {c: float(np.abs(j[c].to_numpy().astype('float64')
                            - j[c + suffix].to_numpy().astype('float64')).max()) for c in cols}


def main() -> None:
    t0 = time.time()
    order = json.load(open(MODELS + r'\feature_order.json', encoding='utf-8'))
    lab = PREP.pair_list(PHASE)
    files = PREP.batch_files(PHASE)
    keys_all = pl.concat([pl.read_parquet(f, columns=KEYS) for f in files])
    pids = sorted(keys_all['pair_id'].unique().to_list())
    rng = np.random.default_rng(257)
    pick = [pids[i] for i in sorted(rng.choice(len(pids), NPAIR, replace=False).tolist())]
    sample = keys_all.filter(pl.col('pair_id').is_in(pick))
    print('sampled pairs', len(pick), 'pair-hands', sample.height, flush=True)
    assert sample.height >= 3000, 'sample too small'

    # ---- chronology through the parameterised path
    tabs = (pl.read_parquet(DATA + r'\hands.parquet', columns=['hand_id', 'table_id', 'phase'])
            .filter(pl.col('phase') == PHASE).drop('phase'))
    ktab = sample.join(tabs, on='hand_id').select('table_id').unique()
    hd, s = PREP.stage_a(PHASE, ktab, lab)
    chrono = PREP.chrono_for(sample.join(lab, on='pair_id'), hd, s)
    del hd, s

    # ---- the 662 through the parameterised path
    PCT169, EQ169 = M.load_pf169()
    allc, inhand, chron = M.kept_columns()
    keys = chrono.select(KEYS + ['p1_is_A']).join(lab, on='pair_id', how='left')
    hl = keys['hand_id'].unique().sort().to_list()
    parts = []
    for a0 in range(0, len(hl), 2500):
        ks = keys.join(pl.DataFrame({'hand_id': hl[a0:a0 + 2500]}), on='hand_id', how='semi')
        parts.append(M.build_chunk(ks, PHASE, PCT169, EQ169).select(KEYS + inhand))
        print('  mech', a0 + 2500, '/', len(hl), round(time.time() - t0, 1), 's', flush=True)
    mech = pl.concat(parts).join(chrono.drop('p1_is_A'), on=KEYS, how='left')
    mech = mech.select(KEYS + allc)
    assert mech.height == sample.height

    ref662 = pl.read_parquet(S254 + r'\step254_features.parquet')
    j = mech.join(ref662, on=KEYS, how='inner', validate='1:1', suffix='_ref')
    assert j.height == mech.height, 'gate rows missing from step254_features.parquet'
    g662 = maxabs(j, allc)
    del ref662, j

    # ---- the 25 through STEP251's evaluation route
    from evidence_aug import step246_invisible_features as S246
    pmap = {r[0]: (r[1], r[2]) for r in lab.iter_rows()}
    tmp = SCRATCH + r'\gate_inv_dev.parquet'
    if os.path.exists(tmp):
        os.remove(tmp)
    f25 = S246.build(sample.select(KEYS), pmap, PHASE, tmp, workers=3)
    nm25 = S246.names()
    ref25 = pl.read_parquet(S237 + r'\step237_features.parquet')
    j = f25.join(ref25, on=KEYS, how='inner', validate='1:1', suffix='_ref')
    assert j.height == f25.height
    g25 = maxabs(j, nm25)
    del ref25, j

    # ---- the 163 through the batch-cache lookup used by step257_eval_build
    import step254_eval as S254E
    h, feats = S254E.load()
    assert feats == order['block_163_production']
    f163 = pl.concat([pl.read_parquet(f).join(sample.select('hand_id').unique(), on='hand_id',
                                              how='semi') for f in files])
    f163 = f163.join(sample, on=KEYS, how='semi').drop('table_id')
    lut = f163.select(KEYS).with_row_index('r')
    take = mech.select(KEYS).join(lut, on=KEYS, how='left')
    assert take['r'].null_count() == 0
    X163 = f163.select(feats).to_numpy()[take['r'].to_numpy()]
    ref163 = mech.select(KEYS).join(h.select(KEYS + feats), on=KEYS, how='left', maintain_order='left')
    assert ref163.select(feats).null_count().to_numpy().sum() == 0
    R163 = ref163.select(feats).to_numpy()
    g163 = {c: float(np.abs(X163[:, i].astype('float64') - R163[:, i].astype('float64')).max())
            for i, c in enumerate(feats)}

    # ---- Booster.predict == the saved out-of-fold probabilities
    oof = pl.read_parquet(S254 + r'\hand_oof_step254.parquet')
    o = mech.select(KEYS).join(oof.select(KEYS + ['fold', 'mechplus_p_any', 'mechplus_p_DT',
                                                  'mechplus_p_SP', 'mechplus_p_CI']),
                               on=KEYS, how='left', maintain_order='left')
    X = np.empty((mech.height, 850), np.float32)
    X[:, :163] = X163
    X[:, 163:188] = mech.select(KEYS).join(f25, on=KEYS, how='left',
                                           maintain_order='left').select(nm25).to_numpy()
    for i, c in enumerate(allc):
        X[:, 188 + i] = mech[c].to_numpy()
    fold = o['fold'].to_numpy()
    gpred = {}
    for hnm in ('any', 'DT', 'SP', 'CI'):
        got = np.full(mech.height, np.nan)
        for k in range(5):
            te = fold == k
            if not te.any():
                continue
            b = lgb.Booster(model_file=MODELS + f'\\mechplus_{hnm}_fold{k}.txt')
            got[te] = b.predict(X[te], num_threads=4)
        gpred[hnm] = float(np.abs(got - o[f'mechplus_p_{hnm}'].to_numpy()).max())

    eqcols = [c for c in allc if c.startswith(('t2_eq_', 't2_str_', 't2_hole_pf169', 't2_strgap_'))]
    hard = {c: v for c, v in g662.items() if v > 0 and c not in eqcols}
    soft = {c: v for c, v in g662.items() if v > 0 and c in eqcols}
    res = {'phase': PHASE, 'pairs': len(pick), 'pair_hands': sample.height,
           'max_abs_662': max(g662.values()), 'max_abs_25': max(g25.values()),
           'max_abs_163': max(g163.values()),
           'nonzero_662_strict': hard, 'nonzero_662_float32_equity_tolerance': soft,
           'nonzero_25': {c: v for c, v in g25.items() if v > 0},
           'nonzero_163': {c: v for c, v in g163.items() if v > 0},
           'booster_predict_vs_step254_oof_max_abs': gpred,
           'seconds': round(time.time() - t0, 1)}
    res['G_parity'] = 'PASS' if (not hard and max(soft.values(), default=0.) < 1e-5
                                 and max(g25.values()) == 0.0 and max(g163.values()) == 0.0
                                 and max(gpred.values()) < 1e-9) else 'FAIL'
    json.dump(res, open(OUT + r'\gate_parity257.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'nonzero_163'}, indent=1))


if __name__ == '__main__':
    main()
