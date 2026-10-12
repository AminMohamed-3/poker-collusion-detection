"""STEP259b step 3: re-score the STEP259 fusion scope with the five saved MECHPLUS boosters per head,
keeping BOTH the arithmetic mean (the STEP257 production form, for the parity gate) and the GEOMETRIC
mean sigmoid(mean_k logit(clip(p_k, 1e-7, 1 - 1e-7))) (the STEP266 BUILD_G4 recommendation).

The 850 columns are rebuilt with the STEP257 functions unchanged - step257_eval_build.score_batch's
recipe, re-implemented here because the STEP257 scripts are hash-registered and must not be edited:

    163  outputs_v2/hands_evaluation/batch_*.parquet, semi-joined to the scope keys
     25  evidence_aug.step246_invisible_features.build(keys, pmap, 'evaluation', ...)   [row-local]
    662  step257_mech_eval.build_chunk(ks, 'evaluation', PCT169, EQ169)                 [row-local]
         + the step257_prep_eval chronology block, which is NOT row-local and is therefore taken from
           the chrono_evaluation/chrono_b*.parquet files built over each pair's FULL evaluation hand
           list (rebuild with `python step257_prep_eval.py evaluation` if they are gone).

Everything is evaluation phase for evaluation pairs; build_chunk and step246 assert the phase on every
hand they touch, and those asserts are kept.

Usage:  python step259b_rescore_scope.py prep
        python step259b_rescore_scope.py shard <i> <n>
        python step259b_rescore_scope.py merge <n>
Never uploads, never writes into an existing directory other than the new outputs_step259b_20260918.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # step257_* import each other flat
sys.path.insert(0, os.path.dirname(_HERE))                  # evidence_aug package
sys.path.insert(0, r'C:\Data\Projects\Money Making\work\poker\Codex Resume\src')
sys.dont_write_bytecode = True

import lightgbm as lgb                                      # noqa: E402
import step257_mech_eval as M                               # noqa: E402
import step257_prep_eval as PREP                            # noqa: E402

ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
S257 = CV + r'\experiments\step257_eval_mech_scores_20260918'
MODELS = S257 + r'\models'
OUT = CV + r'\outputs_step259b_20260918'
SCRATCH = PREP.SCRATCH                                      # ...\scratchpad\step257
WORK = (r'C:\Users\USER\AppData\Local\Temp\claude\C--Data-Projects-Money-Making-work-poker'
        r'\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step259b')
KEYS = ['pair_id', 'hand_id']
PHASE = 'evaluation'
SUBHANDS = int(os.environ.get('STEP259B_SUBHANDS', '8000'))
NJ = int(os.environ.get('STEP259B_NJOBS', '4'))
INV_WORKERS = int(os.environ.get('STEP259B_INV_WORKERS', '8'))
HEADS = ('DT', 'SP')


# ----------------------------------------------------------------------------- helpers
def read_filtered_keys(files, keys: pl.DataFrame, columns=None) -> pl.DataFrame:
    """step257_eval_build.read_filtered, semi-joined on (pair_id, hand_id) instead of hand_id."""
    parts = []
    for f in files:
        d = pl.read_parquet(f, columns=columns)
        d = d.join(keys, on=KEYS, how='semi')
        if d.height:
            parts.append(d)
    return pl.concat(parts)


def order_json() -> dict:
    o = json.load(open(MODELS + r'\feature_order.json', encoding='utf-8'))
    assert o['n'] == 850
    return o


def scope_keys() -> pl.DataFrame:
    s = pl.read_parquet(OUT + r'\scope_keys.parquet')
    assert s.height == 321200, s.height
    return s


def shard_of_hands(n: int) -> pl.DataFrame:
    """deterministic, balanced-by-hand assignment of the scope's hands to n shards."""
    hs = scope_keys()['hand_id'].unique().sort()
    return pl.DataFrame({'hand_id': hs, 'shard': (np.arange(len(hs)) % n).astype(np.int32)})


# ----------------------------------------------------------------------------- prep
def prep() -> None:
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    sk = scope_keys().select(KEYS)
    order = order_json()
    lab = PREP.pair_list(PHASE)

    fc = WORK + r'\chrono_scope.parquet'
    if not os.path.exists(fc):
        files = sorted(glob.glob(SCRATCH + f'\\chrono_{PHASE}\\chrono_b*.parquet'))
        assert len(files) == 20, ('chronology files missing - rerun '
                                  '`python step257_prep_eval.py evaluation`', files)
        chrono = read_filtered_keys(files, sk)
        assert chrono.height == sk.height, (chrono.height, sk.height)
        chrono.write_parquet(fc)
        print('chrono_scope', chrono.shape, round(time.time() - t0, 1), 's', flush=True)
        del chrono
        gc.collect()

    f1 = WORK + r'\f163_scope.parquet'
    if not os.path.exists(f1):
        f163 = read_filtered_keys(PREP.batch_files(PHASE), sk).drop('table_id')
        feats163 = [x for x in f163.columns if x not in KEYS]
        assert feats163 == order['block_163_production'], 'the 163-column layout differs'
        assert f163.height == sk.height, (f163.height, sk.height)
        f163.write_parquet(f1)
        print('f163_scope', f163.shape, round(time.time() - t0, 1), 's', flush=True)
        del f163
        gc.collect()

    f2 = WORK + r'\f25_scope.parquet'
    if not os.path.exists(f2):
        from evidence_aug import step246_invisible_features as S246
        pmap = {r[0]: (r[1], r[2]) for r in lab.iter_rows()}
        chrono = pl.read_parquet(fc, columns=KEYS)
        f25 = S246.build(chrono, pmap, PHASE, f2, workers=INV_WORKERS)
        assert S246.names() == order['block_25_step237'], 'the 25-column layout differs'
        assert f25.height == sk.height
        print('f25_scope', f25.shape, round(time.time() - t0, 1), 's', flush=True)
        del f25, chrono
        gc.collect()

    rec = {'rows': sk.height, 'chrono': fc, 'f163': f1, 'f25': f2,
           'subhands': SUBHANDS, 'seconds': round(time.time() - t0, 1)}
    json.dump(rec, open(WORK + r'\prep259b.json', 'w', encoding='utf-8'), indent=1)
    print('PREP', json.dumps(rec), flush=True)


# ----------------------------------------------------------------------------- shard
def shard(i: int, n: int) -> None:
    t0 = time.time()
    fn = WORK + f'\\scores_shard_{i:02d}_of_{n:02d}.parquet'
    if os.path.exists(fn):
        print('shard', i, 'already done', flush=True)
        return
    order = order_json()
    allc, inhand, chron = M.kept_columns()
    assert order['block_662_mechanical'] == allc, 'the 662-column layout differs'
    pos = {nm: j for j, nm in enumerate(order['all'])}
    feats163 = order['block_163_production']
    nm25 = order['block_25_step237']
    assert [pos[x] for x in feats163] == list(range(163))
    assert [pos[x] for x in nm25] == list(range(163, 188))
    PCT169, EQ169 = M.load_pf169()
    bs = {h: [lgb.Booster(model_file=MODELS + f'\\mechplus_{h}_fold{k}.txt') for k in range(5)]
          for h in HEADS}
    for h in HEADS:
        assert all(b.num_feature() == 850 for b in bs[h])

    hmap = shard_of_hands(n).filter(pl.col('shard') == i).select('hand_id')
    chrono = pl.read_parquet(WORK + r'\chrono_scope.parquet').join(hmap, on='hand_id', how='semi')
    f163 = pl.read_parquet(WORK + r'\f163_scope.parquet').join(hmap, on='hand_id', how='semi')
    f25 = pl.read_parquet(WORK + r'\f25_scope.parquet').join(hmap, on='hand_id', how='semi')
    assert chrono.height == f163.height == f25.height, (chrono.height, f163.height, f25.height)
    lab = PREP.pair_list(PHASE)
    keys = chrono.select(KEYS + ['p1_is_A']).join(lab, on='pair_id', how='left')
    assert keys['player_1'].null_count() == 0

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

    hl = hmap['hand_id'].to_list()
    parts, nrow = [], 0
    for a0 in range(0, len(hl), SUBHANDS):
        hsub = pl.DataFrame({'hand_id': hl[a0:a0 + SUBHANDS]})
        ks = keys.join(hsub, on='hand_id', how='semi')
        if not ks.height:
            continue
        mech = M.build_chunk(ks, PHASE, PCT169, EQ169).select(KEYS + inhand)
        mech = mech.join(lut163, on=KEYS, how='left').join(lut25, on=KEYS, how='left') \
                   .join(lutc, on=KEYS, how='left')
        assert mech['r163'].null_count() == 0 and mech['r25'].null_count() == 0 \
            and mech['rc'].null_count() == 0, 'a pair-hand lost its feature row'
        i163 = mech['r163'].to_numpy()
        i25 = mech['r25'].to_numpy()
        ic = mech['rc'].to_numpy()
        m_ = mech.height
        X = np.empty((m_, len(order['all'])), np.float32)
        X[:, :163] = X163[i163]
        X[:, 163:188] = X25[i25]
        for x in inhand:
            X[:, pos[x]] = mech[x].to_numpy()
        for x in chron:
            X[:, pos[x]] = chrono_np[x][ic]
        kdf = mech.select(KEYS)
        del mech
        for j0 in range(0, X.shape[1], 64):
            assert np.isfinite(X[:, j0:j0 + 64]).all(), 'non-finite value in the evaluation matrix'
        cols = {}
        for h in HEADS:
            pk = []
            acc = np.zeros(m_, np.float64)              # step257_eval_build's own accumulation
            for bst in bs[h]:
                pr = bst.predict(X, num_threads=NJ)
                pk.append(pr)
                acc += pr
            arith = (acc / 5.0).astype(np.float32)      # == step257_eval_build's `acc / 5.0`
            q = np.clip(np.stack(pk), 1e-7, 1 - 1e-7)   # == step266 arm G4's geometric mean
            geo = (1.0 / (1.0 + np.exp(-np.log(q / (1 - q)).mean(0)))).astype(np.float32)
            cols[f'p_{h}_arith'] = arith
            cols[f'p_{h}_geo'] = geo
        parts.append(kdf.with_columns([pl.Series(k, v) for k, v in cols.items()]))
        nrow += m_
        del X, kdf
        gc.collect()
        print(f'   s{i:02d} hands {a0 + len(hsub):6d}/{len(hl)}  rows {nrow}  '
              f'{time.time() - t0:.0f}s', flush=True)
    out = pl.concat(parts)
    assert out.height == nrow == len(X163) and out.select(KEYS).n_unique() == nrow
    out.write_parquet(fn)
    r = {'shard': i, 'of': n, 'hands': len(hl), 'rows': out.height,
         'seconds': round(time.time() - t0, 1)}
    json.dump(r, open(WORK + f'\\log_shard_{i:02d}_of_{n:02d}.json', 'w', encoding='utf-8'), indent=1)
    print('SHARD', json.dumps(r), flush=True)


# ----------------------------------------------------------------------------- merge + GATE A
def merge(n: int) -> None:
    t0 = time.time()
    sk = scope_keys()
    parts = []
    for i in range(n):
        fn = WORK + f'\\scores_shard_{i:02d}_of_{n:02d}.parquet'
        assert os.path.exists(fn), fn
        parts.append(pl.read_parquet(fn))
    new = pl.concat(parts).sort(KEYS)
    assert new.height == sk.height == 321200, (new.height, sk.height)
    assert new.select(KEYS).n_unique() == new.height
    assert new.select(KEYS).equals(sk.select(KEYS).sort(KEYS)), 'the re-scored key set is not the scope'

    files = sorted(glob.glob(S257 + r'\eval_hand_scores_batch_*.parquet'))
    assert len(files) == 20, files
    old = read_filtered_keys(files, sk.select(KEYS),
                             columns=KEYS + ['p_any', 'p_DT', 'p_SP', 'p_CI']).sort(KEYS)
    assert old.height == new.height, (old.height, new.height)
    assert old.select(KEYS).equals(new.select(KEYS))

    gate = {}
    for h in HEADS:
        a = new[f'p_{h}_arith'].to_numpy().astype(np.float64)
        b = old[f'p_{h}'].to_numpy().astype(np.float64)
        d = np.abs(a - b)
        gate[h] = {'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
                   'n_exact': int((d == 0).sum()), 'n': int(d.size),
                   'share_exact': round(float((d == 0).mean()), 6),
                   'spearman_like_max_rank_move': None}
    gate['tolerance'] = 1e-5
    gate['PASS'] = all(gate[h]['max_abs_diff'] <= 1e-5 for h in HEADS)

    # descriptive: how far the geometric mean moves the fused rows
    disp = {}
    for h, famcol in (('DT', 'directed_transfer'), ('SP', 'soft_play')):
        m = (sk.sort(KEYS)['predicted_behavior'].to_numpy() == famcol)
        a = new[f'p_{h}_arith'].to_numpy().astype(np.float64)[m]
        g = new[f'p_{h}_geo'].to_numpy().astype(np.float64)[m]
        lo = a <= np.quantile(a, .10)
        disp[h] = {'fused_rows': int(m.sum()),
                   'mean_abs_diff_arith_geo': float(np.abs(a - g).mean()),
                   'median_ratio_low_decile': float(np.median(a[lo] / np.maximum(g[lo], 1e-12))),
                   'spearman_rank_corr': float(np.corrcoef(
                       np.argsort(np.argsort(a)), np.argsort(np.argsort(g)))[0, 1]),
                   'share_geo_below_arith': round(float((g < a).mean()), 6)}

    out = new.join(old.select(KEYS + ['p_any', 'p_CI']), on=KEYS, validate='1:1',
                   maintain_order='left').select(
        KEYS + ['p_DT_arith', 'p_SP_arith', 'p_DT_geo', 'p_SP_geo', 'p_any', 'p_CI'])
    assert out.height == 321200
    fn = OUT + r'\scope_scores_geo.parquet'
    assert not os.path.exists(fn), 'preserve a built file'
    out.write_parquet(fn)
    import hashlib
    rec = {'rows': out.height, 'gate_A': gate, 'arith_vs_geo': disp,
           'output': 'outputs_step259b_20260918/scope_scores_geo.parquet',
           'output_sha256': hashlib.sha256(open(fn, 'rb').read()).hexdigest(),
           'shards': n, 'subhands': SUBHANDS,
           'seconds': round(time.time() - t0, 1)}
    json.dump(rec, open(OUT + r'\gateA259b.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rec, indent=1), flush=True)
    assert gate['PASS'], 'GATE A FAILED: the rebuilt feature matrix does not reproduce STEP257'
    print('GATE A PASS', flush=True)


def main() -> None:
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep()
    elif cmd == 'shard':
        shard(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == 'merge':
        merge(int(sys.argv[2]))
    else:
        raise SystemExit('usage: prep | shard <i> <n> | merge <n>')


if __name__ == '__main__':
    main()
