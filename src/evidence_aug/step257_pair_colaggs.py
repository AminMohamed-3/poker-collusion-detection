"""STEP257 second pass: per-evaluation-pair aggregates of the 662 mechanical + 25 STEP237 columns.

Requested by the coordinator after the scoring pass had started; the scoring pass partitions by HAND
(one hand's seats / actions / strengths are prepared once and shared by the ~12 pairs that contain
it), so a pair's history is spread over all 20 hand batches and complete per-pair aggregates cannot
be formed inside that loop.  This second pass therefore runs in two stages:

  shard  rebuild the 687 columns hand batch by hand batch (the same code path, same phase assert)
         and write them out regrouped by PAIR batch - the outputs_v2/hands_evaluation batching, in
         which every evaluation pair lies entirely inside one batch;
  agg    per pair batch, aggregate each pair's full evaluation history with STEP247's convention
         (src/pu_pairs/step247_pair_aggregates_new_block.py:agg_block, IMPORTED, not retyped):
         mean, std ddof=1 (zeros when n = 1), max, q95 numpy method="nearest", in that column block
         order, float32, plus pair_id and n_hands.
  gate   the parity check the coordinator asked for: on the development sample of gate_parity257,
         these aggregates through this code path against aggregates taken directly from
         step254_features.parquet / step237_features.parquet.  Required max abs < 1e-4.

Usage: python step257_pair_colaggs.py shard <c0> <c1> | agg <pb0> <pb1> | gate
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import gc
import glob
import json
import sys
import time

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pu_pairs'))
sys.dont_write_bytecode = True

import step247_pair_aggregates_new_block as S247    # noqa: E402  (agg_block, block_names)
import step257_mech_eval as M                       # noqa: E402
import step257_prep_eval as PREP                    # noqa: E402
import step257_eval_build as B                      # noqa: E402

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
CV = ROOT + r'\Claude Version'
OUT = CV + r'\experiments\step257_eval_mech_scores_20260918'
MODELS = OUT + r'\models'
S254 = CV + r'\experiments\step254_mechanical_tensor_20260917'
S237 = CV + r'\experiments\step237_invisible_events_20260917'
SCRATCH = PREP.SCRATCH
SHARDS = SCRATCH + r'\colshards'
KEYS = ['pair_id', 'hand_id']
SUBHANDS = int(os.environ.get('STEP257_SUBHANDS', '2500'))
WORKERS = int(os.environ.get('STEP257_WORKERS', '3'))


def pair_batch_map(phase='evaluation') -> dict:
    path = SCRATCH + f'\\pair_batch_{phase}.parquet'
    if os.path.exists(path):
        d = pl.read_parquet(path)
    else:
        rows = []
        for b, f in enumerate(PREP.batch_files(phase)):
            u = pl.read_parquet(f, columns=['pair_id'])['pair_id'].unique()
            rows.append(pl.DataFrame({'pair_id': u, 'pb': np.full(len(u), b, np.int32)}))
        d = pl.concat(rows)
        assert d['pair_id'].n_unique() == d.height, 'a pair spans two batches'
        d.write_parquet(path)
    return d


def columns_687(order) -> tuple[list[str], list[str]]:
    allc, inhand, chron = M.kept_columns()
    return allc, list(order['block_25_step237'])


def build_687(ks: pl.DataFrame, phase, PCT169, EQ169, chrono: pl.DataFrame,
              f25: pl.DataFrame, allc, nm25) -> pl.DataFrame:
    mech = M.build_chunk(ks, phase, PCT169, EQ169)
    inhand = [c for c in allc if c in set(mech.columns)]
    mech = mech.select(KEYS + inhand)
    mech = mech.join(chrono, on=KEYS, how='left').join(f25, on=KEYS, how='left')
    assert mech.height == ks.height
    out = mech.select(KEYS + allc + nm25)
    assert out.null_count().to_numpy().sum() == 0, 'null in the 687-column block'
    return out


def shard(c0: int, c1: int) -> None:
    phase = 'evaluation'
    os.makedirs(SHARDS, exist_ok=True)
    order = json.load(open(MODELS + r'\feature_order.json', encoding='utf-8'))
    allc, nm25 = columns_687(order)
    PCT169, EQ169 = M.load_pf169()
    hc = B.hand_chunks(phase)
    pbm = pair_batch_map(phase)
    lab = PREP.pair_list(phase)
    from evidence_aug import step246_invisible_features as S246
    pmap = {r[0]: (r[1], r[2]) for r in lab.iter_rows()}
    for c in range(c0, c1):
        if os.path.exists(SHARDS + f'\\done_h{c:03d}.json'):
            print('hand batch', c, 'already sharded', flush=True)
            continue
        t0 = time.time()
        hands = hc.filter(pl.col('chunk') == c).select('hand_id')
        chrono = B.read_filtered(
            sorted(glob.glob(SCRATCH + f'\\chrono_{phase}\\chrono_b*.parquet')), hands)
        keys = chrono.select(KEYS + ['p1_is_A']).join(lab, on='pair_id', how='left')
        chrono = chrono.select(KEYS + [x for x in chrono.columns if x.startswith('t4_')])
        tmp = SCRATCH + f'\\inv2_{phase}_b{c:03d}.parquet'
        f25 = pl.read_parquet(tmp) if os.path.exists(tmp) else \
            S246.build(keys.select(KEYS), pmap, phase, tmp, workers=WORKERS)
        f25 = f25.select(KEYS + nm25)
        hl = hands['hand_id'].to_list()
        buf: dict = {}
        nrow = 0
        flush_k = [0]

        def flush():
            for pb_, gs in buf.items():
                pl.concat(gs).write_parquet(
                    SHARDS + f'\\p{pb_:02d}_h{c:03d}_{flush_k[0]:02d}.parquet')
            buf.clear()
            flush_k[0] += 1
            gc.collect()

        for a0 in range(0, len(hl), SUBHANDS):
            ks = keys.join(pl.DataFrame({'hand_id': hl[a0:a0 + SUBHANDS]}), on='hand_id', how='semi')
            if not ks.height:
                continue
            blk = build_687(ks, phase, PCT169, EQ169, chrono, f25, allc, nm25)
            blk = blk.join(pbm, on='pair_id', how='left')
            assert blk['pb'].null_count() == 0
            for pb, g in blk.partition_by('pb', as_dict=True).items():
                key = pb[0] if isinstance(pb, tuple) else pb
                buf.setdefault(int(key), []).append(g.drop('pb'))
            nrow += blk.height
            del blk
            if (a0 // SUBHANDS) % 4 == 3:
                flush()
            print(f'   shard h{c:03d} hands {a0 + SUBHANDS}/{len(hl)} rows {nrow} '
                  f'{time.time() - t0:.0f}s', flush=True)
        flush()
        os.remove(tmp)
        json.dump({'hand_batch': c, 'rows': nrow, 'seconds': round(time.time() - t0, 1)},
                  open(SHARDS + f'\\done_h{c:03d}.json', 'w', encoding='utf-8'), indent=1)
        print('SHARD', c, nrow, round(time.time() - t0, 1), 's', flush=True)


def agg(pb0: int, pb1: int) -> None:
    order = json.load(open(MODELS + r'\feature_order.json', encoding='utf-8'))
    allc, nm25 = columns_687(order)
    names = allc + nm25
    outnames = S247.block_names(names)
    for pb in range(pb0, pb1):
        fn = OUT + f'\\eval_pair_mech_colaggs_batch_{pb:03d}.parquet'
        if os.path.exists(fn):
            print('pair batch', pb, 'already aggregated', flush=True)
            continue
        t0 = time.time()
        fs = sorted(glob.glob(SHARDS + f'\\p{pb:02d}_h*.parquet'))
        hbs = {os.path.basename(f).split('_h')[1][:3] for f in fs}
        assert len(hbs) == 20, (pb, sorted(hbs))
        d = pl.concat([pl.read_parquet(f) for f in fs]).sort(KEYS)
        X = d.select(names).to_numpy()
        pid = d['pair_id'].to_numpy()
        del d
        gc.collect()
        ids, start = np.unique(pid, return_index=True)
        o = np.argsort(start)
        ids, start = ids[o], start[o]
        ends = np.append(start[1:], len(pid))
        A = np.zeros((len(ids), 4 * len(names)), np.float32)
        nh = np.zeros(len(ids), np.int64)
        for i in range(len(ids)):
            xs = X[start[i]:ends[i]]
            nh[i] = xs.shape[0]
            A[i] = S247.agg_block(xs)
        del X
        gc.collect()
        out = pl.DataFrame({'pair_id': ids, 'n_hands': nh}).hstack(
            pl.DataFrame(A, schema=outnames))
        out.write_parquet(fn)
        print('AGG', pb, out.shape, round(time.time() - t0, 1), 's', flush=True)
        del out, A
        gc.collect()


def gate() -> None:
    """the coordinator's parity check, on the same development sample as gate_parity257."""
    phase = 'development'
    t0 = time.time()
    order = json.load(open(MODELS + r'\feature_order.json', encoding='utf-8'))
    allc, nm25 = columns_687(order)
    names = allc + nm25
    lab = PREP.pair_list(phase)
    keys_all = pl.concat([pl.read_parquet(f, columns=KEYS) for f in PREP.batch_files(phase)])
    pids = sorted(keys_all['pair_id'].unique().to_list())
    rng = np.random.default_rng(257)
    pick = [pids[i] for i in sorted(rng.choice(len(pids), 60, replace=False).tolist())]
    sample = keys_all.filter(pl.col('pair_id').is_in(pick))
    tabs = (pl.read_parquet(DATA + r'\hands.parquet', columns=['hand_id', 'table_id', 'phase'])
            .filter(pl.col('phase') == phase).drop('phase'))
    hd, s = PREP.stage_a(phase, sample.join(tabs, on='hand_id').select('table_id').unique(), lab)
    chrono = PREP.chrono_for(sample.join(lab, on='pair_id'), hd, s)
    keys = chrono.select(KEYS + ['p1_is_A']).join(lab, on='pair_id', how='left')
    chrono = chrono.select(KEYS + [x for x in chrono.columns if x.startswith('t4_')])
    from evidence_aug import step246_invisible_features as S246
    pmap = {r[0]: (r[1], r[2]) for r in lab.iter_rows()}
    tmp = SCRATCH + r'\gate_inv_dev_colagg.parquet'
    if os.path.exists(tmp):
        os.remove(tmp)
    f25 = S246.build(sample.select(KEYS), pmap, phase, tmp, workers=3).select(KEYS + nm25)
    PCT169, EQ169 = M.load_pf169()
    hl = keys['hand_id'].unique().sort().to_list()
    parts = []
    for a0 in range(0, len(hl), 2500):
        ks = keys.join(pl.DataFrame({'hand_id': hl[a0:a0 + 2500]}), on='hand_id', how='semi')
        parts.append(build_687(ks, phase, PCT169, EQ169, chrono, f25, allc, nm25))
    mine = pl.concat(parts).sort(KEYS)

    ref = (pl.read_parquet(S254 + r'\step254_features.parquet')
           .join(pl.read_parquet(S237 + r'\step237_features.parquet'), on=KEYS, how='inner')
           .join(sample, on=KEYS, how='semi').sort(KEYS))
    assert ref.height == mine.height == sample.height

    def blocks(d):
        X = d.select(names).to_numpy()
        pid = d['pair_id'].to_numpy()
        ids, start = np.unique(pid, return_index=True)
        o = np.argsort(start)
        ids, start = ids[o], start[o]
        ends = np.append(start[1:], len(pid))
        A = np.stack([S247.agg_block(X[start[i]:ends[i]]) for i in range(len(ids))])
        return ids, A, np.array([ends[i] - start[i] for i in range(len(ids))])

    i1, A1, n1 = blocks(mine)
    i2, A2, n2 = blocks(ref)
    assert list(i1) == list(i2) and np.array_equal(n1, n2)
    d = np.abs(A1.astype(np.float64) - A2.astype(np.float64))
    onames = S247.block_names(names)
    worst = int(np.unravel_index(np.argmax(d), d.shape)[1])
    res = {'phase': phase, 'pairs': len(i1), 'pair_hands': int(sample.height),
           'n_aggregate_columns': A1.shape[1], 'max_abs': float(d.max()),
           'worst_column': onames[worst],
           'n_columns_with_any_difference': int((d.max(axis=0) > 0).sum()),
           'convention': 'step247_pair_aggregates_new_block.agg_block / block_names, imported',
           'seconds': round(time.time() - t0, 1)}
    res['G_colagg_parity'] = 'PASS' if d.max() < 1e-4 else 'FAIL'
    json.dump(res, open(OUT + r'\gate_colagg_parity257.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'shard':
        shard(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == 'agg':
        agg(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == 'gate':
        gate()
    else:
        raise SystemExit('usage: shard c0 c1 | agg pb0 pb1 | gate')
