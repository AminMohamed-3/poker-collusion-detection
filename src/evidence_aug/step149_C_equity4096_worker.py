"""Step 149 (Claude): 256- and 4096-runout eq_* fields for explicit (pair_id, hand_id, player_1, player_2) keys, in one pass.
The root equity_evidence_features.py is loaded unchanged; only its preflop branch is replaced by a function that walks the first 4096 rows of
the production sample generator (whose first 256 rows are the production 256) and returns the 256 or 4096 mean. Postflop stays the exact
original branch. Run as a script so the process pool imports only this light module:
python step149_C_equity4096_worker.py --input keys.parquet --out256 a.parquet --out4096 b.parquet --receipt r.json --procs 14"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
ORIGINAL, ORIGINAL_SHA = POKER / 'src/equity_evidence_features.py', '9ff36fdee8f965fb32fba111a53fc006382141f62e2fb27320937a2d3f6678f8'
HIGH, BASE = 4096, 256
KEYS = ['pair_id', 'hand_id']
M, MODE = None, {'n': BASE}


def load():
    spec = importlib.util.spec_from_file_location('step149_original_equity', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PREFLOP_SAMPLES == BASE
    idx = np.argsort(np.random.default_rng(module.SAMPLE_SEED).random((HIGH, 48)), axis=1)[:, :5]
    assert np.array_equal(idx[:BASE], module._SAMPLE_INDICES), 'the 4096 generator does not keep the production 256 prefix'
    rows = [tuple(int(i) for i in r) for r in idx]
    exact, preflop, evaluate = module._canonical_equity, {}, module.evaluate_cards

    def canonical(first, second, board):
        if board:
            return exact(first, second, board)
        if (first, second) not in preflop:
            known = first + second
            if len(set(known)) != len(known):
                raise ValueError('Invalid or duplicated cards')
            remaining = tuple(c for c in range(52) if c not in known)
            points, low = 0., None
            for k, row in enumerate(rows):
                if k == BASE:
                    low = points / BASE
                runout = tuple(remaining[i] for i in row)
                a = evaluate(*first, *runout)
                b = evaluate(*second, *runout)
                points += float(a < b) + .5 * float(a == b)
            preflop[(first, second)] = (low, points / HIGH)
        return preflop[(first, second)][0 if MODE['n'] == BASE else 1]

    module._canonical_equity = canonical
    return module


def init():
    global M
    M = load()


def task(chunk):
    out = []
    for pair, hand, first, second, blind, board, seats, actions in chunk:
        MODE['n'] = BASE
        low = M.extract_pair_hand(first, second, blind, seats, board, actions)
        MODE['n'] = HIGH
        high = M.extract_pair_hand(first, second, blind, seats, board, actions)
        out.append((pair, hand, low, high))
    return out


def swap_check(module):
    c = module.CARDS
    seats = {'a': (c['As'], c['Ad']), 'b': (c['Kc'], c['Kd'])}
    acts = [{'action_no': 0, 'player_id': 'a', 'action': 'raise', 'amount': 6, 'to_call': 2, 'pot_before': 3, 'stack_before': 100, 'street': 'preflop'},
            {'action_no': 1, 'player_id': 'b', 'action': 'call', 'amount': 4, 'to_call': 4, 'pot_before': 9, 'stack_before': 100, 'street': 'preflop'}]
    for n in (BASE, HIGH):
        MODE['n'] = n
        assert module.heads_up_equity(seats['a'], seats['b'], []) + module.heads_up_equity(seats['b'], seats['a'], []) == 1.0
        assert module.extract_pair_hand('a', 'b', 2, seats, [], acts) == module.extract_pair_hand('b', 'a', 2, seats, [], acts[::-1])


def main():
    ap = argparse.ArgumentParser()
    for a in ('--input', '--out256', '--out4096', '--receipt'):
        ap.add_argument(a, required=True)
    ap.add_argument('--procs', type=int, default=14)
    args = ap.parse_args()
    t0 = time.perf_counter()
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
    module = load()
    swap_check(module)
    keys = pl.read_parquet(args.input).select(KEYS + ['player_1', 'player_2']).sort(KEYS)
    assert keys.select(KEYS).n_unique() == keys.height
    hids = pl.DataFrame({'hand_id': keys['hand_id'].unique()}).lazy()
    hands = pl.scan_parquet(POKER / 'data/hands.parquet').join(hids, on='hand_id').select('hand_id', 'big_blind', 'board_cards').collect()
    seats = pl.scan_parquet(POKER / 'data/seats.parquet').join(hids, on='hand_id').select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2').collect()
    actions = (pl.scan_parquet(POKER / 'data/actions.parquet').join(hids, on='hand_id')
               .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call', 'pot_before', 'stack_before').collect().sort('hand_id', 'action_no'))
    assert hands.height == hids.collect().height
    hl = {h: (bb, [module.CARDS[c] for c in (b or '').split()]) for h, bb, b in hands.iter_rows()}
    sl, al = {}, {}
    for h, p, a, b in seats.iter_rows():
        sl.setdefault(h, {})[p] = (module.CARDS[a], module.CARDS[b])
    for r in actions.iter_rows(named=True):
        al.setdefault(r.pop('hand_id'), []).append(r)
    items = [(p, h, a, b, hl[h][0], hl[h][1], sl[h], al.get(h, [])) for p, h, a, b in keys.select(KEYS + ['player_1', 'player_2']).iter_rows()]
    items.sort(key=lambda x: x[1])
    size = max(1, len(items) // (args.procs * 40))
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    print(json.dumps({'rows': len(items), 'chunks': len(chunks), 'prepared_s': round(time.perf_counter() - t0, 1)}), flush=True)
    rows, done = [], 0
    with Pool(args.procs, initializer=init) as pool:
        for part in pool.imap_unordered(task, chunks):
            rows += part
            done += 1
            if done % max(1, len(chunks) // 10) == 0:
                print(json.dumps({'chunks_done': done, 'rows': len(rows), 's': round(time.perf_counter() - t0, 1)}), flush=True)
    names = list(module.FEATURES)
    out = {}
    for tag, j in (('256', 2), ('4096', 3)):
        f = pl.DataFrame([(r[0], r[1], *r[j]) for r in rows], schema=KEYS + names, orient='row').with_columns(pl.col(names).cast(pl.Float32)).sort(KEYS)
        assert f.select(KEYS).equals(keys.select(KEYS))
        out[tag] = f
    out['256'].write_parquet(args.out256)
    out['4096'].write_parquet(args.out4096)
    diff = np.abs(out['4096'].select(names).to_numpy().astype(float) - out['256'].select(names).to_numpy().astype(float))
    receipt = {'rows': len(rows), 'procs': args.procs, 'equity_source_sha256': ORIGINAL_SHA, 'prefix_256_of_4096_verified': True, 'swap_checks': True,
               'out256_sha256': hashlib.sha256(Path(args.out256).read_bytes()).hexdigest(), 'out4096_sha256': hashlib.sha256(Path(args.out4096).read_bytes()).hexdigest(),
               'rows_any_field_changed': int((diff.max(axis=1) > 0).sum()), 'mean_abs_change_by_field': dict(zip(names, np.round(diff.mean(axis=0), 6).tolist())),
               'seconds': round(time.perf_counter() - t0, 1)}
    Path(args.receipt).write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('rows', 'rows_any_field_changed', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
