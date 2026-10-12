"""STEP280 stage 4: roll out whole hands under the fitted honest policy.

Each simulated hand copies a real honest TWO_SAMPLE hand's table, button, blinds, seat
occupancy and starting stacks, deals a fresh 52-card shuffle, and plays every seat with
the population honest policy. Hands are advanced in lockstep so the two LightGBM heads
are called on batches rather than single rows.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_lib as S
from prep_data import SCRATCH

HERE = Path(__file__).resolve().parent
BATCH = 512
HS_K = 128
SEED = 280057


def features_for(st, ctx, rng, pf_table):
    if st['street'] == 0:
        hs = 0.0
    else:
        key = ('hs', st['street'])
        cache = ctx['cache']
        if key not in cache:
            cache[key] = S.street_hs(ctx['board'], ctx['all_hole'], range(6), rng, K=HS_K)
        hs = cache[key][st['seat']]
    return S.state_features(st, ctx['hole'], ctx['board'], hs,
                            pf_table[S.preflop_bucket(*ctx['hole'])])


def main(limit=None, seed=SEED, tag=''):
    t0 = time.time()
    pf_table = S.build_preflop_equity()
    with open(SCRATCH / 'policy.pkl', 'rb') as fh:
        pol = pickle.load(fh)
    models_a, models_s, pools = pol['action'], pol['size'], pol['pools']
    pool_keys = {k: v for k, v in pools.items()}

    sel = pl.read_parquet(SCRATCH / 'sel_hands.parquet').filter(pl.col('grp') == 'TWO_SAMPLE')
    if limit:
        sel = sel.head(limit)
    keep = set(sel['hand_id'].to_list())
    seats = pl.read_parquet(SCRATCH / 'sel_seats.parquet').filter(pl.col('hand_id').is_in(keep))
    sm = defaultdict(list)
    for r in seats.iter_rows(named=True):
        sm[r['hand_id']].append(r)
    cfgs = []
    for h in sel.iter_rows(named=True):
        rows = sorted(sm[h['hand_id']], key=lambda r: r['seat_no'])
        cfgs.append((h, rows))
    print(f'{len(cfgs)} hand configurations ({time.time() - t0:.0f}s)', flush=True)

    rng = np.random.default_rng(seed)
    H, SE, AC = [], [], []
    ptr = 0
    active = []      # list of [gen, request, hrow, seatrows, sim_hand_id]
    done = 0
    illegal = 0

    def launch():
        nonlocal ptr
        while len(active) < BATCH and ptr < len(cfgs):
            h, rows = cfgs[ptr]
            ptr += 1
            g = S.hand_gen({'button_seat': h['button_seat'], 'small_blind': h['small_blind'],
                            'big_blind': h['big_blind'],
                            'starting_stack': [r['starting_stack'] for r in rows]}, rng=rng)
            try:
                req = next(g)
            except StopIteration as stop:
                finish(stop.value, h, rows)
                continue
            active.append([g, req, h, rows, 'S' + h['hand_id'][1:]])

    def finish(out, h, rows):
        nonlocal done
        done += 1
        hid = 'S' + h['hand_id'][1:]
        H.append({'hand_id': hid, 'table_id': h['table_id'], 'button_seat': out['button_seat'],
                  'small_blind': out['small_blind'], 'big_blind': out['big_blind'],
                  'board_cards': out['board_cards'], 'final_pot': out['final_pot'],
                  'players_dealt': 6, 'players_at_showdown': out['players_at_showdown']})
        sd = set(out['showdown_seats'])
        for i, r in enumerate(rows):
            SE.append({'hand_id': hid, 'player_id': r['player_id'], 'seat_no': i,
                       'starting_stack': out['start_stacks'][i],
                       'hole_card_1': S.ID_CARD[out['hole'][i][0]],
                       'hole_card_2': S.ID_CARD[out['hole'][i][1]],
                       'total_contribution': out['contrib'][i],
                       'net_chips': out['won'][i] - out['contrib'][i],
                       'folded': bool(out['folded'][i]), 'went_to_showdown': i in sd,
                       'won_share': (out['won'][i] / out['final_pot']) if out['final_pot'] else 0.})
        for a in out['actions']:
            act = a['action']
            if act == 'raise' and a['to_call'] == 0 and a['street'] > 0:
                act = 'bet'
            AC.append({'hand_id': hid, 'action_no': a['action_no'],
                       'street': S.STREETS[a['street']],
                       'player_id': rows[a['seat']]['player_id'], 'action': act,
                       'amount': a['amount'], 'amount_to': a['amount_to'],
                       'pot_before': a['pot_before'], 'stack_before': a['stack_before'],
                       'to_call': a['to_call'], 'players_active': a['players_active']})

    launch()
    while active:
        feats = np.asarray([features_for(e[1][0], e[1][2], rng, pf_table) for e in active],
                           dtype=np.float32)
        pa = np.zeros((len(active), 5))
        for m in models_a:
            pa += m.predict(feats) / len(models_a)
        mask = np.zeros_like(pa)
        for i, e in enumerate(active):
            for a in e[1][1]:
                mask[i, S.CLASS_IDX[a]] = 1.0
        pa = pa * mask
        pa /= np.clip(pa.sum(1, keepdims=True), 1e-12, None)
        u = rng.random(len(active))
        choice = (pa.cumsum(1) < u[:, None]).sum(1).clip(0, 4)
        need_size = [i for i in range(len(active)) if S.ACTION_CLASSES[choice[i]] == 'raise']
        sizes = {}
        if need_size:
            fs = feats[need_size]
            ps = np.zeros((len(need_size), S.N_SIZE))
            for m in models_s:
                ps += m.predict(fs) / len(models_s)
            uu = rng.random(len(need_size))
            bidx = (ps.cumsum(1) < uu[:, None]).sum(1).clip(0, S.N_SIZE - 1)
            for j, i in enumerate(need_size):
                st = active[i][1][0]
                key = f"{st['street']}_{1 if st['to_call'] > 0 else 0}_{bidx[j]}"
                pool = pool_keys.get(key)
                if pool is None or len(pool) == 0:
                    frac = (S.SIZE_EDGES[bidx[j]] + min(S.SIZE_EDGES[bidx[j] + 1], 5.0)) / 2
                else:
                    frac = float(pool[rng.integers(len(pool))])
                amt = int(round(frac * max(st['pot'], 1)))
                sizes[i] = st['street_committed'] + max(amt, 1)
        nxt = []
        for i, e in enumerate(active):
            act = S.ACTION_CLASSES[choice[i]]
            try:
                req = e[0].send((act, sizes.get(i)))
                e[1] = req
                nxt.append(e)
            except StopIteration as stop:
                finish(stop.value, e[2], e[3])
            except S.IllegalAction:
                illegal += 1
        active[:] = nxt
        launch()
        if done and done % 10000 < BATCH and len(active) == BATCH:
            print(f'  {done}/{len(cfgs)} hands ({time.time() - t0:.0f}s)', flush=True)

    suffix = f'_{tag}' if tag else ''
    pl.DataFrame(H).write_parquet(SCRATCH / f'sim_hands{suffix}.parquet')
    pl.DataFrame(SE).write_parquet(SCRATCH / f'sim_seats{suffix}.parquet')
    pl.DataFrame(AC).write_parquet(SCRATCH / f'sim_actions{suffix}.parquet')
    rep = {'hands_requested': len(cfgs), 'hands_emitted': done, 'illegal_aborts': illegal,
           'actions': len(AC), 'seed': seed, 'batch': BATCH, 'HS_K': HS_K,
           'seconds': round(time.time() - t0, 1)}
    (HERE / f'simulate_report{suffix}.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    main(a.limit, a.seed, a.tag)
