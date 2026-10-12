"""STEP280 stage 2: replay real honest hands through the engine.

Two jobs at once:
  (a) VALIDATE the engine - every recorded action must be legal under our rules and the
      engine's derived pot_before / stack_before / to_call / players_active and the hand's
      final_pot / board / showdown / won_share must reproduce the recorded values exactly;
  (b) EMIT the decision dataset the policy is fitted on, through the same code path that
      will later sample actions.
"""
from __future__ import annotations

import argparse
import json
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
HS_K = 128


class Mismatch(Exception):
    pass


def replay_hand(hrow, seats, acts, pf_table, rng, collect):
    """Replay one recorded hand. Returns (rows, targets, report)."""
    n = 6
    seats = sorted(seats, key=lambda r: r['seat_no'])
    assert [r['seat_no'] for r in seats] == list(range(n))
    hole = [(S.CARD_ID[r['hole_card_1']], S.CARD_ID[r['hole_card_2']]) for r in seats]
    seat_of = {r['player_id']: r['seat_no'] for r in seats}
    rec_board = [S.CARD_ID[c] for c in (hrow['board_cards'] or '').split()]
    used = set(rec_board) | {c for p in hole for c in p}
    pad = [c for c in range(52) if c not in used]
    board_full = rec_board + pad[:5 - len(rec_board)]

    hs_cache = {}

    def hs_for(street, seat):
        if street == 0:
            return 0.0
        if street not in hs_cache:
            b = board_full[:S.BOARD_LEN[street]]
            hs_cache[street] = S.street_hs(b, hole, range(n), rng, K=HS_K)
        return hs_cache[street][seat]

    rows, tgt_a, tgt_s, tgt_f, ctxs, masks = [], [], [], [], [], []
    it = iter(acts)
    problems = []

    def decide(st, legal, ctx):
        r = next(it)
        seat = seat_of[r['player_id']]
        if seat != st['seat']:
            raise Mismatch(f'seat order: recorded {seat} engine {st["seat"]}')
        for key, col in [('pot', 'pot_before'), ('stack', 'stack_before'),
                         ('to_call', 'to_call'), ('players_live', 'players_active')]:
            if st[key] != r[col]:
                raise Mismatch(f'{col}: recorded {r[col]} engine {st[key]} '
                               f'(street {st["street"]} seat {seat})')
        if S.STREET_IDX[r['street']] != st['street']:
            raise Mismatch(f'street: recorded {r["street"]} engine {st["street"]}')
        act = 'raise' if r['action'] == 'bet' else r['action']
        if act not in legal:
            raise Mismatch(f'illegal by our rules: {r["action"]} not in {legal}')
        if collect:
            hs = hs_for(st['street'], st['seat'])
            pf = pf_table[S.preflop_bucket(*ctx['hole'])]
            rows.append(S.state_features(st, ctx['hole'], ctx['board'], hs, pf))
            tgt_a.append(S.CLASS_IDX[act if r['action'] != 'bet' else 'raise'])
            if r['action'] in ('bet', 'raise'):
                frac = r['amount'] / max(r['pot_before'], 1)
                tgt_s.append(int(np.searchsorted(S.SIZE_EDGES, frac, side='right') - 1))
                tgt_f.append(frac)
            else:
                tgt_s.append(-1)
                tgt_f.append(-1.0)
            ctxs.append((st['street'], 1 if st['to_call'] > 0 else 0))
            masks.append(sum(1 << S.CLASS_IDX[x] for x in legal))
        return act, r['amount_to']

    out = S.run_hand(
        {'button_seat': hrow['button_seat'], 'small_blind': hrow['small_blind'],
         'big_blind': hrow['big_blind'],
         'starting_stack': [r['starting_stack'] for r in seats]},
        decide, cards={'hole': hole, 'board': board_full})
    leftover = sum(1 for _ in it)
    if leftover:
        problems.append(f'{leftover} recorded actions unconsumed')
    if out['final_pot'] != hrow['final_pot']:
        problems.append(f'final_pot {out["final_pot"]} vs {hrow["final_pot"]}')
    if out['board_cards'] != (hrow['board_cards'] or ''):
        problems.append(f'board {out["board_cards"]!r} vs {hrow["board_cards"]!r}')
    if out['players_at_showdown'] != hrow['players_at_showdown']:
        problems.append(f'showdown {out["players_at_showdown"]} vs {hrow["players_at_showdown"]}')
    for r in seats:
        i = r['seat_no']
        if bool(out['folded'][i]) != bool(r['folded']):
            problems.append(f'folded seat {i}')
        if out['contrib'][i] != r['total_contribution']:
            problems.append(f'contrib seat {i}: {out["contrib"][i]} vs {r["total_contribution"]}')
        share = out['won'][i] / out['final_pot'] if out['final_pot'] else 0.0
        if abs(share - r['won_share']) > 2e-3:
            problems.append(f'won_share seat {i}: {share:.4f} vs {r["won_share"]:.4f}')
    return rows, tgt_a, tgt_s, tgt_f, ctxs, masks, problems


def run(groups, limit=None, seed=280031, out_prefix='dec'):
    t0 = time.time()
    pf_table = S.build_preflop_equity()
    print(f'preflop table ready ({time.time() - t0:.0f}s)', flush=True)
    sel = pl.read_parquet(SCRATCH / 'sel_hands.parquet').filter(pl.col('grp').is_in(groups))
    if limit:
        sel = sel.head(limit)
    keep = set(sel['hand_id'].to_list())
    seats = pl.read_parquet(SCRATCH / 'sel_seats.parquet').filter(pl.col('hand_id').is_in(keep))
    acts = (pl.read_parquet(SCRATCH / 'sel_actions.parquet')
            .filter(pl.col('hand_id').is_in(keep)).sort('hand_id', 'action_no'))
    sm, am = defaultdict(list), defaultdict(list)
    for r in seats.iter_rows(named=True):
        sm[r['hand_id']].append(r)
    for r in acts.iter_rows(named=True):
        am[r['hand_id']].append(r)
    print(f'loaded {sel.height} hands ({time.time() - t0:.0f}s)', flush=True)

    X, YA, YS, YF, TAB, GRP, CTX, MK = [], [], [], [], [], [], [], []
    bad, nbad, examples = 0, 0, []
    rng = np.random.default_rng(seed)
    for i, h in enumerate(sel.iter_rows(named=True)):
        try:
            rows, ta, ts, tf, cx, mk, problems = replay_hand(h, sm[h['hand_id']], am[h['hand_id']],
                                                             pf_table, rng, True)
        except (Mismatch, S.IllegalAction, StopIteration) as exc:
            bad += 1
            if len(examples) < 8:
                examples.append(f'{h["hand_id"]}: {type(exc).__name__}: {exc}')
            continue
        if problems:
            nbad += 1
            if len(examples) < 8:
                examples.append(f'{h["hand_id"]}: {problems[:3]}')
            continue
        X.extend(rows); YA.extend(ta); YS.extend(ts); YF.extend(tf); CTX.extend(cx)
        MK.extend(mk)
        TAB.extend([h['table_id']] * len(rows)); GRP.extend([h['grp']] * len(rows))
        if (i + 1) % 20000 == 0:
            print(f'  {i + 1}/{sel.height} hands, {len(X)} decisions, '
                  f'{bad} illegal, {nbad} outcome-mismatch ({time.time() - t0:.0f}s)', flush=True)
    X = np.asarray(X, dtype=np.float32)
    frame = pl.DataFrame({n: X[:, j] for j, n in enumerate(S.FEAT_NAMES)}).with_columns(
        pl.Series('y_action', np.asarray(YA, dtype=np.int8)),
        pl.Series('y_size', np.asarray(YS, dtype=np.int8)),
        pl.Series('y_frac', np.asarray(YF, dtype=np.float32)),
        pl.Series('table_id', TAB), pl.Series('grp', GRP),
        pl.Series('ctx_street', np.asarray([c[0] for c in CTX], dtype=np.int8)),
        pl.Series('ctx_facing', np.asarray([c[1] for c in CTX], dtype=np.int8)),
        pl.Series('legal_mask', np.asarray(MK, dtype=np.int8)),
    )
    frame.write_parquet(SCRATCH / f'{out_prefix}_{"_".join(groups)}.parquet')
    report = {
        'hands_attempted': sel.height, 'hands_illegal_under_our_rules': bad,
        'hands_outcome_mismatch': nbad,
        'hands_reproduced_exactly': sel.height - bad - nbad,
        'exact_rate': round((sel.height - bad - nbad) / max(sel.height, 1), 6),
        'decisions': int(frame.height), 'examples': examples,
        'seconds': round(time.time() - t0, 1), 'HS_K': HS_K, 'seed': seed,
    }
    print(json.dumps(report, indent=1), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--groups', nargs='+', default=['POLICY_TRAIN', 'POLICY_HELD'])
    p.add_argument('--limit', type=int)
    p.add_argument('--out', default='dec')
    p.add_argument('--report')
    a = p.parse_args()
    rep = run(a.groups, a.limit, out_prefix=a.out)
    if a.report:
        (HERE / a.report).write_text(json.dumps(rep, indent=1), encoding='utf-8')
