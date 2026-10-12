"""STEP193: build per-pair gameplay packets for LLM rule induction on the LISTING convention.

The data lever is the biggest one measured in this project (+.034 decoded per doubling of
labelled lists, STEP133 = +0.61 % composite) and every branch of it is shut, all for the same
reason: we can generate hands but we cannot label them the way the tournament does. STEP115
measured the best explicit listing rule at DT .67/.67, SP .57/.54, CI .27/.27 precision/recall.

So the task worth giving a strong LLM is NOT labelling. It is INDUCING the annotators' listing
rule from the 372 development pairs, where the ground truth is visible. That is verifiable
(hold pairs out and measure against STEP115's baseline), small (CI is 92 pairs), and if it
succeeds it is what unlocks labelling generated hands.

This script emits, per labelled target pair, a chronological narrative of every hand the two
players shared, each tagged LISTED(rank k) or UNLISTED, plus a manifest naming the withheld
pairs.

HARD CONSTRAINTS honoured here:
  * players.parquet is NEVER opened. Player metadata (account age, region, stake, client) is
    off-limits, and the induced rule has to be computable from gameplay alone anyway.
  * player_ids are replaced by seat-derived aliases P1 / P2 / O1..On per hand, so no identity
    is carried across hands or out of the packet.
  * development phase only. No evaluation pair, and no cross-phase linkage.

Usage: python step193_C_rule_induction_packets.py FAMILY HOLDOUT_FRACTION OUTDIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
DATA = POKER / 'data'
STREETS = ['preflop', 'flop', 'turn', 'river']

# BURNED: experiments/label_process_20260911/qualitative.txt dumps these nine pairs, three per
# family, WITH their per-hand LISTED/unlisted tags. Any agent that has ever had this working
# directory has seen their ground truth, so they can never serve as held-out evidence. They stay
# in the TRAINING half -- their labels are already public to such an agent, so showing them costs
# nothing -- but they are force-excluded from the holdout. An OpenCode session on 2026-09-16 was
# handed the induction task, could not see any packets, and answered from these artefacts plus the
# aggregate audit statistics instead; that is what made the exclusion necessary rather than tidy.
BURNED = {'PBE03EF8D984E', 'P9A6A6BB5C30D', 'P029F9BC3A7A4',
          'PA4A4744909BA', 'P83DF23CBD312', 'P3FC221188111',
          'P85CFE95257D0', 'P442C0ACBF0B9', 'P6350879DBF8F'}


def detz(df):
    """Drop time zones before materialising rows.

    polars converts a tz-aware Datetime to a python object through zoneinfo, and this venv has no
    tzdata, so iter_rows panics with ZoneInfoNotFoundError('UTC'). started_at is only ever used to
    order hands within a pair, and that order is unchanged by dropping the zone.
    """
    tz = [c for c, t in df.schema.items() if isinstance(t, pl.Datetime) and t.time_zone is not None]
    return df.with_columns([pl.col(c).dt.replace_time_zone(None) for c in tz]) if tz else df


def card(a, b):
    return f'[{a} {b}]' if a and b else '[--]'


def fmt_hand(h, seats, acts, alias, listed_rank, idx, n):
    L = []
    tag = f'LISTED(rank {listed_rank})' if listed_rank else 'UNLISTED'
    L.append(f'--- HAND {idx + 1} of {n}   {tag}')
    L.append(f'    blinds {h["small_blind"]}/{h["big_blind"]}  button_seat {h["button_seat"]}  '
             f'dealt {h["players_dealt"]}  showdown {h["players_at_showdown"]}  final_pot {h["final_pot"]}')
    if h['board_cards']:
        L.append(f'    board {h["board_cards"]}')
    row = []
    for s in seats.iter_rows(named=True):
        who = alias[s['player_id']]
        row.append(f'{who}(seat {s["seat_no"]}, stack {s["starting_stack"]}) '
                   f'{card(s["hole_card_1"], s["hole_card_2"])} '
                   f'put_in {s["total_contribution"]} net {s["net_chips"]}'
                   + (' SHOWDOWN' if s['went_to_showdown'] else '')
                   + (' folded' if s['folded'] else ''))
    L.append('    seats: ' + ' | '.join(row))
    for st in STREETS:
        a = acts.filter(pl.col('street') == st)
        if not a.height:
            continue
        parts = []
        for r in a.iter_rows(named=True):
            who = alias[r['player_id']]
            act = r['action']
            if act in ('bet', 'raise', 'call', 'all_in', 'post'):
                parts.append(f'{who} {act} {r["amount"]}'
                             + (f'->{r["amount_to"]}' if r['amount_to'] not in (None, r['amount']) else '')
                             + f' (pot {r["pot_before"]}, to_call {r["to_call"]}, stack {r["stack_before"]})')
            else:
                parts.append(f'{who} {act} (pot {r["pot_before"]}, to_call {r["to_call"]})')
        L.append(f'    {st}: ' + '; '.join(parts))
    return '\n'.join(L)


def hand_record(h, seats, acts, alias, listed_rank, idx):
    """The same facts fmt_hand narrates, as fields a rule can be written against.

    The narrative is for reading; this is for computing. A rule induced by eyeballing ten packets
    is worth little -- the point of emitting both is that a hypothesis formed on the prose can be
    tested across all training pairs at once.
    """
    seat_rows, acts_out = [], []
    for s in seats.iter_rows(named=True):
        seat_rows.append({'who': alias[s['player_id']], 'seat_no': s['seat_no'],
                          'starting_stack': s['starting_stack'],
                          'hole': [s['hole_card_1'], s['hole_card_2']],
                          'total_contribution': s['total_contribution'], 'net_chips': s['net_chips'],
                          'went_to_showdown': bool(s['went_to_showdown']), 'folded': bool(s['folded'])})
    for r in acts.iter_rows(named=True):
        acts_out.append({'street': r['street'], 'who': alias[r['player_id']], 'action': r['action'],
                         'amount': r['amount'], 'amount_to': r['amount_to'],
                         'pot_before': r['pot_before'], 'to_call': r['to_call'],
                         'stack_before': r['stack_before']})
    return {'hand_index': idx, 'chrono_frac': None, 'listed_rank': listed_rank,
            'small_blind': h['small_blind'], 'big_blind': h['big_blind'],
            'button_seat': h['button_seat'], 'players_dealt': h['players_dealt'],
            'players_at_showdown': h['players_at_showdown'], 'final_pot': h['final_pot'],
            'board_cards': h['board_cards'], 'seats': seat_rows, 'actions': acts_out}


def main():
    family, holdout, outdir = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    lab = pl.read_csv(DATA / 'development_labels.csv')
    ev = pl.read_csv(DATA / 'development_evidence.csv')
    tgt = lab.filter((pl.col('label') == 1) & (pl.col('behavior_family') == family))
    assert tgt.height, f'no labelled targets for {family}'
    pids = set(tgt['player_1']) | set(tgt['player_2'])
    print(f'{family}: {tgt.height} labelled target pairs, {len(pids)} distinct players', flush=True)

    # scan rather than read: seats is 94 MB and actions 150 MB, and this may run alongside a fit
    seats = (pl.scan_parquet(DATA / 'seats.parquet')
             .filter(pl.col('player_id').is_in(list(pids)))
             .select('hand_id', 'player_id').collect())
    hands_of = {p: set(g['hand_id']) for (p,), g in seats.group_by('player_id')}
    pair_hands = {}
    for r in tgt.iter_rows(named=True):
        shared = hands_of.get(r['player_1'], set()) & hands_of.get(r['player_2'], set())
        pair_hands[r['pair_id']] = shared
    allh = sorted(set().union(*pair_hands.values()))
    print(f'{len(allh)} distinct shared hands; median per pair '
          f'{int(np.median([len(v) for v in pair_hands.values()]))}', flush=True)

    H = detz(pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(allh)).collect())
    S = detz(pl.scan_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(allh)).collect())
    A = detz(pl.scan_parquet(DATA / 'actions.parquet').filter(pl.col('hand_id').is_in(allh))
             .sort('hand_id', 'action_no').collect())
    print(f'loaded {H.height} hands, {S.height} seat rows, {A.height} actions', flush=True)

    hmap = {r['hand_id']: r for r in H.iter_rows(named=True)}
    smap = {k[0]: g for k, g in S.sort('seat_no').group_by('hand_id')}
    amap = {k[0]: g for k, g in A.group_by('hand_id')}
    evmap = {}
    for r in ev.iter_rows(named=True):
        evmap.setdefault(r['pair_id'], {})[r['hand_id']] = r['evidence_rank']

    rng = np.random.default_rng(1930)
    order = sorted(pair_hands)
    # The holdout is drawn only from pairs whose labels have never been exposed. BURNED pairs stay
    # in the packet set as TRAINING -- withholding a tag an agent can already look up would measure
    # nothing -- but they can never score the rule.
    eligible = [p for p in order if p not in BURNED]
    burned_here = [p for p in order if p in BURNED]
    n_held = max(1, int(round(len(order) * holdout)))
    assert n_held <= len(eligible), 'holdout larger than the unexposed pool'
    held = set(rng.choice(eligible, size=n_held, replace=False).tolist())
    manifest = {'family': family, 'n_pairs': len(order), 'held_out': sorted(held),
                'holdout_fraction': holdout, 'rng_seed': 1930,
                'excluded_from_holdout_as_burned': burned_here,
                'burned_reason': 'ground truth for these pairs is printed in '
                                 'experiments/label_process_20260911/qualitative.txt, so any agent '
                                 'with repo access has already seen their labels',
                'baseline_step115_precision_recall': {'directed_transfer': [0.67, 0.67],
                                                      'soft_play': [0.57, 0.54],
                                                      'coordinated_isolation': [0.27, 0.27]}[family],
                'note': 'players.parquet was never opened; aliases are per-hand and carry no identity across hands'}

    for pid in order:
        hs = sorted(pair_hands[pid], key=lambda h: (hmap[h]['started_at'], h))
        r = tgt.filter(pl.col('pair_id') == pid).row(0, named=True)
        lines = [f'PAIR {pid}   family {family}   {"WITHHELD" if pid in held else "TRAINING"}',
                 f'{len(hs)} shared hands, chronological. '
                 + (f'{len(evmap.get(pid, {}))} of them were LISTED as evidence by the annotators.'
                    if pid not in held else
                    'How many were LISTED is withheld along with which ones.'), '']
        recs = []
        for i, hid in enumerate(hs):
            seats_h = smap[hid]
            alias, o = {}, 0
            for s in seats_h.iter_rows(named=True):
                p = s['player_id']
                if p == r['player_1']:
                    alias[p] = 'P1'
                elif p == r['player_2']:
                    alias[p] = 'P2'
                else:
                    o += 1
                    alias[p] = f'O{o}'
            rank = evmap.get(pid, {}).get(hid)
            shown = None if pid in held else rank
            acts_h = amap.get(hid, A.head(0))
            lines.append(fmt_hand(hmap[hid], seats_h, acts_h, alias, shown, i, len(hs)))
            rec = hand_record(hmap[hid], seats_h, acts_h, alias, shown, i)
            rec['chrono_frac'] = round(i / max(1, len(hs) - 1), 6)
            recs.append(rec)
        (outdir / f'pair_{pid}.txt').write_text('\n'.join(lines), encoding='utf-8')
        (outdir / f'pair_{pid}.json').write_text(json.dumps(
            {'pair_id': pid, 'family': family, 'split': 'WITHHELD' if pid in held else 'TRAINING',
             'n_hands': len(hs),
             'n_listed': None if pid in held else len(evmap.get(pid, {})),
             'hands': recs}), encoding='utf-8')
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    # The answer key is written to outdir's PARENT on purpose: outdir is the directory handed to an
    # outside agent, and it must contain no way to recover the withheld tags.
    korder = {p: sorted(pair_hands[p], key=lambda x: (hmap[x]['started_at'], x)) for p in sorted(held)}
    key = {p: {h: evmap.get(p, {}).get(h) for h in korder[p] if evmap.get(p, {}).get(h)}
           for p in sorted(held)}
    (outdir.parent / f'ANSWER_KEY_{family}.json').write_text(
        json.dumps({'family': family, 'held_out': sorted(held), 'listed_by_pair': key,
                    'hand_order': korder}, indent=1), encoding='utf-8')
    assert not (outdir / f'ANSWER_KEY_{family}.json').exists()
    print(f'wrote {len(order)} packets + manifest.json to {outdir}', flush=True)
    print(f'withheld {len(held)} pairs: their hands carry NO listed/unlisted tag', flush=True)


if __name__ == '__main__':
    main()
