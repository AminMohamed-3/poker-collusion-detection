"""STEP248 Part B.5: the sceptic's checks on the candidates.

(1) DECOY CONTROL (OBS 15.3).  A confirmed colluder is ORDINARY with everyone but the partner;
    the host's decoys -- tilt, weak play, streaks, repeated opponent selection -- are odd with
    EVERYONE.  For every candidate pair we compute excess = S1 percentile of the pair minus the
    larger of the two members' median S1 percentile over their 28 other pairs, and compare the
    candidates with the STEP207 block and with the population.
(2) SESSION CONCENTRATION.  Coordination is episodic but runs from the start of the history at
    a constant per-hand rate, so its flagged hands should span several sessions; a per-session
    artefact (one lineup, one hot streak) concentrates in one.
(3) FLAGGED HANDS.  Re-replay the candidates' tables and record exactly which hands carry each
    significant event, in time order, so the hands can be read.
(4) RENDER.  Print the action log of flagged hands and of matched hands from unflagged pairs.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_c_common import ORIGINAL, OUT, attach_players, load_detail, load_phase
from step248_b_events import NAMES, NN, SCRATCH
from step248_b_run import IU, JU, load_chunk, emit_table, N_PL

TABLES_PER_CHUNK = 10
ACTN = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
STN = ['preflop', 'flop', 'turn', 'river']
SESSION_GAP_US = 600 * 1_000_000


def decoy_control(cand, groups):
    """excess = pair S1 percentile - max over the two members of their median other-pair pct."""
    s1 = cand['S1'].to_numpy()
    order = np.argsort(np.argsort(s1))
    pct = (order + .5) / len(s1)
    cand = cand.with_columns(pl.Series('S1_pct', pct))
    pa = cand.select(pl.col('player_a').alias('p'), 'S1_pct', 'pair_id')
    pb = cand.select(pl.col('player_b').alias('p'), 'S1_pct', 'pair_id')
    allp = pl.concat([pa, pb])
    pctmap = cand.select('pair_id', 'S1_pct')
    out = {}
    for name, sel in groups.items():
        sel = sel.drop('S1_pct', strict=False).join(pctmap, on='pair_id', how='left')
        ids = set(sel['pair_id'].to_list())
        if not ids:
            out[name] = None
            continue
        exc = []
        sub = allp.filter(~pl.col('pair_id').is_in(list(ids)))
        med = sub.group_by('p').agg(pl.col('S1_pct').median().alias('base'))
        m = dict(zip(med['p'].to_list(), med['base'].to_list()))
        for r in sel.select('player_a', 'player_b', 'S1_pct').iter_rows():
            base = max(m.get(r[0], .5), m.get(r[1], .5))
            exc.append(r[2] - base)
        out[name] = {'n': len(exc), 'median_pair_pct': float(np.median(sel['S1_pct'])),
                     'median_member_baseline': float(np.median(
                         [max(m.get(x, .5), m.get(y, .5))
                          for x, y in sel.select('player_a', 'player_b').iter_rows()])),
                     'median_excess': float(np.median(exc))}
    return cand, out


def flagged_hands(tag, want):
    """want: {(tidx, A, B): set(event codes)} -> rows (tidx, A, B, event, hloc)."""
    by_tab = defaultdict(list)
    for (t, A, B), evs in want.items():
        by_tab[t].append((A, B, evs))
    rows = []
    for t, items in sorted(by_tab.items()):
        ch = t // TABLES_PER_CHUNK
        h, s, a = load_chunk(tag, ch)
        hm = h.filter(pl.col('tidx') == t).sort('hloc')
        sf = s.filter(pl.col('tidx') == t).sort(['hidx', 'seat_no'])
        af = a.filter(pl.col('tidx') == t).sort(['hidx', 'action_no'])
        v, P6, HR, PART, nH = emit_table(hm, sf, af)
        hl = (v >> 17).astype(np.int64)
        al = ((v >> 14) & 7).astype(np.int64)
        mk = ((v >> 8) & 63).astype(np.int64)
        cd = (v & 255).astype(np.int64)
        A6 = P6[hl, al]
        for j in range(6):
            sel = (mk >> j) & 1 == 1
            if not sel.any():
                continue
            B6 = P6[hl[sel], j]
            for (A, B, evs) in items:
                m = (A6[sel] == A) & (B6 == B) & np.isin(cd[sel], list(evs))
                if m.any():
                    for hloc, code in zip(hl[sel][m], cd[sel][m]):
                        rows.append((t, A, B, int(code), int(hloc)))
    return pl.DataFrame(rows, schema=['tidx', 'A', 'B', 'event', 'hloc'], orient='row')


def render_hand(tag, hidx, mark=()):
    d = SCRATCH / tag
    hd = pl.read_parquet(d / 'hands.parquet').filter(pl.col('hidx') == hidx)
    t = int(hd['tidx'][0])
    ch = t // TABLES_PER_CHUNK
    s = pl.read_parquet(d / f's_{ch:03d}.parquet').filter(pl.col('hidx') == hidx).sort('seat_no')
    a = pl.read_parquet(d / f'a_{ch:03d}.parquet').filter(
        pl.col('hidx') == hidx).sort('action_no')
    bb = int(hd['big_blind'][0])
    lines = [f'hand {hd["hand_id"][0]} (table {t}, index {int(hd["hloc"][0])}) '
             f'button seat {int(hd["button_seat"][0])} bb {bb} '
             f'board [{hd["board_cards"][0]}] pot {int(hd["final_pot"][0])}']
    names = {}
    for r in s.iter_rows(named=True):
        tagm = ' <<<' if r['ploc'] in mark else ''
        names[r['ploc']] = f'P{r["ploc"]:02d}'
        lines.append(f'  seat {r["seat_no"]} P{r["ploc"]:02d} {r["hole_card_1"]}'
                     f'{r["hole_card_2"]} stack? contrib {r["total_contribution"]:>4} '
                     f'net {r["net_chips"]:>5} {"FOLD" if r["folded"] else "    "} '
                     f'{"SD" if r["went_to_showdown"] else "  "}{tagm}')
    cur = -1
    for r in a.iter_rows(named=True):
        if r['street'] != cur:
            cur = r['street']
            lines.append(f'  -- {STN[cur]} --')
        tagm = ' <<<' if r['ploc'] in mark else ''
        lines.append(f'     {names[r["ploc"]]} {ACTN[r["action"]]:<7} amt {r["amount"]:>4} '
                     f'to {r["amount_to"]:>4} pot {r["pot_before"]:>5} '
                     f'call {r["to_call"]:>4}{tagm}')
    return '\n'.join(lines)


def main():
    t0 = time.time()
    cand = pl.read_parquet(OUT / 'step248b-eval-pair-scores.parquet')
    hidden = pl.read_parquet(OUT / 'candidates_hidden.parquet')
    lowdose = pl.read_parquet(OUT / 'candidates_lowdose_known.parquet')
    block = cand.filter(pl.col('is_block'))
    print(f'candidates: hidden {hidden.height}, low-dose-known {lowdose.height}, '
          f'block {block.height}')

    cand2, dec = decoy_control(cand, {'hidden': hidden, 'lowdose_known': lowdose,
                                      'block': block,
                                      'block_top': block.sort('risk_score',
                                                              descending=True).head(200)})
    print('\n--- decoy control (OBS 15.3): are the members ordinary with everyone else? ---')
    for k, v in dec.items():
        if v:
            print(f'  {k:>16} n={v["n"]:>4} pair pct {v["median_pair_pct"]:.4f} '
                  f'member baseline {v["median_member_baseline"]:.4f} '
                  f'EXCESS {v["median_excess"]:+.4f}')

    det = load_detail('evaluation')
    key = cand.select('tidx', 'a', 'b', 'pair_id')
    de = pl.concat([det.join(key, on=['tidx', 'a', 'b'], how='inner'),
                    det.join(key.rename({'a': 'b', 'b': 'a'}), on=['tidx', 'a', 'b'],
                             how='inner')])
    want = defaultdict(set)
    ids = set(hidden['pair_id'].to_list()) | set(lowdose['pair_id'].to_list())
    for r in de.filter(pl.col('pair_id').is_in(list(ids))).iter_rows(named=True):
        want[(r['tidx'], r['a'], r['b'])].add(r['event'])
    print(f'\nextracting flagged hands for {len(want)} directed cells '
          f'({len(ids)} pairs)  [{time.time()-t0:.0f}s]')
    fh = flagged_hands('evaluation', want)
    hd = pl.read_parquet(SCRATCH / 'evaluation' / 'hands.parquet')
    fh = fh.join(hd.select('tidx', 'hloc', 'hidx', 'hand_id', 'ts'), on=['tidx', 'hloc'],
                 how='left')
    pm = pl.read_parquet(SCRATCH / 'evaluation' / 'players.parquet')
    fh = fh.join(pm.rename({'ploc': 'A', 'player_id': 'player_a'}), on=['tidx', 'A'],
                 how='left')
    fh = fh.join(pm.rename({'ploc': 'B', 'player_id': 'player_b'}), on=['tidx', 'B'],
                 how='left')
    fh = fh.with_columns(pl.min_horizontal('player_a', 'player_b').alias('p_lo'),
                         pl.max_horizontal('player_a', 'player_b').alias('p_hi'))
    ckey = cand.select('player_a', 'player_b', 'pair_id').with_columns(
        pl.min_horizontal('player_a', 'player_b').alias('p_lo'),
        pl.max_horizontal('player_a', 'player_b').alias('p_hi')).select('p_lo', 'p_hi',
                                                                        'pair_id')
    fh = fh.join(ckey, on=['p_lo', 'p_hi'], how='left')
    fh = fh.with_columns(pl.Series('event_name', [NAMES[i] for i in fh['event'].to_list()]))
    fh.write_parquet(OUT / 'step248b-flagged-hands.parquet')

    # ---- session concentration -----------------------------------------------------------
    ses = {}
    for t, g in hd.partition_by('tidx', as_dict=True).items():
        tt = t[0] if isinstance(t, tuple) else t
        g = g.sort('ts')
        ts = g['ts'].to_numpy()
        sid = np.r_[0, np.cumsum(np.diff(ts) > SESSION_GAP_US)]
        ses[tt] = dict(zip(g['hloc'].to_list(), sid.tolist()))
    fh = fh.with_columns(pl.Series('session', [ses[t][h] for t, h in
                                               zip(fh['tidx'].to_list(), fh['hloc'].to_list())]))
    conc = (fh.group_by('pair_id').agg(pl.col('session').n_unique().alias('n_sessions'),
                                       pl.col('hloc').n_unique().alias('n_hands'),
                                       pl.col('hloc').min().alias('first_hand'),
                                       pl.col('hloc').max().alias('last_hand')))
    hid2 = hidden.join(conc, on='pair_id', how='left')
    low2 = lowdose.join(conc, on='pair_id', how='left')
    blk = block.join(conc, on='pair_id', how='left').filter(pl.col('n_hands').is_not_null())
    print('\n--- session spread of the flagged hands ---')
    for nm, fr in (('hidden', hid2), ('lowdose_known', low2)):
        if fr.height:
            print(f'  {nm:>14} n={fr.height} median flagged hands '
                  f'{fr["n_hands"].median()} median distinct sessions '
                  f'{fr["n_sessions"].median()} share in ONE session '
                  f'{(fr["n_sessions"] == 1).mean():.2f}')
    if blk.height:
        print(f'  {"block(ref)":>14} n={blk.height} median flagged hands '
              f'{blk["n_hands"].median()} median distinct sessions '
              f'{blk["n_sessions"].median()} share in ONE session '
              f'{(blk["n_sessions"] == 1).mean():.2f}')
    else:
        print('  block(ref)     no flagged hands extracted (hands were only extracted for '
              'the two candidate lists)')

    hid2.sort('S1', descending=True).write_parquet(OUT / 'candidates_hidden.parquet')
    low2.sort('S1', descending=True).write_parquet(OUT / 'candidates_lowdose_known.parquet')
    (OUT / 'step248b-decoy-control.json').write_text(
        json.dumps({'decoy': dec,
                    'session': {k: ({'n': int(v.height),
                                     'median_sessions': float(v['n_sessions'].median() or 0),
                                     'one_session_share': float((v['n_sessions'] == 1).mean())}
                                    if v.height else None)
                                for k, v in (('hidden', hid2), ('lowdose', low2),
                                             ('block', blk))}}, indent=1), encoding='utf-8')

    # ---- render ----------------------------------------------------------------------------
    top = hid2.sort('S1', descending=True).head(4)
    txt = []
    for r in top.iter_rows(named=True):
        sub = fh.filter(pl.col('pair_id') == r['pair_id']).sort('hloc')
        evs = sorted(set(sub['event_name'].to_list()))
        txt.append('=' * 100)
        txt.append(f'HIDDEN CANDIDATE {r["pair_id"]} table {r["table_id"]} '
                   f'players {r["player_a"]} / {r["player_b"]} (local {r["a"]},{r["b"]}) '
                   f'shared {r["n_shared"]} S1 {r["S1"]:.2f} S2 {r["S2"]} '
                   f'risk207 {r["risk_score"]:.4f} events {evs} '
                   f'sessions {r["n_sessions"]} flagged hands {r["n_hands"]}')
        for hidx in sub['hidx'].unique().to_list()[:12]:
            txt.append(render_hand('evaluation', int(hidx), mark={r['a'], r['b']}))
            txt.append('')
    # matched unflagged controls: same tables, same shared-hand band, not flagged
    ctrl = cand.filter((~pl.col('flag30')) & (~pl.col('is_block')) &
                       pl.col('table_id').is_in(top['table_id'].to_list()))
    ctrl = ctrl.sort('n_shared', descending=True).head(4)
    hdz = pl.read_parquet(SCRATCH / 'evaluation' / 'hands.parquet')
    for r in ctrl.iter_rows(named=True):
        sfr = pl.read_parquet(SCRATCH / 'evaluation' /
                              f's_{r["tidx"] // TABLES_PER_CHUNK:03d}.parquet').filter(
            pl.col('tidx') == r['tidx'])
        both = (sfr.filter(pl.col('ploc').is_in([r['a'], r['b']]))
                .group_by('hidx').len().filter(pl.col('len') == 2))
        txt.append('=' * 100)
        txt.append(f'MATCHED UNFLAGGED {r["pair_id"]} table {r["table_id"]} '
                   f'shared {r["n_shared"]} S1 {r["S1"]:.2f} risk207 {r["risk_score"]:.4f}')
        for hidx in both['hidx'].to_list()[:5]:
            txt.append(render_hand('evaluation', int(hidx), mark={r['a'], r['b']}))
            txt.append('')
    (OUT / 'step248b-hand-readings.txt').write_text('\n'.join(txt), encoding='utf-8')
    print(f'\nrendered {len(top)} candidates and {ctrl.height} controls -> '
          f'step248b-hand-readings.txt  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
