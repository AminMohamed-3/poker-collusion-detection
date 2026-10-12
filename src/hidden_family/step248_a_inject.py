"""STEP248 Part A positive control: PLANT a one-way laundering chain a -> c -> b and check
that the CORRECTED statistic (chain_max) recovers it while the ORIGINAL (chain_sym) does not.

OBS 16.4 measured `chain` blind at 4x a real directed transfer's strength.  Line 115 of
hf10_mechanism_stats.py is `chain = np.minimum(chain, chain.T)`, which is identically zero
unless the SAME pair launders in both directions, so that null was uninterpretable.  Without
this control the Part A null on laundering means nothing (OBS 16.8).

Injection is hf12's M3, at hf12's calibration (strength 1.0 = 7 hands x 175 chips = one real
directed_transfer's worth).  Development phase only, truncated to 2,000 hands per table.
"""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_a_mechstats import ORIGINAL, OUT, SESSION_GAP_US, load_hands, table_stats

RNG = np.random.default_rng(20260917)
N_TABLES = 60
STRENGTHS = [0.25, 0.5, 1.0, 2.0, 4.0]
UNIT_CHIPS = 175.0
UNIT_HANDS = 7
WATCH = ['chain_sym', 'chain_max', 'chain_z_sym', 'chain_z_max', 'flow_z', 'direct']


def table_arrays(frame):
    frame = frame.sort('ts', 'seat_no')
    uh = frame.select('hand_id', 'ts').unique(subset='hand_id').sort('ts')
    hidx = {h: i for i, h in enumerate(uh['hand_id'].to_list())}
    tss = uh['ts'].to_numpy()
    sess = np.r_[0, np.cumsum(np.diff(tss) > SESSION_GAP_US)].astype(int)
    up = frame['player_id'].unique(maintain_order=True).to_list()
    pidx = {p: i for i, p in enumerate(up)}
    return {'seat_no': frame['seat_no'].to_numpy().copy(),
            'pid': np.fromiter((pidx[p] for p in frame['player_id'].to_list()), int,
                               frame.height),
            'net': frame['net_chips'].to_numpy().astype(float),
            'contrib': frame['total_contribution'].to_numpy().astype(float),
            'folded': frame['folded'].to_numpy().copy(),
            'showdown': frame['went_to_showdown'].to_numpy().copy(),
            'hand_idx': np.fromiter((hidx[h] for h in frame['hand_id'].to_list()), int,
                                    frame.height),
            'sess': sess, 'n_hands': len(hidx), 'n_sess': int(sess.max()) + 1, 'players': up}


def stats_from_arrays(A):
    return table_stats(A['seat_no'], A['pid'], A['net'], A['contrib'], A['folded'],
                       A['showdown'], A['hand_idx'], A['sess'], A['n_hands'], A['n_sess'],
                       len(A['players']))


def hands_with(A, p):
    m = A['pid'] == p
    return A['hand_idx'][m], np.where(m)[0]


def inject_chain(A, a, b, s, rng):
    """hf12's M3: move chips a -> c in some hands and c -> b in others."""
    B = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in A.items()}
    ha, ra = hands_with(B, a)
    hb, rb = hands_with(B, b)
    if len(np.intersect1d(ha, hb)) < 20:
        return None
    others = [p for p in range(len(B['players'])) if p not in (a, b)]
    hc = {c: hands_with(B, c)[0] for c in others}
    c = max(others, key=lambda c: min(len(np.intersect1d(hc[c], ha)),
                                      len(np.intersect1d(hc[c], hb))))
    rc = {h: r for h, r in zip(*hands_with(B, c))}
    ia = {h: r for h, r in zip(ha, ra)}
    ib = {h: r for h, r in zip(hb, rb)}
    nt = max(1, int(round(s * UNIT_HANDS)))
    leg1 = np.intersect1d(ha, hc[c])
    leg2 = np.intersect1d(hb, hc[c])
    if len(leg1) < nt or len(leg2) < nt:
        return None
    for h in rng.permutation(leg1)[:nt]:
        B['net'][ia[h]] -= UNIT_CHIPS
        B['net'][rc[h]] += UNIT_CHIPS
    for h in rng.permutation(leg2)[:nt]:
        B['net'][rc[h]] -= UNIT_CHIPS
        B['net'][ib[h]] += UNIT_CHIPS
    return B


def midrank(sorted_ref, v):
    lo = np.searchsorted(sorted_ref, v, side='left')
    hi = np.searchsorted(sorted_ref, v, side='right')
    return (lo + hi) / (2.0 * len(sorted_ref))


def main():
    t0 = time.time()
    hands = load_hands('development', 2000)
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet',
                            columns=['hand_id', 'player_id', 'seat_no', 'net_chips',
                                     'total_contribution', 'folded', 'went_to_showdown'])
    seats = seats.join(hands, on='hand_id', how='inner')
    parts = seats.partition_by('table_id', as_dict=True)
    keys = sorted(parts.keys(), key=lambda k: str(k))
    ref = pl.read_parquet(OUT / 'step248a-mechstats-development2000.parquet')
    sorted_ref = {w: np.sort(ref[w].to_numpy().astype(float)) for w in WATCH}
    dl = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    labelled = set(dl['player_1'].to_list()) | set(dl['player_2'].to_list())
    print(f'{len(keys)} tables  [{time.time()-t0:.0f}s]', flush=True)

    chosen = RNG.choice(len(keys), size=N_TABLES, replace=False)
    res = {w: {str(s): [] for s in STRENGTHS} for w in WATCH}
    baseline = {w: [] for w in WATCH}
    n_done = 0
    for ti in chosen:
        frame = parts[keys[ti]]
        A = table_arrays(frame)
        free = [i for i, p in enumerate(A['players']) if p not in labelled]
        if len(free) < 6:
            continue
        a, b = RNG.choice(free, size=2, replace=False)
        base = stats_from_arrays(A)
        for w in WATCH:
            baseline[w].append(midrank(sorted_ref[w], base[w][a, b]))
        for s in STRENGTHS:
            B = inject_chain(A, int(a), int(b), s, RNG)
            if B is None:
                continue
            st = stats_from_arrays(B)
            for w in WATCH:
                res[w][str(s)].append(midrank(sorted_ref[w], st[w][a, b]))
        n_done += 1
        if n_done % 10 == 0:
            print(f'  {n_done} tables injected  [{time.time()-t0:.0f}s]', flush=True)

    summary = {'n_tables': n_done, 'unit_chips': UNIT_CHIPS, 'unit_hands': UNIT_HANDS,
               'baseline_median_pct': {w: float(np.median(baseline[w])) for w in WATCH},
               'by_strength': {}}
    top = len(sorted_ref['chain_max']) - 174        # "in the top 174 of 174,000", hf12's rule
    for w in WATCH:
        summary['by_strength'][w] = {}
        for s in STRENGTHS:
            v = np.array(res[w][str(s)])
            if len(v) == 0:
                continue
            summary['by_strength'][w][str(s)] = {
                'n': int(len(v)), 'median_pct': float(np.median(v)),
                'share_top174': float((v >= top / len(sorted_ref[w])).mean()),
                'share_above_999': float((v >= 0.999).mean())}
    (OUT / 'step248a-inject-chain.json').write_text(json.dumps(summary, indent=1),
                                                    encoding='utf-8')
    for w in WATCH:
        line = ' '.join(f"{s}:{summary['by_strength'][w].get(str(s),{}).get('median_pct',float('nan')):.4f}"
                        for s in STRENGTHS)
        print(f"{w:>12} base {summary['baseline_median_pct'][w]:.4f} | {line}")
    print(f'done  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
