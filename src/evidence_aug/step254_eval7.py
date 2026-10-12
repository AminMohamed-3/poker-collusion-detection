"""STEP254 helper: vectorised 5/6/7-card poker evaluator returning phevaluator-compatible ranks.

Card id convention used everywhere in STEP254: id = rank * 4 + suit, rank 0..12 for '2'..'A',
suit 0..3 for 'cdhs'.  evaluate(cards) takes an (N, k) int array (k in 5, 6, 7) and returns the
canonical Cactus-Kev equivalence-class rank, 1 = royal flush ... 7462 = 7-5-4-3-2 offsuit, which
is exactly what phevaluator.evaluate_cards returns (asserted in self_test()).

Strength percentile convention (STEP237): (7463 - rank) / 7462.
"""
from __future__ import annotations

import itertools

import numpy as np

RANK_CHARS = '23456789TJQKA'
SUIT_CHARS = 'cdhs'
CARD_ID = {r + s: i * 4 + j for i, r in enumerate(RANK_CHARS) for j, s in enumerate(SUIT_CHARS)}

POPC = np.array([bin(m).count('1') for m in range(8192)], dtype=np.int8)
POW5 = (5 ** np.arange(13)).astype(np.int64)


def _straight_high(mask: int) -> int:
    """highest straight top-rank inside a 13-bit rank mask, -1 if none (wheel returns 3 = '5')."""
    for hi in range(12, 3, -1):
        if all(mask >> (hi - k) & 1 for k in range(5)):
            return hi
    if (mask & 1) and (mask >> 1 & 1) and (mask >> 2 & 1) and (mask >> 3 & 1) and (mask >> 12 & 1):
        return 3
    return -1


def _build_class_tables():
    """canonical 7462 ordering."""
    sf, quads, boats, flushes, straights, trips, twopair, pairs, highs = {}, {}, {}, {}, {}, {}, {}, {}, {}
    v = 0
    for hi in range(12, 2, -1):
        v += 1
        sf[hi] = v                                   # 1..10
    for q in range(12, -1, -1):
        for k in range(12, -1, -1):
            if k == q:
                continue
            v += 1
            quads[(q, k)] = v                        # 11..166
    for t in range(12, -1, -1):
        for p in range(12, -1, -1):
            if p == t:
                continue
            v += 1
            boats[(t, p)] = v                        # 167..322
    combos = sorted(itertools.combinations(range(13), 5), key=lambda c: tuple(sorted(c, reverse=True)),
                    reverse=True)
    nonstr = [c for c in combos if _straight_high(sum(1 << r for r in c)) < 0]
    for c in nonstr:
        v += 1
        flushes[tuple(sorted(c, reverse=True))] = v  # 323..1599
    for hi in range(12, 2, -1):
        v += 1
        straights[hi] = v                            # 1600..1609
    for t in range(12, -1, -1):
        for kk in sorted(itertools.combinations([r for r in range(13) if r != t], 2),
                         key=lambda x: tuple(sorted(x, reverse=True)), reverse=True):
            v += 1
            trips[(t, tuple(sorted(kk, reverse=True)))] = v          # 1610..2467
    for p1 in range(12, -1, -1):
        for p2 in range(p1 - 1, -1, -1):
            for k in range(12, -1, -1):
                if k in (p1, p2):
                    continue
                v += 1
                twopair[(p1, p2, k)] = v                              # 2468..3325
    for p in range(12, -1, -1):
        for kk in sorted(itertools.combinations([r for r in range(13) if r != p], 3),
                         key=lambda x: tuple(sorted(x, reverse=True)), reverse=True):
            v += 1
            pairs[(p, tuple(sorted(kk, reverse=True)))] = v           # 3326..6185
    for c in nonstr:
        v += 1
        highs[tuple(sorted(c, reverse=True))] = v                     # 6186..7462
    assert v == 7462, v
    return sf, quads, boats, flushes, straights, trips, twopair, pairs, highs


_SF, _QUADS, _BOATS, _FLUSHES, _STRAIGHTS, _TRIPS, _2P, _PAIRS, _HIGH = _build_class_tables()

FLUSH_TABLE = np.full(8192, 9999, dtype=np.int16)
for m in range(8192):
    if POPC[m] < 5:
        continue
    sh = _straight_high(m)
    if sh >= 0:
        FLUSH_TABLE[m] = _SF[sh]
    else:
        top = sorted([r for r in range(13) if m >> r & 1], reverse=True)[:5]
        FLUSH_TABLE[m] = _FLUSHES[tuple(top)]


def _nonflush_value(counts: tuple[int, ...]) -> int:
    present = [r for r in range(13) if counts[r]]
    mask = sum(1 << r for r in present)
    best = 9999
    c4 = [r for r in present if counts[r] == 4]
    c3 = sorted([r for r in present if counts[r] == 3], reverse=True)
    c2 = sorted([r for r in present if counts[r] == 2], reverse=True)
    if c4:
        q = max(c4)
        k = max(r for r in present if r != q)
        best = min(best, _QUADS[(q, k)])
    if c3:
        t = c3[0]
        pool = [r for r in present if r != t and counts[r] >= 2]
        if pool:
            best = min(best, _BOATS[(t, max(pool))])
    sh = _straight_high(mask)
    if sh >= 0:
        best = min(best, _STRAIGHTS[sh])
    if c3:
        t = c3[0]
        kk = sorted([r for r in present if r != t], reverse=True)[:2]
        if len(kk) == 2:
            best = min(best, _TRIPS[(t, tuple(kk))])
    if len(c2) >= 2:
        p1, p2 = c2[0], c2[1]
        rest = [r for r in present if r not in (p1, p2)]
        if rest:
            best = min(best, _2P[(p1, p2, max(rest))])
    if len(c2) == 1 and not c3 and not c4:
        p = c2[0]
        kk = sorted([r for r in present if r != p], reverse=True)[:3]
        if len(kk) == 3:
            best = min(best, _PAIRS[(p, tuple(kk))])
    if len(present) >= 5 and not c2 and not c3 and not c4:
        top = sorted(present, reverse=True)[:5]
        best = min(best, _HIGH.get(tuple(top), 9999))  # a straight top-5 lives in _STRAIGHTS
    assert best < 9999, counts
    return best


def _build_nonflush(n: int):
    keys, vals = [], []
    cnt = [0] * 13

    def rec(r: int, left: int):
        if r == 13:
            if left == 0:
                keys.append(int(sum(cnt[i] * POW5[i] for i in range(13))))
                vals.append(_nonflush_value(tuple(cnt)))
            return
        remaining_slots = (13 - r) * 4
        if left > remaining_slots:
            return
        for c in range(min(4, left) + 1):
            cnt[r] = c
            rec(r + 1, left - c)
        cnt[r] = 0

    rec(0, n)
    k = np.array(keys, dtype=np.int64)
    o = np.argsort(k)
    return k[o], np.array(vals, dtype=np.int16)[o]


_NF = {n: _build_nonflush(n) for n in (5, 6, 7)}


def evaluate(cards: np.ndarray) -> np.ndarray:
    """cards: (N, k) int array of card ids, k in {5,6,7}. returns (N,) int16 ranks (1 = best)."""
    cards = np.asarray(cards)
    n = cards.shape[1]
    ranks = (cards >> 2).astype(np.int64)
    suits = (cards & 3).astype(np.int64)
    bits = (np.int64(1) << ranks)
    N = cards.shape[0]
    masks = np.zeros((N, 4), dtype=np.int64)
    for s in range(4):
        masks[:, s] = np.bitwise_or.reduce(np.where(suits == s, bits, 0), axis=1)
    pc = POPC[masks]
    fs = pc.argmax(axis=1)
    fmask = masks[np.arange(N), fs]
    has_flush = pc.max(axis=1) >= 5
    key = POW5[ranks].sum(axis=1)
    kk, vv = _NF[n]
    idx = np.searchsorted(kk, key)
    out = vv[idx].astype(np.int16)
    if has_flush.any():
        out = np.where(has_flush, FLUSH_TABLE[fmask], out)
    return out


def strength(cards: np.ndarray) -> np.ndarray:
    return (7463.0 - evaluate(cards)) / 7462.0


def self_test(n_samples: int = 20000, seed: int = 254) -> None:
    from phevaluator import Card, evaluate_cards
    inv = {v: k for k, v in CARD_ID.items()}
    rng = np.random.default_rng(seed)
    for k in (5, 6, 7):
        deals = np.array([rng.permutation(52)[:k] for _ in range(n_samples // 3)])
        mine = evaluate(deals)
        for i in range(0, len(deals), 1):
            if i >= 4000:
                break
            ref = evaluate_cards(*[inv[int(c)] for c in deals[i]])
            assert int(mine[i]) == int(ref), (k, deals[i], int(mine[i]), ref)
    print('step254_eval7 self-test OK (matches phevaluator on 5/6/7 cards)')


if __name__ == '__main__':
    self_test()
