"""HF12 stage 3: PLANT each hypothesised mechanism in the real hands and see whether the
detector finds it.  This is the step the whole investigation has been missing.

hf11 shows some statistics are blind to the three disclosed families (shared-victim AUC .52,
laundering .51, co-presence .51, seat geometry .52).  A null on a blind statistic says nothing
whatsoever.  So here the mechanism is manufactured at a known strength inside real tables, the
statistics are recomputed through the identical hf10 pipeline, and the recovery rate is
measured against the untouched population.  The output is a DETECTION FLOOR per mechanism:
the smallest planted effect the statistic pulls into the top of 174,000 pairs.

Only then does a null mean "the mechanism is not there" rather than "we cannot see it".

Injections, all applied to the raw seat rows of one table:

  M1 avoidance      flip `folded` so the pair stops reaching the end of the same pot
  M2 shared victim  move chips from one common third player to BOTH partners
  M3 laundering     move chips a -> c in some hands and c -> b in others
  M4 churn          alternate the direction of a pair transfer session by session, net ~ 0
  M5 selection      swap seats so the pair co-occurs, and sits adjacent, more than chance

Each runs at four strengths so the answer is a curve, not a yes/no.  Read-only.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import ORIGINAL, CV, write_json
from hidden_family.hf10_mechanism_stats import table_arrays, stats_from_arrays

OUT = CV / 'experiments' / 'hidden_family_20260912'
RNG = np.random.default_rng(20260912)
N_TABLES = 60
STRENGTHS = [0.25, 0.50, 1.00, 2.00, 4.00]
# Calibration against the real families, measured directly from seats.parquet:
#   directed_transfer  ~7 shared-contested hands, 1,215 chips moved in total, biggest one 513
#   soft_play          ~6 hands,   114 chips  -- moves NO net money, it is an action pattern
#   coordinated_isolat ~2 hands,   181 chips  -- the pair barely interacts; the victim carries it
#   non-target         ~3 hands,   117 chips
# so strength 1.0 is defined as one real directed_transfer's worth of value moved, and the
# first injection sweep was ~1000x too weak because it used median |net| (= 1 chip) as its unit.
UNIT_CHIPS = 175.0
UNIT_HANDS = 7
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']

# statistic each mechanism is supposed to move, and the direction of the signal
TARGET = {'M0_directed_transfer': [('flow_z', +1), ('direct', +1)],
          'M1_avoidance': [('avoid_lr', -1), ('contest_both', -1), ('avoid_z', -1)],
          'M2_shared_victim': [('shared_victim', +1), ('sv_overlap', +1), ('sv_conc', +1)],
          'M3_laundering': [('chain', +1), ('chain_z', +1)],
          'M4_churn': [('churn', +1), ('churn_ratio', +1), ('churn_cancel', +1)],
          'M5_selection': [('copresence', +1), ('adjacency', +1), ('n_both', +1)]}


def midrank(sorted_ref, v):
    """Tie-aware percentile.  Half the pairs share the same integer value on statistics like
    contest_both, so a plain left-searchsorted reports a planted pair at percentile 1.0 purely
    for joining a huge tied mass.  The mid-rank refuses to call that a detection."""
    lo = np.searchsorted(sorted_ref, v, side='left')
    hi = np.searchsorted(sorted_ref, v, side='right')
    return (lo + hi) / (2.0 * len(sorted_ref))


def hands_with(A, p):
    """hand indices where player p is seated, and the seat-row positions."""
    m = A['pid'] == p
    return A['hand_idx'][m], np.where(m)[0]


def inject(A, a, b, mech, s, rng):
    """Return a mutated copy of the table arrays with `mech` planted in pair (a,b)."""
    B = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in A.items()}
    ha, ra = hands_with(B, a)
    hb, rb = hands_with(B, b)
    shared = np.intersect1d(ha, hb)
    if len(shared) < 20:
        return None
    pot = UNIT_CHIPS
    nt = max(1, int(round(s * UNIT_HANDS)))

    ia_all = {h: r for h, r in zip(ha, ra)}
    ib_all = {h: r for h, r in zip(hb, rb)}
    if mech == 'M0_directed_transfer':
        for h in rng.permutation(shared)[:nt]:
            B['net'][ia_all[h]] -= pot
            B['net'][ib_all[h]] += pot
            B['folded'][ia_all[h]] = False
            B['folded'][ib_all[h]] = False
        return B

    if mech == 'M1_avoidance':
        # both currently reach the end of the same pot -> make one of them fold
        ia = {h: r for h, r in zip(ha, ra)}
        ib = {h: r for h, r in zip(hb, rb)}
        both = [h for h in shared if not B['folded'][ia[h]] and not B['folded'][ib[h]]]
        if not both:
            return None
        k = int(round(s * len(both)))
        for h in rng.permutation(both)[:k]:
            r = ia[h] if rng.random() < .5 else ib[h]
            B['folded'][r] = True
            B['showdown'][r] = False
        return B

    others = [p for p in range(len(B['players'])) if p not in (a, b)]
    if mech == 'M2_shared_victim':
        # find the third player who shares most hands with both, then feed both partners
        hc = {c: hands_with(B, c)[0] for c in others}
        c = max(others, key=lambda c: len(np.intersect1d(hc[c], ha)) +
                len(np.intersect1d(hc[c], hb)))
        rc = {h: r for h, r in zip(*hands_with(B, c))}
        for who, hw, rw in ((a, ha, ra), (b, hb, rb)):
            idx = {h: r for h, r in zip(hw, rw)}
            common = np.intersect1d(hw, hc[c])
            for h in rng.permutation(common)[:nt]:
                B['net'][idx[h]] += pot
                B['net'][rc[h]] -= pot
        return B

    if mech == 'M3_laundering':
        hc = {c: hands_with(B, c)[0] for c in others}
        c = max(others, key=lambda c: min(len(np.intersect1d(hc[c], ha)),
                                          len(np.intersect1d(hc[c], hb))))
        rc = {h: r for h, r in zip(*hands_with(B, c))}
        ia = {h: r for h, r in zip(ha, ra)}
        ib = {h: r for h, r in zip(hb, rb)}
        leg1 = np.intersect1d(ha, hc[c])
        leg2 = np.intersect1d(hb, hc[c])
        for h in rng.permutation(leg1)[:nt]:
            B['net'][ia[h]] -= pot
            B['net'][rc[h]] += pot
        for h in rng.permutation(leg2)[:nt]:
            B['net'][rc[h]] -= pot
            B['net'][ib[h]] += pot
        return B

    if mech == 'M4_churn':
        # ordinary pairs reach the end of the same pot a MEDIAN of once in ~110 shared hands
        # (hf11), so an alternating dump has to put them in the pot together first -- which is
        # what any generator implementing it would also have to do.
        for h in rng.permutation(shared)[:2 * nt]:          # both directions need room
            sign = 1.0 if B['sess'][h] % 2 == 0 else -1.0   # direction flips every session
            B['folded'][ia_all[h]] = False
            B['folded'][ib_all[h]] = False
            B['net'][ia_all[h]] += sign * pot
            B['net'][ib_all[h]] -= sign * pot
        return B

    if mech == 'M5_selection':
        # put b into hands a plays but b does not, in the seat right next to a
        solo_a = np.setdiff1d(ha, hb)
        if len(solo_a) < 10:
            return None
        rowof = {}
        for r, (h, p) in enumerate(zip(B['hand_idx'], B['pid'])):
            rowof.setdefault(h, []).append(r)
        seat_a = {h: B['seat_no'][r] for h, r in zip(ha, ra)}
        k = int(round(min(s, 1.0) * 0.6 * len(solo_a)))     # selection saturates at s=1
        for h in rng.permutation(solo_a)[:k]:
            rows = [r for r in rowof[h] if B['pid'][r] != a]
            if not rows:
                continue
            want = (seat_a[h] + 1) % 6
            tgt = min(rows, key=lambda r: 0 if B['seat_no'][r] == want else 1)
            B['pid'][tgt] = b
        return B
    raise ValueError(mech)


def main():
    t0 = time.time()
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet',
                            columns=['hand_id', 'table_id', 'started_at', 'phase'])
    hands = hands.filter(pl.col('phase') == 'development').with_columns(
        pl.col('started_at').dt.replace_time_zone(None).cast(pl.Int64).alias('ts')
    ).drop('started_at', 'phase')
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet',
                            columns=['hand_id', 'player_id', 'seat_no', 'net_chips',
                                     'total_contribution', 'folded', 'went_to_showdown'])
    seats = seats.join(hands, on='hand_id', how='inner')
    parts = seats.partition_by('table_id', as_dict=True)
    keys = sorted(parts.keys(), key=lambda k: str(k))
    print(f'{len(keys)} tables loaded [{time.time() - t0:.0f}s]', flush=True)

    ref = pl.read_parquet(OUT / 'hf11-dev-mechstats-labelled.parquet')
    unl = ref.filter(pl.col('label') == -1)
    fam = ref['behavior_family'].to_numpy()
    lab = {p: pl.DataFrame(ref).filter(pl.col('pair_id').is_not_null()) for p in []}

    chosen = RNG.choice(len(keys), size=N_TABLES, replace=False)
    labelled_players = set(ref.filter(pl.col('label') >= 0)['player_a'].to_list()) | \
        set(ref.filter(pl.col('label') >= 0)['player_b'].to_list())

    # ---- yardstick: where do the DISCLOSED families sit on each statistic? -------
    print()
    print('=' * 100)
    print('YARDSTICK -- percentile of the disclosed families within the 172,140 unlabelled '
          'pairs')
    print('=' * 100)
    all_stats = sorted({s for v in TARGET.values() for s, _ in v})
    yard = {}
    print(f'{"statistic":<16}{"dir":>5}' + ''.join(f'{f[:14]:>16}' for f in FAMS))
    for s in all_stats:
        u = np.sort(unl[s].to_numpy().astype(float))
        for sgn in (1, -1):
            row = []
            for f in FAMS:
                v = ref.filter(pl.col('behavior_family') == f)[s].to_numpy().astype(float)
                pc = midrank(u, v)
                pc = pc if sgn > 0 else 1 - pc
                row.append(float(np.median(pc)))
            yard[f'{s}:{sgn}'] = dict(zip(FAMS, row))
            print(f'{s:<16}{sgn:>5}' + ''.join(f'{x:>16.4f}' for x in row))

    # ---- the injection sweep -----------------------------------------------------
    print()
    print('=' * 100)
    print('INJECTION -- plant each mechanism in 60 real tables, replay hf10, and measure the')
    print('             planted pair\'s percentile among the 172,140 untouched unlabelled pairs')
    print('=' * 100)
    print(f'{"mechanism":<18}{"stat":<15}{"strength":>9}{"n":>5}'
          f'{"med pct":>10}{"p>=.99":>9}{"p>=.999":>9}{"top-174":>9}')
    results = []
    for mech, targets in TARGET.items():
        for s in STRENGTHS:
            planted = {st: [] for st, _ in targets}
            n_ok = 0
            for ti in chosen:
                key = keys[ti]
                A = table_arrays(parts[key])
                P = len(A['players'])
                cand = [(i, j) for i in range(P) for j in range(i + 1, P)
                        if A['players'][i] not in labelled_players
                        and A['players'][j] not in labelled_players]
                if not cand:
                    continue
                a, b = cand[RNG.integers(len(cand))]
                B = inject(A, a, b, mech, s, RNG)
                if B is None:
                    continue
                st2 = stats_from_arrays(B)
                n_ok += 1
                for stat, _ in targets:
                    planted[stat].append(st2[stat][a, b])
            for stat, sgn in targets:
                v = np.array(planted[stat])
                if not len(v):
                    continue
                u = np.sort(unl[stat].to_numpy().astype(float))
                pc = midrank(u, v)
                pc = pc if sgn > 0 else 1 - pc
                top174 = float(np.mean(pc >= 1 - 174 / len(u)))
                rec = {'mechanism': mech, 'statistic': stat, 'strength': s, 'n': int(len(v)),
                       'median_pct': float(np.median(pc)),
                       'frac_ge_99': float(np.mean(pc >= .99)),
                       'frac_ge_999': float(np.mean(pc >= .999)), 'frac_top174': top174}
                results.append(rec)
                print(f'{mech:<18}{stat:<15}{s:>9.2f}{len(v):>5}{np.median(pc):>10.4f}'
                      f'{rec["frac_ge_99"]:>9.2f}{rec["frac_ge_999"]:>9.2f}{top174:>9.2f}',
                      flush=True)
        print(f'{"":<18}{"":<15}{"":>9}  [{time.time() - t0:.0f}s]')

    write_json(OUT / 'hf12-injection.json',
               {'yardstick': yard, 'injection': results, 'n_tables': N_TABLES,
                'strengths': STRENGTHS})
    print()
    print(f'written {OUT / "hf12-injection.json"}')


if __name__ == '__main__':
    main()
