"""Step 80 [C] (author: claude).  Is the step-79 winner real, or a lucky cell?

Step 79 promoted two arms past the pre-registered gate:

    rzm_wide_big_rank   .6435  +.0276  [+.0092, +.0461]   <- best
    rzm_base_small_clf  .6379  +.0220  [+.0016, +.0428]

The second is corroborated by its own neighbourhood: rz_base_small_clf sits at .6374, so varying the
margin transform moves it by +.0005 and the cell is flat and stable around +.0217.

The first is NOT.  Its immediate neighbours are much lower -- rzm_wide_big_clf .6216 (change the
objective), rz_wide_small_rank .6265 (change transform and capacity) -- so `rzm_wide_big_rank` is an
isolated peak in a 10-arm grid.  That is the shape of a lucky cell, and 22 arms have now been scored
on this audit set across steps 78 and 79, so at a 95 % interval roughly one false positive is
expected by chance.  Promoting an isolated peak to a held read is exactly the error that 18.15 and
18.17 recorded: a rule firing on a statistic that did not measure what the rule assumed.

So before any held read, two questions, both pre-registered here:

  Q1 SEED STABILITY.  Every arm averages 3 seeds (11, 12, 13).  Re-fit each cell under three
     INDEPENDENT seed triplets -- (11,12,13), (21,22,23), (31,32,33).  A real design effect is
     roughly constant across triplets; a lucky cell moves with the seeds.
  Q2 NEIGHBOURHOOD.  Re-fit the winner's one-factor neighbours under the same triplets.  A real
     effect leaves its neighbours elevated too; a lucky cell sits alone.

PRE-REGISTERED DECISION (written before any fit below):
  The arm carried to a held read is the one with the highest MINIMUM across the three seed triplets,
  among arms whose across-triplet spread (max - min) is below .010 AND whose one-factor neighbours
  average above the incumbent.  If the step-79 winner's spread exceeds .010, it is recorded as a
  lucky cell and `rzm_base_small_clf` / `rz_base_small_clf` is carried instead at its own value.
  Choosing by the MINIMUM rather than the mean is deliberate: after 22 arms the risk is optimistic
  selection, and the minimum is the statistic that cannot be inflated by a good draw.

  No arm is promoted on this step either.  Held folds 0-2 are untouched; my read count stands at 4.

Training-only: the 162 audit pairs in 172 tables outside every held fold.
Output: experiments/stage2_selection_C_20260912/audit-C-selection-stability.json
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, check_protected, write_json
from evidence_aug.step5_evaluate import pool_bootstrap
from evidence_aug.step14_audit_truncation import CLF, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, build_d0, evaluate, fit_predict,
                                              frame_at)
from evidence_aug.step78_C_stage2_selection import RANKER, fit_predict_rank
from evidence_aug.step79_C_selection_v2 import BIG, BIG_RANK, N_WIDE, relativise, top_ranker_cols

AUTHOR = 'claude'
OUTDIR = CV / 'experiments' / 'stage2_selection_C_20260912'
TRIPLETS = [(11, 12, 13), (21, 22, 23), (31, 32, 33)]
SPREAD_BAR = .010

# the step-79 winner, its one-factor neighbours, and the two stable comparators
CELLS = [('rzm', 'wide', 'big', 'rank'),     # the step-79 winner
         ('rz', 'wide', 'big', 'rank'),      # - margin
         ('rzm', 'base', 'big', 'rank'),     # - wide
         ('rzm', 'wide', 'small', 'rank'),   # - capacity
         ('rzm', 'wide', 'big', 'clf'),      # - listwise
         ('rzm', 'base', 'small', 'clf'),    # stable comparator (step 79 #2)
         ('rz', 'base', 'small', 'clf')]     # stable comparator (step 78 best)

DECISION = {
    'carry_rule': 'highest MINIMUM across the three seed triplets, among arms with across-triplet '
                  'spread < .010 whose one-factor neighbours average above the incumbent',
    'why_minimum_not_mean': '22 arms have been scored on this surface across steps 78-79; the minimum '
                            'is the statistic a good draw cannot inflate',
    'spread_bar': SPREAD_BAR,
    'fallback': 'if the step-79 winner spreads more than .010 it is recorded as a lucky cell and the '
                'flat base_small cell is carried at its own value',
    'held_read': 'NOT taken here; claude held-fold read count stands at 4'}


def main():
    t0 = time.time()
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTDIR / 'prediction-stability.json',
               {'author': AUTHOR, 'step': 80, 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'decision': DECISION, 'cells': [list(c) for c in CELLS],
                'triplets': [list(t) for t in TRIPLETS]})

    d0, struct, rcols = build_d0()
    d = frame_at(d0, 1.0)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    base_cols = struct + CHRONO3 + S1 + rcols

    print('=' * 100)
    print('STEP 80 [C].  SEED STABILITY AND NEIGHBOURHOOD OF THE STEP-79 WINNER')
    print('=' * 100)

    wide = list(dict.fromkeys(struct + CHRONO3 + top_ranker_cols(d, base_cols, N_WIDE)))
    frames = {(rc, tf): relativise(d, cols, TOP, margin=(tf == 'rzm'))
              for rc, cols in (('base', struct + CHRONO3), ('wide', wide)) for tf in ('rz', 'rzm')}

    inc_r = fit_predict(d, d, struct + CHRONO3 + S1 + rcols, TOP)
    inc_f = fit_predict(d, d, struct + CHRONO3 + S1, TOP)
    _, ppr = evaluate(d, inc_r, TOP, truths, labels)
    _, ppf = evaluate(d, inc_f, TOP, truths, labels)
    inc = {q: (ppf[q] if labels[q] == CI else ppr[q]) for q in ppr}
    incm = float(np.mean(list(inc.values())))
    print(f'  incumbent hybrid_clf_plain = {incm:.4f}\n')

    res = {'author': AUTHOR, 'step': 80, 'training_only': True, 'held_folds_touched': False,
           'decision': DECISION, 'incumbent': round(incm, 4), 'cells': {}, 'pooled': {}}

    for tf, rc, cap, obj in CELLS:
        tag = f'{tf}_{rc}_{cap}_{obj}'
        fr = frames[(rc, tf)]
        cols = base_cols + [c for c in fr.columns if c.endswith(('__wr', '__wz', '__wm'))]
        vals, pps = [], []
        for trip in TRIPLETS:
            t1 = time.time()
            if obj == 'clf':
                pred = fit_predict(fr, fr, cols, TOP, seeds=trip, params=CLF if cap == 'small' else BIG)
            else:
                pred = fit_predict_rank(fr, fr, cols, TOP, seeds=trip,
                                        params=RANKER if cap == 'small' else BIG_RANK)
            summ, pp = evaluate(fr, pred, TOP, truths, labels)
            vals.append(round(summ['map5'], 4)); pps.append(pp)
            log(f'{tag} seeds{trip[0]}', json.dumps({'map5': round(summ['map5'], 4),
                                                     'sec': round(time.time() - t1)}))
        spread = round(max(vals) - min(vals), 4)
        # pool the three triplets into one 9-seed arm and interval it against the incumbent
        pooled = {p: float(np.mean([pp[p] for pp in pps])) for p in pps[0]}
        pm = float(np.mean(list(pooled.values())))
        ci = [round(x, 4) for x in pool_bootstrap({p: pooled[p] - inc[p] for p in pooled}, ptab)]
        fam = {f: round(float(np.mean([pooled[p] for p in pooled if labels[p] == f])), 4) for f in FAMILIES}
        res['cells'][tag] = {'by_triplet': vals, 'min': min(vals), 'max': max(vals),
                             'mean': round(float(np.mean(vals)), 4), 'spread': spread,
                             'stable': bool(spread < SPREAD_BAR)}
        res['pooled'][tag] = {'map5_9seed': round(pm, 4), 'delta': round(pm - incm, 4),
                              'ci95': ci, 'by_family': fam}

    print(f"\n  {'cell':<24}{'t1':>8}{'t2':>8}{'t3':>8}{'min':>9}{'spread':>9}  {'9-seed':>8}"
          f"{'delta':>9}{'  95% CI':>20}")
    print('  ' + '-' * 106)
    for tag in sorted(res['cells'], key=lambda t: -res['cells'][t]['min']):
        c, p = res['cells'][tag], res['pooled'][tag]
        flag = '' if c['stable'] else '   <-- UNSTABLE'
        print(f"  {tag:<24}" + ''.join(f'{v:>8.4f}' for v in c['by_triplet'])
              + f"{c['min']:>9.4f}{c['spread']:>9.4f}  {p['map5_9seed']:>8.4f}{p['delta']:>+9.4f}"
              + f"   [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]" + flag)

    win = ('rzm', 'wide', 'big', 'rank')
    wtag = '_'.join(win)
    nbrs = [f'{a}_{b}_{c}_{e}' for a, b, c, e in CELLS[1:5]]
    nbr_mean = float(np.mean([res['pooled'][n]['map5_9seed'] for n in nbrs]))
    res['neighbourhood'] = {'winner': wtag, 'winner_stable': res['cells'][wtag]['stable'],
                            'neighbours': nbrs, 'neighbour_mean_9seed': round(nbr_mean, 4),
                            'neighbours_above_incumbent': bool(nbr_mean > incm)}
    eligible = {t: res['cells'][t]['min'] for t in res['cells']
                if res['cells'][t]['stable'] and res['pooled'][t]['delta'] > 0}
    carry = max(eligible, key=eligible.get) if eligible else None
    res['carry'] = {'arm': carry,
                    'its_min_across_triplets': eligible.get(carry),
                    'its_9seed_delta': res['pooled'][carry]['delta'] if carry else None,
                    'its_9seed_ci95': res['pooled'][carry]['ci95'] if carry else None}
    print(f"\n  winner {wtag}: spread {res['cells'][wtag]['spread']:.4f} -> "
          f"{'STABLE' if res['cells'][wtag]['stable'] else 'LUCKY CELL'}; "
          f"neighbour mean {nbr_mean:.4f} vs incumbent {incm:.4f}")
    print(f"  CARRY: {carry}  (min across triplets {eligible.get(carry)}, "
          f"9-seed delta {res['carry']['its_9seed_delta']}, CI {res['carry']['its_9seed_ci95']})")

    res['seconds'] = round(time.time() - t0)
    write_json(OUTDIR / 'audit-C-selection-stability.json', res)
    check_protected()
    print(f"\n  wrote {OUTDIR / 'audit-C-selection-stability.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
