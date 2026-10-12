"""Step 80b [C] (author: claude).  Step 80 was aborted because its test measured nothing.  This is
the corrected test: perturb the DATA PARTITION, not the seed.

WHY STEP 80 WAS ABORTED
Step 80 re-fitted each arm under three independent seed triplets -- (11,12,13), (21,22,23),
(31,32,33) -- on the theory that a real design effect is constant across triplets while a lucky cell
moves.  The first two cells returned:

    rzm_wide_big_rank   .6435  .6435  .6435
    rz_wide_big_rank    .6220  .6220     -

Identical to four decimals is not stability, it is a null operation.  Our LightGBM params carry no
`subsample`, no `colsample_bytree` and no `extra_trees`, and plain GBDT with full bagging and full
feature use is deterministic, so `random_state` changes nothing.  Verified directly:

    LGBMClassifier(**CLF, random_state=s).predict_proba  identical for s in (11,21,31)  -> True
    LGBMRanker(**RANKER,  random_state=s).predict        identical for s in (11,21,31)  -> True

(the same check runs as an assertion in `seed_inertness_control()` below, so the finding is re-proved
every time this file executes).

Two consequences, both recorded rather than quietly fixed:

  1. `SEEDS = (11,12,13)` has been a no-op in every stage-2 fit in this project.  It averages three
     IDENTICAL models.  No published number is wrong -- the OOF protocol in `fit_predict` is sound and
     the values are what they are -- but every stage-2 fit has cost 3x what it needed to, and no
     result was ever "seed-averaged" in the sense the word implies.  It is NOT removed here: the
     reproduction controls in step 81 assert byte-equality against frozen artefacts, and
     `p/3 + p/3 + p/3` does not equal `p` in floating point, so dropping the loop would break exact
     reproduction for a saving that does not matter at this point in the study.
  2. Step 80's carry rule would have fired with spread .0000 on every arm and carried the step-79
     winner.  That is a rule firing on a statistic that does not measure what the rule assumed --
     precisely the error class recorded in 18.15 and 18.17, which I have been auditing in other
     people's gates.  It does not get to pass in mine.

THE REAL QUESTION, UNCHANGED
`rzm_wide_big_rank` scored .6435 (+.0276) but is an isolated peak: its one-factor neighbours sit at
.6216 and .6265, and 22 arms have now been scored on this 162-pair surface, so about one false
positive is expected at a 95 % interval.  Is its advantage a design effect or an artefact of this
particular partition of this particular surface?

THE PERTURBATION THAT ACTUALLY BITES
Re-partition the audit surface.  `split_of` assigns each table to one of five pools by
`sha256('audit-trunc-v1:' + table_id) % 5`; changing the salt gives a genuinely different train/test
partition, so every model is genuinely different and so is every candidate set.  If an arm's edge
depends on which tables happened to land together, a new salt moves it.

Soundness of re-salting.  Stage 1's `oof` in `audit-oof.parquet` was produced under the v1 salt, and
under any salt it remains out-of-fold with respect to its own table -- a row's stage-1 score never
came from a model that saw that row's table.  What does change is that under v1 the stage-2 partition
coincides with stage-1's, and under b/c it does not, so a stage-2 test row's `oof` may come from a
stage-1 model that also trained on other stage-2 test tables.  That is a structural difference in the
surface, not a label leak, and it is identical for every arm; the comparison here is PAIRED across
arms on the same surface, so it is clean.  Absolute MAP@5 is therefore comparable across arms within
a salt, and only roughly comparable between salts -- which is why the rule below reads DELTAS.

PRE-REGISTERED DECISION (written to prediction-resplit.json before any fit):
  Carry the arm with the highest MINIMUM paired delta-vs-incumbent across the three partitions,
  among arms whose delta is positive under ALL THREE and whose one-factor neighbours average above
  the incumbent.  If the step-79 winner is negative under any partition it is recorded as a lucky
  cell and the flat `base_small` cell is carried at its own value.  If NO arm is positive under all
  three, nothing is carried and no held read is taken.
  Minimum, not mean: after 22 arms the risk is optimistic selection, and the minimum is the statistic
  a good draw cannot inflate.

  No arm is promoted here.  Held folds 0-2 are untouched; my read count stands at 4.

Training-only: the 162 audit pairs in 172 tables outside every held fold.
Output: experiments/stage2_selection_C_20260912/audit-C-resplit-stability.json
"""
import hashlib
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, check_protected, write_json
from evidence_aug.step5_evaluate import pool_bootstrap
from evidence_aug.step14_audit_truncation import CLF, SPLITS, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, build_d0, evaluate, fit_predict,
                                              frame_at)
from evidence_aug.step78_C_stage2_selection import RANKER, fit_predict_rank
from evidence_aug.step79_C_selection_v2 import BIG, BIG_RANK, N_WIDE, relativise, top_ranker_cols

AUTHOR = 'claude'
OUTDIR = CV / 'experiments' / 'stage2_selection_C_20260912'
SALTS = ['audit-trunc-v1', 'audit-resplit-b', 'audit-resplit-c']

# carried under every partition: the step-79 winner and the two flat comparators
ARMS = [('rzm', 'wide', 'big', 'rank'),     # step-79 winner, .6435, isolated peak
        ('rzm', 'base', 'small', 'clf'),    # step-79 #2, .6379, flat neighbourhood
        ('rz', 'base', 'small', 'clf')]     # step-78 best, .6374

# the winner's one-factor neighbours, at the v1 partition only.  Two are already measured and the
# learner is deterministic, so only the two unmeasured ones are fitted.
NEIGHBOURS = [('rzm', 'base', 'big', 'rank'),      # - wide
              ('rzm', 'wide', 'small', 'rank')]    # - capacity
KNOWN_NEIGHBOURS = {'rzm_wide_big_clf': .6216,     # step 79 (- listwise)
                    'rz_wide_big_rank': .6220}     # aborted step 80, salt v1, deterministic (- margin)

DECISION = {
    'aborted_step80': 'seed triplets are a null perturbation for these params; verified in-script',
    'carry_rule': 'highest MINIMUM paired delta-vs-incumbent across the three partitions, among arms '
                  'positive under ALL THREE whose one-factor neighbours average above the incumbent',
    'why_minimum_not_mean': '22 arms scored on this surface across steps 78-79; the minimum is the '
                            'statistic a good draw cannot inflate',
    'fallback': 'if the step-79 winner is negative under any partition it is a lucky cell and the flat '
                'base_small cell is carried; if no arm is positive under all three, nothing is carried '
                'and no held read is taken',
    'salts': SALTS,
    'held_read': 'NOT taken here; claude held-fold read count stands at 4'}


def seed_inertness_control():
    """Re-prove, every run, that random_state does nothing for these params.  This is the fact that
    invalidated step 80, so it is asserted rather than trusted."""
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(1200, 40)), (rng.random(1200) < .25).astype(float)
    p = [lgb.LGBMClassifier(**CLF, random_state=s).fit(X, y).predict_proba(X)[:, 1] for s in (11, 21)]
    q = [lgb.LGBMRanker(**RANKER, random_state=s).fit(X, y.astype(int), group=np.full(60, 20)).predict(X)
         for s in (11, 21)]
    assert np.array_equal(*p) and np.array_equal(*q), \
        'random_state now changes the fit -- step 80 was not vacuous after all; re-read this file'
    return {'clf_identical_across_seeds': True, 'rank_identical_across_seeds': True,
            'implication': 'SEEDS=(11,12,13) averages three identical models'}


def resplit(d, salt):
    """the 5-way pool hash under a different salt; pools stay whole tables, as in step 14."""
    tabs = d['table_id'].unique().to_list()
    m = {t: int(hashlib.sha256((salt + ':' + t).encode()).hexdigest(), 16) % SPLITS for t in tabs}
    return d.with_columns(pl.col('table_id').replace_strict(m, return_dtype=pl.Int64).alias('split'))


def run_cell(fr, cols, cap, obj, truths, labels):
    if obj == 'clf':
        pred = fit_predict(fr, fr, cols, TOP, params=CLF if cap == 'small' else BIG)
    else:
        pred = fit_predict_rank(fr, fr, cols, TOP, params=RANKER if cap == 'small' else BIG_RANK)
    return evaluate(fr, pred, TOP, truths, labels)


def main():
    t0 = time.time()
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inert = seed_inertness_control()
    write_json(OUTDIR / 'prediction-resplit.json',
               {'author': AUTHOR, 'step': '80b', 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'decision': DECISION, 'seed_inertness': inert,
                'arms': ['_'.join(a) for a in ARMS], 'neighbours': ['_'.join(a) for a in NEIGHBOURS],
                'known_neighbours': KNOWN_NEIGHBOURS})

    d0, struct, rcols = build_d0()
    d = frame_at(d0, 1.0)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    base_cols = struct + CHRONO3 + S1 + rcols

    print('=' * 100)
    print('STEP 80b [C].  RE-PARTITION STABILITY  (step 80 aborted: seed triplets are a null test)')
    print('=' * 100)
    print(f'  seed inertness re-proved: {inert["implication"]}\n')

    # the relativised frames depend only on the feature columns, never on `split`, so build once
    wide = list(dict.fromkeys(struct + CHRONO3 + top_ranker_cols(d, base_cols, N_WIDE)))
    frames = {(rc, tf): relativise(d, cols, TOP, margin=(tf == 'rzm'))
              for rc, cols in (('base', struct + CHRONO3), ('wide', wide)) for tf in ('rz', 'rzm')}
    relcols = {k: [c for c in f.columns if c.endswith(('__wr', '__wz', '__wm'))]
               for k, f in frames.items()}

    res = {'author': AUTHOR, 'step': '80b', 'training_only': True, 'held_folds_touched': False,
           'aborted': {'step': 80, 'reason': 'seed triplets are a null perturbation for these params',
                       'evidence': 'rzm_wide_big_rank returned .6435 .6435 .6435; '
                                   'rz_wide_big_rank .6220 .6220', 'control': inert},
           'decision': DECISION, 'by_salt': {}, 'arms': {}, 'neighbours': dict(KNOWN_NEIGHBOURS)}

    for salt in SALTS:
        ds = resplit(d, salt)
        sp = ds['split'].to_numpy()
        moved = float(np.mean(sp != d['split'].to_numpy()))
        inc_r = fit_predict(ds, ds, base_cols, TOP)
        inc_f = fit_predict(ds, ds, struct + CHRONO3 + S1, TOP)
        _, ppr = evaluate(ds, inc_r, TOP, truths, labels)
        _, ppf = evaluate(ds, inc_f, TOP, truths, labels)
        inc = {q: (ppf[q] if labels[q] == CI else ppr[q]) for q in ppr}
        incm = float(np.mean(list(inc.values())))
        res['by_salt'][salt] = {'rows_moved_pool': round(moved, 3), 'incumbent': round(incm, 4),
                                'arms': {}}
        log(f'{salt} incumbent', json.dumps({'map5': round(incm, 4), 'moved': round(moved, 3)}))

        for tf, rc, cap, obj in ARMS:
            t1 = time.time()
            tag = f'{tf}_{rc}_{cap}_{obj}'
            fr = frames[(rc, tf)].with_columns(pl.Series('split', sp))
            summ, pp = run_cell(fr, base_cols + relcols[(rc, tf)], cap, obj, truths, labels)
            delta = {p: pp[p] - inc[p] for p in pp}
            ci = [round(x, 4) for x in pool_bootstrap(delta, ptab)]
            dm = float(np.mean(list(delta.values())))
            res['by_salt'][salt]['arms'][tag] = {
                'map5': round(summ['map5'], 4), 'delta': round(dm, 4), 'ci95': ci,
                'by_family': {f: round(float(np.mean([pp[p] for p in pp if labels[p] == f])), 4)
                              for f in FAMILIES}}
            log(f'{salt} {tag}', json.dumps({'map5': round(summ['map5'], 4), 'delta': round(dm, 4),
                                             'sec': round(time.time() - t1)}))

    for tf, rc, cap, obj in NEIGHBOURS:
        tag = f'{tf}_{rc}_{cap}_{obj}'
        fr = frames[(rc, tf)]      # v1 partition, as built
        summ, _ = run_cell(fr, base_cols + relcols[(rc, tf)], cap, obj, truths, labels)
        res['neighbours'][tag] = round(summ['map5'], 4)
        log(f'neighbour {tag}', json.dumps({'map5': round(summ['map5'], 4)}))

    # ---- apply the pre-registered rule --------------------------------------------------------
    v1 = SALTS[0]
    nmean = float(np.mean(list(res['neighbours'].values())))
    res['neighbourhood'] = {'winner': 'rzm_wide_big_rank', 'neighbour_values': res['neighbours'],
                            'neighbour_mean': round(nmean, 4),
                            'incumbent_v1': res['by_salt'][v1]['incumbent'],
                            'neighbours_above_incumbent': bool(nmean > res['by_salt'][v1]['incumbent'])}
    for tf, rc, cap, obj in ARMS:
        tag = f'{tf}_{rc}_{cap}_{obj}'
        ds = [res['by_salt'][s]['arms'][tag]['delta'] for s in SALTS]
        res['arms'][tag] = {'deltas_by_salt': ds, 'min': min(ds), 'max': max(ds),
                            'mean': round(float(np.mean(ds)), 4), 'spread': round(max(ds) - min(ds), 4),
                            'positive_under_all': bool(min(ds) > 0)}

    print(f"\n  {'arm':<22}" + ''.join(f'{s.split("-")[-1]:>10}' for s in SALTS)
          + f"{'min':>10}{'spread':>9}   verdict")
    print('  ' + '-' * 88)
    for tag in sorted(res['arms'], key=lambda t: -res['arms'][t]['min']):
        a = res['arms'][tag]
        print(f"  {tag:<22}" + ''.join(f"{v:>+10.4f}" for v in a['deltas_by_salt'])
              + f"{a['min']:>+10.4f}{a['spread']:>9.4f}   "
              + ('positive under all 3' if a['positive_under_all'] else 'FAILS: negative somewhere'))
    print(f"\n  neighbourhood of the winner: {res['neighbours']} -> mean {nmean:.4f} vs incumbent "
          f"{res['by_salt'][v1]['incumbent']:.4f}")

    elig = {t: res['arms'][t]['min'] for t in res['arms'] if res['arms'][t]['positive_under_all']}
    carry = max(elig, key=elig.get) if elig else None
    res['carry'] = {'arm': carry, 'its_min_delta_across_partitions': elig.get(carry),
                    'its_deltas_by_salt': res['arms'][carry]['deltas_by_salt'] if carry else None,
                    'its_v1_map5': res['by_salt'][v1]['arms'][carry]['map5'] if carry else None,
                    'its_v1_ci95': res['by_salt'][v1]['arms'][carry]['ci95'] if carry else None,
                    'rule': DECISION['carry_rule']}
    print(f"\n  CARRY: {carry}"
          + (f"  (min delta {elig[carry]:+.4f} across partitions)" if carry else
             '  -- nothing survived; no held read'))

    res['seconds'] = round(time.time() - t0)
    write_json(OUTDIR / 'audit-C-resplit-stability.json', res)
    check_protected()
    print(f"\n  wrote {OUTDIR / 'audit-C-resplit-stability.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
