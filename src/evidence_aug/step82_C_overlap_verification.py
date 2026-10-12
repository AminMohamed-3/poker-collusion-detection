"""
author: claude

step82_C_overlap_verification.py -- verify OBSERVATIONS 18.6's HELD column, which has no saved
artefact, and measure the same statistic for the relative-encoding arm.

Why this exists.  18.6 reports top-5 overlap with the stage-1 list by predicted family:

    predicted family        held    evaluation
    directed_transfer       .890    .786
    soft_play               .908    .905
    coordinated_isolation   .769    .546

and from the held-vs-evaluation gap infers `delta_CI ~ -.048` on evaluation.  That inference is
load-bearing: it is the basis for "the isolation branch is the broken component", which I put to
Codex twice.  A survey of the repo found that the EVALUATION column is reproducible from
`experiments/public_ledger_20260912/step68b-routing.json` (DT .7860, SP .9046, CI .5461, exact) but
the HELD column is not saved anywhere -- step68b only persisted per-fold, all-family-pooled values
(.8898 / .8578 / .8604).  So half of a load-bearing comparison is unverified.

Step 81 left behind per-hand scores for held folds 0-2 for chrono1 (the stage-1 baseline),
inc_struct, inc_r382, inc_hybrid and the new arm, and its reproduction controls proved chrono1 and
inc_struct reproduce the frozen artefacts byte-exactly.  So the held column can be recomputed from
those scores directly, with the pooled per-fold values as a reproduction control.

THIS CONSUMES NO HELD READ.  The overlap statistic is label-free: it compares two models' top-5
lists with each other, never with held truth.  That is exactly the property that lets 18.6 compute
it on the evaluation set at all.  No file under outputs_* is written.
"""
import io
import json
import time

import numpy as np
import polars as pl

from evidence_aug.paths import CV, check_protected, write_json
from evidence_aug.step5_evaluate import TIE_SEED

HELD = CV / 'experiments' / 'stage2_selection_C_20260912' / 'held'
OUT = CV / 'experiments' / 'stage2_selection_C_20260912' / 'overlap-C-verification.json'
FOLDS = (0, 1, 2)
CANDS = ('chrono1', 'inc_struct', 'inc_r382', 'inc_hybrid', 'new_rz_base_small_clf')
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')

# what the record claims, so the script states its own target before computing anything
OBS_18_6_HELD = {'directed_transfer': .890, 'soft_play': .908, 'coordinated_isolation': .769}
OBS_18_6_EVAL = {'directed_transfer': .786, 'soft_play': .905, 'coordinated_isolation': .546}
STEP68B_POOLED_HELD = {0: .8898, 1: .8578, 2: .8604}   # the only saved held numbers


def top5(frame):
    """the production top-5 rule: sort by score desc, break ties with the TIE_SEED draw.

    step5_evaluate draws the tie column over the WHOLE frame after sorting by (pair_id, hand_id),
    so the draw depends on frame height; reproduced here per candidate frame.  Ties in continuous
    LightGBM scores are vanishingly rare, so this affects nothing measurable -- it is here so the
    selection rule is identical to production rather than merely equivalent.
    """
    f = frame.sort('pair_id', 'hand_id')
    f = f.with_columns(pl.Series('tie', np.random.default_rng(TIE_SEED).random(f.height)))
    return (f.sort(['pair_id', 'score', 'tie'], descending=[False, True, False])
            .group_by('pair_id', maintain_order=True)
            .agg(pl.col('hand_id').head(5).alias('hands')))


def overlap(a, b):
    """mean |A n B| / 5 over the pairs present in both, and the per-pair vector."""
    j = a.join(b, on='pair_id', suffix='_b')
    v = [len(set(x) & set(y)) / 5.0 for x, y in zip(j['hands'].to_list(), j['hands_b'].to_list())]
    return float(np.mean(v)), v, j['pair_id'].to_list()


def main():
    t0 = time.time()
    protected = check_protected()

    lists, roles = {}, {}
    for k in FOLDS:
        roles[k] = pl.read_csv(HELD / 'scores' / f'fold{k}-roles.csv')
        for c in CANDS:
            lists[(k, c)] = top5(pl.read_parquet(HELD / 'scores' / f'fold{k}-{c}.parquet'))

    out = {'author': 'claude', 'step': 82, 'consumes_held_read': False,
           'why_no_read': 'the statistic is model-vs-model top-5 overlap; held truth is never opened',
           'protected': protected, 'obs_18_6_claimed_held': OBS_18_6_HELD,
           'obs_18_6_claimed_eval': OBS_18_6_EVAL}

    # ---------------------------------------------------------------- reproduction control
    # step68b saved pooled-over-families held overlap chrono1 -> struct, per fold.
    ctrl = {}
    for k in FOLDS:
        m, _, _ = overlap(lists[(k, 'chrono1')], lists[(k, 'inc_struct')])
        ctrl[k] = {'recomputed': round(m, 4), 'step68b_saved': STEP68B_POOLED_HELD[k],
                   'delta': round(m - STEP68B_POOLED_HELD[k], 4)}
    out['reproduction_control_pooled_per_fold'] = ctrl
    worst = max(abs(v['delta']) for v in ctrl.values())
    out['reproduction_control_max_abs_delta'] = round(worst, 4)
    out['reproduction_control_verdict'] = ('reproduces step68b' if worst <= .002 else
                                           'DOES NOT REPRODUCE step68b -- read this before using anything below')

    # ---------------------------------------------------------------- the 18.6 held column
    # pooled across folds, grouped by PREDICTED family (18.6 groups the evaluation side by the
    # submission's predicted_behavior column, so the held side must use predicted, not true, family)
    def by_family(cand_a, cand_b):
        acc = {f: [] for f in FAMS}
        for k in FOLDS:
            m, v, pids = overlap(lists[(k, cand_a)], lists[(k, cand_b)])
            fam = dict(zip(roles[k]['pair_id'].to_list(), roles[k]['predicted_behavior'].to_list()))
            for p, x in zip(pids, v):
                acc[fam[p]].append(x)
        return {f: {'overlap': round(float(np.mean(acc[f])), 4), 'n': len(acc[f])} for f in FAMS}

    held = by_family('chrono1', 'inc_struct')
    out['held_by_predicted_family_chrono1_to_struct'] = held
    out['verification'] = {
        f: {'claimed_18_6': OBS_18_6_HELD[f], 'recomputed': held[f]['overlap'],
            'delta': round(held[f]['overlap'] - OBS_18_6_HELD[f], 4), 'n': held[f]['n']}
        for f in FAMS}

    # ---------------------------------------------------------------- the transfer question
    # 19.4's mechanism claim: a within-pair rank is invariant to a monotone scale shift between
    # development and evaluation, so the relative-encoding arm should rewrite the stage-1 list LESS
    # aggressively on the family where struct is suspected of over-rewriting.  Measured here on held
    # only -- the evaluation half needs a production fit and is not attempted in this script.
    for a, b, tag in [('chrono1', 'inc_r382', 'chrono1_to_r382'),
                      ('chrono1', 'inc_hybrid', 'chrono1_to_hybrid'),
                      ('chrono1', 'new_rz_base_small_clf', 'chrono1_to_new'),
                      ('inc_hybrid', 'new_rz_base_small_clf', 'hybrid_to_new')]:
        out['held_by_predicted_family_' + tag] = by_family(a, b)

    out['seconds'] = round(time.time() - t0, 1)
    assert check_protected() == protected
    write_json(OUT, out)

    # ---------------------------------------------------------------- report
    L = []
    L.append('reproduction control (pooled chrono1 -> struct, per fold, vs step68b saved):')
    for k in FOLDS:
        c = ctrl[k]
        L.append(f"  fold{k}  recomputed {c['recomputed']:.4f}  saved {c['step68b_saved']:.4f}  "
                 f"delta {c['delta']:+.4f}")
    L.append(f"  -> {out['reproduction_control_verdict']}")
    L.append('')
    L.append('OBS 18.6 HELD column, verified:')
    L.append('  family                  claimed  recomputed   delta     n')
    for f in FAMS:
        v = out['verification'][f]
        L.append(f"  {f:<22} {v['claimed_18_6']:.3f}     {v['recomputed']:.4f}   {v['delta']:+.4f}  {v['n']:4d}")
    L.append('')
    L.append('held overlap with stage 1, by predicted family, all candidates:')
    L.append(f"  {'comparison':<26}" + ''.join(f'{f[:14]:>16}' for f in FAMS))
    for tag in ('chrono1_to_struct', 'chrono1_to_r382', 'chrono1_to_hybrid', 'chrono1_to_new'):
        d = held if tag == 'chrono1_to_struct' else out['held_by_predicted_family_' + tag]
        L.append(f"  {tag:<26}" + ''.join(f"{d[f]['overlap']:>16.4f}" for f in FAMS))
    L.append('')
    L.append('for reference, the EVALUATION column of 18.6 (saved, reproducible):')
    L.append(f"  {'stage1_to_struct(eval)':<26}" + ''.join(f'{OBS_18_6_EVAL[f]:>16.4f}' for f in FAMS))
    text = '\n'.join(L)
    print(text)
    io.open(OUT.with_suffix('.md'), 'w', encoding='utf-8').write(
        '# step 82 [C]: verification of OBS 18.6\'s held column\n\n```\n' + text + '\n```\n')
    print(f"\n  wrote {OUT} in {out['seconds']} s")


if __name__ == '__main__':
    main()
