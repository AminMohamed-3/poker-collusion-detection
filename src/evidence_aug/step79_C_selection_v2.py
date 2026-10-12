"""Step 79 [C] (author: claude).  Escalation of the within-pair relative encoding of step 78.

Step 78 result.  Within-pair relative features (rank and z-score of each structural column computed
inside the pair's own 20 candidates) beat their plain counterpart in ALL FOUR cells: ref_clf +.0206,
ref_rank +.0338, r382_clf +.0152, r382_rank +.0118.  The listwise objective ALONE is null
(ref -.0004, r382 -.0027) and only pays in combination, and only on ref.  Best arm r382_clf_rel
.6374, +.0215 over the shipping stage 2, paired 95 % CI [-.0002, +.0422] -- over the +.010 bar with
the lower bound two ten-thousandths short of zero, so the step-78 gate correctly did not fire.

Two things step 78 settled that this step inherits rather than re-tests:

  * THE CANDIDATE COUNT IS NOT BINDING.  select_oracle is .5823 / .8153 / .9481 / .9852 at K = 5 / 10
    / 20 / 30 while we score .6374 at K = 20.  Raising K raises a ceiling we are nowhere near and only
    adds distractors; lowering it throws away reachable truth.  TOP stays 20.
  * THE CI ROUTING IS STALE.  Production routes coordinated_isolation to the `ref` design because ref
    beat r382 on that family when the hybrid was built.  With relative features r382 scores .6230 on
    CI against ref's .5979 and the shipping hybrid's .5645, so the routing now COSTS .0585 on the
    family it exists to protect.  Arms here are un-routed r382 unless a family split says otherwise,
    and the routed hybrid is reported beside them so the cost stays visible.

What this step varies, all on the r382 base with the learner held at the incumbent LightGBM (Codex
owns the learner axis):

  TRANSFORM  rz  = within-pair rank + z          (step 78)
             rzm = rank + z + margin-to-pair-best (v - max(v) over the pair's candidates)
  RELCOLS    base = structural + chrono3 (73 columns)
             wide = base + the top-50 ranker columns by gain, which are the columns carrying the
                    r382 advantage in the first place and the ones step 78 left on an absolute scale
  CAPACITY   small = the incumbent CLF (150 trees, 7 leaves, depth 3)
             big   = 300 trees, 15 leaves, depth 5, min_child_samples 20 -- because 599 features on
                     ~2600 training rows at depth 3 may simply be underfit, and if capacity is the
                     binding constraint that is a cheaper finding than any new feature

Reproduction control: arm `rz_base_small_clf` must reproduce step 78's r382_clf_rel = 0.6374 exactly.

PRE-REGISTERED GATE (written before any fit; identical bar to step 78 so the two are comparable).
  incumbent : hybrid_clf_plain, the stage 2 that ships (audit MAP@5 .6159)
  promote to a held read only if > +.010 MAP@5, paired 95 % lower bound > 0, no family below -.005.
  STANDING CHECK (18.15 / 18.17): the gate must reject an arm already known to be worse.
  `ref_clf_plain` = .5867, delta -.0292 < +.010 -> FAILS.  Carried from step 78 and re-asserted here.
  Note on multiplicity: step 78 tried 12 arms and this step tries 10 on the same audit set.  The
  audit set is the designated selection surface and folds 0-2 are the confirmation, but the arm count
  is recorded so the held read is judged knowing how many arms preceded it.

Training-only: the 162 audit pairs in 172 tables outside every held fold.  No held read here.
Output: experiments/stage2_selection_C_20260912/audit-C-selection-v2.json
"""
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
from evidence_aug.step14_audit_truncation import CLF, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, build_d0, evaluate, fit_predict,
                                              frame_at)
from evidence_aug.step78_C_stage2_selection import GATE, RANKER, SEEDS, fit_predict_rank

AUTHOR = 'claude'
OUTDIR = CV / 'experiments' / 'stage2_selection_C_20260912'
BIG = dict(n_estimators=300, num_leaves=15, max_depth=5, min_child_samples=20,
           learning_rate=.05, n_jobs=2, verbosity=-1)
BIG_RANK = dict(objective='lambdarank', label_gain=[0, 1], lambdarank_truncation_level=TOP, **BIG)
STEP78_CONTROL = 0.6374
N_WIDE = 50


def relativise(d, cols, topk, margin, rankcol='oof_rank'):
    """within-pair rank, z-score and (optionally) margin-to-best over the pair's candidate set.

    `rankcol` is the stage-1 rank column: 'oof_rank' on the audit frame, 's1_rank' on a held fold.
    Non-candidates get rank 0 / z 0 / margin 0; they never enter a fit or an evaluation because both
    are masked to the same candidate set.
    """
    cand = pl.col(rankcol) <= topk
    exprs = []
    for c in cols:
        v = pl.when(cand).then(pl.col(c)).otherwise(None)
        exprs.append(v.rank('average').over('pair_id').fill_null(0.0).alias(f'{c}__wr'))
        exprs.append(((v - v.mean().over('pair_id')) / (v.std().over('pair_id') + 1e-9))
                     .fill_null(0.0).alias(f'{c}__wz'))
        if margin:
            exprs.append((v - v.max().over('pair_id')).fill_null(0.0).alias(f'{c}__wm'))
    return d.with_columns(exprs)


def top_ranker_cols(d, cols, n):
    """the n ranker columns carrying most gain, fitted once on the audit rows (selection surface)."""
    m = (d['oof_rank'] <= TOP).to_numpy()
    clf = lgb.LGBMClassifier(**CLF, random_state=SEEDS[0])
    clf.fit(d.select(cols).to_numpy().astype(np.float64)[m], d['relevance'].to_numpy().astype(float)[m])
    imp = clf.booster_.feature_importance('gain')
    return [cols[i] for i in np.argsort(-imp)[:n]]


def main():
    t0 = time.time()
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTDIR / 'prediction-v2.json',
               {'author': AUTHOR, 'step': 79, 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()), 'gate': GATE,
                'inherits_from_step78': {'candidate_count_not_binding': 'TOP stays 20',
                                         'CI_routing_stale': 'arms are un-routed r382; the routed '
                                                             'hybrid is reported beside them'},
                'reproduction_control': {'rz_base_small_clf': STEP78_CONTROL},
                'arms_tried_on_this_surface': {'step78': 12, 'step79': 10}})

    d0, struct, rcols = build_d0()
    d = frame_at(d0, 1.0)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    base_cols = struct + CHRONO3 + S1 + rcols

    print('=' * 100)
    print('STEP 79 [C].  ESCALATING THE WITHIN-PAIR RELATIVE ENCODING')
    print('=' * 100)

    wide = struct + CHRONO3 + top_ranker_cols(d, base_cols, N_WIDE)
    wide = list(dict.fromkeys(wide))
    print(f'\n  relcol sets: base = {len(struct + CHRONO3)} cols, wide = {len(wide)} cols '
          f'(base + top-{N_WIDE} ranker by gain, deduplicated)')

    frames = {}
    for rc, cols in (('base', struct + CHRONO3), ('wide', wide)):
        for tf in ('rz', 'rzm'):
            frames[(rc, tf)] = relativise(d, cols, TOP, margin=(tf == 'rzm'))

    res = {'author': AUTHOR, 'step': 79, 'training_only': True, 'held_folds_touched': False,
           'gate': GATE, 'n_wide': len(wide), 'arms': {}, 'paired_vs_incumbent': {}, 'gate_result': {}}

    # the incumbent that the gate protects, rebuilt here so the comparison is paired on this frame
    inc_r = fit_predict(d, d, struct + CHRONO3 + S1 + rcols, TOP)
    inc_f = fit_predict(d, d, struct + CHRONO3 + S1, TOP)
    _, ppr = evaluate(d, inc_r, TOP, truths, labels)
    _, ppf = evaluate(d, inc_f, TOP, truths, labels)
    inc = {q: (ppf[q] if labels[q] == CI else ppr[q]) for q in ppr}
    res['incumbent_reproduced'] = round(float(np.mean(list(inc.values()))), 4)
    print(f"  incumbent hybrid_clf_plain reproduced at {res['incumbent_reproduced']:.4f}")

    per_pair_all = {}
    ARMS = [('rz', 'base', 'small', 'clf'), ('rzm', 'base', 'small', 'clf'),
            ('rz', 'wide', 'small', 'clf'), ('rzm', 'wide', 'small', 'clf'),
            ('rz', 'base', 'big', 'clf'), ('rzm', 'wide', 'big', 'clf'),
            ('rz', 'wide', 'big', 'clf'), ('rzm', 'base', 'big', 'clf'),
            ('rz', 'wide', 'small', 'rank'), ('rzm', 'wide', 'big', 'rank')]
    for tf, rc, cap, obj in ARMS:
        tag = f'{tf}_{rc}_{cap}_{obj}'
        fr = frames[(rc, tf)]
        extra = [c for c in fr.columns if c.endswith(('__wr', '__wz', '__wm'))]
        cols = base_cols + extra
        t1 = time.time()
        if obj == 'clf':
            pred = fit_predict(fr, fr, cols, TOP, params=CLF if cap == 'small' else BIG)
        else:
            pred = fit_predict_rank(fr, fr, cols, TOP,
                                    params=RANKER if cap == 'small' else BIG_RANK)
        summ, per_pair = evaluate(fr, pred, TOP, truths, labels)
        per_pair_all[tag] = per_pair
        res['arms'][tag] = {'map5': round(summ['map5'], 4), 'n_features': len(cols),
                            'by_family': {f: round(v, 4) for f, v in summ['by_family'].items()}}
        log(tag, json.dumps({'map5': round(summ['map5'], 4), 'nf': len(cols),
                             'sec': round(time.time() - t1)}))

    got = res['arms']['rz_base_small_clf']['map5']
    assert abs(got - STEP78_CONTROL) < 1e-9, f'control FAILED: {got} vs step 78 {STEP78_CONTROL}'
    print(f'\n  reproduction control PASSED: rz_base_small_clf = {got:.4f} == step 78 r382_clf_rel\n')

    print(f"  {'arm':<24}{'MAP@5':>9}{'vs inc':>10}{'  95% CI':>20}   "
          + '  '.join(f'{f[:9]:>9}' for f in FAMILIES) + '   gate')
    print('  ' + '-' * 110)
    for tag in sorted(per_pair_all, key=lambda t: -res['arms'][t]['map5']):
        pp = per_pair_all[tag]
        m = res['arms'][tag]['map5']
        ci = [round(x, 4) for x in pool_bootstrap({p: pp[p] - inc[p] for p in pp}, ptab)]
        famd = {f: round(res['arms'][tag]['by_family'][f]
                         - float(np.mean([inc[p] for p in inc if labels[p] == f])), 4) for f in FAMILIES}
        ok = bool(m - res['incumbent_reproduced'] > GATE['promote_if']['beats_incumbent_by']
                  and ci[0] > 0 and min(famd.values()) > GATE['promote_if']['family_floor'])
        res['paired_vs_incumbent'][tag] = {'delta': round(m - res['incumbent_reproduced'], 4),
                                           'ci95': ci, 'by_family_delta': famd}
        res['gate_result'][tag] = ok
        print(f"  {tag:<24}{m:>9.4f}{m - res['incumbent_reproduced']:>+10.4f}   [{ci[0]:+.4f}, {ci[1]:+.4f}]   "
              + '  '.join(f"{res['arms'][tag]['by_family'][f]:>9.4f}" for f in FAMILIES)
              + ('   CLEARS' if ok else ''))

    passed = [t for t, v in res['gate_result'].items() if v]
    res['arms_clearing_gate'] = passed
    print(f"\n  arms clearing the gate: {passed if passed else 'NONE'}")
    res['seconds'] = round(time.time() - t0)
    write_json(OUTDIR / 'audit-C-selection-v2.json', res)
    check_protected()
    print(f"  wrote {OUTDIR / 'audit-C-selection-v2.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
