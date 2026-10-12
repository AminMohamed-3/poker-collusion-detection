"""Step 78 [C] (author: claude).  The stage-2 SELECTION formulation: listwise objective and
within-pair relative features.

Why (step 76).  On the 162-pair audit set the shipping hybrid scores MAP@5 .6159.  Stage 1's top-20
already holds 4.642 of the 5 listed hands; stage 2 puts 3.333 of them in the top 5.  A perfect choice
of 5 out of the SAME 20 would score .9481.  The decomposition is ORDER +.0673, SELECTION +.2649,
RETRIEVAL +.0519 (MAP@5), i.e. +.0135 / +.0530 / +.0104 composite.  The competition target lives
entirely in the selection step, so this step attacks the selection step and nothing else.

Two formulation changes.  The learner stays the incumbent LightGBM in both -- Codex owns the
"different learner" axis (TabPFN3) and this must not duplicate it.

  (a) WITHIN-PAIR RELATIVE FEATURES.  Stage 2 is pointwise: it scores each candidate on an absolute
      scale.  The task is comparative -- "which 5 of THESE 20".  For each structural column, add its
      rank and z-score computed within the pair's own candidate set.  The evidence that this is the
      right axis is already in the record: the single largest stage-2 gain in 65f is `s1_rank`
      (.644 DT / .719 SP / .227 CI), and `s1_rank` IS a within-pair rank.  We give the model one
      such column and it dominates; this asks what happens when every column gets one.

  (b) LISTWISE OBJECTIVE.  LGBMRanker lambdarank, the pair as the group, truncation at the candidate
      count, against the identical feature set.  "Choose 5 of 20" is a ranking problem and we are
      currently training it as independent binary classification.

Also measured, free from the same frame: the CANDIDATE-SET SWEEP.  TOP is fixed at 20 by inheritance.
A smaller candidate set lowers the ceiling but makes the choice easier; a larger one raises the
ceiling but adds distractors.  The oracle and the incumbent are reported at each K so the choice of 20
stops being an assumption.

PRE-REGISTERED GATE (written to disk by this script BEFORE any model is fitted).
  incumbent  : hybrid_clf_plain, the stage 2 that actually ships (audit MAP@5 .6159)
  promote to a held read only if, against that incumbent, an arm shows
      > +.010 MAP@5, paired 95 % pool-bootstrap lower bound > 0, and no family below -.005.
  STANDING CHECK (18.15 / 18.17, applied before the study runs): name the incumbent the gate protects
  and verify that an arm already known to be WORSE fails it.  `ref_clf_plain` is the struct design,
  measured at .5867 in step 74 and worse than the hybrid by -.0292.  -.0292 < +.010, so it FAILS the
  gate.  The gate discriminates against the known-worse arm; it is not measuring a design gap.

This step SELECTS on the audit set only.  It does not read folds 0-2 and it does not build a
candidate.  A held read, if any arm clears the gate, is a separate pre-registered step; my held-fold
read count stands at 4 and the next read makes 5.

Training-only: the 162 audit pairs in 172 tables outside every held fold.
Output: experiments/stage2_selection_C_20260912/audit-C-stage2-selection.json
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
from evidence_aug.step14_audit_truncation import CLF, SPLITS, TOP
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, build_d0, evaluate, fit_predict,
                                              frame_at)

AUTHOR = 'claude'
SEEDS = (11, 12, 13)
KSWEEP = (5, 10, 15, 20, 30, 40)
OUTDIR = CV / 'experiments' / 'stage2_selection_C_20260912'

RANKER = dict(objective='lambdarank', n_estimators=150, num_leaves=7, max_depth=3,
              min_child_samples=40, learning_rate=.05, label_gain=[0, 1],
              lambdarank_truncation_level=TOP, n_jobs=2, verbosity=-1)

GATE = {
    'incumbent': 'hybrid_clf_plain (the stage 2 that ships); audit MAP@5 0.6159',
    'promote_if': {'beats_incumbent_by': .010, 'ci95_lower_gt': 0.0, 'family_floor': -.005},
    'standing_check_18_15_18_17': {
        'known_worse_arm': 'ref_clf_plain', 'its_measured_value': 0.5867,
        'its_delta_vs_incumbent': -0.0292,
        'does_it_fail_the_gate': True,
        'why_this_check_exists': 'two of my own pre-registered gates fired on statistics that did not '
                                 'measure what the rule assumed; naming the incumbent and checking a '
                                 'known-worse arm fails would have caught both'},
    'on_promotion': 'a held read is a SEPARATE pre-registered step; this step does not read folds 0-2',
    'held_fold_reads_by_claude_so_far': 4}


def relativise(d, cols, topk):
    """within-pair rank and z-score of each column, computed over the pair's own candidate set.

    Non-candidates are given the pair's worst rank and z = 0 so the columns are defined everywhere;
    they never enter a fit or an evaluation because both are masked to oof_rank <= topk.
    """
    cand = pl.col('oof_rank') <= topk
    exprs = []
    for c in cols:
        v = pl.when(cand).then(pl.col(c)).otherwise(None)
        exprs.append(v.rank('average').over('pair_id').fill_null(0.0).alias(f'{c}__wr'))
        exprs.append(((v - v.mean().over('pair_id')) / (v.std().over('pair_id') + 1e-9))
                     .fill_null(0.0).alias(f'{c}__wz'))
    return d.with_columns(exprs)


def fit_predict_rank(dtr, dte, cols, topk, seeds=SEEDS, params=RANKER):
    """per-family, 5-way pool-hash OOF, lambdarank with the pair as the group."""
    pred = np.full(dte.height, -1e6)
    ftr, fte = dtr['behavior_family'].to_numpy(), dte['behavior_family'].to_numpy()
    str_, ste = dtr['split'].to_numpy(), dte['split'].to_numpy()
    ytr = dtr['relevance'].to_numpy().astype(int)
    ptr = dtr['pair_id'].to_numpy()
    ktr = (dtr['oof_rank'] <= topk).to_numpy()
    kte = (dte['oof_rank'] <= topk).to_numpy()
    Xtr_all = dtr.select(cols).to_numpy().astype(np.float64)
    Xte_all = dte.select(cols).to_numpy().astype(np.float64)
    for fam in FAMILIES:
        for s in range(SPLITS):
            tr = np.where((ftr == fam) & ktr & (str_ != s))[0]
            te = np.where((fte == fam) & kte & (ste == s))[0]
            if len(te) == 0 or len(tr) == 0 or ytr[tr].sum() == 0:
                continue
            tr = tr[np.argsort(ptr[tr], kind='stable')]          # lambdarank needs contiguous groups
            _, grp = np.unique(ptr[tr], return_counts=True)
            acc = np.zeros(len(te))
            for seed in seeds:
                r = lgb.LGBMRanker(**params, random_state=seed)
                r.fit(Xtr_all[tr], ytr[tr], group=grp)
                acc += r.predict(Xte_all[te]) / len(seeds)
            pred[te] = acc
    return pred


def main():
    t0 = time.time()
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTDIR / 'prediction.json',
               {'author': AUTHOR, 'step': 78, 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()), 'gate': GATE,
                'arms': 'base {ref, r382} x objective {clf, rank} x features {plain, rel}, '
                        'plus the r382/ref hybrid formed from each cell',
                'hypothesis': 'stage 2 loses .2649 MAP@5 to a perfect choice of 5 from its own top-20 '
                              '(step 76). A pointwise absolute-scale classifier is the wrong '
                              'formulation for "choose 5 of these 20"; a listwise objective and/or '
                              'within-pair relative encoding should recover part of it.',
                'learner_axis_owned_by_codex': 'TabPFN3 -- deliberately not varied here'})

    d0, struct, rcols = build_d0()
    d = frame_at(d0, 1.0)
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())

    print('=' * 100)
    print('STEP 78 [C].  STAGE-2 SELECTION: LISTWISE OBJECTIVE AND WITHIN-PAIR RELATIVE FEATURES')
    print('=' * 100)

    res = {'author': AUTHOR, 'step': 78, 'training_only': True, 'held_folds_touched': False,
           'gate': GATE, 'n_structural': len(struct), 'n_ranker': len(rcols),
           'candidate_sweep': {}, 'arms': {}, 'hybrid': {}, 'paired_vs_incumbent': {}, 'gate_result': {}}

    # ---- candidate-set sweep: is TOP = 20 the right candidate count at all? -------------------
    print('\n  CANDIDATE-SET SWEEP (oracle = best possible 5 out of the top-K)')
    print(f"    {'K':>4}{'listed_in_topK':>17}{'select_oracle':>15}")
    lst = d.filter(pl.col('relevance') == 1).select('pair_id', 'oof_rank')
    den = {p: min(5, len(truths[p])) for p in truths}
    for K in KSWEEP:
        inK = float((lst['oof_rank'] <= K).mean())
        h = dict(lst.filter(pl.col('oof_rank') <= K).group_by('pair_id').agg(pl.len()).iter_rows())
        orc = float(np.mean([min(h.get(p, 0), 5) / den[p] for p in truths]))
        res['candidate_sweep'][str(K)] = {'listed_in_topK': round(inK, 4), 'select_oracle': round(orc, 4)}
        print(f'    {K:>4}{inK:>17.4f}{orc:>15.4f}')

    # ---- arms -------------------------------------------------------------------------------
    base_cols = {'ref': struct + CHRONO3 + S1, 'r382': struct + CHRONO3 + S1 + rcols}
    drel = relativise(d, struct + CHRONO3, TOP)
    rel_extra = [c for c in drel.columns if c.endswith('__wr') or c.endswith('__wz')]
    res['n_relative_added'] = len(rel_extra)
    print(f'\n  relative encoding adds {len(rel_extra)} columns (rank + z of {len(struct + CHRONO3)} base columns)')

    per_pair_all = {}
    for base in ('ref', 'r382'):
        for feats in ('plain', 'rel'):
            cols = base_cols[base] + (rel_extra if feats == 'rel' else [])
            frame = drel if feats == 'rel' else d
            for obj in ('clf', 'rank'):
                tag = f'{base}_{obj}_{feats}'
                t1 = time.time()
                fn = fit_predict if obj == 'clf' else fit_predict_rank
                pred = fn(frame, frame, cols, TOP)
                summ, per_pair = evaluate(frame, pred, TOP, truths, labels)
                per_pair_all[tag] = per_pair
                res['arms'][tag] = {'map5': round(summ['map5'], 4), 'n_features': len(cols),
                                    'by_family': {f: round(v, 4) for f, v in summ['by_family'].items()}}
                log(tag, json.dumps({'map5': round(summ['map5'], 4), 'nf': len(cols),
                                     'sec': round(time.time() - t1)}))

    # ---- hybrids (r382 for transfer and soft play, ref for isolation) -------------------------
    for obj in ('clf', 'rank'):
        for feats in ('plain', 'rel'):
            a, b = per_pair_all[f'r382_{obj}_{feats}'], per_pair_all[f'ref_{obj}_{feats}']
            hyb = {q: (b[q] if labels[q] == CI else a[q]) for q in a}
            per_pair_all[f'hybrid_{obj}_{feats}'] = hyb
            res['hybrid'][f'hybrid_{obj}_{feats}'] = round(float(np.mean(list(hyb.values()))), 4)

    inc = per_pair_all['hybrid_clf_plain']
    res['incumbent_reproduced'] = round(float(np.mean(list(inc.values()))), 4)

    print(f"\n  {'arm':<24}{'MAP@5':>9}{'vs incumbent':>14}{'  95% CI':>20}   "
          + '  '.join(f'{f[:9]:>9}' for f in FAMILIES))
    print('  ' + '-' * 104)
    for tag in sorted(per_pair_all, key=lambda t: -float(np.mean(list(per_pair_all[t].values())))):
        pp = per_pair_all[tag]
        m = float(np.mean(list(pp.values())))
        dl = {p: pp[p] - inc[p] for p in pp}
        ci = [round(x, 4) for x in pool_bootstrap(dl, ptab)]
        fam = {f: float(np.mean([pp[p] for p in pp if labels[p] == f])) for f in FAMILIES}
        famd = {f: round(fam[f] - float(np.mean([inc[p] for p in inc if labels[p] == f])), 4) for f in FAMILIES}
        res['paired_vs_incumbent'][tag] = {'delta': round(m - res['incumbent_reproduced'], 4),
                                           'ci95': ci, 'by_family_delta': famd}
        res['gate_result'][tag] = bool(m - res['incumbent_reproduced'] > GATE['promote_if']['beats_incumbent_by']
                                       and ci[0] > 0 and min(famd.values()) > GATE['promote_if']['family_floor'])
        star = '  <== CLEARS GATE' if res['gate_result'][tag] else ''
        print(f"  {tag:<24}{m:>9.4f}{m - res['incumbent_reproduced']:>+14.4f}   [{ci[0]:+.4f}, {ci[1]:+.4f}]   "
              + '  '.join(f'{fam[f]:>9.4f}' for f in FAMILIES) + star)

    passed = [t for t, v in res['gate_result'].items() if v]
    res['arms_clearing_gate'] = passed
    print(f"\n  STANDING CHECK: ref_clf_plain delta "
          f"{res['paired_vs_incumbent']['ref_clf_plain']['delta']:+.4f} -> "
          f"gate {'FAILS (correct)' if not res['gate_result']['ref_clf_plain'] else 'PASSES (GATE IS BROKEN)'}")
    print(f"  arms clearing the gate: {passed if passed else 'NONE'}")

    res['seconds'] = round(time.time() - t0)
    write_json(OUTDIR / 'audit-C-stage2-selection.json', res)
    check_protected()
    print(f"\n  wrote {OUTDIR / 'audit-C-stage2-selection.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
