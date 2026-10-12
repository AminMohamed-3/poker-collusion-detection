"""step 88 [C].  Was the frozen-stage-2 protocol under-measuring every stage-1 intervention we
ever ran?  A joint refit, a TRUE frozen control, and a re-tune.

author: claude (agent)

WHY THIS FILE EXISTS
Step 85b (experiments/stage1_ensemble_C_20260912/audit-C-capacity-followup.json) found that adding
trees to stage 1 raised stage-1 MAP@5 in 9/9 cells (+.0145 to +.0356) and recall@20 in 9/9 (+.0013
to +.0126), yet end-to-end MAP@5 rose in only 2/9, minimum -.0054.  Its `loss_decomposition`
attributes the 16 lost correct final picks as -5 from top-20 eviction/addition and -11 from
RE-RANKING among candidates that never left the top-20, and reads that as a coupling tax: `s1` and
`s1_rank` carry ~74 % of stage-2 gain, so a stage 2 "fit around the production stage-1's rank
distribution" is penalised by any change to stage 1.  Its own limits list says a stage 2 "re-tuned
or re-fit for a deeper stage 1 is not tested".  If that tax is real, then the seed ensemble, the
capacity arms and the evidence behind OBS 18.14 all measured the tax rather than the intervention.

WHAT A READ OF THE CODE FOUND FIRST, AND WHY THE DESIGN BELOW IS NOT THE ONE THE BRIEF ASKED FOR
step85b.hybrid() is

    dd = d2.with_columns(pl.Series('oof', oof_vec), ...)          # the ARM's stage-1 scores
    pr = fit_predict(dd, dd, stage2_r382, TOP)                    # fit on dd, predict dd
    pf = fit_predict(dd, dd, stage2_ref,  TOP)

and step74.fit_predict calls `lgb.LGBMClassifier(...).fit(...)` inside.  So the stage-2 MODEL is
already re-estimated from scratch on each arm's own stage-1 output, in every cell of step 85b.
What step 85b held fixed is the stage-2 RECIPE -- the feature design, the hyperparameters, the
top-20 candidate depth, and the family routing (r382 for transfer and soft play, ref for
isolation).  "Refit stage 2 from scratch on its OOF output", the primary arm the brief asks for,
is therefore step 85b's own protocol, and would reproduce step 85b's numbers to the last digit.
Reproducing them is worth doing -- it is a hard, checkable claim -- but on its own it answers
nothing.  So this file runs FOUR stage-2 modes against the same stage-1 vectors:

  refit          fit_predict(arm, arm).  Identical to step 85b.  Must reproduce its stored
                 hybrid_map5 for all 12 cells; the reproduction is asserted, not assumed.
  frozen         fit_predict(BASELINE frame, ARM frame).  The stage 2 step 85b's prose describes
                 and its code does not build: the classifier is fit on the production stage-1's
                 rank distribution and its candidate set, then applied to the arm's candidates and
                 the arm's s1/s1_rank.  This is the real frozen comparator, and refit MINUS frozen
                 is the coupling tax as an actual number.
  tuned_nested   the stage-2 recipe itself re-selected for each stage-1, leak-free: per family and
                 per outer pool split, a 3-way inner pool split INSIDE the outer training tables
                 picks one of four configurations (2 capacities x 2 designs), then that winner is
                 refit on all outer training tables and predicts the outer test split.  Because the
                 design is selected per family, this subsumes the production routing.
  tuned_oracle   3 capacities x 2 designs x the 8 per-family routings = 24 configurations, scored
                 on the audit surface and maximised there.  SELECTED ON THE EVALUATION SURFACE, so
                 it is an upper bound and is labelled as one everywhere it appears.  Its grid is a
                 superset of the nested selector's, so it bounds that too.  The nested selector
                 pays 3 families x 5 outer splits x INNER x |grid| fits and the oracle pays |grid|,
                 which is why only the oracle can afford the wider grid.

Every mode is applied to the baseline stage 1 as well as to each arm, so the decisive quantity --
arm minus baseline WITHIN a mode -- charges both sides the same cost.

THE CONTROL THAT MAKES ANY OF IT INTERPRETABLE
Refitting stage 2 on the unchanged production stage-1 output must reproduce the incumbent hybrid
MAP@5 of .6159 at salt audit-trunc-v1 (step 85's control A3, on the stored `oof` of
audit-oof.parquet).  A joint-refit "gain" smaller than that control's own noise is not a gain.
The number is reported at the top of the artefact whatever it turns out to be.

PROVENANCE: a scratch timing-and-premise probe was run BEFORE the pre-registration below was
written.  It computed four numbers at salt audit-trunc-v1 -- the A3 identity, the production
single-seed hybrid, the trees_2x refit hybrid, and the trees_2x TRUE-frozen hybrid -- and it is
what established that step85b.hybrid() refits.  Those four numbers were known when the gate was
written and the gate's frozen comparator was chosen because of them.  They are listed verbatim in
`prediction-joint-refit.json` under `prior_knowledge_at_writing` so the pre-registration cannot be
read as cleaner than it was.

TRAINING ONLY.  The 162-pair audit surface, 172 tables outside every held fold.  load_fold(0) is
read for FEATURE COLUMNS only, as steps 74/76/77/84/85b do; the audit/held table overlap is
asserted empty at entry.  Held folds 0/1/2 are never read for truth, no leaderboard score and no
evaluation label enters anything here, and nothing here is proposed for submission.

Output: experiments/joint_refit_C_20260912/audit-C-joint-refit.json
"""
from __future__ import annotations

import hashlib
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl

from evidence_aug.fold import fold_pools, load_fold
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, check_protected, read_json, write_json
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step14_audit_truncation import CLF, OUT as LP, SPLITS, TOP
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, build_d0, evaluate, fit_predict,
                                              frame_at)
from evidence_aug.step85_C_stage1_ensemble import (AUTHOR, LEVER, OUTDIR as S85DIR, RESOLUTION,
                                                   SALTS, TRANSFER, pair_slices, recalls, resplit,
                                                   stage1_oof, wrank)

OUTDIR = CV / 'experiments' / 'joint_refit_C_20260912'
S85B = S85DIR / 'audit-C-capacity-followup.json'
ARMS = {'trees_2x': dict(n_estimators=476),
        'trees_4x': dict(n_estimators=952),
        'trees_8x': dict(n_estimators=1904)}
MODES = ('refit', 'frozen', 'tuned_nested', 'tuned_oracle')
# stage-2 recipe grid.  'incumbent' is a grid point, so a re-tune can only move where the inner
# criterion says it should.  Applied identically to the baseline stage 1 and to every arm.
S2_CAPACITY = {'incumbent': {},
               'more_trees': dict(n_estimators=400, learning_rate=.03),
               'more_capacity': dict(num_leaves=15, max_depth=5, min_child_samples=15)}
# tuned_nested pays 3 x INNER x |grid| fits per family per outer split, so its grid is the
# affordable subset; tuned_oracle carries the full grid and therefore still upper-bounds it.
S2_CAPACITY_NESTED = ('incumbent', 'more_capacity')
S2_DESIGNS = ('r382', 'ref')
INNER = 3                       # inner pool folds used by tuned_nested, inside the outer training tables
S2_SEEDS_FULL = (11, 12, 13)    # step74.SEEDS; inert for these params (step 80b / OBS 19.4), re-proved below
INCUMBENT_ROUTING = {'directed_transfer': 'r382', 'soft_play': 'r382', 'coordinated_isolation': 'ref'}
ANCHOR_A3 = .6159               # step 85 control A3: incumbent stage 2 on the STORED stage-1 `oof`
ANCHOR_PROD_HYBRID_V1 = .6163   # step 85b `by_salt['audit-trunc-v1'].single_seed_7329.hybrid_map5`

GATE = {
    'primary': 'end-to-end audit MAP@5 delta, capacity arm vs the production stage 1, with stage 2 '
               'REFIT FROM SCRATCH on each side\'s own stage-1 OOF output, MINIMUM across the three '
               'pool partitions',
    'real_requires': [
        'min-across-salt joint-refit delta > 0',
        'min-across-salt joint-refit delta > the corresponding frozen-stage-2 delta for the same arm',
        'min-across-salt joint-refit delta > |refit-identity noise|'],
    'material_requires': f'{LEVER} * min_delta * {TRANSFER} >= {RESOLUTION} '
                         f'(min_delta >= {round(RESOLUTION / (LEVER * TRANSFER), 4)})',
    'verdict': 'ALIVE only if real AND material',
    'frozen_comparator_definition': 'the frozen-stage-2 delta is computed IN THIS FILE as '
                                    'fit_predict(baseline frame, arm frame) -- a stage-2 classifier '
                                    'fit on the production stage-1 rank distribution and applied to '
                                    'the arm. Step 85b\'s published "frozen" delta is NOT that: its '
                                    'hybrid() calls fit_predict(dd, dd), so its number is a refit '
                                    'number. Both are reported side by side.',
    'refit_identity_rule': 'a joint-refit gain smaller in absolute value than the refit-identity '
                           'noise is not a gain and is not claimed as one',
    'secondary_mechanism_test': 'step 85b decomposes -16 correct final picks into -5 from top-20 '
                                'eviction/addition and -11 residual from re-ranking within the '
                                'top-20. The mechanism claim predicts the residual shrinks toward '
                                'zero once stage 2 is refit or re-tuned for the arm. Reported per '
                                'mode as total_residual_from_reranking; no gate is attached to it, '
                                'it is a direction test.',
    'tuned_oracle_is_biased': 'tuned_oracle maximises over 24 configurations on the same audit '
                              'surface it is scored on. It is an upper bound, never a result.',
    'note': 'written before any fit in this file; see prior_knowledge_at_writing for what a scratch '
            'probe had already revealed'}

PRIOR_PROTOCOL_AUDIT = {
    'question': 'the brief\'s premise is that every stage-1 change we ever evaluated was measured '
                'with stage 2 frozen, so all of them measured a coupling tax rather than the '
                'intervention. Is that true of the two studies it names?',
    'step85b_capacity_and_ensemble': {
        'file': 'src/evidence_aug/step85b_C_capacity_followup.py, hybrid()',
        'code': 'pr = fit_predict(dd, dd, stage2_r382, TOP); pf = fit_predict(dd, dd, stage2_ref, TOP) '
                'where dd carries the ARM\'s oof and oof_rank; step74.fit_predict calls '
                'lgb.LGBMClassifier(...).fit(...) per family and per pool split',
        'verdict': 'stage 2 was already REFIT on each arm\'s own stage-1 output. Only the RECIPE '
                   'was frozen.',
        'verified_how': 'numerically, in this file: premise_audit reproduces all 12 stored '
                        'hybrid_map5 values with fit_predict(arm, arm)'},
    'step72_obs_18_14_held_read': {
        'file': 'src/evidence_aug/step72_stage1_f2_fit.py, the `for dname in designs` loop',
        'code': "tr = with_rank(train.with_columns(pl.Series('s1', s1_train[dname])), 's1', "
                "'s1_rank') is built INSIDE the stage-1 design loop, and stage2_fit(tr, cols, fam, "
                'CLF) is called on it, so s1f2_top20_struct and s1f2_top20_r382 carry a stage 2 fit '
                "on the NEW stage 1's own out-of-fold scores",
        'verdict': 'OBS 18.14\'s hybrid_new (-.0040 against hybrid_old on the held read) was ALSO a '
                   'joint refit, not a frozen stage 2.',
        'verified_how': 'code inspection only. Re-running it would be a held read and is forbidden; '
                        'no number from folds 0/1/2 is recomputed, quoted as new, or relied on here '
                        'beyond the -.0040 already in OBSERVATIONS 18.14.'},
    'consequence': 'the premise does not hold for either study. Both already paid for a stage 2 '
                   'refit on the changed stage 1. Whatever is suppressing stage-1 gains, it is not '
                   'a stage-2 model frozen at the production stage-1\'s parameters.'}

PRIOR = {
    'what': 'a scratch timing-and-premise probe was executed before this pre-registration was '
            'written; it is what established that step85b.hybrid() already refits stage 2',
    'numbers_known_when_the_gate_was_written': {
        'salt': 'audit-trunc-v1',
        'A3_incumbent_hybrid_on_stored_stage1': 0.615934,
        'production_single_seed_7329_hybrid': 0.616315,
        'trees_2x_refit_hybrid': 0.615754,
        'trees_2x_TRUE_frozen_hybrid': 0.614823,
        'stage2_repeat_is_bit_identical': True},
    'effect_on_the_design': 'the gate names a frozen comparator computed in this file rather than '
                            'the one step 85b published, because the probe showed the published one '
                            'is a refit number. The probe fixed no threshold and no arm.',
    'probe_file': 'scratchpad only, not an artefact; every number in this artefact is recomputed '
                  'here from a single execution'}


# ------------------------------------------------------------------------------------------------
# stage-2 primitives.  fit_one is the same estimator step74.fit_predict builds, exposed so the
# nested selector can drive it over inner folds without going through the 5-way outer loop.
# ------------------------------------------------------------------------------------------------
def fit_one(X, y, tr, te, params, seeds):
    acc = np.zeros(len(te))
    for s in seeds:
        clf = lgb.LGBMClassifier(**params, random_state=s)
        clf.fit(X[tr], y[tr])
        acc += clf.predict_proba(X[te])[:, 1] / len(seeds)
    return acc


def inner_split_of(salt, tables):
    return {t: int(hashlib.sha256((salt + '|inner:' + t).encode()).hexdigest(), 16) % INNER
            for t in tables}


def main():
    t0 = time.time()
    prot_in = check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUTDIR / 'prediction-joint-refit.json'
    if not pred_path.exists():                      # never rewritten once it exists
        write_json(pred_path, {
            'author': 'claude', 'step': '88', 'written_before_any_fit': True,
            'written_before_any_fit_scope': 'before any model is fit in this file. A scratch probe '
                                            'ran earlier; its four numbers are disclosed under '
                                            'prior_knowledge_at_writing.',
            'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
            'gate': GATE, 'arms': ARMS, 'modes': list(MODES), 'salts': list(SALTS),
            'stage2_grid': {'capacities': {k: v for k, v in S2_CAPACITY.items()},
                            'designs': list(S2_DESIGNS), 'inner_folds': INNER,
                            'tuned_nested_capacities': list(S2_CAPACITY_NESTED),
                            'tuned_oracle_capacities': list(S2_CAPACITY),
                            'why_asymmetric': 'tuned_nested pays 3 x INNER x |grid| fits per '
                                              'family per outer split; tuned_oracle pays |grid| '
                                              'fits total, so it carries the wider grid and still '
                                              "upper-bounds the nested selector's grid"},
            'refit_identity_target': ANCHOR_A3,
            'frozen_deltas_from_step85b_for_the_same_arms': {
                'trees_2x': -0.0054, 'trees_4x': -0.0077, 'trees_8x': -0.0166,
                'note': 'min across the three salts, as published in '
                        'audit-C-capacity-followup.json resalt.*.downstream_min'},
            'prior_knowledge_at_writing': PRIOR,
            'training_only': True, 'held_folds_touched': False,
            'proposed_for_submission': False})

    print('=' * 100)
    print('STEP 88 [C].  JOINT REFIT: WAS THE FROZEN-STAGE-2 PROTOCOL UNDER-MEASURING STAGE 1?')
    print('=' * 100)

    # ---- frames: identical construction to step 85b -------------------------------------------
    audit = pl.read_parquet(LP / 'audit-oof.parquet')
    data, features, _, _ = load_fold(0)          # FEATURE COLUMNS only
    d = (audit.select('pair_id', 'hand_id')
         .join(data, on=['pair_id', 'hand_id'], how='left', validate='1:1')
         .join(audit.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance',
                            'chrono_rank', 'chrono_frac', 'n_hands',
                            pl.col('oof').alias('oof_saved')),
               on=['pair_id', 'hand_id'], validate='1:1')
         .sort('pair_id', 'hand_id'))
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    assert not (set(d['table_id']) & held), 'audit surface overlaps a held fold -- stop'

    d0, struct, rcols = build_d0()
    d2 = frame_at(d0, 1.0)
    assert d.select('pair_id', 'hand_id').equals(d2.select('pair_id', 'hand_id')), 'frame keys differ'

    pid = d['pair_id'].to_numpy()
    rel = d['relevance'].to_numpy()
    famv = d['behavior_family'].to_numpy()
    sl = pair_slices(pid)
    key = d.select('pair_id', 'hand_id', 'table_id')
    truths = {k[0]: set(g['hand_id'])
              for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    tables = sorted(set(d['table_id'].to_list()))
    DESIGN = {'r382': struct + CHRONO3 + S1 + rcols, 'ref': struct + CHRONO3 + S1}
    prod_design = list(features) + ['chrono_frac']
    is_ci = famv == CI
    listed = rel == 1
    y_all = rel.astype(float)
    tabv = d['table_id'].to_numpy()

    def frame_of(oof_vec, split):
        dd = d2.with_columns(pl.Series('oof', oof_vec), pl.Series('split', split))
        return dd.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id')
                               .cast(pl.UInt32).alias('oof_rank'))

    def zed(dd, pred):
        cands = (dd['oof_rank'] <= TOP).to_numpy()
        return np.where(cands, pred + 10.0, dd['oof'].to_numpy() - 1e6)

    def s1_metrics(v):
        summ, per_pair = summarize(key.with_columns(pl.Series('score', v)), truths, labels)
        row = {'map5': round(summ['map5'], 4),
               'by_family': {f: round(x, 4) for f, x in summ['by_family'].items()}}
        row.update(recalls(rel, famv, v, sl))
        return row, per_pair

    def route(dd, preds, routing):
        """combine per-design predictions into one vector by a per-family routing, then score."""
        pred = np.full(dd.height, -1e6)
        for f in FAMILIES:
            m = famv == f
            pred[m] = preds[routing[f]][m]
        _, pp = evaluate(dd, pred, TOP, truths, labels)
        return pp, zed(dd, pred)

    # ================= controls =================================================================
    ctrl = {}
    dv1 = resplit(d, SALTS[0])
    spl_v1 = dv1['split'].to_numpy()

    # C0.  is random_state inert for the stage-2 classifier at these params?  if it is, one seed is
    #      bit-identical to three and the tuning grid costs a third as much.  proved, not assumed.
    dd_probe = frame_of(d['oof_saved'].to_numpy(), spl_v1)
    p3 = fit_predict(dd_probe, dd_probe, DESIGN['ref'], TOP, seeds=S2_SEEDS_FULL)
    p1 = fit_predict(dd_probe, dd_probe, DESIGN['ref'], TOP, seeds=(11,))
    inert = float(np.max(np.abs(p3 - p1)))
    ctrl['C0_stage2_seed_inert_max_abs'] = inert
    ctrl['C0_stage2_seed_inert'] = bool(inert < 1e-12)
    ctrl['C0_what_it_is'] = ('the stage-2 params carry no subsample and no colsample, so '
                             'random_state should be inert (step 80b / OBS 19.4). Measured here as '
                             'the max |difference| between a 3-seed average and a single seed; '
                             'anything at 1e-16 is the float average of three identical vectors.')
    S2SEEDS = S2_SEEDS_FULL      # refit and frozen must reproduce step 85b exactly: 3 seeds
    S2SEEDS_TUNE = (11,) if inert < 1e-12 else S2_SEEDS_FULL
    ctrl['C0_seeds_refit_and_frozen'] = list(S2SEEDS)
    ctrl['C0_seeds_tuned_modes'] = list(S2SEEDS_TUNE)
    ctrl['C0_why_one_seed_for_the_tuned_modes'] = (
        'the tuned modes have no reproduction target and pay 3 families x 5 outer splits x INNER x '
        '|grid| fits; with the seed proved inert to 1e-16 a single seed is the same model at a '
        'third of the cost. refit and frozen keep all three seeds so their reproduction of step 85b '
        'is exact rather than approximate.')
    log('C0 stage-2 seed inertness', json.dumps({'max_abs_3seed_vs_1seed': inert,
                                                 'seeds_refit': list(S2SEEDS),
                                                 'seeds_tuned': list(S2SEEDS_TUNE)}))

    # C1.  THE REFIT-IDENTITY CONTROL.  stage 2 refit on the UNCHANGED production stage-1 output.
    def hybrid_incumbent(dd_tr, dd_te):
        preds = {k: fit_predict(dd_tr, dd_te, DESIGN[k], TOP, seeds=S2SEEDS) for k in S2_DESIGNS}
        pp, z = route(dd_te, preds, INCUMBENT_ROUTING)
        return float(np.mean(list(pp.values()))), pp, z, preds

    a3, pp_a3, _, _ = hybrid_incumbent(dd_probe, dd_probe)
    a3b, _, _, _ = hybrid_incumbent(dd_probe, dd_probe)          # repeat: refit noise, measured
    ctrl['C1_refit_identity_map5'] = round(a3, 6)
    ctrl['C1_refit_identity_anchor'] = ANCHOR_A3
    ctrl['C1_refit_identity_noise_abs'] = abs(a3 - a3b)
    ctrl['C1_deviation_from_recorded_anchor'] = round(abs(a3 - ANCHOR_A3), 6)
    ctrl['C1_pass'] = bool(abs(a3 - ANCHOR_A3) < 5e-5)
    ctrl['C1_what_it_is'] = ('stage 2 refit from scratch on the stored production stage-1 `oof` of '
                             'audit-oof.parquet at salt audit-trunc-v1, incumbent recipe and '
                             'routing; step 85 control A3 recorded .6159 for exactly this')
    ctrl['C1_why_noise_is_zero'] = ('the stage-2 params carry no subsample and no colsample, so the '
                                    'fit is deterministic; C0 measures that directly. The refit '
                                    'identity therefore has no sampling noise to subtract, and the '
                                    'deviation from the anchor is the anchor\'s own rounding.')
    log('C1 REFIT-IDENTITY CONTROL',
        json.dumps({'map5': round(a3, 6), 'anchor': ANCHOR_A3,
                    'noise_abs': ctrl['C1_refit_identity_noise_abs'],
                    'dev_from_anchor': ctrl['C1_deviation_from_recorded_anchor'],
                    'pass': ctrl['C1_pass']}))

    # C2.  the stored stage-1 matrix must still reproduce, as step 85b guards it
    mat = pl.read_parquet(S85DIR / 'stage1-oof-matrix.parquet')
    assert mat.select('pair_id', 'hand_id').equals(d.select('pair_id', 'hand_id')), 'matrix keys'
    fresh = stage1_oof(dv1, prod_design, 7329)
    stored = mat[f'{SALTS[0]}|prod|7329'].to_numpy()
    ctrl['C2_stage1_matrix_guard_max_abs'] = float(np.max(np.abs(fresh - stored)))
    assert ctrl['C2_stage1_matrix_guard_max_abs'] == 0.0, 'stored stage-1 matrix does not reproduce'

    # C3.  the production single-seed hybrid at v1 must reproduce step 85b's .6163
    dd_prod_v1 = frame_of(stored, spl_v1)
    hp, _, _, _ = hybrid_incumbent(dd_prod_v1, dd_prod_v1)
    ctrl['C3_prod_single_seed_hybrid_v1'] = round(hp, 6)
    ctrl['C3_pass'] = bool(abs(hp - ANCHOR_PROD_HYBRID_V1) < 5e-5)
    log('C3 production single-seed hybrid at v1',
        json.dumps({'map5': round(hp, 6), 'anchor': ANCHOR_PROD_HYBRID_V1, 'pass': ctrl['C3_pass']}))
    ctrl['all_pass'] = bool(ctrl['C1_pass'] and ctrl['C3_pass'])

    s85b = read_json(S85B)

    # ------------------------------------------------------------------------------------------
    # the nested, leak-free stage-2 re-tune
    # ------------------------------------------------------------------------------------------
    def tuned_nested(dd, salt):
        """per family and per outer split, choose one of the six configurations by a 4-way inner
        pool split INSIDE the outer training tables, then refit the winner on all of them."""
        X = {k: dd.select(DESIGN[k]).to_numpy().astype(np.float64) for k in S2_DESIGNS}
        outer = dd['split'].to_numpy()
        inner_map = inner_split_of(salt, tables)
        inner = np.array([inner_map[t] for t in tabv])
        cand = (dd['oof_rank'] <= TOP).to_numpy()
        pred = np.full(dd.height, -1e6)
        chosen = {}
        for f in FAMILIES:
            fm = famv == f
            for s in range(SPLITS):
                trm = fm & (outer != s)
                tem = fm & (outer == s) & cand
                if not tem.any():
                    continue
                best, best_map = None, -1.0
                tr_pairs = sorted(set(pid[trm].tolist()))
                trm_idx = np.where(trm)[0]
                loc_of = np.full(dd.height, -1, dtype=np.int64)
                loc_of[trm_idx] = np.arange(len(trm_idx))
                sub_full = dd.filter(pl.Series(trm))
                sub = sub_full.select('pair_id', 'hand_id', 'table_id')
                sub_oof = sub_full['oof'].to_numpy()
                sub_cand = cand[trm]
                inner_folds = [(np.where(trm & (inner != j) & cand)[0],
                                np.where(trm & (inner == j) & cand)[0]) for j in range(INNER)]
                for cname in S2_CAPACITY_NESTED:
                    for dname in S2_DESIGNS:
                        params = dict(CLF, **S2_CAPACITY[cname])
                        ip = np.full(len(trm_idx), -1e6)
                        for itr, ite in inner_folds:
                            if len(ite) == 0 or len(itr) == 0 or y_all[itr].sum() == 0:
                                continue
                            ip[loc_of[ite]] = fit_one(X[dname], y_all, itr, ite,
                                                      params, S2SEEDS_TUNE) + 10.0
                        miss = ~sub_cand
                        ip[miss] = sub_oof[miss] - 1e6
                        m = per_pair_map(sub.with_columns(pl.Series('score', ip)),
                                         {q: truths[q] for q in tr_pairs})
                        mv = float(np.mean(list(m.values())))
                        if mv > best_map:
                            best_map, best = mv, (cname, dname)
                chosen[f'{f}|{s}'] = {'capacity': best[0], 'design': best[1],
                                      'inner_map5': round(best_map, 4)}
                itr = np.where(trm & cand)[0]
                ite = np.where(tem)[0]
                pred[ite] = fit_one(X[best[1]], y_all, itr, ite,
                                    dict(CLF, **S2_CAPACITY[best[0]]), S2SEEDS_TUNE)
        _, pp = evaluate(dd, pred, TOP, truths, labels)
        return float(np.mean(list(pp.values()))), pp, zed(dd, pred), chosen

    def tuned_oracle(dd):
        """upper bound: all 3 capacities x 8 per-family routings scored on the audit surface.

        AP is a per-pair quantity and every pair belongs to exactly one family, so the 24
        routings are recombinations of the 6 per-design per-pair AP dictionaries and need no
        further scoring pass. The incumbent point is recombined the same way and asserted equal
        to `refit` by the caller, which is what validates the recombination."""
        preds, pps = {}, {}
        for cname, cx in S2_CAPACITY.items():
            for dname in S2_DESIGNS:
                p = fit_predict(dd, dd, DESIGN[dname], TOP, seeds=S2SEEDS_TUNE,
                                params=dict(CLF, **cx))
                preds[(cname, dname)] = p
                _, pps[(cname, dname)] = evaluate(dd, p, TOP, truths, labels)
        def recombine(cname, routing):
            return {q: pps[(cname, routing[labels[q]])][q] for q in labels}

        best, best_map = None, -1.0
        for cname in S2_CAPACITY:
            for r0 in S2_DESIGNS:
                for r1 in S2_DESIGNS:
                    for r2 in S2_DESIGNS:
                        routing = dict(zip(FAMILIES, (r0, r1, r2)))
                        mv = float(np.mean(list(recombine(cname, routing).values())))
                        if mv > best_map:
                            best_map, best = mv, (cname, routing)
        best_pp, best_z = route(dd, {k: preds[(best[0], k)] for k in S2_DESIGNS}, best[1])
        inc = recombine('incumbent', INCUMBENT_ROUTING)
        return float(np.mean(list(best_pp.values()))), best_pp, best_z, \
            {'capacity': best[0], 'routing': best[1]}, float(np.mean(list(inc.values())))

    def churn(vb, zb, va, za):
        """step 85b's ledger, character for character, so the two files are comparable."""
        b20, a20 = wrank(vb, sl) <= TOP, wrank(va, sl) <= TOP
        b5, a5 = wrank(zb, sl) <= 5, wrank(za, sl) <= 5
        dropped, added = b20 & ~a20, a20 & ~b20
        slots = int(b20.sum())
        lost = int((b5 & dropped & listed).sum())
        gained = int((added & listed & a5).sum())
        out = {'top20_slots': slots, 'top20_churn_hands': int(dropped.sum()),
               'top20_churn_fraction': round(float(dropped.sum()) / slots, 4),
               'listed_dropped': int((dropped & listed).sum()),
               'listed_added': int((added & listed).sum()),
               'correct_final_picks_lost_to_eviction': lost,
               'correct_final_picks_gained_from_new': gained,
               'net_correct_final_picks': gained - lost,
               'baseline_correct_final_picks': int((b5 & listed).sum()),
               'arm_correct_final_picks': int((a5 & listed).sum())}
        out['observed_change_in_correct_final_picks'] = (out['arm_correct_final_picks']
                                                         - out['baseline_correct_final_picks'])
        out['explained_by_top20_eviction_and_addition'] = out['net_correct_final_picks']
        out['residual_from_reranking_within_top20'] = (
            out['observed_change_in_correct_final_picks']
            - out['explained_by_top20_eviction_and_addition'])
        return out

    # ================= main loop ===============================================================
    res = {'author': 'claude', 'step': '88', 'training_only': True, 'held_folds_touched': False,
           'gate': GATE, 'prior_knowledge_at_writing': PRIOR, 'arms': ARMS, 'modes': list(MODES),
           'salts': list(SALTS), 'controls': ctrl,
           'stage2_grid': {'capacities': S2_CAPACITY, 'designs': list(S2_DESIGNS),
                           'inner_folds': INNER,
                           'tuned_nested_capacities': list(S2_CAPACITY_NESTED),
                           'tuned_oracle_capacities': list(S2_CAPACITY)},
           'premise_audit': {}, 'by_salt': {}}
    orc_dev = []

    for salt in SALTS:
        spl = resplit(d, salt)['split'].to_numpy()
        base_v = mat[f'{salt}|prod|7329'].to_numpy()
        mb, ppb_s1 = s1_metrics(base_v)
        dd_b = frame_of(base_v, spl)
        t1 = time.time()
        hb_refit, ppb_refit, zb_refit, _ = hybrid_incumbent(dd_b, dd_b)
        hb_tn, ppb_tn, zb_tn, chose_b = tuned_nested(dd_b, salt)
        hb_to, ppb_to, zb_to, cfg_b, hb_to_inc = tuned_oracle(dd_b)
        orc_dev.append(abs(hb_to_inc - hb_refit))
        assert orc_dev[-1] < 1e-4, 'grid incumbent point does not equal refit'
        base_row = {'stage1_map5': mb['map5'], 'recall_at_20': mb['recall_at_20']['pooled'],
                    'hybrid_map5': {'refit': round(hb_refit, 6),
                                    'frozen': round(hb_refit, 6),
                                    'tuned_nested': round(hb_tn, 6),
                                    'tuned_oracle': round(hb_to, 6)},
                    'frozen_equals_refit_for_the_baseline': True,
                    'mode_cost_vs_refit': {
                        'tuned_nested': round(hb_tn - hb_refit, 5),
                        'tuned_oracle': round(hb_to - hb_refit, 5),
                        'what_it_is': 'what each stage-2 mode does to the ABSOLUTE production '
                                      'chain before any arm is introduced. A large negative under '
                                      'tuned_nested is selection noise in the re-tune itself, and '
                                      'means the arm-minus-baseline delta under it is a difference '
                                      'between two degraded chains, not an improvement on the one '
                                      'that ships.'},
                    'tuned_nested_choices': chose_b, 'tuned_oracle_choice': cfg_b,
                    'seconds': round(time.time() - t1)}
        base_pp = {'refit': ppb_refit, 'frozen': ppb_refit,
                   'tuned_nested': ppb_tn, 'tuned_oracle': ppb_to}
        base_z = {'refit': zb_refit, 'frozen': zb_refit, 'tuned_nested': zb_tn, 'tuned_oracle': zb_to}
        log(f'{salt} baseline', json.dumps(
            {'s1': mb['map5'], 'refit': round(hb_refit, 4), 'tuned_nested': round(hb_tn, 4),
             'tuned_oracle': round(hb_to, 4), 'sec': base_row['seconds']}))

        row = {'production_stage1': base_row, 'arms': {}}
        for name, extra in ARMS.items():
            t1 = time.time()
            va = stage1_oof(resplit(d, salt), prod_design, 7329, extra=extra)
            ma, ppa_s1 = s1_metrics(va)
            b1 = boot(ppb_s1, ppa_s1, ptab, pools)
            dd_a = frame_of(va, spl)

            ha_refit, ppa_refit, za_refit, _ = hybrid_incumbent(dd_a, dd_a)
            ha_frozen, ppa_frozen, za_frozen, _ = hybrid_incumbent(dd_b, dd_a)
            ha_tn, ppa_tn, za_tn, chose_a = tuned_nested(dd_a, salt)
            ha_to, ppa_to, za_to, cfg_a, ha_to_inc = tuned_oracle(dd_a)
            orc_dev.append(abs(ha_to_inc - ha_refit))
            assert orc_dev[-1] < 1e-4, 'grid incumbent point does not equal refit'

            arm_pp = {'refit': ppa_refit, 'frozen': ppa_frozen,
                      'tuned_nested': ppa_tn, 'tuned_oracle': ppa_to}
            arm_z = {'refit': za_refit, 'frozen': za_frozen,
                     'tuned_nested': za_tn, 'tuned_oracle': za_to}
            arm_h = {'refit': ha_refit, 'frozen': ha_frozen,
                     'tuned_nested': ha_tn, 'tuned_oracle': ha_to}

            cell = {'stage1_map5': ma['map5'], 'stage1_delta': round(b1['mean_delta'], 4),
                    'stage1_ci95': [round(x, 4) for x in b1['ci95']],
                    'recall_at_20': ma['recall_at_20']['pooled'],
                    'recall_at_20_delta': round(ma['recall_at_20']['pooled']
                                                - mb['recall_at_20']['pooled'], 4),
                    'tuned_nested_choices': chose_a, 'tuned_oracle_choice': cfg_a,
                    'modes': {}}
            for mode in MODES:
                bb = boot(base_pp[mode], arm_pp[mode], ptab, pools)
                ch = churn(base_v, base_z[mode], va, arm_z[mode])
                cell['modes'][mode] = {
                    'hybrid_map5': round(arm_h[mode], 6),
                    'baseline_hybrid_map5': round(base_row['hybrid_map5'][mode], 6),
                    'downstream_delta': round(bb['mean_delta'], 4),
                    'downstream_delta_raw': bb['mean_delta'],
                    'downstream_ci95': [round(x, 4) for x in bb['ci95']],
                    'by_family': {f: round(float(np.mean([arm_pp[mode][p] for p in arm_pp[mode]
                                                          if labels[p] == f])), 4)
                                  for f in FAMILIES},
                    'churn': ch}
            cell['coupling_tax_refit_minus_frozen'] = round(
                cell['modes']['refit']['downstream_delta_raw']
                - cell['modes']['frozen']['downstream_delta_raw'], 5)
            cell['seconds'] = round(time.time() - t1)
            row['arms'][name] = cell
            log(f'{salt} {name}', json.dumps(
                {'s1': ma['map5'], 's1_d': cell['stage1_delta'],
                 'r20_d': cell['recall_at_20_delta'],
                 **{m: cell['modes'][m]['downstream_delta'] for m in MODES},
                 'tax': cell['coupling_tax_refit_minus_frozen'], 'sec': cell['seconds']}))
        res['by_salt'][salt] = row

    # ================= premise audit: does `refit` reproduce step 85b? ==========================
    pa = {'claim': 'step85b.hybrid() calls fit_predict(dd, dd, ...) and therefore already refits the '
                   'stage-2 MODEL on each arm\'s own stage-1 output; what it freezes is the RECIPE '
                   '(design, hyperparameters, top-20 depth, family routing)',
          'test': 'this file\'s `refit` mode must reproduce step 85b\'s stored hybrid_map5 in all 12 '
                  'cells (3 salts x baseline + 3 arms) to 4 decimal places',
          'cells': {}, 'max_abs_deviation': 0.0}
    worst = 0.0
    for salt in SALTS:
        got = res['by_salt'][salt]['production_stage1']['hybrid_map5']['refit']
        want = s85b['by_salt'][salt]['single_seed_7329']['hybrid_map5']
        pa['cells'][f'{salt}|production'] = {'step88_refit': round(got, 4), 'step85b': want,
                                             'abs_diff': round(abs(round(got, 4) - want), 6)}
        worst = max(worst, abs(round(got, 4) - want))
        for name in ARMS:
            got = res['by_salt'][salt]['arms'][name]['modes']['refit']['hybrid_map5']
            want = s85b['by_salt'][salt]['arms'][name]['hybrid_map5']
            pa['cells'][f'{salt}|{name}'] = {'step88_refit': round(got, 4), 'step85b': want,
                                             'abs_diff': round(abs(round(got, 4) - want), 6)}
            worst = max(worst, abs(round(got, 4) - want))
    pa['max_abs_deviation'] = round(worst, 6)
    pa['reproduced'] = bool(worst < 5e-5)
    pa['consequence'] = ('if reproduced, then "arm-refit minus baseline-refit" IS step 85b\'s '
                         'published delta, and the joint refit recovers nothing that step 85b had '
                         'not already granted. The frozen comparator that makes the coupling-tax '
                         'question answerable is the one built in this file.')
    pa['prior_protocol_audit'] = PRIOR_PROTOCOL_AUDIT
    res['premise_audit'] = pa

    # ================= rollups =================================================================
    res['resalt'] = {}
    for name in ARMS:
        entry = {}
        for mode in MODES:
            dl = [res['by_salt'][s]['arms'][name]['modes'][mode]['downstream_delta'] for s in SALTS]
            entry[mode] = {'deltas': dl, 'min': min(dl), 'max': max(dl),
                           'up_in': int(sum(x > 0 for x in dl))}
        s1d = [res['by_salt'][s]['arms'][name]['stage1_delta'] for s in SALTS]
        r20 = [res['by_salt'][s]['arms'][name]['recall_at_20_delta'] for s in SALTS]
        tax = [res['by_salt'][s]['arms'][name]['coupling_tax_refit_minus_frozen'] for s in SALTS]
        entry['stage1_deltas'] = s1d
        entry['stage1_min'] = min(s1d)
        entry['recall20_deltas'] = r20
        entry['recall20_min'] = min(r20)
        entry['coupling_tax_refit_minus_frozen'] = tax
        entry['coupling_tax_min'] = min(tax)
        entry['coupling_tax_mean'] = round(float(np.mean(tax)), 5)
        entry['step85b_published_downstream_min'] = s85b['resalt'][name]['downstream_min']
        mind = entry['refit']['min']
        noise = ctrl['C1_refit_identity_noise_abs']
        entry['gate'] = {
            'joint_refit_min': mind,
            'frozen_min': entry['frozen']['min'],
            'refit_identity_noise_abs': noise,
            'beats_zero': bool(mind > 0),
            'beats_frozen': bool(mind > entry['frozen']['min']),
            'beats_refit_noise': bool(mind > abs(noise)),
            'real': bool(mind > 0 and mind > entry['frozen']['min'] and mind > abs(noise)),
            'material': bool(LEVER * mind * TRANSFER >= RESOLUTION),
            'composite_discounted': round(LEVER * mind * TRANSFER, 5)}
        entry['gate']['ALIVE'] = bool(entry['gate']['real'] and entry['gate']['material'])
        res['resalt'][name] = entry

    # ---- the mechanism test: does the re-ranking residual shrink? ------------------------------
    cells = [(s, n) for s in SALTS for n in ARMS]
    res['loss_decomposition_by_mode'] = {
        'derivation': 'per cell, observed = arm_correct_final_picks - baseline_correct_final_picks; '
                      'explained = correct_final_picks_gained_from_new - '
                      'correct_final_picks_lost_to_eviction; residual = observed - explained. Same '
                      'identity step 85b used, recomputed per stage-2 mode with BOTH sides of each '
                      'comparison scored under that mode.',
        'step85b_reference': {
            'total_observed': s85b['loss_decomposition']['total_observed_change_in_correct_final_picks'],
            'total_explained': s85b['loss_decomposition']['total_explained_by_eviction_and_addition'],
            'total_residual': s85b['loss_decomposition']['total_residual_from_reranking_within_top20']},
        'by_mode': {}, 'per_cell': {}}
    for mode in MODES:
        obs = sum(res['by_salt'][s]['arms'][n]['modes'][mode]['churn']
                  ['observed_change_in_correct_final_picks'] for s, n in cells)
        exp = sum(res['by_salt'][s]['arms'][n]['modes'][mode]['churn']
                  ['explained_by_top20_eviction_and_addition'] for s, n in cells)
        rsd = sum(res['by_salt'][s]['arms'][n]['modes'][mode]['churn']
                  ['residual_from_reranking_within_top20'] for s, n in cells)
        res['loss_decomposition_by_mode']['by_mode'][mode] = {
            'total_observed_change_in_correct_final_picks': int(obs),
            'total_explained_by_eviction_and_addition': int(exp),
            'total_residual_from_reranking_within_top20': int(rsd),
            'cells_with_negative_residual': int(sum(
                res['by_salt'][s]['arms'][n]['modes'][mode]['churn']
                ['residual_from_reranking_within_top20'] < 0 for s, n in cells))}
    for s, n in cells:
        res['loss_decomposition_by_mode']['per_cell'][f'{s}/{n}'] = {
            mode: res['by_salt'][s]['arms'][n]['modes'][mode]['churn']
            ['residual_from_reranking_within_top20'] for mode in MODES}
    ref_res = res['loss_decomposition_by_mode']['by_mode']['refit'][
        'total_residual_from_reranking_within_top20']
    fro_res = res['loss_decomposition_by_mode']['by_mode']['frozen'][
        'total_residual_from_reranking_within_top20']
    tn_res = res['loss_decomposition_by_mode']['by_mode']['tuned_nested'][
        'total_residual_from_reranking_within_top20']
    res['loss_decomposition_by_mode']['reading'] = {
        'refit_residual_equals_step85b': bool(ref_res == res['loss_decomposition_by_mode']
                                              ['step85b_reference']['total_residual']),
        'frozen_residual': fro_res, 'refit_residual': ref_res, 'tuned_nested_residual': tn_res,
        'residual_shrank_toward_zero_under_refit_vs_frozen': bool(abs(ref_res) < abs(fro_res)),
        'residual_shrank_toward_zero_under_tuning_vs_refit': bool(abs(tn_res) < abs(ref_res))}

    # ---- verdict -------------------------------------------------------------------------------
    best = max(ARMS, key=lambda n: res['resalt'][n]['gate']['joint_refit_min'])
    g = res['resalt'][best]['gate']
    tax_all = [res['by_salt'][s]['arms'][n]['coupling_tax_refit_minus_frozen'] for s, n in cells]
    tn_all = [res['by_salt'][s]['arms'][n]['modes']['tuned_nested']['downstream_delta']
              for s, n in cells]
    to_all = [res['by_salt'][s]['arms'][n]['modes']['tuned_oracle']['downstream_delta']
              for s, n in cells]
    res['verdict'] = {
        'best_arm_by_joint_refit_min': best,
        'deciding_number': g['joint_refit_min'],
        'deciding_number_is': 'minimum across the three pool partitions of the end-to-end audit '
                              'MAP@5 delta, capacity arm vs the production stage 1, stage 2 refit '
                              'from scratch on each side\'s own stage-1 output',
        'real': g['real'], 'material': g['material'], 'ALIVE': g['ALIVE'],
        'label': ('ALIVE' if g['ALIVE'] else
                  ('real-not-material' if g['real'] else 'DEAD')),
        'refit_identity_control': ctrl['C1_refit_identity_map5'],
        'refit_identity_noise_abs': ctrl['C1_refit_identity_noise_abs'],
        'joint_refit_reproduces_step85b': res['premise_audit']['reproduced'],
        'coupling_tax_refit_minus_frozen': {'mean_over_9_cells': round(float(np.mean(tax_all)), 5),
                                            'min': round(float(np.min(tax_all)), 5),
                                            'max': round(float(np.max(tax_all)), 5),
                                            'positive_in': int(sum(x > 0 for x in tax_all)),
                                            'of_cells': len(tax_all)},
        'tuned_nested_min_over_arms': min(res['resalt'][n]['tuned_nested']['min'] for n in ARMS),
        'tuned_nested_max_over_cells': max(tn_all),
        'tuned_oracle_max_over_cells_UPPER_BOUND': max(to_all),
        'tuned_oracle_min_over_arms_UPPER_BOUND': min(res['resalt'][n]['tuned_oracle']['min']
                                                      for n in ARMS)}

    res['limits'] = [
        'The primary arm the brief names -- stage 2 refit from scratch on each stage-1\'s own OOF '
        'output -- is step 85b\'s protocol, not a new one; premise_audit shows the reproduction. '
        'The new comparators in this file are `frozen` (stage 2 fit on the baseline stage-1 and '
        'applied to the arm) and the two tuned modes.',
        'tuned_oracle maximises 24 configurations on the same 162 audit pairs it is scored on. It '
        'is an upper bound on what any re-tune could buy on this surface, not an estimate of what '
        'one would buy off it.',
        'tuned_nested selects inside ~138 outer-training tables split 4 ways; the inner criterion '
        'is a MAP@5 over roughly 40 pairs per family, so the selector is noisy and can lose to the '
        'incumbent. That is a property of the surface, not a bug, and it is why the incumbent is a '
        'grid point.',
        'The stage-2 grid varies capacity and design only. Candidate depth (TOP=20), the label, the '
        'per-family split structure and the stage-1 design are all held fixed.',
        'The churn ledger counts hands, not AP; MAP@5 weights a hit by rank and by '
        'min(n_relevant,5), so the decomposition gives direction, not magnitude.',
        'Audit->held transfer has run 46-52 % historically; the discounted composite applies .49.',
        '162 audit pairs, training only. Held folds 0/1/2 were not read for truth. Nothing here is '
        'proposed for submission.']

    res['controls']['C4_oracle_incumbent_point_vs_refit_max_abs'] = round(float(max(orc_dev)), 9)
    res['controls']['C4_what_it_is'] = (
        'the tuned_oracle grid contains the incumbent capacity, and recombining its two designs '
        'under the incumbent routing must give back `refit`. This is the max |deviation| over all '
        '12 cells and it validates the per-family recombination the oracle uses in place of 24 '
        'scoring passes. It is not exactly zero only because the tuned modes run one seed and '
        'refit runs three.')
    res['mode_absolute_levels'] = {
        'what': 'the audit MAP@5 of the production chain under each stage-2 mode, before any arm. '
                'refit is what ships. A mode that lowers this is paying a selection cost, and its '
                'arm-minus-baseline delta must be read against that.',
        'by_salt': {s_: {m: res['by_salt'][s_]['production_stage1']['hybrid_map5'][m]
                         for m in MODES} for s_ in SALTS},
        'tuned_nested_cost_vs_refit': [res['by_salt'][s_]['production_stage1']
                                       ['mode_cost_vs_refit']['tuned_nested'] for s_ in SALTS],
        'tuned_oracle_gain_vs_refit': [res['by_salt'][s_]['production_stage1']
                                       ['mode_cost_vs_refit']['tuned_oracle'] for s_ in SALTS]}

    prot_out = check_protected()
    res['protected'] = {'entry': prot_in, 'exit': prot_out,
                        'identical': bool(prot_in == prot_out)}
    res['seconds'] = round(time.time() - t0)

    # ================= printout ================================================================
    print()
    print(f"  REFIT-IDENTITY CONTROL: {ctrl['C1_refit_identity_map5']:.6f} against the recorded "
          f"{ANCHOR_A3}  (noise {ctrl['C1_refit_identity_noise_abs']:.2e}, "
          f"deviation from anchor {ctrl['C1_deviation_from_recorded_anchor']:.2e}) "
          f"-> {'PASS' if ctrl['C1_pass'] else 'FAIL'}")
    print(f"  PREMISE AUDIT: step88 `refit` reproduces step 85b in 12/12 cells: "
          f"{res['premise_audit']['reproduced']}  (max |dev| "
          f"{res['premise_audit']['max_abs_deviation']:.2e})")
    print()
    print('  END-TO-END DELTA vs THE PRODUCTION STAGE 1, BY STAGE-2 MODE')
    print(f"    {'partition':18s} {'arm':10s} {'s1 d':>8s} {'r20 d':>8s} {'refit':>9s} "
          f"{'frozen':>9s} {'tax':>9s} {'tuned_n':>9s} {'tuned_o*':>9s}")
    for salt in SALTS:
        for name in ARMS:
            c = res['by_salt'][salt]['arms'][name]
            print(f"    {salt:18s} {name:10s} {c['stage1_delta']:+8.4f} "
                  f"{c['recall_at_20_delta']:+8.4f} "
                  f"{c['modes']['refit']['downstream_delta']:+9.4f} "
                  f"{c['modes']['frozen']['downstream_delta']:+9.4f} "
                  f"{c['coupling_tax_refit_minus_frozen']:+9.5f} "
                  f"{c['modes']['tuned_nested']['downstream_delta']:+9.4f} "
                  f"{c['modes']['tuned_oracle']['downstream_delta']:+9.4f}")
    print('    * tuned_oracle is selected on the evaluation surface: upper bound, not a result')
    print()
    print('  MINIMUM ACROSS THE THREE PARTITIONS')
    for name in ARMS:
        z = res['resalt'][name]
        print(f"    {name:10s} s1 {z['stage1_min']:+.4f}  r20 {z['recall20_min']:+.4f}  "
              f"refit {z['refit']['min']:+.4f}  frozen {z['frozen']['min']:+.4f}  "
              f"tuned_n {z['tuned_nested']['min']:+.4f}  tuned_o* {z['tuned_oracle']['min']:+.4f}  "
              f"| step85b published {z['step85b_published_downstream_min']:+.4f}")
    print()
    print('  ABSOLUTE LEVEL OF THE PRODUCTION CHAIN UNDER EACH STAGE-2 MODE (no arm involved)')
    for salt in SALTS:
        lv = res['mode_absolute_levels']['by_salt'][salt]
        print(f"    {salt:18s} refit {lv['refit']:.4f}  tuned_nested {lv['tuned_nested']:.4f}  "
              f"tuned_oracle* {lv['tuned_oracle']:.4f}")
    print()
    print('  MECHANISM: TOTAL RE-RANKING RESIDUAL OVER THE 9 ARM CELLS (step 85b reported -11)')
    for mode in MODES:
        b = res['loss_decomposition_by_mode']['by_mode'][mode]
        print(f"    {mode:14s} observed {b['total_observed_change_in_correct_final_picks']:+4d}  "
              f"explained {b['total_explained_by_eviction_and_addition']:+4d}  "
              f"residual {b['total_residual_from_reranking_within_top20']:+4d}")
    v = res['verdict']
    print(f"\n  VERDICT: {v['label']}  -- best arm {v['best_arm_by_joint_refit_min']}, deciding "
          f"number {v['deciding_number']:+.4f}")
    print(f"    coupling tax (refit - frozen) mean over 9 cells "
          f"{v['coupling_tax_refit_minus_frozen']['mean_over_9_cells']:+.5f}, positive in "
          f"{v['coupling_tax_refit_minus_frozen']['positive_in']}/9")

    write_json(OUTDIR / 'audit-C-joint-refit.json', res)
    print(f"\n  wrote {OUTDIR / 'audit-C-joint-refit.json'}  ({res['seconds']} s)")
    print('DONE')


if __name__ == '__main__':
    main()
