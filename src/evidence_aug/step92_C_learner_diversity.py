"""Step 92 [C].  Do genuinely different LEARNER CLASSES on the frozen stage-2 surface decorrelate
from r382, and does BLENDING them beat the best single model?
author: claude

WHAT STEP 89 DID NOT DO
  Step 89 (`step89_C_complementarity.py`, `experiments/complementarity_C_20260912/`) returned DEAD on
  composition: mean off-diagonal per-pair AP correlation .680/.639/.656 over three salts, an oracle
  ceiling SMALLER than its matched null (p = 1.0/1.0/.9975), and seven architectures buying +0.0042
  of oracle over seven seeds of one model.  Two holes in it:

  (1) ALL SEVEN of its variants were LightGBM (GBDT classifiers, two LambdaRank rankers, and raw
      stage 1).  Zero non-tree learners.  The only confirmed PASS anyone on this project holds is
      TabPFN-3 (+0.0140873, CI [+0.0004109, +0.0282275]) -- a LEARNER swap on identical columns.
      Learner-CLASS diversity has never been tested on our own surface.
  (2) It tested ROUTING only (per-family LOTO, a learned per-pair router, pairwise top-2).  It never
      tested a BLEND.  Blending is far more robust than routing and is the obvious omission.

  This step fits genuinely different learner classes -- L2 logistic regression (additive, no
  interactions), a small MLP, ExtraTrees and RandomForest (bagging, not boosting), and k-NN
  (local, non-parametric) -- on the SAME rows, target, fold assignment, top-20 candidate restriction
  and 453-column feature matrix as `r382`, and then blends them.

THE CORRELATION IS DESCRIPTIVE, NOT A GATE  (scope correction received before the production run;
  see `provenance`).  Per-pair AP is a lossy summary that discards the hand-level ranking errors a
  blend actually operates on: two models can have per-pair AP correlation 1 and identical AP vectors
  ([.5, 1] each) and a blend of their HAND-LEVEL scores can still lift mean AP from .75 to 1.  The
  blend arm therefore runs in full whatever Phase 2 says, and a high correlation is never used to
  shorten the run or to pre-empt Phase 3.

CORRELATION IS REPORTED TWO WAYS, because two conventions are in circulation:
  WITHIN-SALT   `np.corrcoef` on the per-salt per-pair AP matrix, mean of the upper triangle.  This
                is step 89's definition; .680/.639/.656 are continuous with it.
  SALT-AVERAGED each model's per-pair AP vector averaged over the three salts FIRST, then correlated.
                This is the convention behind the other agent's TabPFN-3 vs LGBM-hybrid .8543.
                Averaging removes per-salt fit noise and MECHANICALLY INFLATES the correlation, so
                the two conventions are not comparable and both are reported for every model pair.

WHAT VARIES ACROSS THE THREE SALTS AND WHAT DOES NOT  (step 90's `cross_salt_consistency/why`)
  `resplit()` re-hashes the STAGE-2 fold assignment ONLY.  `oof` and `oof_rank` are stored columns.
  Therefore, IDENTICAL BY CONSTRUCTION across all three salts: the 162 pairs, their families, their
  tables, the truth, the top-20 candidate set of every pair, and the per-pair AP vector of `s1_pure`
  (no stage 2 at all).  GENUINELY VARYING: every quantity that depends on a stage-2 fit -- every
  learner's MAP@5 and per-pair AP, every correlation between two fitted models, every blend, every
  oracle ceiling, every seed-family number.  But the three salts are three RE-PARTITIONS OF THE SAME
  162 PAIRS, not three independent samples: the min over salts is a robustness check against
  fold-assignment luck, and nothing in this file is three independent replications.

PROTOCOL
  Training only: the 162 audit pairs in 104 tables outside every held fold.  Folds 0/1/2 are NOT read
  for truth; `load_fold(0)` supplies FEATURE COLUMNS only, exactly as steps 74/76/77/84/85b/87/89.
  Overlap with the held tables is asserted empty at entry.
  `check_protected()` at entry and exit; both hash dicts recorded.
  Three salts (audit-trunc-v1, audit-resplit-b, audit-resplit-c); the MINIMUM across partitions is
  the reported result.
  `prediction-learner-diversity.json` is written BEFORE any fit and never edited.
  Materiality: 0.2 * min_delta * 0.49 >= 0.007, i.e. min_delta >= 0.0714 on audit MAP@5.
  A result below the bar is a FAILURE and is recorded as one.  No rescue, no post-hoc subgroup, no
  re-thresholding.
  Data loading, fold logic, MAP@5 scoring, per-pair AP, the oracle algebra, the matched nulls and the
  cross-salt transfer test are IMPORTED FROM STEP 89, not reimplemented.
  Artefact: experiments/learner_diversity_C_20260913/audit-C-learner-diversity.json
  Nothing is proposed for submission: this is diagnostic.
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evidence_aug.fold import fold_pools
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, check_protected, sha, write_json
from evidence_aug.step14_audit_truncation import CLF, SPLITS, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step74_length_shift import CHRONO3, CI, S1, SEEDS, build_d0, fit_predict, frame_at
from evidence_aug.step78_C_stage2_selection import fit_predict_rank
# --- everything below is step 89's machinery, imported rather than reimplemented -----------------
from evidence_aug.step89_C_complementarity import (fam_means, null_full_perm, null_resid_perm,
                                                   null_summary, oracle_block, r4, resplit, score_of)

AUTHOR = 'claude'
STEP = 92
OUTDIR = CV / 'experiments' / 'learner_diversity_C_20260913'
SALTS = ['audit-trunc-v1', 'audit-resplit-b', 'audit-resplit-c']
NPERM = 400
PERM_SEED = 20260913
LEVER, TRANSFER, RESOLUTION = 0.2, 0.49, 0.007
MIN_DELTA = round(RESOLUTION / (LEVER * TRANSFER), 4)     # 0.0714
STORED_HYBRID = 0.6159         # step 74/76/85/89, the shipping r382DTSP hybrid on this surface
STORED_REF = 0.5867            # step 74/89, ref_full_chrono3 at prefix 1.0
STORED_R382 = 0.6222           # step 74/89, r382_full_chrono3 at prefix 1.0
STORED_S89_MEAN_OFFDIAG = {'audit-trunc-v1': 0.6800, 'audit-resplit-b': 0.6388,
                           'audit-resplit-c': 0.6562}    # step 89 phase2, within-salt convention

# the six-member REAL LEARNER-CLASS FAMILY.  r382 (GBDT) is the incumbent anchor.
FAMILY = ['r382', 'logreg_l2', 'mlp', 'extra_trees', 'random_forest', 'knn']
SEED_FAMILY_N = 6                                    # matched to len(FAMILY)
MLP_SEED_TRIPLES = [(11, 12, 13), (21, 22, 23), (31, 32, 33),
                    (41, 42, 43), (51, 52, 53), (61, 62, 63)]
SINGLE_SEEDS = (11, 12, 13, 14, 15, 16)
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]     # 0.0 .. 1.0, weight ON THE NEW LEARNER
EQUAL_W = 0.5

LOGREG_C = 1.0
LOGREG_C_SENSITIVITY = [0.1, 10.0]
MLP_KW = dict(hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-3, learning_rate_init=1e-3,
              max_iter=300, early_stopping=True, n_iter_no_change=10, validation_fraction=0.15)
TREE_KW = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=5, n_jobs=2)
KNN_KW = dict(n_neighbors=25, weights='distance', n_jobs=2)

GATE = {
    'question': 'do genuinely different LEARNER CLASSES on the frozen stage-2 surface produce '
                'per-pair predictions decorrelated from r382, and does BLENDING them beat the best '
                'single model by more than the materiality bar?',
    'surface': 'the 162 audit pairs in 104 tables outside every held fold; per-pair AP@5, the '
               'quantity the composite is 0.2 x the mean of',
    'why_this_step_exists': 'step 89 tested seven LightGBM variants and only ROUTING between them. '
                            'Zero non-tree learners and zero blends. The only confirmed PASS on this '
                            'project (TabPFN-3, +0.0140873) is a LEARNER swap on identical columns.',
    'learner_family': 'SIX stage-2 arms on ONE frozen production stage-1 score and one identical '
                      'top-20 candidate set: r382 (LightGBM GBDT, the incumbent anchor), logreg_l2 '
                      '(L2 logistic regression, standardised -- additive, no interactions, the most '
                      'genuinely different inductive bias available on CPU), mlp (64-32 ReLU), '
                      'extra_trees and random_forest (BAGGING rather than boosting: the declared '
                      'intermediate case, a weaker form of diversity), knn (k=25, distance-weighted, '
                      'local and non-parametric)',
    'held_constant_across_arms': 'rows, target (relevance), 5-way pool-hash fold assignment, top-20 '
                                 'candidate restriction, and the identical 453-column feature matrix '
                                 '(70 structural + 3 chrono + 2 stage-1 + 378 ranker). Only the '
                                 'LEARNER changes.',
    'preprocessing_for_non_tree_learners':
        'declared in advance: SimpleImputer(median) -> StandardScaler -> estimator, the imputer and '
        'the scaler FITTED ON THE TRAINING FOLD ONLY inside every (family, split) cell. The audit '
        'candidate matrix was inspected for missingness BEFORE this pre-registration was written and '
        'contains 0 NaN, 0 inf and 0 of the -999 join sentinel in its 3240 x 453 candidate block, so '
        'the imputer is expected to be a no-op and that expectation is recorded and checked. Three '
        'columns are constant; StandardScaler leaves a zero-variance column unscaled. Tree learners '
        'get the raw matrix, exactly as r382 does.',
    'seed_convention': 'stochastic arms are 3-seed bags with seeds (11, 12, 13), exactly step 74 / '
                       'step 89 fit_predict convention. logreg_l2 and knn are deterministic and are '
                       'fitted once.',
    'PHASE_2_IS_DESCRIPTIVE_NOT_A_GATE':
        'the per-pair AP correlation is reported as a descriptive statistic and is explicitly NOT a '
        'stop rule. Per-pair AP is a lossy summary that discards the hand-level ranking errors a '
        'blend operates on: two models can have AP correlation 1 and identical AP vectors ([.5, 1]) '
        'and a hand-level blend can still lift mean AP from .75 to 1. Phase 3 runs in full whatever '
        'Phase 2 returns.',
    'two_correlation_conventions': {
        'within_salt': "np.corrcoef on the per-salt per-pair AP matrix, mean of the upper triangle. "
                       "Step 89's definition; its .680 / .639 / .656 are continuous with it.",
        'salt_averaged': 'each per-pair AP vector averaged over the three salts FIRST, then '
                         'correlated. Removes per-salt fit noise and MECHANICALLY INFLATES the '
                         'correlation. Reported so that the other agent\'s TabPFN-3 vs LGBM-hybrid '
                         '.8543, which uses this convention, is comparable.',
        'both_reported_for_every_pair': True},
    'blends': {
        'rank_average': 'within each pair, average-rank the top-20 candidates by each model score '
                        '(1 = best, ties averaged), take the weighted mean of the ranks, negate.',
        'score_average_z': 'within each pair, z-score each model score across the top-20 candidates, '
                           'take the weighted mean.',
        'PRIMARY_PRE_REGISTERED_ESTIMATOR':
            'the ALL-LEARNER EQUAL-WEIGHT RANK-AVERAGE blend over all six family members. One fixed '
            'estimator, no selection of any kind, chosen before any number existed.',
        'secondary_declared_set': 'the maximum over a set declared here in advance: rank-average and '
                                  'z-average equal-weight pairwise blends of r382 with each of the '
                                  'five new learners, plus the two all-learner equal-weight blends. '
                                  'Twelve estimators; the max over them is a selection over twelve '
                                  'and is labelled as such.',
        'weight_sweep': 'w in 0.0 .. 1.0 by 0.1 on the RANK-average pairwise blends only. The '
                        'selected weight is a POST-HOC SELECTION. Its result is reported labelled '
                        'post-hoc and is NEVER promoted above the equal-weight number.'},
    'baseline': 'BEST SINGLE MODEL = the in-sample argmax MAP@5 over the six family members on that '
                'salt. This is generous to the baseline and therefore conservative for the blend.',
    'primary_estimate': 'MINIMUM across the three pool partitions of (PRIMARY all-learner '
                        'equal-weight rank-average blend MAP@5 - best single model MAP@5)',
    'secondary_estimate': 'the same minimum for the max over the twelve declared equal-weight blends',
    'nulls': {
        'null_resid_perm_global': "PRIMARY, step 89's sharp matched null, imported unchanged: "
                                  'ap[v,p] = m[p] + e[v,p] with m the across-variant pair mean; '
                                  'permute e[v,.] across ALL pairs independently per variant. '
                                  'Preserves each arm POOLED mean and the common per-pair difficulty '
                                  'component; destroys all complementarity.',
        'null_resid_perm_family': 'the same permuted WITHIN family. Declared in advance as VACUOUS '
                                  'for the per-family oracle.',
        'null_full_perm_global': 'HARD null: permute each arm whole per-pair vector. Also '
                                 'decorrelates and therefore INFLATES the oracle.',
        'MATCHED_seed_family_mlp_bag3':
            'THE MATCHED SEED-FAMILY INSTRUMENT (OBS 19.20). Six copies of ONE model -- the MLP -- '
            'differing ONLY by random seed, on the SAME frozen stage 1 as every other arm, each a '
            '3-seed bag over a disjoint seed triple so that a seed-family member is constructed '
            'exactly like a real family member. Unlike step 89 null_seed_family, which varied '
            'STAGE 1 and therefore paid the 19.16 coupling tax while its real variants did not, this '
            'control is TAX-FREE and matched. PRIMARY seed instrument.',
        'seed_family_mlp_single': 'the same six as single seeds rather than 3-seed bags: noisier '
                                  'members, so a HARDER (higher) seed floor. Secondary.',
        'seed_family_lgbm_r382_single':
            'six single-seed LightGBM r382 fits. Step 88 measured stage-2 LightGBM seed inertness at '
            'max_abs 1.11e-16 and this family is expected to be DEGENERATE. If it is, the seed floor '
            'for a stage-2 LightGBM variant is ~0 and step 89 +0.0042 was measured against the wrong '
            'floor. The measured max_abs prediction difference between members is recorded either '
            'way; no number is forced.',
        'cross_salt_transfer': 'no synthetic null at all, and the most trustworthy number here: (a) '
                               'choose the per-pair oracle winner on partition A and apply it on B; '
                               '(b) choose the blend WEIGHT on A and apply it on B. If a pattern is '
                               'genuine it transfers; if it is selection noise it collapses.'},
    'diversity_exists_if': 'min over salts of (learner-family oracle gain - null_resid_perm_global '
                           'oracle gain) > 0',
    'blend_helps_if': 'min over salts of (blend MAP@5 - best single model MAP@5) > 0',
    'material_if': f'{LEVER} * min_delta * {TRANSFER} >= {RESOLUTION}  (min_delta >= {MIN_DELTA})',
    'verdict': 'ALIVE only if blend_helps AND material, on the PRIMARY pre-registered estimator. A '
               'blend that helps but not materially is recorded as REAL BUT IMMATERIAL, which is not '
               'ALIVE. A result below the bar is a FAILURE and is preserved as one.',
    'admission_rule': 'nothing in this step is admitted to production under any outcome; it is '
                      'diagnostic. "Admission" here means only: does the write-up record the blend '
                      'lane as ALIVE and worth a held-fold confirmation, or as DEAD.',
    'known_worse_arm_check': 'the standing 18.15/18.17 check: an arm known to be worse must fail the '
                             'gate. Two of them here: (a) the single arm struct_nos1 (MAP@5 ~.499 '
                             'against r382 .622) must give a large negative delta; (b) the '
                             'equal-weight rank-average blend of r382 with s1_pure (no stage 2 at '
                             'all) must not clear the bar.',
    'reproduction_controls': ['PHASE 0 GATE, the study STOPS on failure: the r382DTSP hybrid must '
                              'return MAP@5 0.6159, ref 0.5867 and r382 0.6222 on audit-trunc-v1, to '
                              '1e-9, reproducing step 89 exactly',
                              "step 89's seven variants are refitted here for the full correlation "
                              'matrix and their per-salt MAP@5 must reproduce step 89 stored values',
                              'the mean within-salt off-diagonal correlation of those seven must '
                              'reproduce step 89 .6800 / .6388 / .6562'],
    'coupling_tax': 'OBS 19.16. Every arm in this step -- the six real learners, all blends, and '
                    'BOTH matched seed families -- sits on ONE frozen production stage-1 score and '
                    'one identical top-20 candidate set. Nothing here changes stage 1, so nothing '
                    'here pays the tax. That is the correction to step 89, whose null_seed_family '
                    'varied stage 1 while its real variants did not.',
    'resolution': 'composite = 0.2 x delta MAP@5; the public leaderboard resolves +-.007 composite',
    'production_branch': 'none: diagnostic only, nothing is proposed for submission',
    'held_folds': 'folds 0, 1, 2 are not read for truth; load_fold(0) supplies FEATURE COLUMNS only, '
                  'and the audit/held table overlap is asserted empty at entry',
    'training_only': 'the 162 audit pairs in 104 tables outside every held fold',
    'gpu': 'not used. CPU only, 2 threads.',
    'note': 'written before any model in this file was fitted'}


# ------------------------------------------------------------------------------------------------
# NOT PART OF THE PRE-REGISTRATION.  The constants below were added after smoke run 1.  They are
# kept OUT of GATE and OUT of prediction-learner-diversity.json on purpose, so the pre-registration
# assertion in main() still holds.  None of them can change the verdict; see `provenance`.
# ------------------------------------------------------------------------------------------------
EXPECTED_PREREG_SHA = '39e3d49d992ce25313ff6bbb4e579804b1a93de9d24da262b22e134e6a8b72e7'   # 01:01 local
PROD_PREREG = CV / 'experiments' / 'learner_diversity_C_20260913' / 'prediction-learner-diversity.json'
STEP88_ARTEFACT = CV / 'experiments' / 'joint_refit_C_20260912' / 'audit-C-joint-refit.json'
POST_PREREG_DESCRIPTIVE_ADDITIONS = {
    'status': 'added after the smoke run; requested by the coordinator before the coordinator read '
              'any smoke number; cannot change the verdict',
    'evidence_of_benefit': {
        'why': 'smaller reliable gains are to be kept and stacked; the 2-3 point target is a '
               'milestone, not an admission threshold. A REAL BUT IMMATERIAL blend is therefore not '
               'worth zero, and whether it is statistically supported has to be known.',
        'B': 2000, 'seed': 20260913,
        'cluster': 'table: the audit tables are resampled with replacement and carry all their pairs',
        'interval': '95% percentile',
        'readouts': ['(a) PRIMARY per salt: mean delta, 95% percentile CI, fraction of draws > 0',
                     '(b) the salt-mean delta, every salt recomputed on ONE shared table draw',
                     '(c) direction count: salts with delta > 0',
                     '(d) per-family mean deltas per salt, descriptive subgroup protection only',
                     '(e) the same readouts for the best-of-twelve secondary, labelled a selection '
                     'over twelve',
                     '(f) composite equivalents 0.2 x delta, without and with the 0.49 transfer '
                     'discount',
                     '(g) a descriptive label SUPPORTED_SMALL_LEAD / UNSUPPORTED kept apart from '
                     'ALIVE/DEAD'],
        'best_single_model': 'the in-sample argmax over the six family members from the POINT '
                             'ESTIMATE on each salt, held FIXED across draws and never re-selected '
                             'per draw. The best-of-twelve blend identity is held fixed the same way.',
        'label_rule': 'SUPPORTED_SMALL_LEAD iff the salt-mean 95% CI lower bound > 0 AND the point '
                      'delta is > 0 on every salt (3 of 3); otherwise UNSUPPORTED',
        'label_is_not_the_verdict': True},
    'quality_vs_decorrelation': {
        'status': 'added by claude after smoke run 1, motivated by seeing the weakest new learners '
                  'post the lowest within-salt correlations with r382 in that smoke run; Phase 2 is '
                  'descriptive and this cannot change the verdict'},
    'blend_machinery_identity_check': {
        'status': 'added by claude after smoke run 1; a check on the code, not a statistic'}}


# --------------------------------------------------------------------------------------------
# learner makers.  Only the estimator changes; the fold/candidate/target logic is step 89's.
# --------------------------------------------------------------------------------------------
def _pipe(est):
    return Pipeline([('impute', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler()), ('est', est)])


def make_logreg(seed, C=LOGREG_C):
    return _pipe(LogisticRegression(penalty='l2', C=C, max_iter=2000, solver='lbfgs'))


def make_mlp(seed):
    return _pipe(MLPClassifier(random_state=seed, **MLP_KW))


def make_et(seed):
    return ExtraTreesClassifier(random_state=seed, **TREE_KW)


def make_rf(seed):
    return RandomForestClassifier(random_state=seed, **TREE_KW)


def make_knn(seed):
    return _pipe(KNeighborsClassifier(**KNN_KW))


SK_LEARNERS = {'logreg_l2': (make_logreg, (11,)), 'mlp': (make_mlp, SEEDS),
               'extra_trees': (make_et, SEEDS), 'random_forest': (make_rf, SEEDS),
               'knn': (make_knn, (11,))}


def fit_predict_sk(dtr, dte, cols, topk, maker, seeds):
    """step 74 / step 89 `fit_predict` with the LightGBM classifier swapped for an sklearn one.

    Same per-family, 5-way pool-hash OOF loop, same candidate restriction, same target, same
    seed-bagging of predict_proba.  Nothing about the protocol changes; only the learner does.
    """
    pred = np.full(dte.height, -1e6)
    ftr, fte = dtr['behavior_family'].to_numpy(), dte['behavior_family'].to_numpy()
    str_, ste = dtr['split'].to_numpy(), dte['split'].to_numpy()
    ytr = dtr['relevance'].to_numpy().astype(int)
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
            acc = np.zeros(len(te))
            for seed in seeds:
                m = maker(seed)
                m.fit(Xtr_all[tr], ytr[tr])
                acc += m.predict_proba(Xte_all[te])[:, 1] / len(seeds)
            pred[te] = acc
    return pred


# --------------------------------------------------------------------------------------------
# blend algebra.  Everything happens WITHIN a pair, among the top-20 candidates only.
# --------------------------------------------------------------------------------------------
def within_pair_negrank(pred, sl, cand):
    """average rank among the pair's candidates, ties averaged, NEGATED so higher is better."""
    out = np.zeros(len(pred))
    for a, b in sl:
        idx = np.arange(a, b)[cand[a:b]]
        x = pred[idx].astype(float)
        order = np.argsort(-x, kind='stable')
        srt = x[order]
        n = len(x)
        r = np.empty(n)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and srt[j + 1] == srt[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        out[idx] = -r
    return out


def within_pair_z(pred, sl, cand):
    """z-score among the pair's candidates."""
    out = np.zeros(len(pred))
    for a, b in sl:
        idx = np.arange(a, b)[cand[a:b]]
        x = pred[idx].astype(float)
        out[idx] = (x - x.mean()) / (x.std() + 1e-12)
    return out


def combine(transformed, weights, cand):
    """weighted mean of already-transformed vectors; non-candidates back to -1e6 for ranked_lists."""
    w = np.asarray(weights, dtype=float)
    z = np.zeros(len(transformed[0]))
    for wi, t in zip(w, transformed):
        z = z + wi * t
    z = z / w.sum()
    out = np.full(len(z), -1e6)
    out[cand] = z[cand]
    return out


def offdiag_mean_min(M):
    V = M.shape[0]
    iu = np.triu_indices(V, 1)
    return float(M[iu].mean()), float(M[iu].min()), float(M[iu].max())


def dig(obj, path):
    cur = obj
    for k in path.split('/'):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return cur


# --------------------------------------------------------------------------------------------
def main(smoke_dir=None):
    t0 = time.time()
    global OUTDIR
    smoke = smoke_dir is not None
    salts = SALTS[:2] if smoke else SALTS
    nperm = 25 if smoke else NPERM
    weights = [0.0, 0.5, 1.0] if smoke else WEIGHTS
    nseed = 3 if smoke else SEED_FAMILY_N
    if smoke:
        OUTDIR = Path(smoke_dir)

    entry = check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- pre-registration, written BEFORE any fit and never edited --------------------------
    pre_path = OUTDIR / 'prediction-learner-diversity.json'
    payload = {'author': AUTHOR, 'step': STEP, 'written_before_any_fit': True,
               'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
               'gate': GATE, 'salts': list(SALTS), 'family': list(FAMILY),
               'seed_family_n': SEED_FAMILY_N, 'mlp_seed_triples': [list(t) for t in MLP_SEED_TRIPLES],
               'single_seeds': list(SINGLE_SEEDS), 'weight_grid': WEIGHTS, 'equal_weight': EQUAL_W,
               'nperm': NPERM, 'perm_seed': PERM_SEED,
               'hyperparameters': {'logreg_C': LOGREG_C, 'logreg_C_sensitivity': LOGREG_C_SENSITIVITY,
                                   'mlp': {k: (list(v) if isinstance(v, tuple) else v)
                                           for k, v in MLP_KW.items()},
                                   'trees': TREE_KW, 'knn': KNN_KW,
                                   'lightgbm_clf_reused_unchanged_from_step14': CLF},
               'materiality': {'lever': LEVER, 'transfer_discount': TRANSFER,
                               'public_resolution': RESOLUTION, 'min_delta_required': MIN_DELTA},
               'reused_without_refit': [],
               'imported_unchanged_from_step89': ['resplit', 'score_of', 'fam_means', 'r4',
                                                  'oracle_block', 'null_resid_perm', 'null_full_perm',
                                                  'null_summary'],
               'not_touched': ['OBSERVATIONS.md', 'held folds 0/1/2 truth',
                               'Codex Resume/ and the original tree (read-only)',
                               'the GPU (reserved for another agent)']}
    prereg_new = not pre_path.exists()
    if prereg_new:
        write_json(pre_path, payload)
    else:                                    # already on disk: prove it was not edited
        old = json.loads(pre_path.read_text(encoding='utf-8'))
        a = {k: v for k, v in old.items() if k != 'utc'}
        b = json.loads(json.dumps({k: v for k, v in payload.items() if k != 'utc'}))
        assert a == b, 'the pre-registration on disk differs from the GATE constant -- STOP'
    prereg_sha = sha(pre_path)
    if not smoke:                            # the production file must be the one written at 01:01
        assert prereg_sha == EXPECTED_PREREG_SHA, 'pre-registration sha256 changed -- STOP'

    # read-only: step 88's measured refit-minus-frozen contrast, read at the START so that a missing
    # or changed file fails in seconds rather than at the end of a long run
    s88 = json.loads(STEP88_ARTEFACT.read_text(encoding='utf-8'))
    s88_block = {'artefact': str(STEP88_ARTEFACT), 'sha256': sha(STEP88_ARTEFACT), 'read_only': True,
                 'file_mtime_utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00',
                                                 time.gmtime(STEP88_ARTEFACT.stat().st_mtime)),
                 'verdict_coupling_tax_refit_minus_frozen': s88['verdict']['coupling_tax_refit_minus_frozen'],
                 'gate_frozen_comparator_definition': s88['gate']['frozen_comparator_definition'],
                 'step88_held_folds_touched': s88.get('held_folds_touched')}
    s88_tax = s88_block['verdict_coupling_tax_refit_minus_frozen']
    assert {'mean_over_9_cells', 'positive_in', 'of_cells', 'min', 'max'} <= set(s88_tax), \
        'step 88 artefact schema changed -- STOP'
    del s88
    prod_prereg = json.loads(PROD_PREREG.read_text(encoding='utf-8'))    # read-only, in smoke runs too
    prod_prereg_utc, prod_prereg_sha = prod_prereg['utc'], sha(PROD_PREREG)
    del prod_prereg

    print('=' * 100)
    print('STEP 92 [C].  LEARNER-CLASS DIVERSITY AND BLENDING ON THE FROZEN STAGE-2 SURFACE'
          + ('   [SMOKE]' if smoke else ''))
    print('=' * 100)

    # ---- frame ------------------------------------------------------------------------------
    d0, struct, rcols = build_d0()
    d = frame_at(d0, 1.0)
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    overlap = sorted(set(d['table_id']) & held)
    assert not overlap, 'audit surface overlaps a held fold -- stop'
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pairs = sorted(truths)
    fams = np.array([labels[p] for p in pairs])
    tabs = np.array([ptab[p] for p in pairs])
    P = len(pairs)
    uniq = d['pair_id'].to_numpy()
    starts = np.sort(np.unique(uniq, return_index=True)[1])
    sl = list(zip(starts, list(starts[1:]) + [d.height]))
    assert [uniq[a] for a, _ in sl] == pairs, 'pair order mismatch between frame slices and pairs list'
    cand = (d['oof_rank'] <= TOP).to_numpy()

    ALLCOLS = struct + CHRONO3 + S1 + rcols
    Xc = d.select(ALLCOLS).to_numpy().astype(np.float64)[cand]
    missing_audit = {'candidate_rows': int(Xc.shape[0]), 'columns': int(Xc.shape[1]),
                     'nan_cells': int(np.isnan(Xc).sum()), 'inf_cells': int(np.isinf(Xc).sum()),
                     'join_sentinel_minus999_cells': int((Xc == -999.0).sum()),
                     'zero_variance_columns': int((Xc.std(axis=0) == 0).sum()),
                     'imputer_is_a_noop': bool(np.isnan(Xc).sum() == 0)}

    res = {'author': AUTHOR, 'step': STEP, 'smoke_run': smoke, 'training_only': True,
           'held_folds_touched': False,
           'held_folds_note': 'folds 0, 1, 2 are not read for truth; load_fold(0) supplies FEATURE '
                              'COLUMNS only, exactly as steps 74/76/77/84/85b/87/89',
           'held_fold_overlap_check': {'held_tables': len(held), 'audit_tables': int(d['table_id'].n_unique()),
                                       'intersection': overlap, 'intersection_is_empty': True},
           'gpu_used': False,
           'production_branch': 'none: diagnostic only, nothing is proposed for submission',
           'question': GATE['question'], 'pre_registration': GATE,
           'prediction_file': str(pre_path), 'prediction_file_sha256': prereg_sha,
           'prediction_written_before_any_fit': True,
           'protected_hashes_entry': entry,
           'population': {'pairs': P, 'tables': int(d['table_id'].n_unique()), 'hands': d.height,
                          'listed_hands': int((d['relevance'] == 1).sum()),
                          'candidate_rows': int(cand.sum()),
                          'pairs_by_family': {f: int((fams == f).sum()) for f in FAMILIES},
                          'n_struct': len(struct), 'n_chrono': len(CHRONO3), 'n_stage1': len(S1),
                          'n_ranker': len(rcols), 'n_feature_columns': len(ALLCOLS)},
           'feature_matrix_missingness': missing_audit,
           'phase0_reproduction_gate': {}, 'phase1_learners': {},
           'phase2_correlation_DESCRIPTIVE_NOT_A_GATE': {}, 'phase3_blends': {},
           'phase4_nulls_and_seed_families': {}, 'cross_salt_structure': {}, 'salt_minima': {},
           'verdict': {}, 'limits': [], 'provenance': []}

    # ---- Phase 0 + 1: fit everything ---------------------------------------------------------
    s89_designs = {'s1_pure': ('none', []),
                   'ref': ('clf', struct + CHRONO3 + S1),
                   'r382': ('clf', struct + CHRONO3 + S1 + rcols),
                   'struct_nos1': ('clf', struct + CHRONO3),
                   'r382_nostruct': ('clf', rcols + CHRONO3 + S1),
                   'rank_ref': ('rank', struct + CHRONO3 + S1),
                   'rank_r382': ('rank', struct + CHRONO3 + S1 + rcols)}
    s89_names = list(s89_designs)
    all_names = s89_names + [n for n in FAMILY if n not in s89_names]

    print(f'\n  PHASE 0/1.  {len(all_names)} arms x {len(salts)} partitions '
          f'({len(s89_names)} step-89 LightGBM variants refitted + {len(SK_LEARNERS)} new learner classes)')
    AP, PRED, OV = {}, {}, {}
    per_arm = {}
    conv_warn = 0
    assert np.array_equal(resplit(d, SALTS[0])['split'].to_numpy(), d['split'].to_numpy()), \
        'resplit under the v1 salt does not reproduce the stored split column'
    for salt in salts:
        ds = resplit(d, salt)
        AP[salt], PRED[salt], OV[salt] = {}, {}, {}
        for n in all_names:
            t1 = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                if n in s89_designs:
                    kind, cols = s89_designs[n]
                    if kind == 'none':
                        pred = ds['oof'].to_numpy().astype(float)
                    elif kind == 'clf':
                        pred = fit_predict(ds, ds, cols, TOP)
                    else:
                        pred = fit_predict_rank(ds, ds, cols, TOP)
                else:
                    maker, sds = SK_LEARNERS[n]
                    pred = fit_predict_sk(ds, ds, ALLCOLS, TOP, maker, sds)
                conv_warn += sum(1 for x in w if 'onverg' in str(x.message))
            ap, ov, _ = score_of(ds, pred, truths, pairs)
            AP[salt][n], PRED[salt][n], OV[salt][n] = ap, pred, ov
            per_arm.setdefault(salt, {})[n] = {
                'map5': r4(ap.mean()), 'mean_overlap': r4(ov.mean()),
                'by_family': {k: r4(v) for k, v in fam_means(ap, fams).items()},
                'learner_class': ('LightGBM GBDT (step 89 variant)' if n in s89_designs
                                  else {'logreg_l2': 'L2 logistic regression (linear, additive)',
                                        'mlp': 'MLP 64-32 ReLU (neural)',
                                        'extra_trees': 'ExtraTrees (bagging)',
                                        'random_forest': 'RandomForest (bagging)',
                                        'knn': 'k-NN k=25 distance-weighted (local)'}[n])}
            log(f'{salt} {n}', json.dumps({'map5': r4(ap.mean()), 'ov': r4(ov.mean()),
                                           'sec': round(time.time() - t1)}))
        jr, jf = 'r382', 'ref'
        hyb = np.where(fams == CI, AP[salt][jf], AP[salt][jr])
        per_arm[salt]['_incumbent_hybrid_r382DTSP'] = {
            'map5': r4(hyb.mean()), 'mean_overlap': None,
            'by_family': {k: r4(v) for k, v in fam_means(hyb, fams).items()},
            'learner_class': 'r382 for DT/SP, ref for CI: the model that ships'}

    v1 = salts[0]
    ctrl = {'hybrid_r382DTSP': per_arm[v1]['_incumbent_hybrid_r382DTSP']['map5'],
            'hybrid_stored': STORED_HYBRID,
            'ref': per_arm[v1]['ref']['map5'], 'ref_stored': STORED_REF,
            'r382': per_arm[v1]['r382']['map5'], 'r382_stored': STORED_R382,
            'salt': v1}
    ctrl['PASS'] = bool(abs(ctrl['hybrid_r382DTSP'] - STORED_HYBRID) < 1e-9
                        and abs(ctrl['ref'] - STORED_REF) < 1e-9
                        and abs(ctrl['r382'] - STORED_R382) < 1e-9)
    ctrl['the_study_stops_on_failure'] = True
    res['phase0_reproduction_gate'] = ctrl
    print(f"  PHASE 0 GATE {'PASSED' if ctrl['PASS'] else 'FAILED'}: hybrid "
          f"{ctrl['hybrid_r382DTSP']:.4f} / ref {ctrl['ref']:.4f} / r382 {ctrl['r382']:.4f}")
    assert ctrl['PASS'], f'PHASE 0 REPRODUCTION GATE FAILED -- the study stops: {ctrl}'

    res['phase1_learners'] = {
        'arms': {n: (per_arm[v1][n]['learner_class']) for n in all_names},
        'n_feature_columns_every_arm': len(ALLCOLS),
        'note_on_step89_arms': 'ref / struct_nos1 / r382_nostruct / rank_* use their own step-89 '
                               'column subsets; the SIX FAMILY members all use the identical '
                               f'{len(ALLCOLS)}-column matrix, so only the learner differs among them.',
        'sklearn_convergence_warnings': conv_warn,
        'per_arm_by_salt': per_arm}

    # ---- Phase 2: correlation, both conventions.  DESCRIPTIVE, NOT A GATE ---------------------
    print('\n  PHASE 2.  per-pair AP correlation (DESCRIPTIVE, not a gate) -- both conventions')
    ph2 = {'IS_THIS_A_GATE': False,
           'why_not': GATE['PHASE_2_IS_DESCRIPTIVE_NOT_A_GATE'],
           'conventions': GATE['two_correlation_conventions'], 'by_salt': {}}
    for salt in salts:
        M12 = np.corrcoef(np.column_stack([AP[salt][n] for n in all_names]), rowvar=False)
        i89 = [all_names.index(n) for n in s89_names]
        M89 = M12[np.ix_(i89, i89)]
        ifam = [all_names.index(n) for n in FAMILY]
        MF = M12[np.ix_(ifam, ifam)]
        m89, n89, x89 = offdiag_mean_min(M89)
        m12, n12, x12 = offdiag_mean_min(M12)
        mfam, nfam, xfam = offdiag_mean_min(MF)
        jr = all_names.index('r382')
        ph2['by_salt'][salt] = {
            'vs_r382': {n: r4(M12[jr, all_names.index(n)]) for n in all_names if n != 'r382'},
            'mean_offdiag_step89_seven': r4(m89), 'min_offdiag_step89_seven': r4(n89),
            'max_offdiag_step89_seven': r4(x89),
            'step89_stored_mean_offdiag': STORED_S89_MEAN_OFFDIAG.get(salt),
            'reproduces_step89_mean_offdiag':
                (None if salt not in STORED_S89_MEAN_OFFDIAG
                 else bool(abs(r4(m89) - STORED_S89_MEAN_OFFDIAG[salt]) < 1e-9)),
            'mean_offdiag_all_twelve': r4(m12), 'min_offdiag_all_twelve': r4(n12),
            'mean_offdiag_six_member_family': r4(mfam), 'min_offdiag_six_member_family': r4(nfam),
            'mean_offdiag_five_new_learners_only':
                r4(offdiag_mean_min(M12[np.ix_([all_names.index(n) for n in FAMILY if n != 'r382'],
                                               [all_names.index(n) for n in FAMILY if n != 'r382'])])[0]),
            'full_matrix': {a: {b: r4(M12[i, j]) for j, b in enumerate(all_names)}
                            for i, a in enumerate(all_names)}}
        z = ph2['by_salt'][salt]
        print(f"    {salt:<18} step89-7 mean off-diag {z['mean_offdiag_step89_seven']:.4f} "
              f"(stored {z['step89_stored_mean_offdiag']})   all-12 {z['mean_offdiag_all_twelve']:.4f}"
              f"   vs r382: " + '  '.join(f"{k} {v:.3f}" for k, v in z['vs_r382'].items()
                                          if k in FAMILY))
    # salt-averaged convention
    APBAR = {n: np.mean([AP[s][n] for s in salts], axis=0) for n in all_names}
    Mbar = np.corrcoef(np.column_stack([APBAR[n] for n in all_names]), rowvar=False)
    jr = all_names.index('r382')
    i89 = [all_names.index(n) for n in s89_names]
    ifam = [all_names.index(n) for n in FAMILY]
    ph2['salt_averaged'] = {
        'what': 'per-pair AP averaged over the salts FIRST, then correlated. Comparable with the '
                'other agent TabPFN-3 vs LGBM-hybrid .8543; NOT comparable with step 89 .680/.639/.656.',
        'n_salts_averaged': len(salts),
        'vs_r382': {n: r4(Mbar[jr, all_names.index(n)]) for n in all_names if n != 'r382'},
        'mean_offdiag_step89_seven': r4(offdiag_mean_min(Mbar[np.ix_(i89, i89)])[0]),
        'mean_offdiag_all_twelve': r4(offdiag_mean_min(Mbar)[0]),
        'mean_offdiag_six_member_family': r4(offdiag_mean_min(Mbar[np.ix_(ifam, ifam)])[0]),
        'full_matrix': {a: {b: r4(Mbar[i, j]) for j, b in enumerate(all_names)}
                        for i, a in enumerate(all_names)}}
    ph2['inflation_from_averaging'] = {
        'mean_offdiag_step89_seven_within_salt_mean':
            r4(np.mean([ph2['by_salt'][s]['mean_offdiag_step89_seven'] for s in salts])),
        'mean_offdiag_step89_seven_salt_averaged': ph2['salt_averaged']['mean_offdiag_step89_seven'],
        'mean_offdiag_all_twelve_within_salt_mean':
            r4(np.mean([ph2['by_salt'][s]['mean_offdiag_all_twelve'] for s in salts])),
        'mean_offdiag_all_twelve_salt_averaged': ph2['salt_averaged']['mean_offdiag_all_twelve'],
        'reading': 'the difference between the two rows is METHOD, not model. Any comparison of a '
                   'salt-averaged correlation with step 89 within-salt .680/.639/.656 must be '
                   'corrected by roughly this amount before it means anything.'}
    ph2['inflation_from_averaging']['delta_step89_seven'] = r4(
        ph2['inflation_from_averaging']['mean_offdiag_step89_seven_salt_averaged']
        - ph2['inflation_from_averaging']['mean_offdiag_step89_seven_within_salt_mean'])
    ph2['inflation_from_averaging']['delta_all_twelve'] = r4(
        ph2['inflation_from_averaging']['mean_offdiag_all_twelve_salt_averaged']
        - ph2['inflation_from_averaging']['mean_offdiag_all_twelve_within_salt_mean'])
    # ---- is a low correlation with r382 a sign of a DIFFERENT learner, or just of a WORSE one? ----
    # (added after smoke run 1; descriptive, like all of Phase 2)
    NEW = [n for n in FAMILY if n != 'r382']
    others = [n for n in all_names if n != 'r382']
    refs = [n for n in s89_names if n != 'r382']     # five LightGBM stage-2 variants + raw stage 1
    qrows = {}
    for n in others:
        qrows[n] = {'map5_mean_over_salts': r4(np.mean([per_arm[s][n]['map5'] for s in salts])),
                    'corr_vs_r382_within_salt_mean':
                        r4(np.mean([ph2['by_salt'][s]['vs_r382'][n] for s in salts])),
                    'corr_vs_r382_salt_averaged': ph2['salt_averaged']['vs_r382'][n],
                    'is_a_new_learner_class': n in NEW}
    xq = np.array([qrows[n]['map5_mean_over_salts'] for n in others])
    yq = np.array([qrows[n]['corr_vs_r382_within_salt_mean'] for n in others])
    xr = np.array([qrows[n]['map5_mean_over_salts'] for n in refs])
    yr = np.array([qrows[n]['corr_vs_r382_within_salt_mean'] for n in refs])
    slope, icpt = (float(v) for v in np.polyfit(xr, yr, 1))
    vsline = {}
    for n in NEW:
        x0, y0 = qrows[n]['map5_mean_over_salts'], qrows[n]['corr_vs_r382_within_salt_mean']
        line = icpt + slope * x0
        vsline[n] = {'map5_mean_over_salts': x0, 'observed_corr_vs_r382': y0,
                     'reference_line_corr_at_this_map5': r4(line),
                     'residual_observed_minus_line': r4(y0 - line),
                     'below_the_reference_line': bool(y0 < line),
                     'map5_inside_reference_range': bool(xr.min() <= x0 <= xr.max())}
    ph2['quality_vs_decorrelation'] = {
        'status': POST_PREREG_DESCRIPTIVE_ADDITIONS['quality_vs_decorrelation']['status'],
        'why': 'the per-pair AP correlation between two models is attenuated by BOTH models own '
               'noise, so a weaker model is mechanically less correlated with r382 whatever its '
               'learner class. Before reading a low correlation as "a different inductive bias" it '
               'has to be set against what LightGBM arms of the same quality already give.',
        'per_arm': qrows,
        'pearson_r_between_arm_map5_and_its_corr_with_r382': r4(float(np.corrcoef(xq, yq)[0, 1])),
        'n_arms_in_that_pearson_r': len(others),
        'reference_line': {
            'arms': refs,
            'what': 'ordinary least squares of within-salt-mean corr-with-r382 on mean MAP@5, over the '
                    'six step-89 arms other than r382 (five LightGBM stage-2 variants and raw stage 1)',
            'slope': r4(slope), 'intercept': r4(icpt),
            'pearson_r_over_reference_arms': r4(float(np.corrcoef(xr, yr)[0, 1])),
            'map5_range': [r4(xr.min()), r4(xr.max())]},
        'new_learners_vs_reference_line': vsline,
        'reading': 'a new learner ON the reference line is as correlated with r382 as a LightGBM arm '
                   'of the same quality: its lower correlation is explained by lower quality, not by '
                   'learner class. A clearly NEGATIVE residual means more decorrelated than quality '
                   'alone predicts.',
        'caveat': 'six reference points; a straight line is a rough descriptive aid, not a fitted '
                  'model of attenuation. A learner whose MAP@5 lies outside the reference range is an '
                  'extrapolation and is flagged. Descriptive only; it cannot change the verdict.'}
    res['phase2_correlation_DESCRIPTIVE_NOT_A_GATE'] = ph2
    print(f"    quality vs decorrelation (descriptive): Pearson(arm MAP@5, corr with r382) over "
          f"{len(others)} arms "
          f"{ph2['quality_vs_decorrelation']['pearson_r_between_arm_map5_and_its_corr_with_r382']:+.4f};"
          f"  residual vs LightGBM reference line: "
          + '  '.join(f"{n} {vsline[n]['residual_observed_minus_line']:+.3f}" for n in NEW))
    print(f"    SALT-AVERAGED convention: step89-7 mean off-diag "
          f"{ph2['salt_averaged']['mean_offdiag_step89_seven']:.4f}  all-12 "
          f"{ph2['salt_averaged']['mean_offdiag_all_twelve']:.4f}   "
          f"(inflation vs within-salt {ph2['inflation_from_averaging']['delta_all_twelve']:+.4f})")

    # ---- Phase 3: BLENDS ----------------------------------------------------------------------
    print('\n  PHASE 3.  blends (rank-average and z-average).  Runs in full regardless of Phase 2.')
    BLEND_AP = {}          # per-pair AP vectors kept for the post-prereg evidence-of-benefit readout
    for salt in salts:
        ds = resplit(d, salt)
        Tr = {n: within_pair_negrank(PRED[salt][n], sl, cand) for n in all_names}
        Tz = {n: within_pair_z(PRED[salt][n], sl, cand) for n in all_names}
        A = np.column_stack([AP[salt][n] for n in FAMILY])
        blk = oracle_block(A, fams, FAMILY)
        best_single, best_name = blk['best_single'], blk['best_single_name']

        def m5(vec):
            ap, _, _ = score_of(ds, vec, truths, pairs)
            return float(ap.mean()), ap

        declared = {}
        for mode, T in (('rank', Tr), ('z', Tz)):
            for n in NEW:
                v, apv = m5(combine([T['r382'], T[n]], [1 - EQUAL_W, EQUAL_W], cand))
                declared[f'{mode}_r382+{n}'] = (v, apv)
            v, apv = m5(combine([T[n] for n in FAMILY], [1.0] * len(FAMILY), cand))
            declared[f'{mode}_all_six'] = (v, apv)
        primary_key = 'rank_all_six'
        primary = declared[primary_key][0]
        best_declared_key = max(declared, key=lambda k: declared[k][0])
        best_declared = declared[best_declared_key][0]
        BLEND_AP[salt] = {'primary': declared[primary_key][1], 'best12': declared[best_declared_key][1],
                          'best12_key': best_declared_key, 'best_single_name': best_name,
                          'best_single': AP[salt][best_name], 'primary_mean': primary,
                          'best12_mean': best_declared, 'best_single_mean': best_single}

        sweep = {}
        for n in NEW:
            row = {}
            for w in weights:
                row[f'{w:.1f}'] = r4(m5(combine([Tr['r382'], Tr[n]], [1 - w, w], cand))[0])
            bw = max(row, key=lambda k: row[k])     # ties resolve to the SMALLEST weight, i.e. to
            sweep[n] = {'by_weight': row,           # "do not blend": the conservative direction
                        'POSTHOC_best_weight': float(bw),
                        'POSTHOC_best_map5': row[bw],
                        'n_weights_tied_at_best': int(sum(1 for v in row.values() if v == row[bw])),
                        'POSTHOC_best_delta_vs_best_single': r4(row[bw] - best_single),
                        'equal_weight_map5': r4(declared[f'rank_r382+{n}'][0]),
                        'equal_weight_delta_vs_best_single':
                            r4(declared[f'rank_r382+{n}'][0] - best_single)}
        posthoc_best_over_learners = max(sweep[n]['POSTHOC_best_map5'] for n in NEW)
        n_prefer_no_blend = int(sum(1 for n in NEW if sweep[n]['POSTHOC_best_weight'] == 0.0))

        # identity check on the blend machinery: at w = 0 the rank blend IS r382 (a monotone
        # within-pair transform with ties averaged preserves the ordering exactly)
        w0 = r4(m5(combine([Tr['r382'], Tr[NEW[0]]], [1.0, 0.0], cand))[0])
        ident = {'what': 'the w=0.0 rank blend must reproduce plain r382 MAP@5 exactly; the rank '
                         'transform is monotone within a pair and averages ties, so it cannot change '
                         'the ordering. If this fails the blend machinery is broken.',
                 'w0_rank_blend_map5': w0, 'plain_r382_map5': r4(float(AP[salt]['r382'].mean())),
                 'EXACT': bool(abs(w0 - r4(float(AP[salt]['r382'].mean()))) < 1e-9)}

        # known-worse-arm checks
        kw_blend, _ = m5(combine([Tr['r382'], Tr['s1_pure']], [1 - EQUAL_W, EQUAL_W], cand))
        kw = {'arm_a_single_struct_nos1': {
                  'map5': r4(float(AP[salt]['struct_nos1'].mean())),
                  'delta_vs_best_single': r4(float(AP[salt]['struct_nos1'].mean()) - best_single),
                  'fails_the_gate': bool(float(AP[salt]['struct_nos1'].mean()) < best_single)},
              'arm_b_rank_blend_r382_with_s1_pure': {
                  'map5': r4(kw_blend), 'delta_vs_best_single': r4(kw_blend - best_single),
                  'clears_the_materiality_bar': bool(kw_blend - best_single >= MIN_DELTA),
                  'fails_the_gate': bool(kw_blend - best_single < MIN_DELTA)}}

        res['phase3_blends'][salt] = {
            'best_single_model': best_name, 'best_single_map5': r4(best_single),
            'per_family_member_map5': {n: r4(x) for n, x in blk['per_variant_mean'].items()},
            'PRIMARY_all_six_equal_weight_rank_average': {
                'what': 'the single pre-registered estimator: equal-weight rank-average of all six '
                        'family members. No selection of any kind.',
                'map5': r4(primary), 'delta_vs_best_single': r4(primary - best_single),
                'composite_0.2_x_delta_x_0.49': round(LEVER * (primary - best_single) * TRANSFER, 6),
                'clears_bar': bool(primary - best_single >= MIN_DELTA)},
            'declared_set_of_twelve_equal_weight_blends': {
                k: {'map5': r4(v), 'delta_vs_best_single': r4(v - best_single),
                    'composite_0.2_x_delta_x_0.49': round(LEVER * (v - best_single) * TRANSFER, 6)}
                for k, (v, _) in declared.items()},
            'best_of_the_declared_twelve': {
                'which': best_declared_key, 'map5': r4(best_declared),
                'delta_vs_best_single': r4(best_declared - best_single),
                'composite_0.2_x_delta_x_0.49': round(LEVER * (best_declared - best_single) * TRANSFER, 6),
                'selection_note': 'a maximum over TWELVE pre-declared estimators; not a clean single '
                                  'estimator, and reported below the PRIMARY one',
                'clears_bar': bool(best_declared - best_single >= MIN_DELTA)},
            'POSTHOC_weight_sweep_rank_blends': {
                'LABEL': 'POST-HOC SELECTION. The weight is chosen by looking at this very number. '
                         'It is never promoted above the equal-weight result.',
                'grid': [float(w) for w in weights], 'per_learner': sweep,
                'best_over_learners_map5': r4(posthoc_best_over_learners),
                'best_over_learners_delta_vs_best_single': r4(posthoc_best_over_learners - best_single),
                'composite_0.2_x_delta_x_0.49':
                    round(LEVER * (posthoc_best_over_learners - best_single) * TRANSFER, 6),
                'n_learners_whose_posthoc_best_weight_is_zero': n_prefer_no_blend,
                'n_new_learners': len(NEW),
                'reading_if_all_zero': 'if the post-hoc optimum weight is 0 for every new learner '
                                       'then the sweep, given full freedom to look at the answer, '
                                       'still prefers pure r382 -- the strongest available form of '
                                       'the negative result'},
            'blend_machinery_identity_check': ident,
            'known_worse_arm_check': kw}
        q = res['phase3_blends'][salt]
        print(f"    {salt:<18} best single {best_name} {best_single:.4f}   PRIMARY blend "
              f"{primary:.4f} ({primary - best_single:+.4f})   best-of-12 {best_declared_key} "
              f"{best_declared:.4f} ({best_declared - best_single:+.4f})   POSTHOC-sweep "
              f"{posthoc_best_over_learners:.4f} ({posthoc_best_over_learners - best_single:+.4f})")

    # ---- Phase 4: matched nulls and the MATCHED seed families ---------------------------------
    print('\n  PHASE 4.  matched nulls + the MATCHED stage-2 seed families (stage 1 and top-20 held fixed)')
    ph4 = {'oracle_and_nulls': {}, 'seed_families': {}}
    for salt in salts:
        A = np.column_stack([AP[salt][n] for n in FAMILY])
        blk = oracle_block(A, fams, FAMILY)
        ph4['oracle_and_nulls'][salt] = {
            'best_single': r4(blk['best_single']), 'best_single_name': blk['best_single_name'],
            'oracle_pair': r4(blk['oracle_pair']), 'oracle_pair_gain': r4(blk['oracle_pair_gain']),
            'oracle_family': r4(blk['oracle_family']),
            'oracle_family_gain': r4(blk['oracle_family_gain']),
            'oracle_family_choice': blk['family_choice'],
            'null_resid_perm_global': null_summary(A, fams, FAMILY, null_resid_perm, nperm,
                                                   PERM_SEED, 'global'),
            'null_resid_perm_family': null_summary(A, fams, FAMILY, null_resid_perm, nperm,
                                                   PERM_SEED + 1, 'family'),
            'null_full_perm_global': null_summary(A, fams, FAMILY, null_full_perm, nperm,
                                                  PERM_SEED + 2, 'global')}
        z = ph4['oracle_and_nulls'][salt]
        print(f"    {salt:<18} oracle/pair gain {z['oracle_pair_gain']:+.4f}   real-null "
              f"{z['null_resid_perm_global']['real_minus_null_oracle_pair']:+.4f}  "
              f"p={z['null_resid_perm_global']['perm_p_oracle_pair']}")

    seed_specs = {
        'MATCHED_mlp_bag3': ('mlp_bag3', [MLP_SEED_TRIPLES[i] for i in range(nseed)]),
        'mlp_single': ('mlp_single', [(s,) for s in SINGLE_SEEDS[:nseed]]),
        'lgbm_r382_single': ('lgbm_single', [(s,) for s in SINGLE_SEEDS[:nseed]])}
    for tag, (kind, seedsets) in seed_specs.items():
        ph4['seed_families'][tag] = {
            'what': GATE['nulls']['MATCHED_seed_family_mlp_bag3'] if tag == 'MATCHED_mlp_bag3'
                    else GATE['nulls'][f'seed_family_{tag}'],
            'members': len(seedsets), 'seed_sets': [list(s) for s in seedsets],
            'stage1_and_top20_candidate_set_identical_across_members': True,
            'why_this_is_the_matched_control': 'every member varies only the STAGE-2 seed on the one '
                                               'frozen production stage-1 score, so stage 1 and the '
                                               'top-20 candidate set are identical across members, '
                                               'exactly as they are across the six real learner arms. '
                                               "Step 89's null_seed_family varied STAGE 1 and so "
                                               'changed the candidate sets, while its real variants '
                                               'did not. (The quoted GATE text in `what` predates '
                                               'step 88; see provenance.)',
            'by_salt': {}}
        for salt in salts:
            ds = resplit(d, salt)
            B, preds = np.zeros((P, len(seedsets))), []
            for j, sset in enumerate(seedsets):
                t1 = time.time()
                if kind == 'lgbm_single':
                    pr = fit_predict(ds, ds, ALLCOLS, TOP, seeds=sset)
                else:
                    pr = fit_predict_sk(ds, ds, ALLCOLS, TOP, make_mlp, sset)
                ap, _, _ = score_of(ds, pr, truths, pairs)
                B[:, j] = ap
                preds.append(pr)
                log(f'{salt} {tag} {sset}', json.dumps({'map5': r4(ap.mean()),
                                                        'sec': round(time.time() - t1)}))
            snames = [f'{tag}_{i}' for i in range(len(seedsets))]
            sblk = oracle_block(B, fams, snames)
            Pm = np.column_stack(preds)[cand]
            maxabs = float(np.abs(Pm - Pm[:, [0]]).max())
            Tsr = [within_pair_negrank(p, sl, cand) for p in preds]
            bap, _, _ = score_of(ds, combine(Tsr, [1.0] * len(Tsr), cand), truths, pairs)
            real_gain = ph4['oracle_and_nulls'][salt]['oracle_pair_gain']
            real_primary = res['phase3_blends'][salt]['PRIMARY_all_six_equal_weight_rank_average'][
                'delta_vs_best_single']
            ph4['seed_families'][tag]['by_salt'][salt] = {
                'per_member_map5': {n: r4(x) for n, x in sblk['per_variant_mean'].items()},
                'spread': r4(max(sblk['per_variant_mean'].values())
                             - min(sblk['per_variant_mean'].values())),
                'max_abs_prediction_difference_between_members': maxabs,
                'DEGENERATE': bool(maxabs < 1e-12),
                'best_single': r4(sblk['best_single']),
                'oracle_pair': r4(sblk['oracle_pair']),
                'oracle_pair_gain': r4(sblk['oracle_pair_gain']),
                'mean_offdiag_correlation':
                    (None if maxabs < 1e-12 else
                     r4(offdiag_mean_min(np.corrcoef(B, rowvar=False))[0])),
                'equal_weight_rank_blend_map5': r4(float(bap.mean())),
                'equal_weight_rank_blend_delta_vs_its_best_single':
                    r4(float(bap.mean()) - sblk['best_single']),
                'REAL_family_oracle_pair_gain': real_gain,
                'REAL_minus_seedfamily_oracle_pair_gain': r4(real_gain - sblk['oracle_pair_gain']),
                'REAL_primary_blend_delta': real_primary,
                'REAL_primary_blend_delta_minus_seedfamily_blend_delta':
                    r4(real_primary - (float(bap.mean()) - sblk['best_single']))}
            y = ph4['seed_families'][tag]['by_salt'][salt]
            print(f"    {tag:<18} {salt:<18} oracle gain {y['oracle_pair_gain']:+.4f}  "
                  f"real-seed {y['REAL_minus_seedfamily_oracle_pair_gain']:+.4f}   blend delta "
                  f"{y['equal_weight_rank_blend_delta_vs_its_best_single']:+.4f}  real-seed "
                  f"{y['REAL_primary_blend_delta_minus_seedfamily_blend_delta']:+.4f}"
                  f"{'   [DEGENERATE]' if y['DEGENERATE'] else ''}")
    res['phase4_nulls_and_seed_families'] = ph4

    # ---- cross-salt transfer: no synthetic null ----------------------------------------------
    print('\n  PHASE 4b.  cross-salt transfer (no synthetic null: the most trustworthy number here)')
    trans, wtrans = {}, {}
    for a in salts:
        for b in salts:
            if a == b:
                continue
            Aa = np.column_stack([AP[a][n] for n in FAMILY])
            Ab = np.column_stack([AP[b][n] for n in FAMILY])
            bb = oracle_block(Ab, fams, FAMILY)
            pick = Aa.argmax(axis=1)
            tp = float(Ab[np.arange(P), pick].mean())
            routed = np.empty(P)
            fam_pick = {}
            for f in FAMILIES:
                idx = np.where(fams == f)[0]
                j = int(np.argmax(Aa[idx].mean(axis=0)))
                fam_pick[f] = FAMILY[j]
                routed[idx] = Ab[idx, j]
            trans[f'{a}->{b}'] = {
                'transferred_pair_oracle': r4(tp),
                'gain_over_best_single_on_b': r4(tp - bb['best_single']),
                'own_pair_oracle_on_b': r4(bb['oracle_pair']),
                'fraction_of_own_oracle_gain_retained':
                    r4((tp - bb['best_single']) / (bb['oracle_pair'] - bb['best_single'] + 1e-12)),
                'transferred_family_router': r4(routed.mean()),
                'family_gain_over_best_single_on_b': r4(routed.mean() - bb['best_single']),
                'family_choice_from_a': fam_pick}
            row = {}
            for n in NEW:
                wa = res['phase3_blends'][a]['POSTHOC_weight_sweep_rank_blends']['per_learner'][n][
                    'POSTHOC_best_weight']
                onb = res['phase3_blends'][b]['POSTHOC_weight_sweep_rank_blends']['per_learner'][n][
                    'by_weight'][f'{wa:.1f}']
                row[n] = {'weight_chosen_on_a': wa, 'map5_on_b_at_that_weight': onb,
                          'delta_vs_best_single_on_b': r4(onb - bb['best_single']),
                          'equal_weight_map5_on_b': res['phase3_blends'][b][
                              'declared_set_of_twelve_equal_weight_blends'][f'rank_r382+{n}']['map5'],
                          'posthoc_best_on_b': res['phase3_blends'][b][
                              'POSTHOC_weight_sweep_rank_blends']['per_learner'][n]['POSTHOC_best_map5']}
            wtrans[f'{a}->{b}'] = row
    tpg = [v['gain_over_best_single_on_b'] for v in trans.values()]
    tfg = [v['family_gain_over_best_single_on_b'] for v in trans.values()]
    wdel = [x['delta_vs_best_single_on_b'] for r in wtrans.values() for x in r.values()]
    res['phase4_nulls_and_seed_families']['cross_salt_transfer'] = {
        'oracle_winner_transfer': trans,
        'blend_weight_transfer': {
            'what': 'the POST-HOC best rank-blend weight chosen on partition A, applied unchanged on '
                    'partition B, against B best single model. This is the honest price of the '
                    'weight sweep.',
            'by_direction': wtrans,
            'delta_min': r4(min(wdel)), 'delta_mean': r4(float(np.mean(wdel))),
            'delta_max': r4(max(wdel))},
        'summary': {'pair_oracle_transfer_gain_min': r4(min(tpg)),
                    'pair_oracle_transfer_gain_mean': r4(float(np.mean(tpg))),
                    'family_transfer_gain_min': r4(min(tfg)),
                    'family_transfer_gain_mean': r4(float(np.mean(tfg))),
                    'reading': 'needs no synthetic null: if the winner pattern or the blend weight '
                               'were genuine they would transfer.'}}
    print(f"    oracle winner transfer gain: min {min(tpg):+.4f} mean {np.mean(tpg):+.4f}   "
          f"blend-weight transfer delta: min {min(wdel):+.4f} mean {np.mean(wdel):+.4f}")

    # ---- cross-salt structure: what is identical by construction ------------------------------
    s1_identical = all(np.array_equal(AP[salts[0]]['s1_pure'], AP[s]['s1_pure']) for s in salts)
    res['cross_salt_structure'] = {
        'why': 'resplit() re-hashes the STAGE-2 fold assignment ONLY; oof and oof_rank are stored '
               'columns. Read step 90 cross_salt_consistency/why: the same warning applies here.',
        'IDENTICAL_BY_CONSTRUCTION_across_all_three_salts': [
            'the 162 pairs, their behaviour families, their tables and the truth',
            'the top-20 candidate set of every pair (oof_rank is a stored column)',
            'the per-pair AP vector of s1_pure (no stage 2 at all) -- asserted below',
            'the feature matrix itself'],
        's1_pure_ap_vector_identical_across_salts': bool(s1_identical),
        's1_pure_map5_each_salt': {s: r4(float(AP[s]['s1_pure'].mean())) for s in salts},
        'GENUINELY_VARYING_across_salts': [
            'every stage-2 fit and therefore every learner MAP@5 and per-pair AP vector',
            'every correlation between two fitted arms',
            'every blend MAP@5 and every blend delta',
            'every oracle ceiling and every seed-family number',
            'the best-single-model identity'],
        'BUT': 'the three salts are three RE-PARTITIONS OF THE SAME 162 PAIRS, not three independent '
               'samples. The min over salts is a robustness check against fold-assignment luck. '
               'Nothing in this file is three independent replications, and no p value here is '
               'combined across salts.'}

    # ---- minima and verdict --------------------------------------------------------------------
    mins = {
        'primary_blend_delta_by_salt':
            [res['phase3_blends'][s]['PRIMARY_all_six_equal_weight_rank_average']['delta_vs_best_single']
             for s in salts],
        'best_of_twelve_delta_by_salt':
            [res['phase3_blends'][s]['best_of_the_declared_twelve']['delta_vs_best_single'] for s in salts],
        'POSTHOC_sweep_delta_by_salt':
            [res['phase3_blends'][s]['POSTHOC_weight_sweep_rank_blends'][
                'best_over_learners_delta_vs_best_single'] for s in salts],
        'oracle_pair_gain_by_salt': [ph4['oracle_and_nulls'][s]['oracle_pair_gain'] for s in salts],
        'oracle_pair_real_minus_residperm_by_salt':
            [ph4['oracle_and_nulls'][s]['null_resid_perm_global']['real_minus_null_oracle_pair']
             for s in salts],
        'oracle_family_real_minus_residperm_by_salt':
            [ph4['oracle_and_nulls'][s]['null_resid_perm_global']['real_minus_null_oracle_family']
             for s in salts],
        'real_minus_MATCHED_seedfamily_oracle_by_salt':
            [ph4['seed_families']['MATCHED_mlp_bag3']['by_salt'][s][
                'REAL_minus_seedfamily_oracle_pair_gain'] for s in salts],
        'real_minus_MATCHED_seedfamily_blend_by_salt':
            [ph4['seed_families']['MATCHED_mlp_bag3']['by_salt'][s][
                'REAL_primary_blend_delta_minus_seedfamily_blend_delta'] for s in salts],
        'mean_offdiag_six_family_within_salt_by_salt':
            [ph2['by_salt'][s]['mean_offdiag_six_member_family'] for s in salts],
        'corr_vs_r382_min_over_new_learners_by_salt':
            [min(ph2['by_salt'][s]['vs_r382'][n] for n in NEW) for s in salts]}
    for k in list(mins):
        mins['MIN_' + k.replace('_by_salt', '')] = min(mins[k])
    res['salt_minima'] = mins

    primary_min = mins['MIN_primary_blend_delta']
    twelve_min = mins['MIN_best_of_twelve_delta']
    blend_helps = bool(primary_min > 0)
    material = bool(primary_min >= MIN_DELTA)
    res['verdict'] = {
        'ALIVE': bool(blend_helps and material),
        'DECIDING_NUMBER': primary_min,
        'deciding_number_is': 'the MINIMUM across the three pool partitions of (PRIMARY all-six '
                              'equal-weight rank-average blend MAP@5 - best single model MAP@5). One '
                              'pre-registered estimator, no selection.',
        'materiality_bar': MIN_DELTA,
        'composite_at_the_deciding_number': round(LEVER * primary_min * TRANSFER, 6),
        'clears_the_bar': material,
        'blend_helps_at_all': blend_helps,
        'secondary_best_of_twelve_min': twelve_min,
        'secondary_best_of_twelve_clears_bar': bool(twelve_min >= MIN_DELTA),
        'secondary_note': 'a maximum over twelve declared estimators; it is a selection and is not '
                          'the deciding number',
        'POSTHOC_sweep_min': mins['MIN_POSTHOC_sweep_delta'],
        'POSTHOC_sweep_note': 'POST-HOC. The weight is chosen on the same numbers it is scored on. '
                              'Reported for completeness only and never promoted. Its honest price '
                              'is the cross-salt blend-weight transfer above.',
        'diversity_exists_oracle_minus_matched_null_min': mins['MIN_oracle_pair_real_minus_residperm'],
        'diversity_exists': bool(mins['MIN_oracle_pair_real_minus_residperm'] > 0),
        'beats_matched_stage2_seed_family_oracle': bool(mins['MIN_real_minus_MATCHED_seedfamily_oracle'] > 0),
        'beats_matched_stage2_seed_family_blend': bool(mins['MIN_real_minus_MATCHED_seedfamily_blend'] > 0),
        'FAILURE_PRESERVED_AS_A_FAILURE': not bool(blend_helps and material),
        'what_a_failure_means_here': 'a result below the 0.0714 bar is a FAILURE of the learner-class '
                                     'diversity lane on this surface and is recorded as one. No '
                                     'rescue, no post-hoc subgroup, no re-thresholding.'}

    # ---- POST-PREREGISTRATION DESCRIPTIVE ADDITION: evidence of benefit -------------------------
    # Declared in POST_PREREG_DESCRIPTIVE_ADDITIONS, not in GATE.  It cannot change the verdict above.
    eob_cfg = POST_PREREG_DESCRIPTIVE_ADDITIONS['evidence_of_benefit']
    tab_keys = sorted(set(tabs.tolist()))
    tix = np.array([tab_keys.index(t) for t in tabs.tolist()])
    n_tab = len(tab_keys)          # the tables that carry at least one audit pair: the resampling unit
    ns_t = np.bincount(tix, minlength=n_tab).astype(float)
    IX = np.random.default_rng(eob_cfg['seed']).integers(0, n_tab, (eob_cfg['B'], n_tab))
    MULT = np.stack([np.bincount(row, minlength=n_tab) for row in IX]).astype(float)  # ONE draw, reused
    den_b = MULT @ ns_t

    def r5(x):
        return round(float(x), 5)

    def r6(x):
        return round(float(x), 6)

    def boot_readout(delta_pairs):
        dsum = np.bincount(tix, weights=delta_pairs, minlength=n_tab)
        draws = (MULT @ dsum) / den_b              # pooled mean per-pair delta over the carried pairs
        pt, bm = float(delta_pairs.mean()), float(draws.mean())
        lo, hi = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
        out = {'point_delta': r5(pt), 'point_delta_positive': bool(pt > 0),
               'bootstrap_mean_delta': r5(bm), 'ci95_percentile': [r5(lo), r5(hi)],
               'ci95_lower_bound_positive': bool(lo > 0),
               'frac_draws_gt_0': r4(float((draws > 0).mean())),
               'composite_0.2_x_delta': {'point': r6(LEVER * pt), 'bootstrap_mean': r6(LEVER * bm),
                                         'ci95': [r6(LEVER * lo), r6(LEVER * hi)]},
               'composite_0.2_x_delta_x_0.49': {'point': r6(LEVER * pt * TRANSFER),
                                                'bootstrap_mean': r6(LEVER * bm * TRANSFER),
                                                'ci95': [r6(LEVER * lo * TRANSFER),
                                                         r6(LEVER * hi * TRANSFER)]}}
        return out, pt, lo

    def fam_delta(delta_pairs):
        return {f: {'n_pairs': int((fams == f).sum()), 'mean_delta': r5(delta_pairs[fams == f].mean())}
                for f in FAMILIES}

    eob = {'LABEL': 'POST-PREREGISTRATION DESCRIPTIVE ADDITION: '
                    + POST_PREREG_DESCRIPTIVE_ADDITIONS['status'],
           'declared_in': 'POST_PREREG_DESCRIPTIVE_ADDITIONS, a constant separate from GATE and absent '
                          'from the pre-registration file',
           'config': eob_cfg,
           'estimator': 'table-clustered paired bootstrap: the audit tables are resampled with '
                        'replacement and carry all their pairs; each draw is the pooled mean per-pair '
                        'delta over the carried pairs (the ratio estimator of step 5 pool_bootstrap, '
                        'with this block own seed and B). ONE set of B table draws is made and reused '
                        'for every salt, for the salt mean, and for both estimators.',
           'best_single_model_choice': eob_cfg['best_single_model'],
           'n_tables': n_tab, 'n_tables_in_population': res['population']['tables'],
           'resampling_unit': 'every table that carries at least one audit pair, with all of its pairs',
           'n_pairs': P, 'n_salts': len(salts),
           'what_this_is_not': 'not the verdict, not a gate, not an admission rule. The pre-registered '
                               'verdict and its ALIVE/DEAD label stay primary.'}
    for tag, key, what in (
            ('PRIMARY_all_six_equal_weight_rank_average', 'primary',
             'the single pre-registered estimator (equal-weight rank-average of all six family members) '
             'minus the in-sample best single model, per pair'),
            ('SECONDARY_best_of_twelve_SELECTION_OVER_TWELVE', 'best12',
             'the best of the twelve declared equal-weight blends on each salt, its identity chosen '
             'in-sample on the point estimate and held fixed across draws, minus the same fixed best '
             'single model. A SELECTION OVER TWELVE: its interval is optimistic.')):
        p3key = ('PRIMARY_all_six_equal_weight_rank_average' if key == 'primary'
                 else 'best_of_the_declared_twelve')
        per_salt, dlist, n_pos = {}, [], 0
        for s in salts:
            dp = BLEND_AP[s][key] - BLEND_AP[s]['best_single']
            rd, pt, _ = boot_readout(dp)
            ref_pt = BLEND_AP[s][key + '_mean'] - BLEND_AP[s]['best_single_mean']
            assert abs(pt - ref_pt) < 1e-12, \
                f'evidence-of-benefit point delta disagrees with phase 3 on {s} {key}'
            rd.update({'blend': 'rank_all_six' if key == 'primary' else BLEND_AP[s]['best12_key'],
                       'best_single_model_held_fixed': BLEND_AP[s]['best_single_name'],
                       'agrees_with_phase3_point_delta': True,
                       'phase3_point_delta_rounded': res['phase3_blends'][s][p3key]['delta_vs_best_single'],
                       'by_family_mean_delta_DESCRIPTIVE': fam_delta(dp)})
            per_salt[s] = rd
            dlist.append(dp)
            n_pos += int(pt > 0)
        dbar = np.mean(np.stack(dlist), axis=0)
        sm, _, sm_lo = boot_readout(dbar)
        sm['by_family_mean_delta_DESCRIPTIVE'] = fam_delta(dbar)
        sm['how'] = ('the per-pair delta averaged over the salts, then the SAME table draws applied, '
                     'so every salt is recomputed on each shared draw')
        supported = bool(sm_lo > 0 and n_pos == len(salts))
        eob[tag] = {'what': what, 'by_salt': per_salt, 'salt_mean': sm,
                    'direction_count_positive': n_pos, 'of_salts': len(salts),
                    'label': 'SUPPORTED_SMALL_LEAD' if supported else 'UNSUPPORTED',
                    'label_rule': eob_cfg['label_rule'], 'label_is_not_the_verdict': True}
        print(f"  EVIDENCE OF BENEFIT (descriptive) {tag.split('_')[0]:<9} salt-mean "
              f"{sm['point_delta']:+.5f} CI {sm['ci95_percentile']}  P(>0) {sm['frac_draws_gt_0']}  "
              f"positive in {n_pos}/{len(salts)}  -> {eob[tag]['label']}")
    res['evidence_of_benefit_DESCRIPTIVE_POST_PREREG'] = eob
    res['step88_coupling_tax_as_read_at_runtime'] = s88_block

    res['limits'] = [
        'All six family members share ONE frozen stage-1 score and one identical top-20 candidate '
        'set. This measures learner-class diversity of STAGE 2 only. Diversity between different '
        'STAGE-1 retrievals would change the candidate sets and is a different experiment (steps 85b '
        'and 88); it is not measured here.',
        '162 pairs, 43 of them coordinated_isolation. Every number here is a training-audit number '
        'on a surface where the audit isolation column has failed to replicate on held twice '
        '(OBS 18.9 / 18.10). Nothing here has been confirmed on a held fold.',
        'The best single model is chosen by IN-SAMPLE argmax over the six members, which flatters '
        'the baseline and therefore understates every blend delta. That is the conservative '
        'direction and is deliberate.',
        'The three salts re-partition the SAME 162 pairs. The min over salts is not three '
        'independent replications; see cross_salt_structure.',
        'The non-tree learners were given the identical 453-column matrix that r382 uses, which was '
        'selected (by earlier steps) for a GBDT. A linear or local learner might do better on a '
        'different column set; no column selection was performed for them, by design, because the '
        'question asked is about the LEARNER holding the columns fixed.',
        'Hyperparameters for the five new learners were fixed in the pre-registration and never '
        'tuned. They are reasonable defaults, not optimised ones, so each new learner is a lower '
        'bound on what its class could do here.',
        'The weight sweep is a post-hoc selection and its number is reported only beside the '
        'cross-salt transfer of the selected weight, which is the honest price of that selection.',
        'AP@5 is bounded and lumpy (six attainable overlap levels on five slots), so the '
        'residual-permutation null produces surrogate values slightly outside the AP@5 support; the '
        'statistic compared is the oracle GAIN, whose per-arm means are preserved exactly.']

    res['protected_hashes_exit'] = check_protected()
    res['protected_hashes_identical'] = res['protected_hashes_exit'] == entry

    # ---- Phase 5: prose self-check --------------------------------------------------------------
    checks = [
        ('verdict ALIVE', 'verdict/ALIVE'),
        ('deciding number (min over salts, PRIMARY blend delta)', 'verdict/DECIDING_NUMBER'),
        ('materiality bar', 'verdict/materiality_bar'),
        ('composite at the deciding number', 'verdict/composite_at_the_deciding_number'),
        ('clears the bar', 'verdict/clears_the_bar'),
        ('blend helps at all', 'verdict/blend_helps_at_all'),
        ('secondary best-of-twelve min', 'verdict/secondary_best_of_twelve_min'),
        ('POSTHOC sweep min', 'verdict/POSTHOC_sweep_min'),
        ('diversity exists (oracle - matched null, min)',
         'verdict/diversity_exists_oracle_minus_matched_null_min'),
        ('beats matched stage-2 seed family (oracle)', 'verdict/beats_matched_stage2_seed_family_oracle'),
        ('beats matched stage-2 seed family (blend)', 'verdict/beats_matched_stage2_seed_family_blend'),
        ('phase 0 gate passed', 'phase0_reproduction_gate/PASS'),
        ('phase 0 hybrid', 'phase0_reproduction_gate/hybrid_r382DTSP'),
        ('phase 0 ref', 'phase0_reproduction_gate/ref'),
        ('phase 0 r382', 'phase0_reproduction_gate/r382'),
        ('held folds touched', 'held_folds_touched'),
        ('held/audit intersection empty', 'held_fold_overlap_check/intersection_is_empty'),
        ('protected hashes identical', 'protected_hashes_identical'),
        ('pre-registration written before any fit', 'prediction_written_before_any_fit'),
        ('phase 2 is not a gate', 'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/IS_THIS_A_GATE'),
        ('candidate rows', 'population/candidate_rows'),
        ('feature columns', 'population/n_feature_columns'),
        ('NaN cells in candidate matrix', 'feature_matrix_missingness/nan_cells'),
        ('s1_pure AP identical across salts',
         'cross_salt_structure/s1_pure_ap_vector_identical_across_salts'),
        ('salt-averaged mean off-diag, all twelve',
         'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/salt_averaged/mean_offdiag_all_twelve'),
        ('salt-averaged mean off-diag, step89 seven',
         'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/salt_averaged/mean_offdiag_step89_seven'),
        ('averaging inflation, all twelve',
         'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/inflation_from_averaging/delta_all_twelve'),
        ('cross-salt oracle transfer gain min',
         'phase4_nulls_and_seed_families/cross_salt_transfer/summary/pair_oracle_transfer_gain_min'),
        ('cross-salt blend-weight transfer delta min',
         'phase4_nulls_and_seed_families/cross_salt_transfer/blend_weight_transfer/delta_min'),
        ('cross-salt blend-weight transfer delta mean',
         'phase4_nulls_and_seed_families/cross_salt_transfer/blend_weight_transfer/delta_mean'),
        ('MIN oracle gain over salts', 'salt_minima/MIN_oracle_pair_gain'),
        ('MIN within-salt mean off-diag of the six-member family',
         'salt_minima/MIN_mean_offdiag_six_family_within_salt'),
        ('MIN corr vs r382 over new learners', 'salt_minima/MIN_corr_vs_r382_min_over_new_learners'),
    ]
    for s in salts:
        checks += [
            (f'[{s}] step89-7 within-salt mean off-diag',
             f'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/by_salt/{s}/mean_offdiag_step89_seven'),
            (f'[{s}] step89-7 reproduces step 89 stored value',
             f'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/by_salt/{s}/reproduces_step89_mean_offdiag'),
            (f'[{s}] all-12 within-salt mean off-diag',
             f'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/by_salt/{s}/mean_offdiag_all_twelve'),
            (f'[{s}] best single model map5', f'phase3_blends/{s}/best_single_map5'),
            (f'[{s}] PRIMARY blend delta',
             f'phase3_blends/{s}/PRIMARY_all_six_equal_weight_rank_average/delta_vs_best_single'),
            (f'[{s}] best-of-twelve delta',
             f'phase3_blends/{s}/best_of_the_declared_twelve/delta_vs_best_single'),
            (f'[{s}] oracle pair gain', f'phase4_nulls_and_seed_families/oracle_and_nulls/{s}/oracle_pair_gain'),
            (f'[{s}] oracle minus matched null',
             f'phase4_nulls_and_seed_families/oracle_and_nulls/{s}/null_resid_perm_global/'
             f'real_minus_null_oracle_pair'),
            (f'[{s}] matched MLP seed-family oracle gain',
             f'phase4_nulls_and_seed_families/seed_families/MATCHED_mlp_bag3/by_salt/{s}/oracle_pair_gain'),
            (f'[{s}] LGBM stage-2 seed family degenerate',
             f'phase4_nulls_and_seed_families/seed_families/lgbm_r382_single/by_salt/{s}/DEGENERATE'),
            (f'[{s}] LGBM stage-2 seed family max abs pred diff',
             f'phase4_nulls_and_seed_families/seed_families/lgbm_r382_single/by_salt/{s}/'
             f'max_abs_prediction_difference_between_members'),
            (f'[{s}] known-worse arm struct_nos1 fails',
             f'phase3_blends/{s}/known_worse_arm_check/arm_a_single_struct_nos1/fails_the_gate'),
            (f'[{s}] known-worse blend with s1_pure fails',
             f'phase3_blends/{s}/known_worse_arm_check/arm_b_rank_blend_r382_with_s1_pure/fails_the_gate'),
        ]
        for n in FAMILY:
            checks.append((f'[{s}] {n} MAP@5', f'phase1_learners/per_arm_by_salt/{s}/{n}/map5'))
        for n in NEW:
            checks.append((f'[{s}] corr({n}, r382) within salt',
                           f'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/by_salt/{s}/vs_r382/{n}'))
    for n in NEW:
        checks.append((f'corr({n}, r382) salt-averaged',
                       f'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/salt_averaged/vs_r382/{n}'))

    # ---- entries for the readouts added after smoke run 1 ----
    QV = 'phase2_correlation_DESCRIPTIVE_NOT_A_GATE/quality_vs_decorrelation'
    PH2 = 'phase2_correlation_DESCRIPTIVE_NOT_A_GATE'
    EOB = 'evidence_of_benefit_DESCRIPTIVE_POST_PREREG'
    SF = 'phase4_nulls_and_seed_families/seed_families'
    CST = 'phase4_nulls_and_seed_families/cross_salt_transfer'
    S88 = 'step88_coupling_tax_as_read_at_runtime/verdict_coupling_tax_refit_minus_frozen'
    checks += [
        ('pre-registration sha256', 'prediction_file_sha256'),
        ('quality: Pearson r over the eleven non-r382 arms',
         f'{QV}/pearson_r_between_arm_map5_and_its_corr_with_r382'),
        ('quality: Pearson r over the six reference arms', f'{QV}/reference_line/pearson_r_over_reference_arms'),
        ('quality: reference-line slope', f'{QV}/reference_line/slope'),
        ('quality: reference-line intercept', f'{QV}/reference_line/intercept'),
        ('quality: reference MAP@5 range', f'{QV}/reference_line/map5_range'),
        ('step 88 refit-minus-frozen mean', f'{S88}/mean_over_9_cells'),
        ('step 88 refit-minus-frozen min', f'{S88}/min'),
        ('step 88 refit-minus-frozen max', f'{S88}/max'),
        ('step 88 refit-minus-frozen positive in', f'{S88}/positive_in'),
        ('step 88 refit-minus-frozen of cells', f'{S88}/of_cells'),
        ('step 88 artefact sha256', 'step88_coupling_tax_as_read_at_runtime/sha256'),
        ('MIN real minus MATCHED seed family (oracle)', 'salt_minima/MIN_real_minus_MATCHED_seedfamily_oracle'),
        ('MIN real minus MATCHED seed family (blend)', 'salt_minima/MIN_real_minus_MATCHED_seedfamily_blend'),
        ('MIN oracle minus matched null', 'salt_minima/MIN_oracle_pair_real_minus_residperm'),
        ('cross-salt oracle transfer gain mean', f'{CST}/summary/pair_oracle_transfer_gain_mean'),
        ('cross-salt family transfer gain min', f'{CST}/summary/family_transfer_gain_min'),
        ('cross-salt blend-weight transfer delta max', f'{CST}/blend_weight_transfer/delta_max'),
        ('averaging inflation, step89 seven', f'{PH2}/inflation_from_averaging/delta_step89_seven'),
        ('within-salt mean of step89-7 mean off-diag',
         f'{PH2}/inflation_from_averaging/mean_offdiag_step89_seven_within_salt_mean'),
        ('within-salt mean of all-12 mean off-diag',
         f'{PH2}/inflation_from_averaging/mean_offdiag_all_twelve_within_salt_mean'),
        ('salt-averaged mean off-diag, six-member family', f'{PH2}/salt_averaged/mean_offdiag_six_member_family'),
        ('evidence: number of tables', f'{EOB}/n_tables'),
        ('evidence: tables in population', f'{EOB}/n_tables_in_population'),
    ]
    for n in NEW:
        checks += [
            (f'quality: {n} residual vs reference line',
             f'{QV}/new_learners_vs_reference_line/{n}/residual_observed_minus_line'),
            (f'quality: {n} below reference line', f'{QV}/new_learners_vs_reference_line/{n}/below_the_reference_line'),
            (f'quality: {n} MAP@5 inside reference range',
             f'{QV}/new_learners_vs_reference_line/{n}/map5_inside_reference_range'),
            (f'quality: {n} line value at its MAP@5',
             f'{QV}/new_learners_vs_reference_line/{n}/reference_line_corr_at_this_map5')]
    for n in others:
        checks += [(f'quality: {n} mean MAP@5 over salts', f'{QV}/per_arm/{n}/map5_mean_over_salts'),
                   (f'quality: {n} within-salt corr with r382, mean over salts',
                    f'{QV}/per_arm/{n}/corr_vs_r382_within_salt_mean')]
    for tag in ('PRIMARY_all_six_equal_weight_rank_average', 'SECONDARY_best_of_twelve_SELECTION_OVER_TWELVE'):
        base = f'{EOB}/{tag}'
        checks += [
            (f'evidence {tag}: label', f'{base}/label'),
            (f'evidence {tag}: direction count positive', f'{base}/direction_count_positive'),
            (f'evidence {tag}: of salts', f'{base}/of_salts'),
            (f'evidence {tag}: salt-mean point delta', f'{base}/salt_mean/point_delta'),
            (f'evidence {tag}: salt-mean bootstrap mean', f'{base}/salt_mean/bootstrap_mean_delta'),
            (f'evidence {tag}: salt-mean CI95', f'{base}/salt_mean/ci95_percentile'),
            (f'evidence {tag}: salt-mean frac draws > 0', f'{base}/salt_mean/frac_draws_gt_0'),
            (f'evidence {tag}: salt-mean composite 0.2x point', f'{base}/salt_mean/composite_0.2_x_delta/point'),
            (f'evidence {tag}: salt-mean composite 0.2x CI95', f'{base}/salt_mean/composite_0.2_x_delta/ci95'),
            (f'evidence {tag}: salt-mean composite 0.2x0.49 point',
             f'{base}/salt_mean/composite_0.2_x_delta_x_0.49/point'),
            (f'evidence {tag}: salt-mean composite 0.2x0.49 CI95',
             f'{base}/salt_mean/composite_0.2_x_delta_x_0.49/ci95')]
        for f in FAMILIES:
            checks.append((f'evidence {tag}: salt-mean {f} delta',
                           f'{base}/salt_mean/by_family_mean_delta_DESCRIPTIVE/{f}/mean_delta'))
        for s in salts:
            checks += [
                (f'evidence {tag} [{s}]: point delta', f'{base}/by_salt/{s}/point_delta'),
                (f'evidence {tag} [{s}]: bootstrap mean', f'{base}/by_salt/{s}/bootstrap_mean_delta'),
                (f'evidence {tag} [{s}]: CI95', f'{base}/by_salt/{s}/ci95_percentile'),
                (f'evidence {tag} [{s}]: frac draws > 0', f'{base}/by_salt/{s}/frac_draws_gt_0'),
                (f'evidence {tag} [{s}]: blend', f'{base}/by_salt/{s}/blend'),
                (f'evidence {tag} [{s}]: fixed best single', f'{base}/by_salt/{s}/best_single_model_held_fixed'),
                (f'evidence {tag} [{s}]: composite 0.2x point', f'{base}/by_salt/{s}/composite_0.2_x_delta/point'),
                (f'evidence {tag} [{s}]: composite 0.2x0.49 point',
                 f'{base}/by_salt/{s}/composite_0.2_x_delta_x_0.49/point')]
            for f in FAMILIES:
                checks.append((f'evidence {tag} [{s}]: {f} delta',
                               f'{base}/by_salt/{s}/by_family_mean_delta_DESCRIPTIVE/{f}/mean_delta'))
    for s in salts:
        checks += [
            (f'[{s}] best single model name', f'phase3_blends/{s}/best_single_model'),
            (f'[{s}] PRIMARY blend MAP@5', f'phase3_blends/{s}/PRIMARY_all_six_equal_weight_rank_average/map5'),
            (f'[{s}] best-of-twelve identity', f'phase3_blends/{s}/best_of_the_declared_twelve/which'),
            (f'[{s}] best-of-twelve MAP@5', f'phase3_blends/{s}/best_of_the_declared_twelve/map5'),
            (f'[{s}] POSTHOC sweep delta',
             f'phase3_blends/{s}/POSTHOC_weight_sweep_rank_blends/best_over_learners_delta_vs_best_single'),
            (f'[{s}] POSTHOC learners whose best weight is zero',
             f'phase3_blends/{s}/POSTHOC_weight_sweep_rank_blends/n_learners_whose_posthoc_best_weight_is_zero'),
            (f'[{s}] blend identity check exact', f'phase3_blends/{s}/blend_machinery_identity_check/EXACT'),
            (f'[{s}] oracle permutation p (resid perm global)',
             f'phase4_nulls_and_seed_families/oracle_and_nulls/{s}/null_resid_perm_global/perm_p_oracle_pair'),
            (f'[{s}] null resid perm global mean oracle gain',
             f'phase4_nulls_and_seed_families/oracle_and_nulls/{s}/null_resid_perm_global/null_oracle_pair_gain_mean'),
            (f'[{s}] MATCHED seed family real minus seed (oracle)',
             f'{SF}/MATCHED_mlp_bag3/by_salt/{s}/REAL_minus_seedfamily_oracle_pair_gain'),
            (f'[{s}] MATCHED seed family real minus seed (blend)',
             f'{SF}/MATCHED_mlp_bag3/by_salt/{s}/REAL_primary_blend_delta_minus_seedfamily_blend_delta'),
            (f'[{s}] MATCHED seed family own blend delta',
             f'{SF}/MATCHED_mlp_bag3/by_salt/{s}/equal_weight_rank_blend_delta_vs_its_best_single'),
            (f'[{s}] MATCHED seed family mean off-diag corr', f'{SF}/MATCHED_mlp_bag3/by_salt/{s}/mean_offdiag_correlation'),
            (f'[{s}] MATCHED seed family max abs pred diff',
             f'{SF}/MATCHED_mlp_bag3/by_salt/{s}/max_abs_prediction_difference_between_members'),
            (f'[{s}] mlp_single seed family oracle gain', f'{SF}/mlp_single/by_salt/{s}/oracle_pair_gain'),
            (f'[{s}] mlp_single real minus seed (oracle)',
             f'{SF}/mlp_single/by_salt/{s}/REAL_minus_seedfamily_oracle_pair_gain'),
            (f'[{s}] lgbm seed family oracle gain', f'{SF}/lgbm_r382_single/by_salt/{s}/oracle_pair_gain'),
            (f'[{s}] six-member family within-salt mean off-diag', f'{PH2}/by_salt/{s}/mean_offdiag_six_member_family')]
        for n in ('s1_pure', 'ref', 'struct_nos1', 'r382_nostruct', 'rank_ref', 'rank_r382'):
            checks.append((f'[{s}] corr({n}, r382) within salt', f'{PH2}/by_salt/{s}/vs_r382/{n}'))
    for n in ('s1_pure', 'ref', 'struct_nos1', 'r382_nostruct', 'rank_ref', 'rank_r382'):
        checks.append((f'corr({n}, r382) salt-averaged', f'{PH2}/salt_averaged/vs_r382/{n}'))

    checked, ok = [], True
    for label, path in checks:
        try:
            v = dig(res, path)
        except (KeyError, IndexError, TypeError):
            v, ok = '<<PATH NOT FOUND>>', False
        checked.append({'label': label, 'path': path, 'value': v})
    res['prose_self_check'] = {'checked': checked, 'n_entries': len(checked), 'all_matched': ok}

    res['provenance'] = [
        'FULL DISCLOSURE.  Before this production run a REDUCED SMOKE RUN of the identical code path '
        '(2 salts, NPERM=25, weight grid {0, .5, 1}, 3 seed-family members) was executed with OUTDIR '
        'redirected to the session scratchpad, to validate the code before spending the real budget. '
        'Its output was written to the scratchpad and is not part of this artefact. The GATE constant '
        '(the full pre-registration) was authored BEFORE that smoke run, and '
        'prediction-learner-diversity.json was written to its real path BEFORE any model was fitted '
        'anywhere, including in the smoke run. On this production run the script re-derives the '
        'payload and ASSERTS it equals the file on disk as parsed JSON in every key but the utc '
        'stamp, and separately ASSERTS the file sha256 is unchanged, so the file provably was not '
        'edited after the fact. (This entry describes smoke run 1; smoke run 2 has its own entry.)',
        'SCOPE ADDITIONS received from the coordinator BEFORE the production run and AFTER the GATE '
        'constant was authored. None of them changes a threshold, a comparison or an admission rule; '
        'all three only widen what is reported. (1) The per-pair AP correlation must NOT be used as a '
        'stop rule -- per-pair AP is a lossy summary that discards the hand-level ranking errors a '
        'blend operates on, so two models with AP correlation 1 can still blend profitably. Phase 3 '
        'therefore ran in full regardless of Phase 2, and the gate key '
        'PHASE_2_IS_DESCRIPTIVE_NOT_A_GATE records that. (2) Correlations must be reported BOTH '
        'within-salt (step 89 convention) and on salt-averaged per-pair AP vectors (the convention '
        'behind the other agent TabPFN-3 vs LGBM-hybrid .8543), because averaging first mechanically '
        'inflates the correlation and the two are not comparable; both are now in '
        'phase2/by_salt and phase2/salt_averaged, with the inflation itself measured. (3) A MATCHED '
        'seed family was requested: step 89 null_seed_family varied STAGE 1, and so changed the top-20 '
        'candidate sets, while its real variants varied only STAGE 2 on one fixed candidate set. All '
        'three seed families here vary only the STAGE-2 seed on the one frozen stage 1, so stage 1 and '
        'the top-20 candidate set are identical across members, exactly as across the six real learner '
        'arms. (This sentence was reworded after smoke run 1 to drop a "tax-free" framing; see the '
        'step-88 entry below.)',
        f'INTERRUPTION.  Smoke run 1 finished at 01:08 local time, UTC+3 (scratchpad/step92smoke, not '
        f'modified since). At about 01:09 local the session died on an API rate limit (HTTP 429) part '
        f'way through an edit to this file. When the session was resumed the coordinator reported that '
        f'no step-92 process was running, that experiments/learner_diversity_C_20260913/ held only '
        f'prediction-learner-diversity.json and no production artefact, that the script passed '
        f'py_compile, and that the payload main() builds from the current GATE equals that file in '
        f'every key but utc. Claude re-checked the last two points independently before any further '
        f'run. The pre-registration was not rewritten: it is still the file written at utc '
        f'{prod_prereg_utc}, sha256 {prod_prereg_sha}, and the production run asserts that sha256 '
        f'before any fit. The edit in progress at the interruption (the quality_vs_decorrelation block) '
        f'referenced an undefined name and would have raised NameError; it never executed and was '
        f'rewritten before any run (change list, item 3).',
        f'COUPLING-TAX WORDING IN GATE IS SUPERSEDED BY STEP 88.  GATE keys coupling_tax and '
        f'nulls/MATCHED_seed_family_mlp_bag3 call the matched seed family "TAX-FREE" and say step 89 '
        f'null_seed_family "paid the 19.16 coupling tax". That wording follows OBS 19.16 and predates '
        f'step 88 being taken into this study: it does not reflect the step 88 refit-versus-frozen '
        f'result. (Timestamps, for the record: the pre-registration utc is {prod_prereg_utc}; the step '
        f'88 result file mtime is {s88_block["file_mtime_utc"]}.) It is frozen in the pre-registration '
        f'and left exactly as written. Step 88 ({s88_block["artefact"]}, sha256 {s88_block["sha256"]}, '
        f'read at runtime into step88_coupling_tax_as_read_at_runtime) found NO coupling tax: refit '
        f'minus frozen averages {s88_tax["mean_over_9_cells"]} over {s88_tax["of_cells"]} cells (min '
        f'{s88_tax["min"]}, max {s88_tax["max"]}) and is positive in only {s88_tax["positive_in"]} of '
        f'{s88_tax["of_cells"]}; its gate/frozen_comparator_definition, recorded verbatim in that '
        f'block, states that the step 85b published "frozen" delta was a refit number. The matched '
        f'seed family is still the right control, for a different reason: it keeps stage 1, and '
        f'therefore the top-20 candidate set, identical across members, exactly like the six real '
        f'learner arms, whereas step 89 null_seed_family changed the candidate sets. No prose written '
        f'for this artefact outside the quoted GATE text calls the matched family tax-free.',
        'POST-PREREGISTRATION DESCRIPTIVE ADDITION: evidence_of_benefit_DESCRIPTIVE_POST_PREREG was '
        'added after the smoke run; requested by the coordinator before the coordinator read any smoke '
        'number; cannot change the verdict. It is declared in the separate constant '
        'POST_PREREG_DESCRIPTIVE_ADDITIONS, which is not part of GATE and not in the pre-registration '
        'file, so the pre-registration assertion is unaffected. Its SUPPORTED_SMALL_LEAD / UNSUPPORTED '
        'label is kept apart from ALIVE/DEAD, and the pre-registered verdict stays primary.',
        'COMPLETE LIST OF CODE CHANGES AFTER SMOKE RUN 1 (none changes GATE, the pre-registration '
        'payload, a threshold, a comparison, the verdict estimator or an admission rule): '
        '(1) Phase 3 sweep bookkeeping: n_weights_tied_at_best, '
        'n_learners_whose_posthoc_best_weight_is_zero, n_new_learners and reading_if_all_zero were '
        'added; the tie-break itself (first maximum in grid order, i.e. the smallest weight) is '
        'unchanged. '
        '(2) blend_machinery_identity_check: the w=0 rank blend must reproduce plain r382 exactly. '
        '(3) Phase 2 quality_vs_decorrelation block, motivated by smoke-run-1 numbers (the weakest new '
        'learners had the lowest correlations with r382). Its first draft referenced an undefined name '
        '(NEW_FOR_Q) and used a clamped np.interp; before any run it was rewritten as an OLS reference '
        'line through the six non-r382 step-89 arms with per-learner residuals and an in-range flag, '
        'and the NEW list was moved ahead of it (it had been defined at the start of Phase 3). '
        '(4) BLEND_AP keeps the per-pair AP vectors of the PRIMARY blend, the best-of-twelve blend and '
        'the fixed best single model for the evidence readout. '
        '(5) The POST_PREREG_DESCRIPTIVE_ADDITIONS, EXPECTED_PREREG_SHA, PROD_PREREG and STEP88_ARTEFACT '
        'constants '
        'and the evidence_of_benefit_DESCRIPTIVE_POST_PREREG block (coordinator request). '
        '(6) A production-only fail-fast assertion that the pre-registration sha256 equals the 01:01 '
        'value. '
        '(7) Step 88 verdict/coupling_tax_refit_minus_frozen and gate/frozen_comparator_definition are '
        'read at the START of main(), with the file sha256 and mtime, into '
        'step88_coupling_tax_as_read_at_runtime (read-only). '
        '(8) Non-GATE prose reworded away from "tax-free": the Phase 4 console line; the seed-family '
        'keys pays_the_coupling_tax / why_tax_free renamed to '
        'stage1_and_top20_candidate_set_identical_across_members / why_this_is_the_matched_control; the '
        'first limits entry; scope item (3) above. '
        '(9) Provenance: the interruption, step-88, addition, change-list and smoke-run-2 entries were '
        'added; a dangling sentence promising a fix list "in the next entry" was replaced by this '
        'list; and the first entry, which called the pre-registration check "byte-equal", now says '
        'what is checked (equality as parsed JSON, plus the sha256 assertion of item 6). '
        '(10) prose_self_check: entries added for every new readout, and every entry is now verified '
        'by re-reading the written artefact from disk (written, re-read, checked, written again). '
        '(11) The final console summary prints the evidence-of-benefit labels and the round-trip result.',
        'SMOKE RUN 2.  Because code changed after smoke run 1, a second REDUCED smoke run of the '
        'identical code path (same reductions: 2 salts, NPERM=25, weight grid {0, .5, 1}, 3 seed-family '
        'members) was executed into a NEW scratchpad directory, scratchpad/step92smoke2, before this '
        'production run, so that the long run could not crash at the end. The smoke run 1 directory '
        '(scratchpad/step92smoke) was neither reused nor modified. Neither smoke output is part of this '
        'artefact. No code was changed between smoke run 2 and this production run.',
        'No code change of any kind was made after seeing any production number.']

    res['seconds'] = round(time.time() - t0)
    out = OUTDIR / 'audit-C-learner-diversity.json'
    write_json(out, res)
    # round trip: every self-check entry must equal the value at its path IN THE FILE AS WRITTEN
    disk = json.loads(out.read_text(encoding='utf-8'))
    mism = []
    for e in res['prose_self_check']['checked']:
        try:
            dv = dig(disk, e['path'])
        except (KeyError, IndexError, TypeError):
            dv = '<<PATH NOT FOUND ON DISK>>'
        if json.dumps(dv, sort_keys=True) != json.dumps(e['value'], sort_keys=True):
            mism.append(e['path'])
    res['prose_self_check']['verified_against_the_artefact_re_read_from_disk'] = True
    res['prose_self_check']['round_trip_mismatches'] = mism
    res['prose_self_check']['all_matched'] = bool(ok and not mism)
    write_json(out, res)
    final = json.loads(out.read_text(encoding='utf-8'))
    assert final['prose_self_check']['all_matched'] == res['prose_self_check']['all_matched']
    assert final['verdict'] == disk['verdict'], 'the second write changed the verdict block'

    print('\n' + '=' * 100)
    print(f"  PHASE 2 (descriptive)  within-salt mean off-diag, six-member family: "
          f"{[ph2['by_salt'][s]['mean_offdiag_six_member_family'] for s in salts]}")
    print(f"  PHASE 2 (descriptive)  corr vs r382, new learners, min over salts: "
          f"{mins['MIN_corr_vs_r382_min_over_new_learners']:+.4f}")
    print(f"  PRIMARY BLEND DELTA    min across salts {primary_min:+.4f}   "
          f"composite {LEVER * primary_min * TRANSFER:+.6f}   bar {MIN_DELTA}")
    print(f"  BEST OF TWELVE         min across salts {twelve_min:+.4f}")
    print(f"  POSTHOC SWEEP          min across salts {mins['MIN_POSTHOC_sweep_delta']:+.4f}  (post-hoc)")
    print(f"  ORACLE - MATCHED NULL  min across salts "
          f"{mins['MIN_oracle_pair_real_minus_residperm']:+.4f}")
    print(f"  REAL - MATCHED SEED FAMILY (oracle)  min "
          f"{mins['MIN_real_minus_MATCHED_seedfamily_oracle']:+.4f}   (blend) min "
          f"{mins['MIN_real_minus_MATCHED_seedfamily_blend']:+.4f}")
    print(f"  VERDICT: {'ALIVE' if res['verdict']['ALIVE'] else 'DEAD'}   deciding number "
          f"{primary_min:+.4f}  (bar {MIN_DELTA})")
    ps = res['prose_self_check']
    ev = res['evidence_of_benefit_DESCRIPTIVE_POST_PREREG']
    print(f"  EVIDENCE LABELS (descriptive, NOT the verdict): PRIMARY "
          f"{ev['PRIMARY_all_six_equal_weight_rank_average']['label']}   best-of-twelve "
          f"{ev['SECONDARY_best_of_twelve_SELECTION_OVER_TWELVE']['label']}")
    print(f"  prose_self_check: {ps['n_entries']} entries, verified against the file re-read from disk, "
          f"all_matched={ps['all_matched']}  mismatches={len(ps['round_trip_mismatches'])}")
    print(f"  wrote {out}  ({res['seconds']} s)")
    return res


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--prereg-only':
        OUTDIR.mkdir(parents=True, exist_ok=True)
        p = OUTDIR / 'prediction-learner-diversity.json'
        assert not p.exists(), f'{p} already exists -- refusing to overwrite a pre-registration'
        write_json(p, {'author': AUTHOR, 'step': STEP, 'written_before_any_fit': True,
                       'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                       'gate': GATE, 'salts': list(SALTS), 'family': list(FAMILY),
                       'seed_family_n': SEED_FAMILY_N,
                       'mlp_seed_triples': [list(t) for t in MLP_SEED_TRIPLES],
                       'single_seeds': list(SINGLE_SEEDS), 'weight_grid': WEIGHTS,
                       'equal_weight': EQUAL_W, 'nperm': NPERM, 'perm_seed': PERM_SEED,
                       'hyperparameters': {'logreg_C': LOGREG_C,
                                           'logreg_C_sensitivity': LOGREG_C_SENSITIVITY,
                                           'mlp': {k: (list(v) if isinstance(v, tuple) else v)
                                                   for k, v in MLP_KW.items()},
                                           'trees': TREE_KW, 'knn': KNN_KW,
                                           'lightgbm_clf_reused_unchanged_from_step14': CLF},
                       'materiality': {'lever': LEVER, 'transfer_discount': TRANSFER,
                                       'public_resolution': RESOLUTION,
                                       'min_delta_required': MIN_DELTA},
                       'reused_without_refit': [],
                       'imported_unchanged_from_step89': ['resplit', 'score_of', 'fam_means', 'r4',
                                                          'oracle_block', 'null_resid_perm',
                                                          'null_full_perm', 'null_summary'],
                       'not_touched': ['OBSERVATIONS.md', 'held folds 0/1/2 truth',
                                       'Codex Resume/ and the original tree (read-only)',
                                       'the GPU (reserved for another agent)']})
        print(f'wrote pre-registration {p}  sha256 {sha(p)}')
    elif args and args[0] == '--smoke':
        main(smoke_dir=args[1])
    else:
        main()
