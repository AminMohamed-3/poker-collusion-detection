"""Step 87 [C].  The evidence confusion matrix: LOOK AT THE MISSES.
author: claude

THE HOLE THIS FILLS
The composite is .7*PairAP + .2*EvidenceMAP@5 + .1*BehaviourMAP.  The pair side has .03024 of
remaining headroom in total; the evidence side needs +.10 to +.15 MAP@5 to reach the user's target,
and the ladder of step 76 says retrieval is nearly solved and the loss is SELECTION:

    incumbent hybrid MAP@5 .6159, mean overlap 3.3333 of 5, oracle-within-top-20 4.642,
    select_oracle by candidate count K = .5823 / .8153 / .9481 / .9852 / .9951 at K = 5/10/20/30/40.

Selection has been attacked four times -- OBSERVATIONS 19.3-19.6 formulations, 19.5's 23 hand
specified rules and 28 greedy cells, 19.12's exhaustive C(20,5) = 15,504 set-cohesion enumeration
over 532 cells -- and every one is dead or failed its gate.  Every one of them proposed a FIX.
Nobody has ever simply partitioned the candidates and looked at which hands are missed.  On the pair
side the equivalent audit (19.10) overturned two standing premises in an afternoon.  This is that
audit for evidence.

THE PARTITION
For each audit pair, the stage-1 top-20 candidates in stage-2 score order, positions 1..20:

    TP  listed and selected        (position <= 5, listed)
    FN  listed but NOT selected    (position >  5, listed)      <- the loss
    FP  selected but NOT listed    (position <= 5, not listed)  <- what displaced it
    TN  neither

and, separately, RETRIEVAL MISSES: listed hands that are not in the top-20 at all and can never be
selected.

THE FIVE QUESTIONS, in order
 1  where do the FNs sit in the stage-2 order?  Full rank histogram 6..20 and its median, against
    the matched null in which FN is a uniform random subset of positions 6..20.
 2  is the loss concentrated or uniform?  Per-pair overlap distribution 0/5..5/5, whole and by
    family (coordinated_isolation is the weakest family at MAP@5 .4205 on the public read, .5645 on
    this audit surface).
 3  THE CENTRAL PAIRED QUESTION: within a pair, is the FN distinguishable from the FP that displaced
    it?  Nobody has used this conditioning.  Every statistic here is PAIRED -- within-pair AUC over
    the FN x FP comparisons of the same pair, plus a sign test and a Wilcoxon on the per-pair mean
    difference -- so pair-level confounds cancel.  Reported for every available column, with a
    family-wise permutation control on the maximum deviation so the 479-column search is priced.
 4  can anything separate FN from FP OUT OF FOLD?  A modest LightGBM on the paired FN-vs-FP problem,
    5-way pool-hash OOF, AUC with a table bootstrap CI and a matched null.  This is the decidable
    version of "is the residual gap noise or signal".
 5  what would it be worth?  The separability is converted into overlap-per-pair through an
    ORACLE-CONDITIONED swap (it is told which candidates are FN and which are FP; no deployable
    selector knows that, so it is an UPPER BOUND) and then to composite at the .2 weight, and
    checked against the +-.007 public resolution.

PROTOCOL (any violation invalidates the run)
  TRAINING ONLY.  Held folds 0, 1, 2 are never read for truth.  load_fold(0) supplies FEATURE
  COLUMNS only, exactly as steps 74/76/77/84 do.  Truth is the 162-pair audit OOF surface.
  check_protected() at entry and exit, both hash dicts recorded.
  MATCHED CONTROL for every real number: a within-pair permutation of which candidates are labelled
  "listed", preserving the count per pair.  The PRIMARY null also preserves the count inside the
  selected-5 and the unselected-15 strata, so |TP|, |FN| and |FP| are identical per pair and the
  rank structure of the paired problem is untouched -- only the association between features and
  listedness is destroyed.  The SECONDARY null preserves only the per-pair count (the literal
  reading), which additionally randomises how many listed hands land in the five.
  RE-SALT: three pool partitions, audit-trunc-v1 / audit-resplit-b / audit-resplit-c; anything
  called a result is the MINIMUM across them.
  prediction-evidence-confusion.json is written BEFORE any fit and never edited.
  Artefacts are append-only: this step asserts its output files do not already exist.
  NOTHING is proposed for submission.  Diagnostic only.

Output: Claude Version/experiments/evidence_confusion_C_20260912/
        prediction-evidence-confusion.json  (before any fit)
        audit-C-evidence-confusion.json     (the study)
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import binomtest, rankdata, wilcoxon
from sklearn.metrics import roc_auc_score

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, ORIGINAL, check_protected, read_json, write_json
from evidence_aug.step5_evaluate import BOOT_SEED, ap5, pool_bootstrap
from evidence_aug.step14_audit_truncation import CLF, SPLITS, TOP
from evidence_aug.step74_length_shift import CHRONO3, S1, SEEDS, build_d0, designs_of, frame_at
from evidence_aug.step76_C_order_vs_set import ranked_lists
from evidence_aug.step77_C_label_process import add_derived
from evidence_aug.step84_C_setlevel_exhaustive import (avg_rank_pct, fit_pointwise,
                                                       ordered_candidates, resplit)

AUTHOR = 'claude'
STEP = 87
OUTDIR = CV / 'experiments' / 'evidence_confusion_C_20260912'
PRED_FILE = OUTDIR / 'prediction-evidence-confusion.json'
RES_FILE = OUTDIR / 'audit-C-evidence-confusion.json'
DATA = ORIGINAL / 'data'

SALTS = ['audit-trunc-v1', 'audit-resplit-b', 'audit-resplit-c']
NPERM_STAT = 200            # permutation replicates for the (cheap) paired column statistics
NPERM_MODEL = 24            # permutation replicates for the (expensive) OOF model; p resolution 1/25
PERM_SEED = 20260912
BOOT_REPS_AUC = 800

# stored reproduction targets (steps 74 / 76 / 77)
STORED_MAP5 = 0.6159
STORED_HITS = 3.3333
STORED_ORACLE = 4.642
STORED_CI_MAP5 = 0.5645     # coordinated_isolation on this audit surface (step 76)

DERIVED = ['n_board', 'pot_bb_raw', 'type_showdown', 'either_member_sd', 'max_stack_bb',
           'min_stack_bb', 'max_put_bb', 'min_put_bb', 'net_ben_bb', 'net_oth_bb', 'chen_ben',
           'chen_oth', 'put_ben_bb', 'put_oth_bb', 'oth_folded', 'flow_to_ben_bb', 'xfer_to_ben_bb',
           'xfer_abs_bb', 'winner_is_ben', 'loser_put_bb', 'others_net_bb', 'pot_over_min_stack',
           'chen_gap_ben', 'folder_chen', 'gap_prev_s', 'gap_next_s']

GATE = {
    'question': 'within a pair, is the listed-but-not-selected hand (FN) distinguishable from the '
                'selected-but-not-listed hand (FP) that displaced it?',
    'primary_estimate': 'out-of-fold WITHIN-PAIR AUC of a modest LightGBM (n_estimators=150, '
                        'num_leaves=7, max_depth=3, min_child_samples=40, lr=.05, 3 seeds) fit on '
                        'the paired FN-vs-FP rows only, label 1 = FN (listed), 0 = FP (unlisted), '
                        'features = within-top-20 average-rank percentiles, 5-way pool-hash OOF by '
                        'table.  Evaluated on the pairs contributing at least one FN and one FP.',
    'secondary_estimate': 'pooled row-level OOF AUC over the same rows',
    'headline_rule': 'the MINIMUM over the three pool partitions of (real - matched null)',
    'matched_control': 'THREE nulls, all declared here before any number exists, each re-run end to '
                       'end and reported beside its real twin.  (a) BLOCK_SWAP (PRIMARY): each '
                       "pair's 20 x C feature block is replaced by another same-family pair's block "
                       'at MATCHED POSITIONS (a cyclic within-family derangement), leaving the '
                       'listed mask, the rank positions and |TP|/|FN|/|FP| exactly as they are.  It '
                       'preserves the rank-to-feature association -- the artefact that FN sits below '
                       'FP in the score order and therefore differs on every score-correlated '
                       'column -- and destroys only the association between the features and WHICH '
                       'candidate is listed.  (b) STRATIFIED: within-pair permutation of the listed '
                       'labels, count preserved inside the selected-5 and inside the unselected-15, '
                       'so |TP|/|FN|/|FP| are identical per pair.  (c) UNSTRATIFIED: within-pair '
                       'permutation of the listed labels preserving only the per-pair count.  '
                       'DECLARED IN ADVANCE: (b) and (c) move the FN off its real rank positions -- '
                       'a relabelled FN is uniform over ranks 6..20 while a real FN is not -- so '
                       'they make the FN-vs-FP problem EASIER than it really is and their real-minus-'
                       'null will read negative whenever the real FNs are near-misses.  That is a '
                       'restatement of question 1, not a measure of separability, so the gate is '
                       'stated on (a), which is the only null that holds the rank structure fixed.  '
                       '(b) and (c) are reported in full beside it.',
    'third_reference': 'a STAGE1_ONLY design (the within-pair percentiles of oof and oof_rank plus '
                       'the family indicators) is fitted alongside, because the partition is defined '
                       'by the stage-2 order and stage 1 is its dominant input: any full-design AUC '
                       'must be read as an increment over that trivial model, and the increment is '
                       'reported with a paired table-bootstrap CI.',
    'separable_if': 'min over the three partitions of (real within-pair AUC - BLOCK_SWAP-null '
                    'within-pair AUC) > 0 AND the real within-pair AUC pool-bootstrap 95 % lower '
                    'bound > .5 under all three partitions',
    'alive_if': 'separable AND the ORACLE-CONDITIONED composite upper bound (real - BLOCK_SWAP '
                'null), minimum over the three partitions, is >= .007',
    'resolution': 'composite = 0.2 * delta_MAP@5; the public leaderboard resolves +-.007 composite',
    'why_the_conversion_is_an_upper_bound': 'the FN-vs-FP problem conditions on already knowing '
                                            'which of the top-20 are listed.  No deployable selector '
                                            'has that information, so any overlap gain measured here '
                                            'bounds the deployable gain from above; and a swap that '
                                            'is told the FN/FP identity sets already gains '
                                            '|FN||FP|/(|FN|+|FP|) hits per pair by pure chance, so '
                                            'the chance level and the null are both reported.',
    'q1_control': 'the FN rank histogram against the stratified null, in which FN is a uniform '
                  'random subset of positions 6..20',
    'q2_control': 'the per-pair overlap distribution against the unstratified null, which is '
                  'hypergeometric(20, listed-in-top-20, 5)',
    'q3_familywise': 'the paired column search is priced by the maximum over all columns of '
                     '|within-pair AUC - .5| against the same maximum on NPERM_STAT null replicates '
                     'of BOTH the block-swap and the stratified kind; no column is called '
                     'separating unless the real maximum beats the BLOCK-SWAP null maximum at '
                     'permutation p <= .05, and BH-FDR q-values are reported beside',
    'stage2_score_is_excluded_from_the_model': 'the stage-2 pointwise score and its rank separate FN '
                                               'from FP perfectly BY CONSTRUCTION (FP is in the five '
                                               'and FN is not).  They are reported as a tautology '
                                               'check and never given to the model.',
    'production_branch': 'none: diagnostic only, nothing is proposed for submission from it',
    'held_folds': 'folds 0, 1, 2 are not read for truth; load_fold(0) supplies FEATURE COLUMNS only, '
                  'exactly as steps 74/76/77/84 do.  claude held-fold read count unchanged at 5.',
    'training_only': 'the 162 audit pairs in 104 tables outside every held fold',
}


def log(*a):
    print(*a, flush=True)


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def r5(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 5)


def bh(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for k in range(n - 1, -1, -1):
        prev = min(prev, p[o[k]] * n / (k + 1))
        q[o[k]] = prev
    return q


def table_boot_mean(values_by_pair, table_by_pair):
    """pool bootstrap of a pair-weighted mean (same machinery as step 5's pool_bootstrap)."""
    return [r4(x) for x in pool_bootstrap(values_by_pair, table_by_pair)]


def table_boot_auc(y, s, tab, reps=BOOT_REPS_AUC, seed=BOOT_SEED):
    """table-resampled 95 % CI for a pooled row-level AUC."""
    tabs = sorted(set(tab))
    ix = {t: np.where(np.asarray(tab) == t)[0] for t in tabs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(tabs), len(tabs))
        rows = np.concatenate([ix[tabs[j]] for j in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, s[rows]))
    return [r4(np.quantile(out, .025)), r4(np.quantile(out, .975))] if out else [None, None]


# --------------------------------------------------------------------------------------------
# the per-pair candidate block: (npair, 20, ncols) percentiles in STAGE-2 order
# --------------------------------------------------------------------------------------------
def blocks(d, pred, cols):
    """Return pids, hand ids, the (npair, 20, ncols) within-pair percentile tensor in stage-2 score
    order, the listed mask, the stage-2 score percentile, and the per-pair listed count."""
    order = ordered_candidates(d, pred, TOP)
    cand = d.filter(pl.col('oof_rank') <= TOP).sort('pair_id', 'hand_id')
    npair = cand.height // TOP
    assert cand.height == npair * TOP, 'not every pair has exactly 20 candidates'
    pid_blk = cand['pair_id'].to_numpy().reshape(npair, TOP)
    assert (pid_blk == pid_blk[:, :1]).all(), 'candidate rows are not contiguous by pair'
    pids = [str(x) for x in pid_blk[:, 0]]
    hand_blk = cand['hand_id'].to_numpy().reshape(npair, TOP)
    rel_blk = cand['relevance'].to_numpy().astype(bool).reshape(npair, TOP)

    M = cand.select(cols).to_numpy().astype(np.float64).reshape(npair, TOP, len(cols))
    PCT = (rankdata(M, method='average', axis=1) - .5) / TOP
    # control: the polars/scipy path must equal step 84's avg_rank_pct hand for hand
    for i in (0, npair // 2, npair - 1):
        for j in (0, len(cols) // 2, len(cols) - 1):
            assert np.allclose(PCT[i, :, j], avg_rank_pct(M[i, :, j])), 'percentile path diverged'

    s2 = pred[(d['oof_rank'] <= TOP).to_numpy()].reshape(npair, TOP)   # d is sorted pair_id, hand_id
    POS = np.empty((npair, TOP), dtype=np.int64)
    for i, p in enumerate(pids):
        where = {h: j for j, h in enumerate(hand_blk[i])}
        POS[i] = [where[h] for h in order[p]]
    Q = np.take_along_axis(PCT, POS[:, :, None], axis=1)
    REL = np.take_along_axis(rel_blk, POS, axis=1)
    HAND = np.take_along_axis(hand_blk, POS, axis=1)
    S2 = np.take_along_axis(s2, POS, axis=1)
    assert (np.diff(S2, axis=1) <= 1e-12).all(), 'stage-2 order is not descending'
    S2PCT = (rankdata(S2, method='average', axis=1) - .5) / TOP
    return pids, HAND, Q, REL, S2, S2PCT


def rows_of(REL):
    """the paired FN-vs-FP row set: (pair index, position, label) with 1 = FN (listed), 0 = FP."""
    npair = REL.shape[0]
    pi, po, y = [], [], []
    for i in range(npair):
        for j in range(TOP):
            if j < 5 and not REL[i, j]:
                pi.append(i); po.append(j); y.append(0.0)
            elif j >= 5 and REL[i, j]:
                pi.append(i); po.append(j); y.append(1.0)
    return np.array(pi, dtype=np.int64), np.array(po, dtype=np.int64), np.array(y, dtype=np.float64)


def swap_blocks(Q, fam_groups, rng):
    """POSITION-MATCHED null.  Replace every pair's 20 x C feature block with another same-family
    pair's block at the SAME positions (a cyclic derangement inside each family).  The listed mask,
    the rank positions and |TP|/|FN|/|FP| are untouched, and position j still receives a position-j
    feature vector, so the rank-to-feature association -- the artefact that FN sits below FP in the
    score order -- survives intact.  Only the association between the features and WHICH candidate
    of the pair is listed is destroyed."""
    donor = np.arange(Q.shape[0])
    for ix in fam_groups:
        o = rng.permutation(ix)
        if len(o) > 1:
            donor[o] = np.roll(o, 1)
    assert (donor != np.arange(Q.shape[0])).all() or Q.shape[0] < 2, 'block swap left a fixed point'
    return Q[donor], donor


def permute_rel(REL, rng, stratified):
    """within-pair permutation of which candidates are labelled listed, count per pair preserved.
    stratified=True additionally preserves the count inside positions 1-5 and 6-20, so |TP|, |FN|
    and |FP| are identical per pair and only the association with the features is destroyed."""
    out = np.zeros_like(REL)
    for i in range(REL.shape[0]):
        if stratified:
            kt, kb = int(REL[i, :5].sum()), int(REL[i, 5:].sum())
            if kt:
                out[i, rng.choice(5, kt, replace=False)] = True
            if kb:
                out[i, 5 + rng.choice(TOP - 5, kb, replace=False)] = True
        else:
            k = int(REL[i].sum())
            if k:
                out[i, rng.choice(TOP, k, replace=False)] = True
    return out


# --------------------------------------------------------------------------------------------
# paired column statistics
# --------------------------------------------------------------------------------------------
def paired_wpa(Q, REL, extra=None):
    """within-pair AUC of every column over the FN x FP comparisons of the same pair, plus the
    per-pair mean-percentile difference.  Returns (wpa_by_pair (np, C), delta_by_pair, mask)."""
    npair, _, C = Q.shape
    Ce = C + (0 if extra is None else extra.shape[2])
    W = np.full((npair, Ce), np.nan)
    D = np.full((npair, Ce), np.nan)
    ok = np.zeros(npair, dtype=bool)
    for i in range(npair):
        fn = np.where(REL[i] & (np.arange(TOP) >= 5))[0]
        fp = np.where(~REL[i] & (np.arange(TOP) < 5))[0]
        if len(fn) == 0 or len(fp) == 0:
            continue
        ok[i] = True
        A = Q[i, fn] if extra is None else np.concatenate([Q[i, fn], extra[i, fn]], axis=1)
        B = Q[i, fp] if extra is None else np.concatenate([Q[i, fp], extra[i, fp]], axis=1)
        gt = (A[:, None, :] > B[None, :, :]).sum((0, 1))
        eq = (A[:, None, :] == B[None, :, :]).sum((0, 1))
        W[i] = (gt + .5 * eq) / (len(fn) * len(fp))
        D[i] = A.mean(0) - B.mean(0)
    return W, D, ok


# --------------------------------------------------------------------------------------------
# the out-of-fold FN-vs-FP model
# --------------------------------------------------------------------------------------------
def oof_scores(Q, FAMOH, pi, po, y, colix, split_of_pair, seeds=SEEDS):
    X = np.concatenate([Q[pi, po][:, colix], FAMOH[pi]], axis=1)
    sp = split_of_pair[pi]
    s = np.full(len(y), .5)
    for k in range(SPLITS):
        te = sp == k
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0 or y[tr].sum() in (0, tr.sum()):
            continue
        acc = np.zeros(int(te.sum()))
        for seed in seeds:
            clf = lgb.LGBMClassifier(**CLF, random_state=seed)
            clf.fit(X[tr], y[tr])
            acc += clf.predict_proba(X[te])[:, 1] / len(seeds)
        s[te] = acc
    return s


def within_pair_auc(pi, y, s, npair):
    """per-pair concordance over the FN x FP comparisons; only pairs with both contribute."""
    out = {}
    for i in np.unique(pi):
        m = pi == i
        a, b = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(a) == 0 or len(b) == 0:
            continue
        out[int(i)] = float(((a[:, None] > b[None, :]).sum() + .5 * (a[:, None] == b[None, :]).sum())
                            / (len(a) * len(b)))
    return out


def ap5_of(rel_vec, order5, nlisted):
    hits, tot = 0, 0.0
    for r, pos in enumerate(order5, 1):
        if rel_vec[pos]:
            hits += 1
            tot += hits / r
    return tot / min(5, nlisted) if nlisted else 0.0


def oracle_swap(REL, NLIST, pi, po, y, s, npair):
    """ORACLE-CONDITIONED conversion.  Told which candidates are FN and which are FP, fill the
    |FP| freed slots with the top-|FP| of FN u FP by model score, keeping the TP hands where they
    are and giving the earliest freed slot to the highest-scoring promotion.  Returns per-pair new
    overlap, new AP@5, and the analytic chance gain."""
    new_hits = np.array([int(REL[i, :5].sum()) for i in range(npair)], dtype=float)
    new_ap = np.array([ap5_of(REL[i], list(range(5)), int(NLIST[i])) for i in range(npair)])
    chance = np.zeros(npair)
    for i in range(npair):
        m = pi == i
        if not m.any():
            continue
        pos_i, y_i, s_i = po[m], y[m], s[m]
        nfp = int((y_i == 0).sum())
        nfn = int((y_i == 1).sum())
        if nfp == 0:
            continue
        chance[i] = nfn * nfp / (nfn + nfp) if (nfn + nfp) else 0.0
        take = np.argsort(-s_i, kind='stable')[:nfp]
        gain = float(y_i[take].sum())
        new_hits[i] += gain
        free = sorted(int(p) for p in pos_i[y_i == 0])
        keep = [j for j in range(5) if REL[i, j]]
        promoted = [int(pos_i[t]) for t in take]
        order5 = list(keep) + []
        slot = {}
        for f, hp in zip(free, promoted):
            slot[f] = hp
        order5 = [slot.get(j, j) for j in range(5) if (REL[i, j] or j in slot)]
        new_ap[i] = ap5_of(REL[i], order5, int(NLIST[i]))
    return new_hits, new_ap, chance


# --------------------------------------------------------------------------------------------
def analyse_partition(d, pred, cols, interp_ix, s1_ix, split_of_pair, ptab, labels, truths, tag,
                      nperm_model, do_unstratified):
    """The whole study on one pool partition: confusion, paired statistics, OOF model, conversion,
    and every matched null."""
    pids, HAND, Q, REL, S2, S2PCT = blocks(d, pred, cols)
    npair = len(pids)
    NLIST = np.array([len(truths[p]) for p in pids], dtype=float)
    MISS = np.array([len(truths[p]) - int(REL[i].sum()) for i, p in enumerate(pids)], dtype=float)
    FAMOH = np.stack([[1.0 if labels[p] == f else 0.0 for f in FAMILIES] for p in pids])
    tabs = [ptab[p] for p in pids]
    fam_groups = [np.array([i for i, p in enumerate(pids) if labels[p] == f]) for f in FAMILIES]

    tp = REL[:, :5].sum(1).astype(int)
    fn = REL[:, 5:].sum(1).astype(int)
    fp = 5 - tp
    tn = (TOP - 5) - fn
    conf = {'TP': int(tp.sum()), 'FN': int(fn.sum()), 'FP': int(fp.sum()), 'TN': int(tn.sum()),
            'retrieval_misses': int(MISS.sum()), 'listed_total': int(NLIST.sum()),
            'listed_in_top20': int(REL.sum()),
            'recall_at_20_hand_level': r4(REL.sum() / NLIST.sum()),
            'pairs': npair, 'pairs_with_a_retrieval_miss': int((MISS > 0).sum()),
            'pairs_with_at_least_one_FN_and_one_FP': int(((fn > 0) & (fp > 0)).sum()),
            'mean_FN_per_pair': r4(fn.mean()), 'mean_FP_per_pair': r4(fp.mean()),
            'mean_overlap': r4(tp.mean())}

    # ---------------------------------------------------------------- Q1  FN rank histogram
    fn_ranks = [j + 1 for i in range(npair) for j in range(5, TOP) if REL[i, j]]
    fp_ranks = [j + 1 for i in range(npair) for j in range(5) if not REL[i, j]]
    q1 = {'fn_rank_histogram': {str(k): int(sum(1 for r in fn_ranks if r == k)) for k in range(6, TOP + 1)},
          'fn_rank_median': float(np.median(fn_ranks)) if fn_ranks else None,
          'fn_rank_mean': r4(np.mean(fn_ranks)) if fn_ranks else None,
          'fn_at_rank_6': int(sum(1 for r in fn_ranks if r == 6)),
          'fn_in_6_to_10': int(sum(1 for r in fn_ranks if r <= 10)),
          'fn_in_11_to_20': int(sum(1 for r in fn_ranks if r > 10)),
          'fn_total': len(fn_ranks),
          'best_fn_rank_per_pair_histogram': {},
          'fp_rank_histogram': {str(k): int(sum(1 for r in fp_ranks if r == k)) for k in range(1, 6)},
          'fp_rank_median': float(np.median(fp_ranks)) if fp_ranks else None,
          'null_uniform_expected_per_rank': r4(len(fn_ranks) / 15.0),
          'null_median_if_uniform': 13.0,
          'null_note': 'under either label-permutation null a FN is a uniform random draw from '
                       'positions 6..20, so the exact matched null histogram is flat at '
                       'n_FN / 15 with median 13',
          'by_family': {}}
    for f in FAMILIES:
        fr = [j + 1 for i in range(npair) if labels[pids[i]] == f
              for j in range(5, TOP) if REL[i, j]]
        q1['by_family'][f] = {
            'n_FN': len(fr), 'median_rank': float(np.median(fr)) if fr else None,
            'mean_rank': r4(np.mean(fr)) if fr else None,
            'share_in_6_to_10': r4(sum(1 for r in fr if r <= 10) / len(fr)) if fr else None,
            'histogram': {str(k): int(sum(1 for r in fr if r == k)) for k in range(6, TOP + 1)}}
    best = [min(j + 1 for j in range(5, TOP) if REL[i, j]) for i in range(npair) if fn[i] > 0]
    q1['best_fn_rank_per_pair_histogram'] = {str(k): int(sum(1 for r in best if r == k))
                                             for k in sorted(set(best))}
    q1['best_fn_rank_median'] = float(np.median(best)) if best else None

    # ---------------------------------------------------------------- Q2  overlap distribution
    q2 = {'overlap_histogram': {str(k): int((tp == k).sum()) for k in range(6)},
          'mean': r4(tp.mean()), 'median': float(np.median(tp)),
          'sd': r4(tp.std(ddof=1)), 'by_family': {}}
    for f in FAMILIES:
        m = np.array([labels[p] == f for p in pids])
        q2['by_family'][f] = {
            'pairs': int(m.sum()), 'mean_overlap': r4(tp[m].mean()),
            'histogram': {str(k): int((tp[m] == k).sum()) for k in range(6)},
            'mean_FN': r4(fn[m].mean()), 'mean_retrieval_miss': r4(MISS[m].mean()),
            'oracle_overlap_top20': r4(np.minimum(REL[m].sum(1), 5).mean())}
    # unstratified-null (hypergeometric) reference for the same histogram
    rng0 = np.random.default_rng(PERM_SEED + 7)
    hh = np.zeros(6)
    for _ in range(400):
        Rn = permute_rel(REL, rng0, stratified=False)
        t = Rn[:, :5].sum(1)
        for k in range(6):
            hh[k] += (t == k).sum() / 400.0
    q2['null_unstratified_overlap_histogram'] = {str(k): r4(hh[k]) for k in range(6)}
    q2['null_unstratified_mean_overlap'] = r4(sum(k * hh[k] for k in range(6)) / npair)

    # ---------------------------------------------------------------- Q3  paired column statistics
    extra = np.stack([S2PCT], axis=2)                     # the tautology column, reported separately
    W, D, ok = paired_wpa(Q, REL, extra=extra)
    names = list(cols) + ['_STAGE2_SCORE_PCT_tautology']
    okix = np.where(ok)[0]
    per_col = {}
    pvals, order_names = [], []
    for c, name in enumerate(names):
        w = W[okix, c]
        dl = D[okix, c]
        pos = int((dl > 0).sum())
        neg = int((dl < 0).sum())
        try:
            sp = float(binomtest(pos, pos + neg, .5).pvalue) if pos + neg else 1.0
        except Exception:
            sp = 1.0
        try:
            wp = float(wilcoxon(dl).pvalue) if np.any(dl != 0) else 1.0
        except Exception:
            wp = 1.0
        per_col[name] = {'within_pair_auc': r4(w.mean()), 'dev_from_half': r4(w.mean() - .5),
                         'mean_delta_pct_FN_minus_FP': r4(dl.mean()),
                         'median_delta': r4(np.median(dl)),
                         'pairs_FN_higher': pos, 'pairs_FP_higher': neg,
                         'sign_test_p': r5(sp), 'wilcoxon_p': r5(wp)}
        if name != '_STAGE2_SCORE_PCT_tautology':
            pvals.append(sp)
            order_names.append(name)
    q = bh(pvals)
    for name, qq in zip(order_names, q):
        per_col[name]['sign_test_bh_q'] = r5(qq)
    wmean = np.nanmean(W[okix], axis=0)                  # unrounded, for the family-wise control
    real_dev = np.abs(wmean[:len(order_names)] - .5)
    real_maxdev = float(np.nanmax(real_dev))
    best_cols = [order_names[j] for j in np.argsort(-real_dev)[:30]]

    # matched nulls for the whole column search: the position-matched block swap (primary) and the
    # within-pair label permutation (the literal control), both re-run end to end
    q3 = {'n_columns_tested': len(order_names), 'pairs_contributing': int(ok.sum()),
          'real_max_abs_dev_from_half': r4(real_maxdev),
          'real_mean_abs_dev': r4(np.nanmean(real_dev)),
          'columns_with_bh_q_below_.05': int(sum(1 for n in order_names
                                                 if per_col[n]['sign_test_bh_q'] is not None
                                                 and per_col[n]['sign_test_bh_q'] < .05)),
          'nulls': {}}
    for kind in ('block_swap', 'stratified'):
        rng = np.random.default_rng(PERM_SEED + (0 if kind == 'block_swap' else 55))
        null_maxdev, null_meandev = [], []
        null_col = np.zeros(len(order_names))
        for _ in range(NPERM_STAT):
            if kind == 'block_swap':
                Qn, _ = swap_blocks(Q, fam_groups, rng)
                Wn, _, okn = paired_wpa(Qn, REL)
            else:
                Wn, _, okn = paired_wpa(Q, permute_rel(REL, rng, stratified=True))
            wn = np.nanmean(Wn[np.where(okn)[0]], axis=0)
            dv = np.abs(wn - .5)
            null_maxdev.append(float(np.nanmax(dv)))
            null_meandev.append(float(np.nanmean(dv)))
            null_col += dv / NPERM_STAT
        p_max = (1 + sum(1 for x in null_maxdev if x >= real_maxdev - 1e-12)) / (1 + NPERM_STAT)
        q3['nulls'][kind] = {
            'replicates': NPERM_STAT,
            'null_max_abs_dev_from_half_mean': r4(np.mean(null_maxdev)),
            'null_max_abs_dev_from_half_p95': r4(np.quantile(null_maxdev, .95)),
            'real_minus_null_max_abs_dev': r4(real_maxdev - np.mean(null_maxdev)),
            'familywise_permutation_p': r4(p_max),
            'null_mean_abs_dev': r4(np.mean(null_meandev)),
            'n_columns_beating_their_own_null_column_mean': int((real_dev > null_col).sum())}
        q3[f'per_column_null_{kind}_mean_abs_dev'] = {order_names[j]: r4(null_col[j])
                                                      for j in np.argsort(-real_dev)[:30]}
    q3['familywise_permutation_p'] = q3['nulls']['block_swap']['familywise_permutation_p']
    q3['top30_by_abs_deviation'] = {n: per_col[n] for n in best_cols}
    q3['tautology_check_stage2_score'] = per_col['_STAGE2_SCORE_PCT_tautology']
    q3['all_columns'] = per_col

    # ---------------------------------------------------------------- Q4  OOF separability
    pi, po, y = rows_of(REL)
    designs = {'pct_all': np.arange(len(cols)), 'pct_interpretable': np.asarray(interp_ix),
               'stage1_only': np.asarray(s1_ix)}
    sp_pair = np.array([split_of_pair[p] for p in pids])
    q4 = {'rows': int(len(y)), 'FN_rows': int(y.sum()), 'FP_rows': int((y == 0).sum()),
          'designs': {}}
    swap_real, exact, wpa_real = {}, {}, {}
    for dn, cix in designs.items():
        s = oof_scores(Q, FAMOH, pi, po, y, cix, sp_pair)
        auc = float(roc_auc_score(y, s))
        wpa = within_pair_auc(pi, y, s, npair)
        wpa_pid = {pids[i]: v for i, v in wpa.items()}
        wpa_real[dn] = wpa_pid
        rowtab = [tabs[i] for i in pi]
        nh, na, chance = oracle_swap(REL, NLIST, pi, po, y, s, npair)
        swap_real[dn] = (nh, na, chance)
        exact[dn] = {'pooled': auc, 'wpa': float(np.mean(list(wpa.values())))}
        q4['designs'][dn] = {
            'n_features': int(len(cix) + len(FAMILIES)),
            'pooled_oof_auc': r4(auc),
            'pooled_oof_auc_ci95': table_boot_auc(y, s, rowtab),
            'within_pair_auc': r4(np.mean(list(wpa.values()))),
            'within_pair_auc_ci95': table_boot_mean(wpa_pid, ptab),
            'pairs_scored': len(wpa)}
    # the increment over the trivial stage-1 model, paired over pairs
    for dn in designs:
        if dn == 'stage1_only':
            continue
        inc = {p: wpa_real[dn][p] - wpa_real['stage1_only'][p] for p in wpa_real[dn]}
        q4['designs'][dn]['within_pair_auc_minus_stage1_only'] = r4(np.mean(list(inc.values())))
        q4['designs'][dn]['within_pair_auc_minus_stage1_only_ci95'] = table_boot_mean(inc, ptab)

    # matched nulls for the model: block swap (primary), stratified, unstratified
    null_kinds = [('block_swap', None), ('stratified', True)] + \
                 ([('unstratified', False)] if do_unstratified else [])
    for kind, strat in null_kinds:
        rngm = np.random.default_rng(
            PERM_SEED + {'block_swap': 0, 'stratified': 55, 'unstratified': 101}[kind])
        acc = {dn: {'pooled': [], 'wpa': [], 'swap_hits': [], 'swap_ap': [], 'chance': []}
               for dn in designs}
        for _ in range(nperm_model):
            if kind == 'block_swap':
                Qn, _ = swap_blocks(Q, fam_groups, rngm)
                Rn, pin, pon, yn = REL, pi, po, y
            else:
                Qn = Q
                Rn = permute_rel(REL, rngm, stratified=strat)
                pin, pon, yn = rows_of(Rn)
            if yn.sum() == 0 or yn.sum() == len(yn):
                continue
            base_h = Rn[:, :5].sum(1).astype(float)
            base_a = np.array([ap5_of(Rn[i], list(range(5)), int(NLIST[i])) for i in range(npair)])
            for dn, cix in designs.items():
                sn = oof_scores(Qn, FAMOH, pin, pon, yn, cix, sp_pair)
                acc[dn]['pooled'].append(float(roc_auc_score(yn, sn)))
                wn = within_pair_auc(pin, yn, sn, npair)
                acc[dn]['wpa'].append(float(np.mean(list(wn.values()))) if wn else .5)
                nh, na, ch = oracle_swap(Rn, NLIST, pin, pon, yn, sn, npair)
                acc[dn]['swap_hits'].append(float((nh - base_h).mean()))
                acc[dn]['swap_ap'].append(float((na - base_a).mean()))
                acc[dn]['chance'].append(float(ch.mean()))
        for dn in designs:
            a = acc[dn]
            v = q4['designs'][dn]
            rw, rp = exact[dn]['wpa'], exact[dn]['pooled']
            v[f'null_{kind}_pooled_auc'] = r4(np.mean(a['pooled'])) if a['pooled'] else None
            v[f'null_{kind}_within_pair_auc'] = r4(np.mean(a['wpa'])) if a['wpa'] else None
            v[f'real_minus_null_{kind}_pooled_auc'] = r4(rp - np.mean(a['pooled'])) if a['pooled'] else None
            v[f'real_minus_null_{kind}_within_pair_auc'] = r4(rw - np.mean(a['wpa'])) if a['wpa'] else None
            v[f'null_{kind}_within_pair_auc_spread'] = [r4(np.min(a['wpa'])), r4(np.max(a['wpa']))] if a['wpa'] else None
            v[f'null_{kind}_perm_p_within_pair_auc'] = r4(
                (1 + sum(1 for x in a['wpa'] if x >= rw - 1e-12)) / (1 + len(a['wpa']))) if a['wpa'] else None
            v[f'null_{kind}_swap_delta_overlap'] = r4(np.mean(a['swap_hits'])) if a['swap_hits'] else None
            v[f'null_{kind}_swap_delta_map5'] = r4(np.mean(a['swap_ap'])) if a['swap_ap'] else None
            v[f'null_{kind}_swap_chance_overlap'] = r4(np.mean(a['chance'])) if a['chance'] else None
        q4[f'null_{kind}_replicates'] = nperm_model

    # ---------------------------------------------------------------- Q5  what it would be worth
    base_h = tp.astype(float)
    base_a = np.array([ap5_of(REL[i], list(range(5)), int(NLIST[i])) for i in range(npair)])
    q5 = {'what': 'ORACLE-CONDITIONED upper bound: the swap is told which candidates are FN and '
                  'which are FP and only has to rank inside that set.  A deployable selector does '
                  'not know that, so this bounds the deployable gain from above.',
          'baseline_overlap': r4(base_h.mean()), 'baseline_map5': r4(base_a.mean()), 'designs': {}}
    for dn in designs:
        nh, na, chance = swap_real[dn]
        dh = {pids[i]: nh[i] - base_h[i] for i in range(npair)}
        da = {pids[i]: na[i] - base_a[i] for i in range(npair)}
        v4 = q4['designs'][dn]
        prim = v4.get('null_block_swap_swap_delta_map5')
        prim_h = v4.get('null_block_swap_swap_delta_overlap')
        q5['designs'][dn] = {
            'oracle_swap_overlap': r4(nh.mean()),
            'delta_overlap': r4(nh.mean() - base_h.mean()),
            'delta_overlap_ci95': table_boot_mean(dh, ptab),
            'chance_swap_delta_overlap': r4(chance.mean()),
            'delta_overlap_above_chance': r4(nh.mean() - base_h.mean() - chance.mean()),
            'oracle_swap_map5': r4(na.mean()),
            'delta_map5': r4(na.mean() - base_a.mean()),
            'delta_map5_ci95': table_boot_mean(da, ptab),
            'composite_delta_upper_bound': r5(.2 * (na.mean() - base_a.mean())),
            'null_block_swap_delta_overlap': prim_h,
            'null_block_swap_delta_map5': prim,
            'real_minus_null_delta_overlap': r4(nh.mean() - base_h.mean() - prim_h) if prim_h is not None else None,
            'real_minus_null_delta_map5': r4((na.mean() - base_a.mean()) - prim) if prim is not None else None,
            'composite_real_minus_null': r5(.2 * ((na.mean() - base_a.mean()) - prim)) if prim is not None else None,
            'null_stratified_delta_map5': v4.get('null_stratified_swap_delta_map5'),
            'null_unstratified_delta_map5': v4.get('null_unstratified_swap_delta_map5'),
            'clears_public_resolution_0.007': bool(prim is not None and
                                                   abs(.2 * ((na.mean() - base_a.mean()) - prim)) >= .007)}
    return {'tag': tag, 'confusion': conf, 'q1_fn_rank': q1, 'q2_overlap': q2,
            'q3_paired_columns': q3, 'q4_separability': q4, 'q5_worth': q5}


# --------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    hashes_in = check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------ prediction, BEFORE any fit
    pred_doc = {'author': AUTHOR, 'step': STEP, 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'gate': GATE, 'salts': SALTS, 'top_k_candidates': TOP,
                'perm_replicates': {'paired_columns': NPERM_STAT, 'oof_model': NPERM_MODEL},
                'model': CLF, 'seeds': list(SEEDS),
                'stored_targets': {'map5': STORED_MAP5, 'overlap': STORED_HITS,
                                   'oracle_top20': STORED_ORACLE, 'ci_map5_audit': STORED_CI_MAP5}}
    if PRED_FILE.exists():
        old = read_json(PRED_FILE)
        assert old['gate'] == GATE, 'the prediction file exists with a DIFFERENT gate; it is ' \
                                    'append-only and must not be edited'
        log(f'  prediction file already present and identical: {PRED_FILE}')
    else:
        write_json(PRED_FILE, pred_doc)
        log(f'  wrote the prediction BEFORE any fit: {PRED_FILE}')
    assert not RES_FILE.exists(), f'{RES_FILE} exists; artefacts are append-only, refusing to overwrite'

    log('=' * 100)
    log('STEP 87 [C].  THE EVIDENCE CONFUSION MATRIX: WHICH HANDS DOES STAGE 2 MISS?')
    log('=' * 100)

    # ------------------------------------------------------ frame
    d0, struct, rcols = build_d0()
    designs2 = designs_of(struct, rcols)
    d = frame_at(d0, 1.0)
    hands = (pl.scan_parquet(DATA / 'hands.parquet')
             .select('hand_id', 'big_blind', 'final_pot', 'board_cards')
             .filter(pl.col('hand_id').is_in(d['hand_id'].implode())).collect())
    n0 = d.height
    d = add_derived(d.join(hands, on='hand_id', validate='m:1'))
    assert d.height == n0, 'hands join changed the row count'
    d = d.sort('pair_id', 'hand_id')

    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    log(f'  frame {d.height} rows, {len(truths)} pairs, {len(set(ptab.values()))} tables, '
        f"{int((d['relevance'] == 1).sum())} listed hands")

    interp = struct + CHRONO3 + S1 + DERIVED
    cols = interp + rcols
    assert len(cols) == len(set(cols)), 'duplicate analysis columns'
    missing = [c for c in cols if c not in d.columns]
    assert not missing, f'columns absent from the frame: {missing}'
    interp_ix = list(range(len(interp)))
    s1_ix = [cols.index(c) for c in S1]
    log(f'  {len(cols)} analysis columns ({len(interp)} interpretable + {len(rcols)} ranker); '
        f'the stage-2 score is EXCLUDED from every model design (tautological); the trivial '
        f'stage1_only design is {S1}')

    # ------------------------------------------------------ control 1: reproduction
    pred = fit_pointwise(d, designs2)
    top5, top20, _ = ranked_lists(d, pred, TOP)
    map5 = float(np.mean([ap5(top5[p], truths[p]) for p in truths]))
    hits = float(np.mean([len(set(top5[p]) & truths[p]) for p in truths]))
    oracle = float(np.mean([min(len(top20[p] & truths[p]), 5) for p in truths]))
    c1 = {'map5': r4(map5), 'stored_map5': STORED_MAP5, 'mean_overlap': r4(hits),
          'stored_overlap': STORED_HITS, 'oracle_overlap': r4(oracle), 'stored_oracle': STORED_ORACLE,
          'passed': abs(round(map5, 4) - STORED_MAP5) < 1e-9 and abs(round(hits, 4) - STORED_HITS) < 1e-9}
    log(f"\n  CONTROL 1 reproduction: MAP@5 {map5:.4f} (stored {STORED_MAP5}), overlap {hits:.4f} "
        f"(stored {STORED_HITS}), oracle {oracle:.4f} (stored {STORED_ORACLE}) -> "
        f"{'PASSED' if c1['passed'] else 'FAILED'}")
    assert c1['passed'], 'REPRODUCTION CONTROL FAILED; STOPPING'

    # ------------------------------------------------------ the three partitions
    parts = {}
    for si, salt in enumerate(SALTS):
        t1 = time.time()
        ds = resplit(d, salt)
        ps = pred if salt == SALTS[0] else fit_pointwise(ds, designs2)
        sp = dict(ds.select('pair_id', 'split').unique().iter_rows())
        parts[salt] = analyse_partition(ds, ps, cols, interp_ix, s1_ix, sp, ptab, labels, truths,
                                        salt, NPERM_MODEL, do_unstratified=(si == 0))
        parts[salt]['seconds'] = round(time.time() - t1)
        c = parts[salt]['confusion']
        log(f"\n  [{salt}]  {time.time() - t1:.0f} s   TP {c['TP']}  FN {c['FN']}  FP {c['FP']}  "
            f"TN {c['TN']}  retrieval-miss {c['retrieval_misses']}  overlap {c['mean_overlap']}")

    head = parts[SALTS[0]]
    # control 1b: the position-space AP@5 reconstruction must equal the chain's own MAP@5, and the
    # confusion partition must account for every listed hand exactly once
    assert abs(head['q5_worth']['baseline_map5'] - round(map5, 4)) < 1e-9, \
        'the position-space AP@5 does not reproduce the chain MAP@5'
    hc = head['confusion']
    assert hc['TP'] + hc['FN'] + hc['retrieval_misses'] == hc['listed_total'], 'partition leaks'
    assert hc['TP'] + hc['FP'] == 5 * hc['pairs'], 'the selected five is not five'
    assert abs(head['q3_paired_columns']['tautology_check_stage2_score']['within_pair_auc']) < 1e-12, \
        'the stage-2 score is not perfectly anti-separating; the ordering is wrong'
    c1['position_space_map5_reproduces'] = True
    c1['partition_accounts_for_every_listed_hand'] = True
    c1['stage2_tautology_within_pair_auc'] = 0.0

    # ------------------------------------------------------ printed answers
    log('\n' + '-' * 100)
    log('  Q1  WHERE DO THE FNs SIT IN THE STAGE-2 ORDER?  (salt audit-trunc-v1)')
    q1 = head['q1_fn_rank']
    log(f"      {q1['fn_total']} FNs; median rank {q1['fn_rank_median']}, mean {q1['fn_rank_mean']}; "
        f"uniform-null median 13.0, uniform-null count per rank {q1['null_uniform_expected_per_rank']}")
    log('      rank : ' + ' '.join(f'{k:>4}' for k in range(6, 21)))
    log('      count: ' + ' '.join(f"{q1['fn_rank_histogram'][str(k)]:>4}" for k in range(6, 21)))
    log(f"      ranks 6-10 hold {q1['fn_in_6_to_10']} of {q1['fn_total']} "
        f"({q1['fn_in_6_to_10'] / max(1, q1['fn_total']):.1%}); rank 6 alone holds {q1['fn_at_rank_6']}")
    log(f"      best FN rank per pair: median {q1['best_fn_rank_median']}, "
        f"histogram {json.dumps(q1['best_fn_rank_per_pair_histogram'])}")
    log(f"      FP ranks inside our five: {json.dumps(q1['fp_rank_histogram'])} "
        f"(median {q1['fp_rank_median']})")
    for f in FAMILIES:
        v = q1['by_family'][f]
        log(f"      {f:<24} n_FN {v['n_FN']:<4} median {v['median_rank']}  mean {v['mean_rank']}  "
            f"share in 6-10 {v['share_in_6_to_10']}")

    log('\n  Q2  IS THE LOSS CONCENTRATED OR UNIFORM?')
    q2 = head['q2_overlap']
    log(f"      per-pair overlap: {json.dumps(q2['overlap_histogram'])}  mean {q2['mean']} "
        f"median {q2['median']} sd {q2['sd']}")
    log(f"      unstratified null (hypergeometric): {json.dumps(q2['null_unstratified_overlap_histogram'])}")
    for f in FAMILIES:
        v = q2['by_family'][f]
        log(f"      {f:<24} n={v['pairs']:<4} mean {v['mean_overlap']:.4f}  oracle {v['oracle_overlap_top20']:.4f}  "
            f"FN/pair {v['mean_FN']:.3f}  miss/pair {v['mean_retrieval_miss']:.3f}  "
            f"hist {json.dumps(v['histogram'])}")

    log('\n  Q3  IS THE FN DISTINGUISHABLE FROM THE FP THAT DISPLACED IT?  (paired, within pair)')
    q3 = head['q3_paired_columns']
    log(f"      {q3['n_columns_tested']} columns, {q3['pairs_contributing']} pairs contribute both a "
        f"FN and a FP;  real max |within-pair AUC - .5| = {q3['real_max_abs_dev_from_half']}, "
        f"real mean |dev| = {q3['real_mean_abs_dev']}, columns at BH q < .05: "
        f"{q3['columns_with_bh_q_below_.05']}")
    for kind, v in q3['nulls'].items():
        log(f"      null[{kind:<11}] max |dev| {v['null_max_abs_dev_from_half_mean']} "
            f"(p95 {v['null_max_abs_dev_from_half_p95']})  mean |dev| {v['null_mean_abs_dev']}  "
            f"real-null {v['real_minus_null_max_abs_dev']:+.4f}  family-wise p "
            f"{v['familywise_permutation_p']}  columns above their own null column mean: "
            f"{v['n_columns_beating_their_own_null_column_mean']}")
    log(f"      tautology check, stage-2 score percentile: within-pair AUC "
        f"{q3['tautology_check_stage2_score']['within_pair_auc']} (must be 0.0)")
    log(f"      {'column':<34}{'wpAUC':>8}{'dev':>8}{'dPct':>8}{'FN>':>6}{'FP>':>6}{'signP':>9}{'BHq':>9}")
    for n, v in list(q3['top30_by_abs_deviation'].items())[:20]:
        log(f"      {n[:33]:<34}{v['within_pair_auc']:>8.4f}{v['dev_from_half']:>+8.4f}"
            f"{v['mean_delta_pct_FN_minus_FP']:>+8.4f}{v['pairs_FN_higher']:>6}{v['pairs_FP_higher']:>6}"
            f"{v['sign_test_p']:>9.4f}{v['sign_test_bh_q']:>9.4f}")

    log('\n  Q4  CAN ANYTHING SEPARATE FN FROM FP OUT OF FOLD?  (label 1 = FN/listed, 0 = FP)')
    log(f"      {'partition':<18}{'design':<19}{'wpAUC':>8}{'CI95':>18}{'-stage1':>9}"
        f"{'nullBS':>8}{'r-nBS':>8}{'pBS':>7}{'nullST':>8}{'nullUN':>8}")
    for salt in SALTS:
        q4 = parts[salt]['q4_separability']
        for dn, v in q4['designs'].items():
            ci = f"[{v['within_pair_auc_ci95'][0]:.3f},{v['within_pair_auc_ci95'][1]:.3f}]"
            inc = v.get('within_pair_auc_minus_stage1_only')
            un = v.get('null_unstratified_within_pair_auc')
            log(f"      {salt:<18}{dn:<19}{v['within_pair_auc']:>8.4f}{ci:>18}"
                f"{('%+.4f' % inc) if inc is not None else '       -':>9}"
                f"{v['null_block_swap_within_pair_auc']:>8.4f}"
                f"{v['real_minus_null_block_swap_within_pair_auc']:>+8.4f}"
                f"{v['null_block_swap_perm_p_within_pair_auc']:>7.3f}"
                f"{v['null_stratified_within_pair_auc']:>8.4f}"
                f"{(('%.4f' % un) if un is not None else '-'):>8}")
    log(f"      pooled row-level OOF AUC, salt 1: " + '  '.join(
        f"{dn} {v['pooled_oof_auc']:.4f} {v['pooled_oof_auc_ci95']} (null BS "
        f"{v['null_block_swap_pooled_auc']:.4f})"
        for dn, v in parts[SALTS[0]]['q4_separability']['designs'].items()))

    log('\n  Q5  WHAT WOULD IT BE WORTH?  (ORACLE-CONDITIONED UPPER BOUND)')
    for salt in SALTS:
        q5 = parts[salt]['q5_worth']
        for dn, v in q5['designs'].items():
            log(f"      [{salt:<17}] {dn:<18} overlap {q5['baseline_overlap']:.4f} -> "
                f"{v['oracle_swap_overlap']:.4f} ({v['delta_overlap']:+.4f}, chance "
                f"{v['chance_swap_delta_overlap']:+.4f}, null-BS {v['null_block_swap_delta_overlap']:+.4f})"
                f"   MAP@5 {v['delta_map5']:+.4f}  composite {v['composite_delta_upper_bound']:+.5f}"
                f"  real-null composite {v['composite_real_minus_null']:+.5f}")

    # ------------------------------------------------------ re-salt minima and the verdict
    def mn(path):
        vals = []
        for s in SALTS:
            node = parts[s]
            for k in path:
                node = node[k]
            vals.append(node)
        return vals

    resalt = {'null_used_for_the_gate': 'block_swap (position-matched); the two label-permutation '
                                        'nulls are reported per partition inside `partitions`'}
    for dn in ('pct_all', 'pct_interpretable', 'stage1_only'):
        wp = mn(['q4_separability', 'designs', dn, 'within_pair_auc'])
        rn = mn(['q4_separability', 'designs', dn, 'real_minus_null_block_swap_within_pair_auc'])
        rns = mn(['q4_separability', 'designs', dn, 'real_minus_null_stratified_within_pair_auc'])
        lo = mn(['q4_separability', 'designs', dn, 'within_pair_auc_ci95'])
        pl_ = mn(['q4_separability', 'designs', dn, 'pooled_oof_auc'])
        prn = mn(['q4_separability', 'designs', dn, 'real_minus_null_block_swap_pooled_auc'])
        cmp_ = mn(['q5_worth', 'designs', dn, 'composite_real_minus_null'])
        dov = mn(['q5_worth', 'designs', dn, 'delta_overlap'])
        dab = mn(['q5_worth', 'designs', dn, 'delta_overlap_above_chance'])
        rnh = mn(['q5_worth', 'designs', dn, 'real_minus_null_delta_overlap'])
        resalt[dn] = {
            'within_pair_auc_by_salt': [r4(x) for x in wp], 'min_within_pair_auc': r4(min(wp)),
            'real_minus_null_block_swap_within_pair_auc_by_salt': [r4(x) for x in rn],
            'MIN_real_minus_null_within_pair_auc': r4(min(rn)),
            'real_minus_null_stratified_within_pair_auc_by_salt': [r4(x) for x in rns],
            'MIN_real_minus_null_stratified_within_pair_auc': r4(min(rns)),
            'within_pair_auc_ci95_lower_by_salt': [x[0] for x in lo],
            'min_ci95_lower': r4(min(x[0] for x in lo)),
            'pooled_auc_by_salt': [r4(x) for x in pl_], 'min_pooled_auc': r4(min(pl_)),
            'MIN_real_minus_null_pooled_auc': r4(min(prn)),
            'oracle_swap_delta_overlap_by_salt': [r4(x) for x in dov],
            'MIN_oracle_swap_delta_overlap': r4(min(dov)),
            'MIN_oracle_swap_delta_overlap_above_chance': r4(min(dab)),
            'MIN_oracle_swap_delta_overlap_real_minus_null': r4(min(rnh)),
            'composite_real_minus_null_by_salt': [r5(x) for x in cmp_],
            'MIN_composite_real_minus_null': r5(min(cmp_)),
            'all_salts_ci_lower_above_half': bool(min(x[0] for x in lo) > .5),
            'all_salts_beat_null': bool(min(rn) > 0)}
    for dn in ('pct_all', 'pct_interpretable'):
        inc = mn(['q4_separability', 'designs', dn, 'within_pair_auc_minus_stage1_only'])
        icl = mn(['q4_separability', 'designs', dn, 'within_pair_auc_minus_stage1_only_ci95'])
        resalt[dn]['within_pair_auc_minus_stage1_only_by_salt'] = [r4(x) for x in inc]
        resalt[dn]['MIN_within_pair_auc_minus_stage1_only'] = r4(min(inc))
        resalt[dn]['min_increment_ci95_lower'] = r4(min(x[0] for x in icl))
        resalt[dn]['all_salts_increment_ci_lower_above_zero'] = bool(min(x[0] for x in icl) > 0)
    for kind in ('block_swap', 'stratified'):
        fw = mn(['q3_paired_columns', 'nulls', kind, 'familywise_permutation_p'])
        resalt[f'q3_{kind}_familywise_permutation_p_by_salt'] = [r4(x) for x in fw]
        resalt[f'q3_{kind}_MAX_familywise_permutation_p'] = r4(max(fw))
    resalt['q3_MAX_familywise_permutation_p'] = resalt['q3_block_swap_MAX_familywise_permutation_p']

    bestd = max(('pct_all', 'pct_interpretable'),
                key=lambda dn: resalt[dn]['MIN_real_minus_null_within_pair_auc'])
    R = resalt[bestd]
    separable = bool(R['all_salts_beat_null'] and R['all_salts_ci_lower_above_half'])
    worth = bool(R['MIN_composite_real_minus_null'] is not None
                 and R['MIN_composite_real_minus_null'] >= .007)
    verdict = {
        'ALIVE_or_DEAD': 'ALIVE' if (separable and worth) else 'DEAD',
        'best_design': bestd,
        'separable': separable,
        'clears_public_resolution_0.007': worth,
        'null_used_for_the_gate': 'block_swap (position-matched)',
        'deciding_number': (
            f"paired FN-vs-FP out of fold, design {bestd}: within-pair AUC "
            f"{R['min_within_pair_auc']:.4f} at the worst of the three partitions "
            f"(95 % CI lower bound {R['min_ci95_lower']}); against the position-matched block-swap "
            f"null the same estimator gives real-minus-null "
            f"{R['MIN_real_minus_null_within_pair_auc']:+.4f} at its worst partition, and over the "
            f"trivial stage-1-only model the increment is "
            f"{R['MIN_within_pair_auc_minus_stage1_only']:+.4f} "
            f"(CI lower {R['min_increment_ci95_lower']}).  The ORACLE-CONDITIONED swap converts the "
            f"real model to {R['MIN_oracle_swap_delta_overlap']:+.4f} overlap per pair "
            f"({R['MIN_oracle_swap_delta_overlap_above_chance']:+.4f} above the chance swap, "
            f"{R['MIN_oracle_swap_delta_overlap_real_minus_null']:+.4f} above the matched null) and "
            f"composite {R['MIN_composite_real_minus_null']:+.5f} real-minus-null, against a "
            f"+-.007 public resolution"),
        'criteria': {'min_real_minus_null_within_pair_auc_gt_0': bool(R['all_salts_beat_null']),
                     'all_salt_ci_lower_above_half': bool(R['all_salts_ci_lower_above_half']),
                     'min_composite_real_minus_null_ge_.007': worth,
                     'increment_over_stage1_only_ci_above_zero_on_all_salts':
                         bool(R['all_salts_increment_ci_lower_above_zero']),
                     'q3_familywise_p_le_.05_on_all_salts':
                         bool(resalt['q3_MAX_familywise_permutation_p'] <= .05)}}

    hashes_out = check_protected()
    res = {'author': AUTHOR, 'step': STEP, 'training_only': True, 'held_folds_touched': False,
           'held_folds_note': GATE['held_folds'],
           'production_branch': 'none: diagnostic only, nothing is proposed for submission',
           'question': GATE['question'], 'gate': GATE,
           'prediction_file': str(PRED_FILE).replace('\\', '/'),
           'prediction_written_before_any_fit': True,
           'protected_hashes_entry': hashes_in, 'protected_hashes_exit': hashes_out,
           'protected_hashes_identical': hashes_in == hashes_out,
           'population': {'pairs': len(truths), 'tables': len(set(ptab.values())), 'hands': d.height,
                          'listed_hands': int((d['relevance'] == 1).sum()),
                          'analysis_columns': len(cols), 'interpretable_columns': len(interp),
                          'ranker_columns': len(rcols)},
           'controls': {'1_reproduction': c1,
                        '2_matched_null': {'kinds': ['block_swap (PRIMARY, position-matched)',
                                                     'stratified label permutation',
                                                     'unstratified label permutation (salt '
                                                     'audit-trunc-v1 only)'],
                                           'paired_column_replicates': NPERM_STAT,
                                           'model_replicates': NPERM_MODEL,
                                           'what': GATE['matched_control']},
                        '3_resalt': {'salts': SALTS, 'rule': 'the MINIMUM across partitions is the '
                                                             'reported result'}},
           'partitions': parts, 'resalt_minima': resalt, 'verdict': verdict,
           'seconds': round(time.time() - t0)}
    write_json(RES_FILE, res)
    log('\n' + '=' * 100)
    log(f"  VERDICT: {verdict['ALIVE_or_DEAD']}  --  {verdict['deciding_number']}")
    log(f"  criteria: {json.dumps(verdict['criteria'])}")
    log(f"  protected hashes identical at entry and exit: {res['protected_hashes_identical']}")
    log(f"  wrote {RES_FILE}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
