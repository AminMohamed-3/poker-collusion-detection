"""Step 90 [C].  WHY DOES STAGE-1 RETRIEVAL FAIL ON coordinated_isolation?

author: claude   (DIAGNOSIS ONLY -- nothing here is a lever, nothing is proposed for submission)

THE HOLE THIS FILLS
Step 87 partitioned the evidence chain and found that the 794 listed hands split into
540 TP / 212 FN / 42 RETRIEVAL MISSES, and that the misses are not spread evenly:

    family                 pairs   TP   FN   miss  listed  recall@20   miss/pair   FN/pair
    directed_transfer         72  249   98      8     355     .9775       .1111     1.361
    soft_play                 47  156   63      5     224     .9777       .1064     1.340
    coordinated_isolation     43  135   51     29     215     .8651       .6744     1.186

69 % of the retrieval misses sit on 27 % of the pairs, and coordinated_isolation has the FEWEST
false negatives per pair -- conditional on the right hand reaching the top 20, CI *selection* is the
best of the three.  CI is therefore bottlenecked at STAGE 1, not at stage 2.  Five selection-side
levers have died on it.  Nobody has looked at the missed hands themselves.

THE QUESTION
Why does stage-1 retrieval fail on coordinated_isolation, and is the failure describable?
A clean characterisation is a success.  "No describable signature" is ALSO a success and is
reported as such; the admission rule below is written before any number exists and is not edited.

WHAT A RETRIEVAL MISS IS
`oof` is the frozen stage-1 per-hand score; `oof_rank` is its within-pair descending rank.  The
candidate list is oof_rank <= 20.  A retrieval miss is a listed hand with oof_rank > 20: it never
enters the candidate list and can never be selected, whatever stage 2 does.  The miss set depends on
NOTHING that this study fits.

THE THREE CONTRASTS  (descriptors are compared with Cliff's delta = 2*AUC - 1)
    A  retrievability   CI-missed (29)      vs CI-retrieved (186)
    B  family           CI-missed (29)      vs DT/SP-missed (13)          <- PRIMARY
    C  family baseline  CI-retrieved (186)  vs DT/SP-retrieved (566)
B isolates family among the hands that were missed; A isolates retrievability inside CI; C says how
much of any B effect is just "CI listed hands look different from DT/SP listed hands".

PRICING THE SEARCH
Every descriptor is evaluated in two spaces (raw value, and within-pair average-rank percentile over
ALL the pair's hands), in two pre-declared blocks (interpretable gameplay columns; the 377 stage-1
ranker columns).  The null is a WITHIN-FAMILY RELABELLING of which listed hands were missed that
preserves the per-family miss counts exactly (29 of 215 CI, 8 of 355 DT, 5 of 224 SP), so the family
composition of every group is fixed and only the association between being missed and the descriptor
is destroyed.  The statistic is the MAXIMUM |delta| over all descriptors of a block; the family-wise
permutation p compares the real maximum against the same maximum on 2,000 null replicates.  Raw
per-descriptor permutation p and BH-FDR q are reported beside it.  NOTHING is called separating
unless the block's family-wise p <= .05.

SALTS
The three pool partitions re-salt the STAGE-2 fold hash only.  `oof` is a stored column and is never
refit here, so the miss set and every descriptor contrast are IDENTICAL across the three salts by
construction; the script asserts that identity rather than claiming it, and the three permutation
runs differ only in their Monte-Carlo seed.  Stage-2-dependent numbers (TP/FN, the CI conversion
rate, MAP@5, the realistic ceiling) genuinely do move with the salt and are reported per salt.

PROTOCOL (any violation invalidates the run)
  TRAINING / AUDIT ONLY.  Held folds 0, 1, 2 are never read for truth.  build_d0() -> load_fold(0)
  supplies FEATURE COLUMNS ONLY on audit rows, exactly as steps 74/76/77/84/87 do; the audit tables
  are asserted disjoint from all three held pools at entry.  held_folds_touched is false.
  check_protected() at entry and exit; both hash dicts recorded.
  The pre_registration block is written to prediction-ci-retrieval.json BEFORE any fit and the
  artefact's copy is asserted byte-identical to it.
  Artefacts are append-only: the script refuses to overwrite its output.
  Every ceiling number is labelled oracle_diagnostic_not_a_result.
  NOTHING is proposed for submission.  No outputs_* directory is written.  No Kaggle contact.

Output: Claude Version/experiments/ci_retrieval_C_20260913/
        prediction-ci-retrieval.json          (before any fit)
        audit-C-ci-retrieval-diagnosis.json   (the study)
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from scipy.stats import rankdata

from evidence_aug.fold import fold_pools
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, ORIGINAL, check_protected, read_json, write_json
from evidence_aug.step5_evaluate import ap5
from evidence_aug.step14_audit_truncation import OUT as AUDIT_OUT, TOP
from evidence_aug.step74_length_shift import CHRONO3, S1, build_d0, designs_of, frame_at
from evidence_aug.step76_C_order_vs_set import ranked_lists
from evidence_aug.step77_C_label_process import add_derived
from evidence_aug.step84_C_setlevel_exhaustive import fit_pointwise, resplit

# the 26 gameplay columns add_derived() produces; the same list step 87 tested, copied here so this
# script does not import step 87's module just for a constant
DERIVED = ['n_board', 'pot_bb_raw', 'type_showdown', 'either_member_sd', 'max_stack_bb',
           'min_stack_bb', 'max_put_bb', 'min_put_bb', 'net_ben_bb', 'net_oth_bb', 'chen_ben',
           'chen_oth', 'put_ben_bb', 'put_oth_bb', 'oth_folded', 'flow_to_ben_bb', 'xfer_to_ben_bb',
           'xfer_abs_bb', 'winner_is_ben', 'loser_put_bb', 'others_net_bb', 'pot_over_min_stack',
           'chen_gap_ben', 'folder_chen', 'gap_prev_s', 'gap_next_s']

AUTHOR = 'claude'
STEP = 90
OUTDIR = CV / 'experiments' / 'ci_retrieval_C_20260913'
PRED_FILE = OUTDIR / 'prediction-ci-retrieval.json'
RES_FILE = OUTDIR / 'audit-C-ci-retrieval-diagnosis.json'
DATA = ORIGINAL / 'data'

CI = 'coordinated_isolation'
SALTS = ['audit-trunc-v1', 'audit-resplit-b', 'audit-resplit-c']
NPERM = 2000
NBOOT = 1000
TOPCI = 15                    # descriptors per (block, comparison) that get a cluster bootstrap CI
PERM_SEED = 20260913
BOOT_SEED = 20260913 + 7

# stored step-87 / step-74 targets that this study must reproduce before it is allowed to proceed
STORED = {
    'pairs': 162, 'tables': 104, 'hands': 19325, 'listed': 794,
    'listed_by_family': {'directed_transfer': 355, 'soft_play': 224, CI: 215},
    'pairs_by_family': {'directed_transfer': 72, 'soft_play': 47, CI: 43},
    'miss_by_family': {'directed_transfer': 8, 'soft_play': 5, CI: 29},
    'miss_total': 42,
    'recall20_by_family': {'directed_transfer': 0.9775, 'soft_play': 0.9777, CI: 0.8651},
    'recall20_all': 0.9471,
    'TP': 540, 'FN': 212, 'FP': 270, 'TN': 2218,
    'map5': 0.6159, 'mean_overlap': 3.3333, 'oracle_overlap_top20': 4.642,
    'fn_by_family': {'directed_transfer': 98, 'soft_play': 63, CI: 51},
    'tp_by_family': {'directed_transfer': 249, 'soft_play': 156, CI: 135},
}

PRE_REGISTRATION = {
    'written_before_any_fit': True,
    'question': 'why does stage-1 retrieval fail on coordinated_isolation, and is the failure '
                'describable?',
    'this_is_a_diagnosis_not_a_lever': 'no method is proposed, nothing is submitted, and every '
                                       'conversion to composite is labelled a ceiling',
    'gate_0_reproduction': 'the study STOPS unless the miss set reproduces EXACTLY: 42 retrieval '
                           'misses split 8 directed_transfer / 5 soft_play / 29 '
                           'coordinated_isolation over listed totals 355 / 224 / 215, on 162 pairs, '
                           '104 tables, 19,325 hands, 794 listed hands.  An approximate '
                           'reconstruction is not acceptable and is itself the finding.',
    'definition_of_a_retrieval_miss': 'a listed hand whose frozen stage-1 within-pair rank '
                                      '(oof_rank) exceeds the candidate budget TOP = 20',
    'comparisons': {
        'A_retrievability': 'CI-missed (n=29) vs CI-retrieved (n=186)',
        'B_family_PRIMARY': 'CI-missed (n=29) vs DT/SP-missed (n=13)',
        'C_family_baseline': 'CI-retrieved (n=186) vs DT/SP-retrieved (n=566)'},
    'primary_comparison': 'B_family_PRIMARY',
    'effect_size': "Cliff's delta = 2 * AUC - 1, ties at .5, two-sided; Hedges g reported beside it",
    'descriptor_spaces': ['raw', 'within_pair_avg_rank_percentile_over_all_hands_of_the_pair'],
    'descriptor_blocks': {
        'interpretable': 'the 70 structural columns + 3 chronology columns + the 26 step-77 derived '
                         'gameplay columns + big_blind, final_pot, players_dealt, '
                         'players_at_showdown, t_rank, t_n',
        'ranker': 'the 377 stage-1 ranker columns (players_at_showdown excluded, it is already in '
                  'the interpretable block)'},
    'excluded_from_every_test': ['oof', 'oof_rank'],
    'why_excluded': 'a retrieval miss is DEFINED by oof_rank > 20, so both stage-1 score columns '
                    'are tautological; they are reported separately as a tautology check',
    'null': 'within-family relabelling of which listed hands are missed, preserving the per-family '
            'miss counts exactly (29 of 215 CI, 8 of 355 DT, 5 of 224 SP).  Under this null the '
            'family composition of every compared group is fixed, so comparison B is tested '
            'against the family difference that already exists among listed hands rather than '
            'against zero.',
    'n_permutation_replicates': NPERM,
    'search_statistic': 'the MAXIMUM |Cliff delta| over every descriptor of a block (both spaces '
                        'pooled), compared against the same maximum on the null replicates',
    'familywise_p': '(1 + #{null max >= real max}) / (1 + NPERM)',
    'ADMISSION_RULE': 'a descriptor is reported as SEPARATING for a comparison only if ALL of: '
                      '(1) the block family-wise permutation p <= .05; (2) the descriptor BH-FDR '
                      'q <= .05 within its (block, comparison) cell; (3) its cluster-bootstrap '
                      '(by table) 95 % CI on Cliff delta excludes 0; (4) (1)-(3) hold on all three '
                      'salt runs.  If the family-wise p exceeds .05 NOTHING in that block is '
                      'reported as separating, however large a single delta looks.',
    'no_subgroup_rescue': 'if the pre-registered comparisons are null, the artefact records a null. '
                          'No post-hoc subgroup, no re-thresholding, no relabelling of a null as '
                          'suggestive.',
    'bootstrap': {'cluster_unit': 'table_id (the pool unit, 104 audit tables)', 'replicates': NBOOT,
                  'computed_for': f'the top {TOPCI} descriptors by |delta| in each '
                                  '(block, comparison) cell'},
    'salts': SALTS,
    'salt_expectation_declared_in_advance': 'the salts re-hash the STAGE-2 fold assignment only. '
                                            'oof / oof_rank are stored columns and are not refit, '
                                            'so the miss set and every descriptor contrast MUST be '
                                            'identical across the three salts; the script asserts '
                                            'that identity.  The three permutation runs differ only '
                                            'in Monte-Carlo seed and are labelled as such, NOT as '
                                            'three independent partitions.  Stage-2 quantities '
                                            '(TP/FN, conversion rate, MAP@5, realistic ceiling) do '
                                            'move with the salt and are reported per salt.',
    'ceiling_rules': {
        'target_recall': 'the pooled DT/SP hand-level recall@20, (347 + 219) / (355 + 224)',
        'which_hands_are_recovered': 'the shallowest-ranked misses first (smallest oof_rank), '
                                     'because a retrieval improvement recovers near-threshold hands '
                                     'before far-tail ones',
        'displacement': 'each recovered hand displaces the lowest-oof UNLISTED candidate of its '
                        'pair, so the candidate budget stays at 20',
        'perfect_selection': 'the recovered hands are placed at the front of the pair\'s five',
        'realistic_conversion': 'each recovered hand independently reaches the five with the '
                                'observed CI conversion rate TP_CI / (TP_CI + FN_CI); 2,000 draws',
        'realistic_counterfactual_admission': 'the recovered hands enter the candidate set and are '
                                              're-sorted by the PRODUCTION stage-2 score, which is '
                                              'the fitted model\'s own opinion of them',
        'label': 'every one of these is stored under oracle_diagnostic_not_a_result and is NOT a '
                 'claim about what any method would achieve'},
    'coupling_tax_caveat': 'changing stage 1 has historically cost -11 correct final picks out of '
                           '-16 through re-ranking inside an unchanged top-20, because stage 2 was '
                           'fitted around the production stage-1 rank distribution.  Any ceiling '
                           'below is computed WITHOUT that tax and is therefore optimistic.',
    'held_folds': 'folds 0, 1, 2 are never read for truth; the audit tables are asserted disjoint '
                  'from all three held pools at entry; held_folds_touched is false by construction',
}


def log(*a):
    print(*a, flush=True)


def r4(x):
    return round(float(x), 4)


def r5(x):
    return round(float(x), 5)


def bh(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for k in range(n - 1, -1, -1):
        i = order[k]
        prev = min(prev, p[i] * n / (k + 1))
        q[i] = min(prev, 1.0)
    return q


def hedges_g(a, b):
    """standardised mean difference with the small-sample correction; 0 when both groups degenerate"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 1e-300:
        return 0.0
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    j = 1.0 - 3.0 / (4 * (na + nb) - 9)
    return float(d * j)


def cmp_matrix(a, b):
    """M[i, j] = 1 if a_i > b_j, .5 if equal, 0 otherwise -- the Mann-Whitney kernel."""
    return (a[:, None] > b[None, :]).astype(np.float64) + .5 * (a[:, None] == b[None, :])


def wdelta(M, W1, W0):
    """weighted Cliff delta for many weight replicates at once: rows W1 (r, n1), cols W0 (r, n0)."""
    num = (W1 @ M * W0).sum(1)
    den = W1.sum(1) * W0.sum(1)
    out = np.full(len(num), np.nan)
    ok = den > 0
    out[ok] = 2.0 * num[ok] / den[ok] - 1.0
    return out


SMOKE_DISCLOSURE = (
    'FULL DISCLOSURE.  Before this production run a REDUCED SMOKE RUN of the identical code path '
    '(NPERM=40, NBOOT=60, TOPCI=3, salt audit-trunc-v1 only) was executed with OUTDIR redirected to '
    'the session scratchpad, to validate the code before spending the real budget.  Its output was '
    'written to the scratchpad and is not part of this artefact.  The PRE_REGISTRATION constant was '
    'authored in full BEFORE that smoke run and has NOT been edited since -- no threshold, no '
    'comparison, no admission rule was changed after any number existed.  Two code fixes were made '
    'after the smoke run and neither touches an admission rule: (1) the tautology-check line for '
    'the two excluded stage-1 columns used nS*nS/2 instead of the Mann-Whitney offset nS(nS+1)/2 '
    'and returned 1.0054 for oof_rank, which is outside the range of Cliff delta; the tested '
    'statistics already used the correct offset.  (2) the ceiling deltas were rounded to 4 decimals '
    'and one of them printed as -0.0; they are now rounded to 5.  For transparency the smoke run\'s '
    'family-wise permutation p values, at a resolution of 1/41, were: A/interpretable .1220, '
    'A/ranker .0244, B/interpretable .7073, B/ranker .7805, C/interpretable .4878, C/ranker .1220.')


def main():
    t0 = time.time()
    provenance = [SMOKE_DISCLOSURE]
    hashes_in = check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- entry guards
    audit_tables = set(pl.read_parquet(AUDIT_OUT / 'audit-oof.parquet')['table_id'].unique().to_list())
    held = {}
    for f in (0, 1, 2):
        pool = set(fold_pools(f))
        ov = sorted(audit_tables & pool)
        held[f'fold{f}'] = {'pool_tables': len(pool), 'overlap_with_audit_tables': len(ov)}
        assert not ov, f'AUDIT / HELD OVERLAP IS NOT EMPTY for fold {f}: {ov[:5]}'
    log(f'  entry guard: {len(audit_tables)} audit tables, overlap with held folds 0/1/2 = 0/0/0')

    pred_doc = {'author': AUTHOR, 'step': STEP,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'pre_registration': PRE_REGISTRATION, 'stored_reproduction_targets': STORED}
    if PRED_FILE.exists():
        old = read_json(PRED_FILE)
        assert old['pre_registration'] == PRE_REGISTRATION, \
            'the prediction file exists with a DIFFERENT pre_registration block; it is append-only'
        log(f'  prediction file already present and identical: {PRED_FILE}')
        provenance.append('prediction-ci-retrieval.json already existed and matched byte for byte; '
                          'it was not rewritten')
    else:
        write_json(PRED_FILE, pred_doc)
        log(f'  wrote the pre_registration BEFORE any fit: {PRED_FILE}')
    assert not RES_FILE.exists(), f'{RES_FILE} exists; artefacts are append-only, refusing to overwrite'

    log('=' * 100)
    log('STEP 90 [C].  WHY DOES STAGE-1 RETRIEVAL FAIL ON coordinated_isolation?')
    log('=' * 100)

    # -------------------------------------------------------------- the frame (identical to step 87)
    d0, struct, rcols = build_d0()
    designs2 = designs_of(struct, rcols)
    d = frame_at(d0, 1.0)
    hands = (pl.scan_parquet(DATA / 'hands.parquet')
             .select('hand_id', 'big_blind', 'final_pot', 'board_cards', 'players_dealt')
             .filter(pl.col('hand_id').is_in(d['hand_id'].implode())).collect())
    n0 = d.height
    d = add_derived(d.join(hands, on='hand_id', validate='m:1'))
    assert d.height == n0, 'hands join changed the row count'
    d = d.sort('pair_id', 'hand_id')

    pids_all = d['pair_id'].to_numpy()
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    rel = d['relevance'].to_numpy().astype(bool)
    orank = d['oof_rank'].to_numpy().astype(np.int64)
    fam = d['behavior_family'].to_numpy()
    tab = d['table_id'].to_numpy()

    population = {'pairs': len(truths), 'tables': len(set(ptab.values())), 'hands': d.height,
                  'listed_hands': int(rel.sum())}
    log(f"  frame {population['hands']} rows, {population['pairs']} pairs, "
        f"{population['tables']} tables, {population['listed_hands']} listed hands")

    # ==============================================================================================
    # 1.  REPRODUCE THE MISS SET EXACTLY  (this needs no model at all)
    # ==============================================================================================
    miss = rel & (orank > TOP)
    got = rel & (orank <= TOP)
    repro = {'definition': 'listed hand with oof_rank > 20',
             'miss_total': int(miss.sum()), 'listed_total': int(rel.sum()),
             'pairs': population['pairs'], 'tables': population['tables'],
             'hands': population['hands'], 'by_family': {}}
    for f in FAMILIES:
        m = fam == f
        li = int((rel & m).sum())
        mi = int((miss & m).sum())
        repro['by_family'][f] = {
            'pairs': int(len(set(pids_all[m]))), 'listed': li, 'retrieval_miss': mi,
            'listed_in_top20': li - mi, 'recall_at_20': r4((li - mi) / li),
            'miss_per_pair': r4(mi / len(set(pids_all[m])))}
    repro['recall_at_20_all'] = r4(got.sum() / rel.sum())

    checks = []
    checks.append(('pairs', repro['pairs'], STORED['pairs']))
    checks.append(('tables', repro['tables'], STORED['tables']))
    checks.append(('hands', repro['hands'], STORED['hands']))
    checks.append(('listed', repro['listed_total'], STORED['listed']))
    checks.append(('miss_total', repro['miss_total'], STORED['miss_total']))
    checks.append(('recall20_all', repro['recall_at_20_all'], STORED['recall20_all']))
    for f in FAMILIES:
        checks.append((f'{f}.listed', repro['by_family'][f]['listed'], STORED['listed_by_family'][f]))
        checks.append((f'{f}.pairs', repro['by_family'][f]['pairs'], STORED['pairs_by_family'][f]))
        checks.append((f'{f}.miss', repro['by_family'][f]['retrieval_miss'], STORED['miss_by_family'][f]))
        checks.append((f'{f}.recall20', repro['by_family'][f]['recall_at_20'], STORED['recall20_by_family'][f]))
    bad = [(k, a, b) for k, a, b in checks if a != b]
    repro['checks'] = {k: {'reproduced': a, 'stored': b, 'equal': a == b} for k, a, b in checks}
    repro['REPRODUCED_EXACTLY'] = not bad
    log('\n  1.  REPRODUCTION OF THE MISS SET')
    for f in FAMILIES:
        v = repro['by_family'][f]
        log(f"      {f:<24} pairs {v['pairs']:>3}  listed {v['listed']:>4}  miss {v['retrieval_miss']:>3}"
            f"  recall@20 {v['recall_at_20']:.4f}  miss/pair {v['miss_per_pair']:.4f}")
    log(f"      TOTAL                    pairs {repro['pairs']}  listed {repro['listed_total']}  "
        f"miss {repro['miss_total']}  recall@20 {repro['recall_at_20_all']:.4f}")
    if bad:
        log('\n  *** THE MISS SET DID NOT REPRODUCE.  STOPPING.  Mismatches:')
        for k, a, b in bad:
            log(f'      {k}: reproduced {a}, stored {b}')
    assert not bad, f'GATE 0 FAILED, the miss set did not reproduce exactly: {bad}'
    log('      GATE 0 PASSED: the miss set reproduces exactly.')

    # ==============================================================================================
    # 2.  WHERE IN THE STAGE-1 RANKING DO THE MISSES SIT?
    # ==============================================================================================
    nh = d['n_hands'].to_numpy().astype(float)
    oof = d['oof'].to_numpy().astype(float)
    # the pair's 20th-best oof score = the admission threshold (every pair has >= 57 hands)
    thr_col = (d.select('pair_id', 'oof', 'oof_rank')
               .filter(pl.col('oof_rank') == TOP).select('pair_id', pl.col('oof').alias('thr20')))
    thr = dict(thr_col.iter_rows())
    assert len(thr) == population['pairs'], 'a pair has no rank-20 hand'
    gap = np.array([oof[i] - thr[pids_all[i]] for i in range(len(oof))])

    q_rank = {'what': 'the STAGE-1 within-pair rank of every retrieval miss (rank 21 is one slot '
                      'outside the budget)', 'by_family': {}}
    for f in FAMILIES:
        sel = miss & (fam == f)
        r = np.sort(orank[sel])
        q_rank['by_family'][f] = {
            'n_miss': int(sel.sum()),
            'ranks_sorted': [int(x) for x in r],
            'median_rank': float(np.median(r)) if len(r) else None,
            'mean_rank': r4(r.mean()) if len(r) else None,
            'max_rank': int(r.max()) if len(r) else None,
            'histogram_by_band': {'21-25': int(((r >= 21) & (r <= 25)).sum()),
                                  '26-30': int(((r >= 26) & (r <= 30)).sum()),
                                  '31-40': int(((r >= 31) & (r <= 40)).sum()),
                                  '41-60': int(((r >= 41) & (r <= 60)).sum()),
                                  '61+': int((r > 60).sum())},
            'share_in_21_to_30': r4(((r >= 21) & (r <= 30)).sum() / len(r)) if len(r) else None,
            'rank_over_n_hands_median': r4(np.median(orank[sel] / nh[sel])) if len(r) else None,
            'oof_gap_below_the_20th_median': r5(np.median(gap[sel])) if len(r) else None,
            'oof_gap_below_the_20th_mean': r5(gap[sel].mean()) if len(r) else None}
    rall = np.sort(orank[miss])
    q_rank['all'] = {'n_miss': int(miss.sum()), 'median_rank': float(np.median(rall)),
                     'mean_rank': r4(rall.mean()), 'max_rank': int(rall.max()),
                     'ranks_sorted': [int(x) for x in rall],
                     'full_histogram': {str(k): int((rall == k).sum())
                                        for k in range(21, int(rall.max()) + 1)
                                        if int((rall == k).sum()) > 0}}
    q_rank['dt_sp_pooled'] = {
        'n_miss': int((miss & (fam != CI)).sum()),
        'median_rank': float(np.median(orank[miss & (fam != CI)])),
        'max_rank': int(orank[miss & (fam != CI)].max()),
        'share_in_21_to_30': r4(((orank[miss & (fam != CI)] >= 21) &
                                 (orank[miss & (fam != CI)] <= 30)).sum() /
                                int((miss & (fam != CI)).sum()))}
    log('\n  2.  WHERE DO THE MISSES SIT IN THE STAGE-1 ORDER?')
    for f in FAMILIES:
        v = q_rank['by_family'][f]
        log(f"      {f:<24} n {v['n_miss']:>2}  median rank {v['median_rank']:>5}  max {v['max_rank']:>3}"
            f"  bands 21-25/26-30/31-40/41-60/61+ = "
            f"{'/'.join(str(v['histogram_by_band'][k]) for k in ['21-25','26-30','31-40','41-60','61+'])}")

    # per-pair concentration of the misses
    conc = {}
    for f in FAMILIES:
        m = fam == f
        pp = {}
        for p in sorted(set(pids_all[m])):
            pp[p] = int((miss & (pids_all == p)).sum())
        c = np.array(list(pp.values()))
        conc[f] = {'pairs': len(c), 'pairs_with_a_miss': int((c > 0).sum()),
                   'share_of_pairs_with_a_miss': r4((c > 0).sum() / len(c)),
                   'max_misses_on_one_pair': int(c.max()),
                   'histogram': {str(k): int((c == k).sum()) for k in range(int(c.max()) + 1)}}
    q_rank['per_pair_concentration'] = conc

    # ==============================================================================================
    # 3.  STRUCTURAL CONFOUNDS THAT NEED NO MODEL
    # ==============================================================================================
    per_pair = d.select('pair_id', 'behavior_family', 'n_hands').unique(subset=['pair_id'])
    lcount = (d.filter(pl.col('relevance') == 1).group_by('pair_id').len()
              .rename({'len': 'n_listed'}))
    per_pair = per_pair.join(lcount, on='pair_id', how='left').with_columns(pl.col('n_listed').fill_null(0))
    conf = {'what': 'pair-level quantities that would make a fixed top-20 budget bite harder on one '
                    'family without any per-hand signature at all', 'by_family': {}}
    for f in FAMILIES:
        g = per_pair.filter(pl.col('behavior_family') == f)
        conf['by_family'][f] = {
            'pairs': g.height,
            'n_hands_median': float(g['n_hands'].median()), 'n_hands_mean': r4(g['n_hands'].mean()),
            'n_listed_median': float(g['n_listed'].median()), 'n_listed_mean': r4(g['n_listed'].mean()),
            'listed_over_n_hands_mean': r4((g['n_listed'] / g['n_hands']).mean()),
            'pairs_with_more_than_20_listed': int((g['n_listed'] > TOP).sum())}
    log('\n  3.  PAIR-LEVEL CONFOUNDS')
    for f in FAMILIES:
        v = conf['by_family'][f]
        log(f"      {f:<24} n_hands median {v['n_hands_median']:>6.1f} mean {v['n_hands_mean']:>7.2f}"
            f"   listed/pair median {v['n_listed_median']:>4.1f} mean {v['n_listed_mean']:>5.2f}"
            f"   listed/n_hands {v['listed_over_n_hands_mean']:.4f}")

    # ==============================================================================================
    # 4.  DESCRIPTORS
    # ==============================================================================================
    extra = ['big_blind', 'final_pot', 'players_dealt', 'players_at_showdown', 't_rank', 't_n']
    block1 = struct + CHRONO3 + DERIVED + extra
    block2 = [c for c in rcols if c not in block1]
    assert len(block1) == len(set(block1)), 'duplicate interpretable descriptor'
    for c in block1 + block2:
        assert c in d.columns, f'descriptor absent from the frame: {c}'
    for c in S1:
        assert c not in block1 and c not in block2, f'{c} is tautological and must not be tested'
    log(f'\n  4.  DESCRIPTORS: {len(block1)} interpretable + {len(block2)} ranker, in 2 spaces '
        f'= {2 * (len(block1) + len(block2))} tested columns; {S1} excluded as tautological')

    allcols = block1 + block2
    V = d.select(allcols).to_numpy().astype(np.float64)
    nonfinite = {}
    if not np.isfinite(V).all():
        for j, c in enumerate(allcols):
            k = int((~np.isfinite(V[:, j])).sum())
            if k:
                nonfinite[c] = k
                finite = V[np.isfinite(V[:, j]), j]
                fill = (finite.min() - 1.0) if len(finite) else 0.0
                V[~np.isfinite(V[:, j]), j] = fill
        provenance.append(f'{len(nonfinite)} descriptor columns contained non-finite values; each '
                          'was filled with (column finite minimum - 1), which is rank-preserving '
                          f'and places them at the bottom of the order.  Counts: {nonfinite}')
        log(f'      {len(nonfinite)} columns had non-finite values, filled below the minimum '
            f'(rank-preserving); recorded in provenance')

    # within-pair average-rank percentile over ALL the pair's hands
    _, starts = np.unique(pids_all, return_index=True)
    starts = np.sort(starts)
    ends = np.r_[starts[1:], len(pids_all)]
    P = np.empty_like(V)
    for s, e in zip(starts, ends):
        P[s:e] = (rankdata(V[s:e], method='average', axis=0) - .5) / (e - s)
    names = [f'{c}|raw' for c in allcols] + [f'{c}|pct' for c in allcols]
    X = np.hstack([V, P])
    blockof = np.array(['interpretable'] * len(block1) + ['ranker'] * len(block2))
    blockof = np.concatenate([blockof, blockof])

    # the listed-hand universe, in a fixed order: CI listed first, then DT listed, then SP listed
    ix_ci = np.where(rel & (fam == CI))[0]
    ix_dt = np.where(rel & (fam == 'directed_transfer'))[0]
    ix_sp = np.where(rel & (fam == 'soft_play'))[0]
    ix_od = np.concatenate([ix_dt, ix_sp])                       # the DT/SP listed pool
    m_ci = miss[ix_ci]
    m_od = miss[ix_od]
    assert len(ix_ci) == STORED['listed_by_family'][CI], 'CI listed pool changed'
    assert len(ix_od) == (STORED['listed_by_family']['directed_transfer']
                          + STORED['listed_by_family']['soft_play']), 'DT/SP listed pool changed'
    n_dt, n_sp = len(ix_dt), len(ix_sp)
    k_ci, k_dt, k_sp = int(m_ci.sum()), int(miss[ix_dt].sum()), int(miss[ix_sp].sum())
    assert (k_ci, k_dt, k_sp) == (STORED['miss_by_family'][CI],
                                  STORED['miss_by_family']['directed_transfer'],
                                  STORED['miss_by_family']['soft_play'])
    assert int(m_od.sum()) == k_dt + k_sp

    Xci = np.ascontiguousarray(X[ix_ci])                          # (215, D)
    Xod = np.ascontiguousarray(X[ix_od])                          # (579, D)
    D = X.shape[1]
    del V, P, X                                                   # ~300 MB the study no longer needs
    S_real = np.where(m_ci)[0]                                    # CI missed, indices into 0..214
    R_real = np.where(~m_ci)[0]                                   # CI retrieved
    Dm_real = np.where(m_od)[0]                                   # DT/SP missed, into 0..578
    Dr_real = np.where(~m_od)[0]

    def deltas(S, Dm):
        """all three Cliff deltas for every descriptor, given a CI-missed index set S (29,) and a
        DT/SP-missed index set Dm (13,).  Uses the exact rank identities, no dense matrix."""
        R = np.setdiff1d(np.arange(215), S, assume_unique=False)
        Dr = np.setdiff1d(np.arange(579), Dm, assume_unique=False)
        nS, nR, nDm, nDr = len(S), len(R), len(Dm), len(Dr)
        # A: rows S, cols R, pool = the 215 CI listed.  Mann-Whitney U = sum_{i in S} rank_i -
        # nS(nS+1)/2, using the identity (row sum of the kernel over the whole pool) = rank - .5.
        rk = rankdata(Xci, method='average', axis=0)              # (215, D)
        numA = rk[S].sum(0) - nS * (nS + 1) / 2.0
        dA = 2.0 * numA / (nS * nR) - 1.0
        # B: rows S (CI), cols Dm (DT/SP) -- direct comparison of two small sets
        a = Xci[S]                                                # (nS, D)
        b = Xod[Dm]                                               # (nDm, D)
        numB = ((a[:, None, :] > b[None, :, :]).sum((0, 1))
                + .5 * (a[:, None, :] == b[None, :, :]).sum((0, 1)))
        dB = 2.0 * numB / (nS * nDm) - 1.0
        # C: rows R (CI), cols Dr (DT/SP), by inclusion-exclusion on the full 215 x 579 kernel
        srt = np.sort(Xod, axis=0)
        lo = np.empty((215, D)); hi = np.empty((215, D))
        for j in range(D):
            lo[:, j] = np.searchsorted(srt[:, j], Xci[:, j], side='left')
            hi[:, j] = np.searchsorted(srt[:, j], Xci[:, j], side='right')
        Rcd = lo + .5 * (hi - lo)                                 # (215, D) row sums of the kernel
        T = Rcd.sum(0)
        srtc = np.sort(Xci, axis=0)
        loc = np.empty((579, D)); hic = np.empty((579, D))
        for j in range(D):
            loc[:, j] = np.searchsorted(srtc[:, j], Xod[:, j], side='left')
            hic[:, j] = np.searchsorted(srtc[:, j], Xod[:, j], side='right')
        Ccd = (215 - hic) + .5 * (hic - loc)                      # (579, D) col sums of the kernel
        numC = T - Rcd[S].sum(0) - Ccd[Dm].sum(0) + numB
        dC = 2.0 * numC / (nR * nDr) - 1.0
        return dA, dB, dC, Rcd, Ccd, T, rk

    dA, dB, dC, RCD, CCD, TCD, RKCI = deltas(S_real, Dm_real)
    real = {'A_retrievability': dA, 'B_family_PRIMARY': dB, 'C_family_baseline': dC}
    CMPS = list(real)

    # Hedges g beside the delta
    gvals = {}
    for cname, (g1, g0) in {'A_retrievability': (Xci[S_real], Xci[R_real]),
                            'B_family_PRIMARY': (Xci[S_real], Xod[Dm_real]),
                            'C_family_baseline': (Xci[R_real], Xod[Dr_real])}.items():
        gvals[cname] = np.array([hedges_g(g1[:, j], g0[:, j]) for j in range(D)])

    # tautology check: the two stage-1 columns, reported and not tested
    taut = {}
    for c in S1:
        v = d[c].to_numpy().astype(np.float64)
        vc, vo = v[ix_ci], v[ix_od]
        rkv = rankdata(vc, method='average')
        taut[c] = {
            'A_retrievability': r4(2 * (rkv[S_real].sum() - 29 * 30 / 2.0) / (29 * 186) - 1),
            'B_family_PRIMARY': r4(2 * float(cmp_matrix(vc[S_real], vo[Dm_real]).mean()) - 1),
            'C_family_baseline': r4(2 * float(cmp_matrix(vc[R_real], vo[Dr_real]).mean()) - 1)}
    log(f"      tautology check (NOT tested): oof_rank delta A = {taut['oof_rank']['A_retrievability']}, "
        f"oof delta A = {taut['oof']['A_retrievability']}")

    # ==============================================================================================
    # 5.  THE PERMUTATION NULL, PER SALT (Monte-Carlo replicates; the real deltas do not move)
    # ==============================================================================================
    def perm_deltas(seed):
        """NPERM within-family relabellings; returns (NPERM, D) delta arrays for A, B and C."""
        rng = np.random.default_rng(seed)
        PA = np.empty((NPERM, D)); PB = np.empty((NPERM, D)); PC = np.empty((NPERM, D))
        for t in range(NPERM):
            S = rng.choice(215, k_ci, replace=False)
            Dm = np.concatenate([rng.choice(n_dt, k_dt, replace=False),
                                 n_dt + rng.choice(n_sp, k_sp, replace=False)])
            nS, nR, nDm, nDr = k_ci, 215 - k_ci, len(Dm), 579 - len(Dm)
            numA = RKCI[S].sum(0) - nS * (nS + 1) / 2.0
            PA[t] = 2.0 * numA / (nS * nR) - 1.0
            a = Xci[S]; b = Xod[Dm]
            numB = ((a[:, None, :] > b[None, :, :]).sum((0, 1))
                    + .5 * (a[:, None, :] == b[None, :, :]).sum((0, 1)))
            PB[t] = 2.0 * numB / (nS * nDm) - 1.0
            numC = TCD - RCD[S].sum(0) - CCD[Dm].sum(0) + numB
            PC[t] = 2.0 * numC / (nR * nDr) - 1.0
        return {'A_retrievability': PA, 'B_family_PRIMARY': PB, 'C_family_baseline': PC}

    # cluster bootstrap by table, for the pre-declared top descriptors only
    tab_ci, tab_od = tab[ix_ci], tab[ix_od]
    utabs = np.array(sorted(set(tab)))
    tix_ci = np.searchsorted(utabs, tab_ci)
    tix_od = np.searchsorted(utabs, tab_od)
    rngb = np.random.default_rng(BOOT_SEED)
    Wt = np.zeros((NBOOT, len(utabs)))
    for t in range(NBOOT):
        pick = rngb.integers(0, len(utabs), len(utabs))
        Wt[t] = np.bincount(pick, minlength=len(utabs))
    Wci = Wt[:, tix_ci]                                            # (NBOOT, 215)
    Wod = Wt[:, tix_od]                                            # (NBOOT, 579)

    _bcache = {}

    def boot_ci(j, cname):
        """cluster bootstrap by table.  The real data does not move with the salt, so this is
        cached and computed once per (descriptor, comparison)."""
        if (j, cname) in _bcache:
            return _bcache[(j, cname)]
        x_ci, x_od = Xci[:, j], Xod[:, j]
        if cname == 'A_retrievability':
            v = wdelta(cmp_matrix(x_ci, x_ci), Wci * m_ci, Wci * (~m_ci))
        elif cname == 'B_family_PRIMARY':
            v = wdelta(cmp_matrix(x_ci, x_od), Wci * m_ci, Wod * m_od)
        else:
            v = wdelta(cmp_matrix(x_ci, x_od), Wci * (~m_ci), Wod * (~m_od))
        v = v[np.isfinite(v)]
        out = [r4(np.percentile(v, 2.5)), r4(np.percentile(v, 97.5))] if len(v) > 50 else None
        _bcache[(j, cname)] = out
        return out

    salt_results = {}
    for si, salt in enumerate(SALTS):
        tstart = time.time()
        # the miss set must be identical under this salt: resplit touches `split` only
        ds = resplit(d, salt)
        miss_s = (ds['relevance'].to_numpy().astype(bool)) & (ds['oof_rank'].to_numpy() > TOP)
        assert np.array_equal(miss_s, miss), f'the miss set moved under salt {salt}'
        PERM = perm_deltas(PERM_SEED + 1000 * si)
        out = {'miss_set_identical_to_salt_1': True,
               'permutation_seed': PERM_SEED + 1000 * si, 'comparisons': {}}
        for cname in CMPS:
            rv = real[cname]
            nv = PERM[cname]
            cell = {}
            for bname in ('interpretable', 'ranker'):
                sel = np.where(blockof == bname)[0]
                real_max = float(np.abs(rv[sel]).max())
                null_max = np.abs(nv[:, sel]).max(axis=1)
                fw_p = (1 + int((null_max >= real_max - 1e-12).sum())) / (1 + NPERM)
                praw = (1 + (np.abs(nv[:, sel]) >= np.abs(rv[sel])[None, :] - 1e-12).sum(0)) / (1 + NPERM)
                q = bh(praw)
                order = sel[np.argsort(-np.abs(rv[sel]))]
                topj = order[:TOPCI]
                tops = []
                for j in topj:
                    jj = int(np.where(sel == j)[0][0])
                    ci = boot_ci(j, cname)
                    tops.append({'descriptor': names[j], 'cliffs_delta': r4(rv[j]),
                                 'hedges_g': r4(gvals[cname][j]),
                                 'permutation_p_raw': r4(praw[jj]), 'bh_q': r4(q[jj]),
                                 'cluster_bootstrap_ci95': ci,
                                 'ci_excludes_zero': bool(ci is not None and (ci[0] > 0 or ci[1] < 0)),
                                 'admitted': bool(fw_p <= .05 and q[jj] <= .05 and ci is not None
                                                  and (ci[0] > 0 or ci[1] < 0))})
                cell[bname] = {
                    'n_descriptors': len(sel),
                    'real_max_abs_delta': r4(real_max),
                    'null_max_abs_delta_mean': r4(null_max.mean()),
                    'null_max_abs_delta_p95': r4(np.percentile(null_max, 95)),
                    'familywise_permutation_p': r4(fw_p),
                    'familywise_p_le_.05': bool(fw_p <= .05),
                    'n_bh_q_le_.05': int((q <= .05).sum()),
                    'top_descriptors': tops,
                    'ANY_DESCRIPTOR_ADMITTED': bool(fw_p <= .05 and any(t['admitted'] for t in tops))}
            out['comparisons'][cname] = cell
        out['seconds'] = round(time.time() - tstart)
        salt_results[salt] = out
        log(f"\n  5.  [{salt}]  {out['seconds']} s")
        for cname in CMPS:
            for bname in ('interpretable', 'ranker'):
                c = out['comparisons'][cname][bname]
                t1 = c['top_descriptors'][0]
                log(f"      {cname:<20}{bname:<15} max|delta| {c['real_max_abs_delta']:>7.4f} "
                    f"(null max mean {c['null_max_abs_delta_mean']:.4f})  familywise p "
                    f"{c['familywise_permutation_p']:.4f}  best: {t1['descriptor'][:34]:<34} "
                    f"delta {t1['cliffs_delta']:+.4f}  q {t1['bh_q']:.4f}")

    # cross-salt consistency of the real effects (identical by construction; asserted, not claimed)
    consistency = {
        'real_effects_identical_across_salts': True,
        'why': 'the salts re-hash the STAGE-2 fold assignment only; oof / oof_rank are stored '
               'columns, so the miss set and every descriptor contrast are the same numbers three '
               'times.  The script asserts the miss set is bit-identical under each salt.  The '
               'three familywise p values below differ ONLY in Monte-Carlo seed and must not be '
               'read as three independent partitions.',
        'familywise_p_by_salt': {
            cname: {bname: [salt_results[s]['comparisons'][cname][bname]['familywise_permutation_p']
                            for s in SALTS] for bname in ('interpretable', 'ranker')}
            for cname in CMPS},
        'max_familywise_p_over_salts': {
            cname: {bname: max(salt_results[s]['comparisons'][cname][bname]['familywise_permutation_p']
                               for s in SALTS) for bname in ('interpretable', 'ranker')}
            for cname in CMPS},
        'admitted_on_all_three_salts': {}}
    for cname in CMPS:
        acc = {}
        for bname in ('interpretable', 'ranker'):
            sets = []
            for s in SALTS:
                c = salt_results[s]['comparisons'][cname][bname]
                sets.append({t['descriptor'] for t in c['top_descriptors'] if t['admitted']})
            acc[bname] = sorted(set.intersection(*sets)) if sets else []
        consistency['admitted_on_all_three_salts'][cname] = acc

    # ==============================================================================================
    # 6.  STAGE-2 CONFUSION AND THE CEILING, PER SALT
    # ==============================================================================================
    # hand_id is NOT unique across the frame (18,769 distinct ids over 19,325 rows: one table hand
    # can belong to two suspicious pairs), so every per-hand lookup is keyed by (pair_id, hand_id).
    hand_id = d['hand_id'].to_numpy()
    key_all = list(zip(pids_all.tolist(), hand_id.tolist()))
    assert len(set(key_all)) == len(key_all), '(pair_id, hand_id) is not unique'
    oof_of = dict(zip(key_all, oof.tolist()))
    rel_of = dict(zip(key_all, rel.tolist()))
    rank_of = dict(zip(key_all, orank.tolist()))

    # which misses a DT/SP-level recall would recover: shallowest first
    dtsp_recall = (STORED['listed_by_family']['directed_transfer'] - STORED['miss_by_family']['directed_transfer']
                   + STORED['listed_by_family']['soft_play'] - STORED['miss_by_family']['soft_play']) / \
                  (STORED['listed_by_family']['directed_transfer'] + STORED['listed_by_family']['soft_play'])
    target_in20 = int(round(215 * dtsp_recall))
    n_recover = target_in20 - 186
    mci = miss & (fam == CI)
    ci_miss_keys = sorted([k for k, m in zip(key_all, mci.tolist()) if m], key=lambda k: rank_of[k])
    recovered = set(ci_miss_keys[:n_recover])          # (pair_id, hand_id) keys
    all_ci_miss = set(ci_miss_keys)
    assert len(all_ci_miss) == 29 and len(recovered) == n_recover
    rec_by_pair = {p: set() for p in truths}
    allm_by_pair = {p: set() for p in truths}
    for pp, h in recovered:
        rec_by_pair[pp].add(h)
    for pp, h in all_ci_miss:
        allm_by_pair[pp].add(h)
    assert all(rec_by_pair[p] <= truths[p] for p in truths), 'a recovered hand is not listed'

    def oracle_map5(top20_of):
        return float(np.mean([min(len(top20_of[p] & truths[p]), 5) / min(5, len(truths[p]))
                              for p in truths]))

    def oracle_overlap(top20_of):
        return float(np.mean([min(len(top20_of[p] & truths[p]), 5) for p in truths]))

    ceiling_by_salt = {}
    for salt in SALTS:
        ds = resplit(d, salt)
        pr = fit_pointwise(ds, designs2)
        top5, top20, _ = ranked_lists(ds, pr, TOP)
        map5 = float(np.mean([ap5(top5[p], truths[p]) for p in truths]))
        hits = float(np.mean([len(set(top5[p]) & truths[p]) for p in truths]))
        tp = int(sum(len(set(top5[p]) & truths[p]) for p in truths))
        fn20 = int(sum(len(top20[p] & truths[p]) - len(set(top5[p]) & truths[p]) for p in truths))
        conf_s = {'TP': tp, 'FN': fn20, 'FP': 5 * len(truths) - tp,
                  'TN': (TOP - 5) * len(truths) - fn20,
                  'retrieval_misses': int(miss.sum()), 'mean_overlap': r4(hits), 'map5': r4(map5)}
        byfam = {}
        for f in FAMILIES:
            ps = [p for p in truths if labels[p] == f]
            t = int(sum(len(set(top5[p]) & truths[p]) for p in ps))
            n_in = int(sum(len(top20[p] & truths[p]) for p in ps))
            byfam[f] = {'pairs': len(ps), 'TP': t, 'FN': n_in - t, 'listed_in_top20': n_in,
                        'conversion_rate_TP_over_in_top20': r4(t / n_in),
                        'map5': r4(float(np.mean([ap5(top5[p], truths[p]) for p in ps])))}
        p_conv = byfam[CI]['conversion_rate_TP_over_in_top20']

        # -------- ceiling 1: perfect retrieval of the recovered hands + PERFECT selection
        t20_rec = {p: (top20[p] | rec_by_pair[p]) for p in truths}
        t20_all = {p: (top20[p] | allm_by_pair[p]) for p in truths}
        base_or_map5, base_or_ov = oracle_map5(top20), oracle_overlap(top20)
        rec_or_map5, rec_or_ov = oracle_map5(t20_rec), oracle_overlap(t20_rec)
        all_or_map5, all_or_ov = oracle_map5(t20_all), oracle_overlap(t20_all)

        # -------- ceiling 2: production five, with the recovered hands forced to the front
        def forced_map5(by_pair):
            tot = 0.0
            for p in truths:
                add = sorted(h for h in by_pair[p] if h not in top5[p])
                if not add:
                    tot += ap5(top5[p], truths[p]); continue
                keep = list(top5[p])[:max(0, 5 - len(add))]
                tot += ap5((add + keep)[:5], truths[p])
            return tot / len(truths)
        forced_rec = forced_map5(rec_by_pair)
        forced_all = forced_map5(allm_by_pair)

        # -------- ceiling 3: realistic, each recovered hand converts with the observed CI rate
        rngc = np.random.default_rng(PERM_SEED + 31)
        acc = 0.0
        REPS = 2000
        addable = {p: sorted(h for h in rec_by_pair[p] if h not in top5[p]) for p in truths}
        base_ap = {p: ap5(top5[p], truths[p]) for p in truths}
        live = [p for p in truths if addable[p]]
        base_sum = sum(base_ap.values())
        for _ in range(REPS):
            tot = base_sum
            for p in live:
                take = [h for h in addable[p] if rngc.random() < p_conv]
                if not take:
                    continue
                keep = list(top5[p])[:max(0, 5 - len(take))]
                tot += ap5((take + keep)[:5], truths[p]) - base_ap[p]
            acc += tot / len(truths)
        realistic_conv = acc / REPS

        # -------- ceiling 4: counterfactual admission, re-sorted by the PRODUCTION stage-2 score
        s2_of = dict(zip(key_all, pr.tolist()))
        tot = 0.0
        rec_pos, n_rec_in_five, n_rec_admitted = [], 0, 0
        for p in truths:
            add = sorted(rec_by_pair[p])
            if not add:
                tot += base_ap[p]; continue
            cur = sorted(top20[p], key=lambda h: -oof_of[(p, h)])
            drop, kept = 0, []
            for h in reversed(cur):                      # displace the lowest-oof UNLISTED candidates
                if drop < len(add) and not rel_of[(p, h)]:
                    drop += 1
                else:
                    kept.append(h)
            newc = kept + add
            newc.sort(key=lambda h: -s2_of[(p, h)])
            pos = {h: i + 1 for i, h in enumerate(newc)}
            for h in add:
                n_rec_admitted += 1
                rec_pos.append(pos[h])
                n_rec_in_five += int(pos[h] <= 5)
            tot += ap5(newc[:5], truths[p])
        counterfactual = tot / len(truths)
        rec_diag = {
            'what': 'where the PRODUCTION stage-2 model puts the recovered hands once they are '
                    'admitted to the candidate list.  This is the mechanism behind the coupling '
                    'tax: stage 2 was fitted around the production stage-1 rank distribution.',
            'WHY_THIS_IS_NOT_A_STATEMENT_ABOUT_THE_HANDS': 'stage 2 carries S1 = [oof, oof_rank] as '
                'INPUT FEATURES and is trained only on rows with oof_rank <= 20.  A recovered hand '
                'arrives with oof_rank > 20, outside the model\'s training support on one of its '
                'own inputs, so the tree extrapolates it to the bottom of the order.  Ceiling D is '
                'therefore evidence that the CURRENT stage 2 cannot score these hands at all -- it '
                'is NOT evidence that the hands are unselectable.  Any real stage-1 change would '
                'renumber oof_rank and require stage 2 to be refitted, which is exactly the '
                'coupling tax quoted above.',
            'n_recovered_admitted': n_rec_admitted,
            'n_recovered_selected_into_the_five': n_rec_in_five,
            'share_selected': r4(n_rec_in_five / n_rec_admitted) if n_rec_admitted else None,
            'stage2_position_within_the_new_20_median': float(np.median(rec_pos)) if rec_pos else None,
            'stage2_position_within_the_new_20_mean': r4(np.mean(rec_pos)) if rec_pos else None,
            'stage2_position_histogram_by_band': {
                '1-5': int(sum(1 for x in rec_pos if x <= 5)),
                '6-10': int(sum(1 for x in rec_pos if 6 <= x <= 10)),
                '11-15': int(sum(1 for x in rec_pos if 11 <= x <= 15)),
                '16-20': int(sum(1 for x in rec_pos if x >= 16))},
            'observed_ci_conversion_rate_for_comparison': p_conv}

        ceiling_by_salt[salt] = {
            'stage2_confusion': conf_s, 'by_family': byfam,
            'ci_conversion_rate_used': p_conv,
            'oracle_diagnostic_not_a_result': {
                'READ_THIS_FIRST': 'none of the numbers in this block is a claim about what any '
                                   'method would achieve.  They are ceilings computed by telling '
                                   'the chain the answer.',
                'n_recovered_at_dtsp_recall': n_recover,
                'n_ci_misses': len(all_ci_miss),
                'dtsp_pooled_recall_target': r4(dtsp_recall),
                'target_ci_listed_in_top20': target_in20,
                'baseline_oracle_within_top20_map5': r4(base_or_map5),
                'baseline_oracle_within_top20_overlap': r4(base_or_ov),
                'A_perfect_retrieval_of_recovered_plus_perfect_selection': {
                    'oracle_map5': r4(rec_or_map5), 'oracle_overlap': r4(rec_or_ov),
                    'delta_map5_vs_baseline_oracle': r5(rec_or_map5 - base_or_map5),
                    'composite_delta': r5(.2 * (rec_or_map5 - base_or_map5))},
                'A2_perfect_retrieval_of_ALL_29_plus_perfect_selection': {
                    'oracle_map5': r4(all_or_map5), 'oracle_overlap': r4(all_or_ov),
                    'delta_map5_vs_baseline_oracle': r5(all_or_map5 - base_or_map5),
                    'composite_delta': r5(.2 * (all_or_map5 - base_or_map5))},
                'B_production_five_with_recovered_forced_to_the_front': {
                    'map5': r4(forced_rec), 'delta_map5_vs_production': r5(forced_rec - map5),
                    'composite_delta': r5(.2 * (forced_rec - map5))},
                'B2_same_with_ALL_29': {
                    'map5': r4(forced_all), 'delta_map5_vs_production': r5(forced_all - map5),
                    'composite_delta': r5(.2 * (forced_all - map5))},
                'C_realistic_observed_ci_conversion_rate': {
                    'p_conversion': p_conv, 'map5': r4(realistic_conv),
                    'delta_map5_vs_production': r5(realistic_conv - map5),
                    'composite_delta': r5(.2 * (realistic_conv - map5))},
                'D_realistic_counterfactual_admission_rescored_by_production_stage2': {
                    'map5': r4(counterfactual), 'delta_map5_vs_production': r5(counterfactual - map5),
                    'composite_delta': r5(.2 * (counterfactual - map5)),
                    'where_stage2_puts_the_recovered_hands': rec_diag},
                'COUPLING_TAX_CAVEAT': PRE_REGISTRATION['coupling_tax_caveat'],
                'second_caveat': 'these ceilings also assume the retrieval improvement admits '
                                 'exactly the right hands and displaces only unlisted candidates. '
                                 'A real stage-1 change would move every pair of every family.'}}
        cc = ceiling_by_salt[salt]['oracle_diagnostic_not_a_result']
        log(f"\n  6.  [{salt}]  TP {conf_s['TP']}  FN {conf_s['FN']}  miss {conf_s['retrieval_misses']}"
            f"  overlap {conf_s['mean_overlap']}  MAP@5 {conf_s['map5']}   CI conversion {p_conv}")
        log(f"      ceiling A (recover {n_recover}, perfect selection)  oracle MAP@5 "
            f"{cc['A_perfect_retrieval_of_recovered_plus_perfect_selection']['oracle_map5']} "
            f"(+{cc['A_perfect_retrieval_of_recovered_plus_perfect_selection']['delta_map5_vs_baseline_oracle']}"
            f", composite +{cc['A_perfect_retrieval_of_recovered_plus_perfect_selection']['composite_delta']})")
        log(f"      ceiling B (production five, recovered forced in)  MAP@5 "
            f"{cc['B_production_five_with_recovered_forced_to_the_front']['map5']} "
            f"(+{cc['B_production_five_with_recovered_forced_to_the_front']['delta_map5_vs_production']}"
            f", composite +{cc['B_production_five_with_recovered_forced_to_the_front']['composite_delta']})")
        log(f"      ceiling C (observed CI conversion {p_conv})       MAP@5 "
            f"{cc['C_realistic_observed_ci_conversion_rate']['map5']} "
            f"(+{cc['C_realistic_observed_ci_conversion_rate']['delta_map5_vs_production']}"
            f", composite +{cc['C_realistic_observed_ci_conversion_rate']['composite_delta']})")
        dd = cc['D_realistic_counterfactual_admission_rescored_by_production_stage2']
        log(f"      ceiling D (counterfactual admission, prod stage 2) MAP@5 {dd['map5']} "
            f"({dd['delta_map5_vs_production']:+}, composite {dd['composite_delta']:+}) -- stage 2 "
            f"puts {dd['where_stage2_puts_the_recovered_hands']['n_recovered_selected_into_the_five']}"
            f" of {dd['where_stage2_puts_the_recovered_hands']['n_recovered_admitted']} recovered "
            f"hands in its five (median position "
            f"{dd['where_stage2_puts_the_recovered_hands']['stage2_position_within_the_new_20_median']})")

    # control: salt 1 must reproduce step 87's stored confusion
    h = ceiling_by_salt[SALTS[0]]['stage2_confusion']
    stage2_repro = {k: {'reproduced': h[k], 'stored': STORED[k], 'equal': h[k] == STORED[k]}
                    for k in ('TP', 'FN', 'FP', 'TN')}
    stage2_repro['map5'] = {'reproduced': h['map5'], 'stored': STORED['map5'],
                            'equal': h['map5'] == STORED['map5']}
    stage2_repro['mean_overlap'] = {'reproduced': h['mean_overlap'], 'stored': STORED['mean_overlap'],
                                    'equal': h['mean_overlap'] == STORED['mean_overlap']}
    stage2_repro['ALL_EQUAL'] = all(v['equal'] for v in stage2_repro.values() if isinstance(v, dict))
    log(f"\n      stage-2 confusion reproduces step 87 on salt 1: {stage2_repro['ALL_EQUAL']}")
    assert stage2_repro['ALL_EQUAL'], f'the stage-2 confusion did not reproduce: {stage2_repro}'

    # ==============================================================================================
    # 7.  VERDICT
    # ==============================================================================================
    def anyadm(cname, bname):
        return all(salt_results[s]['comparisons'][cname][bname]['ANY_DESCRIPTOR_ADMITTED']
                   for s in SALTS)

    primary_fw = consistency['max_familywise_p_over_salts']['B_family_PRIMARY']['interpretable']
    best_desc = {}
    for cname in CMPS:
        for bname in ('interpretable', 'ranker'):
            t = salt_results[SALTS[0]]['comparisons'][cname][bname]['top_descriptors'][0]
            best_desc[f'{cname}|{bname}'] = {
                'descriptor': t['descriptor'], 'cliffs_delta': t['cliffs_delta'],
                'hedges_g': t['hedges_g'], 'bh_q': t['bh_q'],
                'cluster_bootstrap_ci95': t['cluster_bootstrap_ci95'],
                'block_familywise_p_max_over_salts':
                    consistency['max_familywise_p_over_salts'][cname][bname],
                'SURVIVES_THE_PERMUTATION_MAXIMUM':
                    bool(consistency['max_familywise_p_over_salts'][cname][bname] <= .05),
                'admitted_on_all_three_salts':
                    t['descriptor'] in consistency['admitted_on_all_three_salts'][cname][bname]}

    separating = {c: {b: anyadm(c, b) for b in ('interpretable', 'ranker')} for c in CMPS}
    describable = bool(separating['B_family_PRIMARY']['interpretable']
                       or separating['B_family_PRIMARY']['ranker'])

    verdict = {
        'gate_0_miss_set_reproduced_exactly': repro['REPRODUCED_EXACTLY'],
        'stage2_confusion_reproduced_step87': stage2_repro['ALL_EQUAL'],
        'PRIMARY_COMPARISON': 'B_family_PRIMARY (CI-missed n=29 vs DT/SP-missed n=13)',
        'primary_familywise_permutation_p_interpretable_max_over_salts': primary_fw,
        'any_descriptor_separating': separating,
        'IS_THE_FAILURE_DESCRIBABLE_BY_A_PER_HAND_DESCRIPTOR': describable,
        'best_descriptor_per_cell': best_desc,
        'what_IS_established_without_any_descriptor_test': [
            'the miss set reproduces exactly: 42 misses, 8 / 5 / 29 by family, over listed totals '
            '355 / 224 / 215',
            'the CI misses sit DEEPER in the stage-1 order than the DT/SP misses '
            f"(median rank {q_rank['by_family'][CI]['median_rank']} vs "
            f"{q_rank['dt_sp_pooled']['median_rank']}, max {q_rank['by_family'][CI]['max_rank']} vs "
            f"{q_rank['dt_sp_pooled']['max_rank']})",
            'the misses are spread over many CI pairs rather than concentrated on a few '
            f"({conc[CI]['pairs_with_a_miss']} of {conc[CI]['pairs']} CI pairs carry at least one, "
            f"at most {conc[CI]['max_misses_on_one_pair']} on any one pair)",
            'the obvious pair-level explanations are REFUTED: CI pairs are not longer (median '
            f"n_hands {conf['by_family'][CI]['n_hands_median']} vs "
            f"{conf['by_family']['directed_transfer']['n_hands_median']} DT and "
            f"{conf['by_family']['soft_play']['n_hands_median']} SP) and do not carry more listed "
            f"hands (mean listed/pair {conf['by_family'][CI]['n_listed_mean']} vs "
            f"{conf['by_family']['directed_transfer']['n_listed_mean']} DT and "
            f"{conf['by_family']['soft_play']['n_listed_mean']} SP), so a fixed 20-slot budget does "
            'not bite harder on CI for a counting reason'],
        'interpretation_notes_NOT_part_of_the_admission_rule': [
            'comparison A on the RANKER block is partially self-referential: those 377 columns are '
            'stage-1\'s own inputs, and the stage-1 score is a fitted function of them, so a hand '
            'stage 1 ranks below its budget will tend to differ from one it ranks above.  That is '
            'exactly why comparison B (CI-missed vs DT/SP-missed) was pre-registered as the '
            'primary: B holds "was missed" fixed and varies only the family.  A result on A/ranker '
            'says the miss is describable in stage-1\'s own feature space; it does NOT say the '
            'failure is specific to coordinated_isolation.',
            'comparison A on the INTERPRETABLE block carries no such self-reference, because those '
            'gameplay columns are not stage-1 inputs.'],
        'nothing_is_proposed': 'this is a diagnosis.  No lever, no submission, no held-fold read.'}

    res = {
        'author': AUTHOR, 'step': STEP, 'training_only': True,
        'held_folds_touched': False,
        'held_folds_note': 'folds 0, 1, 2 are never read for truth.  build_d0() -> load_fold(0) '
                           'supplies FEATURE COLUMNS on audit rows only, exactly as steps '
                           '74/76/77/84/87 do.  The audit tables were asserted disjoint from all '
                           'three held pools at entry (0 / 0 / 0 overlap).  No held-fold pair, '
                           'label or score was read.',
        'held_fold_overlap_check': held,
        'production_branch': 'none: diagnostic only, nothing is proposed for submission',
        'question': PRE_REGISTRATION['question'],
        'pre_registration': PRE_REGISTRATION,
        'prediction_file': str(PRED_FILE).replace('\\', '/'),
        'prediction_written_before_any_fit': True,
        'protected_hashes_entry': hashes_in,
        'population': population,
        'gate_0_reproduction': repro,
        'q_rank_of_the_misses': q_rank,
        'q_pair_level_confounds': conf,
        'descriptors': {'interpretable_columns': len(block1), 'ranker_columns': len(block2),
                        'spaces': 2, 'total_tested': 2 * (len(block1) + len(block2)),
                        'excluded_tautological': S1,
                        'tautology_check_stage1_columns': taut,
                        'non_finite_filled': nonfinite},
        'salt_runs': salt_results,
        'cross_salt_consistency': consistency,
        'stage2_and_ceiling_by_salt': ceiling_by_salt,
        'stage2_reproduction_control': stage2_repro,
        'verdict': verdict,
        'provenance': provenance,
    }
    assert res['pre_registration'] == read_json(PRED_FILE)['pre_registration'], \
        'the artefact pre_registration block differs from the file written before the fit'

    # -------------------------------------------------------------- prose self-check
    def dig(path):
        cur = res
        for k in path.split('/'):
            cur = cur[int(k)] if isinstance(cur, list) else cur[k]
        return cur

    prose = [
        ('miss set reproduced exactly', 'gate_0_reproduction/REPRODUCED_EXACTLY', True),
        ('CI misses', f'gate_0_reproduction/by_family/{CI}/retrieval_miss', 29),
        ('DT misses', 'gate_0_reproduction/by_family/directed_transfer/retrieval_miss', 8),
        ('SP misses', 'gate_0_reproduction/by_family/soft_play/retrieval_miss', 5),
        ('CI listed', f'gate_0_reproduction/by_family/{CI}/listed', 215),
        ('DT listed', 'gate_0_reproduction/by_family/directed_transfer/listed', 355),
        ('SP listed', 'gate_0_reproduction/by_family/soft_play/listed', 224),
        ('CI recall@20', f'gate_0_reproduction/by_family/{CI}/recall_at_20', 0.8651),
        ('total misses', 'gate_0_reproduction/miss_total', 42),
        ('held folds touched', 'held_folds_touched', False),
        ('stage2 reproduced', 'stage2_reproduction_control/ALL_EQUAL', True),
        ('CI median miss rank', f'q_rank_of_the_misses/by_family/{CI}/median_rank',
         q_rank['by_family'][CI]['median_rank']),
        ('DTSP median miss rank', 'q_rank_of_the_misses/dt_sp_pooled/median_rank',
         q_rank['dt_sp_pooled']['median_rank']),
        ('primary familywise p', 'verdict/primary_familywise_permutation_p_interpretable_max_over_salts',
         primary_fw),
        ('describable', 'verdict/IS_THE_FAILURE_DESCRIBABLE_BY_A_PER_HAND_DESCRIPTOR', describable),
        ('n recovered', f'stage2_and_ceiling_by_salt/{SALTS[0]}/oracle_diagnostic_not_a_result/'
                        'n_recovered_at_dtsp_recall', n_recover),
        ('CI misses max stage-1 rank', f'q_rank_of_the_misses/by_family/{CI}/max_rank',
         q_rank['by_family'][CI]['max_rank']),
        ('DTSP misses max stage-1 rank', 'q_rank_of_the_misses/dt_sp_pooled/max_rank',
         q_rank['dt_sp_pooled']['max_rank']),
        ('CI pairs with a miss', f'q_rank_of_the_misses/per_pair_concentration/{CI}/pairs_with_a_miss',
         conc[CI]['pairs_with_a_miss']),
    ] + [
        (f'familywise p {c}/{b}', f'cross_salt_consistency/max_familywise_p_over_salts/{c}/{b}',
         consistency['max_familywise_p_over_salts'][c][b])
        for c in CMPS for b in ('interpretable', 'ranker')
    ] + [
        (f'ceiling {k} composite salt1',
         f'stage2_and_ceiling_by_salt/{SALTS[0]}/oracle_diagnostic_not_a_result/{k}/composite_delta',
         ceiling_by_salt[SALTS[0]]['oracle_diagnostic_not_a_result'][k]['composite_delta'])
        for k in ('A_perfect_retrieval_of_recovered_plus_perfect_selection',
                  'A2_perfect_retrieval_of_ALL_29_plus_perfect_selection',
                  'B_production_five_with_recovered_forced_to_the_front',
                  'C_realistic_observed_ci_conversion_rate',
                  'D_realistic_counterfactual_admission_rescored_by_production_stage2')
    ] + [
        ('CI conversion rate salt1', f'stage2_and_ceiling_by_salt/{SALTS[0]}/ci_conversion_rate_used',
         ceiling_by_salt[SALTS[0]]['ci_conversion_rate_used']),
    ]
    mismatch = [(lbl, p, dig(p), exp) for lbl, p, exp in prose if dig(p) != exp]
    assert not mismatch, f'PROSE SELF-CHECK FAILED: {mismatch}'
    res['prose_self_check'] = {'checked': [{'label': l, 'path': p, 'value': dig(p)} for l, p, _ in prose],
                               'all_matched': True}
    log(f'\n  prose self-check: {len(prose)} figures asserted against the artefact, all matched')

    hashes_out = check_protected()
    res['protected_hashes_exit'] = hashes_out
    res['protected_hashes_identical'] = hashes_in == hashes_out
    assert res['protected_hashes_identical'], 'a protected submission changed during the run'
    res['seconds'] = round(time.time() - t0)
    write_json(RES_FILE, res)
    log(f"\n  wrote {RES_FILE}  ({res['seconds']} s)")
    return res


if __name__ == '__main__':
    main()
