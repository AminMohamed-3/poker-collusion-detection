"""Step 77 [C].  What process chose the five listed evidence hands?
author: claude (agent)

Motivation.  Step 76 measured where the evidence MAP@5 loss sits: stage 1's top-20 already holds
4.642 of the (at most) 5 listed hands, stage 2 keeps only 3.333 of them, and a perfect 5-of-20
selection would take MAP@5 from .6159 to .9481.  Every remaining point is in the SELECTION step, so
any regularity in the organisers' choice of five is directly worth score.

This step is descriptive only.  It asks five questions on the 162-pair audit set (172 tables outside
every held fold; folds 0/1/2 are never read, nothing is fitted, no candidate is produced):

  1. UNIVARIATE   for every per-hand column, the within-pair AUC and the mean within-pair percentile
                  of listed hands, over two populations: all of the pair's hands (the organisers'
                  universe) and stage 1's top-20 (the universe our selection actually chooses from).
  2. HARD RULES   deterministic "take the five ..." rules, scored by exact set match and by mean
                  overlap with the truth set, in both universes.  `oof_desc` on the all-hands
                  universe is a reproduction control: it must return step 76's 2.827.
  3. PER FAMILY   1 and 2 split by behavior_family.
  4. SPACING      the observed chronological spacing of the five against two Monte-Carlo nulls
                  (5 uniform from all hands, 5 uniform from the top-20) and against a span-conditioned
                  null that fixes the first and last listed hand and re-draws the middle three.
  5. CONDITIONAL  the part a pointwise classifier cannot represent.  Leave-one-out: hold out one
                  listed hand, build a context S from the other four, and ask whether the held-out
                  hand is more "similar to S" than the unlisted top-20 candidates are.  A
                  random-context control (S drawn from unlisted candidates) removes the part of that
                  AUC which is really a marginal effect.

Prior work not repeated here (audit-truncation.json, 14.11 / 14.16 / 14.17 / 14.21 / 14.22 / 14.42):
the cap of five and its distribution, evidence_rank being (type, time)-sorted, the first-k / earlier-
top-10 profile, earliest-of-top-K as an ORDERING rule, direction consistency as a feature, mechanism
signatures, sizing fingerprints, and the Poisson-binomial decode of the cap.  All were measured; this
step measures the SET, not the order, and reports rules and conditional structure that were not.

A consistency check on all 372 labelled positive pairs is run for the columns that need only
hands.parquet + seats.parquet, and is labelled as such (it is a description, not a fit).

Output: Claude Version/experiments/label_process_20260911/audit-C-label-process.json
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
from evidence_aug.paths import ORIGINAL, check_protected, write_json
from evidence_aug.step5_evaluate import pool_bootstrap

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'label_process_20260911'
DATA = ORIGINAL / 'data'
AUTHOR = 'claude'
TOP = 20
SEED = 20260912
MC = 2000                       # Monte-Carlo draws per pair for the spacing / dispersion nulls
RCTX = 24                       # random-context repetitions in the conditional test
# reproduction controls (step 76, audit-C-order-vs-set.json / 14.16)
CTRL_OOF_TOP5_OVERLAP = 2.8272
CTRL_TYPE_SHARE = {('coordinated_isolation', 'fold'): 210, ('coordinated_isolation', 'showdown'): 5,
                   ('directed_transfer', 'fold'): 271, ('directed_transfer', 'showdown'): 84,
                   ('soft_play', 'fold'): 157, ('soft_play', 'showdown'): 67}


def log(*a):
    print(*a, flush=True)


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


# ----------------------------------------------------------------------------------------------
# 1.  frame
# ----------------------------------------------------------------------------------------------
def add_derived(d):
    """Gameplay-derived per-hand columns.  Direction is aligned by the pair's own net chip flow over
    its whole shared history (a gameplay quantity), never by the label file's player order."""
    d = d.with_columns(
        (pl.col('board_cards').str.strip_chars().str.split(' ').list.len()
         * (pl.col('board_cards').str.strip_chars().str.len_chars() > 0).cast(pl.Int64)).alias('n_board'),
        (pl.col('final_pot') / pl.col('big_blind')).alias('pot_bb_raw'),
        # 14.16's list type: 'showdown' = BOTH members reached showdown (nobody in the pair folded);
        # this reproduces its listed type counts exactly and is asserted as a control in main().
        (pl.col('pair_sd') == 2).cast(pl.Float64).alias('type_showdown'),
        (pl.col('pair_sd') > 0).cast(pl.Float64).alias('either_member_sd'),
        pl.max_horizontal('A_stack_bb', 'B_stack_bb').alias('max_stack_bb'),
        pl.min_horizontal('A_stack_bb', 'B_stack_bb').alias('min_stack_bb'),
        pl.max_horizontal('A_put_bb', 'B_put_bb').alias('max_put_bb'),
        pl.min_horizontal('A_put_bb', 'B_put_bb').alias('min_put_bb'),
    )
    # which member benefits over the whole shared history -> ben / oth alignment
    d = d.with_columns((pl.col('A_net_bb').sum().over('pair_id') >= pl.col('B_net_bb').sum().over('pair_id')).alias('ben_is_A'))
    d = d.with_columns(
        pl.when('ben_is_A').then(pl.col('A_net_bb')).otherwise(pl.col('B_net_bb')).alias('net_ben_bb'),
        pl.when('ben_is_A').then(pl.col('B_net_bb')).otherwise(pl.col('A_net_bb')).alias('net_oth_bb'),
        pl.when('ben_is_A').then(pl.col('A_chen')).otherwise(pl.col('B_chen')).alias('chen_ben'),
        pl.when('ben_is_A').then(pl.col('B_chen')).otherwise(pl.col('A_chen')).alias('chen_oth'),
        pl.when('ben_is_A').then(pl.col('A_put_bb')).otherwise(pl.col('B_put_bb')).alias('put_ben_bb'),
        pl.when('ben_is_A').then(pl.col('B_put_bb')).otherwise(pl.col('A_put_bb')).alias('put_oth_bb'),
        pl.when('ben_is_A').then(pl.col('B_folded')).otherwise(pl.col('A_folded')).alias('oth_folded'),
    )
    d = d.with_columns(
        (pl.col('net_ben_bb') - pl.col('net_oth_bb')).alias('flow_to_ben_bb'),
        pl.min_horizontal(pl.col('net_ben_bb'), -pl.col('net_oth_bb')).clip(lower_bound=0).alias('xfer_to_ben_bb'),
        pl.min_horizontal(pl.max_horizontal('A_net_bb', 'B_net_bb'),
                          -pl.min_horizontal('A_net_bb', 'B_net_bb')).clip(lower_bound=0).alias('xfer_abs_bb'),
        (pl.col('net_ben_bb') > pl.col('net_oth_bb')).cast(pl.Float64).alias('winner_is_ben'),
        pl.when(pl.col('A_net_bb') >= pl.col('B_net_bb')).then(pl.col('B_put_bb')).otherwise(pl.col('A_put_bb')).alias('loser_put_bb'),
        pl.when(pl.col('A_folded') > 0).then(pl.col('A_chen')).otherwise(-1.0).alias('_fcA'),
        pl.when(pl.col('B_folded') > 0).then(pl.col('B_chen')).otherwise(-1.0).alias('_fcB'),
        (-pl.col('pair_net_bb')).alias('others_net_bb'),
        (pl.col('pot_bb') / pl.max_horizontal(pl.col('min_stack_bb'), pl.lit(1.0))).alias('pot_over_min_stack'),
        (pl.col('chen_ben') - pl.col('chen_oth')).alias('chen_gap_ben'),
    )
    d = d.with_columns(pl.max_horizontal('_fcA', '_fcB').alias('folder_chen')).drop('_fcA', '_fcB')
    # timing within the pair's shared history
    d = d.sort('pair_id', 'started_at')
    d = d.with_columns(
        (pl.col('started_at').diff().over('pair_id').dt.total_seconds().cast(pl.Float64)).alias('gap_prev_s'),
        (pl.col('started_at').diff(-1).over('pair_id').dt.total_seconds().cast(pl.Float64) * -1).alias('gap_next_s'))
    d = d.with_columns(pl.col('gap_prev_s').fill_null(-1.0), pl.col('gap_next_s').fill_null(-1.0))
    return d.sort('pair_id', 'hand_id')


def build_frame():
    oof = pl.read_parquet(OUT / 'audit-oof.parquet')
    struct = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = oof.join(struct, on=['pair_id', 'hand_id'], validate='1:1')
    assert d.height == oof.height
    hands = (pl.scan_parquet(DATA / 'hands.parquet')
             .select('hand_id', 'started_at', 'big_blind', 'final_pot', 'players_dealt',
                     'players_at_showdown', 'board_cards')
             .filter(pl.col('hand_id').is_in(d['hand_id'].implode())).collect())
    d = d.join(hands, on='hand_id', validate='m:1')
    assert d.height == oof.height, 'hands join changed the row count'
    return add_derived(d)


# ----------------------------------------------------------------------------------------------
# 2.  within-pair univariate separation
# ----------------------------------------------------------------------------------------------
def within_pair_stats(d, cols):
    """pooled within-pair AUC (Mann-Whitney, ties at .5) and mean within-pair percentile of listed."""
    rk = d.select(['pair_id', 'relevance'] + [pl.col(c).rank('average').over('pair_id').alias(c) for c in cols])
    n = d.select(pl.len().over('pair_id').alias('n'))['n'].to_numpy().astype(float)
    y = d['relevance'].to_numpy().astype(bool)
    pid = d['pair_id'].to_numpy()
    # group boundaries (frame is sorted by pair_id)
    _, start = np.unique(pid, return_index=True)
    start = np.sort(start)
    ends = np.r_[start[1:], len(pid)]
    out = {}
    for c in cols:
        r = rk[c].to_numpy().astype(float)
        pct = (r - 0.5) / n
        num = 0.0
        den = 0.0
        for s, e in zip(start, ends):
            yy = y[s:e]
            n1 = int(yy.sum())
            n0 = int((~yy).sum())
            if n1 == 0 or n0 == 0:
                continue
            u = r[s:e][yy].sum() - n1 * (n1 + 1) / 2.0
            num += u
            den += n1 * n0
        v = d[c].to_numpy().astype(float)
        out[c] = {'auc': num / den if den else None, 'mean_listed_pct': float(pct[y].mean()),
                  'mean_unlisted_pct': float(pct[~y].mean()),
                  'listed_mean': float(v[y].mean()), 'unlisted_mean': float(v[~y].mean())}
    return out


def univariate_block(d, cols, tag):
    res = {'ALL': within_pair_stats(d, cols)}
    for f in FAMILIES:
        res[f] = within_pair_stats(d.filter(pl.col('behavior_family') == f), cols)
    log(f'  univariate {tag}: {len(cols)} columns x 4 groups')
    return res


# ----------------------------------------------------------------------------------------------
# 3.  deterministic rules
# ----------------------------------------------------------------------------------------------
RULES = [
    ('largest_pot_bb', [('pot_bb', True)]),
    ('largest_pot_chips', [('final_pot', True)]),
    ('largest_abs_net_flow', [('pair_net_gap', True)]),
    ('largest_flow_to_beneficiary', [('flow_to_ben_bb', True)]),
    ('largest_xfer_to_beneficiary', [('xfer_to_ben_bb', True)]),
    ('largest_pair_joint_profit', [('pair_net_bb', True)]),
    ('largest_victim_loss', [('others_net_bb', False)]),
    ('largest_loser_contribution', [('loser_put_bb', True)]),
    ('largest_min_member_contribution', [('min_put_bb', True)]),
    ('earliest', [('chrono_rank', False)]),
    ('latest', [('chrono_rank', True)]),
    ('earliest_fold_type_then_showdown', [('type_showdown', False), ('chrono_rank', False)]),
    ('earliest_showdown_type_first', [('type_showdown', True), ('chrono_rank', False)]),
    ('pair_showdown_then_pot', [('pair_sd', True), ('pot_bb', True)]),
    ('pair_reached_showdown_then_time', [('pair_sd', True), ('chrono_rank', False)]),
    ('most_actions', [('n_actions', True)]),
    ('heads_up_pair_then_pot', [('hu_pair_only', True), ('pot_bb', True)]),
    ('biggest_laydown_by_member', [('folder_chen', True)]),
    ('pair_opened_then_time', [('pf_open_pair', True), ('chrono_rank', False)]),
    ('iso_raise_then_time', [('pf_iso_raise', True), ('chrono_rank', False)]),
    ('iso_raise_then_pot', [('pf_iso_raise', True), ('pot_bb', True)]),
    ('victim_folds_to_pair_bet_then_pot', [('post_victim_fold_to_pair_bet', True), ('pot_bb', True)]),
    ('deepest_street_then_pot', [('max_street', True), ('pot_bb', True)]),
    ('flow_positive_then_pot', [('winner_is_ben', True), ('pot_bb', True)]),
    ('flow_positive_then_time', [('winner_is_ben', True), ('chrono_rank', False)]),
    ('largest_pot_bb_then_earliest', [('pot_bb', True), ('chrono_rank', False)]),
    ('oof_desc', [('oof', True)]),                        # reproduction control on the all-hands universe
]


def rule_sets(d, keys, kmap):
    """top-k rows per pair under the sort keys (ties broken by time then by our ranker)."""
    by = [k for k, _ in keys] + ['chrono_rank', 'oof']
    desc = [v for _, v in keys] + [False, True]
    s = d.sort(by, descending=desc, nulls_last=True)
    s = s.with_columns(pl.int_range(pl.len()).over('pair_id').alias('_r'))
    s = s.join(pl.DataFrame({'pair_id': list(kmap.keys()), '_k': [kmap[p] for p in kmap]}), on='pair_id')
    s = s.filter(pl.col('_r') < pl.col('_k'))
    return {k[0]: set(g['hand_id']) for k, g in s.group_by('pair_id')}


def score_rule(sets, truths, labels):
    """Every rule emits five hands, exactly as our chain does.

    exact_match       the five ARE the truth set (only possible for the 150 pairs with |R| = 5)
    contains_all      the five contain every listed hand (the honest 'this rule explains the pair')
    mean_overlap      |rule5 n truth|, directly comparable to stage 2's 3.333 and the 4.642 oracle
    """
    ex, ca, ov, mp, fam = [], [], [], [], {f: [] for f in FAMILIES}
    for p, R in truths.items():
        S = sets.get(p, set())
        hit = len(S & R)
        ov.append(hit)
        ex.append(float(S == R))
        ca.append(float(hit == len(R)))
        mp.append(hit / min(5, len(R)))
        fam[labels[p]].append(hit)
    return {'exact_match_pairs': int(sum(ex)), 'exact_match_rate': r4(np.mean(ex)),
            'contains_all_listed_pairs': int(sum(ca)), 'contains_all_listed_rate': r4(np.mean(ca)),
            'mean_overlap': r4(np.mean(ov)), 'mean_overlap_over_truth': r4(np.mean(mp)),
            'by_family_mean_overlap': {f: (r4(np.mean(v)) if v else None) for f, v in fam.items()}}


def pair_hits(sets, truths):
    return {p: len(sets.get(p, set()) & R) for p, R in truths.items()}


def blend_frame(df, col, higher_is_listed, w):
    """rank-blend of the stage-1 score with one column, both as within-pair percentiles."""
    return (df.with_columns((pl.col('oof').rank('average').over('pair_id') / pl.len().over('pair_id')).alias('_zoof'),
                            (pl.col(col).rank('average', descending=not higher_is_listed).over('pair_id')
                             / pl.len().over('pair_id')).alias('_zc'))
            .with_columns(((1 - w) * pl.col('_zoof') + w * pl.col('_zc')).alias('_blend')))


def random_baseline(truths, labels, universe_sizes):
    ov, fam = [], {f: [] for f in FAMILIES}
    for p, R in truths.items():
        e = 5 * len(R) / universe_sizes[p]      # expected overlap of 5 uniform draws without replacement
        ov.append(e)
        fam[labels[p]].append(e)
    return {'exact_match_pairs': None, 'exact_match_rate': None, 'contains_all_listed_pairs': None,
            'contains_all_listed_rate': None, 'mean_overlap': r4(np.mean(ov)),
            'mean_overlap_over_truth': None, 'by_family_mean_overlap': {f: r4(np.mean(v)) for f, v in fam.items()}}


# ----------------------------------------------------------------------------------------------
# 4.  spacing
# ----------------------------------------------------------------------------------------------
def spacing_block(d, labels, rng):
    obs = {'min_gap': [], 'mean_gap': [], 'span_frac': [], 'adjacent': [], 'first_frac': []}
    null_pct = {'min_gap': [], 'span_frac': []}
    null_pct20 = {'min_gap': [], 'span_frac': []}
    cond_pct = {'min_gap': []}
    nulls = {'min_gap': [], 'span_frac': [], 'adjacent': []}
    nulls20 = {'min_gap': [], 'span_frac': [], 'adjacent': []}
    cond_adj = []
    per_fam = {f: {'obs_min_gap': [], 'null_min_gap': []} for f in FAMILIES}
    for (pid,), g in d.group_by('pair_id'):
        n = g.height
        c = np.sort(g.filter(pl.col('relevance') == 1)['chrono_rank'].to_numpy().astype(int))
        k = len(c)
        if k < 3:
            continue
        cand = np.sort(g.filter(pl.col('oof_rank') <= TOP)['chrono_rank'].to_numpy().astype(int))
        gaps = np.diff(c)
        o_min, o_span = int(gaps.min()), float((c[-1] - c[0]) / (n - 1))
        obs['min_gap'].append(o_min)
        obs['mean_gap'].append(float(gaps.mean()))
        obs['span_frac'].append(o_span)
        obs['adjacent'].append(float((gaps == 1).mean()))
        obs['first_frac'].append(float(c[0] / (n - 1)))
        # null A: k uniform from all n hands
        s = np.sort(rng.random((MC, n)).argsort(1)[:, :k], axis=1)
        gg = np.diff(s, axis=1)
        nmin, nspan, nadj = gg.min(1), (s[:, -1] - s[:, 0]) / (n - 1), (gg == 1).mean(1)
        nulls['min_gap'].append(nmin.mean()); nulls['span_frac'].append(nspan.mean()); nulls['adjacent'].append(nadj.mean())
        null_pct['min_gap'].append(float((nmin < o_min).mean() + .5 * (nmin == o_min).mean()))
        null_pct['span_frac'].append(float((nspan < o_span).mean() + .5 * (nspan == o_span).mean()))
        per_fam[labels[pid]]['obs_min_gap'].append(o_min)
        per_fam[labels[pid]]['null_min_gap'].append(float(nmin.mean()))
        # null B: k uniform from the pair's top-20 candidate positions
        if len(cand) >= k:
            s2 = np.sort(cand[rng.random((MC, len(cand))).argsort(1)[:, :k]], axis=1)
            g2 = np.diff(s2, axis=1)
            m2, sp2 = g2.min(1), (s2[:, -1] - s2[:, 0]) / (n - 1)
            nulls20['min_gap'].append(m2.mean()); nulls20['span_frac'].append(sp2.mean()); nulls20['adjacent'].append((g2 == 1).mean(1).mean())
            null_pct20['min_gap'].append(float((m2 < o_min).mean() + .5 * (m2 == o_min).mean()))
            null_pct20['span_frac'].append(float((sp2 < o_span).mean() + .5 * (sp2 == o_span).mean()))
        # span-conditioned null: keep the first and the last listed hand, redraw the middle k-2
        inner = np.arange(c[0] + 1, c[-1])
        if len(inner) >= k - 2 and k > 2:
            mid = np.sort(inner[rng.random((MC, len(inner))).argsort(1)[:, :k - 2]], axis=1)
            full = np.concatenate([np.full((MC, 1), c[0]), mid, np.full((MC, 1), c[-1])], axis=1)
            g3 = np.diff(full, axis=1)
            m3 = g3.min(1)
            cond_pct['min_gap'].append(float((m3 < o_min).mean() + .5 * (m3 == o_min).mean()))
            cond_adj.append(float((g3 == 1).mean()))
    out = {
        'pairs': len(obs['min_gap']),
        'observed': {k: r4(np.mean(v)) for k, v in obs.items()},
        'null_uniform_all_hands': {k: r4(np.mean(v)) for k, v in nulls.items()},
        'null_uniform_top20': {k: r4(np.mean(v)) for k, v in nulls20.items()},
        'mean_null_percentile_of_observed': {'min_gap_vs_all': r4(np.mean(null_pct['min_gap'])),
                                             'span_vs_all': r4(np.mean(null_pct['span_frac'])),
                                             'min_gap_vs_top20': r4(np.mean(null_pct20['min_gap'])),
                                             'span_vs_top20': r4(np.mean(null_pct20['span_frac'])),
                                             'min_gap_vs_span_conditioned': r4(np.mean(cond_pct['min_gap']))},
        'adjacent_share_observed_vs_span_conditioned_null': [r4(np.mean(obs['adjacent'])), r4(np.mean(cond_adj))],
        'by_family_min_gap_obs_vs_null': {f: [r4(np.mean(v['obs_min_gap'])), r4(np.mean(v['null_min_gap']))]
                                          for f, v in per_fam.items()}}
    return out


# ----------------------------------------------------------------------------------------------
# 5.  set dispersion and conditional structure (top-20 universe)
# ----------------------------------------------------------------------------------------------
DISP_COLS = ['pot_bb', 'chrono_rank', 'flow_to_ben_bb', 'oof', 'max_street', 'n_actions']
BIN_COLS = ['type_showdown', 'winner_is_ben', 'pf_open_pair', 'pair_sd']


def dispersion_block(d, rng):
    """Is the listed 5-set more homogeneous (clustered) or more spread than a random 5 of the top-20?
    Values are within-pair percentiles so every column is on the same scale."""
    res = {c: [] for c in DISP_COLS}
    resb = {c: [] for c in BIN_COLS}
    for (pid,), g in d.filter(pl.col('oof_rank') <= TOP).group_by('pair_id'):
        y = g['relevance'].to_numpy().astype(bool)
        k = int(y.sum())
        m = g.height
        if k < 3 or m - k < 3:
            continue
        idx = rng.random((MC, m)).argsort(1)[:, :k]
        for c in DISP_COLS:
            v = g[c].to_numpy().astype(float)
            p = (np.argsort(np.argsort(v)) + .5) / m
            o = float(p[y].std())
            nn = p[idx].std(1)
            res[c].append(float((nn < o).mean() + .5 * (nn == o).mean()))
        for c in BIN_COLS:
            v = (g[c].to_numpy().astype(float) > 0).astype(float)
            o = float(v[y].mean())
            o = max(o, 1 - o)                       # majority share among the listed
            nn = v[idx].mean(1)
            nn = np.maximum(nn, 1 - nn)
            resb[c].append(float((nn < o).mean() + .5 * (nn == o).mean()))
    return {'note': 'mean percentile of the observed statistic in the random-5-of-top20 null; .5 = no structure',
            'dispersion_percentile_lower_means_more_homogeneous': {c: r4(np.mean(v)) for c, v in res.items()},
            'majority_share_percentile_higher_means_more_uniform': {c: r4(np.mean(v)) for c, v in resb.items()},
            'pairs': len(res['pot_bb'])}


CONT_SIM = ['pot_bb', 'flow_to_ben_bb', 'oof', 'n_actions']
CAT_SIM = ['type_showdown', 'winner_is_ben', 'pf_open_pair']


def conditional_block(d, rng, marginal):
    """Leave-one-out: does knowing four of the listed hands help identify the fifth among the
    unlisted top-20 candidates, beyond what the same statistic gives without the coordination?

    Two controls, because the obvious one is not enough.
      random-context  the four context hands are drawn from the UNLISTED candidates.  This inverts
                      the marginal effect rather than removing it (a context of low-scoring hands
                      makes 'near the context' mean 'low scoring'), so it brackets rather than
                      isolates the conditional part.
      cross-pair      the context is another pair's listed set of the same family, carried across as
                      within-pair percentiles / chronological fractions.  The context then has the
                      same marginal profile as a real listed context but knows nothing about THIS
                      pair, so real minus cross-pair is the coordination proper.
    The marginal AUC of the underlying column inside the top-20 is reported beside each row: a
    conditional feature only earns its keep if it beats that.
    """
    names = [f'near_{c}' for c in CONT_SIM] + [f'same_{c}' for c in CAT_SIM] + ['near_in_time', 'far_in_time']
    real = {n: [0.0, 0.0] for n in names}          # [concordant, comparisons]
    ctrl = {n: [0.0, 0.0] for n in names}
    xpair = {n: [0.0, 0.0] for n in names}
    fam_real = {f: {n: [0.0, 0.0] for n in names} for f in FAMILIES}
    fam_xpair = {f: {n: [0.0, 0.0] for n in names} for f in FAMILIES}
    # donor pool: per family, the listed sets of every pair as within-pair percentiles / fractions
    donors = {f: [] for f in FAMILIES}
    for (pid,), g in d.filter(pl.col('oof_rank') <= TOP).group_by('pair_id'):
        y = g['relevance'].to_numpy().astype(bool)
        m = g.height
        if y.sum() < 3:
            continue
        rec = {c: float((((np.argsort(np.argsort(g[c].to_numpy().astype(float))) + .5) / m)[y]).mean()) for c in CONT_SIM}
        for c in CAT_SIM:
            rec[c] = float((g[c].to_numpy().astype(float) > 0)[y].mean() > .5)
        rec['chrono_frac'] = (g['chrono_rank'].to_numpy().astype(float)[y] / max(float(g['n_hands'][0]), 1.0)).tolist()
        donors[g['behavior_family'][0]].append((pid, rec))
    for (pid,), g in d.filter(pl.col('oof_rank') <= TOP).group_by('pair_id'):
        y = g['relevance'].to_numpy().astype(bool)
        m = g.height
        li = np.where(y)[0]
        ui = np.where(~y)[0]
        if len(li) < 3 or len(ui) < 4:
            continue
        fam = g['behavior_family'][0]
        nh = float(g['n_hands'][0])
        pct = {c: (np.argsort(np.argsort(g[c].to_numpy().astype(float))) + .5) / m for c in CONT_SIM}
        cat = {c: (g[c].to_numpy().astype(float) > 0).astype(float) for c in CAT_SIM}
        chrono = g['chrono_rank'].to_numpy().astype(float)

        def ctx_local(S):
            C = {c: float(pct[c][S].mean()) for c in CONT_SIM}
            for c in CAT_SIM:
                C[c] = float(cat[c][S].mean() > .5)
            C['times'] = chrono[S]
            return C

        def ctx_donor(rec):
            C = {c: rec[c] for c in CONT_SIM}
            for c in CAT_SIM:
                C[c] = float(rec[c])
            C['times'] = np.asarray(rec['chrono_frac']) * nh
            return C

        def sims(h, C):
            out = {}
            for c in CONT_SIM:
                out[f'near_{c}'] = -abs(pct[c][h] - C[c])
            for c in CAT_SIM:
                out[f'same_{c}'] = 1.0 - abs(cat[c][h] - C[c])
            dt = np.abs(chrono[h] - C['times']).min() / nh
            out['near_in_time'] = -dt
            out['far_in_time'] = dt
            return out

        def accumulate(store, pos, C, comp, w=1.0):
            sp = sims(pos, C)
            sc = {n: [] for n in names}
            for u in comp:
                su = sims(u, C)
                for n in names:
                    sc[n].append(su[n])
            for n in names:
                a = np.asarray(sc[n])
                store[n][0] += w * float((sp[n] > a).sum() + .5 * (sp[n] == a).sum())
                store[n][1] += w * len(a)

        pool = [rec for q, rec in donors[fam] if q != pid]
        for h in li:
            S = np.array([x for x in li if x != h])
            C = ctx_local(S)
            accumulate(real, h, C, ui)
            accumulate(fam_real[fam], h, C, ui)
            for _ in range(RCTX):
                Sr = rng.choice(ui, size=min(len(S), len(ui) - 2), replace=False)
                comp = np.array([u for u in ui if u not in set(Sr.tolist())])
                if len(comp):
                    accumulate(ctrl, h, ctx_local(Sr), comp, w=1.0 / RCTX)
                if pool:
                    Cd = ctx_donor(pool[int(rng.integers(len(pool)))])
                    accumulate(xpair, h, Cd, ui, w=1.0 / RCTX)
                    accumulate(fam_xpair[fam], h, Cd, ui, w=1.0 / RCTX)

    def auc(store):
        return {n: r4(v[0] / v[1]) if v[1] else None for n, v in store.items()}
    r, c, x = auc(real), auc(ctrl), auc(xpair)
    base = {f'near_{c_}': c_ for c_ in CONT_SIM}
    base.update({f'same_{c_}': c_ for c_ in CAT_SIM})
    base.update({'near_in_time': 'chrono_rank', 'far_in_time': 'chrono_rank'})
    marg = {}
    for n, col in base.items():
        a = marginal[col]['auc']
        marg[n] = r4(max(a, 1 - a))          # best orientation of the underlying column
    return {'note': 'AUC of a held-out listed hand vs the unlisted top-20 candidates given a context. '
                    'real = the four other listed hands; random = four unlisted candidates (inverts the '
                    'marginal, does not remove it); cross_pair = another pair of the same family, carried '
                    'across as percentiles (same marginal profile, no knowledge of this pair). '
                    'marginal_auc_of_column is the best-orientation within-pair AUC of the underlying '
                    'column inside the top-20: a conditional feature has to beat that to be worth anything.',
            'real_context_auc': r, 'random_context_auc': c, 'cross_pair_context_auc': x,
            'marginal_auc_of_column': marg,
            'delta_vs_random_context': {n: r4(r[n] - c[n]) for n in r if r[n] is not None and c[n] is not None},
            'delta_vs_cross_pair_context': {n: r4(r[n] - x[n]) for n in r if r[n] is not None and x[n] is not None},
            'delta_vs_own_marginal': {n: r4(r[n] - marg[n]) for n in r if r[n] is not None},
            'by_family_real_context_auc': {f: auc(v) for f, v in fam_real.items()},
            'by_family_cross_pair_context_auc': {f: auc(v) for f, v in fam_xpair.items()},
            'by_family_delta_vs_cross_pair': {
                f: {n: r4(auc(fam_real[f])[n] - auc(fam_xpair[f])[n])
                    for n in names if auc(fam_real[f])[n] is not None and auc(fam_xpair[f])[n] is not None}
                for f in FAMILIES}}


# ----------------------------------------------------------------------------------------------
# 6.  does any of this survive STAGE 2?  (the only comparison that decides whether it is a lever)
# ----------------------------------------------------------------------------------------------
def stage2_block(d, cols, res, truths, labels, ptab):
    """Rebuild the shipping hybrid stage 2 (step 74/76 machinery, audit set only) and ask whether the
    per-family columns still add once stage 2 has already seen the structural features."""
    from evidence_aug.step74_length_shift import CI, build_d0, designs_of, fit_predict, frame_at
    from evidence_aug.step76_C_order_vs_set import ranked_lists
    from evidence_aug.step5_evaluate import ap5

    d0, struct, rcols = build_d0()
    designs = designs_of(struct, rcols)
    dd = frame_at(d0, 1.0)
    extra = [c for c in cols if c not in dd.columns]
    dd = dd.join(d.select(['pair_id', 'hand_id'] + extra), on=['pair_id', 'hand_id'], validate='1:1')
    preds = {}
    for name in ('r382_chrono3', 'ref_chrono3'):
        preds[name] = fit_predict(dd, dd, designs[name], TOP)
        log(f'    fitted {name}')
    isci = (dd['behavior_family'].to_numpy() == CI)
    hyb = np.where(isci, preds['ref_chrono3'], preds['r382_chrono3'])
    top5, top20, _ = ranked_lists(dd, hyb, TOP)
    m = float(np.mean([ap5(top5[p], truths[p]) for p in truths]))
    hits2 = {p: len(set(top5[p]) & truths[p]) for p in truths}
    out = {'reproduction_control_map5': r4(m), 'stored_step76': 0.6159,
           'passed': abs(round(m, 4) - 0.6159) < 1e-9,
           'stage2_mean_hits': r4(np.mean(list(hits2.values()))),
           'stage2_mean_hits_by_family': {f: r4(np.mean([v for p, v in hits2.items() if labels[p] == f]))
                                          for f in FAMILIES},
           'blend_on_stage2': {}}
    assert out['passed'], f"stage-2 reproduction control FAILED: {m}"
    dd = dd.with_columns(pl.Series('_s2', hyb)).filter(pl.col('oof_rank') <= TOP)

    def bframe(df, col, hi, w):
        return (df.with_columns((pl.col('_s2').rank('average').over('pair_id') / pl.len().over('pair_id')).alias('_zs'),
                                (pl.col(col).rank('average', descending=not hi).over('pair_id')
                                 / pl.len().over('pair_id')).alias('_zc'))
                .with_columns(((1 - w) * pl.col('_zs') + w * pl.col('_zc')).alias('_blend')))

    for f in FAMILIES:
        df = dd.filter(pl.col('behavior_family') == f)
        tf = {p: R for p, R in truths.items() if labels[p] == f}
        kf = {p: 5 for p in tf}
        base = {p: hits2[p] for p in tf}
        cand = [r['col'] for r in res['per_family_selection'][f]['best_single_columns']]
        grid = [(c, w) for c in cand for w in (.1, .2, .3, .4)]
        allb = {}
        for c, w in grid:
            hi = res['univariate_top20'][f][c]['auc'] > .5
            allb[f'{c}@w={w}'] = score_rule(rule_sets(bframe(df, c, hi, w), [('_blend', True)], kf), tf, labels)['mean_overlap']
        best = max(allb.items(), key=lambda kv: kv[1])
        bc, bw = best[0].split('@w=')
        hb = pair_hits(rule_sets(bframe(df, bc, res['univariate_top20'][f][bc]['auc'] > .5, float(bw)),
                                 [('_blend', True)], kf), tf)
        delta = {p: hb[p] - base[p] for p in tf}
        # honest: choose column and weight on the other tables
        hon = {}
        for t in sorted({ptab[p] for p in tf}):
            tr = {p: R for p, R in tf.items() if ptab[p] != t}
            te = {p: R for p, R in tf.items() if ptab[p] == t}
            if not tr or not te:
                continue
            dtr = df.filter(pl.col('pair_id').is_in(list(tr)))
            bb, bv = None, -1.0
            for c, w in grid:
                hi = (within_pair_stats(dtr, [c])[c]['auc'] or .5) > .5
                v = score_rule(rule_sets(bframe(dtr, c, hi, w), [('_blend', True)], {p: 5 for p in tr}),
                               tr, labels)['mean_overlap']
                if v > bv:
                    bb, bv = (c, w, hi), v
            dte = bframe(df.filter(pl.col('pair_id').is_in(list(te))), bb[0], bb[2], bb[1])
            hon.update(pair_hits(rule_sets(dte, [('_blend', True)], {p: 5 for p in te}), te))
        hg = {p: hon[p] - base[p] for p in hon}
        out['blend_on_stage2'][f] = {
            'stage2_hits': r4(np.mean(list(base.values()))),
            'best_blend': best[0], 'best_blend_hits': best[1], 'gain': r4(best[1] - np.mean(list(base.values()))),
            'paired_ci95': [r4(x) for x in pool_bootstrap(delta, ptab)],
            'pairs_better': int(sum(v > 0 for v in delta.values())), 'pairs_worse': int(sum(v < 0 for v in delta.values())),
            'leave_one_table_out_gain': r4(np.mean(list(hg.values()))),
            'leave_one_table_out_ci95': [r4(x) for x in pool_bootstrap(hg, ptab)],
            'all_blends': allb}
        b = out['blend_on_stage2'][f]
        log(f"    {f}: stage2 {b['stage2_hits']:.3f} hits -> best blend {b['best_blend']} {b['best_blend_hits']:.3f} "
            f"({b['gain']:+.3f}, CI [{b['paired_ci95'][0]:+.3f}, {b['paired_ci95'][1]:+.3f}]); honest LOTO "
            f"{b['leave_one_table_out_gain']:+.3f} CI [{b['leave_one_table_out_ci95'][0]:+.3f}, {b['leave_one_table_out_ci95'][1]:+.3f}]")
    # is the truth set a TIME TRUNCATION of a quality set?  (the step-14 test used stage-1 ranks; this
    # repeats it with the far better stage-2 probability, and adds the (type, time) variant of 14.16)
    out['earliest_of_stage2_topK'] = {}
    dk = dd.with_columns(pl.col('_s2').rank('ordinal', descending=True).over('pair_id').alias('_s2rank'))
    for K in (5, 6, 7, 8, 10, 12, 20):
        sub = dk.filter(pl.col('_s2rank') <= K)
        kall = {p: 5 for p in truths}
        out['earliest_of_stage2_topK'][f'K={K}'] = {
            'earliest': score_rule(rule_sets(sub, [('chrono_rank', False)], kall), truths, labels)['mean_overlap'],
            'fold_type_then_earliest': score_rule(rule_sets(sub, [('type_showdown', False), ('chrono_rank', False)], kall),
                                                  truths, labels)['mean_overlap'],
            'by_family_earliest': score_rule(rule_sets(sub, [('chrono_rank', False)], kall), truths, labels)['by_family_mean_overlap']}
    log('    earliest-of-stage2-top-K (pointwise five = %.3f):' % out['stage2_mean_hits'])
    for k, v in out['earliest_of_stage2_topK'].items():
        log(f"      {k:<6} earliest {v['earliest']:.3f}   (type,time) {v['fold_type_then_earliest']:.3f}   "
            + '  '.join(f"{f[:4]} {v['by_family_earliest'][f]:.2f}" for f in FAMILIES))
    out['greedy_set_aware'] = greedy_block(dd, hits2, truths, labels, ptab)
    return out


GREEDY_SIMS = ('flow_to_ben_bb', 'pot_bb', 'chrono_rank', 'oof')
GREEDY_W = (0.0, .05, .1, .15, .2, .3, .4)


def greedy_hits(dd, sim, w, truths, keep=None):
    """Pick five sequentially: the stage-2 percentile plus w x similarity to the hands already picked.
    w = 0 is the pointwise chain itself and is used as a control."""
    hits = {}
    for (pid,), g in dd.group_by('pair_id'):
        if keep is not None and pid not in keep:
            continue
        m = g.height
        s = (np.argsort(np.argsort(g['_s2'].to_numpy().astype(float))) + .5) / m
        v = (np.argsort(np.argsort(g[sim].to_numpy().astype(float))) + .5) / m
        hid = g['hand_id'].to_list()
        picked = [int(np.argmax(s))]
        for _ in range(min(4, m - 1)):
            free = np.array([i for i in range(m) if i not in picked])
            score = (1 - w) * s[free] + w * (1.0 - np.abs(v[free] - v[picked].mean()))
            picked.append(int(free[int(np.argmax(score))]))
        hits[pid] = len({hid[i] for i in picked} & truths[pid])
    return hits


def greedy_block(dd, base_hits, truths, labels, ptab):
    """The only signal that survived the cross-pair control is a SET property (a hand resembling the
    flow of the hands already listed), which a pointwise stage 2 cannot represent.  This measures what
    a set-aware greedy selection buys over the pointwise chain."""
    grid = {}
    for sim in GREEDY_SIMS:
        for w in GREEDY_W:
            grid[(sim, w)] = greedy_hits(dd, sim, w, truths)
    ctrl = float(np.mean(list(grid[(GREEDY_SIMS[0], 0.0)].values())))
    b0 = float(np.mean(list(base_hits.values())))
    out = {'control_w0_equals_pointwise': [r4(ctrl), r4(b0), abs(ctrl - b0) < 1e-9],
           'pooled': {f'{s}@w={w}': r4(np.mean(list(h.values()))) for (s, w), h in grid.items()},
           'by_family': {}}
    assert abs(ctrl - b0) < 1e-9, f'greedy w=0 control FAILED: {ctrl} vs {b0}'
    for f in FAMILIES:
        keep = {p for p in truths if labels[p] == f}
        base = float(np.mean([base_hits[p] for p in keep]))
        cells = {(s, w): float(np.mean([h[p] for p in keep])) for (s, w), h in grid.items()}
        best = max(cells.items(), key=lambda kv: kv[1])
        delta = {p: grid[best[0]][p] - base_hits[p] for p in keep}
        hon = {}
        for t in sorted({ptab[p] for p in keep}):
            tr = {p for p in keep if ptab[p] != t}
            te = {p for p in keep if ptab[p] == t}
            if not tr or not te:
                continue
            bb = max(((s, w) for s in GREEDY_SIMS for w in GREEDY_W),
                     key=lambda k: float(np.mean([grid[k][p] for p in tr])))
            for p in te:
                hon[p] = grid[bb][p]
        hg = {p: hon[p] - base_hits[p] for p in hon}
        out['by_family'][f] = {
            'pointwise_hits': r4(base), 'best': f'{best[0][0]}@w={best[0][1]}', 'best_hits': r4(best[1]),
            'gain': r4(best[1] - base), 'paired_ci95': [r4(x) for x in pool_bootstrap(delta, ptab)],
            'pairs_better': int(sum(v > 0 for v in delta.values())), 'pairs_worse': int(sum(v < 0 for v in delta.values())),
            'leave_one_table_out_gain': r4(np.mean(list(hg.values()))),
            'leave_one_table_out_ci95': [r4(x) for x in pool_bootstrap(hg, ptab)],
            'cells': {f'{s}@w={w}': r4(v) for (s, w), v in cells.items()}}
        o = out['by_family'][f]
        log(f"    greedy {f}: pointwise {o['pointwise_hits']:.3f} -> {o['best']} {o['best_hits']:.3f} "
            f"({o['gain']:+.3f}, CI [{o['paired_ci95'][0]:+.3f}, {o['paired_ci95'][1]:+.3f}]); honest LOTO "
            f"{o['leave_one_table_out_gain']:+.3f} CI [{o['leave_one_table_out_ci95'][0]:+.3f}, {o['leave_one_table_out_ci95'][1]:+.3f}]")
    return out


# ----------------------------------------------------------------------------------------------
# 7.  consistency check on all 372 labelled positive pairs (hands + seats columns only)
# ----------------------------------------------------------------------------------------------
def all372_frame():
    lab = pl.read_csv(DATA / 'development_labels.csv').filter(pl.col('label') == 1)
    players = set(lab['player_1']) | set(lab['player_2'])
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('phase') == 'development')
             .select('hand_id', 'table_id', 'started_at', 'big_blind', 'final_pot', 'players_dealt', 'board_cards'))
    seats = (pl.scan_parquet(DATA / 'seats.parquet')
             .filter(pl.col('player_id').is_in(sorted(players)))
             .select('hand_id', 'player_id', 'net_chips', 'total_contribution', 'folded', 'went_to_showdown'))
    s = seats.join(hands.select('hand_id'), on='hand_id').collect()
    a = s.join(lab.select('pair_id', pl.col('player_1').alias('player_id')), on='player_id')
    b = s.join(lab.select('pair_id', pl.col('player_2').alias('player_id')), on='player_id')
    d = a.join(b, on=['pair_id', 'hand_id'], suffix='_B').rename({
        'net_chips': 'net_A', 'total_contribution': 'put_A', 'folded': 'folded_A', 'went_to_showdown': 'sd_A',
        'net_chips_B': 'net_B', 'total_contribution_B': 'put_B', 'folded_B': 'folded_B', 'went_to_showdown_B': 'sd_B'})
    d = d.join(hands.collect(), on='hand_id', validate='m:1')
    ev = pl.read_csv(DATA / 'development_evidence.csv')
    d = d.join(ev.select('pair_id', 'hand_id', pl.lit(1).alias('relevance')), on=['pair_id', 'hand_id'], how='left')
    d = d.with_columns(pl.col('relevance').fill_null(0).cast(pl.Int64))
    d = d.join(lab.select('pair_id', 'behavior_family'), on='pair_id')
    d = d.sort('pair_id', 'started_at').with_columns(
        pl.int_range(pl.len()).over('pair_id').cast(pl.Float64).alias('chrono_rank'),
        pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'))
    d = d.with_columns(
        (pl.col('final_pot') / pl.col('big_blind')).alias('pot_bb'),
        (pl.col('net_A') / pl.col('big_blind')).alias('A_net_bb'),
        (pl.col('net_B') / pl.col('big_blind')).alias('B_net_bb'),
        ((pl.col('sd_A').cast(pl.Int64) + pl.col('sd_B').cast(pl.Int64)) > 0).cast(pl.Float64).alias('type_showdown'),
        (pl.col('put_A') + pl.col('put_B')).cast(pl.Float64).alias('pair_put'))
    d = d.with_columns((pl.col('A_net_bb') - pl.col('B_net_bb')).abs().alias('pair_net_gap'),
                       (pl.col('A_net_bb') + pl.col('B_net_bb')).alias('pair_net_bb'),
                       (pl.col('A_net_bb').sum().over('pair_id') >= pl.col('B_net_bb').sum().over('pair_id')).alias('ben_is_A'))
    d = d.with_columns((pl.when('ben_is_A').then(pl.col('A_net_bb') - pl.col('B_net_bb'))
                        .otherwise(pl.col('B_net_bb') - pl.col('A_net_bb'))).alias('flow_to_ben_bb'),
                       pl.lit(0.0).alias('oof'))
    return d.sort('pair_id', 'hand_id')


# ----------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    check_protected()
    rng = np.random.default_rng(SEED)
    d = build_frame()
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    kmap = {p: 5 for p in truths}                 # every rule emits five, exactly as the submission does
    d20 = d.filter(pl.col('oof_rank') <= TOP)

    log('=' * 100)
    log('STEP 77 [C].  WHAT PROCESS CHOSE THE FIVE LISTED HANDS?')
    log('=' * 100)
    res = {'author': AUTHOR, 'step': 77, 'training_only': True, 'held_folds_touched': False,
           'production_branch': 'none: descriptive study, nothing is proposed for submission from it',
           'population': {'pairs': len(truths), 'hands': d.height,
                          'listed_hands': int((d['relevance'] == 1).sum()),
                          'top20_hands': d20.height,
                          'listed_in_top20': int((d20['relevance'] == 1).sum()),
                          'by_family': {f: int(d.filter(pl.col('behavior_family') == f)['pair_id'].n_unique()) for f in FAMILIES}},
           'controls': {}}

    # control: the fold/showdown type counts of 14.16 must reproduce from pair_sd
    tc = (d.filter(pl.col('relevance') == 1)
          .with_columns(pl.when(pl.col('type_showdown') > 0).then(pl.lit('showdown')).otherwise(pl.lit('fold')).alias('t'))
          .group_by('behavior_family', 't').len())
    got = {(r[0], r[1]): int(r[2]) for r in tc.iter_rows()}
    res['controls']['listed_type_share_matches_14_16'] = (got == CTRL_TYPE_SHARE)
    assert got == CTRL_TYPE_SHARE, f'type control FAILED: {got}'

    # ---------------- 1 / 3. univariate ----------------
    skip = {'pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance', 'started_at', 'board_cards', 'ben_is_A'}
    cols = [c for c in d.columns if c not in skip and d.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.Boolean)]
    res['univariate_all_hands'] = univariate_block(d, cols, 'all-hands')
    res['univariate_top20'] = univariate_block(d20, cols, 'top-20')
    rank_by = sorted(cols, key=lambda c: -abs((res['univariate_top20']['ALL'][c]['auc'] or .5) - .5))
    res['univariate_top20_ranked'] = [
        {'col': c, 'auc_top20': r4(res['univariate_top20']['ALL'][c]['auc']),
         'listed_pct_top20': r4(res['univariate_top20']['ALL'][c]['mean_listed_pct']),
         'auc_all_hands': r4(res['univariate_all_hands']['ALL'][c]['auc']),
         'listed_pct_all_hands': r4(res['univariate_all_hands']['ALL'][c]['mean_listed_pct']),
         'auc_top20_by_family': {f: r4(res['univariate_top20'][f][c]['auc']) for f in FAMILIES}}
        for c in rank_by]
    res['univariate_top20_ranked_by_family'] = {}
    for f in FAMILIES:
        rb = sorted(cols, key=lambda c: -abs((res['univariate_top20'][f][c]['auc'] or .5) - .5))
        res['univariate_top20_ranked_by_family'][f] = [
            {'col': c, 'auc_top20': r4(res['univariate_top20'][f][c]['auc']),
             'listed_mean': r4(res['univariate_top20'][f][c]['listed_mean']),
             'unlisted_mean': r4(res['univariate_top20'][f][c]['unlisted_mean']),
             'auc_all_hands': r4(res['univariate_all_hands'][f][c]['auc'])} for c in rb[:20]]
    log('\n  TOP 25 COLUMNS BY WITHIN-PAIR AUC INSIDE THE TOP-20 CANDIDATES')
    log(f"    {'column':<34}{'AUC20':>8}{'pct20':>8}{'AUCall':>9}{'pctall':>8}   " + ''.join(f'{f[:6]:>8}' for f in FAMILIES))
    for r in res['univariate_top20_ranked'][:25]:
        log(f"    {r['col']:<34}{r['auc_top20']:>8.3f}{r['listed_pct_top20']:>8.3f}{r['auc_all_hands']:>9.3f}"
            f"{r['listed_pct_all_hands']:>8.3f}   " + ''.join(f"{r['auc_top20_by_family'][f]:>8.3f}" for f in FAMILIES))

    # ---------------- 2 / 3. deterministic rules ----------------
    res['rules_all_hands'], res['rules_top20'] = {}, {}
    for name, keys in RULES:
        res['rules_all_hands'][name] = score_rule(rule_sets(d, keys, kmap), truths, labels)
        res['rules_top20'][name] = score_rule(rule_sets(d20, keys, kmap), truths, labels)
    nall = dict(d.group_by('pair_id').len().iter_rows())
    n20 = dict(d20.group_by('pair_id').len().iter_rows())
    res['rules_all_hands']['random_5'] = random_baseline(truths, labels, nall)
    res['rules_top20']['random_5'] = random_baseline(truths, labels, n20)
    # earliest-of-top-K as a SELECTION rule inside the top-20 (14.11 measured it as an ORDERING rule)
    res['earliest_of_topK_selection'] = {}
    for K in (5, 6, 7, 8, 10, 12, 15, 20):
        sub = d.filter(pl.col('oof_rank') <= K)
        res['earliest_of_topK_selection'][f'K={K}'] = score_rule(rule_sets(sub, [('chrono_rank', False)], kmap), truths, labels)
    got = res['rules_all_hands']['oof_desc']['mean_overlap']
    res['controls']['oof_top5_overlap'] = {'got': got, 'stored_step76': CTRL_OOF_TOP5_OVERLAP,
                                           'passed': abs(got - CTRL_OOF_TOP5_OVERLAP) < 5e-3}
    assert res['controls']['oof_top5_overlap']['passed'], f'oof control FAILED: {got}'
    log('\n  reproduction controls PASSED (14.16 type counts; step-76 oof top-5 overlap 2.827)')

    log('\n  DETERMINISTIC RULES  (every rule emits five hands per pair)')
    log(f"    {'rule':<38}{'------ all hands ------':>25}      {'----- inside top-20 -----':>25}")
    log(f"    {'':<38}{'exact':>7}{'covers':>8}{'overlap':>10}      {'exact':>7}{'covers':>8}{'overlap':>10}")
    order = sorted(res['rules_top20'], key=lambda k: -(res['rules_top20'][k]['mean_overlap'] or 0))
    for name in order:
        a, b = res['rules_all_hands'][name], res['rules_top20'][name]
        log(f"    {name:<38}{str(a['exact_match_pairs']):>7}{str(a['contains_all_listed_pairs']):>8}{a['mean_overlap']:>10.3f}"
            f"      {str(b['exact_match_pairs']):>7}{str(b['contains_all_listed_pairs']):>8}{b['mean_overlap']:>10.3f}")
    log('\n  earliest-of-top-K as a SELECTION rule (mean overlap inside the candidate set)')
    for k, v in res['earliest_of_topK_selection'].items():
        log(f"    {k:<8}overlap {v['mean_overlap']:.3f}   exact {v['exact_match_pairs']:>3}   "
            + '  '.join(f"{f[:4]} {v['by_family_mean_overlap'][f]:.2f}" for f in FAMILIES))

    # single-column selection inside the top-20, the actionable table
    sel = {}
    for c in cols:
        a = res['univariate_top20']['ALL'][c]['auc']
        if a is None:
            continue
        sel[c] = score_rule(rule_sets(d20, [(c, a > .5)], kmap), truths, labels)['mean_overlap']
    res['single_column_top5_of_top20_overlap'] = dict(sorted(sel.items(), key=lambda kv: -kv[1])[:30])
    res['reference_overlaps'] = {'our_stage2_five': 3.333, 'best_five_in_top20_capped': 4.642,
                                 'stage1_top5': res['rules_all_hands']['oof_desc']['mean_overlap'],
                                 'random_5_of_top20': res['rules_top20']['random_5']['mean_overlap']}
    log('\n  SINGLE-COLUMN SELECTION OF 5 FROM THE TOP-20 (mean hits; stage 2 = 3.333, oracle = 4.642)')
    for c, v in list(res['single_column_top5_of_top20_overlap'].items())[:15]:
        log(f'    {c:<38}{v:>7.3f}')

    # per family: does any single column, or a rank blend of it with the ranker, beat the ranker's
    # own five inside that family?  Post-hoc column selection on the audit set: an upper bound.
    res['per_family_selection'] = {'note': 'POST-HOC: the column is chosen on this same audit set, so '
                                          'these are upper bounds, not expected gains'}
    for f in FAMILIES:
        df = d20.filter(pl.col('behavior_family') == f)
        tf = {p: R for p, R in truths.items() if labels[p] == f}
        kf = {p: 5 for p in tf}
        oofv = score_rule(rule_sets(df, [('oof', True)], kf), tf, labels)['mean_overlap']
        rows = []
        for c in cols:
            if c in ('oof', 'oof_rank'):
                continue
            a = res['univariate_top20'][f][c]['auc']
            if a is None:
                continue
            rows.append((c, score_rule(rule_sets(df, [(c, a > .5)], kf), tf, labels)['mean_overlap']))
        rows.sort(key=lambda kv: -kv[1])
        # rank blend of the ranker with the family's best three columns
        blend = {}
        for c, _ in rows[:3]:
            hi = res['univariate_top20'][f][c]['auc'] > .5
            for w in (.1, .2, .3, .4, .5):
                blend[f'{c}@w={w}'] = score_rule(rule_sets(blend_frame(df, c, hi, w), [('_blend', True)], kf),
                                                 tf, labels)['mean_overlap']
        best = max(blend.items(), key=lambda kv: kv[1])
        bcol, bw = best[0].split('@w=')
        bw = float(bw)
        # paired per-pair delta of the best blend, with a table bootstrap
        hit_o = pair_hits(rule_sets(df, [('oof', True)], kf), tf)
        hit_b = pair_hits(rule_sets(blend_frame(df, bcol, res['univariate_top20'][f][bcol]['auc'] > .5, bw),
                                    [('_blend', True)], kf), tf)
        delta = {p: hit_b[p] - hit_o[p] for p in tf}
        ci = pool_bootstrap(delta, ptab)
        # honest version: pick the column AND the weight on the other tables, apply to the held-out one
        tabs = sorted({ptab[p] for p in tf})
        hon = {}
        grid = [(c, w) for c, _ in rows[:8] for w in (.1, .2, .3, .4, .5)]
        for t in tabs:
            tr = {p: R for p, R in tf.items() if ptab[p] != t}
            te = {p: R for p, R in tf.items() if ptab[p] == t}
            if not tr or not te:
                continue
            dtr = df.filter(pl.col('pair_id').is_in(list(tr)))
            best_tr, best_v = None, -1.0
            for c, w in grid:
                up = within_pair_stats(dtr, [c])[c]['auc']
                v = score_rule(rule_sets(blend_frame(dtr, c, (up or .5) > .5, w), [('_blend', True)],
                                         {p: 5 for p in tr}), tr, labels)['mean_overlap']
                if v > best_v:
                    best_tr, best_v = (c, w, (up or .5) > .5), v
            dte = blend_frame(df.filter(pl.col('pair_id').is_in(list(te))), best_tr[0], best_tr[2], best_tr[1])
            h = pair_hits(rule_sets(dte, [('_blend', True)], {p: 5 for p in te}), te)
            hon.update(h)
        hon_gain = {p: hon[p] - hit_o[p] for p in hon}
        res['per_family_selection'][f] = {
            'pairs': len(tf), 'tables': len(tabs), 'ranker_top5_overlap': oofv,
            'best_single_columns': [{'col': c, 'overlap': v} for c, v in rows[:8]],
            'best_rank_blend': {'spec': best[0], 'overlap': best[1], 'gain_vs_ranker': r4(best[1] - oofv),
                                'paired_ci95_table_bootstrap': [r4(x) for x in ci],
                                'pairs_better': int(sum(v > 0 for v in delta.values())),
                                'pairs_worse': int(sum(v < 0 for v in delta.values()))},
            'leave_one_table_out_honest': {'overlap': r4(np.mean(list(hon.values()))),
                                           'gain_vs_ranker': r4(np.mean(list(hon_gain.values()))),
                                           'paired_ci95_table_bootstrap': [r4(x) for x in pool_bootstrap(hon_gain, ptab)],
                                           'pairs': len(hon)},
            'all_blends': blend}
        log(f'\n  {f}: ranker five = {oofv:.3f} hits;  best single column '
            f'{rows[0][0]} = {rows[0][1]:.3f};  best blend {best[0]} = {best[1]:.3f} '
            f'({best[1] - oofv:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}]);  '
            f"leave-one-table-out {res['per_family_selection'][f]['leave_one_table_out_honest']['gain_vs_ranker']:+.3f} "
            f"CI [{res['per_family_selection'][f]['leave_one_table_out_honest']['paired_ci95_table_bootstrap'][0]:+.3f}, "
            f"{res['per_family_selection'][f]['leave_one_table_out_honest']['paired_ci95_table_bootstrap'][1]:+.3f}]")

    # ---------------- 4. spacing ----------------
    res['spacing'] = spacing_block(d, labels, rng)
    log('\n  SPACING: ' + json.dumps(res['spacing']['observed']))
    log('    null(all hands): ' + json.dumps(res['spacing']['null_uniform_all_hands']))
    log('    null(top-20)   : ' + json.dumps(res['spacing']['null_uniform_top20']))
    log('    percentiles    : ' + json.dumps(res['spacing']['mean_null_percentile_of_observed']))

    # ---------------- 5. dispersion + conditional ----------------
    res['set_dispersion'] = dispersion_block(d, rng)
    log('\n  SET DISPERSION (percentile in the random-5-of-top-20 null): '
        + json.dumps(res['set_dispersion']['dispersion_percentile_lower_means_more_homogeneous']))
    log('    majority-share: ' + json.dumps(res['set_dispersion']['majority_share_percentile_higher_means_more_uniform']))
    res['conditional'] = conditional_block(d, rng, res['univariate_top20']['ALL'])
    C = res['conditional']
    log('\n  CONDITIONAL (leave-one-out context of the four other listed hands)')
    log(f"    {'similarity feature':<24}{'real':>7}{'random':>8}{'xpair':>8}{'marginal':>10}"
        f"{'d_xpair':>9}{'d_marg':>8}")
    for n in sorted(C['real_context_auc'], key=lambda k: -abs(C['delta_vs_cross_pair_context'].get(k, 0))):
        log(f"    {n:<24}{C['real_context_auc'][n]:>7.3f}{C['random_context_auc'][n]:>8.3f}"
            f"{C['cross_pair_context_auc'][n]:>8.3f}{C['marginal_auc_of_column'][n]:>10.3f}"
            f"{C['delta_vs_cross_pair_context'][n]:>+9.3f}{C['delta_vs_own_marginal'][n]:>+8.3f}")

    # ---------------- 6. does it survive stage 2? ----------------
    log('\n  STAGE-2 REFERENCE (rebuilding the shipping hybrid on the audit set)')
    res['stage2'] = stage2_block(d, cols, res, truths, labels, ptab)

    # ---------------- 7. all-372 consistency check ----------------
    log('\n  consistency check on all 372 labelled positive pairs (hands+seats columns only)')
    f372 = all372_frame()
    t372 = {k[0]: set(g['hand_id']) for k, g in f372.filter(pl.col('relevance') == 1).group_by('pair_id')}
    l372 = dict(f372.select('pair_id', 'behavior_family').unique().iter_rows())
    k372 = {p: min(5, len(R)) for p, R in t372.items()}
    cols372 = ['pot_bb', 'chrono_rank', 'pair_net_gap', 'pair_net_bb', 'flow_to_ben_bb', 'type_showdown',
               'players_dealt', 'big_blind', 'pair_put', 'n_hands']
    res['all372'] = {'label': 'CONSISTENCY CHECK ONLY - all 372 labelled positive pairs, not the audit surface',
                     'pairs': len(t372), 'hands': f372.height,
                     'univariate': {c: {'auc': r4(v['auc']), 'listed_pct': r4(v['mean_listed_pct'])}
                                    for c, v in within_pair_stats(f372, cols372).items()},
                     'univariate_by_family': {f: {c: r4(v['auc']) for c, v in
                                                  within_pair_stats(f372.filter(pl.col('behavior_family') == f), cols372).items()}
                                              for f in FAMILIES},
                     'rules': {}}
    for name, keys in [('largest_pot_bb', [('pot_bb', True)]), ('earliest', [('chrono_rank', False)]),
                       ('largest_abs_net_flow', [('pair_net_gap', True)]),
                       ('largest_flow_to_beneficiary', [('flow_to_ben_bb', True)]),
                       ('largest_pair_joint_profit', [('pair_net_bb', True)]),
                       ('earliest_fold_type_then_showdown', [('type_showdown', False), ('chrono_rank', False)])]:
        res['all372']['rules'][name] = score_rule(rule_sets(f372, keys, k372), t372, l372)
    log('    ' + json.dumps({k: v['mean_overlap'] for k, v in res['all372']['rules'].items()}))
    log('    ' + json.dumps({k: v['auc'] for k, v in res['all372']['univariate'].items()}))

    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-C-label-process.json', res)
    check_protected()
    log(f"\n  wrote {OUT / 'audit-C-label-process.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
