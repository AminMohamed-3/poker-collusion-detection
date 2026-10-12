"""Step 84 [C].  Exhaustive set-level selection of the five evidence hands.
author: claude (agent)

THE QUESTION
Stage 2 scores each of the 20 stage-1 candidates POINTWISE and keeps the top 5.  Step 77 (19.5)
established that the five listed hands are more homogeneous than random on every axis it measured and
more clustered in time than random (adjacency .0715 against a .0459 uniform null, 1.56x), and read
that as a common-cause signature rather than a selection rule.  But homogeneity is exploitable if you
select SETS rather than POINTS: the planted five should resemble each other more than they resemble
the unplanted candidates.

Step 77 tested that with a GREEDY selector over 28 hand-specified cells
(4 similarity columns {flow_to_ben_bb, pot_bb, chrono_rank, oof} x 7 weights {0,.05,.1,.15,.2,.3,.4}),
each cell being: seed with the pointwise argmax, then add hands one at a time maximising
    (1 - w) * s_pct(h) + w * (1 - |v_pct(h) - mean v_pct(picked)|)
i.e. ONE column at a time, distance to the running centroid, path-dependent, no interaction terms
beyond that.  Best pooled cell 3.3457 against 3.3333 pointwise (+.012).  Greedy can be arbitrarily far
from the optimum of an objective with interaction terms, so that is a weak test of set cohesion.

C(20,5) = 15,504.  162 pairs is 2.51 M subsets.  This step enumerates ALL of them.

    score(S) = z[ sum_{h in S} pointwise(h) ]  +  lambda * z[ cohesion(S) ]

both terms standardised over the 15,504-subset distribution WITHIN each pair, so lambda is a
scale-free "cohesion sd per utility sd" ratio that means the same thing in every pair and for every
cohesion definition (which is also what makes the matched random control below honest).
lambda = 0 must reproduce the pointwise chain EXACTLY, set for set: that is control 2.

CONTROLS (all four must pass or the result is not reported as a result)
  1  reproduction    the shipping hybrid must return MAP@5 .6159 and mean overlap 3.3333.
  2  lambda = 0      every arm must return the pointwise five, hand for hand, in every pair.
  3  matched random  the same lambda sweep with the cohesion statistic computed on a WITHIN-PAIR
                     PERMUTATION of the candidates' feature rows.  That is a relabelling, so the
                     distribution of cohesion over the 15,504 subsets is unchanged to the last value
                     and only its association with which hands were listed is destroyed.  Any gain
                     that survives lambda-sweeping a pure noise statistic is a perturbation artefact.
                     Run as a full independent replicate of the CV procedure, R times, and averaged.
     additive twin   second null of a different kind: replace the SET term with its own pointwise
                     projection, g(h) = -mean_j M[h, j] over the 20 candidates (a per-hand typicality
                     in the same feature space).  If the interaction term buys nothing over its own
                     marginal, there is no set-level structure, only a re-weighting.
  4  re-salt         the best cell re-run under two further 5-way pool partitions (step 80b's salts).
                     Single-partition numbers on 162 pairs swing +-.008-.017.

HONEST SELECTION
The pooled in-sample best cell is reported but is NOT the headline.  The headline is leave-one-table-out
cross-validation over the whole (arm x lambda x utility) grid: for each of the 104 tables the cell is
chosen on the other 103 tables and applied to that table's pairs.  A 5-way pool-hash CV is reported
beside it.  The selection surface is declared in prediction-setlevel.json, written before the frame is
built and before any number in this file exists.

Training-only: the 162 audit pairs in 104 tables (of the 172 outside every held fold).  Folds 0/1/2
are never read.  No leaderboard score and no evaluation label is used.  check_protected() at entry and
exit.  No production branch: this is a diagnostic and nothing is proposed for submission from it.

Output: Claude Version/experiments/setlevel_C_20260912/audit-C-setlevel-exhaustive.json
"""
import hashlib
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, ORIGINAL, check_protected, write_json
from evidence_aug.step5_evaluate import TIE_SEED, ap5, pool_bootstrap
from evidence_aug.step14_audit_truncation import SPLITS, TOP
from evidence_aug.step74_length_shift import CI, build_d0, designs_of, fit_predict, frame_at
from evidence_aug.step76_C_order_vs_set import ranked_lists
from evidence_aug.step77_C_label_process import add_derived

AUTHOR = 'claude (agent)'
OUTDIR = CV / 'experiments' / 'setlevel_C_20260912'
DATA = ORIGINAL / 'data'
K = 5                                    # slots
NPERM = 40                               # matched-random replicates (permutation p resolution 1/41)
PERM_SEED = 20260912
SALTS = ['audit-trunc-v1', 'audit-resplit-b', 'audit-resplit-c']   # step 80b
LAMBDAS = (0.0, .05, .1, .15, .2, .3, .4, .6, .8, 1.0, 1.5, 2.0, 3.0, 1e6)
UTILITIES = ('rank', 'prob')

# stored reproduction targets
STORED_MAP5 = 0.6159            # step 74 / 76 / 77, shipping hybrid full chrono3
STORED_HITS = 3.3333            # step 77, stage2_mean_hits
STORED_ORACLE = 4.642           # step 76, mean_relevant_in_top20_capped5
STORED_GREEDY_BEST = 3.3457     # step 77, best of the 28 greedy cells (oof@w=.15 / .20)

DECISION = {
    'hypothesis': 'the five listed hands are selectable as a SET: within-set cohesion carries '
                  'information that a pointwise stage 2 cannot represent, and greedy (step 77) was '
                  'too weak a test of it',
    'primary_estimate': 'leave-one-table-out CV over the full (utility x arm x lambda) grid, 104 '
                        'tables; the cell is chosen on the other 103 tables and applied to the '
                        'held-out table',
    'secondary_estimate': '5-way pool-hash CV (salt audit-trunc-v1)',
    'reported_but_not_headline': 'the pooled in-sample best cell',
    'alive_if': 'LOTO paired delta over the pointwise baseline is positive with pool-bootstrap 95 % '
                'lower bound > 0, AND exceeds the matched-random-cohesion LOTO delta, AND is positive '
                'under both re-salted partitions',
    'resolution': 'composite = 0.2 * delta_MAP5; the public leaderboard resolves +-.007 composite',
    'production_branch': 'none: diagnostic only',
    'held_folds': 'not read; claude held-fold read count unchanged',
    'amendment_declared_after_the_first_run': (
        'the pre-registered estimator searches all 532 cells, and on a flat surface that search is '
        'itself harmful, so it under-states any real signal.  A SECOND, more generous family of '
        'estimators was therefore added after seeing the first run and is reported beside it: for '
        'each cohesion definition on its own, LOTO over that arm\'s 28 lambda x utility cells only '
        '(the pre-specified version of the test), and the in-sample best of those 28.  Each has its '
        'own matched-random twin.  This amendment can only help the hypothesis, never hurt it, and '
        'it is disclosed rather than folded silently into the headline.')}


def log(*a):
    print(*a, flush=True)


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def resplit(d, salt):
    """the 5-way pool hash under a different salt; pools stay whole tables (step 80b)."""
    tabs = d['table_id'].unique().to_list()
    m = {t: int(hashlib.sha256((salt + ':' + t).encode()).hexdigest(), 16) % SPLITS for t in tabs}
    return d.with_columns(pl.col('table_id').replace_strict(m, return_dtype=pl.Int64).alias('split'))


# ----------------------------------------------------------------------------------------------
# enumeration scaffolding (built once; independent of the data)
# ----------------------------------------------------------------------------------------------
IDX = np.array(list(itertools.combinations(range(TOP), K)), dtype=np.int64)      # (15504, 5)
NSUB = IDX.shape[0]
_ab = list(itertools.combinations(range(K), 2))                                  # 10 slot pairs
PAIRIDX = np.stack([IDX[:, a] * TOP + IDX[:, b] for a, b in _ab])                # (10, 15504)


def zsub(v):
    """standardise a per-subset vector over the 15,504-subset distribution of this pair."""
    s = float(v.std())
    return (v - float(v.mean())) / s if s > 1e-12 else np.zeros_like(v)


def pairsum(M):
    """cohesion kernel: -sum_{i<j in S} M[i, j] for every subset, from a 20x20 matrix."""
    return -M.reshape(-1)[PAIRIDX].sum(axis=0)


def pairmin(M):
    """-min_{i<j in S} M[i, j] (used for the minimum chronological gap)."""
    return -M.reshape(-1)[PAIRIDX].min(axis=0)


def avg_rank_pct(x):
    """within-candidate average-rank percentile; ties share a rank, so a constant column gives a
    constant percentile and contributes exactly zero distance (an ordinal rank would not)."""
    n = len(x)
    order = np.argsort(x, kind='stable')
    xs = x[order]
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return (r + .5) / n


# ----------------------------------------------------------------------------------------------
# cohesion arms.  Every arm is declared here, a priori and by semantics -- never by a measured AUC
# on this surface, which would be selection before the CV could see it.
# ----------------------------------------------------------------------------------------------
FLOW = ['flow_to_ben_bb', 'xfer_to_ben_bb', 'xfer_abs_bb', 'net_ben_bb', 'net_oth_bb',
        'pair_net_bb', 'pair_net_gap', 'others_net_bb', 'loser_put_bb']
HAND = ['chen_max', 'chen_min', 'chen_ben', 'chen_oth', 'chen_gap_ben', 'folder_chen',
        'pos_gap', 'max_stack_bb', 'min_stack_bb']
POT = ['pot_bb', 'final_pot', 'pot_over_min_stack', 'max_put_bb', 'min_put_bb', 'n_board', 'dealt']


def arm_specs(struct):
    pre = [c for c in struct if c.startswith('pf_')]
    post = [c for c in struct if c.startswith('post_')] + [
        'max_street', 'n_to_flop', 'members_to_flop', 'others_to_flop', 'n_actions', 'ended_preflop',
        'pair_won_no_sd', 'opener_won', 'opener_sd', 'iso_one_member_one_other',
        'iso_one_member_vs_others', 'hu_pair_only', 'pair_won_preflop']
    return [
        # (a) structural homogeneity in a standardised feature space
        ('struct_all_dist', 'pdist', list(struct)),
        ('flow_dist', 'pdist', FLOW),
        ('flow_to_ben_dist', 'pdist', ['flow_to_ben_bb']),
        ('xfer_to_ben_dist', 'pdist', ['xfer_to_ben_bb']),
        ('preflop_dist', 'pdist', pre),
        ('postflop_dist', 'pdist', post),
        ('hand_dist', 'pdist', HAND),
        ('pot_dist', 'pdist', POT),
        # (b) temporal cohesion -- the 1.56x adjacency finding, five ways.  chrono_frac / chrono_rank
        #     enter RAW (real position in the pair's history), not as a percentile among candidates,
        #     because chronological clustering is the thing being tested; time_rankdist is the
        #     percentile-among-candidates variant, kept beside it.
        ('time_pairdist', 'pdist', ['chrono_frac']),
        ('time_var', 'sqdist', ['chrono_frac']),
        ('time_span', 'span', ['chrono_frac']),
        ('time_adjacent', 'adjacent', ['chrono_rank']),
        ('time_mingap', 'mingap', ['chrono_rank']),
        ('time_rankdist', 'pctdist', ['chrono_frac']),
        # (c) chip-flow / direction coherence (the +.082 AUC that survived step 77's cross-pair ctrl)
        ('dir_same_winner', 'agree', ['winner_is_ben']),
        ('same_showdown', 'agree', ['type_showdown']),
        # (d) cohesion in the pointwise score itself -- the greedy's best cell, and an internal probe
        #     of "is set cohesion just a re-weighting of stage 2"
        ('score_dist', 'pdist', ['_s2pct']),
        # (e) combinations
        ('flow_time_mix', 'mix', ['flow_dist', 'time_pairdist']),
        ('struct_time_mix', 'mix', ['struct_all_dist', 'time_pairdist']),
    ]


def kernel(kind, P):
    """P: (20, ncols) percentile matrix (or raw values for the time kinds).  Returns the 20x20
    matrix M with cohesion(S) = -sum_{i<j in S} M[i, j], plus the reduction to use."""
    if kind in ('pdist', 'pctdist'):
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) / P.shape[1])
        return D, 'sum'
    if kind == 'sqdist':
        return (P[:, 0][:, None] - P[:, 0][None, :]) ** 2, 'sum'
    if kind == 'adjacent':
        x = P[:, 0]
        return -(np.abs(x[:, None] - x[None, :]) == 1.0).astype(np.float64), 'sum'
    if kind == 'mingap':
        x = P[:, 0]
        M = np.abs(x[:, None] - x[None, :])
        np.fill_diagonal(M, 1e9)
        return M, 'min'
    if kind == 'agree':
        x = P[:, 0]
        return -(x[:, None] == x[None, :]).astype(np.float64), 'sum'
    raise ValueError(kind)


def cohesion_of(kind, P):
    if kind == 'span':
        x = P[:, 0]
        return -(x[IDX].max(1) - x[IDX].min(1))
    M, red = kernel(kind, P)
    return pairmin(M) if red == 'min' else pairsum(M)


def marginal_of(kind, P):
    """the arm's own pointwise projection: a per-hand typicality in the same space."""
    if kind == 'span':
        x = P[:, 0]
        return -np.abs(x - np.median(x))
    M, red = kernel(kind, P)
    Mm = M.copy()
    if red == 'min':
        np.fill_diagonal(Mm, np.nan)
        return -np.nanmin(Mm, axis=1)
    np.fill_diagonal(Mm, 0.0)
    return -Mm.sum(axis=1) / (TOP - 1)


# ----------------------------------------------------------------------------------------------
# per-pair candidate blocks
# ----------------------------------------------------------------------------------------------
def ordered_candidates(d, pred, topk):
    """the stage-1 top-20 of each pair, ordered exactly as the shipping chain orders them
    (score desc, TIE_SEED tie asc, over the pair_id/hand_id-sorted frame).  Position 0..19 IS the
    pointwise rank, so the pointwise five is positions 0..4."""
    cands = (d['oof_rank'] <= topk).to_numpy()
    z = np.where(cands, pred + 10.0, d['oof'].to_numpy() - 1e6)
    key = (d.select('pair_id', 'hand_id')
           .with_columns(pl.Series('score', z), pl.Series('cand', cands))
           .sort('pair_id', 'hand_id'))
    key = key.with_columns(pl.Series('tie', np.random.default_rng(TIE_SEED).random(key.height)))
    out = {}
    for (pid,), fr in key.group_by('pair_id'):
        srt = fr.sort(['score', 'tie'], descending=[True, False]).filter(pl.col('cand'))
        out[pid] = srt['hand_id'].to_list()
    return out


def build_blocks(d, pred, specs, cols):
    """For every pair: the ordered candidate hand ids, the two utility vectors, the truth mask, and
    the percentile / raw matrices each arm needs."""
    order = ordered_candidates(d, pred, TOP)
    dd = (d.filter(pl.col('oof_rank') <= TOP)
          .with_columns(pl.Series('_s2', pred[(d['oof_rank'] <= TOP).to_numpy()])))
    per = {}
    for (pid,), g in dd.group_by('pair_id'):
        pos = {h: i for i, h in enumerate(order[pid])}
        gg = (g.with_columns(pl.col('hand_id').replace_strict(pos, return_dtype=pl.Int64).alias('_pos'))
              .sort('_pos'))
        assert gg['hand_id'].to_list() == order[pid]
        blk = {'hands': order[pid], 'rel': gg['relevance'].to_numpy().astype(bool),
               'family': gg['behavior_family'][0]}
        raw = {c: gg[c].to_numpy().astype(np.float64) for c in cols}
        raw['_s2pct'] = avg_rank_pct(gg['_s2'].to_numpy().astype(np.float64))
        blk['pct'] = {c: avg_rank_pct(v) for c, v in raw.items()}
        blk['raw'] = raw
        u_rank = (TOP - np.arange(TOP) - .5) / TOP
        p = gg['_s2'].to_numpy().astype(np.float64)
        sd = p.std()
        blk['u'] = {'rank': u_rank,
                    'prob': ((p - p.mean()) / sd if sd > 1e-12 else np.zeros(TOP)) + 1e-9 * u_rank}
        per[pid] = blk
    return per, order


RAWCOLS = {'chrono_frac', 'chrono_rank'}


def arm_matrix(blk, name, kind, spec, perm=None):
    """the (20, ncols) matrix an arm consumes.  `perm` permutes the candidates' feature rows."""
    if kind == 'agree':
        P = np.stack([(blk['raw'][c] > 0).astype(np.float64) for c in spec], axis=1)
    elif kind == 'pctdist':
        P = np.stack([blk['pct'][c] for c in spec], axis=1)
    else:
        P = np.stack([blk['raw'][c] if c in RAWCOLS else blk['pct'][c] for c in spec], axis=1)
    return P if perm is None else P[perm]


def cohesion_table(blk, specs, perm=None):
    """every arm's standardised per-subset cohesion vector for one pair."""
    base, out = {}, {}
    for name, kind, spec in specs:
        if kind == 'mix':
            continue
        P = arm_matrix(blk, name, kind, spec, perm)
        base[name] = cohesion_of(kind, P)
        out[name] = zsub(base[name])
    for name, kind, spec in specs:
        if kind == 'mix':
            out[name] = zsub(sum(out[s] for s in spec))
    return out


def marginal_table(blk, specs):
    """the additive twin of each arm: sum over S of a per-hand typicality, standardised the same way."""
    per_hand, out = {}, {}
    for name, kind, spec in specs:
        if kind == 'mix':
            continue
        g = marginal_of(kind, arm_matrix(blk, name, kind, spec))
        s = float(g.std())
        per_hand[name] = (g - float(g.mean())) / s if s > 1e-12 else np.zeros_like(g)
        out[name] = zsub(per_hand[name][IDX].sum(1))
    for name, kind, spec in specs:
        if kind == 'mix':
            out[name] = zsub(sum(per_hand[s] for s in spec)[IDX].sum(1))
    return out


def sweep(blk, coh, truths_pid, lambdas):
    """all (utility, arm, lambda) cells for one pair -> chosen subset -> overlap and AP@5."""
    R = set(truths_pid)
    lam = np.asarray(lambdas)
    res = {}
    for uname, u in blk['u'].items():
        b = zsub(u[IDX].sum(1))
        for aname, c in coh.items():
            sc = b[None, :] + lam[:, None] * c[None, :]
            best = np.argmax(sc, axis=1)
            for li, s in enumerate(best):
                sel = IDX[s]
                hs = [blk['hands'][i] for i in sel]          # sel is ascending = pointwise order
                res[(uname, aname, float(lambdas[li]))] = (len(R & set(hs)), ap5(hs, R))
    return res


# ----------------------------------------------------------------------------------------------
# cross-validation over the cell grid
# ----------------------------------------------------------------------------------------------
def cv_apply(H, groups):
    """H: (ncells, npairs) of per-pair values.  groups: list of boolean test masks.
    For each group choose the cell that maximises the mean on the complement, apply to the group."""
    out = np.empty(H.shape[1])
    picks = []
    for te in groups:
        tr = ~te
        if not tr.any() or not te.any():
            out[te] = H[0, te]
            picks.append(None)
            continue
        j = int(np.argmax(H[:, tr].mean(axis=1)))
        out[te] = H[j, te]
        picks.append(j)
    return out, picks


def groups_by(values):
    vals = sorted(set(values))
    arr = np.asarray(values)
    return [arr == v for v in vals], vals


# ----------------------------------------------------------------------------------------------
def fit_pointwise(d, designs):
    """the shipping hybrid stage 2: r382 for transfer and soft play, ref for isolation."""
    preds = {n: fit_predict(d, d, designs[n], TOP) for n in ('r382_chrono3', 'ref_chrono3')}
    isci = (d['behavior_family'].to_numpy() == CI)
    return np.where(isci, preds['ref_chrono3'], preds['r382_chrono3'])


def main():
    t0 = time.time()
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTDIR / 'prediction-setlevel.json',
               {'author': AUTHOR, 'step': 84, 'written_before_the_frame_is_built': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'decision': DECISION, 'lambdas': list(LAMBDAS), 'utilities': list(UTILITIES),
                'n_subsets_per_pair': int(NSUB), 'matched_random_replicates': NPERM,
                'salts': SALTS, 'stored_targets': {'map5': STORED_MAP5, 'hits': STORED_HITS,
                                                   'oracle': STORED_ORACLE,
                                                   'greedy_best_step77': STORED_GREEDY_BEST}})

    log('=' * 100)
    log('STEP 84 [C].  EXHAUSTIVE C(20,5) SET-LEVEL SELECTION OF THE FIVE EVIDENCE HANDS')
    log('=' * 100)

    d0, struct, rcols = build_d0()
    designs = designs_of(struct, rcols)
    d = frame_at(d0, 1.0)
    # the derived gameplay columns step 77 defined (direction alignment, transfers, showdown type)
    # only the three columns add_derived needs and the frame lacks (players_at_showdown is already
    # a ranker column of d0; re-joining it would collide)
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
    pids = sorted(truths)
    npair = len(pids)
    log(f'  frame {d.height} rows, {npair} pairs, {len(set(ptab.values()))} tables, '
        f'{int((d["relevance"] == 1).sum())} listed hands')

    # ---------------------------------------------------------------- control 1: reproduction
    pred = fit_pointwise(d, designs)
    top5, top20, _ = ranked_lists(d, pred, TOP)
    map5 = float(np.mean([ap5(top5[p], truths[p]) for p in truths]))
    hits0 = {p: len(set(top5[p]) & truths[p]) for p in truths}
    base_hits = float(np.mean(list(hits0.values())))
    oracle = float(np.mean([min(len(top20[p] & truths[p]), 5) for p in truths]))
    c1 = {'map5': r4(map5), 'stored_map5': STORED_MAP5, 'mean_overlap': r4(base_hits),
          'stored_overlap': STORED_HITS, 'oracle_overlap': r4(oracle), 'stored_oracle': STORED_ORACLE,
          'passed': abs(round(map5, 4) - STORED_MAP5) < 1e-9 and abs(round(base_hits, 4) - STORED_HITS) < 1e-9}
    log(f"\n  CONTROL 1 reproduction: MAP@5 {map5:.4f} (stored {STORED_MAP5}), mean overlap "
        f"{base_hits:.4f} (stored {STORED_HITS}), oracle {oracle:.4f} (stored {STORED_ORACLE})  "
        f"-> {'PASSED' if c1['passed'] else 'FAILED'}")
    assert c1['passed'], f'REPRODUCTION CONTROL FAILED: map5={map5} overlap={base_hits}; STOPPING'

    # ---------------------------------------------------------------- blocks and the cell grid
    specs = arm_specs(struct)
    cols = sorted({c for _, k, s in specs if k != 'mix' for c in s} - {'_s2pct'})
    missing = [c for c in cols if c not in d.columns]
    assert not missing, f'columns absent from the frame: {missing}'
    per, order = build_blocks(d, pred, specs, cols)
    assert all(order[p][:5] == top5[p] for p in pids), 'candidate ordering does not match the chain'
    log(f'  {len(specs)} cohesion arms x {len(LAMBDAS)} lambdas x {len(UTILITIES)} utilities = '
        f'{len(specs) * len(LAMBDAS) * len(UTILITIES)} cells; {NSUB} subsets per pair '
        f'({NSUB * npair:,} enumerations per cell family)')

    t1 = time.time()
    cells = [(u, a, float(l)) for u in UTILITIES for a, _, _ in specs for l in LAMBDAS]
    cix = {c: i for i, c in enumerate(cells)}
    H = np.zeros((len(cells), npair))
    A = np.zeros((len(cells), npair))
    for pi, p in enumerate(pids):
        coh = cohesion_table(per[p], specs)
        for cell, (h, a) in sweep(per[p], coh, truths[p], LAMBDAS).items():
            H[cix[cell], pi] = h
            A[cix[cell], pi] = a
    log(f'  enumerated in {time.time() - t1:.0f} s')

    # ---------------------------------------------------------------- control 2: lambda = 0
    base_vec = np.array([hits0[p] for p in pids], dtype=float)
    z0 = [i for i, c in enumerate(cells) if c[2] == 0.0]
    c2 = {'cells_at_lambda_0': len(z0),
          'all_identical_to_pointwise_overlap': bool(np.all(H[z0] == base_vec[None, :])),
          'all_identical_to_pointwise_map5': bool(
              np.allclose(A[z0], np.array([ap5(top5[p], truths[p]) for p in pids])[None, :]))}
    c2['passed'] = c2['all_identical_to_pointwise_overlap'] and c2['all_identical_to_pointwise_map5']
    log(f"  CONTROL 2 lambda=0 reproduces the pointwise five in all {len(z0)} cells "
        f"-> {'PASSED' if c2['passed'] else 'FAILED'}")
    assert c2['passed'], 'lambda=0 CONTROL FAILED'

    # ---------------------------------------------------------------- CV
    tabs = [ptab[p] for p in pids]
    fams = [labels[p] for p in pids]
    salt_split = {t: int(hashlib.sha256(('audit-trunc-v1:' + t).encode()).hexdigest(), 16) % SPLITS
                  for t in set(tabs)}
    pools = [salt_split[t] for t in tabs]
    g_tab, _ = groups_by(tabs)
    g_pool, _ = groups_by(pools)

    def report(vec_h, vec_a, tag):
        dh = {pids[i]: vec_h[i] - base_vec[i] for i in range(npair)}
        da = {pids[i]: vec_a[i] - A[z0[0], i] for i in range(npair)}
        return {'tag': tag, 'mean_overlap': r4(vec_h.mean()), 'gain_overlap': r4(vec_h.mean() - base_vec.mean()),
                'overlap_ci95': [r4(x) for x in pool_bootstrap(dh, ptab)],
                'mean_map5': r4(vec_a.mean()), 'gain_map5': r4(vec_a.mean() - A[z0[0]].mean()),
                'map5_ci95': [r4(x) for x in pool_bootstrap(da, ptab)],
                'composite_delta': r4(.2 * (vec_a.mean() - A[z0[0]].mean())),
                'pairs_better': int(sum(v > 0 for v in dh.values())),
                'pairs_worse': int(sum(v < 0 for v in dh.values())),
                'by_family': {f: r4(np.mean([vec_h[i] for i in range(npair) if fams[i] == f]))
                              for f in FAMILIES}}

    arm_rows = {a: np.array([i for i, c in enumerate(cells) if c[1] == a]) for a, _, _ in specs}
    fam_masks = {f: np.array([x == f for x in fams]) for f in FAMILIES}

    def estimators(Hx, Ax):
        """every honest estimator, run identically on the real surface and on each null replicate.
        `loto`/`pool5` search the FULL 532-cell grid; `arm:<name>` searches only that one cohesion
        definition's 28 lambda x utility cells, which is the pre-specified version of the test."""
        out = {}
        h, picks = cv_apply(Hx, g_tab)
        a, _ = cv_apply(Ax, g_tab)
        out['loto'] = (h, a, picks)
        h, picks = cv_apply(Hx, g_pool)
        a, _ = cv_apply(Ax, g_pool)
        out['pool5'] = (h, a, picks)
        fh, fa = np.empty(npair), np.empty(npair)
        for f, m in fam_masks.items():
            gm = [g[m] for g in g_tab if (g & m).any()]
            fh[m], _ = cv_apply(Hx[:, m], gm)
            fa[m], _ = cv_apply(Ax[:, m], gm)
        out['loto_per_family'] = (fh, fa, None)
        for a_name, ix in arm_rows.items():
            h, picks = cv_apply(Hx[ix], g_tab)
            aa, _ = cv_apply(Ax[ix], g_tab)
            out['arm:' + a_name] = (h, aa, [ix[j] for j in picks if j is not None])
        return out

    real_est = estimators(H, A)
    res_cv = {}
    res_cv['loto'] = report(*real_est['loto'][:2], 'leave-one-table-out over the full 532-cell grid')
    res_cv['loto']['distinct_cells_picked'] = sorted(
        {f'{cells[j][0]}|{cells[j][1]}@{cells[j][2]:g}' for j in real_est['loto'][2] if j is not None})
    res_cv['pool5'] = report(*real_est['pool5'][:2], '5-way pool-hash CV (salt audit-trunc-v1)')
    res_cv['pool5']['cells_picked'] = [f'{cells[j][0]}|{cells[j][1]}@{cells[j][2]:g}'
                                       for j in real_est['pool5'][2] if j is not None]
    res_cv['loto_per_family'] = report(*real_est['loto_per_family'][:2],
                                       'leave-one-table-out, cell chosen per family')

    # in-sample best (reported, not the headline)
    jj = int(np.argmax(H.mean(axis=1)))
    res_cv['in_sample_best'] = report(H[jj], A[jj], f'IN-SAMPLE best cell {cells[jj]}')
    res_cv['in_sample_best']['cell'] = f'{cells[jj][0]}|{cells[jj][1]}@{cells[jj][2]:g}'

    log('\n  CROSS-VALIDATED RESULT (overlap out of 5; pointwise = %.4f, oracle = %.3f)' % (base_hits, oracle))
    for k in ('loto', 'pool5', 'loto_per_family', 'in_sample_best'):
        v = res_cv[k]
        log(f"    {k:<16} overlap {v['mean_overlap']:.4f} ({v['gain_overlap']:+.4f}, CI "
            f"[{v['overlap_ci95'][0]:+.4f}, {v['overlap_ci95'][1]:+.4f}])   MAP@5 {v['mean_map5']:.4f} "
            f"({v['gain_map5']:+.4f})   composite {v['composite_delta']:+.5f}")

    # per-arm pooled table (in-sample; diagnostic only)
    pooled = {}
    for i, c in enumerate(cells):
        pooled[f'{c[0]}|{c[1]}@{c[2]:g}'] = r4(H[i].mean())
    best_per_arm = {}
    for a, _, _ in specs:
        rows = [(c, i) for i, c in enumerate(cells) if c[1] == a]
        j = max(rows, key=lambda t: H[t[1]].mean())
        best_per_arm[a] = {'best_cell': f'{j[0][0]}@{j[0][2]:g}', 'overlap': r4(H[j[1]].mean()),
                           'gain': r4(H[j[1]].mean() - base_hits)}
    log('\n  best IN-SAMPLE cell per cohesion arm (pointwise %.4f):' % base_hits)
    for a, v in sorted(best_per_arm.items(), key=lambda kv: -kv[1]['overlap']):
        log(f"    {a:<20} {v['best_cell']:<14} {v['overlap']:.4f}  ({v['gain']:+.4f})")

    # pure cohesion (lambda -> infinity): IS the listed five the most cohesive five?  The direct
    # form of the hypothesis, with no pointwise term to hide behind.  Random five inside the top-20
    # is 1.225 (step 77, analytic).
    pure = {}
    for a, _, _ in specs:
        ix = [i for i, c in enumerate(cells) if c[1] == a and c[2] == 1e6]
        pure[a] = r4(max(H[i].mean() for i in ix))
    log('\n  PURE COHESION, lambda -> infinity: the most cohesive five (random five = 1.225, '
        'pointwise = %.3f)' % base_hits)
    for a, v in sorted(pure.items(), key=lambda kv: -kv[1]):
        log(f'    {a:<24}{v:.4f}')

    # ---------------------------------------------------------------- control 3a: matched random
    # EVERY estimator above is re-run on the null surface, so no number is reported without its twin.
    t2 = time.time()
    rng = np.random.default_rng(PERM_SEED)
    keys = list(real_est)
    null_h = {k: [] for k in keys}
    null_a = {k: [] for k in keys}
    null_best, null_best_map5 = [], []
    null_best_arm = {a: [] for a in arm_rows}
    null_best_arm_map5 = {a: [] for a in arm_rows}
    for _ in range(NPERM):
        Hn = np.zeros_like(H)
        An = np.zeros_like(A)
        for pi, p in enumerate(pids):
            coh = cohesion_table(per[p], specs, perm=rng.permutation(TOP))
            for cell, (h, a) in sweep(per[p], coh, truths[p], LAMBDAS).items():
                Hn[cix[cell], pi] = h
                An[cix[cell], pi] = a
        est = estimators(Hn, An)
        for k in keys:
            null_h[k].append(float(est[k][0].mean()))
            null_a[k].append(float(est[k][1].mean()))
        mm = Hn.mean(axis=1)
        jn = int(np.argmax(mm))
        null_best.append(float(mm[jn]))
        null_best_map5.append(float(An[jn].mean()))
        for a_name, ix in arm_rows.items():
            ma = Hn[ix].mean(axis=1)
            ja = int(np.argmax(ma))
            null_best_arm[a_name].append(float(ma[ja]))
            null_best_arm_map5[a_name].append(float(An[ix[ja]].mean()))
    ap0 = A[z0[0]].mean()

    def pval(nulls, real):
        """one-sided permutation p: how often a matched noise statistic reaches the real one."""
        return r4((1 + sum(1 for x in nulls if x >= real - 1e-12)) / (1 + len(nulls)))

    table = {}
    for k in keys:
        rh = float(real_est[k][0].mean())
        ra = float(real_est[k][1].mean())
        nh = float(np.mean(null_h[k]))
        na = float(np.mean(null_a[k]))
        table[k] = {'real_overlap': r4(rh), 'real_gain': r4(rh - base_hits),
                    'null_overlap': r4(nh), 'null_gain': r4(nh - base_hits),
                    'real_minus_null_overlap': r4(rh - nh),
                    'real_gain_map5': r4(ra - ap0), 'null_gain_map5': r4(na - ap0),
                    'real_minus_null_map5': r4(ra - na),
                    'real_minus_null_composite': r4(.2 * (ra - na)),
                    'perm_p_overlap': pval(null_h[k], rh), 'perm_p_map5': pval(null_a[k], ra),
                    'null_spread': [r4(min(null_h[k])), r4(max(null_h[k]))]}
    jg = int(np.argmax(H.mean(axis=1)))
    table['in_sample_best_of_grid'] = {
        'cell': f'{cells[jg][0]}|{cells[jg][1]}@{cells[jg][2]:g}',
        'real_overlap': r4(H[jg].mean()), 'real_gain': r4(H[jg].mean() - base_hits),
        'null_overlap': r4(np.mean(null_best)), 'null_gain': r4(np.mean(null_best) - base_hits),
        'real_minus_null_overlap': r4(H[jg].mean() - np.mean(null_best)),
        'real_gain_map5': r4(A[jg].mean() - ap0),
        'null_gain_map5': r4(np.mean(null_best_map5) - ap0),
        'real_minus_null_map5': r4(A[jg].mean() - np.mean(null_best_map5)),
        'real_minus_null_composite': r4(.2 * (A[jg].mean() - np.mean(null_best_map5))),
        'perm_p_overlap': pval(null_best, float(H[jg].mean())),
        'null_spread': [r4(min(null_best)), r4(max(null_best))]}
    for a_name, ix in arm_rows.items():
        ja = ix[int(np.argmax(H[ix].mean(axis=1)))]
        table['in_sample_best_of_arm:' + a_name] = {
            'cell': f'{cells[ja][0]}|{cells[ja][1]}@{cells[ja][2]:g}',
            'real_overlap': r4(H[ja].mean()), 'real_gain': r4(H[ja].mean() - base_hits),
            'null_overlap': r4(np.mean(null_best_arm[a_name])),
            'null_gain': r4(np.mean(null_best_arm[a_name]) - base_hits),
            'real_minus_null_overlap': r4(H[ja].mean() - np.mean(null_best_arm[a_name])),
            'real_minus_null_map5': r4(A[ja].mean() - np.mean(null_best_arm_map5[a_name])),
            'real_minus_null_composite': r4(.2 * (A[ja].mean() - np.mean(null_best_arm_map5[a_name]))),
            'perm_p_overlap': pval(null_best_arm[a_name], float(H[ja].mean()))}
    c3 = {'replicates': NPERM,
          'what': "cohesion recomputed on a within-pair permutation of the candidates' feature rows: "
                  'the subset-distribution of the statistic is identical to the last value, only its '
                  'association with which hands were listed is destroyed.  Every estimator is re-run '
                  'end to end on each null surface, so each real number has a matched twin.',
          'table': table,
          'headline_real_minus_null_loto': table['loto']['real_minus_null_overlap'],
          'headline_real_minus_null_best_prespecified_arm': max(
              table['in_sample_best_of_arm:' + a]['real_minus_null_overlap'] for a in arm_rows)}
    c3['passed'] = bool(max(table[k]['real_minus_null_overlap'] for k in keys) > 0)
    log(f'\n  CONTROL 3a matched random cohesion, {NPERM} replicates ({time.time() - t2:.0f} s).'
        f'\n    every estimator against its own null twin (overlap out of 5):')
    log(f"    {'estimator':<28}{'real':>8}{'null':>9}{'real-null':>11}{'composite(r-n)':>16}{'perm p':>9}")
    for k in ('loto', 'pool5', 'loto_per_family'):
        v = table[k]
        log(f"    {k:<28}{v['real_overlap']:>8.4f}{v['null_overlap']:>9.4f}"
            f"{v['real_minus_null_overlap']:>+11.4f}{v['real_minus_null_composite']:>+16.5f}"
            f"{v['perm_p_overlap']:>9.3f}")
    v = table['in_sample_best_of_grid']
    log(f"    {'best-of-532 (in sample)':<28}{v['real_overlap']:>8.4f}{v['null_overlap']:>9.4f}"
        f"{v['real_minus_null_overlap']:>+11.4f}{'':>16}{v['perm_p_overlap']:>9.3f}")
    log('\n    pre-specified single-arm grids (28 cells each), LOTO over lambda x utility only:')
    log(f"    {'arm':<24}{'LOTO real':>10}{'null':>8}{'r-n':>9}{'p':>7}   "
        f"{'best real':>9}{'null':>8}{'r-n':>9}{'p':>7}")
    for a_name in sorted(arm_rows, key=lambda x: -table['arm:' + x]['real_minus_null_overlap']):
        v = table['arm:' + a_name]
        w = table['in_sample_best_of_arm:' + a_name]
        log(f"    {a_name:<24}{v['real_overlap']:>10.4f}{v['null_overlap']:>8.4f}"
            f"{v['real_minus_null_overlap']:>+9.4f}{v['perm_p_overlap']:>7.3f}   "
            f"{w['real_overlap']:>9.4f}{w['null_overlap']:>8.4f}{w['real_minus_null_overlap']:>+9.4f}"
            f"{w['perm_p_overlap']:>7.3f}")

    # ---------------------------------------------------------------- control 3b: additive twin
    Hm = np.zeros_like(H)
    Am = np.zeros_like(A)
    for pi, p in enumerate(pids):
        mg = marginal_table(per[p], specs)
        for cell, (h, a) in sweep(per[p], mg, truths[p], LAMBDAS).items():
            Hm[cix[cell], pi] = h
            Am[cix[cell], pi] = a
    add_h, _ = cv_apply(Hm, g_tab)
    add_a, _ = cv_apply(Am, g_tab)
    c3b = {'what': 'the SET term replaced by its own pointwise projection g(h) = -mean_j M[h, j]; '
                   'if the interaction buys nothing over its own marginal there is no set structure',
           'additive_loto': report(add_h, add_a, 'additive twin, LOTO'),
           'additive_in_sample_best_overlap': r4(Hm.mean(axis=1).max())}
    c3b['set_minus_additive_loto'] = r4(res_cv['loto']['gain_overlap'] - c3b['additive_loto']['gain_overlap'])
    log(f"  CONTROL 3b additive twin: LOTO {c3b['additive_loto']['gain_overlap']:+.4f} vs set-level "
        f"{res_cv['loto']['gain_overlap']:+.4f}   -> interaction worth "
        f"{c3b['set_minus_additive_loto']:+.4f}")

    # ---------------------------------------------------------------- control 4: re-salt
    # the cell the LOTO procedure picks most often is the one whose stability is worth checking
    cnt = Counter(j for j in real_est['loto'][2] if j is not None)
    modal = cells[cnt.most_common(1)[0][0]]
    probe = [('modal_loto_cell', modal), ('in_sample_best_cell', cells[jj])]
    # the best cell that actually USES cohesion, so the re-salt is never vacuous even when the CV
    # settles on lambda = 0
    knz = int(np.argmax(np.where(np.array([c[2] > 0 for c in cells]), H.mean(axis=1), -1e9)))
    probe.append(('best_nonzero_lambda_cell', cells[knz]))
    seen, todo = set(), []
    for tag, c in probe:
        if c not in seen:
            seen.add(c)
            todo.append((tag, c))
    c4 = {'chosen_as': {t: f'{c[0]}|{c[1]}@{c[2]:g}' for t, c in todo},
          'modal_loto_share': r4(cnt.most_common(1)[0][1] / max(1, sum(cnt.values()))),
          'partitions': {}}
    log(f"\n  CONTROL 4 re-salt.  cells probed: "
        + ';  '.join(f'{t} = {c[0]}|{c[1]}@{c[2]:g}' for t, c in todo))
    for salt in SALTS:
        ds = resplit(d, salt)
        ps = pred if salt == SALTS[0] else fit_pointwise(ds, designs)
        t5s, _, _ = ranked_lists(ds, ps, TOP)
        hb = {p: len(set(t5s[p]) & truths[p]) for p in truths}
        ab = {p: ap5(t5s[p], truths[p]) for p in truths}
        pers, _ = build_blocks(ds, ps, specs, cols)
        got = {t: ({}, {}) for t, _ in todo}
        for p in pids:
            sw = sweep(pers[p], cohesion_table(pers[p], specs), truths[p], LAMBDAS)
            for t, c in todo:
                got[t][0][p], got[t][1][p] = sw[c]
        c4['partitions'][salt] = {'pointwise_overlap': r4(np.mean(list(hb.values()))),
                                  'pointwise_map5': r4(np.mean(list(ab.values()))), 'cells': {}}
        for t, _ in todo:
            hs, as_ = got[t]
            dh = {p: hs[p] - hb[p] for p in pids}
            da = {p: as_[p] - ab[p] for p in pids}
            c4['partitions'][salt]['cells'][t] = {
                'set_overlap': r4(np.mean(list(hs.values()))), 'set_map5': r4(np.mean(list(as_.values()))),
                'delta_overlap': r4(np.mean(list(dh.values()))), 'delta_map5': r4(np.mean(list(da.values()))),
                'delta_overlap_ci95': [r4(x) for x in pool_bootstrap(dh, ptab)],
                'composite_delta': r4(.2 * np.mean(list(da.values())))}
        v = c4['partitions'][salt]
        log(f"    {salt:<18} pointwise {v['pointwise_overlap']:.4f}  " + '  '.join(
            f"{t.split('_')[0]} {v['cells'][t]['set_overlap']:.4f} "
            f"({v['cells'][t]['delta_overlap']:+.4f})" for t, _ in todo))
    c4['min_delta_overlap'] = {t: r4(min(c4['partitions'][s]['cells'][t]['delta_overlap'] for s in SALTS))
                               for t, _ in todo}
    c4['positive_under_all_partitions'] = {t: bool(v > 0) for t, v in c4['min_delta_overlap'].items()}
    c4['modal_cell_positive_under_all_partitions'] = c4['positive_under_all_partitions']['modal_loto_cell']

    # ---------------------------------------------------------------- verdict
    lo = res_cv['loto']
    alive = (lo['gain_overlap'] > 0 and lo['overlap_ci95'][0] > 0
             and table['loto']['real_minus_null_overlap'] > 0
             and c4['modal_cell_positive_under_all_partitions'])
    # the most generous honest reading available: best pre-specified single-arm grid, against its
    # own matched null.  Reported so the negative cannot be blamed on the search being too wide.
    ba = max(arm_rows, key=lambda a: table['in_sample_best_of_arm:' + a]['real_minus_null_overlap'])
    gen = table['in_sample_best_of_arm:' + ba]
    verdict = {
        'ALIVE_or_DEAD': 'ALIVE' if alive else 'DEAD',
        'deciding_number': 'the best of the 532 exhaustive set-level cells buys '
                           f"{table['in_sample_best_of_grid']['real_gain']:+.4f} overlap in sample; "
                           'the best of a matched-random grid of the same size buys '
                           f"{table['in_sample_best_of_grid']['null_gain']:+.4f}.  Difference "
                           f"{table['in_sample_best_of_grid']['real_minus_null_overlap']:+.4f} "
                           'hits/pair.  Cross-validated (leave-one-table-out over the grid) the set '
                           f"selector returns {lo['gain_overlap']:+.4f} hits/pair "
                           f"[{lo['overlap_ci95'][0]:+.4f}, {lo['overlap_ci95'][1]:+.4f}], MAP@5 "
                           f"{lo['gain_map5']:+.4f}, composite {lo['composite_delta']:+.5f}",
        'most_generous_reading': {
            'arm': ba, 'cell': gen['cell'],
            'what': "in-sample best of that one arm's 28 cells, against its own matched null",
            'real_gain': gen['real_gain'], 'null_gain': gen['null_gain'],
            'real_minus_null_overlap': gen['real_minus_null_overlap'],
            'real_minus_null_map5': gen['real_minus_null_map5'],
            'real_minus_null_composite': gen['real_minus_null_composite']},
        'clears_public_resolution_0.007': bool(abs(lo['composite_delta']) >= .007),
        'criteria': {'loto_gain_positive': bool(lo['gain_overlap'] > 0),
                     'loto_ci_lower_above_zero': bool(lo['overlap_ci95'][0] > 0),
                     'beats_matched_random': bool(table['loto']['real_minus_null_overlap'] > 0),
                     'positive_under_all_resalts': c4['positive_under_all_partitions']}}

    res = {'author': AUTHOR, 'step': 84, 'training_only': True, 'held_folds_touched': False,
           'production_branch': 'none: diagnostic only, nothing proposed for submission',
           'question': DECISION['hypothesis'],
           'what_step77_tested': {
               'selector': 'GREEDY, path-dependent: seed with the pointwise argmax, then add hands one '
                           'at a time maximising (1-w)*s_pct(h) + w*(1 - |v_pct(h) - mean v_pct(picked)|)',
               'cells': '28 = 4 similarity columns x 7 weights',
               'similarity_columns': ['flow_to_ben_bb', 'pot_bb', 'chrono_rank', 'oof'],
               'weights': [0.0, .05, .1, .15, .2, .3, .4],
               'set_features_available': 'one column at a time, absolute distance to the running '
                                         'centroid of the already-picked hands, in within-pair '
                                         'percentile space; no multi-column distance, no variance, '
                                         'span, adjacency or agreement statistic, no interaction '
                                         'beyond the centroid term, and no exhaustive optimisation',
               'result': f'best pooled cell {STORED_GREEDY_BEST} against {STORED_HITS} pointwise'},
           'population': {'pairs': npair, 'tables': len(set(tabs)), 'hands': d.height,
                          'listed_hands': int((d['relevance'] == 1).sum()),
                          'subsets_per_pair': int(NSUB), 'cells': len(cells)},
           'arms': {a: {'kind': k, 'columns': s if k != 'mix' else s} for a, k, s in specs},
           'lambdas': list(LAMBDAS), 'utilities': list(UTILITIES),
           'controls': {'1_reproduction': c1, '2_lambda_zero': c2, '3a_matched_random': c3,
                        '3b_additive_twin': c3b, '4_resalt': c4},
           'cross_validated': res_cv, 'best_in_sample_cell_per_arm': best_per_arm,
           'pure_cohesion_lambda_infinity': {
               'note': 'the most cohesive five, no pointwise term.  Random five inside the top-20 is '
                       '1.225 (step 77, analytic); the pointwise chain is 3.3333.  The binary-agreement '
                       'and min-gap arms sit high only because their cohesion saturates on many subsets '
                       'and the pointwise term breaks the tie -- they are the pointwise five inside a '
                       'homogeneous class, and they are still WORSE than the pointwise five.',
               'overlap': pure},
           'pooled_in_sample_cells': pooled,
           'reference_points': {'pointwise_overlap': r4(base_hits), 'pointwise_map5': r4(map5),
                                'oracle_overlap': r4(oracle),
                                'step77_greedy_best_pooled': STORED_GREEDY_BEST},
           'verdict': verdict, 'seconds': round(time.time() - t0)}
    write_json(OUTDIR / 'audit-C-setlevel-exhaustive.json', res)
    check_protected()
    log(f"\n  VERDICT: {verdict['ALIVE_or_DEAD']}  --  {verdict['deciding_number']}")
    log(f"  criteria: {verdict['criteria']}")
    log(f"  wrote {OUTDIR / 'audit-C-setlevel-exhaustive.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
