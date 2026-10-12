"""Step 89 [C].  Do we hold several models that are each good at DIFFERENT things?
author: claude

THE QUESTION
The classic Kaggle move is composition: keep N models that each win on a different slice and route
between them.  Nobody here has measured whether our variants are actually complementary.  The
scattered evidence that they might be is unusually consistent and all points one way --
coordinated_isolation (CI) behaves differently in kind from directed_transfer (DT) and soft_play (SP):

  * 19.17  retrieval misses per pair DT .111 / SP .106 / CI .674; top-20 oracle 4.819 / 4.660 / 4.326.
           CI is bottlenecked at STAGE 1; DT and SP at SELECTION, which is dead five times over.
  * 14.41  the 382 stage-1 ranker columns inside the per-family stage 2 help DT and SP; the shipping
           model is the resulting `r382DTSP` hybrid, i.e. we already do family composition once, and
           it is labelled post-hoc.
  * Codex RZ  overall +0.011996 but coordinated_isolation -0.010697, independent lane and family.
  * step 84  of four set-level estimators only `loto_per_family` -- the cell chosen PER FAMILY -- was
           positive after its matched null (+.0872 overlap, composite +.0034, p=.122).
  * ledger  `family_routing_CI_to_struct` NEAR-NULL on held, +.0029 [-.0029, +.0084].

WHAT THIS STEP DOES
  Phase 1  build one tidy per-pair, per-variant table on the 162-pair training audit surface.
  Phase 2  correlation, pairwise win fractions, and the ORACLE-ROUTER CEILING (per pair, per family).
  Phase 3  price the ceiling against MATCHED NULLS.  An oracle over N variants on 162 pairs looks
           excellent from selection noise alone; a router that does not beat its matched null is
           measuring noise.  Three nulls, all declared before any number exists:
             null_resid_perm_global  (PRIMARY)  decompose ap[v,p] = m[p] + e[v,p] with m the
                              across-variant pair mean, then permute each variant's residual across
                              ALL pairs.  Preserves each variant's POOLED mean (marginal quality) and
                              the common "some pairs are easy" component that induces most of the
                              between-variant correlation, and destroys all complementarity, per pair
                              AND per family.  This is the sharp null.
             null_resid_perm_family  the same permuted WITHIN FAMILY.  Stricter for the per-pair
                              question because it grants the family effect to the null, and -- said
                              in advance -- VACUOUS for the per-family oracle, which it cannot move
                              at all because it preserves every per-family mean exactly.  That is
                              precisely why the global scope, not this one, is primary.
             null_full_perm_{global,family}  permute each variant's whole per-pair vector.
                              Preserves the score DISTRIBUTION exactly but also decorrelates the
                              variants, which INFLATES the oracle -- so these are hard nulls and are
                              reported as such, not used alone to kill anything.
             null_seed_family 7 seed-variants of the production stage-1 ranker (read, not refitted,
                              from step 85's stage1-oof-matrix.parquet) each run through the incumbent
                              stage 2.  Seven "different models" that differ only by a random seed:
                              the noise floor for an oracle over 7 models of this quality.
           plus CROSS-SALT TRANSFERRED ORACLE, which needs no synthetic null at all: choose the
           per-pair winner on partition A and apply it to partition B.  If the winner pattern is
           genuine it transfers; if it is selection noise it collapses to the mean variant.
  Phase 4  is it exploitable?  (a) PER-FAMILY routing, leave-one-table-out -- the cheap special case
           that could actually ship, since it needs only the predicted family; (b) a learned per-pair
           router over label-free features (family, candidate-set statistics, stage-1 score
           distribution, between-variant agreement), 5-way pool-hash OOF.  Both against their nulls.
  Phase 5  the CI retrieval question 19.17 says is the live one, as an explicit UPPER BOUND.

THE COUPLING TAX (OBS 19.16) AND WHERE IT APPLIES
  `experiments/stage1_ensemble_C_20260912/audit-C-capacity-followup.json` -> `loss_decomposition`:
  of 16 correct final picks lost when stage 1 changed, only -5 came from top-20 eviction/addition and
  -11 came from RE-RANKING among candidates that never left the top 20.  Stage 2 is fitted around one
  particular rank distribution, so ANY change of the stage-1 output feeding a frozen stage 2 pays
  ~11/16 of the loss through re-ranking alone.
    -> The seven REAL variants here all sit on ONE frozen production stage-1 score and therefore on
       one identical top-20 candidate set.  Routing between them never changes stage 1, so the
       primary routing numbers of this step are tax-free BY CONSTRUCTION.  That is a design choice,
       recorded, not an assumption.
    -> `null_seed_family` DOES change stage 1 and DOES pay the tax; its numbers are a property of the
       incumbent pipeline, not of those models.  Said again beside every number it touches.
    -> Phase 5's CI-retrieval counterfactual is a stage-1 intervention and pays the tax in full.
       Its conversion to a composite is therefore an upper bound twice over.
  Step 88 is separately measuring the tax by refitting both stages; this step does not duplicate it
  and does not touch experiments/joint_refit_C_20260912/.

PROTOCOL
  Training only: the 162 audit pairs in 104 tables outside every held fold.  Folds 0/1/2 are NOT read
  for truth; load_fold(0) supplies FEATURE COLUMNS only, exactly as steps 74/76/77/84/85b/87.
  check_protected() at entry and exit, both hash dicts recorded.
  Three salts (audit-trunc-v1, audit-resplit-b, audit-resplit-c); the MINIMUM across partitions is
  the reported result, because single-partition numbers on 162 pairs swing +-.008-.017 (step 80b).
  prediction-complementarity.json is written BEFORE any fit and never edited.
  Materiality: 0.2 * min_delta * 0.49 >= 0.007, i.e. min_delta >= 0.0714.
  Artefacts append-only into experiments/complementarity_C_20260912/.  Nothing is proposed for
  submission: this is diagnostic.
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl

from evidence_aug.fold import fold_pools
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, check_protected, write_json
from evidence_aug.step5_evaluate import ap5, pool_bootstrap
from evidence_aug.step14_audit_truncation import SPLITS, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step74_length_shift import CHRONO3, CI, S1, build_d0, fit_predict, frame_at
from evidence_aug.step76_C_order_vs_set import ranked_lists
from evidence_aug.step78_C_stage2_selection import fit_predict_rank

AUTHOR = 'claude'
STEP = 89
OUTDIR = CV / 'experiments' / 'complementarity_C_20260912'
S1MATRIX = CV / 'experiments' / 'stage1_ensemble_C_20260912' / 'stage1-oof-matrix.parquet'
SALTS = ['audit-trunc-v1', 'audit-resplit-b', 'audit-resplit-c']
NULL_SEEDS = (7329, 101, 202, 303, 404, 505, 606)   # 7 of step 85's 8, to match N=7 real variants
NPERM = 400              # replicates for the oracle nulls (pure numpy)
NPERM_ROUTER = 20        # replicates for the router null (each one refits the router)
PERM_SEED = 20260913
LEVER, TRANSFER, RESOLUTION = 0.2, 0.49, 0.007
MIN_DELTA = round(RESOLUTION / (LEVER * TRANSFER), 4)     # 0.0714
STORED_HYBRID = 0.6159         # step 74/76/85, the shipping r382DTSP hybrid on this surface
STORED_REF = 0.5867            # step 74, ref_full_chrono3 at prefix 1.0
STORED_R382 = 0.6222           # step 74, r382_full_chrono3 at prefix 1.0
STORED_CI_MISS = {'directed_transfer': 0.1111, 'soft_play': 0.1064, 'coordinated_isolation': 0.6744}
STORED_CI_ORACLE = {'directed_transfer': 4.8194, 'soft_play': 4.6596, 'coordinated_isolation': 4.3256}
ROUTER = dict(n_estimators=120, num_leaves=5, max_depth=3, min_child_samples=20,
              learning_rate=.05, n_jobs=2, verbosity=-1)

GATE = {
    'question': 'do we hold several model variants that are each good at DIFFERENT pairs, such that '
                'routing between them beats the best single variant by more than selection noise?',
    'surface': 'the 162 audit pairs in 104 tables outside every held fold; per-pair AP@5, the '
               'quantity the composite is 0.2 x the mean of',
    'variant_family': 'SEVEN stage-2 variants on ONE frozen production stage-1 score: s1_pure, ref, '
                      'r382, struct_nos1, r382_nostruct, rank_ref, rank_r382',
    'why_one_stage1': 'so that routing never changes stage 1 and the coupling tax of 19.16 cannot '
                      'contaminate the routing number',
    'primary_estimate': 'MINIMUM across the three pool partitions of (deployable routed MAP@5 - best '
                        'single variant MAP@5), where "deployable" means the LEAVE-ONE-TABLE-OUT '
                        'per-family router and the OOF learned per-pair router, both compared with a '
                        'best single variant chosen by the same honest rule',
    'secondary_estimate': 'the ORACLE-ROUTER CEILING (per-pair best, and separately per-family best) '
                          'minus its matched null; an oracle is not a lever and is reported as a '
                          'ceiling only',
    'nulls': {
        'null_resid_perm_global': 'PRIMARY for every headline. ap[v,p] = m[p] + e[v,p] with m the '
                                  'across-variant pair mean; permute e[v,.] across ALL pairs, '
                                  'independently per variant.  Preserves each variant POOLED mean '
                                  '(marginal quality) and the common per-pair component that induces '
                                  'most of the between-variant correlation, and destroys all '
                                  'complementarity, per pair AND per family.',
        'null_resid_perm_family': 'STRICTER, and only meaningful for the per-pair question: the same '
                                  'construction permuted WITHIN family, so each variant per-family '
                                  'mean is preserved exactly.  Declared in advance as VACUOUS for '
                                  'the per-family oracle -- it cannot move that statistic at all, '
                                  'which is why the global scope is primary.',
        'null_full_perm_global': 'permute each variant whole per-pair vector.  Preserves the pooled '
                                 'score distribution exactly but also decorrelates the variants, '
                                 'which INFLATES the oracle: a HARD null, declared here as such and '
                                 'not used alone to kill anything.',
        'null_full_perm_family': 'the same, permuted within family.',
        'null_seed_family': '7 production-stage-1 SEED variants (read from step 85 stage1-oof-matrix, '
                            'not refitted) run through the incumbent stage 2.  Different models that '
                            'differ only by a seed.  NOTE: this family changes stage 1 and therefore '
                            'PAYS THE COUPLING TAX of 19.16; the real family does not.',
        'cross_salt_transfer': 'no synthetic null: choose the per-pair (and per-family) winner on '
                               'partition A, apply it to partition B.'},
    'complementarity_exists_if': 'min over salts of (oracle gain - null_resid_perm_global oracle '
                                 'gain) > 0, evaluated separately for the per-pair and the '
                                 'per-family oracle',
    'exploitable_if': 'min over salts of (routed - best_single) > 0 for a DEPLOYABLE router AND that '
                      'quantity exceeds the same quantity under null_resid_perm_global',
    'material_if': f'{LEVER} * min_delta * {TRANSFER} >= {RESOLUTION}  (min_delta >= {MIN_DELTA})',
    'verdict': 'ALIVE only if exploitable AND material.  An oracle that beats its null but a router '
               'that does not is recorded as REAL BUT NOT EXPLOITABLE, which is not ALIVE.',
    'per_family_routing_is_reported_separately_and_prominently':
        'it needs only the predicted family, which we already have, so it is the one form that could '
        'ship; the incumbent r382DTSP hybrid is exactly one fixed instance of it and is measured here '
        'against plain r382 on the same surface',
    'known_worse_arm_check': 'the standing 18.15/18.17 check: an arm known to be worse must fail the '
                             'gate.  Here: routing every pair to `s1_pure` (no stage 2 at all) must '
                             'produce a large negative delta.',
    'reproduction_controls': ['the r382DTSP hybrid must return MAP@5 0.6159 (steps 74/76/85)',
                              'ref must return 0.5867 and r382 0.6222 (step 74 prefix 1.0)',
                              'CI retrieval misses per pair must be 0.6744 and the CI top-20 oracle '
                              'overlap 4.3256 (19.17), and DT/SP 0.1111/0.1064 and 4.8194/4.6596'],
    'coupling_tax': 'the seven real variants share one frozen stage-1 score, so routing among them is '
                    'tax-free; null_seed_family and the Phase 5 counterfactual both change stage 1 '
                    'and pay it, and are labelled wherever they appear',
    'resolution': 'composite = 0.2 x delta MAP@5; the public leaderboard resolves +-.007 composite',
    'production_branch': 'none: diagnostic only, nothing is proposed for submission',
    'held_folds': 'folds 0, 1, 2 are not read for truth; load_fold(0) supplies FEATURE COLUMNS only',
    'training_only': 'the 162 audit pairs in 104 tables outside every held fold',
    'note': 'written before any number in this file was computed'}


# --------------------------------------------------------------------------------------------
# frame / scoring helpers
# --------------------------------------------------------------------------------------------
def resplit(d, salt):
    """the 5-way pool hash under a different salt; pools stay whole tables (step 14 / 80b / 85)."""
    tabs = d['table_id'].unique().to_list()
    m = {t: int(hashlib.sha256((salt + ':' + t).encode()).hexdigest(), 16) % SPLITS for t in tabs}
    return d.with_columns(pl.col('table_id').replace_strict(m, return_dtype=pl.Int64).alias('split'))


def score_of(d, pred, truths, pairs):
    """per-pair AP@5 and per-pair overlap for one prediction vector, using step 76's exact ordering."""
    top5, top20, _ = ranked_lists(d, pred, TOP)
    ap = np.array([ap5(top5[p], truths[p]) for p in pairs])
    ov = np.array([len(set(top5[p]) & truths[p]) for p in pairs], dtype=float)
    return ap, ov, top5


def fam_means(a, fams):
    return {f: float(a[fams == f].mean()) for f in FAMILIES}


def r4(x):
    return round(float(x), 4)


# --------------------------------------------------------------------------------------------
# oracle-router algebra.  A is (P, V) of per-pair scores; fams is (P,) of family strings.
# --------------------------------------------------------------------------------------------
def oracle_block(A, fams, names):
    P, V = A.shape
    mean_v = A.mean(axis=0)
    j_best = int(np.argmax(mean_v))
    best_single = float(mean_v[j_best])
    oracle_pair = float(A.max(axis=1).mean())
    routed = np.empty(P)
    fam_choice = {}
    for f in FAMILIES:
        idx = np.where(fams == f)[0]
        j = int(np.argmax(A[idx].mean(axis=0)))
        fam_choice[f] = names[j]
        routed[idx] = A[idx, j]
    oracle_family = float(routed.mean())
    return {'best_single': best_single, 'best_single_name': names[j_best],
            'oracle_pair': oracle_pair, 'oracle_pair_gain': oracle_pair - best_single,
            'oracle_family': oracle_family, 'oracle_family_gain': oracle_family - best_single,
            'family_choice': fam_choice, 'per_variant_mean': {n: float(m) for n, m in zip(names, mean_v)}}


def _blocks(fams, scope):
    """the permutation blocks: one per family ('family' scope) or one big block ('global')."""
    if scope == 'family':
        return [np.where(fams == f)[0] for f in FAMILIES]
    return [np.arange(len(fams))]


def null_resid_perm(A, fams, rng, scope='global'):
    """keep the across-variant pair mean m[p], permute each variant's residual e[v,.] within block.

    scope='global'  every variant's POOLED mean is preserved and its per-family profile is
                    randomised: the matched null for "is there any complementarity at all",
                    per pair OR per family.
    scope='family'  every variant's PER-FAMILY mean is preserved exactly, so the family effect is
                    granted to the null for free and only within-family per-pair alignment is
                    destroyed.  A STRICTER null for the per-pair question and, by construction, a
                    VACUOUS one for the per-family oracle (which it cannot move at all).
    """
    m = A.mean(axis=1, keepdims=True)
    E = A - m
    B = np.empty_like(A)
    for idx in _blocks(fams, scope):
        for v in range(A.shape[1]):
            B[idx, v] = m[idx, 0] + E[rng.permutation(idx), v]
    return B


def null_full_perm(A, fams, rng, scope='global'):
    """HARD null: permute each variant's whole per-pair vector within block (also decorrelates)."""
    B = np.empty_like(A)
    for idx in _blocks(fams, scope):
        for v in range(A.shape[1]):
            B[idx, v] = A[rng.permutation(idx), v]
    return B


def null_summary(A, fams, names, maker, nperm, seed, scope):
    rng = np.random.default_rng(seed)
    gp, gf = [], []
    for _ in range(nperm):
        b = oracle_block(maker(A, fams, rng, scope), fams, names)
        gp.append(b['oracle_pair_gain'])
        gf.append(b['oracle_family_gain'])
    gp, gf = np.array(gp), np.array(gf)
    real = oracle_block(A, fams, names)
    return {'replicates': nperm,
            'null_oracle_pair_gain_mean': r4(gp.mean()), 'null_oracle_pair_gain_p95': r4(np.quantile(gp, .95)),
            'null_oracle_family_gain_mean': r4(gf.mean()), 'null_oracle_family_gain_p95': r4(np.quantile(gf, .95)),
            'real_minus_null_oracle_pair': r4(real['oracle_pair_gain'] - gp.mean()),
            'real_minus_null_oracle_family': r4(real['oracle_family_gain'] - gf.mean()),
            'perm_p_oracle_pair': r4((1 + int((gp >= real['oracle_pair_gain']).sum())) / (nperm + 1)),
            'perm_p_oracle_family': r4((1 + int((gf >= real['oracle_family_gain']).sum())) / (nperm + 1))}


# --------------------------------------------------------------------------------------------
# deployable routers
# --------------------------------------------------------------------------------------------
def loto_routers(A, fams, tabs, names):
    """leave-one-table-out: choose the variant on the other 103 tables, apply to this table's pairs."""
    P = A.shape[0]
    per_fam = np.empty(P)
    glob = np.empty(P)
    picks_fam, picks_glob = [], []
    for t in sorted(set(tabs)):
        te = np.where(tabs == t)[0]
        tr = np.where(tabs != t)[0]
        jg = int(np.argmax(A[tr].mean(axis=0)))
        glob[te] = A[te, jg]
        picks_glob.append(names[jg])
        for f in FAMILIES:
            k = te[fams[te] == f]
            if len(k) == 0:
                continue
            trf = tr[fams[tr] == f]
            j = int(np.argmax(A[trf].mean(axis=0))) if len(trf) else jg
            per_fam[k] = A[k, j]
            picks_fam.append((f, names[j]))
    return per_fam, glob, picks_fam, picks_glob


def pair_features(d, preds, names, pairs, labels, sl, top5s):
    """label-free per-pair router features: family, candidate-set stats, stage-1 distribution,
    per-variant score shape, and between-variant agreement.  No truth is used anywhere here."""
    cand = (d['oof_rank'] <= TOP).to_numpy()
    oof = d['oof'].to_numpy()
    cfrac = d['chrono_frac'].to_numpy()
    nh = d['n_hands'].to_numpy()
    rows = []
    for i, (a, b) in enumerate(sl):
        c = cand[a:b]
        o = np.sort(oof[a:b][c])[::-1]
        oall = oof[a:b]
        cf = np.sort(cfrac[a:b][c])
        row = {'n_hands': float(nh[a]), 'log_n_hands': float(np.log1p(nh[a])), 'n_cand': float(c.sum()),
               'oof_max': float(o[0]), 'oof_mean': float(o.mean()), 'oof_sd': float(o.std()),
               'oof_min': float(o[-1]), 'oof_gap12': float(o[0] - o[1]),
               'oof_gap15': float(o[0] - o[min(4, len(o) - 1)]),
               'oof_gap56': float(o[min(4, len(o) - 1)] - o[min(5, len(o) - 1)]),
               'oof_gap520': float(o[min(4, len(o) - 1)] - o[-1]),
               'oof_all_sd': float(oall.std()),
               'oof_frac_above_cut': float((oall >= o[-1]).mean()),
               'cand_cfrac_mean': float(cf.mean()), 'cand_cfrac_sd': float(cf.std()),
               'cand_cfrac_span': float(cf[-1] - cf[0]),
               'cand_cfrac_mingap': float(np.min(np.diff(cf))) if len(cf) > 1 else 0.0}
        rows.append(row)
    for n in names:                                    # per-variant score shape, within-pair z
        pv = preds[n]
        for i, (a, b) in enumerate(sl):
            c = cand[a:b]
            x = pv[a:b][c].astype(float)
            z = (x - x.mean()) / (x.std() + 1e-12)
            s = np.sort(z)[::-1]
            rows[i][f'{n}__pmax'] = float(s[0])
            rows[i][f'{n}__pgap56'] = float(s[min(4, len(s) - 1)] - s[min(5, len(s) - 1)])
            rows[i][f'{n}__ptop5sum'] = float(s[:5].sum())
    V = len(names)
    for i, p in enumerate(pairs):                      # between-variant agreement on the chosen five
        sets = [set(top5s[n][p]) for n in names]
        ov = []
        per_v = np.zeros(V)
        for x in range(V):
            for y in range(x + 1, V):
                k = len(sets[x] & sets[y])
                ov.append(k)
                per_v[x] += k
                per_v[y] += k
        ov = np.array(ov, dtype=float)
        rows[i]['agree_mean'] = float(ov.mean())
        rows[i]['agree_min'] = float(ov.min())
        rows[i]['agree_sd'] = float(ov.std())
        rows[i]['agree_union'] = float(len(set().union(*sets)))
        for x, n in enumerate(names):
            rows[i][f'{n}__agree'] = float(per_v[x] / (V - 1))
    for f in FAMILIES:
        for i, p in enumerate(pairs):
            rows[i]['fam_' + f] = 1.0 if labels[p] == f else 0.0
    cols = sorted(rows[0])
    X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    return X, cols


def learned_router(X, A, split, names, seed=11):
    """per-variant OOF regression on label-free features; route each pair to the argmax prediction."""
    P, V = A.shape
    hat = np.zeros((P, V))
    for v in range(V):
        for s in range(SPLITS):
            te = np.where(split == s)[0]
            tr = np.where(split != s)[0]
            if len(te) == 0 or len(tr) == 0:
                continue
            m = lgb.LGBMRegressor(**ROUTER, random_state=seed).fit(X[tr], A[tr, v])
            hat[te, v] = m.predict(X[te])
    pick = hat.argmax(axis=1)
    return float(A[np.arange(P), pick].mean()), pick, hat


def pairwise_router(X, A, split, ia, ib, seed=11):
    """the cheapest learned form: A-vs-B for the two best variants, OOF classifier on the sign."""
    P = A.shape[0]
    y = (A[:, ia] > A[:, ib]).astype(int)
    tie = A[:, ia] == A[:, ib]
    out = np.empty(P)
    for s in range(SPLITS):
        te = np.where(split == s)[0]
        tr = np.where((split != s) & ~tie)[0]
        if len(te) == 0 or len(tr) == 0 or len(set(y[tr])) < 2:
            out[te] = A[te, ia]
            continue
        m = lgb.LGBMClassifier(**ROUTER, random_state=seed).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        out[te] = np.where(p >= .5, A[te, ia], A[te, ib])
    return float(out.mean())


# --------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    entry = check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    write_json(OUTDIR / 'prediction-complementarity.json',
               {'author': AUTHOR, 'step': STEP, 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'gate': GATE, 'salts': list(SALTS), 'null_seeds': list(NULL_SEEDS),
                'nperm_oracle': NPERM, 'nperm_router': NPERM_ROUTER,
                'materiality': {'lever': LEVER, 'transfer_discount': TRANSFER,
                                'public_resolution': RESOLUTION, 'min_delta_required': MIN_DELTA},
                'reused_without_refit': [str(S1MATRIX)],
                'not_touched': ['experiments/joint_refit_C_20260912/ (step 88, another agent)',
                                'OBSERVATIONS.md', 'held folds 0/1/2 truth']})

    print('=' * 100)
    print('STEP 89 [C].  ARE OUR VARIANTS COMPLEMENTARY, AND CAN A ROUTER EXPLOIT IT?')
    print('=' * 100)

    # ---- frame ------------------------------------------------------------------------------
    d0, struct, rcols = build_d0()
    d = frame_at(d0, 1.0)
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    assert not (set(d['table_id']) & held), 'audit surface overlaps a held fold -- stop'
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pairs = sorted(truths)
    fams = np.array([labels[p] for p in pairs])
    tabs = np.array([ptab[p] for p in pairs])
    P = len(pairs)
    uniq = d['pair_id'].to_numpy()
    first = np.unique(uniq, return_index=True)[1]
    starts = np.sort(first)
    ends = list(starts[1:]) + [d.height]
    sl = list(zip(starts, ends))
    assert [uniq[a] for a, _ in sl] == pairs, 'pair order mismatch between frame slices and pairs list'

    designs = {'s1_pure': ('none', []),
               'ref': ('clf', struct + CHRONO3 + S1),
               'r382': ('clf', struct + CHRONO3 + S1 + rcols),
               'struct_nos1': ('clf', struct + CHRONO3),
               'r382_nostruct': ('clf', rcols + CHRONO3 + S1),
               'rank_ref': ('rank', struct + CHRONO3 + S1),
               'rank_r382': ('rank', struct + CHRONO3 + S1 + rcols)}
    names = list(designs)
    V = len(names)

    res = {'author': AUTHOR, 'step': STEP, 'training_only': True, 'held_folds_touched': False,
           'held_folds_note': 'folds 0, 1, 2 are not read for truth; load_fold(0) supplies FEATURE '
                              'COLUMNS only, exactly as steps 74/76/77/84/85b/87',
           'production_branch': 'none: diagnostic only, nothing is proposed for submission',
           'question': GATE['question'], 'gate': GATE,
           'prediction_file': str(OUTDIR / 'prediction-complementarity.json'),
           'prediction_written_before_any_fit': True,
           'protected_hashes_entry': entry,
           'population': {'pairs': P, 'tables': int(d['table_id'].n_unique()), 'hands': d.height,
                          'listed_hands': int((d['relevance'] == 1).sum()),
                          'pairs_by_family': {f: int((fams == f).sum()) for f in FAMILIES},
                          'n_struct': len(struct), 'n_ranker': len(rcols)},
           'coupling_tax': {
               'source': 'OBS 19.16 / experiments/stage1_ensemble_C_20260912/audit-C-capacity-'
                         'followup.json -> loss_decomposition',
               'magnitude': '-11 of -16 lost correct final picks came from RE-RANKING among '
                            'candidates that never left the top 20, only -5 from eviction/addition',
               'applies_to_this_step': {
                   'seven_real_variants': 'NO. They share one frozen production stage-1 score and '
                                          'therefore one identical top-20 candidate set; routing '
                                          'between them never changes stage 1. Tax-free by design.',
                   'null_seed_family': 'YES. Seven stage-1 seeds through a frozen stage 2: every '
                                       'number in that block is a property of the incumbent pipeline '
                                       'and not of those models.',
                   'phase5_ci_retrieval': 'YES, in full. A CI-specific retrieval improvement is a '
                                          'stage-1 intervention; its realised value is below the '
                                          'proportional estimate for exactly this reason.'},
               'frozen_vs_refit': 'a frozen/refit contrast is not applicable to the primary routing '
                                  'number because stage 1 is common to all seven arms and stage 2 is '
                                  'refit inside every arm by construction (fit_predict is per arm). '
                                  'The refit side of the stage-1 question is step 88 and is not '
                                  'duplicated here.'},
           'phase1_inventory': {}, 'phase2_complementarity': {}, 'phase3_nulls': {},
           'phase4_deployable': {}, 'phase5_ci_retrieval': {}, 'resalt_minima': {}, 'verdict': {},
           'limits': []}

    res['phase1_inventory']['variants'] = {
        n: {'kind': {'none': 'no stage 2 (stage-1 order inside the top 20)',
                     'clf': 'per-family LightGBM classifier, 5-way pool-hash OOF (step 14 CLF)',
                     'rank': 'per-family LightGBM lambdarank, 5-way pool-hash OOF (step 78 RANKER)'}[k],
            'n_columns': len(c), 'stage1': 'frozen production oof (audit-oof.parquet)'}
        for n, (k, c) in designs.items()}
    res['phase1_inventory']['reused_without_refit'] = [
        {'artefact': str(S1MATRIX), 'what': '24 stage-1 OOF score vectors (8 seeds x 3 salts) at hand '
                                            'level from step 85', 'used_for': 'null_seed_family; '
                                            'no stage-1 ranker is refitted anywhere in this step'},
        {'artefact': 'experiments/label_process_20260911/audit-oof.parquet',
         'what': 'the frozen production stage-1 score and the audit truth',
         'used_for': 'the common stage 1 of all seven real variants'},
        {'artefact': 'experiments/label_process_20260911/audit-structural-features.parquet',
         'what': 'the 70 structural stage-2 columns', 'used_for': 'every stage-2 design'}]
    res['phase1_inventory']['could_not_reuse'] = [
        {'what': 'per-pair AP@5 vectors for any model variant',
         'searched': ['experiments/label_process_20260911/audit-{stage2-ensemble,r382-variants,'
                      'hybrid-combos,length-shift,f2,family-stage2}.json',
                      'experiments/stage2_selection_C_20260912/*.json',
                      'experiments/setlevel_C_20260912/audit-C-setlevel-exhaustive.json',
                      'experiments/pair_side_C_20260912/*.json',
                      'experiments/evidence_confusion_C_20260912/audit-C-evidence-confusion.json',
                      'experiments/stage1_ensemble_C_20260912/audit-C-stage1-ensemble.json'],
         'finding': 'every one of them stores POOLED and PER-FAMILY MAP@5 (and for some, paired '
                    'mean deltas and pairs_better/pairs_worse counts) but NOT the per-pair vector. '
                    'The 40+ variants already scored in those files are therefore an inventory of '
                    'which designs exist and how good they are, not a reusable per-pair table.',
         'consequence': 'the seven stage-2 arms are re-fitted here. That is not a duplicated '
                        'experiment: the stage-2 classifier random_state is inert (OBS 14.41 quirk, '
                        're-proved in step 80b), so ref and r382 must reproduce step 74 exactly, and '
                        'the reproduction control asserts it.'},
        {'what': 'a per-pair table for Codex RZ or any TabPFN lane',
         'finding': 'those live on a different pool reference (common65) whose absolute MAP is not '
                    'comparable with this surface, and outside Claude Version; not used.'}]

    # ---- Phase 1: per-pair, per-variant scores, three salts -----------------------------------
    print('\n  PHASE 1.  per-pair scoring of 7 variants x 3 partitions')
    Amat, Ovmat, PREDS, TOP5 = {}, {}, {}, {}
    per_variant = {}
    assert np.array_equal(resplit(d, SALTS[0])['split'].to_numpy(), d['split'].to_numpy()), \
        'resplit under the v1 salt does not reproduce the stored split column'
    for salt in SALTS:
        ds = resplit(d, salt)
        A = np.zeros((P, V))
        Ov = np.zeros((P, V))
        preds, top5s = {}, {}
        for j, n in enumerate(names):
            t1 = time.time()
            kind, cols = designs[n]
            if kind == 'none':
                pred = ds['oof'].to_numpy().astype(float)
            elif kind == 'clf':
                pred = fit_predict(ds, ds, cols, TOP)
            else:
                pred = fit_predict_rank(ds, ds, cols, TOP)
            ap, ov, t5 = score_of(ds, pred, truths, pairs)
            A[:, j], Ov[:, j] = ap, ov
            preds[n], top5s[n] = pred, t5
            per_variant.setdefault(salt, {})[n] = {
                'map5': r4(ap.mean()), 'by_family': {k: r4(v) for k, v in fam_means(ap, fams).items()},
                'mean_overlap': r4(ov.mean())}
            log(f'{salt} {n}', json.dumps({'map5': r4(ap.mean()), 'ov': r4(ov.mean()),
                                           'sec': round(time.time() - t1)}))
        # the incumbent shipping composition, free from ref and r382
        jr, jf = names.index('r382'), names.index('ref')
        hyb = np.where(fams == CI, A[:, jf], A[:, jr])
        per_variant[salt]['_incumbent_hybrid_r382DTSP'] = {
            'map5': r4(hyb.mean()), 'by_family': {k: r4(v) for k, v in fam_means(hyb, fams).items()},
            'mean_overlap': None, 'what': 'r382 for DT/SP, ref for CI: the model that ships'}
        Amat[salt], Ovmat[salt], PREDS[salt], TOP5[salt] = A, Ov, preds, top5s

    A1 = Amat[SALTS[0]]
    ctrl = {'hybrid_r382DTSP': per_variant[SALTS[0]]['_incumbent_hybrid_r382DTSP']['map5'],
            'hybrid_stored': STORED_HYBRID,
            'ref': per_variant[SALTS[0]]['ref']['map5'], 'ref_stored': STORED_REF,
            'r382': per_variant[SALTS[0]]['r382']['map5'], 'r382_stored': STORED_R382}
    ctrl['pass'] = bool(abs(ctrl['hybrid_r382DTSP'] - STORED_HYBRID) < 1e-9
                        and abs(ctrl['ref'] - STORED_REF) < 1e-9
                        and abs(ctrl['r382'] - STORED_R382) < 1e-9)
    assert ctrl['pass'], f'reproduction control FAILED: {ctrl}'
    res['phase1_inventory']['reproduction_control'] = ctrl
    res['phase1_inventory']['per_variant'] = per_variant
    print(f"  reproduction control PASSED: hybrid {ctrl['hybrid_r382DTSP']:.4f} == {STORED_HYBRID}, "
          f"ref {ctrl['ref']:.4f}, r382 {ctrl['r382']:.4f}")

    # ---- Phase 2: is there complementarity at all? --------------------------------------------
    print('\n  PHASE 2.  correlation, win fractions, oracle ceilings')
    for salt in SALTS:
        A = Amat[salt]
        blk = oracle_block(A, fams, names)
        corr = np.corrcoef(A, rowvar=False)
        wins = {}
        for x in range(V):
            for y in range(x + 1, V):
                a, b = A[:, x], A[:, y]
                wins[f'{names[x]}|{names[y]}'] = {
                    'corr': r4(corr[x, y]),
                    'A_beats_B': r4((a > b).mean()), 'B_beats_A': r4((b > a).mean()),
                    'tied': r4((a == b).mean()),
                    'mean_delta': r4((a - b).mean()),
                    'pair_oracle_gain_over_better': r4(np.maximum(a, b).mean() - max(a.mean(), b.mean()))}
        fam_blocks = {}
        for f in FAMILIES:
            idx = np.where(fams == f)[0]
            mv = A[idx].mean(axis=0)
            fam_blocks[f] = {'best_variant': names[int(np.argmax(mv))],
                             'best_value': r4(mv.max()),
                             'per_variant': {n: r4(x) for n, x in zip(names, mv)},
                             'oracle_pair': r4(A[idx].max(axis=1).mean()),
                             'oracle_pair_gain': r4(A[idx].max(axis=1).mean() - mv.max())}
        res['phase2_complementarity'][salt] = {
            'per_variant_mean': {n: r4(x) for n, x in blk['per_variant_mean'].items()},
            'best_single': r4(blk['best_single']), 'best_single_name': blk['best_single_name'],
            'oracle_pair': r4(blk['oracle_pair']), 'oracle_pair_gain': r4(blk['oracle_pair_gain']),
            'oracle_family': r4(blk['oracle_family']), 'oracle_family_gain': r4(blk['oracle_family_gain']),
            'oracle_family_choice': blk['family_choice'],
            'mean_offdiag_correlation': r4(corr[np.triu_indices(V, 1)].mean()),
            'min_offdiag_correlation': r4(corr[np.triu_indices(V, 1)].min()),
            'pairwise': wins, 'by_family': fam_blocks,
            'pairs_where_all_variants_agree_on_score': int((A.max(axis=1) == A.min(axis=1)).sum())}
        print(f"    {salt:<18} best {blk['best_single_name']:<14}{blk['best_single']:.4f}   "
              f"oracle/pair {blk['oracle_pair']:.4f} (+{blk['oracle_pair_gain']:.4f})   "
              f"oracle/family {blk['oracle_family']:.4f} (+{blk['oracle_family_gain']:.4f})")

    # ---- Phase 3: matched nulls ---------------------------------------------------------------
    print('\n  PHASE 3.  matched nulls for the oracle ceiling')
    for salt in SALTS:
        A = Amat[salt]
        res['phase3_nulls'][salt] = {
            'null_resid_perm_global': null_summary(A, fams, names, null_resid_perm, NPERM,
                                                   PERM_SEED, 'global'),
            'null_resid_perm_family': null_summary(A, fams, names, null_resid_perm, NPERM,
                                                   PERM_SEED + 1, 'family'),
            'null_full_perm_global': null_summary(A, fams, names, null_full_perm, NPERM,
                                                  PERM_SEED + 2, 'global'),
            'null_full_perm_family': null_summary(A, fams, names, null_full_perm, NPERM,
                                                  PERM_SEED + 3, 'family'),
            'note_on_the_family_scope': 'null_*_perm_family cannot move the per-family oracle by '
                                        'construction (it preserves every per-family mean exactly); '
                                        'its per-family columns are therefore reported as an '
                                        'identity check, not as a null. Declared in the gate.'}
        r = res['phase3_nulls'][salt]['null_resid_perm_global']
        rf = res['phase3_nulls'][salt]['null_resid_perm_family']
        print(f"    {salt:<18} GLOBAL resid null: oracle/pair gain {r['null_oracle_pair_gain_mean']:+.4f}"
              f"  real-null {r['real_minus_null_oracle_pair']:+.4f} p={r['perm_p_oracle_pair']}"
              f"  |  family real-null {r['real_minus_null_oracle_family']:+.4f}"
              f" p={r['perm_p_oracle_family']}"
              f"   ||  FAMILY-scope resid null: pair real-null "
              f"{rf['real_minus_null_oracle_pair']:+.4f} p={rf['perm_p_oracle_pair']}")

    # cross-salt transferred oracle: no synthetic null needed
    trans = {}
    for a in SALTS:
        for b in SALTS:
            if a == b:
                continue
            Aa, Ab = Amat[a], Amat[b]
            pick = Aa.argmax(axis=1)
            tp = float(Ab[np.arange(P), pick].mean())
            fam_pick = {}
            routed = np.empty(P)
            for f in FAMILIES:
                idx = np.where(fams == f)[0]
                j = int(np.argmax(Aa[idx].mean(axis=0)))
                fam_pick[f] = names[j]
                routed[idx] = Ab[idx, j]
            bb = oracle_block(Ab, fams, names)
            trans[f'{a}->{b}'] = {
                'transferred_pair_oracle': r4(tp),
                'gain_over_best_single_on_b': r4(tp - bb['best_single']),
                'own_pair_oracle_on_b': r4(bb['oracle_pair']),
                'fraction_of_own_oracle_gain_retained':
                    r4((tp - bb['best_single']) / (bb['oracle_pair'] - bb['best_single'] + 1e-12)),
                'transferred_family_router': r4(routed.mean()),
                'family_gain_over_best_single_on_b': r4(routed.mean() - bb['best_single']),
                'family_choice_from_a': fam_pick}
    res['phase3_nulls']['cross_salt_transfer'] = trans
    tpg = [v['gain_over_best_single_on_b'] for v in trans.values()]
    tfg = [v['family_gain_over_best_single_on_b'] for v in trans.values()]
    res['phase3_nulls']['cross_salt_transfer_summary'] = {
        'pair_oracle_transfer_gain_min': r4(min(tpg)), 'pair_oracle_transfer_gain_mean': r4(np.mean(tpg)),
        'family_transfer_gain_min': r4(min(tfg)), 'family_transfer_gain_mean': r4(np.mean(tfg)),
        'reading': 'the per-pair winner chosen on one partition, applied to another. This needs no '
                   'synthetic null: if the winner pattern were genuine it would transfer.'}
    print(f"    cross-salt transferred per-pair oracle gain: min {min(tpg):+.4f} "
          f"mean {np.mean(tpg):+.4f}   per-family: min {min(tfg):+.4f} mean {np.mean(tfg):+.4f}")

    # null_seed_family: 7 stage-1 seed variants through the incumbent stage 2.  PAYS THE COUPLING TAX.
    print('\n  PHASE 3b.  null_seed_family (7 stage-1 seeds x incumbent stage 2) -- PAYS THE COUPLING TAX')
    mat = pl.read_parquet(S1MATRIX).sort('pair_id', 'hand_id')
    assert d.select('pair_id', 'hand_id').equals(mat.select('pair_id', 'hand_id')), 'matrix keys differ'
    seed_names = [f's1seed_{s}' for s in NULL_SEEDS]
    res['phase3_nulls']['null_seed_family'] = {
        'what': f'{len(NULL_SEEDS)} production stage-1 ranker seeds {list(NULL_SEEDS)} READ from step '
                f"85's stage1-oof-matrix.parquet (no ranker is refitted), each run through the "
                f'incumbent stage 2 (r382 for DT/SP, ref for CI)',
        'why_seven': 'to match N=7 real variants exactly, since an oracle ceiling grows with N; '
                     "step 85's eighth seed (707) is dropped for that reason and for no other",
        'COUPLING_TAX': 'this family changes stage 1 under a frozen stage 2, so every number below '
                        'carries the 19.16 tax and is a property of the incumbent pipeline, not of '
                        'these models. The seven REAL variants do not carry it.',
        'by_salt': {}}
    for salt in SALTS:
        ds = resplit(d, salt)
        B = np.zeros((P, len(NULL_SEEDS)))
        for j, s in enumerate(NULL_SEEDS):
            t1 = time.time()
            v = mat[f'{salt}|prod|{s}'].to_numpy().astype(float)
            dd = ds.with_columns(pl.Series('oof', v))
            dd = dd.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id')
                                 .cast(pl.UInt32).alias('oof_rank'))
            pr = fit_predict(dd, dd, struct + CHRONO3 + S1 + rcols, TOP)
            pf = fit_predict(dd, dd, struct + CHRONO3 + S1, TOP)
            apr, _, _ = score_of(dd, pr, truths, pairs)
            apf, _, _ = score_of(dd, pf, truths, pairs)
            B[:, j] = np.where(fams == CI, apf, apr)
            log(f'{salt} s1seed {s}', json.dumps({'map5': r4(B[:, j].mean()),
                                                  'sec': round(time.time() - t1)}))
        blk = oracle_block(B, fams, seed_names)
        res['phase3_nulls']['null_seed_family']['by_salt'][salt] = {
            'per_variant_mean': {n: r4(x) for n, x in blk['per_variant_mean'].items()},
            'best_single': r4(blk['best_single']), 'best_single_name': blk['best_single_name'],
            'spread': r4(max(blk['per_variant_mean'].values()) - min(blk['per_variant_mean'].values())),
            'oracle_pair': r4(blk['oracle_pair']), 'oracle_pair_gain': r4(blk['oracle_pair_gain']),
            'oracle_family': r4(blk['oracle_family']), 'oracle_family_gain': r4(blk['oracle_family_gain']),
            'mean_offdiag_correlation': r4(np.corrcoef(B, rowvar=False)[np.triu_indices(len(NULL_SEEDS), 1)].mean()),
            'real_family_oracle_pair_gain': res['phase2_complementarity'][salt]['oracle_pair_gain'],
            'real_minus_seedfamily_oracle_pair':
                r4(res['phase2_complementarity'][salt]['oracle_pair_gain'] - blk['oracle_pair_gain']),
            'real_minus_seedfamily_oracle_family':
                r4(res['phase2_complementarity'][salt]['oracle_family_gain'] - blk['oracle_family_gain'])}
        z = res['phase3_nulls']['null_seed_family']['by_salt'][salt]
        print(f"    {salt:<18} seed-family oracle/pair gain {z['oracle_pair_gain']:+.4f}   "
              f"real - seedfamily {z['real_minus_seedfamily_oracle_pair']:+.4f}")

    # ---- Phase 4: is it exploitable? ----------------------------------------------------------
    print('\n  PHASE 4.  deployable routers')
    for salt in SALTS:
        A = Amat[salt]
        ds = resplit(d, salt)
        sp = dict(ds.select('pair_id', 'split').unique().iter_rows())
        split = np.array([sp[p] for p in pairs])
        blk = oracle_block(A, fams, names)

        pf, gl, picks_fam, picks_glob = loto_routers(A, fams, tabs, names)
        dpair = {pairs[i]: float(pf[i] - gl[i]) for i in range(P)}
        ci95 = pool_bootstrap(dpair, ptab)

        X, fcols = pair_features(ds, PREDS[salt], names, pairs, labels, sl, TOP5[salt])
        routed, pick, _ = learned_router(X, A, split, names)
        ia = names.index(blk['best_single_name'])
        second = int(np.argsort(-A.mean(axis=0))[1])
        pw = pairwise_router(X, A, split, ia, second)

        nulls = {}
        for scope in ('global', 'family'):
            rng = np.random.default_rng(PERM_SEED + 7 + (0 if scope == 'global' else 100))
            nl_pf, nl_ro, nl_pw = [], [], []
            for _ in range(NPERM_ROUTER):
                Bn = null_resid_perm(A, fams, rng, scope)
                npf, ngl, _, _ = loto_routers(Bn, fams, tabs, names)
                nl_pf.append(float(npf.mean() - ngl.mean()))
                nb = oracle_block(Bn, fams, names)
                nr, _, _ = learned_router(X, Bn, split, names)
                nl_ro.append(nr - nb['best_single'])
                nl_pw.append(pairwise_router(X, Bn, split, int(np.argmax(Bn.mean(axis=0))),
                                             int(np.argsort(-Bn.mean(axis=0))[1])) - nb['best_single'])
            nulls[scope] = {'pf': nl_pf, 'ro': nl_ro, 'pw': nl_pw}
        nl_pf, nl_ro, nl_pw = nulls['global']['pf'], nulls['global']['ro'], nulls['global']['pw']
        jr, jf = names.index('r382'), names.index('ref')
        hyb = np.where(fams == CI, A[:, jf], A[:, jr])

        res['phase4_deployable'][salt] = {
            'best_single_in_sample': r4(blk['best_single']), 'best_single_name': blk['best_single_name'],
            'per_family_loto': {
                'what': 'for each of the 104 tables, the variant is chosen per family on the other '
                        '103 tables and applied to this table pairs. THE CHEAP SPECIAL CASE: it needs '
                        'only the predicted family, which the pipeline already has.',
                'routed_map5': r4(pf.mean()),
                'global_loto_map5': r4(gl.mean()),
                'delta_vs_global_loto': r4(pf.mean() - gl.mean()),
                'delta_vs_best_single_in_sample': r4(pf.mean() - blk['best_single']),
                'paired_ci95_vs_global_loto': [r4(x) for x in ci95],
                'modal_family_pick': {f: max({n: sum(1 for a, b in picks_fam if a == f and b == n)
                                              for n in names}.items(), key=lambda kv: kv[1])[0]
                                      for f in FAMILIES},
                'modal_global_pick': max({n: picks_glob.count(n) for n in names}.items(),
                                         key=lambda kv: kv[1])[0],
                'null_resid_perm_global_delta_mean': r4(np.mean(nl_pf)),
                'null_resid_perm_global_delta_p95': r4(np.quantile(nl_pf, .95)),
                'real_minus_null': r4((pf.mean() - gl.mean()) - np.mean(nl_pf)),
                'null_resid_perm_family_delta_mean': r4(np.mean(nulls['family']['pf'])),
                'real_minus_null_family_scope':
                    r4((pf.mean() - gl.mean()) - np.mean(nulls['family']['pf'])),
                'which_null_is_primary': 'global. The family-scope null preserves each variant '
                                         'per-family mean exactly, which is the very quantity a '
                                         'per-family router exploits, so it grants the signal to '
                                         'the null and is reported only for completeness.'},
            'learned_pair_router': {
                'what': f'{len(fcols)} label-free features -> per-variant OOF LightGBM regression, '
                        f'route to the argmax. 5-way pool-hash OOF under this salt.',
                'n_features': len(fcols),
                'routed_map5': r4(routed),
                'delta_vs_best_single': r4(routed - blk['best_single']),
                'null_resid_perm_global_delta_mean': r4(np.mean(nl_ro)),
                'null_resid_perm_global_delta_p95': r4(np.quantile(nl_ro, .95)),
                'real_minus_null': r4((routed - blk['best_single']) - np.mean(nl_ro)),
                'null_resid_perm_family_delta_mean': r4(np.mean(nulls['family']['ro'])),
                'real_minus_null_family_scope':
                    r4((routed - blk['best_single']) - np.mean(nulls['family']['ro'])),
                'pick_distribution': {n: int((pick == i).sum()) for i, n in enumerate(names)}},
            'pairwise_router_top2': {
                'what': f'the cheapest learned form: {names[ia]} vs {names[second]}, OOF sign classifier',
                'routed_map5': r4(pw), 'delta_vs_best_single': r4(pw - blk['best_single']),
                'null_resid_perm_global_delta_mean': r4(np.mean(nl_pw)),
                'real_minus_null': r4((pw - blk['best_single']) - np.mean(nl_pw)),
                'null_resid_perm_family_delta_mean': r4(np.mean(nulls['family']['pw'])),
                'real_minus_null_family_scope':
                    r4((pw - blk['best_single']) - np.mean(nulls['family']['pw']))},
            'incumbent_hybrid_as_a_fixed_family_router': {
                'what': 'r382 for DT/SP, ref for CI: the family routing that already ships',
                'map5': r4(hyb.mean()), 'plain_r382_map5': r4(A[:, jr].mean()),
                'delta_vs_plain_r382': r4(hyb.mean() - A[:, jr].mean()),
                'delta_vs_best_single': r4(hyb.mean() - blk['best_single'])},
            'known_worse_arm_check': {
                'arm': 'route every pair to s1_pure (no stage 2 at all)',
                'delta_vs_best_single': r4(A[:, names.index('s1_pure')].mean() - blk['best_single']),
                'fails_the_gate': bool(A[:, names.index('s1_pure')].mean() < blk['best_single'])}}
        q = res['phase4_deployable'][salt]
        print(f"    {salt:<18} per-family LOTO {q['per_family_loto']['routed_map5']:.4f} vs global "
              f"{q['per_family_loto']['global_loto_map5']:.4f} "
              f"(delta {q['per_family_loto']['delta_vs_global_loto']:+.4f}, "
              f"real-null {q['per_family_loto']['real_minus_null']:+.4f})   learned router "
              f"{q['learned_pair_router']['delta_vs_best_single']:+.4f} "
              f"(real-null {q['learned_pair_router']['real_minus_null']:+.4f})")

    # ---- Phase 5: the CI retrieval upper bound ------------------------------------------------
    print('\n  PHASE 5.  CI retrieval upper bound (19.17 Q5 style)')
    g = (d.filter(pl.col('relevance') == 1).group_by('pair_id')
         .agg(pl.len().alias('n_listed'), (pl.col('oof_rank') <= TOP).sum().alias('in20')))
    gmap = {r[0]: (int(r[1]), int(r[2])) for r in g.iter_rows()}
    n_listed = np.array([gmap[p][0] for p in pairs], dtype=float)
    in20 = np.array([gmap[p][1] for p in pairs], dtype=float)
    miss = n_listed - in20
    den = np.minimum(n_listed, 5.0)
    orc_ov = np.minimum(in20, 5.0)
    sel_orc = orc_ov / den
    obs_miss = {f: r4(miss[fams == f].mean()) for f in FAMILIES}
    obs_orc = {f: r4(orc_ov[fams == f].mean()) for f in FAMILIES}
    for f in FAMILIES:
        assert abs(obs_miss[f] - STORED_CI_MISS[f]) < 1e-4, f'19.17 retrieval miss control failed {f}'
        assert abs(obs_orc[f] - STORED_CI_ORACLE[f]) < 1e-4, f'19.17 top-20 oracle control failed {f}'
    ci_idx = np.where(fams == CI)[0]
    dtsp_rate = float(miss[fams != CI].mean())
    target_total = dtsp_rate * len(ci_idx)
    rng = np.random.default_rng(PERM_SEED + 3)
    donors = []                       # (pair index, one missed hand) expanded
    for i in ci_idx:
        donors += [i] * int(round(miss[i]))
    donors = np.array(donors)
    keep_n = int(round(target_total))
    all_missing = np.bincount(donors, minlength=P).astype(float)
    draws_ov, draws_sel = [], []
    for _ in range(2000):
        keep = (rng.choice(len(donors), size=keep_n, replace=False) if keep_n < len(donors)
                else np.arange(len(donors)))
        still_missing = np.bincount(donors[keep], minlength=P).astype(float)
        new_in20 = in20 + (all_missing - still_missing)     # rescued = total misses - those left
        o = np.minimum(new_in20, 5.0)
        draws_ov.append(float(o[ci_idx].mean()))
        draws_sel.append(float((o[ci_idx] / den[ci_idx]).mean()))
    perfect_ov = float(np.minimum(n_listed[ci_idx], 5.0).mean())
    perfect_sel = float((np.minimum(n_listed[ci_idx], 5.0) / den[ci_idx]).mean())
    now_ov = float(orc_ov[ci_idx].mean())
    now_sel = float(sel_orc[ci_idx].mean())
    pooled_now = float(sel_orc.mean())
    share = len(ci_idx) / P
    actual = float(A1[:, names.index('r382')].mean())
    hyb1 = float(np.where(fams == CI, A1[:, names.index('ref')], A1[:, names.index('r382')]).mean())
    conv = hyb1 / pooled_now
    rows = {}
    for tag, (nov, nsel) in {'ci_miss_rate_matched_to_DTSP': (float(np.mean(draws_ov)), float(np.mean(draws_sel))),
                             'ci_retrieval_perfect': (perfect_ov, perfect_sel)}.items():
        d_ci = nsel - now_sel
        d_pool = share * d_ci
        rows[tag] = {
            'ci_top20_oracle_overlap_now': r4(now_ov), 'ci_top20_oracle_overlap_after': r4(nov),
            'ci_select_oracle_now': r4(now_sel), 'ci_select_oracle_after': r4(nsel),
            'ci_select_oracle_delta': r4(d_ci),
            'pooled_select_oracle_now': r4(pooled_now), 'pooled_select_oracle_delta': r4(d_pool),
            'UPPER_BOUND_composite_at_0.2_weight': round(LEVER * d_pool, 5),
            'proportional_estimate_map5_delta': r4(conv * d_pool),
            'proportional_estimate_composite': round(LEVER * conv * d_pool, 5),
            'clears_public_resolution_0.007_as_upper_bound': bool(LEVER * d_pool >= RESOLUTION),
            'clears_public_resolution_0.007_proportionally': bool(LEVER * conv * d_pool >= RESOLUTION)}
    res['phase5_ci_retrieval'] = {
        'controls_reproduce_19_17': {'retrieval_miss_per_pair': obs_miss, 'top20_oracle_overlap': obs_orc,
                                     'pooled_select_oracle': r4(pooled_now)},
        'ci_pairs': len(ci_idx), 'ci_share_of_surface': r4(share),
        'ci_retrieval_misses_total': int(miss[ci_idx].sum()),
        'all_retrieval_misses_total': int(miss.sum()),
        'dtsp_miss_rate_per_pair': r4(dtsp_rate),
        'misses_that_would_have_to_be_rescued': int(round(miss[ci_idx].sum() - target_total)),
        'conversion_actual_over_select_oracle': r4(conv),
        'conversion_note': f'the incumbent hybrid scores {hyb1:.4f} against a pooled select-oracle of '
                           f'{pooled_now:.4f}; {conv:.3f} of the top-20 content is converted into '
                           f'final MAP@5. The proportional estimate assumes newly retrieved CI hands '
                           f'convert at that same rate, which is itself optimistic for CI.',
        'rows': rows,
        'THIS_IS_AN_UPPER_BOUND': 'twice over. (1) it grants CI a stage-1 retrieval rate nobody has '
                                  'shown is reachable; (2) it is a change to STAGE 1 under a stage 2 '
                                  'fitted around the incumbent rank distribution, so it pays the '
                                  'coupling tax of 19.16 in full: -11 of every -16 lost correct picks '
                                  'in step 85b came from re-ranking among candidates that never left '
                                  'the top 20. Neither figure below is a forecast of a realised gain.',
        'r382_alone_reference_map5': r4(actual)}
    for tag, v in rows.items():
        print(f"    {tag:<32} CI oracle overlap {v['ci_top20_oracle_overlap_now']:.3f} -> "
              f"{v['ci_top20_oracle_overlap_after']:.3f}   pooled select-oracle "
              f"{v['pooled_select_oracle_delta']:+.4f}   UPPER BOUND composite "
              f"{v['UPPER_BOUND_composite_at_0.2_weight']:+.5f}   proportional "
              f"{v['proportional_estimate_composite']:+.5f}")

    # ---- minima and verdict --------------------------------------------------------------------
    mins = {
        'oracle_pair_gain_by_salt': [res['phase2_complementarity'][s]['oracle_pair_gain'] for s in SALTS],
        'oracle_family_gain_by_salt': [res['phase2_complementarity'][s]['oracle_family_gain'] for s in SALTS],
        'oracle_pair_real_minus_residperm_by_salt':
            [res['phase3_nulls'][s]['null_resid_perm_global']['real_minus_null_oracle_pair']
             for s in SALTS],
        'oracle_family_real_minus_residperm_by_salt':
            [res['phase3_nulls'][s]['null_resid_perm_global']['real_minus_null_oracle_family']
             for s in SALTS],
        'oracle_pair_real_minus_residperm_familyscope_by_salt':
            [res['phase3_nulls'][s]['null_resid_perm_family']['real_minus_null_oracle_pair']
             for s in SALTS],
        'oracle_pair_real_minus_fullperm_by_salt':
            [res['phase3_nulls'][s]['null_full_perm_global']['real_minus_null_oracle_pair']
             for s in SALTS],
        'oracle_pair_real_minus_seedfamily_by_salt':
            [res['phase3_nulls']['null_seed_family']['by_salt'][s]['real_minus_seedfamily_oracle_pair']
             for s in SALTS],
        'per_family_loto_delta_by_salt':
            [res['phase4_deployable'][s]['per_family_loto']['delta_vs_global_loto'] for s in SALTS],
        'per_family_loto_delta_vs_best_single_by_salt':
            [res['phase4_deployable'][s]['per_family_loto']['delta_vs_best_single_in_sample']
             for s in SALTS],
        'per_family_loto_real_minus_null_by_salt':
            [res['phase4_deployable'][s]['per_family_loto']['real_minus_null'] for s in SALTS],
        'learned_router_delta_by_salt':
            [res['phase4_deployable'][s]['learned_pair_router']['delta_vs_best_single'] for s in SALTS],
        'learned_router_real_minus_null_by_salt':
            [res['phase4_deployable'][s]['learned_pair_router']['real_minus_null'] for s in SALTS],
        'pairwise_router_delta_by_salt':
            [res['phase4_deployable'][s]['pairwise_router_top2']['delta_vs_best_single'] for s in SALTS],
        'incumbent_hybrid_minus_plain_r382_by_salt':
            [res['phase4_deployable'][s]['incumbent_hybrid_as_a_fixed_family_router']['delta_vs_plain_r382']
             for s in SALTS]}
    for k in list(mins):
        mins['MIN_' + k.replace('_by_salt', '')] = min(mins[k])
    res['resalt_minima'] = mins

    exists = mins['MIN_oracle_pair_real_minus_residperm'] > 0
    exists_fam = mins['MIN_oracle_family_real_minus_residperm'] > 0
    best_router = max(mins['MIN_per_family_loto_delta'], mins['MIN_learned_router_delta'],
                      mins['MIN_pairwise_router_delta'])
    best_router_vs_null = max(mins['MIN_per_family_loto_real_minus_null'],
                              mins['MIN_learned_router_real_minus_null'])
    exploitable = bool(best_router > 0 and best_router_vs_null > 0)
    material = bool(best_router >= MIN_DELTA)
    res['verdict'] = {
        'complementarity_exists_per_pair': bool(exists),
        'complementarity_exists_per_family': bool(exists_fam),
        'exploitable': exploitable, 'material': material,
        'ALIVE': bool(exploitable and material),
        'deciding_number': best_router,
        'deciding_number_is': 'the MINIMUM across the three pool partitions of (best deployable '
                              'routed MAP@5 - best single variant MAP@5), taken over the per-family '
                              'LOTO router, the learned per-pair router and the pairwise top-2 router',
        'oracle_ceiling_min': mins['MIN_oracle_pair_gain'],
        'oracle_ceiling_minus_matched_null_min': mins['MIN_oracle_pair_real_minus_residperm'],
        'per_family_routing_min_delta': mins['MIN_per_family_loto_delta'],
        'per_family_routing_min_real_minus_null': mins['MIN_per_family_loto_real_minus_null'],
        'materiality_bar': MIN_DELTA,
        'composite_at_the_deciding_number_discounted': round(LEVER * best_router * TRANSFER, 5)}
    res['limits'] = [
        'The seven variants share ONE frozen stage-1 score by design, so this measures complementarity '
        'among STAGE-2 designs only. Complementarity between different STAGE-1 retrievals is not '
        'measured here and could not be measured tax-free without a joint refit (step 88).',
        'The routing features use the TRUE behaviour family of each audit pair. In deployment the '
        'router would see the PREDICTED family, so every per-family number here is optimistic by the '
        'family classifier error rate; 14.41 built the shipping hybrid on predicted family and lost '
        'part of the audit gain to misrouted true-CI pairs for exactly this reason.',
        'OBS 18.9 / the 18.10 lesson: the audit isolation column has failed to replicate on held '
        'twice (14.41 r382-for-CI audit +.006 / held -.0068; ff50 audit +.0196 / held -.0170). Any CI '
        'routing signal measured on this surface should be treated as non-predictive until something '
        'changes. That warning applies to the CI arm of every routing number in this file.',
        '162 pairs, 43 of them CI. A per-family router chooses among 7 variants on 43 CI pairs; the '
        'matched nulls price that search but they cannot create power that is not there.',
        'The per-pair metric is AP@5, which is bounded and lumpy (six attainable overlap levels). '
        'The residual-permutation null therefore produces surrogate values slightly outside the AP@5 '
        'support; the statistic compared is the oracle GAIN, whose per-variant means are preserved '
        'exactly by construction, not the absolute level.',
        'Phase 5 is an upper bound and is not a forecast: no CI-specific retrieval model was built, '
        'and any such model pays the coupling tax in full.']

    res['protected_hashes_exit'] = check_protected()
    res['protected_hashes_identical'] = res['protected_hashes_exit'] == entry
    res['seconds'] = round(time.time() - t0)
    write_json(OUTDIR / 'audit-C-complementarity.json', res)

    print('\n' + '=' * 100)
    print(f"  ORACLE CEILING (per pair)      min across salts {mins['MIN_oracle_pair_gain']:+.4f}"
          f"   minus matched null {mins['MIN_oracle_pair_real_minus_residperm']:+.4f}")
    print(f"  ORACLE CEILING (per family)    min across salts {mins['MIN_oracle_family_gain']:+.4f}"
          f"   minus matched null {mins['MIN_oracle_family_real_minus_residperm']:+.4f}")
    print(f"  PER-FAMILY LOTO ROUTER         min across salts {mins['MIN_per_family_loto_delta']:+.4f}"
          f"   minus matched null {mins['MIN_per_family_loto_real_minus_null']:+.4f}")
    print(f"  LEARNED PER-PAIR ROUTER        min across salts {mins['MIN_learned_router_delta']:+.4f}"
          f"   minus matched null {mins['MIN_learned_router_real_minus_null']:+.4f}")
    print(f"  INCUMBENT HYBRID - PLAIN r382  min across salts "
          f"{mins['MIN_incumbent_hybrid_minus_plain_r382']:+.4f}")
    print(f"  VERDICT: {'ALIVE' if res['verdict']['ALIVE'] else 'DEAD'}   deciding number "
          f"{res['verdict']['deciding_number']:+.4f}  (bar {MIN_DELTA})")
    print(f"  wrote {OUTDIR / 'audit-C-complementarity.json'}  ({res['seconds']} s)")


if __name__ == '__main__':
    main()
