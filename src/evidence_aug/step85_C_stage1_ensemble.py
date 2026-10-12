"""Step 85 [C] (author: claude (agent)).  Does an ENSEMBLED stage 1 beat the single-seed stage 1
that ships?

WHY THIS QUESTION
Stage 1 is an LGBMRanker (lambdarank) over the 382 conditional features + chrono_frac with
`rank_evidence.PARAMS`, which pins `random_state=7329, deterministic=True`.  Its per-hand score and
within-pair rank are the two dominant stage-2 features (recorded stage-2 gains for directed_transfer
`s1_rank` .644, `s1` .095 -- about 74 % of the model), and it also decides which 20 candidates stage 2
is even allowed to see.  Two facts make ensembling it worth a test:

  1. Stage 1 has real seed variance.  OBSERVATIONS 18.13 (step 71b) re-ran arms at 7329 / 101 / 202
     and the production `chrono` design moved .5574 / .5532 / .5452 -- a .0122 swing with nothing
     changed but the draw.  Production ships ONE draw.
     (This is specific to the stage-1 RANKER, whose `colsample_bytree=.8` makes `random_state` live.
     The stage-2 LightGBM params carry no subsample/colsample, so `random_state` is INERT there --
     OBSERVATIONS 19.4 / step 80b.  The two must not be confused; step 80b was aborted over exactly
     that confusion, and this file re-proves the stage-1 seed is live before it uses it.)
  2. Ranker-derived features are the only family that has ever transferred to the public board.
     OBSERVATIONS 18.3: the structural stage 2 transferred at -1 % (held +.0411 -> public -.0002),
     the r382 ranker-feature increment at 88 % (held +.0163 -> public +.0143).

WHAT IS MEASURED (all training-only, 162 audit pairs in 172 tables outside every held fold)
  A  reproduction controls
  B  stage-1 seed variance at 8 seeds on the production configuration
  C  the ensemble-size curve, 1 / 2 / 4 / 8 seeds, three combiners (score mean, per-pair z mean,
     within-pair rank mean)
  D  the downstream end-to-end number: rebuild `oof`/`oof_rank` from the ensemble, refit the
     INCUMBENT stage 2 unchanged, compare to the shipping hybrid's .6159
  E  ensemble-disagreement columns as new stage-2 features, with a within-pair permutation control
  F  controls: 1-seed identity, capacity (more trees / more leaves / bagging), re-salted partitions

THE 0.5943 REPRODUCTION TARGET IS NOT ATTEMPTED, AND THAT IS DELIBERATE
The brief asks for the production stage-1 5-way pool-hash OOF on the 372 targets, MAP@5 .5943.
Computing that MAP requires the listed-evidence truth of every one of the 372 development targets,
including the pairs inside held folds 0/1/2.  That is a held read, which is forbidden.  It is
substituted by three reproduction controls on the legal surface that pin exactly the same objects:
  A1  the 382-feature ranker refit at seed 7329 must reproduce `audit-oof.parquet` bit-for-bit
      (step 71 recorded max |delta| = 0; MAP@5 .4896)
  A2  the PRODUCTION stage-1 design (382 + chrono_frac) at seed 7329 must reproduce step 71/71b's
      .5574 with DT .5942 / SP .6264 / CI .4205 and recall@20 .9710
  A3  the incumbent stage 2 on the stored stage-1 must reproduce the shipping hybrid's .6159
If any of the three misses, the run stops and everything downstream is void.

PRE-REGISTERED DECISION (written to prediction-stage1-ensemble.json before any fit)
  PRIMARY = the downstream end-to-end audit MAP@5 delta of the 8-seed ensemble stage 1 against the
  matched single-seed (7329) stage 1, incumbent stage 2 unchanged, taken as the MINIMUM across the
  three pool partitions -- the minimum, not the best, because single-partition numbers on this
  surface swing +-.008-.017 and the bootstrap CI does not cover that (step 80b).
    real     := min-across-salt downstream delta > 0
                AND min-across-salt stage-1 delta > 0
                AND the capacity control (a single model with 2x/4x/8x trees, or more leaves, or
                    bagging) does not reach within .005 of the ensemble's stage-1 gain
    material := 0.2 * min_delta * 0.49 >= .007, i.e. min_delta >= .0714
                (composite lever 0.2; historical audit->held transfer 46-52 %, midpoint 49 %;
                 public composite resolution +-.007)
    ALIVE only if real AND material.  Otherwise DEAD, with the failing number named.

Held folds 0-2 are not touched; no leaderboard score and no evaluation label enters anything here.
Output: experiments/stage1_ensemble_C_20260912/audit-C-stage1-ensemble.json
"""
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
from evidence_aug.frame import FAMILIES, family_columns, score as rank_score
from evidence_aug.paths import CV, check_protected, write_json
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step5_evaluate import pool_bootstrap
from evidence_aug.step14_audit_truncation import OUT as LP, SPLITS, TOP
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, build_d0, evaluate, fit_predict,
                                              frame_at)
from rank_evidence import PARAMS

AUTHOR = 'claude (agent)'
OUTDIR = CV / 'experiments' / 'stage1_ensemble_C_20260912'
SEEDS = (7329, 101, 202, 303, 404, 505, 606, 707)   # 7329 first: seeds[:1] is production
SALTS = ('audit-trunc-v1', 'audit-resplit-b', 'audit-resplit-c')
COMBINERS = ('score', 'zscore', 'rank')
SIZES = (1, 2, 4, 8)
# disjoint subsets so a size-k point is not just a prefix of the size-2k point
SUBSETS = {1: [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
           2: [(0, 1), (2, 3), (4, 5), (6, 7)],
           4: [(0, 1, 2, 3), (4, 5, 6, 7)],
           8: [(0, 1, 2, 3, 4, 5, 6, 7)]}
CAPACITY = {'trees_2x': dict(n_estimators=476),
            'trees_4x': dict(n_estimators=952),
            'trees_8x': dict(n_estimators=1904),
            'leaves_31': dict(num_leaves=31),
            'bagged': dict(subsample=.8, subsample_freq=1, colsample_bytree=.5)}
LEVER, TRANSFER, RESOLUTION = .2, .49, .007
ANCHORS = {'base382_oof_map5': .4896, 'prod_map5': .5574,
           'prod_by_family': {'directed_transfer': .5942, 'soft_play': .6264,
                              'coordinated_isolation': .4205},
           'prod_recall20': .9710, 'incumbent_hybrid': .6159}
GATE = {'primary': 'downstream end-to-end audit MAP@5 delta, ens8 vs single-seed-7329, incumbent '
                   'stage 2 unchanged, MINIMUM across the three pool partitions',
        'real_requires': ['min-across-salt downstream delta > 0',
                          'min-across-salt stage-1 delta > 0',
                          'no capacity control within .005 of the ensemble stage-1 gain'],
        'material_requires': f'{LEVER} * min_delta * {TRANSFER} >= {RESOLUTION} '
                             f'(min_delta >= {round(RESOLUTION / (LEVER * TRANSFER), 4)})',
        'verdict': 'ALIVE only if real AND material',
        'why_minimum': 'single-partition numbers on this surface swing +-.008-.017 (step 80b)',
        'note': 'written before any number in this file was computed'}


# ---------------------------------------------------------------------------------------------
# within-pair helpers.  every frame here is sorted by (pair_id, hand_id), so pairs are contiguous.
# ---------------------------------------------------------------------------------------------
def pair_slices(pid):
    uniq, first, cnt = np.unique(pid, return_index=True, return_counts=True)
    order = np.argsort(first)
    return [(int(first[i]), int(first[i] + cnt[i])) for i in order]


def wrank(v, sl):
    """1-based descending ordinal rank inside each pair; ties by row order, as polars 'ordinal'."""
    r = np.empty(len(v), dtype=np.float64)
    for a, b in sl:
        x = v[a:b]
        idx = np.argsort(-x, kind='stable')
        rr = np.empty(len(x))
        rr[idx] = np.arange(1, len(x) + 1)
        r[a:b] = rr
    return r


def wz(v, sl):
    z = np.empty(len(v), dtype=np.float64)
    for a, b in sl:
        x = v[a:b]
        z[a:b] = (x - x.mean()) / (x.std() + 1e-12)
    return z


def combine(vs, mode, sl):
    if mode == 'score':
        return np.mean(vs, axis=0)
    if mode == 'zscore':
        return np.mean([wz(v, sl) for v in vs], axis=0)
    if mode == 'rank':
        return -np.mean([wrank(v, sl) for v in vs], axis=0)
    raise ValueError(mode)


def topk_sets(v, sl, hands, k=5):
    out = []
    for a, b in sl:
        x = v[a:b]
        idx = np.argsort(-x, kind='stable')[:k]
        out.append(frozenset(hands[a:b][idx].tolist()))
    return out


def mean_overlap(sa, sb):
    return float(np.mean([len(x & y) for x, y in zip(sa, sb)]))


def recalls(rel, fam, v, sl, ks=(5, 10, 20)):
    """fraction of LISTED hands inside each pair's top-K under this score, pooled and per family"""
    r = wrank(v, sl)
    lis = rel == 1
    out = {}
    for k in ks:
        hit = (r[lis] <= k)
        out[f'recall_at_{k}'] = {'pooled': round(float(hit.mean()), 4)}
        for f in FAMILIES:
            m = fam[lis] == f
            out[f'recall_at_{k}'][f] = round(float(hit[m].mean()), 4)
    return out


def resplit(d, salt):
    """the 5-way pool hash under a different salt; pools stay whole tables (step 14 / step 80b)."""
    tabs = d['table_id'].unique().to_list()
    m = {t: int(hashlib.sha256((salt + ':' + t).encode()).hexdigest(), 16) % SPLITS for t in tabs}
    return d.with_columns(pl.col('table_id').replace_strict(m, return_dtype=pl.Int64).alias('split'))


def stage1_oof(ds, cols, seed, extra=None):
    """5-way pool-hash out-of-fold stage-1 scores; the step-16/71 loop, one ranker seed."""
    params = dict(PARAMS, random_state=seed, **(extra or {}))
    split_col = ds['split'].to_numpy()
    oof = np.full(ds.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(split_col == s)[0]
        mdl = fit_ranker(ds.filter(pl.col('split') != s), cols, params)
        oof[idx] = rank_score(mdl, family_columns(ds[idx], 'behavior_family'), cols)
    assert np.isfinite(oof).all()
    return oof


def seed_liveness_control():
    """Re-prove, every run, that random_state IS live for the STAGE-1 ranker params (colsample .8),
    while it is inert for the stage-2 classifier.  Step 80b was aborted over this exact confusion."""
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(1200, 40)), (rng.random(1200) < .25).astype(int)
    g = np.full(60, 20)
    a, b = [lgb.LGBMRanker(**dict(PARAMS, random_state=s)).fit(X, y, group=g).predict(X)
            for s in (7329, 101)]
    live = not np.array_equal(a, b)
    assert live, ('random_state no longer changes the stage-1 ranker -- the premise of this file is '
                  'gone; re-read step 71b and step 80b before trusting anything below')
    return {'stage1_ranker_seed_live': True, 'max_abs_score_diff_synthetic': float(np.abs(a - b).max()),
            'why': 'PARAMS carries colsample_bytree=.8, so the per-tree feature subsample is seeded',
            'contrast': 'the stage-2 classifier params carry no subsample/colsample: seed inert '
                        '(step 80b re-proves that separately)'}


def main():
    t0 = time.time()
    check_protected()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    live = seed_liveness_control()
    write_json(OUTDIR / 'prediction-stage1-ensemble.json',
               {'author': AUTHOR, 'step': '85', 'written_before_any_fit': True,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                'gate': GATE, 'seeds': list(SEEDS), 'salts': list(SALTS),
                'combiners': list(COMBINERS), 'sizes': list(SIZES),
                'capacity_controls': {k: v for k, v in CAPACITY.items()},
                'anchors_to_reproduce': ANCHORS, 'seed_liveness': live,
                'not_attempted': {'target': 'production stage-1 5-way pool-hash OOF on the 372 '
                                            'targets, MAP@5 .5943',
                                  'reason': 'its MAP needs the listed-evidence truth of pairs inside '
                                            'held folds 0/1/2; that is a held read and is forbidden',
                                  'substituted_by': ['base382 reproduces audit-oof.parquet',
                                                     'production design reproduces .5574 / .9710',
                                                     'incumbent stage 2 reproduces .6159']}})

    print('=' * 100)
    print('STEP 85 [C].  IS AN ENSEMBLED STAGE 1 BETTER THAN THE SINGLE DRAW THAT SHIPS?')
    print('=' * 100)
    print(f"  stage-1 seed liveness re-proved: max |delta| {live['max_abs_score_diff_synthetic']:.4g} "
          f"on the synthetic probe\n")

    # ---- frames -------------------------------------------------------------------------------
    audit = pl.read_parquet(LP / 'audit-oof.parquet')
    data, features, _, _ = load_fold(0)
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
    hands = d['hand_id'].to_numpy()
    rel = d['relevance'].to_numpy()
    famv = d['behavior_family'].to_numpy()
    sl = pair_slices(pid)
    key = d.select('pair_id', 'hand_id', 'table_id')
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    stage2_r382 = struct + CHRONO3 + S1 + rcols
    stage2_ref = struct + CHRONO3 + S1

    designs = {'prod': list(features) + ['chrono_frac'], 'base382': list(features)}
    res = {'author': AUTHOR, 'step': '85', 'training_only': True, 'held_folds_touched': False,
           'gate': GATE, 'seed_liveness': live, 'seeds': list(SEEDS), 'salts': list(SALTS),
           'population': {'pairs': int(d['pair_id'].n_unique()), 'hands': d.height,
                          'listed_hands': int(rel.sum()), 'tables': int(d['table_id'].n_unique()),
                          'queries_by_family': {f: int(sum(1 for p in labels if labels[p] == f))
                                                for f in FAMILIES}},
           'designs': {k: len(v) for k, v in designs.items()},
           'n_struct': len(struct), 'n_ranker_stage2': len(rcols)}

    def s1_metrics(v):
        summ, per_pair = summarize(key.with_columns(pl.Series('score', v)), truths, labels)
        row = {'map5': round(summ['map5'], 4),
               'by_family': {f: round(x, 4) for f, x in summ['by_family'].items()}}
        row.update(recalls(rel, famv, v, sl))
        return row, per_pair

    def hybrid(oof_vec, extra=None, cols_r=None, cols_f=None, split=None):
        """the incumbent stage 2, unchanged, on top of a supplied stage-1 score vector.

        `split` re-partitions stage 2 to match the salt that produced `oof_vec`, so under a re-salt
        BOTH stages move together and a stage-2 test row's score never came from a stage-1 model
        that saw another test table."""
        dd = d2.with_columns(pl.Series('oof', oof_vec))
        if split is not None:
            dd = dd.with_columns(pl.Series('split', split))
        if extra is not None:
            dd = dd.with_columns([pl.Series(c, extra[c]) for c in extra])
        dd = dd.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id')
                             .cast(pl.UInt32).alias('oof_rank'))
        pr = fit_predict(dd, dd, cols_r or stage2_r382, TOP)
        pf = fit_predict(dd, dd, cols_f or stage2_ref, TOP)
        _, ppr = evaluate(dd, pr, TOP, truths, labels)
        _, ppf = evaluate(dd, pf, TOP, truths, labels)
        pp = {q: (ppf[q] if labels[q] == CI else ppr[q]) for q in ppr}
        return float(np.mean(list(pp.values()))), pp

    # ================= A. reproduction controls ================================================
    dv1 = resplit(d, SALTS[0])
    assert np.array_equal(dv1['split'].to_numpy(),
                          d.with_columns(pl.col('table_id').map_elements(
                              lambda t: int(hashlib.sha256(('audit-trunc-v1:' + t).encode())
                                            .hexdigest(), 16) % SPLITS,
                              return_dtype=pl.Int64).alias('split'))['split'].to_numpy())

    ctrl = {}
    v_base = stage1_oof(dv1, designs['base382'], 7329)
    ctrl['A1_base382_vs_audit_oof_max_abs'] = float(np.max(np.abs(v_base - d['oof_saved'].to_numpy())))
    m_base, _ = s1_metrics(v_base)
    ctrl['A1_base382_map5'] = m_base['map5']
    ctrl['A1_pass'] = bool(ctrl['A1_base382_vs_audit_oof_max_abs'] == 0.0
                           and abs(m_base['map5'] - ANCHORS['base382_oof_map5']) < 5e-5)
    log('A1 base382 reproduces audit-oof.parquet',
        json.dumps({'max_abs': ctrl['A1_base382_vs_audit_oof_max_abs'], 'map5': m_base['map5'],
                    'anchor': ANCHORS['base382_oof_map5'], 'pass': ctrl['A1_pass']}))

    v_prod = stage1_oof(dv1, designs['prod'], 7329)
    m_prod, pp_prod = s1_metrics(v_prod)
    ctrl['A2_prod_map5'] = m_prod['map5']
    ctrl['A2_prod_by_family'] = m_prod['by_family']
    ctrl['A2_prod_recall20'] = m_prod['recall_at_20']['pooled']
    ctrl['A2_pass'] = bool(abs(m_prod['map5'] - ANCHORS['prod_map5']) < 5e-5
                           and abs(m_prod['recall_at_20']['pooled'] - ANCHORS['prod_recall20']) < 5e-5
                           and all(abs(m_prod['by_family'][f] - ANCHORS['prod_by_family'][f]) < 5e-5
                                   for f in FAMILIES))
    log('A2 production stage-1 design at seed 7329',
        json.dumps({'map5': m_prod['map5'], 'fam': m_prod['by_family'],
                    'recall20': m_prod['recall_at_20']['pooled'], 'pass': ctrl['A2_pass']}))

    inc_map, inc_pp = hybrid(d['oof_saved'].to_numpy())
    ctrl['A3_incumbent_hybrid'] = round(inc_map, 4)
    ctrl['A3_pass'] = bool(abs(inc_map - ANCHORS['incumbent_hybrid']) < 5e-5)
    log('A3 incumbent stage 2 on the stored stage-1',
        json.dumps({'map5': round(inc_map, 4), 'anchor': ANCHORS['incumbent_hybrid'],
                    'pass': ctrl['A3_pass']}))
    ctrl['all_pass'] = bool(ctrl['A1_pass'] and ctrl['A2_pass'] and ctrl['A3_pass'])
    ctrl['not_attempted_372_target'] = {
        'value': .5943, 'reason': 'its MAP needs held-fold evidence truth; forbidden'}
    res['reproduction_control'] = ctrl
    if not ctrl['all_pass']:
        res['ABORTED'] = 'reproduction control failed; everything downstream is void'
        write_json(OUTDIR / 'audit-C-stage1-ensemble.json', res)
        check_protected()
        print('\n  *** REPRODUCTION CONTROL FAILED -- STOPPING.  Everything downstream is void. ***')
        return

    # ================= B/C. seed variance and the ensemble curve, per salt =====================
    oofs = {}          # (salt, design, seed) -> vector
    res['by_salt'] = {}
    for salt in SALTS:
        ds = resplit(d, salt)
        moved = float(np.mean(ds['split'].to_numpy() != dv1['split'].to_numpy()))
        sal = {'rows_moved_pool': round(moved, 3), 'per_seed': {}, 'curve': {}, 'downstream': {}}
        for seed in SEEDS:
            t1 = time.time()
            if salt == SALTS[0] and seed == 7329:
                v = v_prod
            else:
                v = stage1_oof(ds, designs['prod'], seed)
            oofs[(salt, 'prod', seed)] = v
            m, _ = s1_metrics(v)
            sal['per_seed'][str(seed)] = {'map5': m['map5'], 'by_family': m['by_family'],
                                          'recall_at_20': m['recall_at_20']['pooled'],
                                          'recall_at_5': m['recall_at_5']['pooled']}
            log(f'B {salt} seed {seed}', json.dumps({'map5': m['map5'],
                                                     'r20': m['recall_at_20']['pooled'],
                                                     'sec': round(time.time() - t1)}))
        vals = [sal['per_seed'][str(s)]['map5'] for s in SEEDS]
        t5 = [topk_sets(oofs[(salt, 'prod', s)], sl, hands, 5) for s in SEEDS]
        pw = [mean_overlap(t5[i], t5[j]) for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
        sal['seed_spread'] = {'min': min(vals), 'max': max(vals), 'spread': round(max(vals) - min(vals), 4),
                              'mean': round(float(np.mean(vals)), 4), 'sd': round(float(np.std(vals, ddof=1)), 4),
                              'production_seed_7329': sal['per_seed']['7329']['map5'],
                              'production_rank_among_seeds': int(1 + sum(v > sal['per_seed']['7329']['map5']
                                                                         for v in vals)),
                              'mean_pairwise_top5_overlap_of_5': round(float(np.mean(pw)), 4)}
        log(f'B {salt} spread', json.dumps(sal['seed_spread']))

        # ---- the curve -------------------------------------------------------------------------
        base_pp = {}
        _, base_pp = s1_metrics(oofs[(salt, 'prod', 7329)])
        for mode in COMBINERS:
            sal['curve'][mode] = {}
            for k in SIZES:
                rows = []
                for sub in SUBSETS[k]:
                    v = combine([oofs[(salt, 'prod', SEEDS[i])] for i in sub], mode, sl)
                    m, pp = s1_metrics(v)
                    b = boot(base_pp, pp, ptab, pools)
                    rows.append({'seeds': [SEEDS[i] for i in sub], 'map5': m['map5'],
                                 'recall_at_20': m['recall_at_20']['pooled'],
                                 'by_family': m['by_family'],
                                 'delta_vs_7329': round(b['mean_delta'], 4),
                                 'ci95': [round(x, 4) for x in b['ci95']],
                                 'top5_overlap_vs_7329': round(
                                     mean_overlap(topk_sets(v, sl, hands, 5), t5[0]), 4)})
                    if k == 8:
                        rows[-1]['mean_top5_overlap_vs_members'] = round(
                            float(np.mean([mean_overlap(topk_sets(v, sl, hands, 5), t) for t in t5])), 4)
                mv = [r['map5'] for r in rows]
                sal['curve'][mode][str(k)] = {
                    'n_subsets': len(rows), 'mean_map5': round(float(np.mean(mv)), 4),
                    'min_map5': min(mv), 'max_map5': max(mv),
                    'mean_delta_vs_7329': round(float(np.mean([r['delta_vs_7329'] for r in rows])), 4),
                    'mean_recall_at_20': round(float(np.mean([r['recall_at_20'] for r in rows])), 4),
                    'subsets': rows}
                log(f'C {salt} {mode} k={k}',
                    json.dumps({'mean': sal['curve'][mode][str(k)]['mean_map5'],
                                'min': min(mv), 'max': max(mv),
                                'd': sal['curve'][mode][str(k)]['mean_delta_vs_7329']}))
        # the 1-seed identity control: combining one model must BE that model
        for mode in COMBINERS:
            one = sal['curve'][mode]['1']['subsets'][0]
            assert one['seeds'] == [7329]
            sal.setdefault('identity_control', {})[mode] = {
                'map5': one['map5'], 'matches_single_seed': bool(one['map5'] ==
                                                                 sal['per_seed']['7329']['map5']),
                'delta': one['delta_vs_7329']}
        res['by_salt'][salt] = sal

    # save the raw stage-1 matrix before anything else can fail
    mat = d.select('pair_id', 'hand_id', 'table_id')
    for (salt, dz, seed), v in oofs.items():
        mat = mat.with_columns(pl.Series(f'{salt}|{dz}|{seed}', v))
    mat.write_parquet(OUTDIR / 'stage1-oof-matrix.parquet')

    # ================= D. downstream, end to end ================================================
    # base382 arm, v1 salt only: the ONLY configuration whose single-seed number is the recorded
    # .6159, so it is the apples-to-apples control for "the only change is the stage-1 source".
    for seed in SEEDS[1:]:
        oofs[(SALTS[0], 'base382', seed)] = stage1_oof(dv1, designs['base382'], seed)
        log('D base382 seed', seed)
    oofs[(SALTS[0], 'base382', 7329)] = v_base
    vb8 = combine([oofs[(SALTS[0], 'base382', s)] for s in SEEDS], 'score', sl)
    mb8, _ = s1_metrics(vb8)
    hb8, ppb8 = hybrid(vb8)
    bb = boot(inc_pp, ppb8, ptab, pools)
    res['downstream_base382_v1'] = {
        'why': 'the audit stage-2 stack was built on the base382 stage 1, so this is the arm whose '
               'single-seed value is exactly the recorded .6159',
        'single_seed_7329': {'stage1_map5': m_base['map5'], 'hybrid_map5': round(inc_map, 4)},
        'ens8_score': {'stage1_map5': mb8['map5'], 'hybrid_map5': round(hb8, 4),
                       'delta': round(bb['mean_delta'], 4),
                       'ci95': [round(x, 4) for x in bb['ci95']]}}
    log('D base382 ens8 downstream',
        json.dumps({'hybrid': round(hb8, 4), 'vs .6159': round(bb['mean_delta'], 4)}))

    # production design, all three salts
    ens8_cache = {}
    for salt in SALTS:
        sal = res['by_salt'][salt]
        spl = resplit(d, salt)['split'].to_numpy()
        single = oofs[(salt, 'prod', 7329)]
        h1, pp1 = hybrid(single, split=spl)
        sal['downstream']['single_7329'] = round(h1, 4)
        for mode in ('score', 'rank'):
            v = combine([oofs[(salt, 'prod', s)] for s in SEEDS], mode, sl)
            h, pp = hybrid(v, split=spl)
            if mode == 'score':
                ens8_cache[salt] = (v, h, pp, spl)
            b = boot(pp1, pp, ptab, pools)
            sal['downstream'][f'ens8_{mode}'] = {
                'hybrid_map5': round(h, 4), 'delta_vs_single': round(b['mean_delta'], 4),
                'ci95': [round(x, 4) for x in b['ci95']],
                'by_family': {f: round(float(np.mean([pp[p] for p in pp if labels[p] == f])), 4)
                              for f in FAMILIES}}
            log(f'D {salt} ens8_{mode} downstream',
                json.dumps({'hybrid': round(h, 4),
                            'delta': sal['downstream'][f'ens8_{mode}']['delta_vs_single']}))
        sal['downstream']['stage1_delta_ens8_score_vs_single'] = \
            sal['curve']['score']['8']['subsets'][0]['delta_vs_7329']

    # ================= E. disagreement features as stage-2 inputs ===============================
    res['disagreement'] = {}
    rng = np.random.default_rng(8501)
    for salt in SALTS:
        vs = [oofs[(salt, 'prod', s)] for s in SEEDS]
        rk = np.vstack([wrank(v, sl) for v in vs])
        zs = np.vstack([wz(v, sl) for v in vs])
        ent = np.zeros(rk.shape[1])
        for j in range(rk.shape[1]):
            _, c = np.unique(rk[:, j], return_counts=True)
            q = c / c.sum()
            ent[j] = float(-(q * np.log(q)).sum() / np.log(len(SEEDS)))
        dis = {'ens_s1_sd_raw': np.vstack(vs).std(axis=0),
               'ens_s1_sd_z': zs.std(axis=0),
               'ens_rank_mean': rk.mean(axis=0), 'ens_rank_sd': rk.std(axis=0),
               'ens_rank_min': rk.min(axis=0), 'ens_rank_max': rk.max(axis=0),
               'ens_rank_spread': rk.max(axis=0) - rk.min(axis=0),
               'ens_top5_frac': (rk <= 5).mean(axis=0), 'ens_top20_frac': (rk <= 20).mean(axis=0),
               'ens_rank_entropy': ent}
        dcols = list(dis)
        # within-pair permutation control: identical marginals, hand-level mapping destroyed
        order = np.arange(len(pid))
        for a, b in sl:
            order[a:b] = rng.permutation(np.arange(a, b))
        shuf = {c: dis[c][order] for c in dcols}

        v8, h8, pp8, spl = ens8_cache[salt]
        row = {'ens8_score_only': round(h8, 4)}
        for tag, block in (('with_disagreement', dis), ('with_disagreement_shuffled', shuf)):
            h, pp = hybrid(v8, extra=block, cols_r=stage2_r382 + dcols, cols_f=stage2_ref + dcols,
                           split=spl)
            b = boot(pp8, pp, ptab, pools)
            row[tag] = {'hybrid_map5': round(h, 4), 'delta_vs_ens8': round(b['mean_delta'], 4),
                        'ci95': [round(x, 4) for x in b['ci95']],
                        'by_family': {f: round(float(np.mean([pp[p] for p in pp if labels[p] == f])), 4)
                                      for f in FAMILIES}}
            log(f'E {salt} {tag}', json.dumps({'hybrid': round(h, 4),
                                               'delta': row[tag]['delta_vs_ens8']}))
        if salt == SALTS[0]:
            # is stage 2 even using them?  normalised gain of the disagreement block, per family
            dd = d2.with_columns(pl.Series('oof', v8), *[pl.Series(c, dis[c]) for c in dcols])
            dd = dd.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id')
                                 .cast(pl.UInt32).alias('oof_rank'))
            top = dd.filter(pl.col('oof_rank') <= TOP)
            row['stage2_gain_share'] = {}
            for f in FAMILIES:
                sub = top.filter(pl.col('behavior_family') == f)
                cols = (stage2_ref if f == CI else stage2_r382) + dcols
                X = sub.select(cols).to_numpy().astype(np.float64)
                clf = lgb.LGBMClassifier(n_estimators=150, num_leaves=7, max_depth=3,
                                         min_child_samples=40, learning_rate=.05, n_jobs=2,
                                         verbosity=-1, random_state=11)
                clf.fit(X, sub['relevance'].to_numpy())
                g = clf.booster_.feature_importance('gain')
                g = g / max(g.sum(), 1e-12)
                idx = {c: i for i, c in enumerate(cols)}
                row['stage2_gain_share'][f] = {
                    'disagreement_block_total': round(float(sum(g[idx[c]] for c in dcols)), 4),
                    'top_disagreement': sorted(((c, round(float(g[idx[c]]), 4)) for c in dcols),
                                               key=lambda kv: -kv[1])[:4],
                    's1_rank': round(float(g[idx['oof_rank']]), 4),
                    's1': round(float(g[idx['oof']]), 4)}
            log('E gain share', json.dumps({f: row['stage2_gain_share'][f]['disagreement_block_total']
                                            for f in FAMILIES}))
        res['disagreement'][salt] = row

    # ================= F. capacity control ======================================================
    ens8_v1 = combine([oofs[(SALTS[0], 'prod', s)] for s in SEEDS], 'score', sl)
    m_ens8, pp_ens8 = s1_metrics(ens8_v1)
    gain = m_ens8['map5'] - m_prod['map5']
    res['capacity_control'] = {'why': 'an 8-seed ensemble is 8x the compute; if one bigger model buys '
                                      'the same lift we are buying capacity, not diversity',
                               'ensemble_stage1_gain_v1': round(gain, 4), 'arms': {}}
    for name, extra in CAPACITY.items():
        t1 = time.time()
        v = stage1_oof(dv1, designs['prod'], 7329, extra=extra)
        m, pp = s1_metrics(v)
        b = boot(pp_prod, pp, ptab, pools)
        res['capacity_control']['arms'][name] = {
            'params': extra, 'map5': m['map5'], 'recall_at_20': m['recall_at_20']['pooled'],
            'delta_vs_single_7329': round(b['mean_delta'], 4),
            'ci95': [round(x, 4) for x in b['ci95']],
            'reaches_ensemble_within_.005': bool(b['mean_delta'] >= gain - .005)}
        log(f'F capacity {name}', json.dumps({'map5': m['map5'],
                                              'delta': round(b['mean_delta'], 4),
                                              'sec': round(time.time() - t1)}))
    res['capacity_control']['any_arm_reaches_ensemble'] = bool(
        any(a['reaches_ensemble_within_.005'] for a in res['capacity_control']['arms'].values()))

    # ================= verdict ==================================================================
    s1d = [res['by_salt'][s]['curve']['score']['8']['subsets'][0]['delta_vs_7329'] for s in SALTS]
    dwn = [res['by_salt'][s]['downstream']['ens8_score']['delta_vs_single'] for s in SALTS]
    dwr = [res['by_salt'][s]['downstream']['ens8_rank']['delta_vs_single'] for s in SALTS]
    dis = [res['disagreement'][s]['with_disagreement']['delta_vs_ens8'] for s in SALTS]
    dsh = [res['disagreement'][s]['with_disagreement_shuffled']['delta_vs_ens8'] for s in SALTS]
    mind = min(dwn)
    real = bool(mind > 0 and min(s1d) > 0
                and not res['capacity_control']['any_arm_reaches_ensemble'])
    material = bool(LEVER * mind * TRANSFER >= RESOLUTION)
    res['resalt'] = {'stage1_ens8_score_deltas': s1d, 'stage1_min': min(s1d),
                     'downstream_ens8_score_deltas': dwn, 'downstream_min': mind,
                     'downstream_ens8_rank_deltas': dwr, 'downstream_rank_min': min(dwr),
                     'disagreement_deltas': dis, 'disagreement_min': min(dis),
                     'disagreement_shuffled_deltas': dsh, 'disagreement_shuffled_min': min(dsh)}
    res['composite'] = {
        'lever': LEVER, 'transfer_discount': TRANSFER, 'public_resolution': RESOLUTION,
        'min_delta_needed_for_resolution_undiscounted': round(RESOLUTION / LEVER, 4),
        'min_delta_needed_for_resolution_discounted': round(RESOLUTION / (LEVER * TRANSFER), 4),
        'downstream_min_delta': mind,
        'composite_undiscounted': round(LEVER * mind, 5),
        'composite_discounted': round(LEVER * mind * TRANSFER, 5),
        'clears_resolution_undiscounted': bool(LEVER * mind >= RESOLUTION),
        'clears_resolution_discounted': material}
    res['verdict'] = {'real': real, 'material': material,
                      'ALIVE': bool(real and material),
                      'deciding_number': mind,
                      'deciding_number_is': 'minimum across the three pool partitions of the '
                                            'end-to-end audit MAP@5 delta, 8-seed ensemble stage 1 '
                                            'vs single-seed-7329 stage 1, incumbent stage 2 unchanged',
                      'reason': None}
    if not real:
        why = []
        if mind <= 0:
            why.append(f'downstream delta is {mind:+.4f} under at least one partition')
        if min(s1d) <= 0:
            why.append(f'stage-1 delta is {min(s1d):+.4f} under at least one partition')
        if res['capacity_control']['any_arm_reaches_ensemble']:
            why.append('a single bigger model reaches the ensemble gain: capacity, not diversity')
        res['verdict']['reason'] = '; '.join(why)
    elif not material:
        res['verdict']['reason'] = (
            f'real but below public resolution: {LEVER} x {mind:.4f} x {TRANSFER} = '
            f'{LEVER * mind * TRANSFER:.5f} composite against +-{RESOLUTION}')

    # ---- console summary ----------------------------------------------------------------------
    print('\n  SEED VARIANCE, production stage-1 configuration, salt v1')
    v1 = res['by_salt'][SALTS[0]]
    print(f"    {'seed':>7}{'MAP@5':>9}{'recall@20':>11}{'DT':>8}{'SP':>8}{'CI':>8}")
    for s in SEEDS:
        r = v1['per_seed'][str(s)]
        print(f"    {s:>7}{r['map5']:>9.4f}{r['recall_at_20']:>11.4f}"
              + ''.join(f"{r['by_family'][f]:>8.4f}" for f in FAMILIES)
              + ('   <- ships' if s == 7329 else ''))
    print(f"    spread {v1['seed_spread']['spread']:.4f}  sd {v1['seed_spread']['sd']:.4f}  "
          f"mean pairwise top-5 overlap {v1['seed_spread']['mean_pairwise_top5_overlap_of_5']:.3f} of 5")

    print('\n  ENSEMBLE-SIZE CURVE, stage-1 MAP@5 (mean over disjoint subsets), salt v1')
    print(f"    {'k':>4}" + ''.join(f'{m:>22}' for m in COMBINERS))
    for k in SIZES:
        cells = []
        for m in COMBINERS:
            c = v1['curve'][m][str(k)]
            cells.append(f"{c['mean_map5']:.4f} [{c['min_map5']:.4f},{c['max_map5']:.4f}]")
        print(f"    {k:>4}" + ''.join(f'{c:>22}' for c in cells))

    print('\n  DOWNSTREAM (incumbent stage 2 unchanged), audit MAP@5')
    print(f"    base382 single 7329 (= the recorded .6159): {inc_map:.4f}")
    print(f"    base382 ens8 score                        : {hb8:.4f}  "
          f"({res['downstream_base382_v1']['ens8_score']['delta']:+.4f})")
    for salt in SALTS:
        s = res['by_salt'][salt]['downstream']
        print(f"    prod {salt:<18} single {s['single_7329']:.4f}   "
              f"ens8_score {s['ens8_score']['hybrid_map5']:.4f} "
              f"({s['ens8_score']['delta_vs_single']:+.4f})   "
              f"ens8_rank {s['ens8_rank']['hybrid_map5']:.4f} "
              f"({s['ens8_rank']['delta_vs_single']:+.4f})")

    print('\n  DISAGREEMENT FEATURES (vs the same ensemble stage 1 without them)')
    for salt in SALTS:
        r = res['disagreement'][salt]
        print(f"    {salt:<20} real {r['with_disagreement']['delta_vs_ens8']:+.4f}   "
              f"shuffled control {r['with_disagreement_shuffled']['delta_vs_ens8']:+.4f}")

    print('\n  CAPACITY CONTROL (single model, salt v1), stage-1 delta vs single seed 7329')
    print(f"    ensemble of 8 gives {gain:+.4f}")
    for name, a in res['capacity_control']['arms'].items():
        print(f"    {name:<12} {a['delta_vs_single_7329']:+.4f}  map5 {a['map5']:.4f}"
              + ('   <- reaches the ensemble' if a['reaches_ensemble_within_.005'] else ''))

    print(f"\n  RE-SALT MINIMUMS: stage1 {min(s1d):+.4f}   downstream {mind:+.4f}   "
          f"disagreement {min(dis):+.4f}")
    print(f"  composite 0.2 x {mind:.4f} = {LEVER * mind:+.5f}; discounted at {TRANSFER} = "
          f"{LEVER * mind * TRANSFER:+.5f} against +-{RESOLUTION}")
    print(f"\n  VERDICT: {'ALIVE' if res['verdict']['ALIVE'] else 'DEAD'}"
          + (f"  -- {res['verdict']['reason']}" if res['verdict']['reason'] else ''))

    res['limits'] = [
        'The 372-target .5943 anchor was not reproduced: its MAP needs held-fold evidence truth. '
        'Three substitute controls on the legal surface all passed exactly.',
        '162 audit pairs, 172 tables. Every delta here is on that surface; audit->held transfer has '
        'run 46-52 % historically and audit->public less than that for non-ranker families.',
        'The downstream test holds stage 2 fixed at the incumbent recipe; a stage 2 re-tuned for an '
        'ensembled stage 1 is not tested.',
        'The ensemble is over ranker seeds only. Bagging inside one model is included solely as a '
        'capacity control, not as a production candidate.',
        'Disagreement features are computed from the same 8 OOF vectors that build the ensemble '
        'score, so they cost nothing extra once the ensemble exists, and nothing at all if it does not.']
    res['seconds'] = round(time.time() - t0)
    write_json(OUTDIR / 'audit-C-stage1-ensemble.json', res)
    check_protected()
    print(f"\n  wrote {OUTDIR / 'audit-C-stage1-ensemble.json'}  ({res['seconds']} s)")
    print('DONE')


if __name__ == '__main__':
    main()
