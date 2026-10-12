"""Step 71b: is the E6 stage-1 gain real?  Seeds, splits, and a within-pair permutation control.

Step 71 found, on the 162 training-only audit pairs, that adding the ~70 structural columns AND the
21 mechanism columns to the stage-1 LambdaRank lifts stage-1 MAP@5 by +.0239 [+.0062, +.0414] over
the production design, while either block alone does nothing (struct +.0078 crossing zero, f2
-.0001).  A super-additive jump from two blocks that are individually null, measured once, at a
single ranker seed, on 162 pairs, is the exact shape of an artefact.  The audit has already
over-estimated isolation variants twice (steps 65h, 68a vs the held read of 65j), and this study has
only one held read of folds 0-2 to spend.  So test it three ways before spending it:

  seeds     re-run every arm at ranker random_state 7329 (production), 101, 202.  `PARAMS` fixes the
            seed and `deterministic=True`, so step 71 measured one draw.
  splits    report the paired delta inside each of the five pool-hash splits: a real effect should
            not live in one split.
  control   `perm` = chrono_struct_f2 with the struct+f2 rows permuted WITHIN each pair (identical
            marginal distributions and column count, hand-level signal destroyed).  If perm also
            gains, the gain is capacity or luck, not the features.

Decision (pre-registered, written before any number below was computed): the arm goes to a held
freeze only if the mean stage-1 delta over the three seeds is >= +.010 with the pooled bootstrap
lower bound > 0, it is positive in at least 4 of the 5 splits, it is positive at all three seeds,
and the permutation control's mean delta is below +.005.

Training-only.  Output: experiments/label_process_20260911/audit-stage1-stability.json
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

from evidence_aug.frame import FAMILIES, family_columns, score as rank_score
from evidence_aug.fold import load_fold
from evidence_aug.paths import check_protected, write_json
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step14_audit_truncation import OUT, SPLITS, split_of
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot
from rank_evidence import PARAMS

SEEDS = (7329, 101, 202)
GATE = {'mean_delta_ge': .010, 'ci95_lower_gt': 0.0, 'splits_positive_ge': 4,
        'positive_at_every_seed': True, 'perm_control_below': .005,
        'note': 'written before any number in this file was computed'}
RNG = np.random.default_rng(710)


def oof_of(d, cols, seed, split_col, key, truths, labels):
    params = dict(PARAMS, random_state=seed)
    oof = np.full(d.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(split_col == s)[0]
        mdl = fit_ranker(d.filter(pl.col('split') != s), cols, params)
        oof[idx] = rank_score(mdl, family_columns(d[idx], 'behavior_family'), cols)
    summ, per_pair = summarize(key.with_columns(pl.Series('score', oof)), truths, labels)
    return summ, per_pair


def main():
    t0 = time.time()
    check_protected()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    struct_f = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    f2_f = pl.read_parquet(OUT / 'audit-f2-features.parquet')
    struct = [c for c in struct_f.columns if c not in ('pair_id', 'hand_id')]
    f2cols = [c for c in f2_f.columns if c.startswith('f2_')]

    data, features, _, _ = load_fold(0)
    struct_new = [c for c in struct if c not in set(features)]
    d = (audit.select('pair_id', 'hand_id')
         .join(data, on=['pair_id', 'hand_id'], how='left', validate='1:1')
         .join(audit.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance',
                            'chrono_rank', 'chrono_frac', 'n_hands'),
               on=['pair_id', 'hand_id'], validate='1:1')
         .join(struct_f, on=['pair_id', 'hand_id'], validate='1:1')
         .join(f2_f.select(['pair_id', 'hand_id'] + f2cols), on=['pair_id', 'hand_id'], validate='1:1')
         .sort('pair_id', 'hand_id'))
    d = d.with_columns(pl.col(f2cols).cast(pl.Float64).fill_nan(0.0).fill_null(0.0))
    d = d.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))

    # the permutation control: shuffle the block's rows inside each pair, nothing else changes
    block = struct_new + f2cols
    order = np.arange(d.height)
    pid = d['pair_id'].to_numpy()
    for p in np.unique(pid):
        m = np.where(pid == p)[0]
        order[m] = RNG.permutation(m)
    perm_block = d.select(block).to_numpy()[order]
    dperm = d.with_columns([pl.Series(c, perm_block[:, i]) for i, c in enumerate(block)])

    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    key = d.select('pair_id', 'hand_id', 'table_id')
    split_col = d['split'].to_numpy()
    pair_split = dict(d.select('pair_id', 'split').unique().iter_rows())

    base = list(features) + ['chrono_frac']
    arms = {'chrono': (d, base),
            'chrono_struct': (d, base + struct_new),
            'chrono_struct_f2': (d, base + struct_new + f2cols),
            'perm_struct_f2': (dperm, base + struct_new + f2cols)}

    res = {'training_only': True, 'gate': GATE, 'seeds': list(SEEDS), 'arms': {}, 'paired': {}}
    maps = {}
    for name, (frame, cols) in arms.items():
        maps[name] = {}
        row = {'n_features': len(cols), 'map5_by_seed': {}}
        for seed in SEEDS:
            t1 = time.time()
            summ, per_pair = oof_of(frame, cols, seed, split_col, key, truths, labels)
            maps[name][seed] = per_pair
            row['map5_by_seed'][seed] = {'map5': round(summ['map5'], 4),
                                         'by_family': {f: round(v, 4) for f, v in summ['by_family'].items()}}
            log(f'{name} seed {seed}', json.dumps({'map5': round(summ['map5'], 4),
                                                   'fam': {f[:2]: round(v, 4) for f, v in summ['by_family'].items()},
                                                   'sec': round(time.time() - t1)}))
        # the seed-averaged pair score is what a 3-seed production ensemble would deliver
        maps[name]['avg'] = {p: float(np.mean([maps[name][s][p] for s in SEEDS])) for p in truths}
        row['map5_seed_avg'] = round(float(np.mean(list(maps[name]['avg'].values()))), 4)
        res['arms'][name] = row

    REF = 'chrono'
    for name in arms:
        if name == REF:
            continue
        per_seed, row = {}, {}
        for seed in SEEDS:
            b = boot(maps[REF][seed], maps[name][seed], ptab, pools)
            per_seed[seed] = {'mean_delta': round(b['mean_delta'], 4),
                              'ci95': [round(v, 4) for v in b['ci95']]}
        ba = boot(maps[REF]['avg'], maps[name]['avg'], ptab, pools)
        row['per_seed'] = per_seed
        row['seed_mean_delta'] = round(float(np.mean([v['mean_delta'] for v in per_seed.values()])), 4)
        row['seed_avg_model'] = {'mean_delta': round(ba['mean_delta'], 4),
                                 'ci95': [round(v, 4) for v in ba['ci95']]}
        row['by_split'] = {}
        for s in range(SPLITS):
            sp = [p for p in truths if pair_split[p] == s]
            row['by_split'][s] = {'n_pairs': len(sp),
                                  'delta': round(float(np.mean([maps[name]['avg'][p] - maps[REF]['avg'][p]
                                                                for p in sp])), 4)}
        row['by_family'] = {}
        for f in FAMILIES:
            fp = [p for p in truths if labels[p] == f]
            fb = boot({p: maps[REF]['avg'][p] for p in fp}, {p: maps[name]['avg'][p] for p in fp},
                      ptab, sorted({ptab[p] for p in fp}))
            row['by_family'][f] = {'mean_delta': round(fb['mean_delta'], 4),
                                   'ci95': [round(v, 4) for v in fb['ci95']], 'n_pairs': len(fp)}
        res['paired'][name] = row
        log(f'{name} - chrono', json.dumps({
            'seed_mean': row['seed_mean_delta'],
            'per_seed': {s: v['mean_delta'] for s, v in per_seed.items()},
            'avg_model': [row['seed_avg_model']['mean_delta'], row['seed_avg_model']['ci95']],
            'splits': {s: v['delta'] for s, v in row['by_split'].items()},
            'fam': {f[:2]: v['mean_delta'] for f, v in row['by_family'].items()}}))

    verdict = {}
    perm_delta = res['paired']['perm_struct_f2']['seed_mean_delta']
    for name, row in res['paired'].items():
        if name == 'perm_struct_f2':
            continue
        pos_splits = sum(1 for v in row['by_split'].values() if v['delta'] > 0)
        verdict[name] = {
            'seed_mean_delta': row['seed_mean_delta'],
            'mean_ge': row['seed_mean_delta'] >= GATE['mean_delta_ge'],
            'ci_lower_gt_0': row['seed_avg_model']['ci95'][0] > GATE['ci95_lower_gt'],
            'splits_positive': pos_splits,
            'splits_ok': pos_splits >= GATE['splits_positive_ge'],
            'positive_every_seed': all(v['mean_delta'] > 0 for v in row['per_seed'].values()),
            'perm_control_delta': perm_delta,
            'perm_ok': perm_delta < GATE['perm_control_below']}
        verdict[name]['gate_met'] = bool(verdict[name]['mean_ge'] and verdict[name]['ci_lower_gt_0']
                                         and verdict[name]['splits_ok']
                                         and verdict[name]['positive_every_seed']
                                         and verdict[name]['perm_ok'])
        log('GATE ' + name, json.dumps(verdict[name]))
    res['verdict'] = verdict
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-stage1-stability.json', res)
    check_protected()
    log('done', res['seconds'])
    print('DONE')


if __name__ == '__main__':
    main()
