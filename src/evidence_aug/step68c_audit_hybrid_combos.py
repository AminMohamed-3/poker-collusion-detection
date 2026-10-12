"""Step 68c (E1/E2 round 2): audit the stage-2 combinations against the PRODUCTION hybrid.

Round 1 (step 68a) measured every recipe against `ref` (the struct stage 2).  But `ref` is not what
ships: the uploaded candidates use the hybrid -- r382 for predicted transfer and soft play, `ref`
for predicted isolation.  Round 2 therefore re-references everything to that hybrid and adds the
combinations round 1 implied but did not build, including the audit twin of the step-69 probe
(isolation reverted to the stage-1 order).

Round 1 findings this is built on:
  r382_ff50   +.0322 [+.0124, +.0535] pooled, isolation +.0196, better in 5 of 5 isolation splits
  ens_rank    +.0280 [+.0155, +.0407] pooled, isolation +.0239, better in 4 of 5
  every hand-designed isolation recipe (ci_reg, ci_struct_only, ci_mono, ci_inter, ci_bag) LOSES
  top-K: 20 beats 15, 25 and 30 in every family
  LambdaRank stage 2 loses to the classifier (rank_ref -.0045; rank_r382 +.0191, CI lower -.0019)

Two bootstraps are reported: pooled over all 162 audit pairs, and restricted to the 92 isolation
pairs and the tables that carry them (round 1's pooled interval for a spliced isolation-only
variant is diluted by the 70 pairs whose lists do not change).

Output: experiments/label_process_20260911/audit-hybrid-combos.json
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import write_json
from evidence_aug.step14_audit_truncation import CLF, OUT, TOP
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step68a_audit_stage2_ensemble import (S3, S5, rank_within_pair, run, score)

CI = 'coordinated_isolation'
DTSP = [f for f in FAMILIES if f != CI]


def uni(cols, params=CLF, seeds=S3):
    return {f: dict(cols=cols, params=params, seeds=seeds, ranker=False) for f in FAMILIES}


def compose(parts):
    """parts: family -> prediction vector; take each family's rows from its own vector"""
    fam = compose.fam
    out = np.full(len(fam), -1e6)
    for f, v in parts.items():
        m = fam == f
        out[m] = v[m]
    return out


def main():
    t0 = time.time()
    from evidence_aug.fold import load_fold
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    sc, chrono = ['oof', 'oof_rank'], ['chrono_rank', 'chrono_frac', 'n_hands']
    data, features, _, _ = load_fold(0)
    rcols = [f for f in features if not f.startswith('family_') and f in data.columns
             and f not in struct + chrono + sc]
    d = d.join(data.select(['pair_id', 'hand_id'] + rcols).with_columns(pl.lit(1).alias('_hit')),
               on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert d['_hit'].null_count() == 0
    d = d.drop('_hit').with_columns(pl.col(rcols).fill_null(-999.0))

    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float)
    groups = d['table_id'].to_numpy()
    key = d.select('pair_id', 'hand_id', 'table_id')
    compose.fam = d['behavior_family'].to_numpy()

    ref, r382 = struct + chrono + sc, struct + chrono + sc + rcols
    designs = {
        'ref': uni(ref),
        'r382': uni(r382),
        'ff50': uni(r382, dict(CLF, colsample_bytree=.5)),
        'ff25': uni(r382, dict(CLF, colsample_bytree=.25)),
        'ff75': uni(r382, dict(CLF, colsample_bytree=.75)),
        'ff50_s5': uni(r382, dict(CLF, colsample_bytree=.5), S5),
    }
    base = {}
    for name, per_fam in designs.items():
        t1 = time.time()
        base[name], _ = run(d, y, groups, per_fam, TOP)
        s, _ = score(d, base[name], TOP, truths, labels, key)
        log('design ' + name, json.dumps({'map5': round(s['map5'], 4),
                                          'fam': {f[:2]: round(v, 4) for f, v in s['by_family'].items()},
                                          'sec': round(time.time() - t1)}))
    rk = {k: rank_within_pair(d, v, TOP) for k, v in base.items()}
    stage1 = d['oof'].to_numpy()                       # reverting a branch to the stage-1 order

    combos = {
        'hyb_prod (DT/SP r382, CI ref)': {**{f: base['r382'] for f in DTSP}, CI: base['ref']},
        'hyb_ci_stage1 (the step-69 probe)': {**{f: base['r382'] for f in DTSP}, CI: stage1},
        'hyb_ci_ff50': {**{f: base['r382'] for f in DTSP}, CI: base['ff50']},
        'hyb_ci_ensr': {**{f: base['r382'] for f in DTSP}, CI: .5 * (rk['ref'] + rk['r382'])},
        'hyb_ci_ensff': {**{f: base['r382'] for f in DTSP}, CI: .5 * (rk['ref'] + rk['ff50'])},
        'hyb_dtsp_ff50': {**{f: base['ff50'] for f in DTSP}, CI: base['ref']},
        'ff50_all': {f: base['ff50'] for f in FAMILIES},
        'ff25_all': {f: base['ff25'] for f in FAMILIES},
        'ff75_all': {f: base['ff75'] for f in FAMILIES},
        'ff50_s5_all': {f: base['ff50_s5'] for f in FAMILIES},
        'ensr_all': {f: .5 * (rk['ref'] + rk['r382']) for f in FAMILIES},
        'ensff_all': {f: .5 * (rk['ref'] + rk['ff50']) for f in FAMILIES},
        'ensff_dtsp_r382ci_ff50': {**{f: .5 * (rk['ref'][:] + rk['ff50'][:]) for f in DTSP},
                                   CI: base['ff50']},
    }
    REF = 'hyb_prod (DT/SP r382, CI ref)'
    maps, sets = {}, {}
    for name, parts in combos.items():
        s, m = score(d, compose(parts), TOP, truths, labels, key)
        maps[name], sets[name] = m, s

    ci_pairs = [p for p in maps[REF] if labels[p] == CI]
    ci_pools = sorted({ptab[p] for p in ci_pairs})
    res = {'training_only': True, 'reference': REF, 'queries': len(truths),
           'n_ci_pairs': len(ci_pairs), 'sets': {}, 'paired_vs_hyb_prod': {}}
    for name, s in sets.items():
        res['sets'][name] = s
        log(name, json.dumps({'map5': round(s['map5'], 4),
                              'fam': {f[:2]: round(v, 4) for f, v in s['by_family'].items()}}))
    for name in combos:
        if name == REF:
            continue
        b = boot(maps[REF], maps[name], ptab, pools)
        sub_a = {p: maps[REF][p] for p in ci_pairs}
        sub_b = {p: maps[name][p] for p in ci_pairs}
        bci = boot(sub_a, sub_b, ptab, ci_pools)
        b['by_family'] = {f: round(float(np.mean([maps[name][p] - maps[REF][p]
                                                  for p in maps[REF] if labels[p] == f])), 4)
                          for f in FAMILIES}
        b['ci_only_bootstrap'] = {'mean_delta': round(bci['mean_delta'], 4),
                                  'ci95': [round(v, 4) for v in bci['ci95']],
                                  'pairs_better': bci['pairs_better'], 'pairs_worse': bci['pairs_worse']}
        res['paired_vs_hyb_prod'][name] = b
        log(f'{name} - prod', json.dumps({'mean': round(b['mean_delta'], 4),
                                          'ci95': [round(v, 4) for v in b['ci95']],
                                          'fam': b['by_family'],
                                          'CIonly': b['ci_only_bootstrap']['mean_delta'],
                                          'CIonly_ci95': b['ci_only_bootstrap']['ci95']}))
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-hybrid-combos.json', res)
    log('done', res['seconds'])
    print('DONE')


if __name__ == '__main__':
    main()
