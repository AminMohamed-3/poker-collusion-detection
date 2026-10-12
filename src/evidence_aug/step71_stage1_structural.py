"""Step 71 (plan E6): put the structural columns inside STAGE 1, not only stage 2.

Why this is worth the CPU after three evidence nulls.  The public reads of P0 split the composite
exactly: the r382 ranker-feature increment transferred at 88 %, the structural stage 2 at -1 %.  The
one thing that demonstrably moves the evaluation set is the stage-1 ranker.  Stage 2 can only
reorder the top 20 it is given, so whatever stage 1 fails to retrieve is unreachable: audit
recall@20 is .972 overall but lowest for coordinated isolation, and MAP@5 is .594.  E6 asks whether
the ~70 structural columns -- which clearly carry signal, since stage 2 learned from them -- do more
good in front of the ranker than behind it.

Arms (all: 5-way pool-hash OOF over the 162 training-only audit pairs, `rank_evidence.PARAMS`
LambdaRank, exactly the step-16 loop):
  base382            the 382 conditional features            <- CONTROL, must reproduce audit-oof.parquet
  chrono             base382 + chrono_frac                   <- the production stage-1 design
  chrono_struct      + the ~70 step-65 structural columns
  chrono_struct_f2   + the step-70 mechanism block
  chrono_f2          chrono + the mechanism block only
Metrics per arm: MAP@5 pooled and per family; recall@5 / @10 / @20 per family (what stage 2 can
still reach); then the `ref` stage 2 refitted on that arm's own OOF, so a stage-1 gain that merely
moves work from stage 2 to stage 1 is visible as a wash.

Pre-registered gate (written before any number below was computed): an arm proceeds to a held freeze
only if, against `chrono`, pooled stage-1 MAP@5 >= +.010 with the paired bootstrap lower bound > 0,
OR coordinated-isolation recall@20 >= +.03; AND the arm's stage-2 MAP@5 is not worse than
`chrono`'s by more than .002.

Training-only.  Development labels for validation and selection; no evaluation label exists and no
leaderboard outcome enters anything here.  Output:
experiments/label_process_20260911/audit-stage1-structural.json
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
from evidence_aug.step14_audit_truncation import CLF, OUT, SPLITS, TOP, split_of
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step68a_audit_stage2_ensemble import S3, run, score as s2_score

GATE = {'stage1_map_ge': .010, 'stage1_ci95_lower_gt': 0.0, 'ci_recall20_ge': .03,
        'stage2_not_worse_than': -.002,
        'note': 'written before any number in this file was computed'}
KS = (5, 10, 20)


def recalls(d, sc):
    """fraction of LISTED hands inside each pair's top-K under this score, pooled and per family"""
    rank = (pl.DataFrame({'pair_id': d['pair_id'], 'relevance': d['relevance'],
                          'fam': d['behavior_family'], 's': sc})
            .with_columns(pl.col('s').rank('ordinal', descending=True).over('pair_id').alias('r')))
    lis = rank.filter(pl.col('relevance') == 1)
    out = {}
    for k in KS:
        out[f'recall_at_{k}'] = {'pooled': round(float((lis['r'] <= k).mean()), 4)}
        for f in FAMILIES:
            g = lis.filter(pl.col('fam') == f)
            out[f'recall_at_{k}'][f] = round(float((g['r'] <= k).mean()), 4)
    return out


def main():
    t0 = time.time()
    check_protected()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    struct_f = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    f2_f = pl.read_parquet(OUT / 'audit-f2-features.parquet')
    struct = [c for c in struct_f.columns if c not in ('pair_id', 'hand_id')]
    f2cols = [c for c in f2_f.columns if c.startswith('f2_')]

    data, features, _, _ = load_fold(0)
    # `pot_bb` (and any other overlap) already lives in the 382 conditional features; the stage-1 arms
    # must not list it twice, while stage 2 keeps the full struct list it has always used.
    struct_new = [c for c in struct if c not in set(features)]
    keys = audit.select('pair_id', 'hand_id')
    d = (keys.join(data, on=['pair_id', 'hand_id'], how='left', validate='1:1')
         .join(audit.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance',
                            'chrono_rank', 'chrono_frac', 'n_hands',
                            pl.col('oof').alias('oof_saved')),
               on=['pair_id', 'hand_id'], validate='1:1')
         .join(struct_f, on=['pair_id', 'hand_id'], validate='1:1')
         .join(f2_f.select(['pair_id', 'hand_id'] + f2cols), on=['pair_id', 'hand_id'], validate='1:1')
         .sort('pair_id', 'hand_id'))
    d = d.with_columns(pl.col(f2cols).cast(pl.Float64).fill_nan(0.0).fill_null(0.0))
    d = d.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
    log('audit frame', d.shape, 'pairs', d['pair_id'].n_unique(), 'listed', int(d['relevance'].sum()))

    chrono = ['chrono_rank', 'chrono_frac', 'n_hands']
    arms = {'base382': list(features),
            'chrono': list(features) + ['chrono_frac'],
            'chrono_struct': list(features) + ['chrono_frac'] + struct_new,
            'chrono_struct_f2': list(features) + ['chrono_frac'] + struct_new + f2cols,
            'chrono_f2': list(features) + ['chrono_frac'] + f2cols}

    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    y = d['relevance'].to_numpy().astype(float)
    groups = d['table_id'].to_numpy()
    key = d.select('pair_id', 'hand_id', 'table_id')
    split_col = d['split'].to_numpy()

    res = {'training_only': True, 'gate': GATE, 'pairs': d['pair_id'].n_unique(),
           'hands': d.height, 'n_struct': len(struct), 'n_struct_new_in_stage1': len(struct_new),
           'struct_already_in_382': sorted(set(struct) - set(struct_new)), 'n_f2': len(f2cols),
           'arms': {}}
    oofs, s1maps, s2maps = {}, {}, {}
    for name, cols in arms.items():
        t1 = time.time()
        oof = np.full(d.height, np.nan)
        for s in range(SPLITS):
            idx = np.where(split_col == s)[0]
            mdl = fit_ranker(d.filter(pl.col('split') != s), cols)
            oof[idx] = rank_score(mdl, family_columns(d[idx], 'behavior_family'), cols)
        oofs[name] = oof
        summ, s1maps[name] = summarize(key.with_columns(pl.Series('score', oof)), truths, labels)
        arm = {'n_features': len(cols), 'map5': round(summ['map5'], 4),
               'by_family': {f: round(v, 4) for f, v in summ['by_family'].items()}}
        arm.update(recalls(d, oof))
        arm['seconds'] = round(time.time() - t1)
        res['arms'][name] = arm
        log('S1 ' + name, json.dumps({'map5': arm['map5'],
                                      'fam': {f[:2]: arm['by_family'][f] for f in FAMILIES},
                                      'r20': arm['recall_at_20'], 'sec': arm['seconds']}))

    # control: the base arm must reproduce the saved audit OOF exactly
    delta = float(np.max(np.abs(oofs['base382'] - d['oof_saved'].to_numpy())))
    res['control_max_abs_delta_vs_audit_oof'] = delta
    log('control |base382 - audit-oof.parquet| max', delta)

    # stage 2 (`ref` design) refitted on each arm's own OOF
    for name, oof in oofs.items():
        t1 = time.time()
        dd = d.with_columns(pl.Series('oof', oof)).with_columns(
            pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        cols = struct + chrono + ['oof', 'oof_rank']
        per_fam = {f: dict(cols=cols, params=CLF, seeds=S3, ranker=False) for f in FAMILIES}
        pred, _ = run(dd, y, groups, per_fam, TOP)
        s, m = s2_score(dd, pred, TOP, truths, labels, key)
        s2maps[name] = m
        res['arms'][name]['stage2_ref'] = {'map5': round(s['map5'], 4),
                                           'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}}
        log('S2 ' + name, json.dumps({'map5': round(s['map5'], 4),
                                      'fam': {f[:2]: round(v, 4) for f, v in s['by_family'].items()},
                                      'sec': round(time.time() - t1)}))

    # paired bootstraps against the production design
    REF = 'chrono'
    res['paired_vs_chrono'] = {}
    for name in arms:
        if name == REF:
            continue
        b1 = boot(s1maps[REF], s1maps[name], ptab, pools)
        b2 = boot(s2maps[REF], s2maps[name], ptab, pools)
        row = {'stage1': {'mean_delta': round(b1['mean_delta'], 4),
                          'ci95': [round(v, 4) for v in b1['ci95']]},
               'stage2': {'mean_delta': round(b2['mean_delta'], 4),
                          'ci95': [round(v, 4) for v in b2['ci95']]},
               'stage1_by_family': {}, 'recall20_delta': {}}
        for f in FAMILIES:
            fp = [p for p in s1maps[REF] if labels[p] == f]
            fb = boot({p: s1maps[REF][p] for p in fp}, {p: s1maps[name][p] for p in fp},
                      ptab, sorted({ptab[p] for p in fp}))
            row['stage1_by_family'][f] = {'mean_delta': round(fb['mean_delta'], 4),
                                          'ci95': [round(v, 4) for v in fb['ci95']]}
            row['recall20_delta'][f] = round(res['arms'][name]['recall_at_20'][f]
                                             - res['arms'][REF]['recall_at_20'][f], 4)
        res['paired_vs_chrono'][name] = row
        log(f'{name} - chrono', json.dumps({'S1': [row['stage1']['mean_delta'], row['stage1']['ci95']],
                                            'S2': [row['stage2']['mean_delta'], row['stage2']['ci95']],
                                            'dR20': row['recall20_delta']}))

    verdict = {}
    for name, row in res['paired_vs_chrono'].items():
        map_ok = (row['stage1']['mean_delta'] >= GATE['stage1_map_ge']
                  and row['stage1']['ci95'][0] > GATE['stage1_ci95_lower_gt'])
        rec_ok = row['recall20_delta']['coordinated_isolation'] >= GATE['ci_recall20_ge']
        s2_ok = row['stage2']['mean_delta'] >= GATE['stage2_not_worse_than']
        verdict[name] = {'stage1_map_gate': map_ok, 'ci_recall20_gate': rec_ok,
                         'stage2_not_worse': s2_ok, 'gate_met': bool((map_ok or rec_ok) and s2_ok)}
        log('GATE ' + name, json.dumps(verdict[name]))
    res['verdict'] = verdict
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-stage1-structural.json', res)
    check_protected()
    log('done', res['seconds'])
    print('DONE')


if __name__ == '__main__':
    main()
