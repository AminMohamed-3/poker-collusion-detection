"""STEP248 Part B.3: POWER of the frozen, unsupervised instrument, measured on DEVELOPMENT
truncated to the first 2,000 hands per table.  Labels are read here and nowhere else.

Because the instrument is unsupervised, its recall on the three DISCLOSED families at a fixed
false-flag budget is an honest estimate of its recall on an unseen in-hand family of similar
dose.  The stopping rule registered before any measurement: if macro recall at 30 expected
false flags (in a 112,540-pair population) is below 30 %, the instrument is too weak and the
study is written up as a documented null.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_c_common import (EVAL_POP, ORIGINAL, OUT, attach_players, load_detail,
                              load_phase, null_thresholds, strata)
from step248_b_events import NAMES

TAG = 'development2000'
MIN_SHARED = 38
BUDGETS = [10, 30, 100]


def main():
    t0 = time.time()
    pairs, null, null_ns, null_td = load_phase(TAG)
    pairs = attach_players(pairs, TAG)
    print(f'{TAG}: {pairs.height} pairs, null rows {null.shape}  [{time.time()-t0:.0f}s]')

    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p_lo'),
        pl.max_horizontal('player_1', 'player_2').alias('p_hi'))
    pairs = pairs.join(lab.select('p_lo', 'p_hi', 'label', 'label_status', 'behavior_family'),
                       on=['p_lo', 'p_hi'], how='left')
    elig = pairs.filter(pl.col('n_shared') >= MIN_SHARED)
    nmask = null_ns >= MIN_SHARED
    ns_e = elig['n_shared'].to_numpy()
    edges = np.quantile(ns_e, np.linspace(.1, .9, 9))
    s1 = elig['S1'].to_numpy()
    s1p = elig['S1_plc'].to_numpy()
    st = strata(ns_e, edges)
    nulls1 = null[nmask, 0]
    nulls1p = null[nmask, 2]
    nns = null_ns[nmask]

    pos = elig.filter(pl.col('label') == 1)
    neg = elig.filter(pl.col('label') == 0)
    print(f'eligible {elig.height}; labelled positives {pos.height} '
          f'(DT {pos.filter(pl.col("behavior_family")=="directed_transfer").height}, '
          f'SP {pos.filter(pl.col("behavior_family")=="soft_play").height}, '
          f'CI {pos.filter(pl.col("behavior_family")=="coordinated_isolation").height}); '
          f'confirmed non-targets {neg.height}')

    res = {'tag': TAG, 'eligible': elig.height, 'min_shared': MIN_SHARED,
           'strata_edges': [float(x) for x in edges], 'budgets': {}}
    islab = elig['label'].to_numpy()
    fam = elig['behavior_family'].fill_null('unlabelled').to_numpy()
    for bud in BUDGETS:
        thr = null_thresholds(nulls1, nns, edges, bud)
        thr_p = null_thresholds(nulls1p, nns, edges, bud)
        flag = s1 >= thr[st]
        flag_p = s1p >= thr_p[st]
        row = {'thresholds_S1': [float(x) for x in thr],
               'flagged_all': int(flag.sum()),
               'flagged_rate_per_112540': float(flag.mean() * EVAL_POP),
               'placebo_flagged_rate_per_112540': float(flag_p.mean() * EVAL_POP),
               'recall': {}, 'flagged_nontarget': int((flag & (islab == 0)).sum()),
               'nontarget_n': int((islab == 0).sum())}
        for f in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
            m = fam == f
            row['recall'][f] = {'n': int(m.sum()),
                                'recall': float(flag[m].mean()) if m.sum() else None}
        row['macro_recall'] = float(np.mean([row['recall'][f]['recall']
                                             for f in row['recall']]))
        row['recall_all_positives'] = float(flag[islab == 1].mean())
        res['budgets'][str(bud)] = row
        print(f'budget {bud:>4}: flagged {row["flagged_all"]:>6} '
              f'(={row["flagged_rate_per_112540"]:.0f} per 112,540; placebo '
              f'{row["placebo_flagged_rate_per_112540"]:.0f}) '
              f'macro recall {row["macro_recall"]:.3f} '
              f'DT {row["recall"]["directed_transfer"]["recall"]:.3f} '
              f'SP {row["recall"]["soft_play"]["recall"]:.3f} '
              f'CI {row["recall"]["coordinated_isolation"]["recall"]:.3f} '
              f'| non-targets flagged {row["flagged_nontarget"]}/{row["nontarget_n"]}')

    # ---- which events carry the disclosed families (defines the KNOWN-FAMILY cluster) -----
    det = load_detail(TAG)
    key = elig.select('tidx', 'a', 'b', 'label', 'behavior_family', 'n_shared')
    d2 = det.join(key, on=['tidx', 'a', 'b'], how='inner')
    d3 = det.join(key.rename({'a': 'b', 'b': 'a'}), on=['tidx', 'a', 'b'], how='inner')
    dd = pl.concat([d2, d3])
    npos = pos.height
    ev = (dd.filter(pl.col('label') == 1).group_by('event')
          .agg(pl.struct('tidx', 'a', 'b').n_unique().alias('n_pairs'))
          .sort('n_pairs', descending=True))
    evn = (dd.filter(pl.col('label').is_null()).group_by('event')
           .agg(pl.struct('tidx', 'a', 'b').n_unique().alias('n_unlab')))
    nunl = elig.filter(pl.col('label').is_null()).height
    ev = ev.join(evn, on='event', how='left').with_columns(
        pl.col('n_unlab').fill_null(0),
        pl.Series('event_name', [NAMES[i] for i in ev['event'].to_list()]))
    ev = ev.with_columns((pl.col('n_pairs') / npos).alias('pos_rate'),
                         (pl.col('n_unlab') / nunl).alias('unlab_rate'))
    ev = ev.with_columns((pl.col('pos_rate') / (pl.col('unlab_rate') + 1e-9)).alias('lift'))
    ev.write_csv(OUT / 'step248b-dev-event-enrichment.csv')
    print('\ntop known-family events (p<1e-3 cell rate among labelled positives):')
    for r in ev.head(25).iter_rows(named=True):
        print(f'  {r["event_name"]:>22} pos {r["pos_rate"]:.3f} unlab {r["unlab_rate"]:.5f} '
              f'lift {r["lift"]:.0f}')

    res['stopping_rule'] = {
        'threshold': 0.30,
        'macro_recall_at_30': res['budgets']['30']['macro_recall'],
        'passes': bool(res['budgets']['30']['macro_recall'] >= 0.30)}
    (OUT / 'step248b-power.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\nSTOPPING RULE: macro recall at 30 false flags = '
          f'{res["budgets"]["30"]["macro_recall"]:.3f} -> '
          f'{"PASS" if res["stopping_rule"]["passes"] else "FAIL"}  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
