"""STEP248 Part B.4: apply the FROZEN instrument to all 112,540 evaluation candidate pairs.

BLOCK  = STEP207 risk >= .5 (909 pairs): the positive control.  Most of these should fire, and
         on the known families' events.
TAIL   = the other 111,631: where an untrained family must live (OBS 15.1 puts an unseen family
         at risk .02-.25).

Two external nulls per event:
  (i)  the permutation null (offset-preserving relabelling), scaled to the tail;
  (ii) the DEVELOPMENT-TRUNCATED population rate, labelled positives removed.

An event is KNOWN-FAMILY if at least 5 % of the development labelled positives carry a
significant cell on it.  A hidden-family candidate is a flagged TAIL pair whose significant
cells contain NO known-family event.
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
from step248_b_events import NAMES, NN
from step248_b_run import R_PERM
from step248_b_events import SCRATCH

SUB207 = (ORIGINAL / 'Claude Version' / 'outputs_step207_20260917' /
          'submission_step207_ens4_ci_dt_rolestrength_on_step169.csv')
MIN_SHARED = 38
BUDGETS = [10, 30, 100]
PRIMARY = 30
KNOWN_POSRATE = 0.05


def ev_totals(tag):
    d = SCRATCH / tag / 'out'
    real = np.concatenate([np.load(p) for p in sorted(d.glob('evreal_*.npy'))]).sum(0)
    nul = np.concatenate([np.load(p) for p in sorted(d.glob('evnull_*.npy'))])
    return real, nul.sum(0) / R_PERM, nul.shape[0] // R_PERM


def main():
    t0 = time.time()
    res = {}
    # ---------------- evaluation side --------------------------------------------------
    pairs, null, null_ns, _ = load_phase('evaluation')
    pairs = attach_players(pairs, 'evaluation')
    ep = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p_lo'),
        pl.max_horizontal('player_1', 'player_2').alias('p_hi'))
    sub = pl.read_csv(SUB207, columns=['pair_id', 'risk_score'])
    ep = ep.join(sub, on='pair_id', how='inner')
    cand = pairs.join(ep.select('p_lo', 'p_hi', 'pair_id', 'shared_hands', 'risk_score'),
                      on=['p_lo', 'p_hi'], how='inner')
    assert cand.height == EVAL_POP, cand.height
    assert (cand['n_shared'] == cand['shared_hands']).all()
    print(f'evaluation candidates {cand.height}  [{time.time()-t0:.0f}s]')

    ns = cand['n_shared'].to_numpy()
    edges = np.quantile(ns, np.linspace(.1, .9, 9))
    st = strata(ns, edges)
    nm = null_ns >= MIN_SHARED
    s1 = cand['S1'].to_numpy()
    s1p = cand['S1_plc'].to_numpy()
    isblock = cand['risk_score'].to_numpy() >= 0.5

    flags = {}
    for bud in BUDGETS:
        thr = null_thresholds(null[nm, 0], null_ns[nm], edges, bud)
        thr_p = null_thresholds(null[nm, 2], null_ns[nm], edges, bud)
        flags[bud] = (s1 >= thr[st], s1p >= thr_p[st], thr)
    for bud in BUDGETS:
        f, fp, thr = flags[bud]
        print(f'budget {bud:>4}: flagged {f.sum():>5} total | BLOCK {int((f & isblock).sum())}'
              f'/{int(isblock.sum())} | TAIL {int((f & ~isblock).sum())}/{int((~isblock).sum())}'
              f' | placebo {int(fp.sum())} (nominal {bud})')
    res['flag_counts'] = {str(b): {'flagged': int(flags[b][0].sum()),
                                   'block_flagged': int((flags[b][0] & isblock).sum()),
                                   'tail_flagged': int((flags[b][0] & ~isblock).sum()),
                                   'placebo_flagged': int(flags[b][1].sum()),
                                   'nominal_false': b} for b in BUDGETS}

    # ---------------- known-family events ------------------------------------------------
    enr = pl.read_csv(OUT / 'step248b-dev-event-enrichment.csv')
    known = set(enr.filter(pl.col('pos_rate') >= KNOWN_POSRATE)['event'].to_list())
    print(f'\nKNOWN-family events ({len(known)} of {NN}): '
          f'{", ".join(NAMES[i] for i in sorted(known))}')
    res['known_events'] = [NAMES[i] for i in sorted(known)]

    # ---------------- per-event discovery table ------------------------------------------
    det_e = load_detail('evaluation')
    key = cand.select('tidx', 'a', 'b', 'pair_id', 'risk_score', 'n_shared', 'S1', 'S2')
    de = pl.concat([det_e.join(key, on=['tidx', 'a', 'b'], how='inner'),
                    det_e.join(key.rename({'a': 'b', 'b': 'a'}), on=['tidx', 'a', 'b'],
                               how='inner')])
    de = de.with_columns((pl.col('risk_score') >= 0.5).alias('is_block'))
    ev_real_all, ev_null_all, ntab = ev_totals('evaluation')
    tail_frac = float((~isblock).sum()) / 174000.0

    # development-truncated external rate (labelled positives removed)
    dpairs, dnull, dnull_ns, _ = load_phase('development2000')
    dpairs = attach_players(dpairs, 'development2000')
    dlab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p_lo'),
        pl.max_horizontal('player_1', 'player_2').alias('p_hi'))
    dpairs = dpairs.join(dlab.select('p_lo', 'p_hi', 'label', 'behavior_family'),
                         on=['p_lo', 'p_hi'], how='left')
    delig = dpairs.filter((pl.col('n_shared') >= MIN_SHARED) &
                          (pl.col('label').fill_null(-1) != 1))
    det_d = load_detail('development2000')
    dkey = delig.select('tidx', 'a', 'b')
    dd = pl.concat([det_d.join(dkey, on=['tidx', 'a', 'b'], how='inner'),
                    det_d.join(dkey.rename({'a': 'b', 'b': 'a'}), on=['tidx', 'a', 'b'],
                               how='inner')])
    n_tail = int((~isblock).sum())
    rows = []
    tailpairs = de.filter(~pl.col('is_block'))
    for e in range(NN):
        t_n = tailpairs.filter(pl.col('event') == e).select('pair_id').n_unique()
        b_n = de.filter((pl.col('event') == e) & pl.col('is_block')).select('pair_id').n_unique()
        d_n = dd.filter(pl.col('event') == e).select(
            pl.struct('tidx', 'a', 'b')).n_unique()
        exp_perm = float(ev_null_all[e]) * tail_frac
        exp_dev = d_n / delig.height * n_tail
        rows.append({'event': NAMES[e], 'known_family': e in known,
                     'placebo': NAMES[e].startswith('plc_'),
                     'tail_pairs': t_n, 'block_pairs': b_n,
                     'exp_permutation': exp_perm, 'exp_development': exp_dev,
                     'excess_vs_perm': t_n - exp_perm, 'excess_vs_dev': t_n - exp_dev,
                     'ratio_vs_perm': t_n / max(exp_perm, .5),
                     'ratio_vs_dev': t_n / max(exp_dev, .5)})
    evt = pl.DataFrame(rows).sort('excess_vs_dev', descending=True)
    evt.write_csv(OUT / 'step248b-event-discovery.csv')
    print('\n--- per-event discovery (evaluation TAIL vs the two nulls) ---')
    print('event                  known  tail  block   exp_perm  exp_dev  exc_dev  ratio_dev')
    for r in evt.head(28).iter_rows(named=True):
        print(f'{r["event"]:>22} {"K" if r["known_family"] else ("P" if r["placebo"] else "-")}'
              f'  {r["tail_pairs"]:>5} {r["block_pairs"]:>5}  {r["exp_permutation"]:>8.1f}'
              f' {r["exp_development"]:>8.1f} {r["excess_vs_dev"]:>+8.1f}'
              f' {r["ratio_vs_dev"]:>8.2f}')
    print('  ... placebo rows:')
    for r in evt.filter(pl.col('placebo')).iter_rows(named=True):
        print(f'{r["event"]:>22} P  {r["tail_pairs"]:>5} {r["block_pairs"]:>5} '
              f' {r["exp_permutation"]:>8.1f} {r["exp_development"]:>8.1f}'
              f' {r["excess_vs_dev"]:>+8.1f} {r["ratio_vs_dev"]:>8.2f}')

    # ---------------- candidate lists ------------------------------------------------------
    f30 = flags[PRIMARY][0]
    cand = cand.with_columns(pl.Series('flag30', f30), pl.Series('flag100', flags[100][0]),
                             pl.Series('is_block', isblock))
    # which known/unknown events each flagged pair fires
    fired = (de.group_by('pair_id').agg(
        pl.col('event').unique().alias('events'),
        pl.col('pval').min().alias('min_pval'),
        pl.len().alias('n_cells')))
    fired = fired.with_columns(
        pl.col('events').list.eval(pl.element().is_in(list(known))).list.any()
        .alias('fires_known'),
        pl.col('events').list.eval(~pl.element().is_in(list(known))).list.any()
        .alias('fires_other'))
    cc = cand.join(fired, on='pair_id', how='left')

    # spill-over: does the pair share a player with a block pair?
    bp = cand.filter(pl.col('is_block'))
    bplayers = set(bp['player_a'].to_list()) | set(bp['player_b'].to_list())
    cc = cc.with_columns(
        (pl.col('player_a').is_in(list(bplayers)) |
         pl.col('player_b').is_in(list(bplayers))).alias('shares_block_player'))

    tail_flag = cc.filter(pl.col('flag30') & ~pl.col('is_block'))
    hidden = tail_flag.filter(pl.col('fires_known').fill_null(False) == False)
    lowdose = tail_flag.filter(pl.col('fires_known').fill_null(False))
    exp_false_perm = PRIMARY * n_tail / EVAL_POP
    devrate = None
    dthr = null_thresholds(dnull[dnull_ns >= MIN_SHARED, 0], dnull_ns[dnull_ns >= MIN_SHARED],
                           edges, PRIMARY)
    dns = delig['n_shared'].to_numpy()
    dflag = delig['S1'].to_numpy() >= dthr[strata(dns, edges)]
    devrate = float(dflag.mean())
    exp_false_dev = devrate * n_tail
    print(f'\nTAIL flagged at budget {PRIMARY}: {tail_flag.height} '
          f'(hidden-type {hidden.height}, low-dose-known {lowdose.height})')
    print(f'  expected under permutation null : {exp_false_perm:.1f}')
    print(f'  expected at the development-truncated population rate ({devrate*1e5:.1f} per 100k)'
          f' : {exp_false_dev:.1f}')
    # the same split for the development null, so the comparison is like for like
    ddet_key = delig.select('tidx', 'a', 'b')
    ddflag = delig.filter(pl.Series(dflag))
    dfired = (dd.group_by(['tidx', 'a', 'b']).agg(pl.col('event').unique().alias('events')))
    dfired = dfired.with_columns(
        pl.col('events').list.eval(pl.element().is_in(list(known))).list.any()
        .alias('fires_known'))
    dj = ddflag.join(dfired, on=['tidx', 'a', 'b'], how='left')
    dj2 = ddflag.join(dfired.rename({'a': 'b', 'b': 'a'}), on=['tidx', 'a', 'b'], how='left')
    dk = (dj['fires_known'].fill_null(False) | dj2['fires_known'].fill_null(False))
    d_hidden_rate = float((~dk).sum()) / delig.height
    print(f'  development-truncated HIDDEN-type flagged rate: {(~dk).sum()} / {delig.height} '
          f'= {d_hidden_rate*1e5:.1f} per 100k -> expected {d_hidden_rate*n_tail:.1f} in the tail')
    res['tail'] = {'n_tail': n_tail, 'flagged': tail_flag.height,
                   'hidden_type': hidden.height, 'lowdose_known': lowdose.height,
                   'expected_permutation': exp_false_perm,
                   'development_rate_per_100k': devrate * 1e5,
                   'expected_development': exp_false_dev,
                   'development_hidden_rate_per_100k': d_hidden_rate * 1e5,
                   'expected_development_hidden': d_hidden_rate * n_tail,
                   'precision_vs_perm': 1 - exp_false_perm / max(tail_flag.height, 1),
                   'precision_vs_dev': 1 - exp_false_dev / max(tail_flag.height, 1),
                   'hidden_precision_vs_dev':
                       1 - d_hidden_rate * n_tail / max(hidden.height, 1)}

    # block positive control: do block pairs fire, and on known events?
    bf = cc.filter(pl.col('is_block'))
    res['block_control'] = {
        'n': bf.height, 'flagged30': int(bf['flag30'].sum()),
        'fires_any_cell': int(bf['n_cells'].is_not_null().sum()),
        'fires_known': int(bf['fires_known'].fill_null(False).sum())}
    print(f"\nBLOCK control: {res['block_control']['flagged30']}/{bf.height} flagged at "
          f"budget {PRIMARY}; {res['block_control']['fires_known']} carry a known-family cell")

    keep = ['pair_id', 'player_a', 'player_b', 'table_id', 'tidx', 'a', 'b', 'n_shared',
            'risk_score', 'S1', 'S2', 'min_pval', 'n_cells', 'fires_known', 'fires_other',
            'shares_block_player']
    hidden.select(keep).sort('S1', descending=True).write_parquet(
        OUT / 'candidates_hidden.parquet')
    lowdose.select(keep).sort('S1', descending=True).write_parquet(
        OUT / 'candidates_lowdose_known.parquet')
    cc.select(keep + ['flag30', 'flag100', 'is_block']).write_parquet(
        OUT / 'step248b-eval-pair-scores.parquet')
    (OUT / 'step248b-discovery.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\nwritten; done  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
