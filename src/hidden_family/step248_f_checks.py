"""STEP248 closing checks.

(1) PARTNER-STYLE CONFOUND.  Events whose TRIGGER depends on the partner's own behaviour
    (iso_raise_b_limp needs b to limp; coldcall_b_raise needs b to raise) are not fully
    controlled by the leave-partner-out baseline: a partner who limps twice as often as a's
    other table-mates produces "excess" with no coordination at all.  27 of the 31 hidden-type
    candidates fire on iso_raise_b_limp, so this is measured directly: the limper's own
    table-wide limp rate, as a percentile among the table's 30 players.

(2) PART A SENSITIVITY, non-circular.  The first sensitivity arm trimmed the development null
    on contest_both and flow_z and was then thresholded on those same statistics, which is
    circular and produced a fake excess on every statistic.  It is replaced here: the
    development null drops labelled positives AND every pair carrying a significant
    known-family EVENT cell from Part B, which is an independent instrument.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_c_common import ORIGINAL, OUT, attach_players, load_detail, load_phase, strata
from step248_b_events import DENI, EVI, NAMES, NN, SCRATCH
from step248_b_run import load_chunk, emit_table, accumulate, N_PL
from step248_a_analyse import BLIND, STATS, boot_threshold_indices, excess_table, pair_key

TABLES_PER_CHUNK = 10
MIN_SHARED = 38


def partner_style_check():
    hid = pl.read_parquet(OUT / 'candidates_hidden.parquet')
    fh = pl.read_parquet(OUT / 'step248b-flagged-hands.parquet')
    tgt = fh.filter(pl.col('event_name') == 'iso_raise_b_limp').select(
        'pair_id', 'tidx', 'A', 'B').unique()
    tgt = tgt.filter(pl.col('pair_id').is_in(hid['pair_id'].to_list()))
    print(f'iso_raise_b_limp directed cells among hidden candidates: {tgt.height}')
    rows = []
    for t, grp in tgt.partition_by('tidx', as_dict=True).items():
        tt = t[0] if isinstance(t, tuple) else t
        ch = tt // TABLES_PER_CHUNK
        h, s, a = load_chunk('evaluation', ch)
        hm = h.filter(pl.col('tidx') == tt).sort('hloc')
        sf = s.filter(pl.col('tidx') == tt).sort(['hidx', 'seat_no'])
        af = a.filter(pl.col('tidx') == tt).sort(['hidx', 'action_no'])
        v, P6, HR, PART, nH = emit_table(hm, sf, af)
        c = accumulate(v, P6, HR, PART)
        limp = c[:, :, EVI['limp_b_live'], :].sum((1, 2))
        pfd = c[:, :, NN + DENI['d_pf_blive'], :].sum((1, 2))
        rate = limp / np.maximum(pfd, 1)
        pct = (np.argsort(np.argsort(rate)) + .5) / N_PL
        for r in grp.iter_rows(named=True):
            rows.append({'pair_id': r['pair_id'], 'raiser': int(r['A']),
                         'limper': int(r['B']),
                         'limper_limp_rate': float(rate[r['B']]),
                         'limper_pct_in_table': float(pct[r['B']]),
                         'raiser_pct_in_table': float(pct[r['A']]),
                         'table_median_limp_rate': float(np.median(rate))})
    df = pl.DataFrame(rows)
    df.write_csv(OUT / 'step248b-partner-style-check.csv')
    print(f'  limper own limp-rate percentile inside his table: median '
          f'{df["limper_pct_in_table"].median():.3f}, '
          f'share >= .80: {(df["limper_pct_in_table"] >= .8).mean():.2f} '
          f'(0.20 expected if partner style were irrelevant)')
    print(f'  limper limp rate {df["limper_limp_rate"].median():.3f} vs table median '
          f'{df["table_median_limp_rate"].median():.3f} '
          f'(ratio {df["limper_limp_rate"].median()/df["table_median_limp_rate"].median():.2f})')
    return {'n': df.height,
            'limper_pct_median': float(df['limper_pct_in_table'].median()),
            'share_pct_ge_80': float((df['limper_pct_in_table'] >= .8).mean()),
            'limper_rate_median': float(df['limper_limp_rate'].median()),
            'table_median_rate': float(df['table_median_limp_rate'].median())}


def part_a_sensitivity():
    rng = np.random.default_rng(2480)
    ev = pl.read_parquet(OUT / 'step248a-mechstats-evaluation.parquet').with_columns(
        *pair_key('player_a', 'player_b'))
    dv = pl.read_parquet(OUT / 'step248a-mechstats-development2000.parquet').with_columns(
        *pair_key('player_a', 'player_b'))
    ep = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').with_columns(
        *pair_key('player_1', 'player_2'))
    sub = pl.read_csv(ORIGINAL / 'Claude Version' / 'outputs_step207_20260917' /
                      'submission_step207_ens4_ci_dt_rolestrength_on_step169.csv',
                      columns=['pair_id', 'risk_score'])
    ep = ep.join(sub, on='pair_id', how='inner')
    ev = ev.join(ep.select('p_lo', 'p_hi', 'pair_id', 'risk_score'), on=['p_lo', 'p_hi'],
                 how='inner')
    tail = ev.filter(pl.col('risk_score') < 0.5)

    dl = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').with_columns(
        *pair_key('player_1', 'player_2'))
    pos = dl.filter(pl.col('label') == 1).select('p_lo', 'p_hi')
    enr = pl.read_csv(OUT / 'step248b-dev-event-enrichment.csv')
    known = enr.filter(pl.col('pos_rate') >= 0.05)['event'].to_list()
    dpairs, _, _, _ = load_phase('development2000')
    dpairs = attach_players(dpairs, 'development2000')
    det = load_detail('development2000').filter(pl.col('event').is_in(known))
    ks = det.select('tidx', 'a', 'b').unique()
    ks2 = det.select('tidx', pl.col('b').alias('a'), pl.col('a').alias('b')).unique()
    flaggedk = pl.concat([ks, ks2]).unique()
    dk = dpairs.join(flaggedk.with_columns(pl.lit(True).alias('kf')), on=['tidx', 'a', 'b'],
                     how='left').with_columns(pl.col('kf').fill_null(False))
    drop = set(zip(dk.filter('kf')['p_lo'].to_list(), dk.filter('kf')['p_hi'].to_list()))
    drop |= set(zip(pos['p_lo'].to_list(), pos['p_hi'].to_list()))
    dv = dv.filter(pl.col('n_both') >= MIN_SHARED)
    keepm = [(lo, hi) not in drop for lo, hi in zip(dv['p_lo'].to_list(), dv['p_hi'].to_list())]
    dvn = dv.filter(pl.Series(keepm))
    print(f'\nPart A sensitivity (non-circular): dev null {dvn.height} '
          f'(dropped {dv.height - dvn.height} labelled positives + known-family-event pairs)')

    ne = tail['n_both'].to_numpy()
    edges = np.quantile(ne, np.linspace(0, 1, 11)); edges[0] -= 1; edges[-1] += 1
    se = np.clip(np.digitize(ne, edges[1:-1]), 0, 9)
    sn = np.clip(np.digitize(dvn['n_both'].to_numpy(), edges[1:-1]), 0, 9)
    rows = []
    for st in STATS:
        e = tail[st].to_numpy().astype(float)
        n = dvn[st].to_numpy().astype(float)
        for q in (0.99, 0.999):
            for up in (True, False):
                obs, exp, exc, lo, hi = excess_table(e, n, se, sn, 10, q, up, rng)
                rows.append({'stat': st, 'q': q, 'tail': 'upper' if up else 'lower',
                             'obs': obs, 'exp': exp, 'excess': exc, 'ci_lo': lo, 'ci_hi': hi,
                             'blind_control': st in BLIND})
    out = pl.DataFrame(rows).sort('excess', descending=True)
    out.write_csv(OUT / 'step248a-sensitivity-noncircular.csv')
    print('  top rows at q99 (upper):')
    for r in out.filter((pl.col('q') == 0.99) & (pl.col('tail') == 'upper')).head(8).iter_rows(
            named=True):
        print(f'    {r["stat"]:>14} obs {r["obs"]:7.0f} exp {r["exp"]:7.1f} '
              f'excess {r["excess"]:+8.1f} CI[{r["ci_lo"]:+8.1f},{r["ci_hi"]:+8.1f}]'
              f'{"  BLIND" if r["blind_control"] else ""}')
    return {'dev_null': dvn.height, 'dropped': dv.height - dvn.height}


def main():
    t0 = time.time()
    res = {'partner_style': partner_style_check()}
    res['part_a_sensitivity'] = part_a_sensitivity()
    (OUT / 'step248-closing-checks.json').write_text(json.dumps(res, indent=1),
                                                     encoding='utf-8')
    print(f'\ndone  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
