"""STEP281 stage B: does the victim instrument SEE the disclosed families?  (gate G1)

Development only.  Labels are used here and nowhere else: this stage decides whether the
instrument has any power at all and fixes the scoring rule stage C applies blind to evaluation.

Power is defined as STEP248 defined it so the numbers are comparable: the threshold is set so
the null produces the stated number of false flags in a 112,540-pair population, and power is
the share of each family's labelled positives flagged.  STEP248's event instrument scored macro
.293 (DT .674, SP .179, CI .024) and failed its gate.

Two things changed after the first run:
  * the block-permutation null failed its blind control (126 flags per 112,540 against a
    nominal 30) and is replaced by the CROSS-TABLE null, which keeps every player intact;
  * the blind control is now MATCHED -- the same statistics read in the direction the mechanism
    does NOT predict, through the identical strata and threshold.
"""
import argparse
import json
import os

os.environ['POLARS_MAX_THREADS'] = '6'
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
EVAL_N = 112_540

# statistic -> the tail the mechanism predicts.  Fixed before any threshold was computed.
CELLS = [
    ('v_fold_excess', +1),        # victims fold more when both members are in the pot
    ('v_foldany_excess', +1),     # including blinds squeezed off preflop
    ('v_bleed_excess', +1),       # victims invest and are then pushed off
    ('v_take_excess', +1),        # victims lose more
    ('v_pressure_excess', +1),    # more aggression while a victim is still live
    ('v_squeeze', +1),            # both aggressive and a victim folds
    ('v_bothaggr', +1),           # both aggressive in the same pot
    ('v_conc', +1),               # the joint take is concentrated on ONE victim
    ('v_victim_take', +1),
    ('v_jointnet_z', +1),         # they win together off the victim (isolation)
    ('v_jointnet_z', -1),         # or off each other (transfer)
    ('v_conflict', -1),           # they limit conflict between themselves
]
# matched placebo: the same statistics read the WRONG way.  Same noise, same strata, no
# mechanism content.  v_jointnet_z is excluded because both of its tails are real cells.
PLACEBO = [(n, -s) for n, s in CELLS if n != 'v_jointnet_z']


def load(name):
    return pl.read_parquet(OUT / name)


def labelled():
    lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv')
    return lab.with_columns(pl.min_horizontal('player_1', 'player_2').alias('p1'),
                            pl.max_horizontal('player_1', 'player_2').alias('p2')) \
              .select('pair_id', 'p1', 'p2', 'label', 'behavior_family')


def strata(n_b, edges):
    return np.clip(np.searchsorted(edges, n_b, side='right') - 1, 0, len(edges) - 2)


def cell_scores(obs, null_vals, null_str, edges, cells):
    """-log10 one-sided tail probability per cell, read inside the pair's n_B stratum."""
    so = strata(obs['n_B'].to_numpy(), edges)
    out = np.zeros((obs.height, len(cells)))
    for k, (name, sign) in enumerate(cells):
        if name not in obs.columns or name not in null_vals:
            continue
        vo = obs[name].to_numpy() * sign
        vn = null_vals[name] * sign
        for s in range(len(edges) - 1):
            mo = so == s
            if not mo.any():
                continue
            mn = null_str == s
            ref = np.sort(vn[mn]) if mn.sum() > 200 else np.sort(vn)
            r = np.searchsorted(ref, vo[mo], side='left')
            out[mo, k] = -np.log10((len(ref) - r + 1) / (len(ref) + 1))
    return out


def run(nullfiles, tag):
    obs = load('victim_development_trunc2000.parquet').filter(pl.col('n_shared') >= 38)
    nulls = [load(f) for f in nullfiles]
    nulls = [n.filter(pl.col('n_shared') >= 38) for n in nulls]
    nn = pl.concat(nulls, how='diagonal')
    print(f'eligible development pairs {obs.height}, null pairs {nn.height}')

    edges = np.unique(np.quantile(obs['n_B'].to_numpy(), np.linspace(0, 1, 11)))
    edges[-1] = np.inf
    null_vals = {c: nn[c].to_numpy() for c in nn.columns if c.startswith(('v_', 'n_'))}
    null_str = strata(nn['n_B'].to_numpy(), edges)

    cells = [c for c in CELLS if c[0] in null_vals]
    plac = [c for c in PLACEBO if c[0] in null_vals]
    dropped = sorted({n for n, _ in CELLS} - {n for n, _ in cells})
    if dropped:
        print(f'cells dropped (not carried by this null): {dropped}')

    S = cell_scores(obs, null_vals, null_str, edges, cells)
    Sp = cell_scores(obs, null_vals, null_str, edges, plac)
    # the null scored against itself, for the threshold and for the calibration check
    nobs = nn.sample(fraction=1.0, shuffle=True, seed=0) if False else nn
    Sn = cell_scores(nobs, null_vals, null_str, edges, cells)
    Snp = cell_scores(nobs, null_vals, null_str, edges, plac)

    lab = labelled()
    key = obs.select('p1', 'p2').with_row_index('i').join(lab, on=['p1', 'p2'], how='left')
    famv = np.full(obs.height, 'unlabelled', object)
    famv[key['i'].to_numpy()] = key['behavior_family'].fill_null('unlabelled').to_numpy()
    is_neg = famv == 'none'
    families = ['directed_transfer', 'soft_play', 'coordinated_isolation']
    print('labelled here:', {f: int((famv == f).sum()) for f in families},
          '| non-targets', int(is_neg.sum()))

    auc = {}
    for k, (name, sign) in enumerate(cells):
        neg = S[is_neg, k]
        row = {}
        for f in families:
            pos = S[famv == f, k]
            rank = np.concatenate([pos, neg]).argsort().argsort()
            row[f] = round(float((rank[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                                 / (len(pos) * len(neg))), 3)
        auc[f'{name}{"+" if sign > 0 else "-"}'] = row

    score, score_n = S.max(axis=1), Sn.max(axis=1)
    plac_s, plac_n = Sp.max(axis=1), Snp.max(axis=1)
    rows = []
    for exp_false in (10, 30, 100):
        rate = exp_false / EVAL_N
        thr = np.quantile(score_n, 1 - rate)
        flag = score >= thr
        rec = {f: round(float((flag & (famv == f)).sum() / max((famv == f).sum(), 1)), 3)
               for f in families}
        thr_p = np.quantile(plac_n, 1 - rate)
        rows.append({'exp_false': exp_false, 'flagged': int(flag.sum()),
                     'flagged_per_112540': round(flag.sum() / obs.height * EVAL_N, 1),
                     'macro_recall': round(float(np.mean(list(rec.values()))), 3), **rec,
                     'non_targets_flagged': int((flag & is_neg).sum()),
                     'placebo_flagged_per_112540':
                         round((plac_s >= thr_p).sum() / obs.height * EVAL_N, 1)})
        print(rows[-1])

    gate = rows[1]
    verdict = {'G1_macro>=.30': bool(gate['macro_recall'] >= .30),
               'G1_CI>=.15': bool(gate['coordinated_isolation'] >= .15)}
    best_ci = max(v['coordinated_isolation'] for v in auc.values())
    res = {'null': nullfiles, 'eligible': obs.height, 'auc_vs_non_targets': auc,
           'power': rows, 'gate_G1': verdict, 'PASS': all(verdict.values()),
           'best_CI_auc': best_ci, 'strata_edges': [float(x) for x in edges[:-1]] + ['inf'],
           'cells': [[n, s] for n, s in cells]}
    (OUT / f'power281_{tag}.json').write_text(json.dumps(res, indent=2))
    np.save(OUT / f'dev_cells_{tag}.npy', S)
    obs.select('table_id', 'p1', 'p2', 'n_shared', 'n_B').write_parquet(OUT / 'dev_keys.parquet')
    print('\nAUC vs confirmed non-targets:')
    for k, v in auc.items():
        print(f'  {k:24s} DT {v["directed_transfer"]:.3f}  SP {v["soft_play"]:.3f}  '
              f'CI {v["coordinated_isolation"]:.3f}')
    print(f'\nbest single-cell CI AUC {best_ci:.3f} (previous project best .632)')
    print('GATE G1:', verdict, '->', 'PASS' if res['PASS'] else 'FAIL')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nulls', nargs='+', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    run(a.nulls, a.tag)
