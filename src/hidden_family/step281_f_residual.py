"""STEP281 stage F: is any part of the evaluation excess coordination-shaped?

Stage D showed the evaluation tail as a whole has the composition of development's ordinary
background (cosine .998) rather than of the disclosed families (.793).  That is a statement
about the tail's average.  This stage asks the sharper question: inside that tail, is there a
SUBSET whose shape is coordination, and is that subset bigger in evaluation than development?

"Coordination-shaped" is defined from the labelled data before any evaluation count is taken:
the three cells that carry the disclosed families in this instrument -- v_pressure_excess,
v_squeeze and v_bothaggr -- supply .746 of a labelled positive's score and only .388 of a
background pair's.  A pair is called coordination-shaped when they supply at least .70.

The count is then split by where the pair sits in the production ranking.  OBS 15.1: an
untrained family collapses to risk .02-.24 but keeps a median percentile near .95, so a fourth
family lives in the upper-middle of the ranking, not in the block.
"""
import json
import os

os.environ['POLARS_MAX_THREADS'] = '6'
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtri

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
PROD = (ROOT / 'Claude Version' / 'outputs_step272_20260919'
        / 'submission_step272_blockM3xKM_on_step259b.csv')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step281_c_discover import CELLS, FAMILIES, Scorer, matched_rate, strat  # noqa: E402

COORD = ['v_pressure_excess', 'v_squeeze', 'v_bothaggr']
SHAPE_CUT = 0.70
BUDGET = 1000


def main():
    dev = pl.read_parquet(OUT / 'victim_development_trunc2000.parquet') \
            .filter(pl.col('n_shared') >= 38)
    ev = pl.read_parquet(OUT / 'victim_evaluation.parquet')
    pairs = pl.read_csv(ROOT / 'data' / 'evaluation_pairs.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p1'),
        pl.max_horizontal('player_1', 'player_2').alias('p2'))
    ev = ev.join(pairs.select('pair_id', 'p1', 'p2'), on=['p1', 'p2'], how='inner')
    lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p1'),
        pl.max_horizontal('player_1', 'player_2').alias('p2'))
    fam = dev.join(lab.select('p1', 'p2', 'behavior_family'), on=['p1', 'p2'], how='left')[
        'behavior_family'].fill_null('unlabelled').to_numpy()

    sc = Scorer(dev, CELLS)
    idx = [j for j, (n, _) in enumerate(CELLS) if n in COORD]

    def contrib(df):
        s = strat(df['n_B'].to_numpy(), sc.edges)
        z = np.zeros((df.height, len(CELLS)))
        for j, (name, sign) in enumerate(CELLS):
            v = df[name].to_numpy() * sign
            for k, ref in enumerate(sc.ref[(name, sign)]):
                m = s == k
                if not m.any() or len(ref) < 50:
                    continue
                p = np.searchsorted(ref, v[m], side='left') / len(ref)
                z[m, j] = np.maximum(ndtri(np.clip(p, .5, 1 - .5 / len(ref))), 0) ** 2
        return z

    zd, ze = contrib(dev), contrib(ev)
    ds, es = zd.sum(axis=1), ze.sum(axis=1)
    shape_d = zd[:, idx].sum(axis=1) / np.maximum(ds, 1e-9)
    shape_e = ze[:, idx].sum(axis=1) / np.maximum(es, 1e-9)
    thr = np.quantile(ds, 1 - BUDGET / 112_540)

    print(f'coordination-share of the score, medians:'
          f'  labelled positives {np.median(shape_d[np.isin(fam, FAMILIES)]):.3f}'
          f' | confirmed non-targets {np.median(shape_d[fam == "none"]):.3f}'
          f' | all development {np.median(shape_d):.3f}')

    # --- coordination-shaped flags, evaluation vs length-matched development -----------
    sh_e = np.unique(np.quantile(ev['n_shared'].to_numpy(), np.linspace(0, 1, 11)))
    sh_e[0], sh_e[-1] = -np.inf, np.inf
    nbin = len(sh_e) - 1
    dev_sh, ev_sh = strat(dev['n_shared'].to_numpy(), sh_e), strat(ev['n_shared'].to_numpy(), sh_e)
    w = np.array([(ev_sh == k).mean() / max((dev_sh == k).mean(), 1e-9) for k in range(nbin)])
    rng = np.random.default_rng(77)
    tabs = dev['table_id'].to_numpy()
    utab = np.unique(tabs)

    rows = []
    for lo, name in ((0.0, 'any shape'), (SHAPE_CUT, f'coordination-shaped (>= {SHAPE_CUT})')):
        dflag = (ds >= thr) & (shape_d >= lo)
        eflag = (es >= thr) & (shape_e >= lo)
        exp = float(np.sum([w[k] * (dflag & (dev_sh == k)).sum() / len(ds)
                            for k in range(nbin)]) * 112_540)
        null = []
        for _ in range(200):
            half = set(rng.choice(utab, len(utab) // 2, replace=False).tolist())
            m = np.fromiter((t in half for t in tabs), bool, len(tabs))
            ta, tb = ~m, m
            wa = np.array([(dev_sh[tb] == k).mean() / max((dev_sh[ta] == k).mean(), 1e-9)
                           for k in range(nbin)])
            null.append(dflag[tb].sum() / tb.sum() * 112_540
                        - float(np.sum([wa[k] * (dflag & ta & (dev_sh == k)).sum() / ta.sum()
                                        for k in range(nbin)]) * 112_540))
        sd = float(np.std(null))
        rows.append({'subset': name, 'eval': int(eflag.sum()), 'dev_expected': round(exp, 1),
                     'excess': round(eflag.sum() - exp, 1), 'sd': round(sd, 1),
                     'z': round((eflag.sum() - exp - np.mean(null)) / max(sd, 1e-9), 2),
                     'dev_labelled_positives_in_subset':
                         int((dflag & np.isin(fam, FAMILIES)).sum())})
        print(rows[-1])

    # --- where the coordination-shaped evaluation flags sit in production ---------------
    prod = pl.read_csv(PROD).select('pair_id', 'risk_score')
    ev2 = ev.select('pair_id', 'p1', 'p2', 'n_shared').with_columns(
        pl.Series('v', es), pl.Series('shape', shape_e)).join(prod, on='pair_id', how='left')
    r = ev2['risk_score'].to_numpy()
    rp = (np.argsort(np.argsort(r)) + 0.5) / len(r)
    ev2 = ev2.with_columns(pl.Series('risk_pct', rp))
    f = ev2.filter((pl.col('v') >= thr))
    print('\ncoordination-share of the evaluation tail, by production-risk band:')
    bands = []
    for lo, hi, nm in ((.0, .90, 'below .90'), (.90, .99, '.90-.99'),
                       (.99, .995, '.99-.995'), (.995, 1.01, 'top .5%')):
        g = f.filter((pl.col('risk_pct') >= lo) & (pl.col('risk_pct') < hi))
        shaped = int((g['shape'] >= SHAPE_CUT).sum())
        bands.append({'band': nm, 'flagged': g.height, 'coordination_shaped': shaped,
                      'median_shape': round(float(g['shape'].median()), 3) if g.height else None})
        print(f'  {nm:10s} flagged {g.height:5d}  coordination-shaped {shaped:4d}'
              f'  median share {bands[-1]["median_shape"]}')

    cands = f.filter((pl.col('shape') >= SHAPE_CUT) & (pl.col('risk_pct') < .995)
                     & (pl.col('risk_score') < .5)).sort('v', descending=True)
    print(f'\nCANDIDATES (flagged + coordination-shaped + outside the production block): '
          f'{cands.height}')
    cands.write_parquet(OUT / 'candidates281.parquet')
    json.dump({'shape_cut': SHAPE_CUT, 'budget': BUDGET, 'subsets': rows, 'bands': bands,
               'n_candidates': cands.height},
              open(OUT / 'residual281.json', 'w'), indent=2)
    print('wrote residual281.json')


if __name__ == '__main__':
    main()
