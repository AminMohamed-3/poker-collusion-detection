"""STEP281 stage D: what IS the evaluation excess?

Stage C measured that the evaluation phase puts ~2.0-2.7 sd more pairs in the victim
instrument's tail than a length-matched development population does.  Three things could
produce that and they have different signatures:

  1. a fourth mechanism          -> the excess fires a DIFFERENT combination of cells from the
                                    one the disclosed families fire;
  2. more disclosed positives    -> the excess fires the SAME combination the labelled
     in evaluation than in dev      development positives fire;
  3. a phase-level shift         -> the excess fires like the ordinary development background,
                                    and the matched placebo moves with it.

So this stage reads the excess cell by cell, compares its composition against the three
reference signatures, and checks whether the pairs carrying it are odd as PAIRS or odd as
PLAYERS (OBS 15.3's decoy statistic, which is what killed the 09-11 candidate population).
"""
import json
import os

os.environ['POLARS_MAX_THREADS'] = '6'
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtri

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
PROD = (ROOT / 'Claude Version' / 'outputs_step272_20260919'
        / 'submission_step272_blockM3xKM_on_step259b.csv')
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step281_c_discover import (CELLS, FAMILIES, Scorer, excess_statistic,  # noqa: E402
                                matched_rate, strat)

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

    sh_e = np.unique(np.quantile(ev['n_shared'].to_numpy(), np.linspace(0, 1, 11)))
    sh_e[0], sh_e[-1] = -np.inf, np.inf
    nbin = len(sh_e) - 1
    dev_sh = strat(dev['n_shared'].to_numpy(), sh_e)
    ev_sh = strat(ev['n_shared'].to_numpy(), sh_e)
    w = np.array([(ev_sh == k).mean() / max((dev_sh == k).mean(), 1e-9) for k in range(nbin)])
    rng = np.random.default_rng(5)
    tabs = dev['table_id'].to_numpy()
    utab = np.unique(tabs)

    # ---- 1. the excess, cell by cell -------------------------------------------------
    print(f'PER-CELL EXCESS at a {BUDGET}-pair budget (evaluation minus length-matched '
          f'development)\n')
    print(f'{"cell":24s} {"eval":>6s} {"dev-exp":>8s} {"excess":>8s} {"sd":>6s} {"z":>6s}'
          f' {"DT":>5s} {"SP":>5s} {"CI":>5s}')
    per_cell = []
    for name, sign in CELLS:
        one = Scorer(dev, [(name, sign)])
        ds, es = one(dev), one(ev)
        thr = np.quantile(ds, 1 - BUDGET / 112_540)
        exp = matched_rate(ds, dev_sh, w, thr, nbin) * 112_540
        obs = int((es >= thr).sum())
        null = []
        for _ in range(120):
            half = set(rng.choice(utab, len(utab) // 2, replace=False).tolist())
            m = np.fromiter((t in half for t in tabs), bool, len(tabs))
            ta, tb = ~m, m
            ta_thr = np.quantile(ds[ta], 1 - BUDGET / 112_540)
            wa = np.array([(dev_sh[tb] == k).mean() / max((dev_sh[ta] == k).mean(), 1e-9)
                           for k in range(nbin)])
            null.append((ds[tb] >= ta_thr).sum() / tb.sum() * 112_540
                        - matched_rate(ds[ta], dev_sh[ta], wa, ta_thr, nbin) * 112_540)
        sd = float(np.std(null))
        flag = ds >= thr
        rec = {f: float((flag & (fam == f)).sum() / max((fam == f).sum(), 1)) for f in FAMILIES}
        per_cell.append({'cell': f'{name}{"+" if sign > 0 else "-"}', 'eval': obs,
                         'dev_expected': round(exp, 1), 'excess': round(obs - exp, 1),
                         'sd': round(sd, 1),
                         'z': round((obs - exp - np.mean(null)) / max(sd, 1e-9), 2),
                         'recall': {k: round(v, 3) for k, v in rec.items()}})
        c = per_cell[-1]
        print(f'{c["cell"]:24s} {obs:6d} {exp:8.1f} {c["excess"]:+8.1f} {sd:6.1f} {c["z"]:+6.2f}'
              f' {rec["directed_transfer"]:5.2f} {rec["soft_play"]:5.2f}'
              f' {rec["coordinated_isolation"]:5.2f}')

    # ---- 2. composition: which cell drives each flagged pair -------------------------
    sc = Scorer(dev, CELLS)
    ds, es = sc(dev), sc(ev)
    thr = np.quantile(ds, 1 - BUDGET / 112_540)

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
    labels = [f'{n}{"+" if s > 0 else "-"}' for n, s in CELLS]
    groups = {'dev labelled positives': np.isin(fam, FAMILIES) & (ds >= thr),
              'dev background flagged': (~np.isin(fam, FAMILIES)) & (ds >= thr)}
    print('\nCOMPOSITION of the flagged tail -- share of the score each cell contributes\n')
    print(f'{"cell":24s} ' + ''.join(f'{g[:18]:>20s}' for g in groups) + f'{"eval flagged":>20s}')
    comp = {}
    for g, m in groups.items():
        comp[g] = zd[m].sum(axis=0) / max(zd[m].sum(), 1e-9)
    comp['eval flagged'] = ze[es >= thr].sum(axis=0) / max(ze[es >= thr].sum(), 1e-9)
    for j, lb in enumerate(labels):
        print(f'{lb:24s} ' + ''.join(f'{comp[g][j]:20.3f}' for g in comp))
    cos = {g: float(np.dot(comp[g], comp['eval flagged'])
                    / (np.linalg.norm(comp[g]) * np.linalg.norm(comp['eval flagged'])))
           for g in groups}
    print('\ncosine of the evaluation tail composition against:', {k: round(v, 4) for k, v in cos.items()})

    # ---- 3. odd pair, or odd players? ------------------------------------------------
    dev_ex = excess_statistic(dev.select('p1', 'p2'), ds)
    ev_ex = excess_statistic(ev.select('p1', 'p2'), es)
    print('\nDECOY STATISTIC (pair percentile minus the better of the two members\' medians)')
    for g, m in (('dev labelled positives', np.isin(fam, FAMILIES)),
                 ('dev positives, flagged', np.isin(fam, FAMILIES) & (ds >= thr)),
                 ('dev confirmed non-targets', fam == 'none'),
                 ('dev background flagged', (~np.isin(fam, FAMILIES)) & (ds >= thr))):
        print(f'  {g:28s} n {int(m.sum()):5d}  median {np.median(dev_ex[m]):+.3f}')
    print(f'  {"evaluation flagged":28s} n {int((es >= thr).sum()):5d}  '
          f'median {np.median(ev_ex[es >= thr]):+.3f}')

    # ---- 4. is the excess spread over tables, or a few? -------------------------------
    evt = ev.with_columns(pl.Series('f', (es >= thr).astype(int)))
    pert = evt.group_by('table_id').agg(pl.col('f').sum()).sort('f', descending=True)
    cnt = pert['f'].to_numpy()
    print(f'\nTABLE SPREAD of the {int(cnt.sum())} evaluation flags over 400 tables: '
          f'max {cnt.max()}, top-10 hold {cnt[:10].sum()}, tables with >=1 flag '
          f'{(cnt > 0).sum()}; Poisson-equivalent max {np.random.default_rng(0).poisson(cnt.mean(), 400).max()}')

    # ---- 5. production risk profile ---------------------------------------------------
    prod = pl.read_csv(PROD).select('pair_id', 'risk_score')
    ev2 = ev.select('pair_id').with_columns(pl.Series('v', es)).join(prod, on='pair_id',
                                                                     how='left')
    r = ev2['risk_score'].to_numpy()
    rp = (np.argsort(np.argsort(r)) + 0.5) / len(r)
    f = es >= thr
    print('\nPRODUCTION RISK of the flagged evaluation pairs:')
    for lo, hi, nm in ((.0, .90, 'below .90 pct'), (.90, .99, '.90-.99 pct'),
                       (.99, .995, '.99-.995 pct'), (.995, 1.01, 'top .5 pct')):
        m = f & (rp >= lo) & (rp < hi)
        exp_share = (hi - min(hi, 1.0)) if False else (min(hi, 1.0) - lo)
        print(f'  {nm:16s} {int(m.sum()):5d} flagged  ({int(m.sum()) / max(f.sum(), 1):.3f} of '
              f'the tail, {exp_share:.3f} expected if unrelated)')
    json.dump({'per_cell': per_cell, 'composition': {k: list(map(float, v)) for k, v in comp.items()},
               'cosine_vs_eval': cos, 'cells': labels},
              open(OUT / 'characterise281.json', 'w'), indent=2)
    print('\nwrote characterise281.json')


if __name__ == '__main__':
    main()
