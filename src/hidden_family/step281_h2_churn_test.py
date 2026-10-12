"""STEP281 stage H2: do the CANCELLING-TRANSFER features add to the production pair model?

Same harness and same protocol as stage G, which killed the victim features: the three prefix-60
fold frames, table-grouped out-of-fold logistic, AP computed INSIDE each held-out group, and 20
row-permuted null banks so "adding columns helps a logistic by construction" is priced in.

Kill criterion fixed before the read, identical to stage G's: delta below the shuffled-column
q95 on def C.
"""
import json
import os

os.environ['POLARS_MAX_THREADS'] = '6'
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
PU = ROOT / 'Claude Version' / 'experiments' / 'pu_pairs_20260911'
CCOLS = ['churn', 'net_flow', 'churn_ratio', 'churn_cancel', 'gross_bb', 'net_bb',
         'cancel_frac']
FAMILIES = ('directed_transfer', 'soft_play', 'coordinated_isolation')


def ap(y, s):
    o = np.argsort(-s)
    y = y[o]
    c = np.cumsum(y)
    return float(((c / np.arange(1, len(y) + 1)) * y).sum() / max(y.sum(), 1))


def rank_normal(v):
    r = (np.argsort(np.argsort(v)) + 0.5) / len(v)
    return np.nan_to_num(ndtri(np.clip(r, 1e-4, 1 - 1e-4)))


def fit_oof(X, y, gi, n_groups, C=0.05):
    out = np.full(len(y), np.nan)
    Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    for g in range(n_groups):
        tr, te = gi != g, gi == g
        if y[tr].sum() < 3 or not te.any():
            continue
        out[te] = LogisticRegression(C=C, max_iter=3000).fit(Xs[tr], y[tr]) \
            .decision_function(Xs[te])
    return out


def grouped_ap(y, s, gi, n_groups, keep):
    num = den = 0.0
    for g in range(n_groups):
        m = (gi == g) & keep & ~np.isnan(s)
        if y[m].sum() < 1:
            continue
        num += ap(y[m], s[m]) * y[m].sum()
        den += y[m].sum()
    return float(num / max(den, 1))


def main():
    ch = pl.read_parquet(OUT / 'churn_dev_prefix60.parquet').drop('table_id')
    known = pl.read_csv(ROOT / 'data' / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p1'),
        pl.max_horizontal('player_1', 'player_2').alias('p2')).select('p1', 'p2', 'label_status')

    rows, diag = [], []
    for fold in (0, 1, 2):
        s = pl.read_parquet(PU / f'fold{fold}-prefix60' / 'scores.parquet').with_columns(
            pl.min_horizontal('player_1', 'player_2').alias('p1'),
            pl.max_horizontal('player_1', 'player_2').alias('p2'))
        j = (s.join(ch, on=['p1', 'p2'], how='inner')
              .join(known, on=['p1', 'p2'], how='left')
              .filter(pl.col('shared_hands') >= 38))
        y = (j['label'].to_numpy() == 1).astype(int)
        lab = j['label_status'].fill_null('unlabelled').to_numpy() != 'unlabelled'
        risk = j['risk'].to_numpy()
        lr = np.log(np.clip(risk, 1e-9, 1) / np.clip(1 - risk, 1e-9, 1))
        Z = np.column_stack([rank_normal(j[c].to_numpy()) for c in CCOLS])

        tab = j['table_id'].to_numpy()
        rg = np.random.default_rng(3)
        grp = {t: i % 5 for i, t in enumerate(rg.permutation(np.unique(tab)))}
        gi = np.fromiter((grp[t] for t in tab), int, len(tab))
        base = fit_oof(lr[:, None], y, gi, 5)
        comb = fit_oof(np.column_stack([lr, Z]), y, gi, 5)
        rn = np.random.default_rng(fold + 100)
        nulls = [fit_oof(np.column_stack([lr, Z[rn.permutation(len(Z))]]), y, gi, 5)
                 for _ in range(20)]

        # is the cancelling shape on its own informative about the pairs production misses?
        cf = j['cancel_frac'].to_numpy()
        fam = j['behavior_family'].to_numpy()
        low = (y == 1) & (risk < .5)
        diag.append({'fold': fold, 'missed': int(low.sum()),
                     'cancel_frac_missed': round(float(np.median(cf[low])), 3) if low.any() else None,
                     'cancel_frac_caught_DT': round(float(np.median(
                         cf[(y == 1) & (risk >= .5) & (fam == 'directed_transfer')])), 3),
                     'cancel_frac_negatives': round(float(np.median(cf[y == 0])), 3)})

        for defn, keep in (('A_labelled', lab), ('C_unlabelled', np.ones(len(y), bool))):
            b = grouped_ap(y, base, gi, 5, keep)
            c = grouped_ap(y, comb, gi, 5, keep)
            nd = [grouped_ap(y, n, gi, 5, keep) - b for n in nulls]
            r = {'fold': fold, 'defn': defn, 'pos': int(y[keep].sum()),
                 'AP_production': round(b, 5),
                 'AP_churn_alone': round(grouped_ap(y, Z[:, 0] - Z[:, 6], gi, 5, keep), 5),
                 'AP_production_plus_churn': round(c, 5), 'delta': round(c - b, 5),
                 'null_q95': round(float(np.quantile(nd, .95)), 5),
                 'beats_null': bool(c - b > np.quantile(nd, .95))}
            rows.append(r)
            print(r, flush=True)
        print(diag[-1], flush=True)

    print('\nPOOLED')
    for defn in ('A_labelled', 'C_unlabelled'):
        g = [r for r in rows if r['defn'] == defn]
        d = np.mean([x['delta'] for x in g])
        q = np.mean([x['null_q95'] for x in g])
        print(f'  {defn:14s} production {np.mean([x["AP_production"] for x in g]):.5f}'
              f' | churn alone {np.mean([x["AP_churn_alone"] for x in g]):.5f}'
              f' | +churn {np.mean([x["AP_production_plus_churn"] for x in g]):.5f}'
              f' | DELTA {d:+.5f} | shuffled null q95 {q:+.5f}'
              f' -> {"PASS" if d > q else "FAIL"}')
    json.dump({'per_fold': rows, 'cancel_frac_diagnostic': diag},
              open(OUT / 'churn281.json', 'w'), indent=2)
    print('wrote churn281.json')


if __name__ == '__main__':
    main()
