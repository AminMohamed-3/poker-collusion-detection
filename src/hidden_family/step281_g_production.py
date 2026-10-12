"""STEP281 stage G: do the victim features add anything to the PRODUCTION pair model?

Amin's question after 19.234: is this instrument useful in production, or only for research?

The precedent to beat is 19.227(e): 38 out-of-fold discriminator columns, every one of them
carrying real signal (standalone def-A AP .840-.913 against .419-.619 for 304 label-permuted
null columns), and EVERY aggregation of them negative.  The reading there was that they were
re-measurements of information the existing views already hold, so they added collinearity the
fit paid for.  So the question is not "does the victim signal exist" -- 19.234 settled that --
but "is it ORTHOGONAL to what production already has, and does it rank pairs production misses".

Harness: the three prefix-60 fold frames in `experiments/pu_pairs_20260911/fold{0,1,2}-prefix60`.
Each is ~43,000 development pairs from 100 held-out pools, scored by the pair model on the FIRST
1,800 of each table's 3,000 hands (identity.json: hands_kept 180,000 / hands_dev 300,000), with
labels attached.  The victim features are rebuilt on exactly that prefix so the two views see
the same history.

Populations, both of the project's definitions:
  def A  positives against the LABELLED negatives          (clean, tiny, saturated)
  def C  positives against the UNLABELLED population       (what evaluation actually scores)
Eligibility is the prefix floor of 38 shared hands, the evaluation pair list's own floor.
"""
import json
import os

os.environ['POLARS_MAX_THREADS'] = '6'
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
PU = ROOT / 'Claude Version' / 'experiments' / 'pu_pairs_20260911'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step281_c_discover import CELLS, Scorer, strat  # noqa: E402

VCOLS = ['v_fold_excess', 'v_foldany_excess', 'v_bleed_excess', 'v_take_excess',
         'v_pressure_excess', 'v_squeeze', 'v_bothaggr', 'v_conc', 'v_victim_take',
         'v_jointnet_z', 'v_conflict']


def ap(y, s):
    o = np.argsort(-s)
    y = y[o]
    c = np.cumsum(y)
    prec = c / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def main():
    vic = pl.read_parquet(OUT / 'victim_dev_prefix60.parquet')
    ref = vic.filter(pl.col('n_shared') >= 38)
    sc = Scorer(ref, CELLS)

    def zmat(df):
        s = strat(df['n_B'].to_numpy(), sc.edges)
        z = np.zeros((df.height, len(CELLS)))
        for j, (name, sign) in enumerate(CELLS):
            v = df[name].to_numpy() * sign
            for k, r in enumerate(sc.ref[(name, sign)]):
                m = s == k
                if not m.any() or len(r) < 50:
                    continue
                p = np.searchsorted(r, v[m], side='left') / len(r)
                z[m, j] = ndtri(np.clip(p, 1e-4, 1 - 1e-4))
            z[:, j] = np.nan_to_num(z[:, j])
        return z

    known = pl.read_csv(ROOT / 'data' / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p1'),
        pl.max_horizontal('player_1', 'player_2').alias('p2')).select('p1', 'p2', 'label_status')

    def fit_oof(X, y, gi, n_groups, C=0.05):
        """Held-out scores, kept INSIDE their group.  Pooling decision_function values from
        separately fitted models mixes incompatible scales and corrupts the global ranking
        (the first run of this stage lost .006-.035 AP on a pure monotone refit that way), so
        every metric below is computed inside a group and averaged."""
        out = np.full(len(y), np.nan)
        Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        for g in range(n_groups):
            tr, te = gi != g, gi == g
            if y[tr].sum() < 3 or not te.any():
                continue
            m = LogisticRegression(C=C, max_iter=3000).fit(Xs[tr], y[tr])
            out[te] = m.decision_function(Xs[te])
        return out

    def grouped_ap(y, s, gi, n_groups, keep):
        """AP inside each held-out group, averaged with weight = positives in the group."""
        num = den = 0.0
        for g in range(n_groups):
            m = (gi == g) & keep & ~np.isnan(s)
            if y[m].sum() < 1:
                continue
            num += ap(y[m], s[m]) * y[m].sum()
            den += y[m].sum()
        return float(num / max(den, 1))

    rows, missed = [], []
    for fold in (0, 1, 2):
        d = PU / f'fold{fold}-prefix60'
        s = pl.read_parquet(d / 'scores.parquet').with_columns(
            pl.min_horizontal('player_1', 'player_2').alias('p1'),
            pl.max_horizontal('player_1', 'player_2').alias('p2'))
        j = (s.join(vic.drop('table_id'), on=['p1', 'p2'], how='inner')
              .join(known, on=['p1', 'p2'], how='left'))
        j = j.filter(pl.col('n_shared') >= 38)      # the evaluation list's own floor
        y = (j['label'].to_numpy() == 1).astype(int)
        status = j['label_status'].fill_null('unlabelled').to_numpy()
        lab = status != 'unlabelled'                # positives + CONFIRMED non-targets only
        risk = j['risk'].to_numpy()
        Z = zmat(j)
        vs = np.maximum(Z, 0).sum(axis=1)           # the 19.234 composite
        lr = np.log(np.clip(risk, 1e-9, 1 - 1e-9) / np.clip(1 - risk, 1e-9, 1))

        tab = j['table_id'].to_numpy()
        rngs = np.random.default_rng(3)
        grp = {t: i % 5 for i, t in enumerate(rngs.permutation(np.unique(tab)))}
        gi = np.fromiter((grp[t] for t in tab), int, len(tab))
        base = fit_oof(lr[:, None], y, gi, 5)
        comb = fit_oof(np.column_stack([lr, Z]), y, gi, 5)

        # null: the same 11 columns, rows shuffled, so "adding columns helps a logistic by
        # construction" is priced in rather than assumed away (19.227(e)'s protocol)
        nulls = []
        rn = np.random.default_rng(fold + 100)
        for _ in range(20):
            Zp = Z[rn.permutation(len(Z))]
            nulls.append(fit_oof(np.column_stack([lr, Zp]), y, gi, 5))

        for defn, keep in (('A_labelled', lab), ('C_unlabelled', np.ones(len(y), bool))):
            b = grouped_ap(y, base, gi, 5, keep)
            c = grouped_ap(y, comb, gi, 5, keep)
            nd = [grouped_ap(y, n, gi, 5, keep) - b for n in nulls]
            r = {'fold': fold, 'defn': defn, 'n': int(keep.sum()), 'pos': int(y[keep].sum()),
                 'AP_production_raw': round(grouped_ap(y, risk, gi, 5, keep), 5),
                 'AP_victim_alone': round(grouped_ap(y, vs, gi, 5, keep), 5),
                 'AP_production_refit': round(b, 5),
                 'AP_production_plus_victim': round(c, 5),
                 'delta': round(c - b, 5),
                 'null_delta_mean': round(float(np.mean(nd)), 5),
                 'null_delta_q95': round(float(np.quantile(nd, .95)), 5)}
            rows.append(r)
            print(r, flush=True)

        # --- the decision-relevant question: the positives production RANKS LOW -------
        rank_r = np.argsort(np.argsort(-risk)) + 1
        rank_v = np.argsort(np.argsort(-vs)) + 1
        low = (y == 1) & (risk < 0.5)
        missed.append({'fold': fold, 'positives': int(y.sum()),
                       'production_misses_risk_lt_.5': int(low.sum()),
                       'their_production_rank_median': int(np.median(rank_r[low])) if low.any() else None,
                       'their_victim_rank_median': int(np.median(rank_v[low])) if low.any() else None,
                       'families': {f: int(((j['behavior_family'].to_numpy() == f) & low).sum())
                                    for f in ('directed_transfer', 'soft_play',
                                              'coordinated_isolation')},
                       'n_rows': len(y)})
        print(missed[-1], flush=True)

    print('\nPOOLED')
    for defn in ('A_labelled', 'C_unlabelled'):
        g = [r for r in rows if r['defn'] == defn]
        print(f'  {defn:14s} production {np.mean([x["AP_production_refit"] for x in g]):.5f}'
              f' | victim alone {np.mean([x["AP_victim_alone"] for x in g]):.5f}'
              f' | +victim {np.mean([x["AP_production_plus_victim"] for x in g]):.5f}'
              f' | DELTA {np.mean([x["delta"] for x in g]):+.5f}'
              f' | shuffled-column null {np.mean([x["null_delta_mean"] for x in g]):+.5f}'
              f' q95 {np.mean([x["null_delta_q95"] for x in g]):+.5f}')
    json.dump({'per_fold': rows, 'missed_positives': missed},
              open(OUT / 'production281.json', 'w'), indent=2)
    print('wrote production281.json')


if __name__ == '__main__':
    main()
