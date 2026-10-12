"""HF7: with the odd-player decoy removed, how many coordinated pairs are there in development,
and can the three disclosed families account for all of them?

hf6 gave a statistic that isolates coordination from individual oddity:

    excess(a,b) = anomaly percentile(a,b) - max(median anomaly pct of a's other pairs,
                                                median anomaly pct of b's other pairs)

Labelled colluders score .358; the population scores -.076; the pairs hf4 flagged score .136
because their players are anomalous everywhere.  Excess is therefore the right screen, and it
makes the existence question answerable in its sharpest form:

  A  How many UNLABELLED pairs are coordination-shaped (high anomaly AND high excess, both
     gated at the 25th percentile of labelled positives)?  Split them by whether the supervised
     production model already recognises them.  The ones it does not are where a fourth family
     would have to live.
  B  Two-component mixture on the excess statistic to estimate the total coordinated population
     of the development period, against the 164 pairs actually labelled here.
  C  THE TEST THAT MATTERS.  Train a 3-class family model on labelled positives, out of fold by
     table.  Held-out positives give the null distribution of "how confidently does a genuine
     disclosed-family pair get assigned to its family".  Apply the same model to the
     coordination-shaped unlabelled pairs.  If they are all disclosed-family, their confidence
     distribution matches the null.  An excess of LOW-confidence assignments is the signature of
     a mechanism the three-class model has never seen -- which is exactly what
     `other_coordination` is.  The excess mass, with a bootstrap interval, is the estimate.
  D  Repairs hf6's control AUC, which was degenerate because every synthetic row shared one
     GroupKFold group.

Read-only.  Consumes hf3's frozen outputs.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

from evidence_aug.paths import CV, write_json

OUT = CV / 'experiments' / 'hidden_family_20260912'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
RNG = np.random.default_rng(20260912)
PARAMS = dict(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=20,
              subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1)


def main():
    fr = pl.read_parquet(OUT / 'hf3-dev-eligible.parquet')
    Z = np.load(OUT / 'hf3-dev-z.npy').astype(np.float64)
    y = fr['label'].to_numpy()
    fam = fr['behavior_family'].to_numpy()
    tab = fr['table_id'].to_numpy()
    pct = fr['anom_pct'].to_numpy()
    rk = fr['risk_known'].to_numpy()
    P1, P2 = fr['player_1'].to_numpy(), fr['player_2'].to_numpy()
    pos, neg, unl = y == 1, y == 0, y == -1
    n = len(fr)

    rows = {}
    for i in range(n):
        rows.setdefault(P1[i], []).append(i)
        rows.setdefault(P2[i], []).append(i)
    rows = {p: np.array(v) for p, v in rows.items()}
    base = np.zeros(n)
    for i in range(n):
        b = []
        for p in (P1[i], P2[i]):
            r = rows[p]
            r = r[r != i]
            b.append(np.median(pct[r]) if len(r) else 0.0)
        base[i] = max(b)
    excess = pct - base

    g_pct = float(np.quantile(pct[pos], .25))
    g_exc = float(np.quantile(excess[pos], .25))
    shaped = (pct >= g_pct) & (excess >= g_exc)
    print('=' * 96)
    print('A.  COORDINATION-SHAPED PAIRS  (anomaly pct >= %.4f AND excess >= %.4f;'
          % (g_pct, g_exc))
    print('     both gates are the 25th percentile of labelled positives, so 75% of a real')
    print('     family passes both by construction)')
    print('=' * 96)
    print(f'  labelled positives passing both gates : '
          f'{int((shaped & pos).sum())} of {int(pos.sum())} '
          f'({(shaped & pos).mean() / pos.mean():.1%})')
    print(f'  confirmed non-targets passing         : {int((shaped & neg).sum())} '
          f'of {int(neg.sum())}')
    print(f'  UNLABELLED passing                    : {int((shaped & unl).sum())} '
          f'of {int(unl.sum())}')
    su = shaped & unl
    print()
    print('  of those unlabelled coordination-shaped pairs, by supervised production risk:')
    for lo, hi in [(0.9, 1.01), (0.5, 0.9), (0.1, 0.5), (0.0, 0.1)]:
        m = su & (rk >= lo) & (rk < hi)
        print(f'    risk in [{lo:.1f}, {hi:.2f}) : {int(m.sum()):>4}')
    unrec = su & (rk < 0.5)
    print(f'  => {int(unrec.sum())} coordination-shaped pairs the supervised model does NOT '
          f'recognise')

    # ---------------- B. mixture estimate of the coordinated population ----------------
    print()
    print('=' * 96)
    print('B.  HOW MANY COORDINATED PAIRS ARE IN THE DEVELOPMENT PERIOD ALTOGETHER?')
    print('=' * 96)
    edges = np.quantile(excess[neg | pos], np.linspace(0, 1, 13))
    edges[0], edges[-1] = -np.inf, np.inf

    def hist(a):
        h = np.histogram(a, bins=edges)[0].astype(float)
        return h / max(h.sum(), 1)

    h_pos, h_neg, h_unl = hist(excess[pos]), hist(excess[neg]), hist(excess[unl])
    ws = np.linspace(0, .2, 401)
    err = [np.abs(w * h_pos + (1 - w) * h_neg - h_unl).sum() for w in ws]
    a_hat = float(ws[int(np.argmin(err))])
    boot = []
    ui = np.where(unl)[0]
    for _ in range(200):
        b = RNG.choice(ui, size=len(ui), replace=True)
        hb = hist(excess[b])
        boot.append(ws[int(np.argmin([np.abs(w * h_pos + (1 - w) * h_neg - hb).sum()
                                      for w in ws]))])
    lo, hi = np.quantile(boot, [.025, .975])
    print(f'  unlabelled = {a_hat:.4f} x positive-shaped + {1 - a_hat:.4f} x ordinary')
    print(f'  => {a_hat * unl.sum():.0f} coordinated pairs among {int(unl.sum())} unlabelled '
          f'[95% CI {lo * unl.sum():.0f} - {hi * unl.sum():.0f}]')
    print(f'  against {int(pos.sum())} labelled positives in the same {len(set(tab))} tables')
    print(f'  (the host labelled a SAMPLE; L1 residual of the fit = {min(err):.4f})')

    # ---------------- C. family-assignment confidence ----------------
    print()
    print('=' * 96)
    print('C.  DO THE UNRECOGNISED PAIRS LOOK LIKE THE THREE DISCLOSED FAMILIES?')
    print('=' * 96)
    pos_idx = np.where(pos)[0]
    ycls = np.array([FAMS.index(f) for f in fam[pos_idx]])
    conf_pos = np.zeros(len(pos_idx))
    conf_tgt = np.zeros((int(unrec.sum()), 3))
    folds = list(GroupKFold(n_splits=5).split(pos_idx, ycls, groups=tab[pos_idx]))
    for tr, te in folds:
        m = lgb.LGBMClassifier(objective='multiclass', num_class=3, **PARAMS).fit(
            Z[pos_idx[tr]], ycls[tr])
        conf_pos[te] = m.predict_proba(Z[pos_idx[te]]).max(axis=1)
        conf_tgt += m.predict_proba(Z[unrec]) / len(folds)
    conf_t = conf_tgt.max(axis=1)
    conf_neg = np.zeros(int(neg.sum()))
    for tr, te in folds:
        m = lgb.LGBMClassifier(objective='multiclass', num_class=3, **PARAMS).fit(
            Z[pos_idx[tr]], ycls[tr])
        conf_neg += m.predict_proba(Z[neg]).max(axis=1) / len(folds)

    def qq(a, lab):
        v = np.quantile(a, [.10, .25, .50, .75])
        print(f'  {lab:<40} n={len(a):>5}  q10={v[0]:.3f} q25={v[1]:.3f} '
              f'med={v[2]:.3f} q75={v[3]:.3f}')
        return [float(x) for x in v]

    q_p = qq(conf_pos, 'held-out labelled positives (the null)')
    q_t = qq(conf_t, 'unrecognised coordination-shaped pairs')
    q_n = qq(conf_neg, 'confirmed non-targets')
    thr = float(np.quantile(conf_pos, .10))
    frac = float((conf_t < thr).mean())
    print()
    print(f'  10th-percentile confidence of a genuine disclosed-family pair = {thr:.3f}')
    print(f'  share of unrecognised pairs below it = {frac:.3f} (expected .100 if all three-family)')
    ex_share = frac - 0.10
    print(f'  => excess low-confidence share {ex_share:+.3f} '
          f'~ {ex_share * unrec.sum():+.1f} pairs of {int(unrec.sum())}')
    bs = []
    for _ in range(400):
        b = RNG.choice(len(conf_t), size=len(conf_t), replace=True)
        bs.append((conf_t[b] < thr).mean() - 0.10)
    blo, bhi = np.quantile(bs, [.025, .975])
    print(f'  bootstrap 95% CI on the excess share: [{blo:+.3f}, {bhi:+.3f}] '
          f'-> [{blo * unrec.sum():+.0f}, {bhi * unrec.sum():+.0f}] pairs')

    # ---------------- D. repaired control ----------------
    print()
    print('=' * 96)
    print('D.  REPAIRED SELECTION-ARTEFACT CONTROL (synthetic rows get real fold groups)')
    print('=' * 96)
    lw_fit = LedoitWolf().fit(Z[~pos])
    mu, cov = Z[~pos].mean(axis=0), lw_fit.covariance_
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[0]))
    Zs = mu + RNG.standard_normal((n, cov.shape[0])) @ L.T
    d2s = LedoitWolf().fit(Zs).mahalanobis(Zs)
    pcts = np.argsort(np.argsort(d2s, kind='mergesort'), kind='mergesort') / (n - 1.0)
    ctrl = []
    for q in [0.0050, 0.0100]:
        sm = np.where(pcts >= 1 - q)[0]
        negi = np.where(neg)[0]
        Xs = np.vstack([Z[negi], Zs[sm]])
        ys = np.r_[np.zeros(len(negi)), np.ones(len(sm))]
        gs = np.r_[tab[negi], RNG.choice(np.unique(tab), size=len(sm))]
        oof = np.zeros(len(ys))
        for tr, te in GroupKFold(n_splits=5).split(Xs, ys, groups=gs):
            mdl = lgb.LGBMClassifier(**PARAMS).fit(Xs[tr], ys[tr])
            oof[te] = mdl.predict_proba(Xs[te])[:, 1]
        auc_s = float(roc_auc_score(ys, oof))
        real = np.where((pct >= 1 - q) & unl & (rk < 0.5))[0]
        Xr = np.vstack([Z[negi], Z[real]])
        yr = np.r_[np.zeros(len(negi)), np.ones(len(real))]
        gr = np.r_[tab[negi], tab[real]]
        oofr = np.zeros(len(yr))
        for tr, te in GroupKFold(n_splits=5).split(Xr, yr, groups=gr):
            mdl = lgb.LGBMClassifier(**PARAMS).fit(Xr[tr], yr[tr])
            oofr[te] = mdl.predict_proba(Xr[te])[:, 1]
        auc_r = float(roc_auc_score(yr, oofr))
        ctrl.append({'top_q': q, 'auc_synth_vs_neg': auc_s, 'auc_real_vs_neg': auc_r})
        print(f'  band {q:.4f}: AUC vs confirmed non-targets  SYNTHETIC {auc_s:.4f}   '
              f'REAL {auc_r:.4f}')
    print()
    print('  A structureless Gaussian tail is already highly separable from the population')
    print('  centre, so separability alone was never evidence of a planted mechanism.')

    write_json(OUT / 'hf7-excess-mass.json',
               {'gates': {'pct': g_pct, 'excess': g_exc},
                'shaped': {'positives': int((shaped & pos).sum()),
                           'negatives': int((shaped & neg).sum()),
                           'unlabelled': int(su.sum()),
                           'unrecognised': int(unrec.sum())},
                'mixture': {'alpha': a_hat, 'n_coordinated_unlabelled': float(a_hat * unl.sum()),
                            'ci': [float(lo * unl.sum()), float(hi * unl.sum())],
                            'l1_residual': float(min(err))},
                'confidence': {'q_positives': q_p, 'q_target': q_t, 'q_negatives': q_n,
                               'threshold': thr, 'frac_below': frac,
                               'excess_share': ex_share,
                               'excess_pairs': float(ex_share * unrec.sum()),
                               'ci_share': [float(blo), float(bhi)],
                               'ci_pairs': [float(blo * unrec.sum()), float(bhi * unrec.sum())]},
                'control': ctrl})
    print()
    print(f'written {OUT / "hf7-excess-mass.json"}')


if __name__ == '__main__':
    main()
