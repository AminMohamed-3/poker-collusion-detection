"""HF6: separating pair-specific coordination from player-general oddity, and controlling for
the selection artefact that hf4 could not rule out.

TWO PROBLEMS WITH hf3-hf5, BOTH ADDRESSED HERE.

(1) THE CANDIDATES MAY BE ODD PLAYERS, NOT COORDINATING PAIRS.  hf5 section B found that the
    players carrying the most candidate pairs sit at anomaly percentile .83-.90 in their OTHER
    pairs too, while players in LABELLED positive pairs sit at .546 -- ordinary -- with their
    non-partners.  Real coordination is a property of a PAIR; an unusual individual policy is a
    property of a PLAYER and contaminates all ~29 of that player's pairs.  The host lists
    exactly this as a decoy ("unusual but non-coordinated play patterns", tilt, weak play,
    strategy changes).  So the statistic that matters is not the pair's anomaly but its EXCESS
    over what its two players look like everywhere else:

        excess(a,b) = pct(a,b) - max( median pct of a's other pairs, median pct of b's other pairs )

    Coordination is pair-specific and must show a large positive excess.  A star hub shows ~0.
    This also gives a better screen: rank by excess, not by raw anomaly.

(2) THE "THIRD POPULATION" MAY BE A SELECTION ARTEFACT.  hf4 selected on (high unsupervised
    anomaly) AND (low supervised risk), then found the selected set far from the family
    centroids and separable from everything.  Both are partly guaranteed by that rule: the tail
    of ANY correlated 33-dimensional distribution consists of points far from the centre that
    share the leading eigendirection, so they look coherent and separable with no planted
    mechanism at all.  hf3's coherence null was drawn from CENTRAL confirmed negatives, which
    is the wrong comparison.  The control here draws a synthetic population with the same
    covariance and no planted families, applies the identical pipeline, and reports what the
    artefact alone produces.

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


def mean_pairwise_cos(Z):
    if len(Z) < 2:
        return float('nan')
    U = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    G = U @ U.T
    n = len(Z)
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))


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

    # ---------------- A. pair-specific excess ----------------
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

    print('=' * 96)
    print('A.  IS THE ANOMALY PAIR-SPECIFIC OR PLAYER-GENERAL?')
    print('=' * 96)
    print(f'{"set":<34}{"n":>6}{"med pct":>10}{"med player base":>18}{"med EXCESS":>13}')
    sets = {'labelled positives': pos,
            'confirmed non-targets': neg,
            'whole eligible population': np.ones(n, bool)}
    for q in [0.0050, 0.0100]:
        sets[f'hf4 candidates top-{q:.4f}'] = (pct >= 1 - q) & unl & (rk < 0.5)
    summ = {}
    for name, m in sets.items():
        summ[name] = {'n': int(m.sum()), 'median_pct': float(np.median(pct[m])),
                      'median_base': float(np.median(base[m])),
                      'median_excess': float(np.median(excess[m]))}
        print(f'{name:<34}{int(m.sum()):>6}{np.median(pct[m]):>10.4f}'
              f'{np.median(base[m]):>18.4f}{np.median(excess[m]):>13.4f}')
    for f in FAMS:
        m = fam == f
        summ[f] = {'n': int(m.sum()), 'median_pct': float(np.median(pct[m])),
                   'median_base': float(np.median(base[m])),
                   'median_excess': float(np.median(excess[m]))}
        print(f'{"  " + f:<34}{int(m.sum()):>6}{np.median(pct[m]):>10.4f}'
              f'{np.median(base[m]):>18.4f}{np.median(excess[m]):>13.4f}')
    print()
    print('  A labelled colluder is anomalous ONLY with the partner (low base, high excess).')
    print('  A pair driven by one odd player has a HIGH base and near-zero excess.')

    # ---------------- B. re-screen by excess ----------------
    print()
    print('=' * 96)
    print('B.  RE-SCREEN: rank by pair-specific EXCESS instead of raw anomaly')
    print('=' * 96)
    ex_thr = float(np.quantile(excess[pos], .25))
    print(f'  gate: excess >= the 25th percentile of labelled positives = {ex_thr:.4f}')
    print(f'{"top-q":>8}{"CAND(raw)":>11}{"survive excess":>16}{"coherence":>11}'
          f'{"AUC vs pos":>12}{"AUC vs neg":>12}')
    screen = []
    for q in [0.0050, 0.0100, 0.0200]:
        raw = (pct >= 1 - q) & unl & (rk < 0.5)
        ref = raw & (excess >= ex_thr)
        coh = mean_pairwise_cos(Z[ref]) if ref.sum() >= 2 else float('nan')
        a_pos = a_neg = float('nan')
        if ref.sum() >= 10:
            for other, key in [(pos, 'pos'), (neg, 'neg')]:
                sel = np.where(other | ref)[0]
                Xs, ys, gs = Z[sel], ref[sel].astype(int), tab[sel]
                oof = np.zeros(len(sel))
                for tr, te in GroupKFold(n_splits=5).split(Xs, ys, groups=gs):
                    mdl = lgb.LGBMClassifier(**PARAMS).fit(Xs[tr], ys[tr])
                    oof[te] = mdl.predict_proba(Xs[te])[:, 1]
                if key == 'pos':
                    a_pos = float(roc_auc_score(ys, oof))
                else:
                    a_neg = float(roc_auc_score(ys, oof))
        screen.append({'top_q': q, 'n_raw': int(raw.sum()), 'n_refined': int(ref.sum()),
                       'coherence': coh, 'auc_vs_positives': a_pos, 'auc_vs_negatives': a_neg})
        print(f'{q:>8.4f}{int(raw.sum()):>11}{int(ref.sum()):>16}{coh:>11.3f}'
              f'{a_pos:>12.4f}{a_neg:>12.4f}')

    # ---------------- C. selection-artefact control ----------------
    print()
    print('=' * 96)
    print('C.  CONTROL: what does the SAME pipeline produce on data with no planted family?')
    print('=' * 96)
    fit_on = ~pos                      # ordinary play + unknowns, families removed
    lw_fit = LedoitWolf().fit(Z[fit_on])
    mu, cov = Z[fit_on].mean(axis=0), lw_fit.covariance_
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[0]))
    Zs = mu + RNG.standard_normal((n, cov.shape[0])) @ L.T
    lw_s = LedoitWolf().fit(Zs)
    d2s = lw_s.mahalanobis(Zs)
    pcts = np.argsort(np.argsort(d2s, kind='mergesort'), kind='mergesort') / (n - 1.0)

    lw_r = LedoitWolf().fit(Z[neg])
    Pm = lw_r.get_precision()
    cent = {f: Z[pos][fam[pos] == f].mean(axis=0) for f in FAMS}

    def off(A):
        return np.stack([((A - cent[f]) @ Pm * (A - cent[f])).sum(1) for f in FAMS]).min(axis=0)

    print(f'{"band":>8}{"":>4}{"coherence":>12}{"med off-manifold":>19}{"AUC vs neg":>13}')
    ctrl = []
    for q in [0.0050, 0.0100]:
        sm = pcts >= 1 - q
        real = (pct >= 1 - q) & unl & (rk < 0.5)
        coh_s = mean_pairwise_cos(Zs[sm])
        off_s = float(np.median(off(Zs[sm])))
        sel = np.concatenate([np.where(neg)[0]])
        Xs = np.vstack([Z[sel], Zs[sm]])
        ys = np.r_[np.zeros(len(sel)), np.ones(int(sm.sum()))]
        gs = np.r_[tab[sel], np.array(['synth'] * int(sm.sum()))]
        oof = np.zeros(len(ys))
        for tr, te in GroupKFold(n_splits=5).split(Xs, ys, groups=gs):
            mdl = lgb.LGBMClassifier(**PARAMS).fit(Xs[tr], ys[tr])
            oof[te] = mdl.predict_proba(Xs[te])[:, 1]
        auc_s = float(roc_auc_score(ys, oof))
        coh_r = mean_pairwise_cos(Z[real])
        off_r = float(np.median(off(Z[real])))
        ctrl.append({'top_q': q, 'synthetic': {'coherence': coh_s, 'median_off': off_s,
                                               'auc_vs_negatives': auc_s},
                     'observed': {'coherence': coh_r, 'median_off': off_r}})
        print(f'{q:>8.4f}  SYNTH{coh_s:>12.3f}{off_s:>19.1f}{auc_s:>13.4f}')
        print(f'{q:>8.4f}  REAL {coh_r:>12.3f}{off_r:>19.1f}{"":>13}')
    print()
    print('  If SYNTH reproduces REAL, the "third population" of hf4 is the tail of a correlated')
    print('  distribution under a selection rule, not a planted mechanism.')

    write_json(OUT / 'hf6-pair-specific-excess.json',
               {'excess_summary': summ, 'excess_gate': ex_thr, 'screen': screen,
                'control': ctrl})
    print()
    print(f'written {OUT / "hf6-pair-specific-excess.json"}')


if __name__ == '__main__':
    main()
