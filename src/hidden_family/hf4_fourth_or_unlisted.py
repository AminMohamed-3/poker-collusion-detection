"""HF4: are the coherent high-anomaly unlabelled pairs a FOURTH mechanism, or unlisted
members of the three disclosed ones?

hf3 established that the top .25 % of the label-free ca_ anomaly, after removing labelled
pairs and pairs the supervised model already likes, is a coherent population (mean pairwise
cosine .419 against a permutation null of .036).  Coherence alone does not make it new: raw
cosine to the disclosed centroids came out .74-.78, but the disclosed families already sit at
.37-.76 cosine to EACH OTHER, so z-space is dominated by one shared "some coordination is
happening" direction and raw cosine cannot answer the question.  This script replaces it with
three calibrated tests.

T1  OFF-MANIFOLD DISTANCE.  Whiten z-space on the confirmed non-targets, then for each pair
    take the smallest Mahalanobis distance to a disclosed-family centroid.  Centroids and
    whitening are fitted OUT OF FOLD (GroupKFold by table), so held-out labelled positives
    give the exact null distribution of "a disclosed-family pair's distance to its own
    family".  A fourth mechanism must sit in the upper tail of that null.

T2  MIXTURE DECOMPOSITION.  Fit the candidates' T1 distribution as a two-component mixture of
    the held-out-positive distribution and the confirmed-negative distribution.  If the fit is
    adequate, the candidates are fully explained as (unlisted disclosed positives + ordinary
    play) and there is no room for a third component.  The residual mass IS the hidden-family
    estimate, with a bootstrap interval.

T3  ADVERSARIAL SEPARABILITY.  Train a classifier to tell candidates from labelled positives,
    out of fold by table.  If the candidates are the same three families, this is chance.  If
    a fourth mechanism is mixed in, AUC rises and the top-scoring candidates are it -- so this
    both tests for and localises the population.

Also completes hf3's pre-registered gates G3 (separability vs confirmed negatives) and G4
(table/player concentration against a permutation null), and runs a k-means sub-structure
search so that, if something IS there, we can describe what it looks like.

Read-only.  Consumes hf3's frozen outputs.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

from evidence_aug.paths import CV, write_json

OUT = CV / 'experiments' / 'hidden_family_20260912'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
TOP_QS = [0.0025, 0.0050]   # both coherent bands; .0025 holds only 13 pairs after dedup
RNG = np.random.default_rng(20260912)
PARAMS = dict(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=20,
              subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mean_pairwise_cos(Z):
    if len(Z) < 2:
        return float('nan')
    U = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    G = U @ U.T
    n = len(Z)
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))


def run(TOP_Q):
    fr = pl.read_parquet(OUT / 'hf3-dev-eligible.parquet')
    Z = np.load(OUT / 'hf3-dev-z.npy').astype(np.float64)
    y = fr['label'].to_numpy()
    fam = fr['behavior_family'].to_numpy()
    tab = fr['table_id'].to_numpy()
    pct = fr['anom_pct'].to_numpy()
    risk_known = fr['risk_known'].to_numpy()
    pos, neg, unl = y == 1, y == 0, y == -1
    cand = (pct >= 1 - TOP_Q) & unl & (risk_known < 0.5)
    print(f'eligible {len(fr)}  positives {pos.sum()}  negatives {neg.sum()}  '
          f'candidates {cand.sum()}')

    # ---------------- T1: out-of-fold off-manifold distance ----------------
    print()
    print('=' * 92)
    print('T1  OFF-MANIFOLD DISTANCE TO THE NEAREST DISCLOSED-FAMILY CENTROID (out of fold)')
    print('=' * 92)
    d_off = np.full(len(fr), np.nan)
    gkf = GroupKFold(n_splits=5)
    pos_idx = np.where(pos)[0]
    for tr, te in gkf.split(pos_idx, groups=tab[pos_idx]):
        tr_i, te_i = pos_idx[tr], pos_idx[te]
        lw = LedoitWolf().fit(Z[neg])            # whiten on confirmed ordinary play
        cent = {f: Z[tr_i][fam[tr_i] == f].mean(axis=0) for f in FAMS}
        P = lw.get_precision()
        for i in te_i:
            d_off[i] = min(float((Z[i] - cent[f]) @ P @ (Z[i] - cent[f])) for f in FAMS)
    # non-positive rows: centroids from ALL labelled positives (they are out of sample anyway)
    lw = LedoitWolf().fit(Z[neg])
    P = lw.get_precision()
    cent_all = {f: Z[pos][fam[pos] == f].mean(axis=0) for f in FAMS}
    rest = np.where(~pos)[0]
    D = np.stack([((Z[rest] - cent_all[f]) @ P * (Z[rest] - cent_all[f])).sum(1) for f in FAMS])
    d_off[rest] = D.min(axis=0)
    nearest = np.array(FAMS)[D.argmin(axis=0)]

    def qs(a, label):
        v = np.quantile(a, [.05, .25, .50, .75, .95])
        print(f'  {label:<34} n={len(a):>6}  ' +
              '  '.join(f'{p}={x:.1f}' for p, x in zip(['q05', 'q25', 'med', 'q75', 'q95'], v)))
        return [float(x) for x in v]

    q_pos = qs(d_off[pos], 'held-out labelled positives')
    q_neg = qs(d_off[neg], 'confirmed non-targets')
    q_cand = qs(d_off[cand], 'CANDIDATES')
    # what fraction of candidates lies beyond the 95th percentile of the positive null?
    thr95 = np.quantile(d_off[pos], .95)
    frac_beyond = float((d_off[cand] > thr95).mean())
    exp_beyond = 0.05
    print()
    print(f'  positive-null 95th percentile = {thr95:.1f}')
    print(f'  candidates beyond it: {frac_beyond:.3f} (expected .050 if they are disclosed-family)')
    print(f'  => excess fraction {frac_beyond - exp_beyond:+.3f}  '
          f'~ {(frac_beyond - exp_beyond) * cand.sum():+.1f} pairs of {cand.sum()}')

    # ---------------- T2: two-component mixture ----------------
    print()
    print('=' * 92)
    print('T2  MIXTURE DECOMPOSITION of the candidate off-manifold distribution')
    print('=' * 92)
    edges = np.quantile(np.concatenate([d_off[pos], d_off[neg]]), np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf

    def hist(a):
        h = np.histogram(a, bins=edges)[0].astype(float)
        return h / h.sum()

    h_pos, h_neg, h_cand = hist(d_off[pos]), hist(d_off[neg]), hist(d_off[cand])
    ws = np.linspace(0, 1, 201)
    err = [np.abs(w * h_pos + (1 - w) * h_neg - h_cand).sum() for w in ws]
    w_hat = float(ws[int(np.argmin(err))])
    resid = w_hat * h_pos + (1 - w_hat) * h_neg - h_cand
    print(f'  best fit: {w_hat:.2f} x (disclosed positive) + {1 - w_hat:.2f} x (ordinary play)')
    print(f'  total absolute residual (L1) = {min(err):.4f}   '
          f'max |residual| in any decile = {np.abs(resid).max():.4f}')
    boot = []
    ci = np.where(cand)[0]
    for _ in range(300):
        b = RNG.choice(ci, size=len(ci), replace=True)
        hb = hist(d_off[b])
        e = [np.abs(w * h_pos + (1 - w) * h_neg - hb).sum() for w in ws]
        boot.append(ws[int(np.argmin(e))])
    lo, hi = np.quantile(boot, [.025, .975])
    print(f'  bootstrap 95% CI on the disclosed-positive share: [{lo:.2f}, {hi:.2f}]')
    print(f'  unexplained (third-component) mass required: {max(0.0, min(err) / 2):.4f} '
          f'~ {max(0.0, min(err) / 2) * cand.sum():.1f} of {cand.sum()} candidates')

    # ---------------- T3: adversarial separability vs labelled positives ----------------
    print()
    print('=' * 92)
    print('T3  CAN A MODEL TELL CANDIDATES FROM LABELLED POSITIVES?  (out of fold by table)')
    print('=' * 92)
    sel = np.where(pos | cand)[0]
    Xs, ys, gs = Z[sel], cand[sel].astype(int), tab[sel]
    oof = np.zeros(len(sel))
    for tr, te in GroupKFold(n_splits=5).split(Xs, ys, groups=gs):
        m = lgb.LGBMClassifier(**PARAMS).fit(Xs[tr], ys[tr])
        oof[te] = m.predict_proba(Xs[te])[:, 1]
    auc_t3 = float(roc_auc_score(ys, oof))
    print(f'  out-of-fold AUC candidates vs labelled positives = {auc_t3:.4f}')
    print('  (.50 = indistinguishable, i.e. the candidates ARE disclosed-family pairs)')

    # G3: candidates vs confirmed negatives
    sel2 = np.where(neg | cand)[0]
    X2, y2, g2 = Z[sel2], cand[sel2].astype(int), tab[sel2]
    oof2 = np.zeros(len(sel2))
    for tr, te in GroupKFold(n_splits=5).split(X2, y2, groups=g2):
        m = lgb.LGBMClassifier(**PARAMS).fit(X2[tr], y2[tr])
        oof2[te] = m.predict_proba(X2[te])[:, 1]
    auc_g3 = float(roc_auc_score(y2, oof2))
    print(f'  [G3] out-of-fold AUC candidates vs confirmed non-targets = {auc_g3:.4f}')

    # ---------------- G4: table / player concentration ----------------
    print()
    print('=' * 92)
    print('G4  DO CANDIDATES CLUSTER IN TABLES AND PLAYERS?')
    print('=' * 92)
    ctab = tab[cand]
    p1, p2 = fr['player_1'].to_numpy()[cand], fr['player_2'].to_numpy()[cand]
    n_tables = len(set(ctab))
    top_tab = max(np.bincount(np.unique(ctab, return_inverse=True)[1]))
    pl_counts = {}
    for a, b in zip(p1, p2):
        pl_counts[a] = pl_counts.get(a, 0) + 1
        pl_counts[b] = pl_counts.get(b, 0) + 1
    top_pl = max(pl_counts.values())
    all_tab = np.unique(tab)
    null_tab, null_pl = [], []
    idx_all = np.arange(len(fr))
    for _ in range(400):
        pick = RNG.choice(idx_all, size=int(cand.sum()), replace=False)
        null_tab.append(len(set(tab[pick])))
        c = {}
        for a, b in zip(fr['player_1'].to_numpy()[pick], fr['player_2'].to_numpy()[pick]):
            c[a] = c.get(a, 0) + 1
            c[b] = c.get(b, 0) + 1
        null_pl.append(max(c.values()))
    print(f'  {int(cand.sum())} candidates span {n_tables} tables '
          f'(random null {np.mean(null_tab):.1f} +- {np.std(null_tab):.1f}); '
          f'busiest table holds {top_tab}')
    print(f'  busiest player appears in {top_pl} candidate pairs '
          f'(random null {np.mean(null_pl):.1f} +- {np.std(null_pl):.1f})')
    # same statistics for labelled positives, as the reference for a real planted mechanism
    ptab = tab[pos]
    pc = {}
    for a, b in zip(fr['player_1'].to_numpy()[pos], fr['player_2'].to_numpy()[pos]):
        pc[a] = pc.get(a, 0) + 1
        pc[b] = pc.get(b, 0) + 1
    print(f'  reference: {int(pos.sum())} labelled positives span {len(set(ptab))} tables, '
          f'busiest player in {max(pc.values())} pairs')

    # ---------------- sub-structure: is there a distinct cluster? ----------------
    print()
    print('=' * 92)
    print('SUB-STRUCTURE  (k-means on the candidates; is any cluster far from all three?)')
    print('=' * 92)
    sub = []
    Zc = Z[cand]
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Zc)
        for c in range(k):
            m = km.labels_ == c
            if m.sum() < 3:
                continue
            cc = Zc[m].mean(axis=0)
            cf = {f: cos(cc, cent_all[f]) for f in FAMS}
            d = min(float((cc - cent_all[f]) @ P @ (cc - cent_all[f])) for f in FAMS)
            row = {'k': k, 'cluster': c, 'n': int(m.sum()),
                   'coherence': mean_pairwise_cos(Zc[m]),
                   'max_cos_to_family': max(cf.values()), 'off_manifold': d,
                   'nearest': max(cf, key=cf.get)}
            sub.append(row)
            print(f'  k={k} c={c}  n={int(m.sum()):>3}  coherence={row["coherence"]:.3f}  '
                  f'max cos to a disclosed family={row["max_cos_to_family"]:.3f} '
                  f'({row["nearest"]})  off-manifold d2={d:.1f}')

    nearest_hist = {f: int((nearest == f).sum()) for f in FAMS}
    cand_nearest = nearest[np.isin(rest, np.where(cand)[0])]
    print()
    print('  nearest disclosed family of the candidates: ' +
          str({f: int((cand_nearest == f).sum()) for f in FAMS}))

    return ({'top_q': TOP_Q, 'n_candidates': int(cand.sum()),
                'T1': {'q_positives': q_pos, 'q_negatives': q_neg, 'q_candidates': q_cand,
                       'pos_null_p95': float(thr95), 'frac_candidates_beyond': frac_beyond,
                       'excess_fraction': frac_beyond - exp_beyond,
                       'excess_pairs': float((frac_beyond - exp_beyond) * cand.sum())},
                'T2': {'disclosed_positive_share': w_hat, 'ci': [float(lo), float(hi)],
                       'l1_residual': float(min(err)),
                       'unexplained_pairs': float(max(0.0, min(err) / 2) * cand.sum())},
                'T3': {'auc_vs_labelled_positives': auc_t3, 'auc_vs_confirmed_negatives': auc_g3},
                'G4': {'n_tables': int(n_tables), 'null_tables_mean': float(np.mean(null_tab)),
                       'busiest_table': int(top_tab), 'busiest_player': int(top_pl),
                       'null_busiest_player_mean': float(np.mean(null_pl))},
                'substructure': sub,
                'candidate_nearest_family': {f: int((cand_nearest == f).sum()) for f in FAMS},
                'population_nearest_family': nearest_hist})


def main():
    out = {}
    for q in TOP_QS:
        print()
        print('#' * 96)
        print(f'#  BAND: top {q:.4%} of the label-free anomaly')
        print('#' * 96)
        out[f'{q:.4f}'] = run(q)
    write_json(OUT / 'hf4-fourth-or-unlisted.json', out)
    print()
    print(f'written {OUT / "hf4-fourth-or-unlisted.json"}')


if __name__ == '__main__':
    main()
