"""Step 73 (plan P3): label-free profile of the evaluation block -- how many of the 909 pairs that the
production risk puts above .5 are plausibly NOT positives, and is there anything in there that the
development testbed does not contain?

Why it still matters after 18.8.  18.8 showed that no view we own re-ranks the eligible tail, so the
pair side is a recall problem rather than an ordering one.  P3 asks the complementary question about
the other end: inside the block itself, is there a false-positive population left to demote?  If the
block is nearly pure, block re-ordering is closed for good and the .7-weighted pair term is finished
at S; if it is a third impostors, a demotion lever is worth building on the fold-0 testbed (P4).

Populations
  testbed   the 60 %-prefix pairs with shared_hands >= 38 (every evaluation pair has >= 38), def-A
            classes only: certified positives vs certified false positives.
  block     multiview_eval rows with the production risk >= .5.
  eligible  all 112,540 evaluation pairs (used only to standardise the evaluation logits).
Transport.  The blend's coefficients live on standardised view logits, and the two populations have
different scales (60 % prefixes vs full histories -- the shift that made the M series look good in
17 and closed it in 18.2).  Each view is therefore standardised WITHIN ITS OWN POPULATION before the
model crosses over: testbed statistics for the fit, evaluation statistics for the application.  The
naive same-scale transport is reported next to it as a sensitivity.

(i)   logistic blend (step-60 recipe, l2 = 10) fitted on the def-A testbed -> P(positive | views);
      applied to the block; F = sum(1 - p) is the implied number of false positives, reported whole
      and by third of the S order.
(ii)  two-component Gaussian mixture fitted WITHOUT labels on the block's own standardised logits;
      its component means compared with the testbed's positive and negative means.
(iii) Mahalanobis distance from each block row to each testbed class; rows beyond the 99th percentile
      of the testbed's own within-class distances are an unexplained cluster, described by
      shared_hands, implied table size (from the number of pairs the table contributes) and view
      profile.

Decision, pre-registered: F < 150 -> block headroom below .01 AP, block re-ordering is closed;
F > 300 -> P4 becomes worth its 2-4 h and any new lever is measured on its eligible false positives.
Between the two: report and leave closed.

No evaluation label exists; nothing here is fitted on or tuned by a leaderboard outcome.
Read-only apart from experiments/viewdis_20260911/block_profile.json.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture

from evidence_aug.paths import check_protected, write_json
from pu_pairs.step60_multiview_blend import lg, logreg, predict
from pu_pairs.step64c_hand_view import DEFS, L2, VD, VIEWS

BLOCK = .5
FLOOR = 38


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def z_of(frame, stats=None):
    """standardised view logits; `stats` = (mean, sd) per view, or None to compute them here"""
    X = np.column_stack([lg(frame[f'r_{v}'].to_numpy()) for v in VIEWS])
    if stats is None:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        stats = (mu, sd)
    return (X - stats[0]) / stats[1], stats


def maha(X, mu, cov_inv):
    d = X - mu
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', d, cov_inv, d), 0.0))


def main():
    check_protected()
    ev = pl.read_parquet(VD / 'multiview_eval.parquet')
    tb = pl.read_parquet(VD / 'multiview_prefix.parquet').filter(pl.col('shared_hands') >= FLOOR)
    posset, negset = DEFS['A']
    dev = tb.filter(pl.col('cls').is_in(posset + negset))
    y = dev['cls'].is_in(posset).to_numpy().astype(float)

    block_mask = (ev['risk'].to_numpy() >= BLOCK)
    blk = ev.filter(pl.Series(block_mask))
    res = {'populations': {'eligible_eval': ev.height, 'block': blk.height,
                           'testbed_eligible': tb.height, 'testbed_defA': dev.height,
                           'testbed_defA_positives': int(y.sum())},
           'block_cut': BLOCK, 'labels': 'development labels of the fold 1 / fold 2 prefix pairs only'}
    print('=' * 100)
    print('STEP 73 (P3).  LABEL-FREE PROFILE OF THE EVALUATION BLOCK')
    print('=' * 100)
    print(f"  eligible evaluation pairs {ev.height}, block (risk >= {BLOCK}) {blk.height}; "
          f"testbed def-A {dev.height} of which {int(y.sum())} positive")

    # ---------- (i) the blend, transported two ways ----------
    Zdev_own, _ = z_of(dev)                               # def-A rows on their own statistics
    _, tb_stats = z_of(tb)                                # whole eligible testbed statistics
    Zdev_pop = z_of(dev, tb_stats)[0]
    _, ev_stats = z_of(ev)                                # whole eligible evaluation statistics
    Zblk_pop = z_of(blk, ev_stats)[0]
    Zblk_naive = z_of(blk, tb_stats)[0]

    res['transport'] = {}
    for tag, (Ztr, Zte) in {'population_standardised': (Zdev_pop, Zblk_pop),
                            'naive_same_scale': (Zdev_own, Zblk_naive)}.items():
        model = logreg(Ztr, y, l2=L2)
        p = sig(predict(model, Zte))                      # `predict` returns the linear predictor
        ptr = sig(predict(model, Ztr))
        F = float((1 - p).sum())
        s = blk['risk'].to_numpy()
        order = np.argsort(-s, kind='stable')
        thirds = np.array_split(order, 3)
        row = {'implied_false_positives_F': round(F, 1),
               'expected_positives': round(float(p.sum()), 1),
               'mean_p': round(float(p.mean()), 4),
               'p_quantiles': [round(float(q), 4) for q in np.quantile(p, [.05, .25, .5, .75, .95])],
               'F_by_third_of_risk_order': [round(float((1 - p[t]).sum()), 1) for t in thirds],
               'mean_p_by_third': [round(float(p[t].mean()), 4) for t in thirds],
               'training_fit': {'mean_p_positives': round(float(ptr[y == 1].mean()), 4),
                                'mean_p_negatives': round(float(ptr[y == 0].mean()), 4)},
               'weights': {v: round(float(w), 3) for v, w in zip(VIEWS, model[0])},
               'intercept': round(float(model[1]), 3)}
        res['transport'][tag] = row
        print()
        print(f'  ({tag})  F = {row["implied_false_positives_F"]}  '
              f'of {blk.height} block pairs; mean p = {row["mean_p"]}')
        print(f'    testbed fit: mean p on positives {row["training_fit"]["mean_p_positives"]}, '
              f'on certified false positives {row["training_fit"]["mean_p_negatives"]}')
        print(f'    F by third of the risk order (top / middle / bottom): '
              f'{row["F_by_third_of_risk_order"]}')
        print(f'    weights: {json.dumps(row["weights"])}')

    F = res['transport']['population_standardised']['implied_false_positives_F']

    # ---------- (ii) unsupervised two-component mixture on the block ----------
    gm = GaussianMixture(n_components=2, covariance_type='full', random_state=731, n_init=5).fit(Zblk_pop)
    order = np.argsort(-gm.weights_)
    zpos = Zdev_pop[y == 1].mean(axis=0)
    zneg = Zdev_pop[y == 0].mean(axis=0)
    comp = []
    for i in order:
        mu = gm.means_[i]
        comp.append({'weight': round(float(gm.weights_[i]), 3),
                     'n_assigned': int((gm.predict(Zblk_pop) == i).sum()),
                     'mean': {v: round(float(m), 3) for v, m in zip(VIEWS, mu)},
                     'euclid_to_testbed_positive_mean': round(float(np.linalg.norm(mu - zpos)), 3),
                     'euclid_to_testbed_negative_mean': round(float(np.linalg.norm(mu - zneg)), 3)})
    res['mixture'] = {'components': comp,
                      'testbed_positive_mean': {v: round(float(m), 3) for v, m in zip(VIEWS, zpos)},
                      'testbed_negative_mean': {v: round(float(m), 3) for v, m in zip(VIEWS, zneg)}}
    print()
    print('  (ii) unsupervised 2-component mixture on the block:')
    for c in comp:
        print(f'    weight {c["weight"]:.3f}  n={c["n_assigned"]:>4}  '
              f'distance to testbed positive mean {c["euclid_to_testbed_positive_mean"]:.3f}, '
              f'to false-positive mean {c["euclid_to_testbed_negative_mean"]:.3f}')

    # ---------- (iii) anything the testbed does not contain ----------
    dists, ref99 = {}, {}
    for name, m in (('positive', y == 1), ('false_positive', y == 0)):
        Z = Zdev_pop[m]
        mu = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False) + 1e-6 * np.eye(len(VIEWS))
        inv = np.linalg.inv(cov)
        ref99[name] = float(np.quantile(maha(Z, mu, inv), .99))
        dists[name] = maha(Zblk_pop, mu, inv)
    far = (dists['positive'] > ref99['positive']) & (dists['false_positive'] > ref99['false_positive'])
    pairs_per_table = dict(ev.group_by('table_id').len().iter_rows())
    seats = blk['table_id'].map_elements(
        lambda t: (1 + (1 + 8 * pairs_per_table[t]) ** .5) / 2, return_dtype=pl.Float64).to_numpy()
    res['outliers'] = {
        'reference_99th_percentiles': {k: round(v, 3) for k, v in ref99.items()},
        'n_beyond_both': int(far.sum()), 'fraction_of_block': round(float(far.mean()), 4),
        'shared_hands': {'block_median': float(np.median(blk['shared_hands'].to_numpy())),
                         'outlier_median': (float(np.median(blk['shared_hands'].to_numpy()[far]))
                                            if far.any() else None)},
        'implied_table_seats': {'block_median': round(float(np.median(seats)), 2),
                                'outlier_median': (round(float(np.median(seats[far])), 2)
                                                   if far.any() else None)},
        'mean_standardised_logits': {
            'block': {v: round(float(Zblk_pop[:, i].mean()), 3) for i, v in enumerate(VIEWS)},
            'outliers': ({v: round(float(Zblk_pop[far, i].mean()), 3) for i, v in enumerate(VIEWS)}
                         if far.any() else None)},
        'R1_rate': {'block': round(float(blk['R1'].to_numpy().mean()), 4),
                    'outliers': (round(float(blk['R1'].to_numpy()[far].mean()), 4)
                                 if far.any() else None)}}
    print()
    print(f'  (iii) block rows beyond the 99th percentile of BOTH testbed classes: '
          f'{int(far.sum())} ({100 * far.mean():.1f} %)')
    if far.any():
        print(f'    shared hands median {res["outliers"]["shared_hands"]["outlier_median"]} '
              f'vs block {res["outliers"]["shared_hands"]["block_median"]}; implied seats '
              f'{res["outliers"]["implied_table_seats"]["outlier_median"]} vs '
              f'{res["outliers"]["implied_table_seats"]["block_median"]}')
        print(f'    mean standardised logits: {json.dumps(res["outliers"]["mean_standardised_logits"]["outliers"])}')

    verdict = ('closed: F < 150, block headroom below .01 AP' if F < 150 else
               ('build P4: F > 300' if F > 300 else 'inconclusive band [150, 300]: report and leave closed'))
    res['decision'] = {'F': F, 'rule': 'F < 150 -> closed; F > 300 -> build P4', 'verdict': verdict}
    print()
    print(f'  DECISION: F = {F} -> {verdict}')
    write_json(VD / 'block_profile.json', res)
    check_protected()
    print()
    print(f'written {VD / "block_profile.json"}')
    print('DONE')


if __name__ == '__main__':
    main()
