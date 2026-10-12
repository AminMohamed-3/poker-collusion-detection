"""Step 73b: is step 73's F a measurement, or a restatement of the risk column?

Step 73 (P3) reported F = sum(1 - p) = 401.3 implied false positives among the 909 evaluation pairs
with production risk >= .5, and its pre-registered rule (F > 300) fires "build P4".  Before that number
is allowed to mean anything, two structural facts about the fit have to be faced.

  1. CIRCULARITY.  The def-A contrast is ['pos', 'undisclosed'] vs ['FP', 'undisclosed_R1'].  On the
     eligible prefix testbed that is 138 + 62 against 49,742 + 7.  `undisclosed` is *defined* as
     risk_full >= .9 and `FP` as risk_full < .1 -- so 62/200 of the positive class and 49,742/49,749
     of the negative class are risk thresholds wearing class labels, and the eight views are sub-models
     of the very pair model that produces risk.  A logistic fitted on that contrast largely re-learns
     "is risk high or low", and F then re-describes where the block sits on the risk axis.  Step 73's
     own docstring calls the negatives "certified false positives"; they are not certified anything,
     they are unlabelled pairs the production model already scores below .1.
  2. PRIOR DEPENDENCE.  F is a sum of probabilities, and a logistic's probabilities are calibrated to
     its training base rate -- 200/49,949 = 0.4 % for def-A.  Change the class balance and F moves,
     with no change whatsoever in the block.  A count that a training-set composition can dial is not
     a count of anything in the evaluation set.

So this step re-asks P3's question three ways and adds a statistic that neither defect can touch.

  (a) fits, transported population-standardised exactly as step 73 did:
        defA         reproduction of step 73's model.
        labels_only  the 138 'pos' against the 526 'neg' -- the only genuinely labelled contrast in
                     the testbed, and the one that actually asks "positive or hard negative".
        risk_only    def-A on the single feature lg(risk) -- how much of defA's p is just the risk.
      For each: F, mean p, and Spearman between p and risk INSIDE the block.  If defA's p is rank-
      equivalent to risk inside the block, F carries no information the risk column did not already
      carry, and neither does the pre-registered rule keyed on it.
  (b) prior sensitivity: F recomputed under the standard prior-shift correction for assumed block
      positive rates, to show the width of the arbitrary.
  (c) PRIOR-FREE RANK STATISTICS on the labels_only score, which is what P3 should have asked:
        above_neg_p95  fraction of block pairs scoring above the 95th percentile of the 526 labelled
                       negatives -- block pairs the labelled negatives cannot account for.
        inside_neg     fraction at or below the labelled negatives' median -- pairs sitting in the
                       middle of the certified-negative distribution, the demotable population.
        below_pos_p05  fraction below the 5th percentile of the 138 labelled positives.
      All three reported whole and by third of the S risk order.  Ranks are immune to the training
      prior, and the labelled contrast is immune to the circularity.

Nothing here is fitted on or tuned by a leaderboard outcome; the labels used are development labels.
Read-only apart from experiments/viewdis_20260911/block_profile_circularity.json.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from evidence_aug.paths import check_protected, write_json
from pu_pairs.step60_multiview_blend import lg, logreg, predict
from pu_pairs.step64c_hand_view import DEFS, L2, VD, VIEWS
from pu_pairs.step73_block_profile import BLOCK, FLOOR, sig, z_of

PRIORS = (.3, .5, .7, .9)


def thirds_of(risk):
    return np.array_split(np.argsort(-risk, kind='stable'), 3)


def main():
    check_protected()
    ev = pl.read_parquet(VD / 'multiview_eval.parquet')
    tb = pl.read_parquet(VD / 'multiview_prefix.parquet').filter(pl.col('shared_hands') >= FLOOR)
    blk = ev.filter(pl.Series(ev['risk'].to_numpy() >= BLOCK))
    risk = blk['risk'].to_numpy()
    parts = thirds_of(risk)

    res = {'block': blk.height, 'testbed_eligible': tb.height, 'block_cut': BLOCK,
           'why': 'defA classes are largely risk thresholds; F is prior-dependent; ranks are neither'}

    print('=' * 100)
    print('STEP 73b.  IS P3 MEASURING THE BLOCK, OR MEASURING THE RISK COLUMN?')
    print('=' * 100)

    # ---------- how much of each def-A class is a risk threshold ----------
    comp = {}
    for c, g in tb.group_by('cls'):
        r = g['risk_full'].to_numpy()
        comp[c[0]] = {'n': g.height, 'risk_full_q': [round(float(q), 4) for q in
                                                     np.quantile(r, [.05, .5, .95])],
                      'labelled': bool(c[0] in ('pos', 'neg'))}
    res['testbed_classes'] = comp
    posset, negset = DEFS['A']
    res['defA'] = {'positives': {k: comp[k]['n'] for k in posset if k in comp},
                   'negatives': {k: comp[k]['n'] for k in negset if k in comp}}
    n_lab_pos = comp.get('pos', {}).get('n', 0)
    n_all_pos = sum(res['defA']['positives'].values())
    n_all_neg = sum(res['defA']['negatives'].values())
    res['defA']['fraction_of_classes_defined_by_a_risk_threshold'] = round(
        1 - (n_lab_pos + 0) / (n_all_pos + n_all_neg), 5)
    print(f"  def-A positives {res['defA']['positives']}  negatives {res['defA']['negatives']}")
    print(f"  labelled rows in def-A: {n_lab_pos} of {n_all_pos + n_all_neg} "
          f"({100 * n_lab_pos / (n_all_pos + n_all_neg):.2f} %); the rest are risk thresholds")
    print(f"  labelled negatives available but UNUSED by def-A: cls 'neg' n={comp.get('neg', {}).get('n')}"
          f"  risk_full 5/50/95 = {comp.get('neg', {}).get('risk_full_q')}")

    # ---------- (a) three fits ----------
    _, tb_stats = z_of(tb)
    _, ev_stats = z_of(ev)
    Zblk = z_of(blk, ev_stats)[0]
    rblk_only = lg(risk)[:, None]

    fits = {}
    dev_a = tb.filter(pl.col('cls').is_in(posset + negset))
    fits['defA'] = (z_of(dev_a, tb_stats)[0], dev_a['cls'].is_in(posset).to_numpy().astype(float), Zblk)
    dev_l = tb.filter(pl.col('cls').is_in(['pos', 'neg']))
    fits['labels_only'] = (z_of(dev_l, tb_stats)[0], (dev_l['cls'] == 'pos').to_numpy().astype(float), Zblk)
    fits['risk_only'] = (lg(dev_a['risk_full'].to_numpy())[:, None],
                         dev_a['cls'].is_in(posset).to_numpy().astype(float), rblk_only)

    res['fits'] = {}
    scores = {}
    for tag, (Xtr, y, Xte) in fits.items():
        model = logreg(Xtr, y, l2=L2)
        z = predict(model, Xte)
        p = sig(z)
        scores[tag] = z
        rho = float(spearmanr(z, risk).statistic)
        ptr = sig(predict(model, Xtr))
        row = {'n_train': int(len(y)), 'train_positive_rate': round(float(y.mean()), 5),
               'implied_false_positives_F': round(float((1 - p).sum()), 1),
               'mean_p': round(float(p.mean()), 4),
               'F_by_third': [round(float((1 - p[t]).sum()), 1) for t in parts],
               'spearman_with_risk_in_block': round(rho, 4),
               'train_mean_p': {'positives': round(float(ptr[y == 1].mean()), 4),
                                'negatives': round(float(ptr[y == 0].mean()), 4)}}
        res['fits'][tag] = row
        print()
        print(f"  ({tag})  n_train {row['n_train']} at {100 * row['train_positive_rate']:.2f} % positive"
              f"  ->  F = {row['implied_false_positives_F']}, mean p {row['mean_p']}")
        print(f"    Spearman(score, risk) inside the block = {rho:+.4f}"
              f"    F by third {row['F_by_third']}")

    res['rank_agreement_defA_vs_risk_only'] = round(
        float(spearmanr(scores['defA'], scores['risk_only']).statistic), 4)
    print()
    print(f"  Spearman(defA score, risk-only score) inside the block = "
          f"{res['rank_agreement_defA_vs_risk_only']:+.4f}")

    # ---------- (b) prior sensitivity of F ----------
    res['prior_sensitivity'] = {}
    for tag in ('defA', 'labels_only'):
        pi_tr = res['fits'][tag]['train_positive_rate']
        z = scores[tag]
        row = {}
        for pi in PRIORS:
            shift = np.log(pi / (1 - pi)) - np.log(pi_tr / (1 - pi_tr))
            row[str(pi)] = round(float((1 - sig(z + shift)).sum()), 1)
        res['prior_sensitivity'][tag] = row
        print(f"  F under assumed block positive rate {dict(row)}   (fit prior {pi_tr:.4f})")

    # ---------- (c) prior-free rank statistics on the labelled contrast ----------
    model = logreg(*fits['labels_only'][:2], l2=L2)
    s_blk = predict(model, Zblk)
    s_pos = predict(model, z_of(tb.filter(pl.col('cls') == 'pos'), tb_stats)[0])
    s_neg = predict(model, z_of(tb.filter(pl.col('cls') == 'neg'), tb_stats)[0])
    cuts = {'above_neg_p95': (float(np.quantile(s_neg, .95)), 'gt'),
            'inside_neg_at_or_below_median': (float(np.median(s_neg)), 'le'),
            'below_pos_p05': (float(np.quantile(s_pos, .05)), 'lt')}
    rank = {'n_labelled_pos': int(len(s_pos)), 'n_labelled_neg': int(len(s_neg)), 'stats': {}}
    for name, (c, op) in cuts.items():
        m = (s_blk > c) if op == 'gt' else ((s_blk <= c) if op == 'le' else (s_blk < c))
        rank['stats'][name] = {'cut': round(c, 3), 'fraction': round(float(m.mean()), 4),
                               'n': int(m.sum()),
                               'by_third': [round(float(m[t].mean()), 4) for t in parts]}
        print(f"  {name:<30} {int(m.sum()):>4} of {blk.height} ({100 * m.mean():5.1f} %)   "
              f"by third {rank['stats'][name]['by_third']}")
    res['rank_free'] = rank

    demotable = rank['stats']['inside_neg_at_or_below_median']['n']
    res['reading'] = {
        'defA_F_is_rank_equivalent_to_risk': res['fits']['defA']['spearman_with_risk_in_block'] >= .95,
        'prior_free_demotable_estimate': demotable,
        'note': ('F is not a count of false positives: its ordering is the risk column and its scale is '
                 'the training prior. The prior-free rank statistics on the labelled contrast are the '
                 'number this question should have been answered with.')}
    write_json(VD / 'block_profile_circularity.json', res)
    check_protected()
    print()
    print(f"written {VD / 'block_profile_circularity.json'}")
    print('DONE')


if __name__ == '__main__':
    main()
