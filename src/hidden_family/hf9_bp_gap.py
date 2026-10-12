"""HF9: pinning the hidden family's prevalence from the measured B - P gap.

hf8 placed hidden-family pairs near the top of our own ranking and concluded the behaviour
component bounds the prevalence tightly.  That placement contradicts hf1, which MEASURED where
an untrained family lands: under leave-one-family-out the withheld family's pairs collapse to
median risk .02-.24 and essentially none reach .90.  Hidden pairs are therefore deep in our
tail, not in the .5 block, and the two public components respond very differently:

  PAIR AP  loses them outright -- they are truth positives sitting at rank ~1,000-8,000.
  BEHAVIOR MAP  barely notices them -- their score inside a disclosed family's ranking is
           `risk`, which is ~.05, so they are weak false positives that sit below the genuine
           members they would otherwise displace.

So a hidden family OPENS A GAP between B and P, and the gap we actually measure on the public
board is B - P = .976 - .938 = .038.  That is a two-equation constraint on one unknown, which
is the strongest identification available anywhere in this investigation.  This script
simulates both components at evaluation scale with the empirical risk distributions --
disclosed positives from the labelled pairs under the trained model, hidden positives from the
LOFO omit runs at shortened history, negatives from our own submission -- and reports which
prevalence reproduces the measured pair.

Caveat kept in view: OBS 14.23-14.35 showed part of the P deficit is length-induced false
positives and short-history recall misses among DISCLOSED families, so every pi here is an
UPPER bound.

Read-only.  No submission is written or modified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import ORIGINAL, CV, write_json

OUT = CV / 'experiments' / 'hidden_family_20260912'
CONF = ORIGINAL / 'reports' / 'conditional-action-confirmation' / 'results'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
SUB = CV / 'outputs_multiview_20260911' / 'submission_M4_stage2_chrono3.csv'
PUBLIC_P, PUBLIC_B = 0.938, 0.976
RNG = np.random.default_rng(20260912)


def stable_ap(scores, truth, ids):
    order = np.argsort(ids, kind='mergesort')
    s, t = np.asarray(scores)[order], np.asarray(truth)[order]
    idx = np.argsort(-s, kind='mergesort')
    t = t[idx]
    if t.sum() == 0:
        return float('nan')
    tp = np.cumsum(t)
    return float(((tp / np.arange(1, len(t) + 1)) * t).sum() / t.sum())


def main():
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    screen = pl.read_csv(ORIGINAL / 'reports' / 'conditional-action-experiment' /
                         'screen-results' / 'pair-predictions.csv').filter(
        (pl.col('candidate') == 'conditional') & (pl.col('history') == 'shortened'))
    risk_map = dict(zip(screen['pair_id'].to_list(), screen['risk'].to_list()))
    pos_ids = labels.filter(pl.col('label') == 1)['pair_id'].to_list()
    disclosed = np.array([risk_map[p] for p in pos_ids if p in risk_map])

    hidden = []
    for fam in FAMS:
        d = pl.read_csv(CONF / f'omit_{fam}' / 'pair-predictions.csv').filter(
            (pl.col('candidate') == 'conditional') & (pl.col('history') == 'shortened'))
        fam_ids = set(labels.filter(pl.col('behavior_family') == fam)['pair_id'])
        hidden += d.filter(pl.col('pair_id').is_in(list(fam_ids)))['risk'].to_list()
    hidden = np.array(hidden)

    sub = pl.read_csv(SUB, columns=['risk_score', 'predicted_behavior'])
    all_risk = sub['risk_score'].to_numpy()
    all_pred = sub['predicted_behavior'].to_numpy()
    neg_pool = all_risk[all_risk < 0.5]
    N = len(all_risk)
    fam_share = np.array([float((all_pred == f).mean()) for f in FAMS])
    fam_share = fam_share / fam_share.sum()

    print(f'disclosed-positive risk pool n={len(disclosed)} median={np.median(disclosed):.4f}')
    print(f'hidden-positive risk pool    n={len(hidden)} median={np.median(hidden):.4f} '
          f'share>=.5={np.mean(hidden >= .5):.3f}')
    print(f'negative pool n={len(neg_pool)}  submission rows {N}')
    print()
    print('=' * 96)
    print('PAIR AP AND BEHAVIOR MAP TOGETHER, AS A FUNCTION OF HIDDEN-FAMILY PREVALENCE')
    print('=' * 96)
    print(f'{"pi":>6}{"n_hidden":>10}{"PairAP":>9}{"BehavMAP":>10}{"B-P gap":>10}'
          f'{"|P-.938|":>10}{"|gap-.038|":>12}')
    rows = []
    n_pos = 930
    for pi in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]:
        Ps, Bs = [], []
        for _ in range(10):
            n_h = int(round(n_pos * pi))
            n_d = n_pos - n_h
            r = np.concatenate([RNG.choice(disclosed, n_d, replace=True),
                                RNG.choice(hidden, n_h, replace=True),
                                RNG.choice(neg_pool, N - n_pos, replace=True)])
            truth = np.array(['none'] * N, dtype=object)
            truth[:n_d] = RNG.choice(FAMS, n_d, p=fam_share)
            truth[n_d:n_pos] = 'other_coordination'
            tf = np.array([str(x) for x in truth])
            y = np.r_[np.ones(n_pos), np.zeros(N - n_pos)]
            ids = np.arange(N)
            RNG.shuffle(ids)
            Ps.append(stable_ap(r, y, ids))
            # our predicted family: correct for disclosed pairs, and for hidden pairs and
            # negatives whatever the production model would say (drawn from its own mix)
            pred = np.empty(N, dtype=object)
            pred[:n_d] = tf[:n_d]
            pred[n_d:] = RNG.choice(FAMS, N - n_d, p=fam_share)
            pred = np.array([str(x) for x in pred])
            aps = []
            for f in FAMS:
                aps.append(stable_ap(np.where(pred == f, r, 0.0),
                                     (tf == f).astype(int), ids))
            Bs.append(float(np.mean(aps)))
        P, B = float(np.mean(Ps)), float(np.mean(Bs))
        rows.append({'pi': pi, 'pair_ap': P, 'behavior_map': B, 'gap': B - P})
        print(f'{pi:>6.2f}{int(n_pos * pi):>10}{P:>9.4f}{B:>10.4f}{B - P:>10.4f}'
              f'{abs(P - PUBLIC_P):>10.4f}{abs((B - P) - (PUBLIC_B - PUBLIC_P)):>12.4f}')

    best_p = min(rows, key=lambda r: abs(r['pair_ap'] - PUBLIC_P))
    best_g = min(rows, key=lambda r: abs(r['gap'] - (PUBLIC_B - PUBLIC_P)))
    print()
    print(f'  measured public: P = {PUBLIC_P:.3f}, B = {PUBLIC_B:.3f}, gap = '
          f'{PUBLIC_B - PUBLIC_P:.3f}')
    print(f'  pi reproducing the measured PAIR AP  : {best_p["pi"]:.2f}')
    print(f'  pi reproducing the measured B-P GAP  : {best_g["pi"]:.2f}')
    print()
    print('  Both are UPPER bounds: disclosed-family length effects and short-history recall')
    print('  misses (OBS 14.23-14.35) already account for part of the same pair-AP deficit.')

    write_json(OUT / 'hf9-bp-gap.json',
               {'public': {'P': PUBLIC_P, 'B': PUBLIC_B, 'gap': PUBLIC_B - PUBLIC_P},
                'hidden_pool_median': float(np.median(hidden)),
                'disclosed_pool_median': float(np.median(disclosed)),
                'rows': rows, 'pi_from_pair_ap': best_p['pi'], 'pi_from_gap': best_g['pi']})
    print()
    print(f'written {OUT / "hf9-bp-gap.json"}')


if __name__ == '__main__':
    main()
