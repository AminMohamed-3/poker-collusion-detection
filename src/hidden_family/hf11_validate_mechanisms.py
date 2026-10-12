"""HF11 stage 2: does each mechanism statistic SEE anything, and does it see the same thing
the disclosed families already are?

This is the inversion.  hf3-hf7 asked "is there an unexplained population?" and answered no,
but could not distinguish MECHANISM ABSENT from DETECTOR BLIND.  Here the order is reversed:
reason forward from what a generator would plausibly implement, build the statistic that
mechanism would move, and first establish on GROUND TRUTH whether the statistic has any power
at all.  Only a statistic with demonstrated power can turn a null into evidence of absence.

Three questions per statistic, in this order:

  Q1 POWER ON KNOWN COORDINATION.  AUC against the 1,488 confirmed non-targets, per family.
     A statistic that cannot separate ANY real coordination is not yet trustworthy, and hf12
     goes on to plant its own mechanism to settle that case.
  Q2 ORTHOGONALITY.  If a statistic is high on a disclosed family, extremes of it in the
     unlabelled population are most likely unlisted members of that family, not a fourth
     mechanism.  The interesting statistics are the ones the disclosed families do NOT move.
  Q3 PAIR-SPECIFICITY.  OBS 15.3's decisive control: coordination is a property of a PAIR, an
     odd individual is a property of a PLAYER and contaminates all 29 of that player's pairs.
     Every statistic is therefore also reported as an excess over the player baseline.

Read-only.  Consumes hf10's frozen table; writes only into experiments/.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from evidence_aug.paths import ORIGINAL, CV, write_json

OUT = CV / 'experiments' / 'hidden_family_20260912'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
STATS = ['n_both', 'contest_both', 'avoid_lr', 'avoid_z', 'avoid_sd', 'shared_victim',
         'sv_overlap', 'sv_conc', 'chain', 'chain_z', 'flow_z', 'direct', 'churn',
         'net_flow', 'churn_ratio', 'churn_cancel', 'copresence', 'seat_conc',
         'adjacency', 'synchrony']
HYP = {'H1 multi-accounting': ['avoid_lr', 'avoid_z', 'avoid_sd', 'synchrony'],
       'H2 shared victim': ['shared_victim', 'sv_overlap', 'sv_conc'],
       'H3 laundering': ['chain', 'chain_z'],
       'H4 alternating churn': ['churn', 'churn_ratio', 'churn_cancel'],
       'H5 selection': ['copresence', 'seat_conc', 'adjacency', 'synchrony']}


def load():
    """mechstats joined to the full development label set, with a player-baseline excess."""
    ms = pl.read_parquet(OUT / 'hf10-mechstats-development.parquet')
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    # labels are unordered pairs; canonicalise both sides before joining
    lab = lab.with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('lo'),
        pl.max_horizontal('player_1', 'player_2').alias('hi')).select(
        'pair_id', 'lo', 'hi', 'label', 'behavior_family')
    ms = ms.with_columns(
        pl.min_horizontal('player_a', 'player_b').alias('lo'),
        pl.max_horizontal('player_a', 'player_b').alias('hi'))
    fr = ms.join(lab, on=['lo', 'hi'], how='left').with_columns(
        pl.col('label').fill_null(-1), pl.col('behavior_family').fill_null('unlabelled'))
    return fr


def add_excess(fr, stats):
    """excess(a,b) = s(a,b) - max(median of a's other pairs, median of b's other pairs).

    Each player sits at exactly one table and therefore has exactly 29 pairs, so the baseline
    is a clean within-player median.  Leave-one-out is unnecessary at n=29 but done anyway.
    """
    n = fr.height
    pa = fr['player_a'].to_numpy()
    pb = fr['player_b'].to_numpy()
    rows = {}
    for i, (a, b) in enumerate(zip(pa, pb)):
        rows.setdefault(a, []).append(i)
        rows.setdefault(b, []).append(i)
    rows = {p: np.array(v) for p, v in rows.items()}
    deg = np.array([len(v) for v in rows.values()])
    assert deg.min() == deg.max() == 29, f'unexpected pair degree {deg.min()}-{deg.max()}'

    out = {}
    for s in stats:
        v = fr[s].to_numpy().astype(float)
        base = np.empty(n)
        med = {p: np.median(v[idx]) for p, idx in rows.items()}
        for i in range(n):
            ba, bb = med[pa[i]], med[pb[i]]
            base[i] = max(ba, bb)
        out[s + '_hi'] = v - base
        base2 = np.array([min(med[pa[i]], med[pb[i]]) for i in range(n)])
        out[s + '_lo'] = v - base2          # for statistics where LOW is the signal
    return fr.with_columns(**{k: pl.Series(k, v) for k, v in out.items()})


def auc(score, pos_mask, neg_mask):
    y = np.r_[np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())]
    s = np.r_[score[pos_mask], score[neg_mask]]
    ok = np.isfinite(s)
    if y[ok].sum() < 3 or (1 - y[ok]).sum() < 3:
        return float('nan')
    return float(roc_auc_score(y[ok], s[ok]))


def main():
    fr = load()
    lab_n = int((fr['label'] == 0).sum())
    lab_p = int((fr['label'] == 1).sum())
    print(f'{fr.height} development pairs; {lab_p} labelled positives, {lab_n} confirmed '
          f'non-targets, {fr.height - lab_p - lab_n} unlabelled')
    fam = fr['behavior_family'].to_numpy()
    neg = fr['label'].to_numpy() == 0
    pos = fr['label'].to_numpy() == 1
    fr = add_excess(fr, STATS)

    print()
    print('=' * 100)
    print('Q1/Q2.  POWER OF EACH MECHANISM STATISTIC ON THE THREE DISCLOSED FAMILIES')
    print('        AUC vs the 1,488 confirmed non-targets.  .50 = the statistic sees nothing;')
    print('        far from .50 either way = it sees that family.')
    print('=' * 100)
    print(f'{"statistic":<16}{"DT":>8}{"SP":>8}{"CI":>8}{"any":>8}   '
          f'{"med(pos)":>11}{"med(neg)":>11}')
    res = {}
    for s in STATS:
        v = fr[s].to_numpy().astype(float)
        a = [auc(v, fam == f, neg) for f in FAMS]
        best = max(abs(x - .5) for x in a if np.isfinite(x))
        res[s] = {'auc': dict(zip(FAMS, a)), 'max_dev': best,
                  'median_pos': float(np.median(v[pos])), 'median_neg': float(np.median(v[neg]))}
        print(f'{s:<16}{a[0]:>8.3f}{a[1]:>8.3f}{a[2]:>8.3f}{best:>8.3f}   '
              f'{np.median(v[pos]):>11.4f}{np.median(v[neg]):>11.4f}')

    print()
    print('=' * 100)
    print('Q3.  THE SAME STATISTICS AS A PAIR-SPECIFIC EXCESS OVER THE PLAYER BASELINE')
    print('     (_hi subtracts the higher of the two player medians, _lo the lower)')
    print('=' * 100)
    print(f'{"statistic":<20}{"DT":>8}{"SP":>8}{"CI":>8}{"any":>8}')
    for s in STATS:
        for suf in ('_hi', '_lo'):
            v = fr[s + suf].to_numpy().astype(float)
            a = [auc(v, fam == f, neg) for f in FAMS]
            best = max(abs(x - .5) for x in a if np.isfinite(x))
            res[s + suf] = {'auc': dict(zip(FAMS, a)), 'max_dev': best}
            print(f'{s + suf:<20}{a[0]:>8.3f}{a[1]:>8.3f}{a[2]:>8.3f}{best:>8.3f}')

    print()
    print('=' * 100)
    print('VERDICT PER HYPOTHESIS (best |AUC-.5| over that hypothesis\'s statistics)')
    print('=' * 100)
    verdict = {}
    for h, ss in HYP.items():
        keys = [k for s in ss for k in (s, s + '_hi', s + '_lo')]
        best_k = max(keys, key=lambda k: res[k]['max_dev'])
        verdict[h] = {'best_statistic': best_k, 'max_dev': res[best_k]['max_dev'],
                      'auc': res[best_k]['auc']}
        a = res[best_k]['auc']
        print(f'  {h:<24} best={best_k:<18} AUC  DT {a["directed_transfer"]:.3f}  '
              f'SP {a["soft_play"]:.3f}  CI {a["coordinated_isolation"]:.3f}')

    fr.write_parquet(OUT / 'hf11-dev-mechstats-labelled.parquet')
    write_json(OUT / 'hf11-validation.json', {'per_statistic': res, 'per_hypothesis': verdict,
                                              'n_pairs': fr.height, 'n_pos': lab_p,
                                              'n_neg': lab_n})
    print()
    print(f'written {OUT / "hf11-validation.json"}')


if __name__ == '__main__':
    main()
