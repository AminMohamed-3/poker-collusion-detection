"""Step 67d: pair AP on the population the evaluation metric actually scores -- every eligible
pair, not the risk >= .5 block.

Why this exists.  Every pair-side experiment since step 53 measured AP inside the block
(`risk_prefix >= .5`, 306 eligible rows), because that is where the S permutation acts: S is a
multiset-preserving rank permutation of the 704 block pairs and leaves all 111,831 other rows at
their frozen risk.  But the competition's pair AP runs over all 112,540 evaluation rows, and every
evaluation pair already has at least 38 shared hands, so the honest analogue is the whole eligible
testbed: 50,761 pairs, 138 labelled positives, all eight views present on every row.

Two numbers motivated it.  Step 67c (P2) found the eligible tail costs +.0370 AP under the S order;
and the AP of S over the eligible class population, .9568, sits .0016 from the pair AP the public
probe measured for the S submission, .9584.  If that correspondence holds, the pair side is not
closed at S at all -- it is closed only inside the block, and the 8 % of eligible positives that
score below .5 are the largest single lever left in the composite (.7 weight).

Populations
  EL    every prefix-testbed pair with shared_hands >= 38 (50,761 rows, 138 positives)
  ELc   the same with the def-A class filter (the step-67c population, for continuity)
Orders
  frozen        risk_prefix, the production v13 risk (the base submission's column)
  S             what the S submission does: the FULL/RAW blend INSIDE the block, frozen outside
  S_global      the same blend applied to every eligible row
  <view>        each of the eight views alone, globally
  rankfuse      mean percentile rank of the eight views, globally
  rankfuse_R1   rankfuse with the R1 rule applied
  M_global      the eight views + log shared_hands, logistic blend, fold 1 <-> fold 2 crossfold
  S_then_M      S inside the block, M_global outside it (a strictly additive change to S)
Labels: development labels only (fold 1 / fold 2 prefix pairs).  No evaluation label exists and no
leaderboard outcome enters any fit here.

Read-only apart from `experiments/viewdis_20260911/global_order.json`.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import check_protected, write_json
from pu_pairs.step60_multiview_blend import ap, lg
from pu_pairs.step64c_hand_view import DEFS, VD, VIEWS
from pu_pairs.step66_evidence_view import crossfold

RNG = np.random.default_rng(20260912)
BLOCK = .5


def sub30_sd(y, a, b, n=400):
    k = max(int(round(.3 * len(y))), 10)
    out = []
    for _ in range(n):
        idx = RNG.choice(len(y), k, replace=False)
        if y[idx].sum() < 3:
            continue
        out.append(ap(y[idx], a[idx]) - ap(y[idx], b[idx]))
    return float(np.std(out)) if out else float('nan')


def prank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind='stable')] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def recall_at(y, s, k):
    return float(y[np.argsort(-s, kind='stable')[:k]].sum() / max(y.sum(), 1))


def orders(c, y):
    r1 = c['R1_prefix'].to_numpy()
    fr = c['risk_prefix'].to_numpy().astype(float)
    blend = lg(c['r_FULL'].to_numpy()) + .5 * lg(c['r_RAW'].to_numpy())
    in_block = fr >= BLOCK

    out = {'frozen': fr}
    # S: a permutation of the block only.  Re-embed the blend order inside the block's own risk
    # values so that nothing outside the block moves -- exactly the step-59 construction.
    s = fr.copy()
    if in_block.any():
        bs = np.where(r1[in_block], blend[in_block] - 100, blend[in_block])
        vals = np.sort(fr[in_block])                       # the block's own multiset of risks
        s[in_block] = vals[np.argsort(np.argsort(bs, kind='stable'), kind='stable')]
    out['S (block permutation)'] = s
    out['S_global (blend everywhere)'] = np.where(r1, blend - 100, blend)
    for v in VIEWS:
        out[f'view {v}'] = lg(c[f'r_{v}'].to_numpy())
    rf = np.mean([prank(lg(c[f'r_{v}'].to_numpy())) for v in VIEWS], axis=0)
    out['rankfuse (8 views)'] = rf
    out['rankfuse + R1'] = np.where(r1, rf - 100, rf)

    m, _, _ = crossfold(c, y, VIEWS + ['sh'])
    out['M_global (8 views + sh)'] = m
    m2, _, _ = crossfold(c, y, VIEWS + ['sh', 'R1'])
    out['M2_global (+ R1)'] = m2
    # S inside the block, the learned blend outside: additive on top of what already ships
    tail = np.where(in_block, 0.0, prank(m) - 10.0)
    out['S + M_global below the block'] = np.where(in_block, s, tail)
    tail2 = np.where(in_block, 0.0, prank(rf) - 10.0)
    out['S + rankfuse below the block'] = np.where(in_block, s, tail2)
    return out


def report(tag, c, y, res):
    print()
    print(f'  {tag}:  n={c.height}  positives={int(y.sum())}  '
          f'in block={int((c["risk_prefix"] >= BLOCK).to_numpy().sum())}')
    print(f'    {"order":<34}{"AP":>9}{"vs frozen":>11}{"sub30 SD":>10}{"R@300":>8}{"R@1000":>8}')
    ords = orders(c, y)
    ref = ords['frozen']
    row = {'n': int(c.height), 'n_pos': int(y.sum()), 'orders': {}}
    for name, s in ords.items():
        a = ap(y, s)
        d = a - ap(y, ref)
        sd = sub30_sd(y, s, ref) if name != 'frozen' else 0.0
        r3, r10 = recall_at(y, s, 300), recall_at(y, s, 1000)
        row['orders'][name] = {'AP': round(a, 4), 'vs_frozen': round(d, 4), 'sub30_sd': round(sd, 4),
                               'recall_at_300': round(r3, 4), 'recall_at_1000': round(r10, 4)}
        print(f'    {name:<34}{a:>9.4f}{d:>+11.4f}{sd:>10.4f}{r3:>8.3f}{r10:>8.3f}')
    res[tag] = row
    return row


def main():
    check_protected()
    mp = pl.read_parquet(VD / 'multiview_prefix.parquet')
    r1 = (pl.read_parquet(VD / 'prefix_rescoring.parquet').select('pair_id', 'R1_prefix')
          .unique(subset=['pair_id'], keep='first'))
    d = (mp.join(r1, on='pair_id', how='left')
         .with_columns(pl.col('R1_prefix').fill_null(False))
         .filter(pl.col('shared_hands') >= 38))
    res = {'population': 'prefix testbed, shared_hands >= 38 (every evaluation pair has >= 38)',
           'block_cut': BLOCK, 'labels': 'development labels of the fold 1 / fold 2 prefix pairs'}

    print('=' * 104)
    print('STEP 67d.  PAIR AP OVER EVERY ELIGIBLE PAIR (the population the evaluation metric scores)')
    print('=' * 104)
    print(f'  eligible testbed pairs {d.height}, labelled positives {int((d["label"] == 1).sum())}, '
          f'confirmed negatives {int((d["label"] == 0).sum())}, '
          f'unlabelled {int(d["label"].is_null().sum() + (d["label"] == 0).sum() * 0)}')

    y_all = (d['label'] == 1).to_numpy().astype(float)
    report('EL (all eligible, positives = labelled)', d, y_all, res)

    posset, negset = DEFS['A']
    c = d.filter(pl.col('cls').is_in(posset + negset))
    report('ELc (def A classes only, for continuity with 67c)',
           c, c['cls'].is_in(posset).to_numpy().astype(float), res)

    # where the positives that the frozen risk misses actually sit
    el = d.filter(pl.col('label') == 1)
    r = el['risk_prefix'].to_numpy()
    bands = [('>= .5', (r >= .5)), ('[.1, .5)', (r >= .1) & (r < .5)),
             ('[.01, .1)', (r >= .01) & (r < .1)), ('< .01', r < .01)]
    print()
    print('  where the 138 eligible positives sit under the frozen risk:')
    res['positive_bands'] = {}
    for name, m in bands:
        res['positive_bands'][name] = int(m.sum())
        print(f'    risk {name:<10}{int(m.sum()):>5}  ({100 * m.mean():.1f} %)')

    # what the eligible tail positives look like against the views
    print()
    print('  mean percentile rank of the eligible positives, by band and view '
          '(1.0 = top of the eligible population):')
    dd = d.with_columns([pl.Series(f'pr_{v}', prank(lg(d[f'r_{v}'].to_numpy()))) for v in VIEWS]
                        + [pl.Series('pr_frozen', prank(d['risk_prefix'].to_numpy().astype(float)))])
    hdr = f'    {"band":<12}{"n":>4}{"frozen":>9}' + ''.join(f'{v:>8}' for v in VIEWS)
    print(hdr)
    res['positive_view_ranks'] = {}
    for name, m in bands:
        g = dd.filter(pl.col('label') == 1).filter(pl.Series(m))
        if g.height == 0:
            continue
        vals = {'frozen': float(g['pr_frozen'].mean())} | {v: float(g[f'pr_{v}'].mean()) for v in VIEWS}
        res['positive_view_ranks'][name] = {k: round(v, 4) for k, v in vals.items()}
        print(f'    {name:<12}{g.height:>4}{vals["frozen"]:>9.3f}'
              + ''.join(f'{vals[v]:>8.3f}' for v in VIEWS))

    write_json(VD / 'global_order.json', res)
    check_protected()
    print()
    print(f'written {VD / "global_order.json"}')
    print('DONE')


if __name__ == '__main__':
    main()
