"""Step 67c (plan P1 + P2): re-ledger every pair-side lever on the population the evaluation can
actually contain, and re-estimate the recall tail there.

17 found that the learned blends M / M2 / M3 / M4 beat S on the 60 %-prefix testbed only through
pairs with fewer than 38 shared hands, which the evaluation eligibility rule excludes.  That was
established for M3 alone; P1 does it for every lever built since the base, so the record is
corrected once and the M series can be closed formally rather than by inference.

P2 then re-asks 14.39's question -- how much pair AP is left in the positives that never reach
the block -- on the eligible population only.  13 % of testbed positives sit below 38 shared
hands, so the old tail estimate counted pairs the evaluation cannot present.

Read-only apart from `experiments/viewdis_20260911/eligible_ledger.json`.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import CV, check_protected, write_json
from pu_pairs.step60_multiview_blend import ap, lg
from pu_pairs.step64c_hand_view import VIEWS, DEFS, VD
from pu_pairs.step66_evidence_view import crossfold

RNG = np.random.default_rng(20260912)
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']


def load():
    tb = (pl.read_parquet(VD / 'evidence_view_prefix_rawonly.parquet')
          .sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first')
          .select('pair_id', 'E_top5', 'E_n'))
    mp = pl.read_parquet(VD / 'multiview_prefix.parquet')
    tv = (pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new')
          .unique(subset=['pair_id'], keep='first').drop('fold_new'))
    r1 = (pl.read_parquet(VD / 'prefix_rescoring.parquet').select('pair_id', 'R1_prefix')
          .unique(subset=['pair_id'], keep='first'))
    return (mp.join(tv, on='pair_id', how='left').join(tb, on='pair_id', how='left')
            .join(r1, on='pair_id', how='left').with_columns(pl.col('R1_prefix').fill_null(False)))


def sub30_sd(y, a, b, n=1500):
    """SD of the AP difference on a 30 % subsample -- the public-split resolution of step 67a"""
    k = max(int(round(.3 * len(y))), 10)
    out = []
    for _ in range(n):
        idx = RNG.choice(len(y), k, replace=False)
        if y[idx].sum() < 3:
            continue
        out.append(ap(y[idx], a[idx]) - ap(y[idx], b[idx]))
    return float(np.std(out)) if out else float('nan')


def levers(c):
    """every ordering built since the base, on one class frame"""
    r1 = c['R1_prefix'].to_numpy()
    s_like = lg(c['r_FULL'].to_numpy()) + .5 * lg(c['r_RAW'].to_numpy())
    out = {'frozen': c['risk_prefix'].to_numpy().astype(float)}
    out['C-like (R1 to bottom)'] = np.where(r1, out['frozen'] - 100, out['frozen'])
    out['S-like (FULL+.5RAW, R1 bottom)'] = np.where(r1, s_like - 100, s_like)
    out['D-like (S + RAW<.05 bottom)'] = np.where(c['r_RAW'].to_numpy() < .05,
                                                  out['S-like (FULL+.5RAW, R1 bottom)'] - 50,
                                                  out['S-like (FULL+.5RAW, R1 bottom)'])
    return out, s_like


def blends(c, y):
    m = VIEWS + ['sh']
    designs = {'M (8 views + sh)': m, 'M2 (M + R1)': m + ['R1'],
               'M3 (M2 + SHORT3, ALL5)': m + ['R1', 'SHORT3', 'ALL5'],
               'M4 (M3 + E_top5)': m + ['R1', 'SHORT3', 'ALL5', 'E_top5']}
    out, weights = {}, {}
    for name, fs in designs.items():
        p, (m1, m2), _ = crossfold(c, y, fs)
        out[name] = p
        weights[name] = {v: round(float((a + b) / 2), 3)
                         for v, a, b in zip(fs, m1[0], m2[0])}
    return out, weights


def p1():
    d = load()
    print('=' * 108)
    print('P1.  EVERY PAIR-SIDE LEVER, LEDGERED ON ELIGIBLE ROWS (shared_hands >= 38, as evaluation)')
    print('=' * 108)
    out = {}
    for dn, (posset, negset) in DEFS.items():
        c0 = d.filter((pl.col('risk_prefix') >= .5) & pl.col('cls').is_in(posset + negset)
                      & pl.col('E_top5').is_not_null() & pl.col('r_SHORT3').is_not_null())
        for tag, cut in [('all rows', 0), ('shared >= 38', 38), ('shared >= 50', 50)]:
            c = c0.filter(pl.col('shared_hands') >= cut)
            y = c['cls'].is_in(posset).to_numpy().astype(float)
            ords, _ = levers(c)
            bl, w = blends(c, y)
            ords.update(bl)
            ref = ords['S-like (FULL+.5RAW, R1 bottom)']
            print()
            print(f'  def {dn}, {tag}:  n={c.height}  positives={int(y.sum())}')
            print(f'    {"lever":<34}{"AP":>9}{"vs S-like":>11}{"sub30 SD":>10}')
            row = {'n': int(c.height), 'n_pos': int(y.sum()), 'levers': {}}
            for name, s in ords.items():
                a = ap(y, s)
                dl = a - ap(y, ref)
                sd = sub30_sd(y, s, ref) if name != 'S-like (FULL+.5RAW, R1 bottom)' else 0.0
                row['levers'][name] = {'AP': round(a, 4), 'vs_S': round(dl, 4), 'sub30_sd': round(sd, 4)}
                print(f'    {name:<34}{a:>9.4f}{dl:>+11.4f}{sd:>10.4f}')
            if cut == 38:
                row['weights'] = w
                r1 = c['R1_prefix'].to_numpy()
                row['R1_on_eligible'] = {'flagged': int(r1.sum()),
                                         'of_which_positive': int(y[r1].sum()),
                                         'precision_as_FP_rule': float(1 - y[r1].mean()) if r1.sum() else None}
                print(f'    R1 on eligible rows: {int(r1.sum())} flagged, {int(y[r1].sum())} of them '
                      f'positive (precision as an FP rule '
                      f'{1 - y[r1].mean() if r1.sum() else float("nan"):.3f})')
                print(f'    standardised weight on log shared_hands: '
                      + ', '.join(f'{k.split()[0]} {v["sh"]:+.2f}' for k, v in w.items()))
            out[f'def {dn} {tag}'] = row
    return out


def p2(d=None):
    d = load() if d is None else d
    print()
    print('=' * 108)
    print('P2.  THE RECALL TAIL ON ELIGIBLE PAIRS')
    print('=' * 108)
    pos = d.filter(pl.col('label') == 1)
    el = pos.filter(pl.col('shared_hands') >= 38)
    out = {'positives_all': pos.height, 'positives_eligible': el.height}
    print(f'  labelled positives in the prefix testbed: {pos.height}; eligible (>= 38): {el.height}'
          f'  ({100 * (1 - el.height / pos.height):.0f} % were ineligible)')
    print()
    print(f'{"family":<26}{"eligible":>10}{"risk<.5":>9}{"[.1,.5)":>9}{"<.1":>7}{"tail %":>9}')
    for f in FAMS + ['ALL']:
        g = el if f == 'ALL' else el.filter(pl.col('behavior_family') == f)
        if g.height == 0:
            continue
        r = g['risk_prefix'].to_numpy()
        lo, mid = int((r < .1).sum()), int(((r >= .1) & (r < .5)).sum())
        out[f] = {'n': g.height, 'below_.5': lo + mid, 'band_.1_.5': mid, 'below_.1': lo}
        print(f'{f:<26}{g.height:>10}{lo + mid:>9}{mid:>9}{lo:>7}{100 * (lo + mid) / g.height:>8.1f}%')

    # what the tail costs: AP with the tail present, against the block alone
    print()
    print('  AP cost of the tail (def A classes, eligible rows only):')
    posset, negset = DEFS['A']
    base = d.filter(pl.col('cls').is_in(posset + negset) & (pl.col('shared_hands') >= 38)
                    & pl.col('E_top5').is_not_null() & pl.col('r_SHORT3').is_not_null())
    print(f'    {"population":<40}{"n":>7}{"pos":>6}{"AP frozen":>11}{"AP S-like":>11}')
    res = {}
    for tag, floor in [('block only (risk >= .5)', .5), ('floor .1', .1), ('floor 0 (all eligible)', -1)]:
        c = base.filter(pl.col('risk_prefix') >= floor) if floor >= 0 else base
        # add back the eligible tail positives that the class filter dropped
        tail = d.filter((pl.col('label') == 1) & (pl.col('shared_hands') >= 38)
                        & (pl.col('risk_prefix') < .5) & (pl.col('risk_prefix') >= max(floor, 0.0)))
        c = pl.concat([c, tail.select(c.columns)]).unique(subset=['pair_id'], keep='first')
        y = (c['cls'].is_in(posset) | (c['label'] == 1)).to_numpy().astype(float)
        ords, _ = levers(c)
        a_f = ap(y, ords['frozen'])
        a_s = ap(y, ords['S-like (FULL+.5RAW, R1 bottom)'])
        res[tag] = {'n': int(c.height), 'n_pos': int(y.sum()), 'AP_frozen': round(a_f, 4),
                    'AP_S_like': round(a_s, 4)}
        print(f'    {tag:<40}{c.height:>7}{int(y.sum()):>6}{a_f:>11.4f}{a_s:>11.4f}')
    out['tail_cost'] = res
    cost = res['block only (risk >= .5)']['AP_S_like'] - res['floor 0 (all eligible)']['AP_S_like']
    print(f'\n  the eligible tail costs {cost:+.4f} AP under the S order '
          f'-> {"NOT a lever; the pair side is closed at S" if cost < .01 else "worth attacking"}')
    out['tail_cost_AP'] = cost
    return out


def main():
    check_protected()
    a = p1()
    b = p2()
    write_json(VD / 'eligible_ledger.json', {'P1': a, 'P2': b})
    check_protected()
    print()
    print(f'written {VD / "eligible_ledger.json"}')
    print('DONE')


if __name__ == '__main__':
    main()
