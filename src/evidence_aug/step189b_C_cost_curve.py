"""STEP189b: POST HOC cost curve on the cached grid. NOT a registered result.

STEP189 is the registered study; this is an engineering addendum written AFTER it read out.
It answers one question only: how much of the BAGS44 gain survives if the production build
uses fewer stage-1 members (the expensive axis, ~2100 s each) or a cheaper stage-2 subset
(the non-bagged configs carry one seed group instead of five)?

Member order is the REGISTRATION order, never an order chosen by score, so the k-member curve
is a cost curve and not a selection path. Every number here is post hoc and is reported as such.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
CV = POKER / 'Claude Version'
OUT = CV / 'experiments/net_ensemble_20260916'
PARTS = ['P0', 'P1651', 'P1652']
INC = 'INC'
S1 = [INC, 'N01_wide_128_32_64', 'N02_narrow_32_8_16', 'N03_wd_1e-1', 'N04_wd_1e-3', 'N05_lr_2e-3',
      'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3', 'N09_two_stage', 'N11_seeds20']
BAGS = ['C00_incumbent', 'A01_ff50_bag', 'A02_ff50_ss80_bag', 'A06_l2reg5_ff30_bag']
NONBAG = ['A03_300t_lr025', 'A04_clf2_l15_d5_300t', 'A05_l3_d2_300t']
# seed groups each configuration costs at production score time (C00 is the 3-seed incumbent)
SEEDS = {'C00_incumbent': 3, 'A01_ff50_bag': 5, 'A02_ff50_ss80_bag': 5, 'A06_l2reg5_ff30_bag': 5,
         'A03_300t_lr025': 1, 'A04_clf2_l15_d5_300t': 1, 'A05_l3_d2_300t': 1}
S2SETS = {
    'BAGS4': BAGS,
    'CHEAP4': ['C00_incumbent'] + NONBAG,
    'CHEAP7': ['C00_incumbent'] + NONBAG + ['A01_ff50_bag', 'A02_ff50_ss80_bag', 'A06_l2reg5_ff30_bag'],
    'C00': ['C00_incumbent'],
    'C00_A01': ['C00_incumbent', 'A01_ff50_bag'],
}


def main():
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    z = np.load(OUT / 'grid186.npz')
    SV = {}
    for k in z.files:
        m, c, n = k.split('~')
        SV[(m, c, n)] = z[k]
    D = S165.load()
    d = D['d']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabd = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabd[q]) for q in qids])
    R = lambda x: round(float(x), 6)

    def evaluate(cells):
        acc = np.zeros(len(qids))
        for n in PARTS:
            s = np.mean([SV[(m, c, n)] for m, c in cells], axis=0)
            _, mm = summarize(key.with_columns(pl.Series('score', D['dec'](s))), truths, fams)
            acc += np.array([mm[q] for q in qids])
        return acc / len(PARTS)

    A0 = evaluate([(INC, 'C00_incumbent')])
    rows = []
    print(f'{"s2set":8s} {"k":>3s} {"cells":>5s} {"seedgrp":>7s} {"delta":>10s} {"h0":>10s} {"h1":>10s}', flush=True)
    for sn, s2 in S2SETS.items():
        sg = sum(SEEDS[c] for c in s2)
        for k in range(1, len(S1) + 1):
            mem = S1[:k]
            A = evaluate([(m, c) for m in mem for c in s2])
            rows.append({'s2set': sn, 'k_members': k, 'n_cells': k * len(s2), 'seed_groups_per_view': sg,
                         'delta': R(A.mean() - A0.mean()), 'half0_delta': R((A - A0)[hv == 0].mean()),
                         'half1_delta': R((A - A0)[hv == 1].mean())})
            print(f'{sn:8s} {k:3d} {k * len(s2):5d} {sg:7d} {A.mean() - A0.mean():+10.6f} '
                  f'{(A - A0)[hv == 0].mean():+10.6f} {(A - A0)[hv == 1].mean():+10.6f}', flush=True)
    dest = OUT / 'results189b_cost_curve.json'
    dest.write_text(json.dumps({'step': '189b', 'status': 'POST HOC, engineering cost curve, not a registered read',
                                'member_order': S1, 'seed_groups': SEEDS, 'rows': rows}, indent=1), encoding='utf-8')
    print('wrote', dest, flush=True)


if __name__ == '__main__':
    main()
