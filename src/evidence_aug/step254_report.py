"""STEP254: assemble hand_oof_step254.parquet, results254.json and the G1 verdict from the arms."""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import json

import numpy as np
import polars as pl

from step254_eval import FAMS, OUT, load

ARMS = ('raw', 'plus', 'mech', 'mechplus', 'perm', 'mechplus_not4', 'mechplus_t4only')
SHORT = {'any': 'any', 'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}


def main() -> None:
    h, _ = load()
    res = {}
    for a in ARMS:
        p = OUT + f'\\results_{a}.json'
        if os.path.exists(p):
            res.update(json.load(open(p, encoding='utf-8')))
    cols = {}
    for a in ARMS:
        f = OUT + f'\\oof_{a}.npy'
        if not os.path.exists(f):
            continue
        o = np.load(f)
        assert o.shape[1] == h.height
        cols[f'{a}_p_any'] = o[0]
        for i, fam in enumerate(FAMS):
            cols[f'{a}_p_{SHORT[fam]}'] = o[i + 1]
        cols[f'{a}_p_fmax'] = o[1:].max(axis=0)
    out = h.select('pair_id', 'hand_id', 'fold', 'listed', 'label',
                   pl.col('behavior_family').alias('family')).with_columns(
        [pl.Series(k, v) for k, v in cols.items()])
    out.write_parquet(OUT + r'\hand_oof_step254.parquet')
    print('wrote hand_oof_step254.parquet', out.shape)

    def ap(arm, fam):
        return res[arm][fam]['ap'] if arm in res else float('nan')

    g = {}
    if 'mechplus' in res and 'plus' in res:
        d_any = ap('mechplus', 'any') - ap('plus', 'any')
        d_ci = ap('mechplus', 'coordinated_isolation') - ap('plus', 'coordinated_isolation')
        d_perm = (ap('mechplus', 'any') - ap('perm', 'any')) if 'perm' in res else float('nan')
        g = {'mechplus_minus_plus_any_AP': round(d_any, 4),
             'mechplus_minus_plus_CI_AP': round(d_ci, 4),
             'mechplus_minus_perm_any_AP': round(d_perm, 4),
             'clause_any_AP_ge_.025': bool(d_any >= .025),
             'clause_CI_AP_ge_.03': bool(d_ci >= .03),
             'clause_perm_ge_half': bool(d_perm >= .5 * d_any),
             'G1': 'PASS' if (d_any >= .025 and d_ci >= .03 and d_perm >= .5 * d_any) else 'FAIL'}
    table = {}
    for a in ARMS:
        if a not in res:
            continue
        table[a] = {SHORT[f]: {k: res[a][f].get(k) for k in
                               ('auc', 'ap', 'tpr@fpr0.0001', 'tpr@fpr0.001', 'tpr@fpr0.01',
                                'invisible_share', 'new_block_gain_share', 'mech_family_gain_share')}
                    for f in ('any', *FAMS)}
        table[a]['_thresholds'] = res[a]['_thresholds']
        table[a]['_runtime_s'] = res[a].get('_runtime_s')
        table[a]['_n_cols'] = res[a].get('_n_cols')
    fx = pl.read_parquet(OUT + r'\step254_features.parquet')
    out_json = {'step': 'STEP254', 'arms': table, 'G1': g,
                'n_mechanical_columns': fx.width - 2,
                'top_gain': {a: res[a]['_top_gain'] for a in ARMS if a in res}}
    json.dump(out_json, open(OUT + r'\results254.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps(g, indent=1))
    for a in table:
        print(a, {k: table[a][k]['ap'] for k in ('any', 'DT', 'SP', 'CI')})


if __name__ == '__main__':
    main()
