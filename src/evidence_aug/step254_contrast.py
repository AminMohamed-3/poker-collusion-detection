"""STEP254 diagnostics: what do the mechanical columns that the model uses actually say?

For every mechanical column that appears in a top-30 split-gain list of the MECHPLUS arm:
  * univariate AUC, listed hands of each family against ALL hands of the confirmed-negative pairs;
  * the mean on listed hands of each family, on the INVISIBLE listed hands of that family
    (p_fmax of the `plus` arm below its honest q99) and on honest hands;
  * the same means restricted to honest hands re-weighted to nothing (raw means, stated as such).
Descriptive only - no arm is selected here. Writes contrast254.json.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import json

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from step254_eval import FAMS, OUT

SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}


def main() -> None:
    res = json.load(open(OUT + r'\results_mechplus.json', encoding='utf-8'))['mechplus']
    fx = pl.read_parquet(OUT + r'\step254_features.parquet')
    oof = pl.read_parquet(OUT + r'\hand_oof_step254.parquet')
    mech = set(c for c in fx.columns if c not in ('pair_id', 'hand_id'))
    top: list[str] = []
    for name in ('any', *FAMS):
        for c, _ in res['_top_gain'][name]:
            if c in mech and c not in top:
                top.append(c)
    d = oof.join(fx, on=['pair_id', 'hand_id'], how='left')
    assert d.height == oof.height
    listed = d['listed'].to_numpy() == 1
    lbl = d['label'].to_numpy()
    fam = d['family'].to_numpy()
    honest = lbl == 0
    pmax_plus = d['plus_p_fmax'].to_numpy()
    q99 = float(np.quantile(pmax_plus[honest], .99))
    out = {'honest_q99_plus_arm': round(q99, 5), 'n_columns': len(top), 'columns': {}}
    for c in top:
        v = d[c].to_numpy().astype(float)
        row = {'mean_honest': round(float(v[honest].mean()), 4)}
        for f in FAMS:
            sel = listed & (fam == f)
            inv = sel & (pmax_plus < q99)
            m = np.concatenate([v[sel], v[honest]])
            y = np.concatenate([np.ones(sel.sum()), np.zeros(honest.sum())])
            row[SHORT[f]] = {
                'auc_listed_vs_honest': round(float(roc_auc_score(y, m)), 4),
                'mean_listed': round(float(v[sel].mean()), 4),
                'mean_invisible_listed': round(float(v[inv].mean()), 4) if inv.sum() else None,
                'n_invisible': int(inv.sum())}
        out['columns'][c] = row
    json.dump(out, open(OUT + r'\contrast254.json', 'w', encoding='utf-8'), indent=1)
    rank = sorted(top, key=lambda c: -max(out['columns'][c][SHORT[f]]['auc_listed_vs_honest']
                                          for f in FAMS))
    print(f'{"column":42s} {"honest":>8s} {"DT auc":>7s} {"SP auc":>7s} {"CI auc":>7s}')
    for c in rank[:30]:
        r = out['columns'][c]
        print(f'{c:42s} {r["mean_honest"]:8.3f} ' +
              ' '.join(f'{r[SHORT[f]]["auc_listed_vs_honest"]:7.3f}' for f in FAMS))


if __name__ == '__main__':
    main()
