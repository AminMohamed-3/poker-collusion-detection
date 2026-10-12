"""STEP248 finalisation: complete the two candidate files and turn the discovery null into an
explicit EXCLUSION BOUND expressed in pairs of a hidden family.

The instrument's recall on each disclosed family (measured on development at the same operating
point) converts the observed excess of hidden-type flagged tail pairs into an upper bound on
the size of a fourth family of comparable dose.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_c_common import ORIGINAL, OUT
from step248_b_events import SCRATCH


def finish_list(name, disc_key, res):
    df = pl.read_parquet(OUT / f'{name}.parquet')
    fh = pl.read_parquet(OUT / 'step248b-flagged-hands.parquet')
    fh = fh.filter(pl.col('pair_id').is_in(df['pair_id'].to_list()))
    agg = (fh.sort(['pair_id', 'ts'])
           .group_by('pair_id', maintain_order=True)
           .agg(pl.col('hand_id').unique(maintain_order=True).alias('flagged_hand_ids'),
                pl.col('event_name').unique().alias('events_fired'),
                pl.col('hand_id').n_unique().alias('n_flagged_hands')))
    df = df.drop('n_hands', strict=False).join(agg, on='pair_id', how='left')
    exp_false = res['tail'][disc_key]
    prec = max(0.0, 1 - exp_false / max(df.height, 1))
    df = df.with_columns(pl.lit(exp_false).alias('expected_false_in_list'),
                         pl.lit(prec).alias('estimated_precision'))
    df.write_parquet(OUT / f'{name}.parquet')
    return {'n': df.height, 'expected_false': exp_false, 'estimated_precision': prec}


def main():
    t0 = time.time()
    res = json.loads((OUT / 'step248b-discovery.json').read_text(encoding='utf-8'))
    pw = json.loads((OUT / 'step248b-power.json').read_text(encoding='utf-8'))
    res['tail']['expected_lowdose'] = (res['tail']['expected_development'] -
                                       res['tail']['expected_development_hidden'])
    out = {'hidden': finish_list('candidates_hidden', 'expected_development_hidden', res),
           'lowdose': finish_list('candidates_lowdose_known', 'expected_lowdose', res)}

    # ---- exclusion bound -----------------------------------------------------------------
    obs = res['tail']['hidden_type']
    exp = res['tail']['expected_development_hidden']
    hi = obs + 1.645 * np.sqrt(max(obs, 1)) - exp          # one-sided 95 % upper bound on excess
    rec = {f: pw['budgets']['30']['recall'][f]['recall']
           for f in pw['budgets']['30']['recall']}
    bound = {f: (hi / r if r > 0 else None) for f, r in rec.items()}
    out['exclusion'] = {'observed_hidden_type': obs, 'expected_hidden_type': exp,
                        'excess': obs - exp, 'excess_upper95': float(hi),
                        'recall_at_budget30': rec,
                        'max_family_size_excluded_95': {k: (float(v) if v else None)
                                                        for k, v in bound.items()}}
    print(f'hidden-type flagged {obs} vs {exp:.1f} expected -> excess {obs-exp:+.1f}, '
          f'95 % upper bound {hi:.1f} pairs')
    for f, r in rec.items():
        print(f'  at {f}-like dose (recall {r:.3f}): a fourth family larger than '
              f'{bound[f]:.0f} pairs is excluded at 95 %')

    # ---- what the two lists look like -------------------------------------------------
    for nm in ('candidates_hidden', 'candidates_lowdose_known'):
        df = pl.read_parquet(OUT / f'{nm}.parquet')
        print(f'\n{nm}: {df.height} pairs, median shared {df["n_shared"].median()}, '
              f'median risk207 {df["risk_score"].median():.4f}, '
              f'share sharing a block player {df["shares_block_player"].mean():.2f}, '
              f'estimated precision {df["estimated_precision"][0]:.2f}')
    (OUT / 'step248-final-summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\ndone  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
