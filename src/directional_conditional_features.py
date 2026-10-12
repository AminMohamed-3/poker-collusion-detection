"""Symmetric actor contrasts of cross-fitted conditional-action residuals."""
import numpy as np
import polars as pl
from run_conditional_policy_screen import hand_features

KEYS = ['pair_id', 'hand_id', 'table_id']


def directional_features(frame, probability, strength_prediction):
    assert set(frame['direction'].unique()).issubset({0, 1})
    parts = []
    for direction in (0, 1):
        mask = frame['direction'].to_numpy() == direction
        part = hand_features(frame.filter(pl.Series(mask)), probability[mask], strength_prediction[mask])
        names = [c for c in part.columns if c.endswith('_mean')]
        parts.append(part.select(KEYS + names).rename({c: f'{c}_{direction}' for c in names}))
    joined = parts[0].join(parts[1], on=KEYS, how='full', coalesce=True, validate='1:1').sort(KEYS)
    both = pl.all_horizontal(pl.col(names[0] + '_0').is_not_null(), pl.col(names[0] + '_1').is_not_null())
    joined = joined.with_columns(both.cast(pl.Float32).alias('cd_both_actors')).with_columns(pl.exclude(KEYS).fill_null(0))
    values = [pl.col('cd_both_actors')]
    for name in names:
        a, b = pl.col(name + '_0'), pl.col(name + '_1')
        suffix = name.removeprefix('ca_').removesuffix('_mean')
        values.extend([
            ((a-b).abs()*pl.col('cd_both_actors')).cast(pl.Float32).alias('cd_' + suffix + '_gap'),
            (a*b*pl.col('cd_both_actors')).cast(pl.Float32).alias('cd_' + suffix + '_product'),
        ])
    for action in (0, 1, 2):
        cross = (pl.col(f'ca_residual_{action}_mean_0')*pl.col('ca_residual_3_mean_1')
                 + pl.col(f'ca_residual_{action}_mean_1')*pl.col('ca_residual_3_mean_0')) / 2
        values.append((cross*pl.col('cd_both_actors')).cast(pl.Float32).alias(f'cd_response_{action}_pressure'))
    result = joined.select(KEYS + values)
    assert len(result.columns) == 39 and np.isfinite(result.select(pl.exclude(KEYS)).to_numpy()).all()
    return result
