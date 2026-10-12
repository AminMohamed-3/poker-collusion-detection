"""Independent checks of the proposed pair-local transforms."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from relative_evidence_experiment import relative_context


def test_hand_calculated_ranks_ties_and_singleton():
    frame = pl.DataFrame({'pair_id': ['a']*4+['b'], 'x': [0., 0., 2., 6., 9.]})
    out, _ = relative_context(frame, ['x'])
    np.testing.assert_allclose(out['percentile_x'], [.25, .25, .625, .875, .5])
    np.testing.assert_allclose(out['context_x'][:4], (np.array([0., 0., 2., 6.])-2)/np.std([0., 0., 2., 6.], ddof=1), rtol=1e-6)
    assert out['context_x'][4] == 0


def test_order_and_identifier_invariance():
    frame = pl.DataFrame({'pair_id': ['a']*3+['b']*2, 'hand_id': list('12345'), 'x': [2., 4., 9., 10., 30.]})
    out, _ = relative_context(frame, ['x'])
    reversed_out, _ = relative_context(frame.reverse(), ['x'])
    assert out.sort('hand_id').equals(reversed_out.sort('hand_id'))
    renamed, _ = relative_context(frame.with_columns(pl.col('pair_id').replace({'a':'zzz', 'b':'q'})), ['x'])
    assert out.drop('pair_id').equals(renamed.drop('pair_id'))


def test_other_pool_cannot_change_pair_context():
    frame = pl.DataFrame({'pair_id': ['a']*3+['b']*2, 'x': [2., 4., 9., 10., 30.]})
    out, _ = relative_context(frame, ['x'])
    altered, _ = relative_context(frame.with_columns(pl.when(pl.col('pair_id')=='b').then(1e9).otherwise(pl.col('x')).alias('x')), ['x'])
    assert out.filter(pl.col('pair_id')=='a').equals(altered.filter(pl.col('pair_id')=='a'))


def test_constant_feature_is_finite_and_neutral():
    frame = pl.DataFrame({'pair_id': ['a']*3, 'v2_x': [4., 4., 4.]})
    out, extra = relative_context(frame, ['v2_x'])
    np.testing.assert_allclose(out['context_v2_x'], 0)
    np.testing.assert_allclose(out['percentile_v2_x'], .5)
    assert extra == ['percentile_v2_x']
