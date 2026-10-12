"""Check metric and ensemble behavior without training real-data models."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from map_evidence_experiment import mean_ranks
from evidence_context_experiment import ap5


def test_equal_rank_blend_hand_calculated():
    frame = pl.DataFrame({'pair_id': ['a', 'a', 'a', 'b']})
    result = mean_ranks(frame, np.array([1., 3., 2., 100.]), np.array([9., 9., 0., -50.]))
    np.testing.assert_allclose(result, [3.5/6, 5.5/6, 3/6, 1.])
    np.testing.assert_allclose(mean_ranks(frame, np.array([1., 3., 2., 100.])*10+50,
                                        np.array([9., 9., 0., -50.])*3), result)


def test_competition_ap5_denominator_and_positions():
    assert ap5(['p', 'x', 'q', 'z', 'r'], {'p', 'q', 'r'}) == (1+2/3+3/5)/3
    assert ap5(['x', 'y', 'z', 'v', 'w'], {'p'}) == 0
    assert ap5(['p', 'q', 'r', 's', 't'], set('pqrstu')) == 1
