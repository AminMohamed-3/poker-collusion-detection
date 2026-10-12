"""Check actor symmetry, missing-actor semantics and action interaction signs."""
import numpy as np
import polars as pl
from directional_conditional_features import directional_features


def fixture(actions=(0, 3), directions=(0, 1)):
    count=len(actions)
    frame=pl.DataFrame({'pair_id':['p']*count,'hand_id':['h']*count,'table_id':['t']*count,
        'direction':directions,'action_class':actions,'partner_strength':[.6]*count,
        'facing_partner':[1]*count,'own_strength':[.7]*count,'price':[.2]*count,
        'observed_log_amount_bb':[1.0]*count,'players_active':[3]*count,
        'street_index':[0]*count,'partner_prior_actions':[0]*count})
    return frame,np.full((count,4),.25),np.full(count,.5)


def test_actor_swap_and_action_order_invariance():
    frame,probability,strength=fixture()
    result=directional_features(frame,probability,strength)
    changed=frame.reverse().with_columns((1-pl.col('direction')).alias('direction'))
    assert result.equals(directional_features(changed,probability[::-1],strength[::-1]))
    repeated=pl.concat([frame,frame])
    assert result.equals(directional_features(repeated,np.tile(probability,(2,1)),np.tile(strength,2)))


def test_missing_actor_is_not_treated_as_an_observed_neutral_action():
    frame,probability,strength=fixture((0,3),(0,0))
    result=directional_features(frame,probability,strength)
    assert np.all(result.select(pl.exclude(['pair_id','hand_id','table_id'])).to_numpy()==0)


def test_opposed_and_matching_choices_have_different_interaction_signs():
    opposed=directional_features(*fixture((0,3)))
    matching=directional_features(*fixture((0,0)))
    assert opposed['cd_response_0_pressure'].item()>0
    assert matching['cd_response_0_pressure'].item()<0
    assert opposed['cd_residual_0_product'].item()<0
    assert matching['cd_residual_0_product'].item()>0
