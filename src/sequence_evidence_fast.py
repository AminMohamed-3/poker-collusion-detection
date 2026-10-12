"""Vectorized implementation of the frozen 23 scalar sequence features."""
from __future__ import annotations
import polars as pl
from sequence_evidence_features import NAMES


def extract(keys, hands, seats, actions):
    group = ['pair_id', 'hand_id']
    street_group = group+['street']
    assert keys.select(pl.struct(group).n_unique()).item() == keys.height
    assert hands['hand_id'].n_unique() == hands.height
    assert seats.select(pl.struct('hand_id','player_id').n_unique()).item() == seats.height
    assert actions.select(pl.struct('hand_id','action_no').n_unique()).item() == actions.height
    keys = keys.select(*group, 'player_1', 'player_2')
    summary = seats.group_by('hand_id').agg(pl.len().alias('seat_count'),
        pl.col('total_contribution').sum().alias('all_contribution'), pl.col('net_chips').sum().alias('all_net'))
    folds = actions.filter(pl.col('action') == 'fold').group_by('hand_id', 'player_id').agg(pl.col('action_no').min().alias('fold_no'))
    meta = keys.join(hands.select('hand_id','big_blind'), on='hand_id', validate='m:1').join(summary, on='hand_id', validate='m:1')
    for i in [1,2]:
        seat = seats.select('hand_id','player_id','total_contribution','net_chips').rename({
            'player_id':f'player_{i}', 'total_contribution':f'contribution_{i}', 'net_chips':f'net_{i}'})
        fold = folds.rename({'player_id':f'player_{i}', 'fold_no':f'fold_{i}'})
        meta = meta.join(seat, on=['hand_id',f'player_{i}'], how='left', validate='m:1').join(
            fold, on=['hand_id',f'player_{i}'], how='left', validate='m:1')
    assert not meta.select(pl.col(['contribution_1','contribution_2','big_blind']).is_null().any()).row(0).count(True)
    meta = meta.with_columns(pl.col(['fold_1','fold_2']).fill_null(10**15))
    frame = actions.join(meta, on='hand_id').sort(*group, 'action_no')
    frame = frame.with_columns(
        pl.when(pl.col('player_id')==pl.col('player_1')).then(1).when(pl.col('player_id')==pl.col('player_2')).then(2).otherwise(0).alias('_role'),
        (pl.col('action').is_in(['bet','raise','all_in']) & (pl.col('amount')>pl.col('to_call'))).alias('_agg'))
    frame = frame.with_columns(
        ((pl.col('_role')==0) & (pl.col('action')=='fold')).cast(pl.Int32).alias('_out_fold'),
        ((pl.col('_role')==1) & pl.col('_agg')).cast(pl.Int32).alias('_raise_1'),
        ((pl.col('_role')==2) & pl.col('_agg')).cast(pl.Int32).alias('_raise_2'),
        pl.when(pl.col('_role')==1).then(pl.col('action_no')<pl.col('fold_2')).otherwise(pl.col('action_no')<pl.col('fold_1')).alias('_partner_live'),
        (pl.col('action').is_in(['call','all_in']) & ~pl.col('_agg')).alias('_call'))
    frame = frame.with_columns(
        pl.when('_agg').then('_role').otherwise(None).forward_fill().shift(1).over(street_group).fill_null(-1).alias('_previous'),
        pl.col('_out_fold').cum_sum().shift(1).over(group).fill_null(0).alias('_out_folds_before'),
        pl.col('_out_fold').cum_sum().shift(1).over(street_group).fill_null(0).alias('_street_out_folds_before'),
        pl.col('_raise_1').cum_sum().shift(1).over(street_group).fill_null(0).alias('_r1_before'),
        pl.col('_raise_2').cum_sum().shift(1).over(street_group).fill_null(0).alias('_r2_before'),
        pl.col('amount').cum_sum().shift(1).over(group+['player_id']).fill_null(0).alias('_invested'))
    role, previous = pl.col('_role'), pl.col('_previous')
    outsider, pair = role==0, role>0
    both = (pl.col('_r1_before')>0) & (pl.col('_r2_before')>0)
    outsiders_live = pl.col('seat_count')-2-pl.col('_out_folds_before')
    to_pair = previous>0
    to_partner = (previous == 3-role) & pl.col('_partner_live') & pair
    fold, call, agg = pl.col('action')=='fold', pl.col('_call'), pl.col('_agg')
    live = pl.col('_partner_live')
    cost = pl.col('amount')/pl.col('big_blind')
    invested = pl.col('_invested')/pl.col('big_blind')
    out_fold_pair = outsider & fold & to_pair
    pair_fold_partner = pair & fold & to_partner
    pair_raise_multi = pair & agg & live & (outsiders_live>0)
    partner_raised = pl.when(role==1).then(pl.col('_r2_before')>0).otherwise(pl.col('_r1_before')>0)
    def total(condition, value=pl.lit(1.)):
        return pl.when(condition).then(value).otherwise(0.).sum()
    expressions = {
        'outsider_fold_to_pair_n': total(out_fold_pair),
        'outsider_fold_to_pair_invested_bb': total(out_fold_pair, invested),
        'outsider_call_to_pair_bb': total(outsider & call & to_pair, cost),
        'outsider_raise_to_pair_n': total(outsider & agg & to_pair),
        'outsider_fold_after_both_raise_n': total(out_fold_pair & both),
        'outsider_fold_after_both_raise_invested_bb': total(out_fold_pair & both, invested),
        'outsider_call_after_both_raise_bb': total(outsider & call & to_pair & both, cost),
        'pair_reraise_multiway_n': total(pair_raise_multi & to_partner),
        'pair_raise_after_partner_multiway_n': total(pair_raise_multi & partner_raised),
        'pair_fold_after_outsider_fold_n': total(pair_fold_partner & (pl.col('_street_out_folds_before')>0)),
        'pair_fold_after_outsider_fold_invested_bb': total(pair_fold_partner & (pl.col('_street_out_folds_before')>0), invested),
        'pair_fold_after_both_raise_n': total(pair_fold_partner & both),
        'pair_fold_after_both_raise_invested_bb': total(pair_fold_partner & both, invested),
        'pair_raise_after_outsiders_gone_n': total(pair & agg & live & (outsiders_live==0)),
        'partner_call_after_outsiders_gone_bb': total(pair & call & to_partner & (outsiders_live==0), cost),
        'max_pair_raise_pot_fraction_multiway': pl.when(pair_raise_multi).then(pl.col('amount')/pl.max_horizontal(pl.col('pot_before')+pl.col('to_call'), 'big_blind')).otherwise(0.).max(),
        'max_outsider_fold_odds_after_both_raise': pl.when(out_fold_pair & both).then(pl.col('to_call')/(pl.col('pot_before')+pl.col('to_call')).clip(lower_bound=1)).otherwise(0.).max(),
    }
    events = frame.group_by(group).agg(*[expr.alias('seq_'+name) for name, expr in expressions.items()])
    check = (pl.col('action')=='check') & pair & live & (outsiders_live==0)
    streets = frame.group_by(street_group).agg(
        pl.col('_raise_1').sum().alias('_r1'), pl.col('_raise_2').sum().alias('_r2'),
        role.filter(check).n_unique().alias('_checking_members'))
    streets = streets.group_by(group).agg(
        (pl.col('_checking_members')==2).sum().alias('seq_heads_up_mutual_check_streets'),
        ((pl.col('_r1')>0) & (pl.col('_r2')>0)).sum().alias('seq_both_pair_raise_streets'),
        pl.min_horizontal('_r1','_r2').max().alias('seq_max_same_street_min_pair_raises'))
    result = meta.join(events, on=group, how='left').join(streets, on=group, how='left').with_columns(
        ((pl.col('all_contribution')-pl.col('contribution_1')-pl.col('contribution_2'))/pl.col('big_blind')).alias('seq_outsiders_contribution_bb'),
        ((pl.col('net_1')+pl.col('net_2')-pl.col('all_net')).clip(lower_bound=0)/pl.col('big_blind')).alias('seq_outsiders_net_loss_bb'),
        pl.when(pl.col('seq_both_pair_raise_streets')>0).then((pl.col('net_1')+pl.col('net_2')).clip(lower_bound=0)/pl.col('big_blind')).otherwise(0.).alias('seq_pair_profit_after_both_raise_bb'))
    features = ['seq_'+n for n in NAMES]
    result = result.select(*group, *features).with_columns(pl.col(features).fill_null(0).cast(pl.Float32))
    assert result.height == keys.height
    assert result.select(pl.col(features).is_finite().all()).row(0) == (True,)*len(features)
    return result
