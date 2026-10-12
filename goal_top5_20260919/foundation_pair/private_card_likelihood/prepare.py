"""Prepare immutable memberships for the nested partner-private likelihood test; fits nothing."""
from __future__ import annotations
import hashlib, json, site
from pathlib import Path
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import polars as pl

HERE = Path(__file__).resolve().parent
POKER = HERE.parents[2]
CACHE = POKER / "reports/conditional-action-experiment/action-cache"
LABELS = POKER / "data/development_labels.csv"
SALT = "private-card-likelihood-v1-35935"
INPUTS = [
    'street_index','own_strength','own_high_rank','own_low_rank','own_pair','own_suited',
    'own_flush_count','own_straight_window','board_high_rank','board_pairs','board_flush_count',
    'position_from_button','partner_position_from_button','price','log_pot_bb','log_stack_bb',
    'players_active','prior_actions','own_prior_actions','own_prior_aggression',
    'own_street_aggression','log_own_prior_amount_bb','partner_prior_actions',
    'partner_prior_aggression','partner_street_aggression','partner_last_action',
    'log_partner_prior_amount_bb','outsider_prior_aggression','facing_partner',
    'own_style_fold','own_style_check','own_style_call','own_style_raise',
    'partner_style_fold','partner_style_check','partner_style_call','partner_style_raise']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bit(value, suffix): return int(hashlib.sha256(f"{SALT}|{suffix}|{value}".encode()).hexdigest(), 16) & 1
def order(pair, direction, hand, action):
    return hashlib.sha256(f"{SALT}|cap|{pair}|{direction}|{hand}|{action}".encode()).hexdigest()

def main():
    paths = sorted(CACHE.glob("*.parquet")); assert len(paths) == 20
    data = pl.read_parquet(paths)
    required = {'pair_id','hand_id','table_id','direction','action_no','action_class','partner_strength',*INPUTS}
    assert required <= set(data.columns) and data.null_count().sum_horizontal().item() == 0
    assert data.select(pl.struct('pair_id','direction','hand_id','action_no').n_unique()).item() == data.height
    # Extractor emits a row only while partner is live; direction is actor orientation.
    tables = sorted(data['table_id'].unique().to_list())
    split = pl.DataFrame({'table_id':tables}).with_columns(
        pl.Series('outer_fold',[bit(t,'outer') for t in tables]),
        pl.Series('inner_fold_outer0',[bit(t,'inner0') for t in tables]),
        pl.Series('inner_fold_outer1',[bit(t,'inner1') for t in tables]))
    assert split.group_by('outer_fold').len().height == 2
    split.write_parquet(HERE/'table_membership.parquet')
    keys = data.select('pair_id','direction','hand_id','action_no','table_id').with_columns(
        pl.Series('_ord',[order(*r[:4]) for r in data.select('pair_id','direction','hand_id','action_no').iter_rows()]))
    train_keys = keys.sort('pair_id','direction','_ord').group_by('pair_id','direction',maintain_order=True).head(128).drop('_ord')
    train_keys.write_parquet(HERE/'capped_training_keys.parquet')
    labels = pl.read_csv(LABELS).select('pair_id','player_1','player_2','label','label_status','behavior_family').join(
        data.group_by('pair_id').agg(pl.col('table_id').first()),on='pair_id',validate='1:1')
    # Recover actor identity from the documented direction convention. It is grouping only.
    actor_map = labels.select('pair_id','player_1','player_2')
    keyed = data.select('pair_id','direction','hand_id','action_no','table_id').join(actor_map,on='pair_id',validate='m:1').with_columns(
        pl.when(pl.col('direction')==0).then(pl.col('player_1')).otherwise(pl.col('player_2')).alias('actor_id')).drop('player_1','player_2')
    assert keyed.select(pl.struct('pair_id','direction','hand_id','action_no').n_unique()).item()==keyed.height
    physical = keyed.group_by('hand_id','action_no','actor_id').agg(pl.len().alias('context_count'))
    raw = pl.scan_parquet(POKER/'data/actions.parquet').select('hand_id','action_no','player_id').collect()
    verified = keyed.join(raw,on=['hand_id','action_no'],validate='m:1')
    assert verified.filter(pl.col('actor_id') != pl.col('player_id')).height == 0
    assert verified.group_by('actor_id').agg(pl.col('table_id').n_unique().alias('n')).filter(pl.col('n')>1).height == 0
    keyed = keyed.join(physical,on=['hand_id','action_no','actor_id'],validate='m:1').with_columns((1/pl.col('context_count')).alias('physical_action_weight'))
    keyed.write_parquet(HERE/'row_membership.parquet')
    labels.write_parquet(HERE/'evaluator_labels.parquet')
    stats = {
      'cache_rows':data.height,'pairs':data['pair_id'].n_unique(),'hands':data['hand_id'].n_unique(),
      'tables':len(tables),'directions':data.group_by('direction').len().sort('direction').to_dicts(),
      'physical_actor_actions':physical.height,'max_contexts_per_physical_action':physical['context_count'].max(),
      'raw_actor_verification_rows':verified.height,'actors_spanning_tables':0,
      'classes':data.group_by('action_class').len().sort('action_class').to_dicts(),
      'capped_training_rows':train_keys.height,
      'folds':data.join(split,on='table_id').group_by('outer_fold').agg(pl.len().alias('actions'),pl.col('pair_id').n_unique().alias('pairs'),pl.col('table_id').n_unique().alias('tables')).sort('outer_fold').to_dicts()}
    manifest={'protocol_version':1,'salt':SALT,'inputs':INPUTS,'action_cohort':'All cached development action rows. The extractor emits actor decisions only while the paired partner is still live. Pair labels do not select training rows. Both models use the identical deterministic cap of 128 actions per directed pair in each applicable training pool; held scoring uses every eligible row.',
      'sources':{str(p.relative_to(POKER)).replace('\\','/'):sha(p) for p in [*paths,LABELS,POKER/'Claude Version/src/conditional_action_features.py',POKER/'Claude Version/src/build_conditional_action_cache.py']},
      'artifacts':{},'stats':stats}
    for name in ['table_membership.parquet','capped_training_keys.parquet','row_membership.parquet','evaluator_labels.parquet','PROTOCOL.md','prepare.py','fit.py','evaluate.py']:
        manifest['artifacts'][name]=sha(HERE/name)
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
