"""Read-only serialized-data audit and training-only source-discrimination test."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_features(h,s,a):
    af=a.group_by('hand_id').agg(pl.len().alias('n_actions'),*[pl.col('action').eq(k).mean().alias('action_'+k) for k in ['fold','check','call','bet','raise','all_in']],(pl.col('amount')/pl.col('stack_before').clip(1)).mean().alias('mean_stack_fraction'))
    sf=s.group_by('hand_id').agg(pl.col('starting_stack').mean().alias('mean_stack'),pl.col('went_to_showdown').mean().alias('showdown_fraction'))
    return h.select('hand_id','table_id','big_blind','final_pot',pl.col('board_cards').fill_null('').str.count_matches(r'[2-9TJQKA][cdhs]').alias('board_count')).join(af,on='hand_id').join(sf,on='hand_id').with_columns((pl.col('final_pot')/pl.col('big_blind')).alias('pot_bb'),(pl.col('mean_stack')/pl.col('big_blind')).alias('stack_bb')).drop('big_blind','final_pot','mean_stack')

def distributions(f):
    cols=[c for c in f.columns if c not in ['hand_id','table_id']]
    return {'hands':f.height,'pools':f['table_id'].n_unique(),'board_counts':f.group_by('board_count').len().sort('board_count').to_dicts(),'means':{c:float(f[c].mean()) for c in cols},'quantiles':{c:{str(q):float(f[c].quantile(q)) for q in [0,.1,.5,.9,1]} for c in ['n_actions','pot_bb','stack_bb']}}

def main():
    plan=json.loads((HERE/'pilot-plan.json').read_text()); generation=json.loads((HERE/'pilot-generation.json').read_text())
    assert generation['plan_sha256']==sha(HERE/'pilot-plan.json')
    hh=[];ss=[];aa=[];ll=[];checks=[];reason_total=Counter();valid_cells=Counter();omitted=[]
    critical={'directed_transfer':{'donor_calls_partner_while_behind','donor_folds_stronger_hand'},'soft_play':{'partner_check','partner_raise_suppressed'},'coordinated_isolation':{'pressure_outsiders'}}
    for r in generation['pools']:
        d=HERE/'pilot'/f"pool-{r['pool_id']:04d}"
        assert all(sha(d/p)==v for p,v in r['files'].items())
        h=pl.read_parquet(d/'hands.parquet').with_columns(pl.col('started_at').cast(pl.String));s=pl.read_parquet(d/'seats.parquet');a=pl.read_parquet(d/'actions.parquet');labels=pl.read_csv(d/'development_labels.csv');pr=json.loads((d/'provenance.json').read_text());tr=json.loads((d/'interventions.json').read_text())
        hh.append(h);ss.append(s);aa.append(a);ll.append(labels)
        assert h.height==3000 and h['hand_id'].n_unique()==3000 and s.height==18000
        assert a.select(pl.struct('hand_id','action_no').n_unique()).item()==a.height
        balances=s.group_by('hand_id').agg(pl.col('net_chips').sum().alias('net'),pl.col('total_contribution').sum().alias('contributed'),pl.col('won_share').sum().alias('shares')).join(h.select('hand_id','final_pot'),on='hand_id')
        assert balances.filter((pl.col('net')!=0)|(pl.col('contributed')!=pl.col('final_pot'))|((pl.col('shares')-1).abs()>1e-8)).height==0
        assert a.filter((pl.col('amount')<0)|(pl.col('amount')>pl.col('stack_before'))|(pl.col('to_call')<0)).height==0
        assert s.filter((pl.col('starting_stack')+pl.col('net_chips')<0)|(pl.col('total_contribution')>pl.col('starting_stack'))).height==0
        reasons=Counter(t['reason'] for row in tr for t in row['interventions']);reason_total.update(reasons)
        mechanism=sum(reasons[k] for k in critical.get(r['behavior'],set()))
        valid=(r['behavior']=='none' or mechanism>0)
        if valid: valid_cells[(r['policy_family'],r['behavior'])]+=1
        if pr['known_positive_omitted_for_no_intervention']: omitted.append(r['pool_id'])
        checks.append({'pool_id':r['pool_id'],'policy_family':r['policy_family'],'behavior':r['behavior'],'labelled_pairs':r['pairs'],'intervention_hands':r['intervention_hands'],'mechanism_actions':mechanism,'mechanism_verified':valid,'reasons':dict(reasons),'minimum_shared_hands':min(pr['shared_hands'].values())})
    h=pl.concat(hh);s=pl.concat(ss);a=pl.concat(aa);labels=pl.concat(ll)
    assert h['hand_id'].n_unique()==h.height
    cards=s.group_by('hand_id').agg(pl.concat_list('hole_card_1','hole_card_2').flatten().alias('hole_cards')).join(h.select('hand_id','board_cards'),on='hand_id')
    card_errors=0
    for row in cards.iter_rows(named=True):
        cs=row['hole_cards']+row['board_cards'].split()
        card_errors+=len(cs)!=len(set(cs))
    assert card_errors==0
    # Independent payload fingerprint excludes synthetic identifiers and timestamps.
    hand_seats=s.sort('hand_id','seat_no').group_by('hand_id',maintain_order=True).agg(pl.struct(pl.exclude('hand_id','player_id')).alias('payload'))
    payloads=hand_seats.join(h.select('hand_id','board_cards','big_blind'),on='hand_id')
    hashes=[hashlib.sha256(json.dumps([r['payload'],r['board_cards'],r['big_blind']],sort_keys=True).encode()).hexdigest() for r in payloads.iter_rows(named=True)]
    duplicate_payloads=len(hashes)-len(set(hashes));assert duplicate_payloads==0
    synthetic=make_features(h,s,a)
    calibration=json.loads((HERE/'training-calibration.json').read_text());pools=calibration['training_pools']
    oh=pl.read_parquet(ROOT/'data/hands.parquet').filter((pl.col('phase')=='development')&pl.col('table_id').is_in(pools)).with_columns(pl.col('started_at').cast(pl.String))
    # Stratified by pool, deterministic uniform hand samples; no held or evaluation rows.
    oh=oh.with_columns(pl.col('hand_id').hash(seed=908117).alias('_order')).sort('table_id','_order').group_by('table_id',maintain_order=True).head(100).drop('_order')
    ids=oh.select('hand_id').lazy()
    os_=pl.scan_parquet(ROOT/'data/seats.parquet').join(ids,on='hand_id',how='inner').collect()
    oa=pl.scan_parquet(ROOT/'data/actions.parquet').join(ids,on='hand_id',how='inner').collect()
    original=make_features(oh,os_,oa)
    sampled=synthetic.with_columns(pl.col('hand_id').hash(seed=908117).alias('_order')).sort('table_id','_order').group_by('table_id',maintain_order=True).head(100).drop('_order')
    features=[c for c in original.columns if c not in ['hand_id','table_id']]
    x=np.concatenate([original.select(features).to_numpy(),sampled.select(features).to_numpy()]);y=np.concatenate([np.zeros(original.height),np.ones(sampled.height)]);groups=np.concatenate([original['table_id'].to_numpy(),sampled['table_id'].to_numpy()]);pred=np.full(len(y),np.nan);folds=[]
    for i,(train,test) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=908117).split(x,y,groups)):
        assert not set(groups[train])&set(groups[test])
        m=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=100,learning_rate=.05,class_weight='balanced',n_jobs=2,verbosity=-1,random_state=908117+i)
        m.fit(x[train],y[train]);pred[test]=m.predict_proba(x[test])[:,1];folds.append(float(roc_auc_score(y[test],pred[test])))
    auc=float(roc_auc_score(y,pred));pre0=original.filter(pl.col('board_count')==0).height/original.height;pre1=synthetic.filter(pl.col('board_count')==0).height/synthetic.height
    domain_flag=auc>.8 or abs(pre0-pre1)>.1
    cell_pass=len(valid_cells)==16
    result={'plan_sha256':sha(HERE/'pilot-plan.json'),'audit_source_sha256':sha(__file__),'generated_hands':h.height,'players':s['player_id'].n_unique(),'labelled_pairs':labels.height,'positives':labels['label'].sum(),'label_counts':labels.group_by('behavior_family').len().to_dicts(),'pool_checks':checks,'zero_intervention_targets_omitted':omitted,'mechanism_verified_cells':len(valid_cells),'duplicate_hand_payloads':duplicate_payloads,'card_errors':card_errors,'serialized_accounting_checks_passed':True,'reasons':dict(reason_total),'synthetic':distributions(synthetic),'original_training_sample':distributions(original),'domain_features':features,'domain_group_oof_auc':auc,'domain_fold_auc':folds,'preflop_ending_fraction_original':pre0,'preflop_ending_fraction_synthetic':pre1,'domain_discrepancy_flag':domain_flag,'domain_threshold_note':'Diagnostic threshold chosen for this audit before outcomes: AUC > .8 or preflop-ending gap > .10. Source discrimination is not proof of downstream harm.','diversity_passed':cell_pass,'scale_up_admitted':cell_pass and not domain_flag,'no_original_held_labels_used':True,'calibration_correction':'Original empty board was counted as one split token in calibration; this audit counts actual cards, so empty board is zero.'}
    out=HERE/'pilot-audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
    synthetic.write_parquet(HERE/'pilot-hand-diagnostics.parquet')
    print(json.dumps({k:v for k,v in result.items() if k not in ['pool_checks','synthetic','original_training_sample','domain_features']},indent=2))

if __name__=='__main__': main()
