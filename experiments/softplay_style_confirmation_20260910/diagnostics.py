"""Paired exogenous-state checks, action parity and training-only source audits."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;R=H.parents[1];OLD=H.parent/'softplay_fast_20260910'
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(OLD))
from conditional_inference_worker import extract_chunk
from conditional_action_features import action_class
from diagnose_actions import summary
import soft_controller
from causal_styles import attach_styles

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(p):
    lab=pl.read_csv(p/'development_labels.csv').filter(pl.col('behavior_family')=='soft_play')
    if not lab.height:return None,0
    h=pl.read_parquet(p/'hands.parquet').with_columns(pl.col('started_at').cast(pl.String));s=pl.read_parquet(p/'seats.parquet');a=pl.read_parquet(p/'actions.parquet');counts=defaultdict(lambda:np.zeros(4,dtype=np.int64))
    for row in a.iter_rows(named=True):counts[row['player_id']][action_class(row)]+=1
    keys=s.select('hand_id','player_id').join(lab,left_on='player_id',right_on='player_1').join(s.select('hand_id',pl.col('player_id').alias('player_2')),on=['hand_id','player_2']).select('pair_id','hand_id','player_id','player_2').rename({'player_id':'player_1'})
    f=extract_chunk(keys,h,s,a,counts);f,style_map=attach_styles(f,h,s,a,lab);checked=0
    for hand in json.loads((p/'interventions.json').read_text()):
        for change in hand['interventions']:
            row=a.filter((pl.col('hand_id')==hand['hand_id'])&(pl.col('action_no')==change['action_no'])).row(0,named=True)
            np.testing.assert_allclose(change['own_past_style'],style_map[hand['hand_id'],row['player_id']],atol=1e-7)
    # Independently reconstruct the controller's observation from action prefixes.
    target=lab.row(0,named=True);player1,player2=target['player_1'],target['player_2']
    for hand_id in sorted(set(f['hand_id']))[:2]:
        hand=h.filter(pl.col('hand_id')==hand_id).row(0,named=True);seats=s.filter(pl.col('hand_id')==hand_id).sort('seat_no').to_dicts();seats.sort(key=lambda x:(x['seat_no']-hand['button_seat']-1)%6);players=[r['player_id'] for r in seats];history=[];last_street=-1;previous=None
        for row in a.filter(pl.col('hand_id')==hand_id).sort('action_no').iter_rows(named=True):
            street=['preflop','flop','turn','river'].index(row['street'])
            if street!=last_street:previous=None;last_street=street
            actor=players.index(row['player_id']);partner=players.index(player2 if row['player_id']==player1 else player1) if row['player_id'] in [player1,player2] else None
            test=f.filter((pl.col('hand_id')==hand_id)&(pl.col('action_no')==row['action_no']))
            if test.height:
                seat=seats[actor];obs=SimpleNamespace(hole=(seat['hole_card_1'],seat['hole_card_2']),board=tuple(hand['board_cards'].split()[:[0,3,4,5][street]]),street=street,position=(seat['seat_no']-hand['button_seat'])%6,to_call=row['to_call'],pot=row['pot_before'],blind=hand['big_blind'],stack=row['stack_before'],active=row['players_active'],prior_actions=row['action_no'],public_history=tuple(history),actor=actor,previous_aggressor=previous,own_past_style=style_map[hand_id,row['player_id']])
                np.testing.assert_allclose(soft_controller.features(obs,partner),test.select(soft_controller.bundle()['inputs']).to_numpy(),rtol=1e-6,atol=1e-6);checked+=1
            kind=action_class(row);history.append((actor,kind,street,row['amount']))
            if kind==3:previous=actor
    return f,checked
def main():
    plan=json.loads((H/'study-plan.json').read_text());paired=[]
    original=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz')['train_pair'][:1403];om=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];keep=np.array([r['behavior_family']=='soft_play' for r in om]);ox=original[keep];og=np.array([r['table_id'] for r in om])[keep]
    output={};parity=0
    for mode in plan['generator_arms']:
        d=H/mode;frames=[]
        for p in sorted((d/'pilot').glob('pool-*')):
            f,n=extract(p);parity+=n
            if f is not None:frames.append(f)
        actions=pl.concat(frames);actions.write_parquet(d/'soft-actions.parquet');m=json.loads((d/'features/metadata.json').read_text())['views'][0];take=np.array([r['behavior_family']=='soft_play' for r in m]);sx=np.load(d/'features/synthetic-pairs.npz')['x'][:len(m)][take];sg=np.array([r['table_id'] for r in m])[take]
        x=np.r_[ox,sx];source=np.r_[np.zeros(len(ox)),np.ones(len(sx))];groups=np.r_[og,sg];pred=np.full(len(source),np.nan);folds=[]
        for k,(tr,te) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=19083).split(x,source,groups)):
            assert not set(groups[tr])&set(groups[te])
            model=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=30,class_weight='balanced',learning_rate=.05,n_jobs=2,verbosity=-1,random_state=19083+k).fit(x[tr],source[tr]);pred[te]=model.predict_proba(x[te])[:,1];folds.append(float(roc_auc_score(source[te],pred[te])))
        auc=float(roc_auc_score(source,pred));contexts=summary(actions)
        conditional={}
        for name,cond in [('preflop_calls',(pl.col('street_index')==0)&(pl.col('action_class')==2)),('postflop_checks',(pl.col('street_index')>0)&(pl.col('action_class')==1))]:
            f=actions.filter(cond);v=f.group_by('pair_id').agg(pl.col('own_strength').mean(),pl.col('price').mean());conditional[name]={'own_strength':float(v['own_strength'].mean()),'price':float(v['price'].mean()),'pairs':v.height}
        audit=json.loads((d/'pilot-audit.json').read_text())
        original=json.loads((OLD/'action-diagnosis.json').read_text())['results']['soft_play'];real_strength=json.loads((OLD/'remaining-mismatch.json').read_text())['results']['real_soft']['preflop_call']['own_strength']
        call_gap=abs(contexts['preflop']['equal_pair_mean_fold_check_call_raise'][2]-original['preflop']['equal_pair_mean_fold_check_call_raise'][2]);strength_gap=abs(conditional['preflop_calls']['own_strength']-real_strength)
        output[mode]={'source_auc':auc,'fold_auc':folds,'supported_pairs':len(sx),'action_summary':contexts,'conditional':conditional,'preflop_call_gap':call_gap,'preflop_call_strength_gap':strength_gap,'realism_gate_passed':bool(auc<=.85 and len(sx)>=12 and audit['policy_families']==4 and call_gap<=.05 and strength_gap<=.05)}
        print(mode,'source AUC',auc,'supported',len(sx),flush=True)
    result={'training_only':True,'paired_scenarios_verified':paired,'paired_seats_hole_cards_profiles_episodes_exact':False,'pairing_not_applicable_single_fresh_arm':True,'controller_feature_parity_rows':parity,'results':output,'source_sha256':sha(__file__),'limitations':'Fresh simulation seeds with unchanged generator; the same learned policy source and original training references remain. Original 100 held pools not loaded.'}
    (H/'diagnostics.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
