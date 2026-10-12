"""Fit the preselected causal-style controller and freeze paired development replays."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import polars as pl
import joblib
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;P=H.parent/'softplay_coupling_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (H/'controller.joblib').exists();probe=json.loads((H/'probe.json').read_text());assert probe['selected']=='own_style'
    pplan=json.loads((H/'probe-plan.json').read_text());assert sha(H/'training-actions-causal.parquet')==pplan['data_sha256'];inputs=probe['results']['own_style']['inputs']
    data=pl.read_parquet(H/'training-actions-causal.parquet').filter(pl.col('behavior_family')=='soft_play');counts=data.group_by('pair_id').len();weights=data.select('pair_id').join(counts,on='pair_id',how='left')['len'].to_numpy()**-1.*data.height/counts.height
    m=LGBMClassifier(**pplan['parameters']).fit(data.select(inputs).to_numpy(),data['action_class'].to_numpy(),sample_weight=weights)
    joblib.dump({'soft':m.booster_,'inputs':inputs,'causal_style_smoothing':1,'information':'own cards, public current-hand history, and own action counts in completed earlier hands only'},H/'controller.joblib')
    receipt={'created_at':datetime.now(timezone.utc).isoformat(),'probe_plan_sha256':sha(H/'probe-plan.json'),'training_pools':json.loads((H/'causal-style-receipt.json').read_text())['training_pools'],'soft_action_rows':data.height,'soft_pairs':data['pair_id'].n_unique(),'parameters':pplan['parameters'],'inputs':inputs,'controller_sha256':sha(H/'controller.joblib'),'selection_before_generation':True};(H/'controller-fit.json').write_text(json.dumps(receipt,indent=2))
    arms=[];hashes={}
    for b in range(2):
        arm=f'b{b}_style';arms.append(arm);d=H/arm;d.mkdir(exist_ok=False);src=P/f'b{b}_coupled';pp=json.loads((src/'pilot-plan.json').read_text())
        for name,hash in pp['source_hashes'].items():assert sha(src/name)==hash;shutil.copy2(src/name,d/name)
        shutil.copy2(H/'controller.joblib',d/'controller.joblib')
        s=(d/'soft_controller.py').read_text().replace("    assert len(values)==len(bundle()['inputs'])","    values.extend(obs.own_past_style)\n    assert len(values)==len(bundle()['inputs'])")
        (d/'soft_controller.py').write_text(s,encoding='utf-8')
        s=(d/'generator.py').read_text().replace('    can_raise: bool = True','    can_raise: bool = True\n    own_past_style: tuple = (.25,.25,.25,.25)')
        s=s.replace('active_episode=True, button_seat=5):','active_episode=True, button_seat=5, public_styles=None):')
        s=s.replace('proposal,base_probabilities=base_decision(obs,profiles[actor],rng)',"obs=replace(obs,own_past_style=tuple((public_styles or {}).get(players[actor],(.25,.25,.25,.25))))\n        proposal,base_probabilities=base_decision(obs,profiles[actor],rng)")
        s=s.replace("'actual_action':name}","'actual_action':name,'own_past_style':obs.own_past_style}")
        (d/'generator.py').write_text(s,encoding='utf-8')
        s=(d/'run_pilot.py').read_text().replace('    previously_seen=set(); resets=[]','    previously_seen=set(); resets=[]\n    past_counts={p:[0,0,0,0] for p in players}')
        s=s.replace('button_seat=button)',"button_seat=button,public_styles={p:tuple((n+1)/(sum(past_counts[p])+4) for n in past_counts[p]) for p in sitting})")
        s=s.replace('        hands.append(hand);seats.extend(ss);actions.extend(aa)',"        hands.append(hand);seats.extend(ss);actions.extend(aa)\n        for row in aa:\n            kind=0 if row['action']=='fold' else 1 if row['action']=='check' else 2 if row['action']=='call' or (row['action']=='all_in' and row['amount']<=row['to_call']) else 3\n            past_counts[row['player_id']][kind]+=1")
        (d/'run_pilot.py').write_text(s,encoding='utf-8');pp['source_hashes']={n:sha(d/n) for n in pp['source_hashes']};pp['paired_existing_control']=str(src.resolve());(d/'pilot-plan.json').write_text(json.dumps(pp,indent=2));hashes[arm]=sha(d/'pilot-plan.json')
    for name in ['soft_controller.py','conditional_action_features.py','advanced_features.py']:shutil.copy2(H/arms[0]/name,H/name)
    for name in ['audit_batch.py','replay_batch.py','prepare_features.py','run_generation.py']:shutil.copy2(P/name,H/name)
    s=(P/'diagnostics.py').read_text()
    s=s.replace('def sha(p):return', 'from causal_styles import attach_styles\n\ndef sha(p):return')
    s=s.replace('f=extract_chunk(keys,h,s,a,counts);checked=0',"f=extract_chunk(keys,h,s,a,counts);f,style_map=attach_styles(f,h,s,a,lab);checked=0\n    for hand in json.loads((p/'interventions.json').read_text()):\n        for change in hand['interventions']:\n            row=a.filter((pl.col('hand_id')==hand['hand_id'])&(pl.col('action_no')==change['action_no'])).row(0,named=True)\n            np.testing.assert_allclose(change['own_past_style'],style_map[hand['hand_id'],row['player_id']],atol=1e-7)")
    s=s.replace('previous_aggressor=previous)','previous_aggressor=previous,own_past_style=style_map[hand_id,row[\'player_id\']])')
    (H/'diagnostics.py').write_text(s,encoding='utf-8')
    s=(P/'run_analysis.py').read_text().replace("assert (H/'coupling-tests.json').exists()","assert (H/'causal-style-tests.json').exists()")
    (H/'run_analysis.py').write_text(s,encoding='utf-8')
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'generator_arms':arms,'generator_plan_sha256':hashes,'batches':[{'arm':f'b{b}_style','existing_control':str((P/f'b{b}_coupled').resolve()),'pool_ids':list(range(start,start+32))} for b,start in enumerate([640,704])],
      'hypothesis':'An average soft-play policy ignores persistent individual tendencies. Own observed action frequencies from earlier hands improve training-pool action prediction and may reduce artificial between-hand action variation.',
      'selected_model':'25 inputs: existing21 plus four own-player past action frequencies. No partner private cards, no whole-history/future style columns. Normal policies and distribution transport unchanged.',
      'controller_sha256':sha(H/'controller.joblib'),'selection_receipt_sha256':sha(H/'probe.json'),'realism_gate':'Unchanged: source AUC<=.85, preflop calling and conditional call-strength gaps<=.05, >=12 targets/all4 ordinary families, in BOTH development populations.',
      'purpose':'Controlled development replay. No outer-held utility scoring unless a later unchanged fresh-seed confirmation passes. Preserve failed predecessor gates.',
      'protected':json.loads((P/'study-plan.json').read_text())['protected']}
    (H/'study-plan.json').write_text(json.dumps(plan,indent=2));print('Causal-style trial frozen with controller '+sha(H/'controller.joblib'))
if __name__=='__main__':main()
