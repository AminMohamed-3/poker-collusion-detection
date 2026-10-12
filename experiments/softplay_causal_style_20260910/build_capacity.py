"""Freeze a larger policy chosen by training-pool action prediction only."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import polars as pl
import joblib
from lightgbm import LGBMClassifier
H=Path(__file__).resolve().parent;E=H.parent;N=E/'softplay_capacity_20260910';C=E/'softplay_style_confirmation_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    probe=json.loads((H/'capacity-probe.json').read_text());choice=probe['selected'];assert choice and probe['results'][choice]['passes'];N.mkdir(exist_ok=False)
    old=joblib.load(H/'controller.joblib');params=json.loads((H/'capacity-probe-plan.json').read_text())['parameters'][choice];f=pl.read_parquet(H/'training-actions-causal.parquet').filter(pl.col('behavior_family')=='soft_play');n=f.group_by('pair_id').len();w=f.select('pair_id').join(n,on='pair_id',how='left')['len'].to_numpy()**-1.*f.height/n.height;m=LGBMClassifier(**params).fit(f.select(old['inputs']).to_numpy(),f['action_class'].to_numpy(),sample_weight=w);joblib.dump({**old,'soft':m.booster_},N/'controller.joblib')
    fit={**json.loads((H/'controller-fit.json').read_text()),'created_at':datetime.now(timezone.utc).isoformat(),'parameters':params,'controller_sha256':sha(N/'controller.joblib'),'selection_receipt_sha256':sha(H/'capacity-probe.json'),'no_outer_held_data_used':True};(N/'controller-fit.json').write_text(json.dumps(fit,indent=2))
    arms=[];hashes={};batches=[]
    for i,src in enumerate([H/'b0_style',H/'b1_style',C/'fresh']):
        arm=f'b{i}_capacity';arms.append(arm);d=N/arm;d.mkdir();pp=json.loads((src/'pilot-plan.json').read_text())
        for name,h in pp['source_hashes'].items():assert sha(src/name)==h;shutil.copy2(src/name,d/name)
        shutil.copy2(N/'controller.joblib',d/'controller.joblib');shutil.copy2(E/'softplay_probability_20260910/candidate/soft_controller.py',d/'soft_controller.py')
        pp.update(created_at=datetime.now(timezone.utc).isoformat(),source_hashes={n:sha(d/n) for n in pp['source_hashes']},paired_existing_control=str(src.resolve()),counterfactual_development_replay=True);pp.pop('fresh_seed_confirmation',None)
        (d/'pilot-plan.json').write_text(json.dumps(pp,indent=2));hashes[arm]=sha(d/'pilot-plan.json');batches.append({'arm':arm,'existing_control':str(src.resolve()),'pool_ids':[s['pool_id'] for s in pp['pools']]})
    for name in ['soft_controller.py','conditional_action_features.py','advanced_features.py']:shutil.copy2(N/arms[0]/name,N/name)
    for name in ['causal_styles.py','audit_batch.py','replay_batch.py','prepare_features.py','run_generation.py','diagnostics.py','run_analysis.py']:shutil.copy2(H/name,N/name)
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'generator_arms':arms,'generator_plan_sha256':hashes,'batches':batches,'controller_sha256':sha(N/'controller.joblib'),'action_model_parameters':params,'selection_receipt_sha256':sha(H/'capacity-probe.json'),
      'hypothesis':'The small100-tree soft policy underfits conditional action probabilities. A300-tree depth5 policy reduces pool-OOF action loss .027554 beyond causal styles and may remove unrealistic action residual tails.',
      'changes':'Only the selected soft-policy model capacity and the mathematically corrected free-call/check representation. The latter was proven gameplay-equivalent for this non-aggression-increasing controller. Inputs, observed past-style process, ordinary policies, environment, scenarios and action evidence rules unchanged.',
      'realism_gate':'Same source AUC<=.85, preflop call-rate/strength gaps<=.05, >=12 supported targets and4ordinary-policy families in EACH of all3 existing populations. The previously failed fresh population is explicitly included. Preserve failures.',
      'progression':'If all3 development populations pass, freeze a NEW independent simulation-seed confirmation before one utility trial. If any fails, inspect training-only diagnostics. No outer validation or leaderboard feedback in this study.',
      'protected':json.loads((H/'study-plan.json').read_text())['protected']}
    (N/'study-plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({'directory':str(N),'choice':choice,'controller_sha256':plan['controller_sha256'],'populations':3}))
if __name__=='__main__':main()
