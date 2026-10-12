"""Create a frozen fresh-seed replication without modifying the prior study."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'softplay_activation_20260910'
NEW=ROOT/'softplay_replication_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    NEW.mkdir(exist_ok=True)
    assert not (NEW/'study-plan.json').exists()
    old=json.loads((OLD/'study-plan.json').read_text())
    for name in ['audit_batch.py','replay_batch.py','prepare_features.py','run_generation.py','soft_controller.py','controller.joblib','conditional_action_features.py','advanced_features.py']:
        source=OLD/name if (OLD/name).exists() else OLD/'calibrated'/name
        shutil.copy2(source,NEW/name)
    arms=[];hashes={};batches=[]
    for b,(start,seed) in enumerate([(640,610031),(704,810119)]):
        arms_b=[]
        for mode in ['calibrated','continuous']:
            arm=f'b{b}_{mode}';arms.append(arm);arms_b.append(arm);dest=NEW/arm;dest.mkdir()
            pp=json.loads((OLD/mode/'pilot-plan.json').read_text())
            for name,h in pp['source_hashes'].items():
                assert sha(OLD/mode/name)==h
                shutil.copy2(OLD/mode/name,dest/name)
            specs=[]
            for i in range(32):
                specs.append(dict(pool_id=start+i,scenario=i,seed=seed+997*i,policy_family=i%4,soft_variant=(i//4)%4,behavior='soft_play',hands=3000,activation_mode=mode,activation_probability=old['calibrated_probability'] if mode=='calibrated' else 1.0))
            pp.update(created_at=datetime.now(timezone.utc).isoformat(),pools=specs,replication_parent_plan_sha256=sha(OLD/mode/'pilot-plan.json'))
            (dest/'pilot-plan.json').write_text(json.dumps(pp,indent=2),encoding='utf-8');hashes[arm]=sha(dest/'pilot-plan.json')
        batches.append(dict(batch=b,pool_ids=list(range(start,start+32)),seed_base=seed,arms=arms_b))
    plan={
      'created_at':datetime.now(timezone.utc).isoformat(),
      'hypothesis':'The apparent utility of increased soft-policy activation survives fresh simulation populations, model seeds, original-only comparison, and multiple real-resampling controls.',
      'generator_arms':arms,'batches':batches,'scenario_pools':64,'hands_per_pool':3000,
      'independent_exogenous_scenarios':192000,'counterfactual_histories':384000,
      'calibrated_probability':old['calibrated_probability'],'controller_sha256':old['controller_sha256'],
      'unchanged_generator':'All generation code, ordinary/controller models, environment calibration and controller RNG separation are byte-identical to the activation study. Only pool IDs and scenario seeds change. No refitting or recalibration.',
      'generator_plan_sha256':hashes,
      'cohort':'Per batch take the intersection of supported calibrated/continuous target pairs. Use N=min(common counts across batches), >=12 and all four ordinary policy families. Select N by deterministic policy-balanced round robin of SHA256(replication-cohort-v1:pair_id) order, never by realism or held score. Each unique target supplies full/70/40 views. If any coverage requirement fails, stop detector fits and report it.',
      'detector_arms':['original','resampled_0','resampled_1','resampled_2',*arms],
      'model_seeds':[19087,29501,40709],'resampling_seeds':[88721,99643,110017],
      'resampling':'Three fixed random permutations of the 92 original-training soft pairs, take first N without replacement in each. Same three draws across generator batches and all model seeds. Report all; do not select a control.',
      'tree_parameters':{k:v for k,v in old['tree_parameters'].items() if k!='random_state'},'detector_fits':24,
      'primary_estimand':'Equal average of individual-model metrics across all three model seeds and both fresh generation batches, each evaluated on the same two histories. Baselines are original-only and the equal mean of metrics from all three resampling draws, with matched model seeds. This is not an ensemble score. All seeds, batches and controls are reported.',
      'utility_gate':'For each activation setting, versus BOTH original-only and mean-resampling: mean multiclass logloss improvement >=.001, mean Pair AP and sparse AP deltas >=0, worst history Pair AP delta >=-.001, nonnegative mean loss improvement in both predefined validation-pool halves. Additionally, mean loss improvement must be >0 for EACH fresh generation batch and EACH model seed versus each baseline. All checks are fixed before generation. No rescue or best-seed selection.',
      'realism_gate':old['realism_gate'],
      'replication_gate':'Pass the utility gate against both baselines and the unchanged realism gate separately in both fresh batches. This is a research continuation gate, not permission for production or broad neural training.',
      'evaluation':'All 24 models/predictions are frozen before loading held labels. Reused 100 original validation pools /457 pairs, full and shortened histories. softplay-fast-v1 SHA256 halves unchanged. Paired whole-pool bootstrap2000 seed20319 applied to the mean per-example loss contrast across all fits; seed/batch dispersion separately reported. Repeated predictions and histories are not new independent observations. Intervals are conditional on these fitted runs, unadjusted, exploratory, not independent confirmation.',
      'stop':'No additional generation, model tuning, leaderboard queries, neural fits or production changes in this replication. Report every outcome including failures.',
      'protected':old['protected'],
      'previous_study_plan_sha256':sha(OLD/'study-plan.json'),
      'source_hashes':{n:sha(NEW/n) for n in ['audit_batch.py','replay_batch.py','prepare_features.py','run_generation.py','soft_controller.py']}
    }
    (NEW/'study-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'directory':str(NEW),'plan_sha256':sha(NEW/'study-plan.json'),'arms':arms,'tree_fits':24}))
if __name__=='__main__':main()
