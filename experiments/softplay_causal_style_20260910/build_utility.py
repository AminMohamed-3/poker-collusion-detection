"""Run only after unchanged fresh-seed realism passes; freeze all utility choices."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).resolve().parent;E=H.parent;P=E/'softplay_replication_20260910';C=E/'softplay_style_confirmation_20260910';N=E/'softplay_style_utility_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((H/'summary.json').read_text())['both_development_realism_gates_passed']
    assert json.loads((C/'summary.json').read_text())['passes_realism']
    N.mkdir(exist_ok=False);old=json.loads((P/'study-plan.json').read_text());arms=['b0_style','b1_style','b2_style'];parents=[H/'b0_style',H/'b1_style',C/'fresh'];hashes={};sources=[];diag={'training_only':True,'results':{}}
    for i,(arm,src) in enumerate(zip(arms,parents)):
        d=N/arm;d.mkdir();(d/'features').mkdir();(d/'pilot').mkdir()
        for name in ['pilot-plan.json','pilot-audit.json','intervention-replay-audit.json','training-calibration.json']:shutil.copy2(src/name,d/name)
        for name in ['metadata.json','synthetic-pairs.npz']:shutil.copy2(src/'features'/name,d/'features'/name)
        for pool in sorted((src/'pilot').glob('pool-*')):
            (d/'pilot'/pool.name).mkdir();shutil.copy2(pool/'provenance.json',d/'pilot'/pool.name/'provenance.json')
        hashes[arm]=sha(d/'pilot-plan.json');sources.append({'arm':arm,'directory':str(src.resolve()),'feature_sha256':sha(src/'features/synthetic-pairs.npz'),'metadata_sha256':sha(src/'features/metadata.json')})
        ds=json.loads((src.parent/'diagnostics.json').read_text());diag['results'][arm]=ds['results'][src.name]
    (N/'diagnostics.json').write_text(json.dumps(diag,indent=2));shutil.copy2(H/'controller.joblib',N/'controller.joblib')
    plan={**old,'created_at':datetime.now(timezone.utc).isoformat(),'hypothesis':'Probability coupling plus causal own-player tendencies passed two development realism populations and an unchanged fresh-seed confirmation. Test whether this repair adds predictive value beyond original training and multiple real-resampling controls.',
      'generator_arms':arms,'generator_plan_sha256':hashes,'batches':[{'batch':i,'arms':[a]} for i,a in enumerate(arms)],'detector_arms':['original','resampled_0','resampled_1','resampled_2',*arms],'detector_fits':21,'controller_sha256':sha(N/'controller.joblib'),'generation_sources':sources,
      'model_seeds':[19087,29501,40709],'resampling_seeds':[88721,99643,110017],
      'cohort':'N is the minimum supported target count across the three populations, at least12 with all4 policies. Deterministic policy-balanced hash order selects N unique targets per population; full/70/40 views. Same number of unique original soft targets in every resampling control.',
      'primary_estimand':'Equal mean of individual-model metrics over all three model seeds and all three synthetic populations, evaluated on both fixed histories. Compare with original-only and the mean of all three resampling draws. No seed, population, control or ensemble selection.',
      'utility_gate':old['utility_gate'].replace('both fresh generation batches','all three generation populations'),
      'realism_gate':'Already passed unchanged rules in two development populations and one fresh confirmation; preserve those receipts.',
      'replication_gate':'Require the fixed utility rule against original-only and mean resampling, including positive loss gain in every population and model seed and no loss degradation in either predefined held-pool half. Passing is a research lead, not production admission.',
      'evaluation':'Freeze all21 fits/predictions before one evaluation on the reused100heldpools/457pairs. Same full/short histories, softplay-fast-v1 halves and2000 whole-pool bootstrap seed20319. Conditional intervals, not untouched holdout or independent seed uncertainty.',
      'stop':'Report all outcomes. No extra utility fit, threshold adjustment, seed replacement or leaderboard feedback in this study. Choose any further generator research from training-only diagnostics.',
      'previous_study_plan_sha256':sha(P/'study-plan.json'),'source_hashes':{},'fresh_confirmation_summary_sha256':sha(C/'summary.json')}
    for key in ['scenario_pools','hands_per_pool','independent_exogenous_scenarios','counterfactual_histories','calibrated_probability','unchanged_generator']:
        plan.pop(key,None)
    (N/'study-plan.json').write_text(json.dumps(plan,indent=2))
    for name in ['prepare_trial.py','train_trial.py','evaluate_trial.py','replication_statistics.py']:
        s=(P/name).read_text()
        if name=='prepare_trial.py':
            s=s.replace("H.parent/'softplay_conditional_20260910/controller-fit.json'","H.parent/'softplay_causal_style_20260910/controller-fit.json'").replace("fit['model_sha256']","fit['controller_sha256']")
        if name=='evaluate_trial.py':
            s=s.replace("['calibrated','continuous']","['style']").replace('realism_passes_both_batches','realism_passes_all_populations').replace('passes_replication_gate','passes_utility_gate')
            s=s.replace('six fitted synthetic models','nine fitted synthetic models').replace('three seeds and two batches','three seeds and three populations')
        (N/name).write_text(s,encoding='utf-8')
    (N/'diagnostics.py').write_text('"""Realism was completed before this utility study; no new source fitting."""\n',encoding='utf-8')
    sources=['prepare_trial.py','train_trial.py','evaluate_trial.py','replication_statistics.py','diagnostics.py']
    (N/'analysis-freeze.json').write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'sources':{n:sha(N/n) for n in sources},'before_utility_fits_and_scores':True,'study_plan_sha256':sha(N/'study-plan.json')},indent=2))
    script='''from pathlib import Path
import subprocess,sys
H=Path(__file__).resolve().parent
for name in ['prepare_trial','train_trial','evaluate_trial']:
    with (H/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(H/(name+'.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name+' complete',flush=True)
'''
    (N/'run.py').write_text(script,encoding='utf-8');print(json.dumps({'directory':str(N),'fits':21,'plan_sha256':sha(N/'study-plan.json')}))
if __name__=='__main__':main()
