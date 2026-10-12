"""One frozen two-dose test; all predictors are frozen before one evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,sys
H=Path(__file__).resolve().parent;N=H.parent/'bet_size_utility_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def main():
    design=json.loads((H/'utility-design-freeze.json').read_text());assert all(sha(H/n)==v for n,v in design['source_hashes'].items())
    subprocess.run([sys.executable,str(H/'build_base_utility.py')],check=True)
    p=json.loads((N/'study-plan.json').read_text());p['detector_arms']+=['resampled_pool_0','resampled_pool_1','resampled_pool_2','pooled'];p['detector_fits']=33;p['secondary_pooled_dose']='Combine the fixedN targets from each of all3 populations, total3N unique targets. Match with3 fixed3N real-soft resampling controls using the same predetermined draw seeds. No generated target selection based on utility. Require3N<=92 available unique original soft targets.'
    p['secondary_pooling_screen']='Compare the pooled arm against original-only and mean of matched high-dose real resampling. Same loss>=.001, AP/sparse nonnegative, worst-history AP>=-.001, both pool halves nonnegative and every seed loss-positive checks. This is one combined generated sample and only a prospective lead, not an independently replicated population effect. Does not replace the primary per-population gate.'
    p['evaluation']=p['evaluation'].replace('all21 fits','all33 fits');p['dose_comparison']='Report both doses regardless of which wins. Descriptive incremental gain beyond dose-matched resampling; quantity and population diversity change together.';(N/'study-plan.json').write_text(json.dumps(p,indent=2))
    s=(N/'prepare_trial.py').read_text()
    a="    assert all(len(set(v))==n for v in control_ids.values())";b=a+"\n    assert 3*n<=len(realsoft)\n    control_ids.update({f'resampled_pool_{i}':np.random.default_rng(seed).permutation(realsoft)[:3*n] for i,seed in enumerate(plan['resampling_seeds'])})\n    assert all(len(set(control_ids[f'resampled_pool_{i}']))==3*n for i in range(3))";s=replace_once(s,a,b)
    a="        elif arm in control_ids:extra=np.concatenate([bx[control_ids[arm]+j*1403] for j in range(3)])";b=a+"\n        elif arm=='pooled':\n            chunks=[]\n            for batch in plan['batches']:\n                name=batch['arms'][0];xx,mm=datasets[name];ii=np.array([maps[name][pair] for pair in selected[str(batch['batch'])]])\n                chunks.extend(xx[ii+j*len(mm)] for j in range(3))\n            extra=np.concatenate(chunks)\n            assert len(extra)==9*n";s=replace_once(s,a,b)
    s=replace_once(s,"'unique_extra_pairs':0 if arm=='original' else n","'unique_extra_pairs':len(extra)//3");(N/'prepare_trial.py').write_text(s,encoding='utf-8')
    s=(N/'evaluate_trial.py').read_text();a="    # Fixed across-seed probability averages are secondary diagnostics, never the primary gate.";b="""    pool_controls=['resampled_pool_0','resampled_pool_1','resampled_pool_2'];secondary={}
    for baseline,bb in [('original',['original']),('mean_resampling',pool_controls)]:
        c=comparison(['pooled'],bb);byseed={str(seed):comparison(['pooled'],bb,[seed]) for seed in plan['model_seeds']};c['by_model_seed']=byseed
        c['screen']=gate(c['mean_deltas'],c['worst_history_pair_ap_delta'],c['pool_half_loss_improvements'],[c['mean_deltas']['logloss_improvement']],[v['mean_deltas']['logloss_improvement'] for v in byseed.values()])
        secondary[baseline]=c
    secondary['passes_screen']=all(secondary[b]['screen']['passed'] for b in ['original','mean_resampling'])
    secondary['unique_generated_targets']=3*f['common_dose'];secondary['one_combined_population_not_independent_replication']=True
    secondary['incremental_loss_gain_beyond_matched_resampling']=secondary['mean_resampling']['mean_deltas']['logloss_improvement']-primary['sized']['mean_resampling']['mean_deltas']['logloss_improvement']
    secondary['high_dose_controls_vs_original']={c:comparison([c],['original']) for c in pool_controls}
"""+a;s=replace_once(s,a,b);s=replace_once(s,"out={'primary':primary,","out={'secondary_pooling':secondary,'primary':primary,");s=s.replace('nine fitted synthetic models','twelve fitted synthetic models including one pooled-dose arm');(N/'evaluate_trial.py').write_text(s,encoding='utf-8')
    s=(N/'train_trial.py').read_text().replace('Twenty-four predetermined small trees','Thirty-three predetermined small tree fits');(N/'train_trial.py').write_text(s,encoding='utf-8')
    names=['prepare_trial.py','train_trial.py','evaluate_trial.py','replication_statistics.py','diagnostics.py'];freeze={'at':datetime.now(timezone.utc).isoformat(),'sources':{n:sha(N/n) for n in names},'before_utility_fits_and_scores':True,'study_plan_sha256':sha(N/'study-plan.json')};(N/'analysis-freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps({'fits':33,'two_prespecified_doses':True,'study_plan_sha256':sha(N/'study-plan.json')}))
if __name__=='__main__':main()
