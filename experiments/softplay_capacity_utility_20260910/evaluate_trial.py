"""Score every frozen run once; compare to original and all resampling controls."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,sys
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'trial'
from replication_statistics import mean_metrics,deltas,pool_interval,gate
sys.path.insert(0,str(H.parent/'softplay_fast_20260910'))
from evaluate_trial import metrics,ce
sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import LABELS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (D/'evaluation.json').exists()
    f=json.loads((D/'freeze.json').read_text());plan=json.loads((H/'study-plan.json').read_text());assert sha(H/'study-plan.json')==f['study_plan_sha256']
    assert all(sha(H/k)==v for k,v in f['source_hashes'].items());probs={};receipts=[]
    for arm in plan['detector_arms']:
        assert sha(D/(arm+'.npz'))==f['inputs'][arm]['sha256']
        for seed in plan['model_seeds']:
            p=D/arm/str(seed);r=json.loads((p/'receipt.json').read_text());assert r['freeze_sha256']==sha(D/'freeze.json') and not r['held_labels_used'] and r['reload_exact']
            assert sha(p/'model.joblib')==r['model_sha256'] and sha(p/'predictions.npz')==r['predictions_sha256']
            assert r['seed']==seed and r['arm']==arm and r['parameters']=={**plan['tree_parameters'],'random_state':seed}
            probs[arm,seed]=dict(np.load(p/'predictions.npz'));receipts.append(r)
    assert len(receipts)==plan['detector_fits']
    mp=H.parent/'mhc_local/fold0/held-metadata.json';assert sha(mp)==f['held_metadata_sha256'];meta=json.loads(mp.read_text());rows=[];data={}
    for tag in ['full','shortened']:
        y=np.array([LABELS[r['behavior_family']] for r in meta[tag]['pairs']]);g=np.array([r['table_id'] for r in meta[tag]['pairs']]);half=np.array([int(hashlib.sha256(('softplay-fast-v1:'+p).encode()).hexdigest(),16)%2 for p in g]);data[tag]=(y,g,half)
        for arm,seed in probs:
            for scope,take in [('all',np.ones(len(y),dtype=bool)),('half0',half==0),('half1',half==1)]:
                rows.append({'arm':arm,'seed':seed,'history':tag,'scope':scope,'pairs':int(take.sum()),'pools':len(set(g[take])),**metrics(y[take],probs[arm,seed][tag][take])})
    def comparison(arms,controls,seeds=None):
        seeds=plan['model_seeds'] if seeds is None else seeds
        def avg(aa,tag,scope):return mean_metrics([r for r in rows if r['arm'] in aa and r['seed'] in seeds and r['history']==tag and r['scope']==scope])
        cases=[dict(history=tag,scope=scope,**deltas(avg(arms,tag,scope),avg(controls,tag,scope))) for scope in ['all','half0','half1'] for tag in ['full','shortened']]
        whole=[r for r in cases if r['scope']=='all'];mean={k:float(np.mean([r[k] for r in whole])) for k in ['logloss_improvement','pair_ap','sparse_ap','soft_play_ap']}
        halves=[float(np.mean([r['logloss_improvement'] for r in cases if r['scope']==s])) for s in ['half0','half1']]
        contrasts=[];groups=[]
        for tag,(y,g,_) in data.items():
            loss=lambda aa:np.mean([ce(y,probs[a,s][tag]) for a in aa for s in seeds],axis=0)
            contrasts.extend(loss(controls)-loss(arms));groups.extend(g)
        interval=pool_interval(contrasts,groups)
        np.testing.assert_allclose(np.mean(contrasts),mean['logloss_improvement'],atol=1e-14)
        return dict(mean_deltas=mean,worst_history_pair_ap_delta=min(r['pair_ap'] for r in whole),pool_half_loss_improvements=halves,pool_bootstrap_logloss_improvement_95pct_interval=interval,cases=cases)
    controls=['resampled_0','resampled_1','resampled_2'];primary={};individual={};ensembles=[]
    diagnostics=json.loads((H/'diagnostics.json').read_text())
    for mode in ['capacity']:
        arms=[a for a in plan['generator_arms'] if a.endswith('_'+mode)];primary[mode]={}
        for baseline,bb in [('original',['original']),('mean_resampling',controls)]:
            c=comparison(arms,bb)
            batch={a:comparison([a],bb) for a in arms};seed={str(s):comparison(arms,bb,[s]) for s in plan['model_seeds']}
            c['by_batch']=batch;c['by_model_seed']=seed
            c['utility_gate']=gate(c['mean_deltas'],c['worst_history_pair_ap_delta'],c['pool_half_loss_improvements'],[v['mean_deltas']['logloss_improvement'] for v in batch.values()],[v['mean_deltas']['logloss_improvement'] for v in seed.values()])
            primary[mode][baseline]=c
        primary[mode]['realism_passes_all_populations']=all(diagnostics['results'][a]['realism_gate_passed'] for a in arms)
        primary[mode]['passes_utility_gate']=bool(primary[mode]['realism_passes_all_populations'] and all(primary[mode][b]['utility_gate']['passed'] for b in ['original','mean_resampling']))
        for a in arms:
            for s in plan['model_seeds']:
                individual[f'{a}/{s}']={b:comparison([a],[b],[s]) for b in ['original',*controls]}
    control_results={c:comparison([c],['original']) for c in controls}
    control_results['mean_resampling']=comparison(controls,['original'])
    # Fixed across-seed probability averages are secondary diagnostics, never the primary gate.
    for arm in plan['detector_arms']:
        for tag,(y,_,_) in data.items():
            p=np.mean([probs[arm,s][tag] for s in plan['model_seeds']],axis=0)
            ensembles.append({'arm':arm,'history':tag,**metrics(y,p)})
    assert all(sha(R/p)==v for p,v in plan['protected'].items())
    out={'primary':primary,'individual_comparisons':individual,'resampling_vs_original':control_results,'metrics':rows,'secondary_seed_ensemble_metrics':ensembles,'fit_count':len(receipts),'fit_seconds':sum(r['seconds'] for r in receipts),'all_frozen_before_held_labels':True,'protected_submissions_unchanged':True,'reused_validation':True,'production_admitted':False,'freeze_sha256':sha(D/'freeze.json'),'uncertainty':'Pool intervals condition on twelve fitted synthetic models and three or nine baseline models. Seed/batch summaries describe sensitivity; three seeds and four populations do not establish their sampling distribution. Histories and repeated fits are not independent validation observations.'}
    (D/'evaluation.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({m:{'original':v['original']['mean_deltas'],'resampling':v['mean_resampling']['mean_deltas'],'gate':v['passes_utility_gate']} for m,v in primary.items()},indent=2))
if __name__=='__main__':main()
