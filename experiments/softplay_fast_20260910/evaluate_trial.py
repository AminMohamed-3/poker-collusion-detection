"""Score all frozen arms once; pool halves are fixed without labels."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,sys
import numpy as np
from sklearn.metrics import average_precision_score
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'trial';sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import LABELS,stable_ap
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ce(y,p):return -np.log(np.maximum(p[np.arange(len(y)),y],1e-12))
def metrics(y,p):
    t=y>0;pre=t.mean();w=np.where(t,.01/pre,.99/(1-pre));risk=1-p[:,0]
    return {'log_loss':float(ce(y,p).mean()),'pair_ap':stable_ap(t,risk),'sparse_ap':stable_ap(t,risk,w),'soft_play_ap':float(average_precision_score(y==2,p[:,2])),'class_log_loss':{str(i):float(ce(y,p)[y==i].mean()) for i in range(4) if np.any(y==i)}}
def main():
    assert not (D/'evaluation.json').exists()
    freeze=json.loads((D/'freeze.json').read_text());plan=json.loads((H/'study-plan.json').read_text())
    assert sha(__file__)==freeze['source_hashes']['evaluate_trial.py']
    assert sha(H/'study-plan.json')==freeze['study_plan_sha256']
    probs={}
    for arm in plan['arms']:
        r=json.loads((D/arm/'receipt.json').read_text());assert r['freeze_sha256']==sha(D/'freeze.json') and not r['held_labels_used'] and r['reload_exact']
        assert sha(D/arm/'model.joblib')==r['model_sha256'] and sha(D/arm/'predictions.npz')==r['predictions_sha256'];probs[arm]=dict(np.load(D/arm/'predictions.npz'))
    # Held labels become available only after every arm's receipt passes.
    mp=R/'experiments/mhc_local/fold0/held-metadata.json';assert sha(mp)==freeze['held_metadata_sha256'];meta=json.loads(mp.read_text())
    rows=[];data={}
    for tag in ['full','shortened']:
        records=meta[tag]['pairs'];y=np.array([LABELS[r['behavior_family']] for r in records]);g=np.array([r['table_id'] for r in records]);half=np.array([int(hashlib.sha256(('softplay-fast-v1:'+p).encode()).hexdigest(),16)%2 for p in g]);data[tag]=(y,g,half)
        for arm in plan['arms']:
            for scope,take in [('all',np.ones(len(y),dtype=bool)),('half0',half==0),('half1',half==1)]:
                rows.append({'arm':arm,'history':tag,'scope':scope,'pairs':int(take.sum()),'pools':len(set(g[take])),**metrics(y[take],probs[arm][tag][take])})
    comparisons={};rng=np.random.default_rng(20319);poolnames=sorted(set(data['full'][1]));boot=rng.integers(0,len(poolnames),(1000,len(poolnames)))
    for arm in plan['arms']:
        if arm=='resampled':continue
        differences=[];halfmeans=[]
        for scope in ['all','half0','half1']:
            dif=[]
            for tag in ['full','shortened']:
                b=next(r for r in rows if r['arm']=='resampled' and r['history']==tag and r['scope']==scope);a=next(r for r in rows if r['arm']==arm and r['history']==tag and r['scope']==scope)
                dif.append({'history':tag,'scope':scope,'logloss_improvement':b['log_loss']-a['log_loss'],**{k:a[k]-b[k] for k in ['pair_ap','sparse_ap','soft_play_ap']}})
            differences.extend(dif)
            if scope!='all':halfmeans.append(float(np.mean([r['logloss_improvement'] for r in dif])))
        whole=[r for r in differences if r['scope']=='all'];mean={k:float(np.mean([r[k] for r in whole])) for k in ['logloss_improvement','pair_ap','sparse_ap','soft_play_ap']};worst=min(r['pair_ap'] for r in whole)
        totals=[];counts=[]
        for name in poolnames:
            dd=[]
            for tag in ['full','shortened']:
                y,g,_=data[tag];delta=ce(y,probs['resampled'][tag])-ce(y,probs[arm][tag]);dd.extend(delta[g==name])
            totals.append(sum(dd));counts.append(len(dd))
        bmean=np.array(totals)[boot].sum(1)/np.array(counts)[boot].sum(1)
        passed=mean['logloss_improvement']>=.001 and mean['pair_ap']>=0 and mean['sparse_ap']>=0 and worst>=-.001 and min(halfmeans)>=0
        comparisons[arm]={'passes_utility_screen':bool(passed),'mean_deltas':mean,'worst_history_pair_ap_delta':worst,'pool_half_mean_logloss_improvement':halfmeans,'pool_bootstrap_logloss_improvement_95pct_interval':np.quantile(bmean,[.025,.975]).tolist(),'cases':differences}
    assert all(sha(R/p)==h for p,h in plan['protected'].items())
    result={'study_plan_sha256':sha(H/'study-plan.json'),'freeze_sha256':sha(D/'freeze.json'),'metrics':rows,'comparisons_to_resampled':comparisons,'all_five_scored_together':True,'production_admitted':False,'protected_submissions_unchanged':True,'heldout_reused':True,'pool_half_method':'sha256(softplay-fast-v1: + table_id) mod 2','bootstrap_seed':20319,'bootstrap_replicates':1000}
    (D/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
