"""All six predictions freeze before one paired, two-dose utility evaluation."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,sys
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'trial'
sys.path.insert(0,str(H.parent/'softplay_fast_20260910'))
from evaluate_trial import metrics,ce
sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import LABELS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (D/'evaluation.json').exists();f=json.loads((D/'freeze.json').read_text());plan=json.loads((H/'study-plan.json').read_text());assert sha(__file__)==f['source_hashes']['evaluate_trial.py'] and sha(H/'study-plan.json')==f['study_plan_sha256'];probs={}
    for arm in plan['detector_arms']:
        r=json.loads((D/arm/'receipt.json').read_text());assert r['freeze_sha256']==sha(D/'freeze.json') and not r['held_labels_used'] and r['reload_exact'];assert sha(D/arm/'model.joblib')==r['model_sha256'] and sha(D/arm/'predictions.npz')==r['predictions_sha256'];probs[arm]=dict(np.load(D/arm/'predictions.npz'))
    mp=R/'experiments/mhc_local/fold0/held-metadata.json';assert sha(mp)==f['held_metadata_sha256'];meta=json.loads(mp.read_text());data={};rows=[]
    for tag in ['full','shortened']:
        y=np.array([LABELS[r['behavior_family']] for r in meta[tag]['pairs']]);g=np.array([r['table_id'] for r in meta[tag]['pairs']]);half=np.array([int(hashlib.sha256(('softplay-fast-v1:'+p).encode()).hexdigest(),16)%2 for p in g]);data[tag]=(y,g,half)
        for arm in plan['detector_arms']:
            for scope,take in [('all',np.ones(len(y),dtype=bool)),('half0',half==0),('half1',half==1)]:rows.append({'arm':arm,'history':tag,'scope':scope,'pairs':int(take.sum()),'pools':len(set(g[take])),**metrics(y[take],probs[arm][tag][take])})
    rng=np.random.default_rng(20319);pools=sorted(set(data['full'][1]));boot=rng.integers(0,len(pools),(1000,len(pools)))
    def compare(arm,control):
        diffs=[];halfmeans=[]
        for scope in ['all','half0','half1']:
            part=[]
            for tag in ['full','shortened']:
                a=next(r for r in rows if r['arm']==arm and r['history']==tag and r['scope']==scope);b=next(r for r in rows if r['arm']==control and r['history']==tag and r['scope']==scope);part.append({'history':tag,'scope':scope,'logloss_improvement':b['log_loss']-a['log_loss'],**{k:a[k]-b[k] for k in ['pair_ap','sparse_ap','soft_play_ap']}})
            diffs+=part
            if scope!='all':halfmeans.append(float(np.mean([r['logloss_improvement'] for r in part])))
        whole=[r for r in diffs if r['scope']=='all'];mean={k:float(np.mean([r[k] for r in whole])) for k in ['logloss_improvement','pair_ap','sparse_ap','soft_play_ap']};worst=min(r['pair_ap'] for r in whole);totals=[];counts=[]
        for pool in pools:
            dd=[]
            for tag in ['full','shortened']:
                y,g,_=data[tag];dd.extend((ce(y,probs[control][tag])-ce(y,probs[arm][tag]))[g==pool])
            totals.append(sum(dd));counts.append(len(dd))
        meanboot=np.array(totals)[boot].sum(1)/np.array(counts)[boot].sum(1)
        return {'mean_deltas':mean,'worst_history_ap_delta':worst,'pool_half_loss_improvements':halfmeans,'pool_bootstrap_logloss_improvement_95pct_interval':np.quantile(meanboot,[.025,.975]).tolist(),'passes_utility_screen':bool(mean['logloss_improvement']>=.001 and mean['pair_ap']>=0 and mean['sparse_ap']>=0 and worst>=-.001 and min(halfmeans)>=0),'cases':diffs}
    comparisons={arm:compare(arm,'resampled_'+arm.split('_')[-1]) for arm in plan['detector_arms'] if not arm.startswith('resampled')};dose={m:compare(m+'_high',m+'_low') for m in ['resampled','strength','learned']}
    netdose={}
    for m in ['strength','learned']:
        totals=[];counts=[]
        for pool in pools:
            dd=[]
            for tag in ['full','shortened']:
                y,g,_=data[tag]
                delta=(ce(y,probs[m+'_low'][tag])-ce(y,probs[m+'_high'][tag]))-(ce(y,probs['resampled_low'][tag])-ce(y,probs['resampled_high'][tag]))
                dd.extend(delta[g==pool])
            totals.append(sum(dd));counts.append(len(dd))
        intervals=np.array(totals)[boot].sum(1)/np.array(counts)[boot].sum(1)
        netdose[m]={k:dose[m]['mean_deltas'][k]-dose['resampled']['mean_deltas'][k] for k in ['logloss_improvement','pair_ap','sparse_ap']}
        netdose[m]['pool_bootstrap_net_loss_improvement_95pct_interval']=np.quantile(intervals,[.025,.975]).tolist()
    assert all(sha(R/p)==v for p,v in plan['protected'].items())
    out={'metrics':rows,'comparisons_to_matched_dose':comparisons,'high_vs_low':dose,'net_high_vs_low_after_resampling_change':netdose,'all_six_frozen_before_scoring':True,'protected_submissions_unchanged':True,'production_admitted':False,'reused_validation':True,'freeze_sha256':sha(D/'freeze.json')}
    (D/'evaluation.json').write_text(json.dumps(out,indent=2));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
