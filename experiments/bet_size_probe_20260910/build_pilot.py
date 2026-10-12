"""One frozen training-only paired simulation test of conditional bet sizes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import numpy as np
H=Path(__file__).resolve().parent;E=H.parent;P=E/'softplay_capacity_20260910';N=E/'bet_size_realism_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((H/'result.json').read_text())['passes']
    # Confirm the physical one-big-blind floor does not erase the OOF advantage.
    aa=[];bb=[];coverage=[];folds=[]
    for i in range(4):
        f=np.load(H/f'family{i}-oof.npz');y=f['y'];pred=np.maximum(np.log(2),f['candidate'])
        loss=lambda p:np.mean([np.maximum(q*(y-p[:,j]),(q-1)*(y-p[:,j])) for j,q in enumerate([.1,.5,.9])],axis=0)
        a=loss(f['baseline']);b=loss(pred);aa.extend(a);bb.extend(b);coverage.extend((y>=pred[:,0])&(y<=pred[:,2]));folds.append(float((a-b).mean()))
    gain=1-np.mean(bb)/np.mean(aa);cov=float(np.mean(coverage));assert gain>=.05 and min(folds)>0 and .7<=cov<=.9
    (H/'projection-check.json').write_text(json.dumps({'one_bb_floor_relative_gain':gain,'coverage':cov,'fold_gains':folds,'passes':True},indent=2))
    N.mkdir(exist_ok=False);d=N/'sized';d.mkdir();src=P/'b0_capacity';pp=json.loads((src/'pilot-plan.json').read_text())
    for name,v in pp['source_hashes'].items():assert sha(src/name)==v;shutil.copy2(src/name,d/name)
    shutil.copy2(H/'quantile-models.joblib',d/'raise-quantiles.joblib')
    s=(d/'learned_policy.py').read_text();s=s.replace('BUNDLE=None','BUNDLE=None\nQUANTILES=None\nUSE_QUANTILE_SIZING=True')
    a=s.index('    amount=max(obs.blind,math.expm1(log_amount)*obs.blind)*rng.uniform(.8,1.2)')
    b=a+len('    amount=max(obs.blind,math.expm1(log_amount)*obs.blind)*rng.uniform(.8,1.2)')
    s=s[:a]+'''    if USE_QUANTILE_SIZING:
        low,high=quantiles()['models'][profile.family]
        values=sorted([float(low.predict(x,num_threads=1)[0]),log_amount,float(high.predict(x,num_threads=1)[0])])
        amount=math.expm1(sample_log_size(values,rng.random()))*obs.blind
    else:
        amount=max(obs.blind,math.expm1(log_amount)*obs.blind)*rng.uniform(.8,1.2)'''+s[b:]
    s+='''
def quantiles():
    global QUANTILES
    if QUANTILES is None:QUANTILES=joblib.load(HERE/'raise-quantiles.joblib')
    return QUANTILES

def sample_log_size(values,u):
    low,median,high=sorted(values)
    knots=[low-.25*(median-low),low,median,high,high+.25*(high-median)]
    return max(math.log(2),float(np.interp(u,[0.,.1,.5,.9,1.],knots)))
'''
    (d/'learned_policy.py').write_text(s,encoding='utf-8')
    pp['source_hashes']={name:sha(d/name) for name in [*pp['source_hashes'],'raise-quantiles.joblib']};pp['created_at']=datetime.now(timezone.utc).isoformat();pp['paired_existing_control']=str(src.resolve());pp['size_probe_sha256']=sha(H/'result.json');pp['size_sampling']='Linear quantile interpolation in log1p(amount/BB), linear tail extrapolation at0/1, oneBB floor; original legal amount/fraction bounds unchanged. One shared RNG draw replaces uniform jitter.';(d/'pilot-plan.json').write_text(json.dumps(pp,indent=2))
    for name in ['controller.joblib','soft_controller.py','conditional_action_features.py','advanced_features.py','causal_styles.py','audit_batch.py','replay_batch.py','prepare_features.py','run_generation.py','run_analysis.py']:shutil.copy2(P/name,N/name)
    shutil.copy2(P/'causal-style-tests.json',N/'inherited-causal-style-tests.json')
    plan={'at':datetime.now(timezone.utc).isoformat(),'generator_arms':['sized'],'generator_plan_sha256':{'sized':sha(d/'pilot-plan.json')},'batches':[{'arm':'sized','existing_control':str(src.resolve()),'pool_ids':list(range(640,672))}],'controller_sha256':sha(d/'controller.joblib'),'quantile_models_sha256':sha(d/'raise-quantiles.joblib'),'hands':96000,'local_workers':2,'hypothesis':'Fix conditional ordinary raise-size dispersion without changing soft action probabilities.','gate':'Preserve all unchanged soft-play realism rules and legal/causal audits; additionally require confirmed-none pair source AUC<=.85. This one development population cannot authorize utility scoring.','next_step':'If this initial batch passes, freeze broader paired/fresh-seed confirmation. If it fails, keep the failed result and inspect training-only size/pressure mismatches. No utility score in this pilot.','protected':json.loads((P/'study-plan.json').read_text())['protected']}
    (N/'study-plan.json').write_text(json.dumps(plan,indent=2));shutil.copy2(P/'diagnostics.py',N/'diagnostics.py');print(json.dumps({'directory':str(N),'one_bb_floor_gain':gain,'plan_sha256':sha(N/'study-plan.json')}))
if __name__=='__main__':main()
