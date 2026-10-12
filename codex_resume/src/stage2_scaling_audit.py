"""Frozen stage-2 label-budget and capacity diagnosis; current upstream cache held fixed."""
from audit_common import *
import time, hashlib, math
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT=HERE/'experiments/stage2_scaling_20260912'
ARMS=('incumbent','labels25','labels50','larger')

def pool_order(pools):
    return sorted(pools,key=lambda p:(hashlib.sha256(('stage2-label-curve-93761|'+p).encode()).hexdigest(),p))

def bce(y,p):
    z=np.clip(np.asarray(p,dtype=float),1e-15,1-1e-15)
    return float(np.mean(-y*np.log(z)-(1-y)*np.log1p(-z)))

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Write-once frozen study'
    best=CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
    bestsha='6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b'
    assert sha(best)==bestsha
    protected=check_protected()
    d,shared,rcols,saved=load_current()
    inputfile=OUT/'input-snapshot.parquet';d.write_parquet(inputfile)
    params={n:dict(CLF,random_state=11) for n in ARMS}
    params['larger'].update(n_estimators=300,num_leaves=15,max_depth=4)
    tasks=[];design=[]
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy())
        q=d[ix];cols=shared+(rcols if fam!='coordinated_isolation' else [])
        x=q.select(cols).to_numpy().astype(float);y=q['relevance'].to_numpy().astype(int);g=q['table_id'].to_numpy()
        assert np.isfinite(x).all()
        assert all(v==20 for v in q.group_by('pair_id').len()['len'])
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,g)):
            training=pool_order(set(g[tr]));held=set(g[te]);assert not set(training)&held
            fractions={'incumbent':1.,'labels25':.25,'labels50':.5,'larger':1.}
            sets={name:set(training[:math.ceil(frac*len(training))]) for name,frac in fractions.items()}
            assert sets['labels25']<=sets['labels50']<=sets['incumbent']==sets['larger']
            for name in ARMS:
                use=tr[np.isin(g[tr],sorted(sets[name]))]
                assert len(np.unique(y[use]))==2,'Stop: selected training subset missing a class'
                assert not set(g[use])&held
                path=OUT/f'training-keys-{name}-{fam}-{fold}.parquet'
                q[use].select('pair_id','hand_id','table_id').write_parquet(path)
                row={'arm':name,'family':fam,'fold':fold,'training_pools':sorted(sets[name]),'held_pools':sorted(held),'train_queries':q[use]['pair_id'].n_unique(),'held_queries':q[te]['pair_id'].n_unique(),'train_rows':len(use),'held_rows':len(te),'features':cols,'training_keys_file':str(path),'training_keys_sha256':sha(path)}
                design.append(row);tasks.append((row,ix,x,y,use,te))
    assert len(tasks)==60
    protocol=HERE/'research/stage2_scaling_protocol_20260912.md'
    sources=[Path(__file__),HERE/'src/audit_common.py',protocol,inputfile,PREV/'scores.parquet',PREV/'predictions-frozen.json',best]
    write(OUT/'freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'arms':list(ARMS),'parameters':params,'fits':60,'cpu_threads':2,'eligible_candidates':['larger'],'diagnostic_only':['labels25','labels50'],'protocol_sha256':sha(protocol),'source_hashes':{str(p):sha(p) for p in sources},'design':design,'gate':{'mean_delta_at_least':.01,'pool_ci_lower_gt':0,'family_floor':-.01,'both_halves_nonnegative':True},'protected':protected,'limits':['Reused nonnested audit, not a pristine holdout.','Existing upstream OOF is fixed; reduced-label arms withhold only stage-2 labels.','One fixed nested pool ordering; intervals condition on this ordering and omit selection/training randomness.','Larger changes trees/leaves/depth jointly; not a neural scaling law.']})
    pred={name:d['oof'].to_numpy().copy()-1e6 for name in ARMS}
    fills={name:np.zeros(d.height,dtype=int) for name in ARMS};receipts=[];losses=[]
    for row,ix,x,y,use,te in tasks:
        name=row['arm'];fam=row['family'];fold=row['fold']
        model=lgb.LGBMClassifier(**params[name]).fit(x[use],y[use])
        p=model.predict_proba(x[te])[:,1];trainp=model.predict_proba(x[use])[:,1]
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        mp=OUT/f'{name}-{fam}-{fold}.txt';model.booster_.save_model(str(mp))
        reload=lgb.Booster(model_file=str(mp));replay=reload.predict(x[te],num_threads=2)
        assert np.array_equal(p,replay)
        item={'arm':name,'family':fam,'fold':fold,'reload_exact':True,'model_sha256':sha(mp)}
        if name=='incumbent':
            assert np.array_equal(p+10,saved['incumbent'].to_numpy()[ix[te]])
            item['reference_exact']=True
        pred[name][ix[te]]=p+10;fills[name][ix[te]]+=1;receipts.append(item)
        losses.append({'arm':name,'family':fam,'fold':fold,'train_queries':row['train_queries'],'held_queries':row['held_queries'],'train_bce':bce(y[use],trainp),'held_bce':bce(y[te],p)})
        if fold==4 and name=='larger':print('Completed',fam,round(time.perf_counter()-start,1),flush=True)
    top=(d['oof_rank']<=20).to_numpy()
    assert all(np.all(f[top]==1) and np.all(f[~top]==0) for f in fills.values())
    assert all(np.array_equal(p[~top],d['oof'].to_numpy()[~top]-1e6) for p in pred.values())
    pred['stage1_only']=d['oof'].to_numpy()
    scores=OUT/'scores.parquet';d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(n,p) for n,p in pred.items()]).write_parquet(scores)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(scores),'controls':receipts,'outer_truth_read':False,'top20_all_once':True,'tail_unchanged':True})
    result=evaluate(d,pred)
    result['passed']=['larger'] if result['comparisons']['larger']['gate_pass'] else []
    result['selected_for_confirmation']='larger' if result['passed'] else None
    result['learning_curve']={}
    for name in ('labels25','labels50'):
        diff=evaluate(d,{'incumbent':pred[name],'all_stage2_labels':pred['incumbent']})['comparisons']['all_stage2_labels']
        diff.pop('gate_pass',None);result['learning_curve']['incumbent_minus_'+name]=diff
    result['loss_by_fit']=losses
    result['loss_summary']={name:{key:float(np.average([l[key] for l in losses if l['arm']==name],weights=[l[weight] for l in losses if l['arm']==name])) for key,weight in (('train_bce','train_queries'),('held_bce','held_queries'))} for name in ARMS}
    result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass'];assert result['negative_control_rejected']
    result['protected_unchanged']=check_protected()==protected and sha(best)==bestsha;assert result['protected_unchanged']
    for path,digest in json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))['source_hashes'].items():assert sha(path)==digest
    result['seconds']=time.perf_counter()-start;write(OUT/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='loss_by_fit'},indent=2))

if __name__=='__main__':main()
