"""Frozen outsider-card linkage screen versus the complete incumbent and hand-card permutation control."""
from audit_common import *
import time
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT=HERE/'experiments'/'outsider_cards_20260912'
FDIR=HERE/'research'/'outsider_decisions_20260912'
CANDIDATES=('linked','card_control')
MODES={name:name for name in CANDIDATES}

def main():
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(), 'Write-once study already started'
    receipt=json.loads((FDIR/'feature-receipt.json').read_text(encoding='utf-8'))
    assert sha(FDIR/'feature-freeze.json')==receipt['feature_freeze_sha256']
    assert receipt['status']=='complete' and receipt['keys']==3240 and receipt['fields_per_arm']==24
    ffreeze=json.loads((FDIR/'feature-freeze.json').read_text(encoding='utf-8'))
    for path,digest in ffreeze['sources'].items(): assert sha(Path(path))==digest,'extraction source changed'
    tables={}; featurefiles={}
    for name,mode in MODES.items():
        path=FDIR/f'{mode}.parquet';featurefiles[name]=path
        assert sha(path)==receipt['feature_hashes'][mode]
        tables[name]=pl.read_parquet(path).sort('pair_id','hand_id')
    newcols=ffreeze['features']; assert len(newcols)==24
    best=CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
    best_sha='6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b'
    assert sha(best)==best_sha
    protected=check_protected()
    sources=[Path(__file__),HERE/'src'/'audit_common.py',FDIR/'feature-freeze.json',FDIR/'feature-receipt.json',*featurefiles.values(),PREV/'scores.parquet',PREV/'predictions-frozen.json',best]
    write(OUT/'freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'eligible_candidates':list(CANDIDATES),'selection':'Larger mean delta among generic-gate passes; linked wins exact tie',
        'arms':['incumbent',*MODES,'stage1_only'],'protocol_sha256':sha(HERE/'research'/'outsider_cards_protocol_20260912.md'),
        'data':'Same162query training-only current383 stage1 audit/top20, original-pool GroupKFold5 per family; all24 fields added to each current complete hybrid branch.',
        'parameters':dict(CLF,random_state=11),'fits':45,'cpu_threads':2,
        'gate':{'map5_gain_at_least':.01,'pool_ci_lower_gt':0,'family_floor':-.01,'both_pool_halves_nonnegative':True},
        'features':newcols,'source_hashes':{str(p):sha(p) for p in sources},'protected':protected,
        'limits':['Adaptively reused audit, not nested confirmation.','Shared first-stage OOF makes this audit nonnested.','Both fixed arms eligible under the same gate; actor-card linkage mechanism is separate. No new arm or blend rescue.','No held evidence/evaluation data/public feedback used.','Actual-card minus permutation-control paired interval reported; point superiority alone is not a confirmed mechanism.']})
    d,shared,rcols,saved=load_current()
    expected=d.filter(pl.col('oof_rank')<=20).select('pair_id','hand_id')
    assert not set(newcols)&set(d.columns)
    aligned={}
    for name,table in tables.items():
        assert table.columns==['pair_id','hand_id']+newcols
        assert table.select('pair_id','hand_id').equals(expected)
        aligned[name]=d.select('pair_id','hand_id').join(table,on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','hand_id')
        assert aligned[name].select('pair_id','hand_id').equals(d.select('pair_id','hand_id'))
    pred={n:d['oof'].to_numpy().copy()-1e6 for n in ('incumbent',*MODES)}
    controls=[];fills={n:np.zeros(d.height,dtype=int) for n in pred}
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy())
        q=d[ix];cols=shared+(rcols if fam!='coordinated_isolation' else [])
        x=q.select(cols).to_numpy().astype(float); assert np.isfinite(x).all()
        augmented={n:np.column_stack([x,aligned[n][ix].select(newcols).to_numpy()]) for n in MODES}
        assert all(np.isfinite(z).all() for z in augmented.values())
        y=q['relevance'].to_numpy().astype(int);g=q['table_id'].to_numpy()
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,g)):
            assert not set(g[tr])&set(g[te])
            ref=lgb.LGBMClassifier(**CLF,random_state=11).fit(x[tr],y[tr]);p=ref.predict_proba(x[te])[:,1]+10
            assert np.array_equal(p,saved['incumbent'].to_numpy()[ix[te]])
            pred['incumbent'][ix[te]]=p
            fills['incumbent'][ix[te]]+=1
            refpath=OUT/f'incumbent-{fam}-{fold}.txt';ref.booster_.save_model(str(refpath))
            refreplay=lgb.Booster(model_file=str(refpath)).predict(x[te],num_threads=2)+10
            assert np.array_equal(p,refreplay),'Serialized reference mismatch'
            controls.append({'arm':'incumbent','family':fam,'fold':fold,'reference_exact':True,'reload_exact':True,'model_sha256':sha(refpath)})
            for name,z in augmented.items():
                model=lgb.LGBMClassifier(**CLF,random_state=11).fit(z[tr],y[tr]);p=model.predict_proba(z[te])[:,1]
                path=OUT/f'{name}-{fam}-{fold}.txt';model.booster_.save_model(str(path))
                replay=lgb.Booster(model_file=str(path)).predict(z[te],num_threads=2)
                assert np.array_equal(p,replay)
                pred[name][ix[te]]=p+10
                fills[name][ix[te]]+=1
                controls.append({'arm':name,'family':fam,'fold':fold,'reload_exact':True,'model_sha256':sha(path)})
        print('Finished',fam,round(time.perf_counter()-start,1),flush=True)
    top=(d['oof_rank']<=20).to_numpy()
    assert all(np.all(f[top]==1) and np.all(f[~top]==0) for f in fills.values()),'Score coverage mismatch'
    pred['stage1_only']=d['oof'].to_numpy()
    scores=OUT/'scores.parquet'
    d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(n,p) for n,p in pred.items()]).write_parquet(scores)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(scores),'controls':controls,'outer_truth_read':False})
    result=evaluate(d,pred)
    contrast=evaluate(d,{'incumbent':pred['card_control'],'linked_minus_control':pred['linked']})['comparisons']['linked_minus_control']
    result['mechanism_contrast_linked_minus_control']=contrast
    result['eligible_arms']=[name for name in CANDIDATES if result['comparisons'][name]['gate_pass']]
    result['passed']=list(result['eligible_arms'])
    result['selected_for_confirmation']=max(result['eligible_arms'],key=lambda name:(result['comparisons'][name]['mean_delta'],name=='linked')) if result['eligible_arms'] else None
    result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass']
    assert result['negative_control_rejected'],'Negative control failed to reject'
    result['protected_unchanged']=check_protected()==protected and sha(best)==best_sha
    assert result['protected_unchanged'],'Protected production file changed'
    result['seconds']=time.perf_counter()-start
    write(OUT/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
