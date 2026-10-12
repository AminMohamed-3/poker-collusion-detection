"""Frozen current-hybrid comparison of real-history means/random/pretrained encodings."""
from audit_common import *
from datetime import datetime,timezone
import time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT=HERE/'experiments/real_history_encoder_20260912'
FDIR=HERE/'research/real_history_encoder_20260912'
ARMS=('incumbent','history_mean','random_encoder','pretrained')


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'write-once study'
    cr=json.loads((FDIR/'corpus-receipt.json').read_text(encoding='utf-8'));tr=json.loads((FDIR/'training-receipt.json').read_text(encoding='utf-8'))
    assert cr['status']==tr['status']=='complete' and tr['models']==5
    assert sha(FDIR/'corpus-freeze.json')==cr['corpus_freeze_sha256'] and sha(FDIR/'training-freeze.json')==tr['training_freeze_sha256']
    cf=json.loads((FDIR/'corpus-freeze.json').read_text(encoding='utf-8'));tf=json.loads((FDIR/'training-freeze.json').read_text(encoding='utf-8'))
    for path,digest in cf['source_hashes'].items():assert sha(path)==digest
    for path,digest in tf['sources'].items():assert sha(path)==digest
    for name,digest in cr['artifacts'].items():assert sha(FDIR/name)==digest
    assert sha(FDIR/'random_encoder.npy')==tr['random_features_sha256'] and tr['random_reload_exact']
    keys=pl.read_parquet(FDIR/'target-keys.parquet').sort('pair_id','hand_id')
    raw=np.load(FDIR/'target-history.npz',allow_pickle=False)
    assert keys.equals(pl.DataFrame({'pair_id':raw['pair_id'],'hand_id':raw['hand_id']}))
    simple=pl.read_parquet(FDIR/'history_mean.parquet').sort('pair_id','hand_id');mcols=cr['simple_features'];ecols=[f'rh_encoder_{i}' for i in range(34)]
    assert simple.select('pair_id','hand_id').equals(keys) and len(mcols)==22
    random=np.load(FDIR/'random_encoder.npy',allow_pickle=False);assert random.shape==(3240,34)
    learned={};featurepaths=[FDIR/'history_mean.parquet',FDIR/'random_encoder.npy']
    for k in range(5):
        er=tr['receipts'][k];assert er['fold']==k and er['reload_exact'] and er['pair_swap_exact']
        path=FDIR/f'pretrained-{k}.npy';assert sha(path)==er['features_sha256'] and sha(FDIR/f'encoder-{k}.pt')==er['model_sha256']
        assert sha(FDIR/f'encoder-{k}-training-indices.npy')==tf['design'][k]['training_indices_sha256']
        assert not set(er['training_detail']['training_pools'])&set(cf['encoder_exclusions'][str(k)])
        learned[k]=np.load(path,allow_pickle=False);assert learned[k].shape==(3240,34);featurepaths.append(path)
    assert np.isfinite(random).all() and all(np.isfinite(z).all() for z in learned.values())
    best=CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv';bestsha='6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b';assert sha(best)==bestsha
    protected=check_protected()
    sources=[Path(__file__),HERE/'src/audit_common.py',HERE/'research/real_history_pretraining_protocol_20260912.md',FDIR/'corpus-freeze.json',FDIR/'corpus-receipt.json',FDIR/'training-freeze.json',FDIR/'training-receipt.json',*featurepaths,PREV/'scores.parquet',PREV/'predictions-frozen.json',best]
    write(OUT/'freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'source_hashes':{str(p):sha(p) for p in sources},'arms':list(ARMS),'eligible_arms':['history_mean','pretrained'],'extra_pretrained_gate':'paired pretrained-minus-random original-pool CI lower>0','selection':'Larger eligible mean gain; exact tie favors history_mean','parameters':dict(CLF,random_state=11),'fits':60,'features':{'history_mean':mcols,'random_encoder':ecols,'pretrained':ecols},'gate':{'minimum_delta':.01,'pool_ci_lower_gt':0,'family_floor':-.01,'both_halves_nonnegative':True},'protected':protected,'limits':['Reused nonnested upstream stage1/conditional cache, not full confirmation.','Every new encoder excludes original outer pools plus union of held pools for its detector fold.','Same encoder k gives both train and held features within detector fold k; never pool different latent coordinate systems.','All features describe observed histories, not identified manipulation or normal-play labels.','No outer evidence/public score read and no posthoc rescue.']})
    d,shared,rcols,saved=load_current();expected=d.filter(pl.col('oof_rank')<=20).select('pair_id','hand_id');assert keys.equals(expected)
    assert not set(mcols+ecols)&set(d.columns)
    def align(table,cols):
        f=d.select('pair_id','hand_id').join(table,on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','hand_id')
        assert f.select('pair_id','hand_id').equals(d.select('pair_id','hand_id'))
        return f.select(cols).to_numpy()
    meanx=align(simple,mcols)
    randomx=align(keys.with_columns(*[pl.Series(n,random[:,i]) for i,n in enumerate(ecols)]),ecols)
    learnedx={k:align(keys.with_columns(*[pl.Series(n,z[:,i]) for i,n in enumerate(ecols)]),ecols) for k,z in learned.items()}
    pred={n:d['oof'].to_numpy().copy()-1e6 for n in ARMS};fills={n:np.zeros(d.height,dtype=int) for n in ARMS};controls=[]
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy());q=d[ix]
        cols=shared+(rcols if fam!='coordinated_isolation' else []);x=q.select(cols).to_numpy().astype(float);y=q['relevance'].to_numpy().astype(int);g=q['table_id'].to_numpy()
        for k,(train,test) in enumerate(GroupKFold(n_splits=5).split(x,y,g)):
            design=next(item for item in cf['detector_design'] if item['family']==fam and item['fold']==k)
            assert set(g[train])==set(design['training_pools']) and set(g[test])==set(design['held_pools']) and not set(g[train])&set(g[test])
            assert not set(g[test])&set(tr['receipts'][k]['training_detail']['training_pools'])
            features={'incumbent':x,'history_mean':np.column_stack([x,meanx[ix]]),'random_encoder':np.column_stack([x,randomx[ix]]),'pretrained':np.column_stack([x,learnedx[k][ix]])}
            for name,z in features.items():
                assert np.isfinite(z).all();model=lgb.LGBMClassifier(**CLF,random_state=11).fit(z[train],y[train]);p=model.predict_proba(z[test])[:,1]
                path=OUT/f'{name}-{fam}-{k}.txt';model.booster_.save_model(str(path));replay=lgb.Booster(model_file=str(path)).predict(z[test],num_threads=2)
                assert np.array_equal(p,replay)
                check={'arm':name,'family':fam,'fold':k,'reload_exact':True,'model_sha256':sha(path)}
                if name=='incumbent':assert np.array_equal(p+10,saved['incumbent'].to_numpy()[ix[test]]);check['reference_exact']=True
                pred[name][ix[test]]=p+10;fills[name][ix[test]]+=1;controls.append(check)
        print('Completed',fam,round(time.perf_counter()-start,1),flush=True)
    top=(d['oof_rank']<=20).to_numpy();assert all(np.all(f[top]==1) and np.all(f[~top]==0) for f in fills.values())
    assert all(np.array_equal(p[~top],d['oof'].to_numpy()[~top]-1e6) for p in pred.values())
    pred['stage1_only']=d['oof'].to_numpy();scores=OUT/'scores.parquet'
    d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(n,p) for n,p in pred.items()]).write_parquet(scores)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(scores),'controls':controls,'top20_all_once':True,'tail_exact':True,'outer_truth_read':False})
    result=evaluate(d,pred)
    contrasts={name:evaluate(d,{'incumbent':pred[control],'pretrained':pred['pretrained']})['comparisons']['pretrained'] for name,control in [('pretrained_minus_random','random_encoder'),('pretrained_minus_mean','history_mean')]}
    result['contrasts']=contrasts
    eligible=[]
    if result['comparisons']['history_mean']['gate_pass']:eligible.append('history_mean')
    if result['comparisons']['pretrained']['gate_pass'] and contrasts['pretrained_minus_random']['ci95'][0]>0:eligible.append('pretrained')
    result['passed']=eligible;result['selected_for_confirmation']=max(eligible,key=lambda n:(result['comparisons'][n]['mean_delta'],n=='history_mean')) if eligible else None
    result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass'];assert result['negative_control_rejected']
    result['protected_unchanged']=check_protected()==protected and sha(best)==bestsha;assert result['protected_unchanged']
    for path,digest in json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))['source_hashes'].items():assert sha(path)==digest
    result['seconds']=time.perf_counter()-start;write(OUT/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
