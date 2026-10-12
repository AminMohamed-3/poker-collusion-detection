"""Frozen higher-precision preflop equity screen; fixed stage1 and top20."""
from audit_common import *
import time
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT=HERE/'experiments/preflop_equity_precision_20260912'
FEATURE_DIR=HERE/'research/preflop_equity_precision_20260912'
PRIMARY='equity4096'

def main():
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Write-once study already started'
    feature_path=FEATURE_DIR/'features4096.parquet'
    new=pl.read_parquet(feature_path).sort('pair_id','hand_id')
    fields=[c for c in new.columns if c not in ('pair_id','hand_id')]
    assert len(fields)==24 and all(c.startswith('eq_') for c in fields)
    receipt=json.loads((FEATURE_DIR/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['exact_float32_256_parity'] and receipt['features4096_sha256']==sha(feature_path)
    protected=check_protected()
    sources=[Path(__file__),HERE/'src/audit_common.py',HERE/'src/preflop_equity_precision_features.py',feature_path,FEATURE_DIR/'receipt.json',PREV/'scores.parquet']
    write(OUT/'freeze.json',{
        'created_at':datetime.now(timezone.utc).isoformat(),'primary':PRIMARY,
        'hypothesis':'Lower preflop equity MonteCarlo noise may improve hand-evidence ordering. Exact postflop engine and every other feature/model decision remain unchanged.',
        'arms':['incumbent',PRIMARY,'stage1_only'],'features_replaced':fields,
        'sample_count':4096,'sample_seed':408331,'params':dict(CLF,random_state=11),
        'data':'Same162query training-only audit outside unionouter0/1/2; unchanged383stage1 scores and top20; per-family whole-pool5folds.',
        'reference':'Complete current hybrid. DT/SP use existing24equity fields as part ofr382; replace those only. CI has no equityinputs in currentrecipe so its scores remain exactincumbent.',
        'budget':'15referencefits plus10DT/SPcandidatefits,2CPU,no parameter/seed/family/weight/cutoff sweep or blend.',
        'gate':{'minimum_map_gain':.01,'pool_bootstrap_lower_gt':0,'family_floor':-.01,'both_fixed_halves_nonnegative':True},
        'limits':['Adaptively reused trainingaudit and sharedstage1OOF; not independent/nestedconfirmation.','4096runouts remains approximate.','Fixedshortlist isolates downstreamfeatureprecision; it does not test higherprecisionstage1retrieval.','No outertruth or evaluationdata or publicoutcome used.'],
        'source_hashes':{str(p):sha(p) for p in sources},'protected':protected})
    d,shared,rcols,saved=load_current()
    expected=d.filter(pl.col('oof_rank')<=20).select('pair_id','hand_id')
    assert expected.equals(new.select('pair_id','hand_id'))
    cached=pl.read_parquet(FEATURE_DIR/'baseline256-cached.parquet').sort('pair_id','hand_id')
    assert cached.select('pair_id','hand_id').equals(expected)
    assert np.array_equal(cached.select(fields).to_numpy(),d.filter(pl.col('oof_rank')<=20).select(fields).to_numpy())
    highcols={c:c+'__4096' for c in fields}
    d=d.join(new.rename(highcols),on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','hand_id')
    assert d.select('pair_id','hand_id').equals(saved.select('pair_id','hand_id'))
    preds={name:d['oof'].to_numpy().copy()-1e6 for name in ('incumbent',PRIMARY)}
    receipts=[]
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy())
        q=d[ix]; cols=shared+(rcols if fam!='coordinated_isolation' else [])
        changed=[highcols.get(c,c) for c in cols]
        x=q.select(cols).to_numpy().astype(float); h=q.select(changed).to_numpy().astype(float)
        assert np.isfinite(x).all() and np.isfinite(h).all()
        if fam=='coordinated_isolation': assert np.array_equal(x,h)
        y=q['relevance'].to_numpy().astype(int); groups=q['table_id'].to_numpy()
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,groups)):
            assert not set(groups[tr])&set(groups[te])
            ref=lgb.LGBMClassifier(**CLF,random_state=11).fit(x[tr],y[tr])
            base=ref.predict_proba(x[te])[:,1]+10
            assert np.array_equal(base,saved['incumbent'].to_numpy()[ix[te]])
            preds['incumbent'][ix[te]]=base
            if fam=='coordinated_isolation':
                preds[PRIMARY][ix[te]]=base
                receipts.append({'family':fam,'fold':fold,'reference_exact':True,'candidate_exact_no_input_change':True})
                continue
            candidate=lgb.LGBMClassifier(**CLF,random_state=11).fit(h[tr],y[tr])
            path=OUT/f'{fam}-{fold}.txt'; candidate.booster_.save_model(str(path))
            p=candidate.predict_proba(h[te])[:,1]
            replay=lgb.Booster(model_file=str(path)).predict(h[te],num_threads=2)
            assert np.array_equal(p,replay)
            preds[PRIMARY][ix[te]]=p+10
            receipts.append({'family':fam,'fold':fold,'reference_exact':True,'reload_exact':True,'model_sha256':sha(path)})
        print('Finished',fam,round(time.perf_counter()-start,1),flush=True)
    scorefile=OUT/'scores.parquet'
    d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(k,v) for k,v in preds.items()]).write_parquet(scorefile)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(scorefile),'controls':receipts,'outer_truth_read':False})
    result=evaluate(d,dict(preds,stage1_only=d['oof'].to_numpy()))
    result['primary_eligible']=result['comparisons'][PRIMARY]['gate_pass']
    result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass']
    result['protected_unchanged']=check_protected()==protected
    result['seconds']=time.perf_counter()-start
    write(OUT/'results.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
