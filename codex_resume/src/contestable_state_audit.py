"""Frozen all-in and contestable-pot evidence screen against current hybrid."""
from audit_common import *
import time
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT=HERE/'experiments'/'contestable_state_20260912'
FEATURE_DIR=HERE/'research'/'contestable_state_20260912'
PRIMARY='contestable_state'

def main():
    start=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(), 'Write-once study already started.'
    feature_path=FEATURE_DIR/'features.parquet'
    new=pl.read_parquet(feature_path).sort('pair_id','hand_id')
    newcols=[c for c in new.columns if c not in ('pair_id','hand_id')]
    assert 0<len(newcols)<=35
    protected=check_protected()
    sources=[Path(__file__),HERE/'src'/'audit_common.py',HERE/'src'/'contestable_state_features.py',feature_path,PREV/'scores.parquet']
    write(OUT/'freeze.json',{
        'created_at':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'Public contribution and remaining-stack state may distinguish legitimate all-in/side-pot decisions from suspicious-looking action patterns in existing simplified equity features.',
        'primary':PRIMARY,'arms':['incumbent',PRIMARY,'stage1_only'],
        'data':'Unchanged162query training-only audit outside union outer0/1/2; unchanged current383stage1 top20; whole-pool5folds per family.',
        'features':newcols,'params':dict(CLF,random_state=11),
        'reference':'Exact complete current hybrid; DT/SP shared+r382, CI shared. Add entire frozen state feature set in every family.',
        'budget':'15 exact reference fits plus15 candidate fits;2CPU; no sweep, blend, family selection or early stopping.',
        'gate':{'minimum_map_gain':.01,'pool_bootstrap_lower_gt':0,'family_floor':-.01,'both_fixed_halves_nonnegative':True},
        'limits':['Audit is adaptively reused; shared first-stage OOF is not nested confirmation.','Unlisted hands remain noisy comparison labels, not certified negatives.','Contestable-pot features are observable chip-accounting proxies, not strategic expected values.','No outer truth, evaluation features or public outcome used in fitting or selection.'],
        'source_hashes':{str(p):sha(p) for p in sources},'protected':protected})
    d,shared,rcols,saved=load_current()
    expected=d.filter(pl.col('oof_rank')<=20).select('pair_id','hand_id')
    assert expected.equals(new.select('pair_id','hand_id')),'Feature key set or ordering mismatch'
    assert not set(newcols)&set(d.columns),'Feature names collide with incumbent'
    d=d.join(new,on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','hand_id')
    assert d.select('pair_id','hand_id').equals(saved.select('pair_id','hand_id'))
    preds={k:d['oof'].to_numpy().copy()-1e6 for k in ('incumbent',PRIMARY)}
    controls=[]
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy())
        q=d[ix]; cols=shared+(rcols if fam!='coordinated_isolation' else [])
        x=q.select(cols).to_numpy().astype(float)
        augmented=q.select(cols+newcols).to_numpy().astype(float)
        assert np.isfinite(x).all() and np.isfinite(augmented).all()
        y=q['relevance'].to_numpy().astype(int); groups=q['table_id'].to_numpy()
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,groups)):
            assert not set(groups[tr])&set(groups[te])
            ref=lgb.LGBMClassifier(**CLF,random_state=11).fit(x[tr],y[tr])
            p=ref.predict_proba(x[te])[:,1]+10
            assert np.array_equal(p,saved['incumbent'].to_numpy()[ix[te]]),'Reference mismatch'
            preds['incumbent'][ix[te]]=p
            model=lgb.LGBMClassifier(**CLF,random_state=11).fit(augmented[tr],y[tr])
            p=model.predict_proba(augmented[te])[:,1]
            path=OUT/f'{fam}-{fold}.txt'; model.booster_.save_model(str(path))
            reload=lgb.Booster(model_file=str(path)); replay=reload.predict(augmented[te],num_threads=2)
            assert np.array_equal(p,replay),'Serialized model mismatch'
            preds[PRIMARY][ix[te]]=p+10
            controls.append({'family':fam,'fold':fold,'reference_exact':True,'reload_exact':True,'model_sha256':sha(path)})
        print('Finished',fam,round(time.perf_counter()-start,1),flush=True)
    scorefile=OUT/'scores.parquet'
    d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(k,v) for k,v in preds.items()]).write_parquet(scorefile)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(scorefile),'controls':controls,'outer_truth_read':False})
    result=evaluate(d,dict(preds,stage1_only=d['oof'].to_numpy()))
    result['primary_eligible']=result['comparisons'][PRIMARY]['gate_pass']
    result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass']
    result['protected_unchanged']=check_protected()==protected
    result['seconds']=time.perf_counter()-start
    write(OUT/'results.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
