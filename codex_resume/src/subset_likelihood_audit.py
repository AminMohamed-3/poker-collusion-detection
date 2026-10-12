"""One frozen exact-subset-loss audit versus the unchanged complete incumbent."""
from audit_common import *
from subset_likelihood import subset_nll
import time
from datetime import datetime,timezone
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
OUT=HERE/'experiments/subset_likelihood_20260912'
MECH=HERE/'research/subset_likelihood_20260912/mechanical-receipt.json'
PROTOCOL=HERE/'research/subset_likelihood_protocol_20260912.md'
BEST=CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
BEST_SHA='6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b'


def make_objective(pair_ids,expected_labels):
    pair_ids=np.asarray(pair_ids);expected_labels=np.asarray(expected_labels,dtype=float)
    boundaries=np.r_[0,np.flatnonzero(pair_ids[1:]!=pair_ids[:-1])+1,len(pair_ids)]
    assert len(set(pair_ids))==len(boundaries)-1 and np.all(np.diff(boundaries)==20)
    calls=[0]
    def objective(y_true,y_pred):
        assert np.array_equal(y_true,expected_labels) and np.isfinite(y_pred).all()
        assert np.max(np.abs(y_pred))<1000,'Outside bounded score range for this audit'
        if calls[0]==0:assert np.all(y_pred==0),'Custom objective did not start at zero'
        gradient=np.empty(len(y_true));hessian=np.empty(len(y_true))
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            _,gradient[a:b],hessian[a:b],_=subset_nll(y_true[a:b],y_pred[a:b])
        assert np.isfinite(gradient).all() and np.isfinite(hessian).all() and np.min(hessian)>=0
        calls[0]+=1
        return gradient,hessian
    return objective,calls,boundaries


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Write-once objective study'
    mech=json.loads(MECH.read_text(encoding='utf-8'))
    assert mech['status']=='mechanics_passed_no_model_fits' and mech['model_fits']==0
    for p,digest in mech['sources'].items():assert sha(Path(p))==digest
    assert sha(BEST)==BEST_SHA
    protected=check_protected()
    d,shared,rcols,saved=load_current()
    sources=[Path(__file__),HERE/'src/subset_likelihood.py',HERE/'src/audit_common.py',MECH,PROTOCOL,PREV/'scores.parquet',PREV/'predictions-frozen.json',BEST]
    write(OUT/'freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'arms':['incumbent','subset_nll','stage1_only'],'fits':30,'parameters':dict(CLF,random_state=11),'objective':'exact cardinality-conditioned subset NLL, exact diagonal curvature, training k only, zero initial raw score','cpu_threads':2,'features_by_family':{fam:shared+(rcols if fam!='coordinated_isolation' else []) for fam in FAMILIES},'source_hashes':{str(p):sha(p) for p in sources},'protected':protected,'gate':{'map5_gain_at_least':.01,'pool_ci_lower_gt':0,'family_floor':-.01,'both_pool_halves_nonnegative':True},'limits':['Adaptive training-only audit; shared stage1 OOF is not nested.','Observed membership is not certified manipulation truth.','Diagonal curvature approximates coupled full Hessian.','No validation/test k or evidence used in inference.','No outer evidence/public evaluation read.']})
    pred={name:d['oof'].to_numpy().copy()-1e6 for name in ('incumbent','subset_nll')}
    controls=[];fills=np.zeros(d.height,dtype=int)
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy())
        q=d[ix];cols=shared+(rcols if fam!='coordinated_isolation' else [])
        x=q.select(cols).to_numpy().astype(float);y=q['relevance'].to_numpy().astype(int);groups=q['table_id'].to_numpy()
        assert np.isfinite(x).all()
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,groups)):
            assert not set(groups[tr])&set(groups[te])
            ref=lgb.LGBMClassifier(**CLF,random_state=11).fit(x[tr],y[tr])
            rp=ref.predict_proba(x[te])[:,1]+10
            assert np.array_equal(rp,saved['incumbent'].to_numpy()[ix[te]])
            path=OUT/f'incumbent-{fam}-{fold}.txt';ref.booster_.save_model(str(path))
            assert np.array_equal(rp,lgb.Booster(model_file=str(path)).predict(x[te],num_threads=2)+10)
            pred['incumbent'][ix[te]]=rp
            controls.append({'arm':'incumbent','family':fam,'fold':fold,'reference_exact':True,'reload_exact':True,'model_sha256':sha(path)})
            objective,calls,boundaries=make_objective(q['pair_id'].to_numpy()[tr],y[tr])
            model=lgb.LGBMRegressor(**CLF,random_state=11,objective=objective).fit(x[tr],y[tr])
            assert calls[0]==CLF['n_estimators']
            score=model.predict(x[te]);assert np.isfinite(score).all() and np.min(score)>-1000
            path=OUT/f'subset_nll-{fam}-{fold}.txt';model.booster_.save_model(str(path))
            assert np.array_equal(score,lgb.Booster(model_file=str(path)).predict(x[te],num_threads=2))
            pred['subset_nll'][ix[te]]=score+10
            fills[ix[te]]+=1
            kp=OUT/f'training-keys-{fam}-{fold}.parquet';q[tr].select('pair_id','hand_id','table_id').write_parquet(kp)
            controls.append({'arm':'subset_nll','family':fam,'fold':fold,'reload_exact':True,'training_queries':len(boundaries)-1,'training_rows':len(tr),'objective_calls':calls[0],'training_keys_sha256':sha(kp),'model_sha256':sha(path)})
        print('Finished',fam,round(time.perf_counter()-start,1),flush=True)
    top=(d['oof_rank']<=20).to_numpy()
    assert np.all(fills[top]==1) and np.all(fills[~top]==0),'Prediction coverage mismatch'
    pred['stage1_only']=d['oof'].to_numpy()
    path=OUT/'scores.parquet';d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(n,v) for n,v in pred.items()]).write_parquet(path)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(path),'controls':controls,'outer_truth_read':False})
    result=evaluate(d,pred)
    result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass'];assert result['negative_control_rejected']
    result['protected_unchanged']=protected==check_protected() and sha(BEST)==BEST_SHA;assert result['protected_unchanged']
    result['seconds']=time.perf_counter()-start
    write(OUT/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
