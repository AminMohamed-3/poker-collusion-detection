"""Two supervised action policies; owner release required. No automatic fit."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '2'
import sys, json, hashlib, time, importlib.util, shutil
from pathlib import Path
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from scipy.stats import rankdata

H = Path(__file__).resolve().parent
P = H.parents[2]
OLD = H.parent / 'private_card_evidence'
L = H.parent / 'private_card_likelihood'
AK = ['pair_id','direction','hand_id','action_no']
K = ['pair_id','hand_id']
spec = importlib.util.spec_from_file_location('frozen_evidence_reference', OLD / 'study.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
E.H = H  # Share exact frozen verifier/evaluator, but all writes go to NEW study.
sha = E.sha
write = E.write

def load_scopes():
    old_manifest = json.loads((OLD / 'MANIFEST.json').read_text())
    for p,h in old_manifest['sources'].items():
        assert sha(P/p) == h, p
    for p,h in old_manifest['artifacts'].items():
        assert sha(OLD/p) == h, p
    receipt = json.loads((OLD / 'FIT_RECEIPT.json').read_text())
    assert receipt['manifest_sha256'] == sha(OLD/'MANIFEST.json')
    for p,h in receipt['artifacts'].items():
        assert sha(OLD/p) == h, p
    d = pl.read_parquet([P/k for k in old_manifest['cache_paths']]).join(
        pl.read_parquet(L/'row_membership.parquet').select(*AK,'actor_id'), on=AK,validate='1:1')
    train = d.join(pl.read_parquet(OLD/'training_tables.parquet'),on='table_id',validate='m:1').join(
        pl.read_parquet(L/'capped_training_keys.parquet').select(AK),on=AK,how='semi')
    helper = E.helper()
    train = helper.weights(train).with_row_index('_i')
    held = d.join(pl.read_parquet(OLD/'candidate_keys.parquet'),on=K+['table_id'],how='semi')
    E.disjoint(train,held)
    saved = pl.read_parquet(OLD/'actual_training_membership.parquet')
    assert train.select(saved.columns).equals(saved)
    membership = json.loads((OLD/'membership.json').read_text())
    assert len(membership['audit_tables'])==104 and len(membership['train_tables'])==293
    assert set(train['table_id'])==set(membership['train_tables'])
    assert not set(train['table_id']).intersection(membership['audit_tables'])
    for item in receipt['membership']:
        j=item['inner']; fitted=helper.weights(train.filter(pl.col('inner_fold')!=j)); scored=train.filter(pl.col('inner_fold')==j)
        E.disjoint(fitted,scored)
        assert sorted(fitted['table_id'].unique().to_list())==item['train_tables']
        assert sorted(scored['table_id'].unique().to_list())==item['score_tables']
        assert hashlib.sha256(fitted.select(AK).write_csv().encode()).hexdigest()==item['train_key_sha256']
    models=joblib.load(OLD/'models.joblib')
    inputs=old_manifest['inputs']
    for key in ['strength_inner0','strength_inner1','strength_full']:
        assert models[key].n_features_in_==len(inputs)==37
    return train,held,models,old_manifest

def prepare():
    assert not (H/'MANIFEST.json').exists()
    train,held,models,old_manifest=load_scopes()
    for name in ['training_tables.parquet','reference.parquet','candidate_keys.parquet','membership.json']:
        assert not (H/name).exists()
        shutil.copyfile(OLD/name,H/name)
    ref=pl.read_parquet(H/'reference.parquet')
    assert ref.height==19325 and ref['pair_id'].n_unique()==162
    assert held.select(K).unique().height==3240
    sources={k:v for k,v in old_manifest['sources'].items()}
    for path in [OLD/'MANIFEST.json',OLD/'FIT_RECEIPT.json',OLD/'models.joblib',OLD/'actual_training_membership.parquet',OLD/'study.py',OLD/'membership.json',OLD/'training_tables.parquet',OLD/'candidate_keys.parquet',OLD/'reference.parquet',P/'data/development_labels.csv']:
        sources[str(path.relative_to(P)).replace('\\','/')]=sha(path)
    write('REUSE_LINEAGE.json',{'all104_audit_excluded':True,'train_tables':293,'actual_training_membership_exact':True,'inner_table_and_key_lineage_exact':True,'nuisance_models_sha256':sha(OLD/'models.joblib'),'new_fits':0,'quality_labels_read':False})
    names=['study.py','PROTOCOL.md','training_tables.parquet','reference.parquet','candidate_keys.parquet','membership.json','REUSE_LINEAGE.json']
    write('MANIFEST.json',{'inputs':old_manifest['inputs'],'sources':sources,'artifacts':{n:sha(H/n) for n in names},'cache_paths':old_manifest['cache_paths'],'fits':2,'runtime_seconds_limit':900,'rank_weights':[9,1]})
    print('Prepared two-policy evidence screen; zero fits or quality reads.')

def policy_weights(frame):
    counts=frame.group_by('pair_id','direction').len().rename({'len':'_pd_n'})
    frame=frame.join(counts,on=['pair_id','direction'],validate='m:1')
    weights=1/frame['_pd_n'].to_numpy(); weights/=weights.mean()
    mass=frame.select('pair_id','direction').with_columns(pl.Series('w',weights)).group_by('pair_id','direction').agg(pl.sum('w'))['w'].to_numpy()
    assert np.allclose(mass,mass[0])
    return frame,weights

def fit():
    m=E.verify(True)
    assert not (H/'FIT_STARTED.json').exists()
    write('FIT_STARTED.json',{'manifest_sha256':sha(H/'MANIFEST.json')})
    start=time.monotonic(); train,held,nuisance,_=load_scopes(); inputs=m['inputs']
    def timed():
        assert time.monotonic()-start<m['runtime_seconds_limit']
    residual_prediction=np.empty(train.height)
    for j in [0,1]:
        scored=train.filter(pl.col('inner_fold')==j)
        residual_prediction[scored['_i'].to_numpy()]=nuisance[f'strength_inner{j}'].predict(scored.select(inputs).to_numpy())
    train=train.with_columns(pl.Series('private_surprise',train['partner_strength'].to_numpy()-residual_prediction))
    held=held.with_columns(pl.Series('private_surprise',held['partner_strength'].to_numpy()-nuisance['strength_full'].predict(held.select(inputs).to_numpy())))
    # Join labels only onto the verified 293-table training rows.
    labels=pl.read_csv(P/'data/development_labels.csv').select('pair_id','label')
    train=train.join(labels,on='pair_id',validate='m:1')
    assert train['label'].null_count()==0 and set(train['label'])=={0,1}
    xheld=held.select(inputs+['private_surprise']).to_numpy()
    models={}; probs={}; artifacts=[]
    for name,label in [('target',1),('normal',0)]:
        subset,w=policy_weights(train.filter(pl.col('label')==label)); timed()
        assert set(subset['action_class'])=={0,1,2,3}
        model=lgb.LGBMClassifier(n_estimators=200,num_leaves=15,min_child_samples=50,learning_rate=.1,random_state=35935,n_jobs=2,verbosity=-1)
        model.fit(subset.select(inputs+['private_surprise']).to_numpy(),subset['action_class'].to_numpy(),sample_weight=w)
        assert np.array_equal(model.classes_,np.arange(4))
        q=model.predict_proba(xheld)
        assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-6)
        models[name]=model; probs[name]=q
        path=f'train_membership_{name}.parquet'; artifacts.append(path)
        subset.select(*AK,'table_id','actor_id','inner_fold').with_columns(pl.Series('fit_weight',w)).write_parquet(H/path)
    joblib.dump(models,H/'models.joblib'); reloaded=joblib.load(H/'models.joblib')
    assert all(np.array_equal(reloaded[name].predict_proba(xheld),q) for name,q in probs.items())
    y=held['action_class'].to_numpy();ix=np.arange(held.height)
    gain=np.log(np.maximum(probs['target'][ix,y],1e-12))-np.log(np.maximum(probs['normal'][ix,y],1e-12))
    actions=held.select(*AK,'table_id','action_class','private_surprise').with_columns(pl.Series('gain',gain))
    for name,q in probs.items():
        actions=actions.with_columns(*[pl.Series(f'{name}_p{k}',q[:,k]) for k in range(4)])
    actions.write_parquet(H/'action_predictions.parquet')
    hands=actions.group_by(K+['direction']).agg(pl.mean('gain')).group_by(K).agg(pl.mean('gain'))
    assert hands.height==3240
    hands.write_parquet(H/'hand_predictions.parquet')
    ref=pl.read_parquet(H/'reference.parquet').sort(K).join(hands,on=K,how='left',validate='1:1',maintain_order='left')
    candidate=ref['current'].to_numpy().copy()
    for _,g in ref.with_row_index('_row').filter(pl.col('is_top20')).group_by('pair_id'):
        assert g.height==20 and g['gain'].null_count()==0
        rb=np.rint(2*rankdata(g['current'].to_numpy(),method='average')).astype(np.int64)
        rp=np.rint(2*rankdata(g['gain'].to_numpy(),method='average')).astype(np.int64)
        candidate[g['_row'].to_numpy()]=10+(9*rb+rp)/400.
    outside=~ref['is_top20'].to_numpy()
    assert np.array_equal(candidate[outside],ref['current'].to_numpy()[outside])
    ref.with_columns(pl.Series('candidate',candidate)).write_parquet(H/'predictions.parquet');timed()
    artifacts+=['models.joblib','action_predictions.parquet','hand_predictions.parquet','predictions.parquet']
    write('FIT_RECEIPT.json',{'manifest_sha256':sha(H/'MANIFEST.json'),'seconds':time.monotonic()-start,'fits':2,'both_policy_reload_exact':True,'artifacts':{n:sha(H/n) for n in artifacts},'quality_reads':0})

def evaluate():
    E.evaluate()

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','fit','evaluate']
    globals()[sys.argv[1]]()
