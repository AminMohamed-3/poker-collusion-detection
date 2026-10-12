"""Fresh union-excluding normal-action probabilities for the ordered-surprise audit."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,warnings
from pathlib import Path
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore',message='X does not have valid feature names')
import numpy as np
import polars as pl
import lightgbm as lgb
from contestable_state_features import target_keys
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
SOURCE=ROOT/'reports/conditional-action-experiment'
OUT=HERE/'research/ordered_surprise_20260912'
AK=['pair_id','hand_id','action_no','direction']
SK=['pair_id','hand_id','table_id','direction','action_no','street_index','action_class']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def fold(pool):return int(hashlib.sha256(str(pool).encode()).hexdigest(),16)%5

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'reference-freeze.json').exists(),'write-once reference experiment'
    original=json.loads((SOURCE/'policy-plan.json').read_text(encoding='utf-8'))
    files=sorted((SOURCE/'action-cache').glob('batch_*.parquet'));assert len(files)==20
    for p in files:assert sha(p)==original['files'][p.relative_to(SOURCE).as_posix()]
    labels_path=ROOT/'data/development_labels.csv';assert sha(labels_path)==original['files']['development_labels.csv']
    styles=json.loads((OUT/'player-style-pool-receipt.json').read_text());assert styles['maximum_tables_per_player']==1
    heldpaths=[ROOT/'reports/tabicl-screen/held-pairs.csv',*[ROOT/f'reports/tabicl-confirmation/fold{i}/held-pairs.csv' for i in (1,2)]]
    outer=set()
    for path in heldpaths:
        part=set(pl.read_csv(path,columns=['table_id'])['table_id']);assert len(part)==100;outer.update(part)
    params=dict(original['parameters'],n_jobs=2)
    inputs=original['inputs'];assert len(inputs)==37 and not set(inputs)&{'pair_id','hand_id','table_id','action_class','partner_strength','observed_log_amount_bb'}
    shortlist=HERE/'experiments/symmetry_censor_20260912'
    shortlist_receipt=json.loads((shortlist/'predictions-frozen.json').read_text(encoding='utf-8'))
    assert sha(shortlist/'scores.parquet')==shortlist_receipt['scores_sha256']
    keys=target_keys()
    data=pl.read_parquet(files).sort(AK)
    assert data.select(AK).n_unique()==data.height
    allowed=set(data['table_id'])-outer
    candidates=data.filter(pl.col('table_id').is_in(sorted(allowed)))
    labels=pl.read_csv(labels_path,columns=['pair_id','label','label_status'])
    normal=candidates.join(labels,on='pair_id',validate='m:1').filter(pl.col('label')==0)
    assert set(normal['label_status'])=={'confirmed_non_target'}
    scored=data.join(keys,on=['pair_id','hand_id'],how='semi').sort(AK)
    assert scored.select(AK).n_unique()==scored.height
    assert not set(scored['table_id'])&outer and scored.select('pair_id','hand_id').unique().height==3240
    normal=normal.with_columns(pl.col('table_id').map_elements(fold,return_dtype=pl.Int64).alias('_fold'))
    scored=scored.with_columns(pl.col('table_id').map_elements(fold,return_dtype=pl.Int64).alias('_fold'))
    sources=[Path(__file__),HERE/'src/contestable_state_features.py',shortlist/'scores.parquet',shortlist/'predictions-frozen.json',SOURCE/'policy-plan.json',OUT/'player-style-pool-receipt.json',OUT/'player-pool-lookup.parquet',HERE/'research/ordered_surprise_protocol_20260912.md',*heldpaths,labels_path,*files]
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'parameters':params,'inputs':inputs,'outer_pools':sorted(outer),'allowed_pools':sorted(allowed),'score_pairs':scored['pair_id'].n_unique(),'score_rows':scored.height,'group_fold':'int(full_SHA256_hex(table_id),16)%5','sample_seed':55193,'sample_cap_per_pair':128,'sample_cap_total':100000,'sources':{str(p):sha(p) for p in sources},'labels_used':'Only confirmed non-target rows outside outer union and score fold train references; no evidence truth read.','public_style':'Full other-hand development style retained; each development player belongs to one pool, so training style cannot draw on held-pool players. Scored-pool gameplay is a permissible descriptor input.'}
    write(OUT/'reference-freeze.json',freeze)
    pred=np.full((scored.height,4),np.nan);receipts=[]
    for k in range(5):
        train=normal.filter(pl.col('_fold')!=k).sort(AK)
        rng=np.random.default_rng(55193)
        train=(train.with_columns(pl.Series('_sample',rng.random(train.height))).sort('pair_id','_sample').group_by('pair_id',maintain_order=True).head(128))
        if train.height>100000:train=train.sample(n=100000,seed=55193)
        train=train.sort(AK)
        ix=np.flatnonzero((scored['_fold']==k).to_numpy());q=scored[ix]
        tp,sp=set(train['table_id']),set(q['table_id'])
        assert len(ix)>0 and train.height>0 and not tp&sp and not tp&outer and not sp&outer
        assert all(fold(t)!=k for t in tp) and all(fold(t)==k for t in sp)
        assert train.group_by('pair_id').len()['len'].max()<=128 and train.height<=100000
        x=train.select(inputs).to_numpy().astype(np.float32);xs=q.select(inputs).to_numpy().astype(np.float32)
        assert np.isfinite(x).all() and np.isfinite(xs).all()
        model=lgb.LGBMClassifier(**params).fit(x,train['action_class'].to_numpy())
        assert np.array_equal(model.classes_,np.arange(4))
        prob=model.predict_proba(xs);mp=OUT/f'reference-{k}.txt';model.booster_.save_model(str(mp))
        replay=lgb.Booster(model_file=str(mp)).predict(xs,num_threads=2)
        assert np.array_equal(prob,replay) and np.isfinite(prob).all() and np.all(prob>=0) and np.allclose(prob.sum(axis=1),1,rtol=0,atol=1e-12)
        pred[ix]=prob
        kp=OUT/f'reference-{k}-training-keys.parquet';train.select(AK+['table_id']).write_parquet(kp)
        receipt={'fold':k,'training_rows':train.height,'scored_rows':q.height,'training_pools':sorted(tp),'scored_pools':sorted(sp),'trained_only_confirmed_non_target':True,'outer_union_and_score_pool_excluded':True,'reload_exact':True,'model_sha256':sha(mp),'training_keys_sha256':sha(kp)}
        write(OUT/f'reference-{k}.json',receipt);receipts.append(receipt)
        print('Reference',k,'training',train.height,'scored',q.height,'seconds',round(time.perf_counter()-start,1),flush=True)
    assert np.isfinite(pred).all()
    result=scored.select(SK).with_columns(*[pl.Series(f'p{i}',pred[:,i]) for i in range(4)])
    pp=OUT/'action-probabilities.parquet';result.write_parquet(pp)
    write(OUT/'reference-receipt.json',{'status':'complete','probabilities_sha256':sha(pp),'reference_freeze_sha256':sha(OUT/'reference-freeze.json'),'rows':result.height,'pair_hands':result.select('pair_id','hand_id').unique().height,'classes':[0,1,2,3],'fits':5,'receipts':receipts,'seconds':time.perf_counter()-start,'outer_evidence_truth_read':False})
    print('Reference probabilities complete',result.shape,round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
