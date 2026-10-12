"""Build frozen label-free ordered and timing-removed residual descriptors."""
import os
for name in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='2'
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from contestable_state_features import target_keys
from ordered_surprise_features import FEATURES32,aggregate_hand
HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'research/ordered_surprise_20260912'
ARMS=('ordered','marginal_control')
KEYS=['pair_id','hand_id']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    start=time.perf_counter()
    assert not (OUT/'feature-freeze.json').exists(),'write-once extraction'
    rr=json.loads((OUT/'reference-receipt.json').read_text(encoding='utf-8'))
    rf=json.loads((OUT/'reference-freeze.json').read_text(encoding='utf-8'))
    assert rr['status']=='complete' and rr['fits']==5 and rr['classes']==[0,1,2,3]
    assert sha(OUT/'reference-freeze.json')==rr['reference_freeze_sha256']
    for path,digest in rf['sources'].items():assert sha(path)==digest,'reference source drift'
    for receipt in rr['receipts']:
        k=receipt['fold']
        assert receipt['outer_union_and_score_pool_excluded'] and receipt['reload_exact']
        assert sha(OUT/f'reference-{k}.txt')==receipt['model_sha256']
        assert sha(OUT/f'reference-{k}-training-keys.parquet')==receipt['training_keys_sha256']
    pp=OUT/'action-probabilities.parquet';assert sha(pp)==rr['probabilities_sha256']
    keys=target_keys()
    data=pl.read_parquet(pp).sort(KEYS+['action_no'])
    assert data.height==rr['rows'] and data.select(KEYS+['action_no']).n_unique()==data.height
    assert data.select(KEYS).unique().sort(KEYS).equals(keys)
    p=data.select([f'p{i}' for i in range(4)]).to_numpy()
    cls=data['action_class'].to_numpy()
    assert np.isin(cls,np.arange(4)).all()
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-12)
    residual=np.clip((np.eye(4)[cls.astype(int)]-p)/np.sqrt(p*(1-p)+.05),-4,4)
    assert np.isfinite(residual).all() and np.max(np.abs(residual))<=4
    data=data.with_columns(*[pl.Series(f'r{i}',residual[:,i]) for i in range(4)])
    sources=[Path(__file__),HERE/'src/ordered_surprise_features.py',OUT/'test_features.py',HERE/'research/ordered_surprise_protocol_20260912.md',OUT/'reference-freeze.json',OUT/'reference-receipt.json',pp]
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'features':list(FEATURES32),'arms':list(ARMS),'keys':keys.height,'sources':{str(p):sha(p) for p in sources},'residual':'clip((one_hot(actual_action)-p)/sqrt(p*(1-p)+.05),-4,4)','label_free':'Only frozen predictions, action categories and public action metadata are read; no relevance, behavior family or outer evidence truth.'}
    write(OUT/'feature-freeze.json',freeze)
    groups={key:q for key,q in data.group_by(KEYS)}
    rows={name:[] for name in ARMS};edge_count=0;zero_edges=0;checked=0;omitted_streets=0
    for pid,hid in keys.iter_rows():
        q=groups.get((pid,hid))
        if q is None:
            args=([],[],[],np.empty((0,4)))
        else:
            q=q.sort('action_no')
            args=(q['direction'].to_numpy(),q['street_index'].to_numpy(),q['action_no'].to_numpy(),q.select([f'r{i}' for i in range(4)]).to_numpy())
        direction,street,actions,r=args
        edges=int(np.sum((direction[:-1]!=direction[1:])&(street[:-1]==street[1:]))) if len(direction)>1 else 0
        edge_count+=edges;zero_edges+=int(edges==0)
        omitted_streets+=int(np.any(np.diff(street)>1))
        agg=aggregate_hand(*args)
        if checked<20 and edges>0:
            flip=aggregate_hand(1-direction,street,actions,r)
            reverse=aggregate_hand(direction[::-1],street[::-1],actions[::-1],r[::-1])
            assert agg==flip==reverse,'pair exchange or input row order changed descriptors'
            checked+=1
        for name in ARMS: rows[name].append({'pair_id':pid,'hand_id':hid,**agg[name]})
    tables={name:pl.DataFrame(rows[name]).select(KEYS+list(FEATURES32)).sort(KEYS) for name in ARMS}
    hashes={}
    for name,table in tables.items():
        assert table.select(KEYS).equals(keys)
        a=table.select(FEATURES32).to_numpy();assert np.isfinite(a).all() and (np.abs(a)<=16+1e-12).all() and (a[:,16:]>=0).all()
        fp=OUT/f'{name}.parquet';table.write_parquet(fp);hashes[name]=sha(fp)
    differences=tables['ordered'].select(FEATURES32).to_numpy()-tables['marginal_control'].select(FEATURES32).to_numpy()
    receipt={'status':'complete','keys':keys.height,'fields_per_arm':32,'feature_hashes':hashes,'feature_freeze_sha256':sha(OUT/'feature-freeze.json'),'action_rows':data.height,'edges':edge_count,'zero_edge_hands':zero_edges,'hands_with_omitted_streets':omitted_streets,'pair_flip_and_row_order_fixtures':checked,'hands_with_different_arms':int(np.any(np.abs(differences)>1e-12,axis=1).sum()),'max_abs_arm_difference':float(np.max(np.abs(differences))),'seconds':time.perf_counter()-start,'outer_evidence_truth_read':False}
    write(OUT/'feature-receipt.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
