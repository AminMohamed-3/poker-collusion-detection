"""Extract existing features and create a matched-data trial; never tune on held results."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from collections import defaultdict
from pathlib import Path
import hashlib,json,sys,shutil
import joblib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'history_repair';O=D/'features'
sys.path.insert(0,str(R/'src'))
import features as basic
import advanced_features as advanced
from conditional_action_features import INPUTS,action_class
from conditional_inference_worker import extract_chunk,pair_aggregate
from run_conditional_policy_screen import hand_features
from conditional_bucket_features import bucket_features
from robust_pair_experiment import thin,aggregate,TRAIN_VIEWS,HELDOUT_VIEW,LABELS,MODEL_PARAMS
from tail_pair_experiment import tails

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False)
    audit=json.loads((D/'pilot-audit.json').read_text())
    generation=json.loads((D/'pilot-generation.json').read_text());dirs=[D/'pilot'/f"pool-{p['pool_id']:04d}" for p in generation['pools']]
    for d,r in zip(dirs,generation['pools']):assert all(sha(d/n)==v for n,v in r['files'].items())
    dat=O/'data';dat.mkdir()
    for n in ['hands','seats','actions']:pl.concat([pl.read_parquet(d/(n+'.parquet')) for d in dirs]).write_parquet(dat/(n+'.parquet'))
    labels=pl.concat([pl.read_csv(d/'development_labels.csv') for d in dirs]);labels.write_csv(dat/'development_labels.csv')
    basic.DATA=dat;basic.OUT=O/'base';basic.build('development',batch_size=8)
    advanced.DATA=dat;advanced.BASE=basic.OUT;advanced.OUT=O/'advanced';advanced.build('development')
    raw=pl.read_parquet(sorted((advanced.OUT/'hands_development').glob('batch_*.parquet')))
    basecols=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']];events=[c for c in basecols if c.startswith('v2_')]
    assert len(basecols)==163 and len(events)==56
    references={}
    for name,folder,filename in [('action','conditional-action-experiment','held-models.joblib'),('bucket','conditional-bucket-experiment','held.joblib')]:
        folder=R/'reports'/folder/'policy-screen';receipt=json.loads((folder/'held.json').read_text());modelpath=folder/filename
        assert sha(modelpath)==receipt['model_sha256'];assert set(receipt['training_pools'])==set(json.loads((H/'training-calibration.json').read_text())['training_pools'])
        assert not set(receipt['training_pools'])&set(receipt['scored_pools'])
        references[name]={'bundle':joblib.load(modelpath),'path':modelpath,'receipt':receipt}
    ca=[];cb=[]
    for d in dirs:
        h=pl.read_parquet(d/'hands.parquet').with_columns(pl.col('started_at').cast(pl.String));s=pl.read_parquet(d/'seats.parquet');a=pl.read_parquet(d/'actions.parquet')
        counts=defaultdict(lambda:np.zeros(4,dtype=np.int64))
        for row in a.iter_rows(named=True):counts[row['player_id']][action_class(row)]+=1
        keys=raw.filter(pl.col('table_id')==h['table_id'][0]).select('pair_id','hand_id','table_id').join(labels.select('pair_id','player_1','player_2'),on='pair_id')
        ids=keys.select('hand_id').unique();f=extract_chunk(keys,h.join(ids,on='hand_id',how='semi'),s.join(ids,on='hand_id',how='semi'),a.join(ids,on='hand_id',how='semi'),counts)
        x=f.select(INPUTS).to_numpy().astype(np.float32);ap=references['action']['bundle']['classifier'].predict_proba(x,num_threads=2)
        ca.append(hand_features(f,ap,references['action']['bundle']['regressor'].predict(x,num_threads=2)))
        cb.append(bucket_features(f,ap,references['bucket']['bundle']['model'].predict_proba(x,num_threads=2)))
    ca=pl.concat(ca);cb=pl.concat(cb);caf=[c for c in ca.columns if c.startswith('ca_')];cbf=[c for c in cb.columns if c.startswith('cb_')]
    ca=raw.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0))
    cb=raw.select('pair_id','hand_id','table_id').join(cb.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(cbf).fill_null(0))
    names=json.loads((R/'reports/mil-pair-experiment/fold0/features.json').read_text())['pair'];names1217=joblib.load(R/'reports/conditional-bucket-experiment/screen/pair.joblib')['features']
    assert names1217[:897]==names
    syn=[];synmeta=[]
    for tag,key in [('full',None),('70',TRAIN_VIEWS[0]),('40',TRAIN_VIEWS[1])]:
        rr=raw if key is None else thin(raw,*key);cc=ca if key is None else thin(ca,*key);bb=cb if key is None else thin(cb,*key)
        pair=aggregate(rr,basecols).join(tails(rr,events),on=['pair_id','table_id']).join(pair_aggregate(cc,caf),on=['pair_id','table_id']).join(pair_aggregate(bb,cbf),on=['pair_id','table_id']).join(labels,on='pair_id',validate='1:1').sort('pair_id')
        assert pair.height==labels.height and np.isfinite(pair.select(names1217).to_numpy()).all()
        syn.append(pair.select(names1217).to_numpy().astype(np.float32));synmeta.append(pair.select('pair_id','table_id','behavior_family').to_dicts())
    np.savez_compressed(O/'synthetic-pairs.npz',x=np.concatenate(syn),y=np.tile(np.array([LABELS[r['behavior_family']] for r in synmeta[0]],dtype=np.int64),3))
    (O/'metadata.json').write_text(json.dumps({'views':synmeta,'features':names1217,'source_sha256':sha(__file__),'generator_plan_sha256':sha(D/'pilot-plan.json'),'no_held_labels_or_new_model_fits':True},indent=2))
    print('Built fresh history-repair features; no model fitting or held scoring.',flush=True)

if __name__=='__main__':main()
