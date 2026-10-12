"""Extract existing features and create a matched-data trial; never tune on held results."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from collections import defaultdict
from pathlib import Path
import hashlib,json,sys,shutil
import joblib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'learned';O=H/'augmentation'
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
    audit=json.loads((D/'pilot-audit.json').read_text());assert audit['scale_up_admitted']
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
        assert pair.height==157 and np.isfinite(pair.select(names1217).to_numpy()).all()
        syn.append(pair.select(names1217).to_numpy().astype(np.float32));synmeta.append(pair.select('pair_id','table_id','behavior_family').to_dicts())
    original=dict(np.load(R/'experiments/mhc_local/fold0/train-arrays.npz'))
    meta=json.loads((R/'reports/mil-pair-experiment/fold0/metadata.json').read_text())
    oldcb=pl.read_parquet(R/'reports/conditional-bucket-experiment/policy-screen/hand-features.parquet');extras={}
    for tag,key in [('train_full',None),('train_70',TRAIN_VIEWS[0]),('train_40',TRAIN_VIEWS[1]),('full',None),('shortened',HELDOUT_VIEW)]:
        f=oldcb if key is None else thin(oldcb,*key);agg=pair_aggregate(f,cbf);ids=pl.DataFrame(meta[tag]['pairs']).select('pair_id')
        part=ids.join(agg,on='pair_id',validate='1:1',maintain_order='left');assert part['pair_id'].to_list()==[p['pair_id'] for p in meta[tag]['pairs']]
        extras[tag]=part.select(names1217[897:]).to_numpy().astype(np.float32)
    all_original={tag:np.concatenate([original[tag+'_pair'],np.concatenate([extras['train_full'],extras['train_70'],extras['train_40']]) if tag=='train' else extras[tag]],axis=1) for tag in ['train','full','shortened']}
    # Verify reconstructed original feature order against archived predictions.
    oldmodel=joblib.load(R/'reports/conditional-bucket-experiment/screen/pair.joblib')['model']
    oldpred=pl.read_csv(R/'reports/conditional-bucket-experiment/screen/pair-predictions.csv')
    for tag in ['full','shortened']:
        expected=oldpred.filter((pl.col('history')==tag)&(pl.col('candidate')=='bucket_risk')).sort('pair_id')
        assert expected['pair_id'].to_list()==[p['pair_id'] for p in meta[tag]['pairs']]
        np.testing.assert_array_equal(1-oldmodel.predict_proba(all_original[tag],num_threads=2)[:,0],expected['risk'].to_numpy())
    y=np.array([LABELS[p['behavior_family']] for p in synmeta[0]],dtype=np.int64);ty=original['train_y'];n=len(meta['train_full']['pairs']);assert np.array_equal(ty[:n],ty[n:2*n]) and np.array_equal(ty[:n],ty[2*n:])
    rng=np.random.default_rng(908117);selected=np.array([rng.choice(np.flatnonzero(ty[:n]==v)) for v in y]);matched=np.concatenate([selected,selected+n,selected+2*n]);sy=np.tile(y,3)
    assert np.array_equal(ty[matched],sy)
    synthetic=np.concatenate(syn);assert synthetic.shape==(471,1217)
    for mode,extra in [('resampled_control',all_original['train'][matched]),('synthetic',synthetic)]:
        d=O/mode;d.mkdir();x=np.concatenate([all_original['train'],extra]);yy=np.r_[ty,sy]
        np.savez_compressed(d/'train-arrays.npz',train_pair=x[:,:897],train_y=yy,full_pair=all_original['full'][:,:897],shortened_pair=all_original['shortened'][:,:897],scaler_pair=all_original['train'][:,:897])
        np.savez_compressed(d/'tree-arrays.npz',train_pair=x,train_y=yy,full_pair=all_original['full'],shortened_pair=all_original['shortened'])
    (O/'metadata.json').write_text(json.dumps({'synthetic_views':synmeta,'matched_original_indices':matched.tolist(),'synthetic_pairs':157,'synthetic_views_count':471,'original_pairs':n,'original_views':len(ty),'synthetic_row_fraction':471/(len(ty)+471),'no_held_labels_in_arrays':True,'all_original_bucket_predictions_reproduced_exactly':True,'source_hashes':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),D/'pilot-audit.json',D/'pilot-plan.json',references['action']['path'],references['bucket']['path']]},'limitation':'Pilot policies share one training source. Legacy tilt/tightness/noise fields are not consumed by the learned policy; this pilot has assigned public style and fixed-temperature variation but does not claim active tilt episodes.'},indent=2))
    (O/'features.json').write_text(json.dumps({'pair897':names,'pair1217':names1217},indent=2));print('Prepared matched trial: 4680 training rows per arm, 10.06% synthetic replacement; held labels absent.',flush=True)

if __name__=='__main__':main()
