"""Fixed CPU pretraining of five pool-excluded observed-behavior encoders."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from real_history_encoder import HistoryEncoder,behavioral_loss
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OUT=HERE/'research/real_history_encoder_20260912'
SEED=94173;BATCH=512

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def encode(model,x,lengths):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(x),BATCH):
            h,_=model(torch.from_numpy(x[start:start+BATCH]),torch.from_numpy(lengths[start:start+BATCH]))
            out.append(h.numpy())
    result=np.concatenate(out);assert result.shape==(len(x),16) and np.isfinite(result).all()
    return result

def pair_features(h,lengths):
    z=np.concatenate([h,lengths[:,:,None]/20],axis=2)
    f=np.concatenate([(z[:,0]+z[:,1])/2,np.abs(z[:,0]-z[:,1])],axis=1)
    flip=np.concatenate([(z[:,1]+z[:,0])/2,np.abs(z[:,1]-z[:,0])],axis=1)
    assert np.array_equal(f,flip) and f.shape==(len(h),34)
    return f

def main():
    begin=time.perf_counter();assert not (OUT/'training-freeze.json').exists(),'write-once model study'
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(SEED)
    receipt=json.loads((OUT/'corpus-receipt.json').read_text(encoding='utf-8'))
    assert receipt['status']=='complete' and receipt['target_pair_hands']==3240
    assert sha(OUT/'corpus-freeze.json')==receipt['corpus_freeze_sha256']
    for name,digest in receipt['artifacts'].items():assert sha(OUT/name)==digest
    cf=json.loads((OUT/'corpus-freeze.json').read_text(encoding='utf-8'))
    for path,digest in cf['source_hashes'].items():assert sha(path)==digest
    corpus=np.load(OUT/'train-corpus.npz',allow_pickle=False);target=np.load(OUT/'target-history.npz',allow_pickle=False)
    x=corpus['x'];n=corpus['lengths'];y=corpus['y'];pools=corpus['pool'];priority=corpus['priority']
    tx=target['x'];tn=target['lengths'];fx=tx.reshape(-1,20,10);fn=tn.reshape(-1)
    assert x.dtype==tx.dtype==y.dtype==np.float32 and n.dtype==tn.dtype==np.int64
    assert x.shape==(len(n),20,10) and y.shape==(len(n),6) and tx.shape==(3240,2,20,10)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(tx).all()
    assert np.all((y[:,4:]==0)|(y[:,4:]==1)) and np.allclose(y[:,:4].sum(1),1,rtol=0,atol=1e-6)
    assert set(pools)<=set(cf['allowed_pools']) and not set(pools)&set(cf['outer_pools'])
    initial=HistoryEncoder();initialpath=OUT/'random-initial-state.pt';torch.save(initial.state_dict(),initialpath)
    selected={};details=[]
    for k in range(5):
        excluded=set(cf['encoder_exclusions'][str(k)]);ix=np.flatnonzero(~np.isin(pools,sorted(excluded)))
        ix=ix[np.argsort(priority[ix],kind='stable')[:50000]]
        assert len(ix)>0 and not set(pools[ix])&excluded and not set(pools[ix])&set(cf['outer_pools'])
        for design in cf['detector_design']:
            if design['fold']==k:assert set(design['held_pools'])<=excluded
        selected[k]=ix;path=OUT/f'encoder-{k}-training-indices.npy';np.save(path,ix)
        details.append({'fold':k,'rows':len(ix),'training_pools':sorted(set(pools[ix])),'excluded_pools':sorted(excluded),'training_indices_sha256':sha(path),'row_indices_reference':'train-keys.parquet','source_metadata_sha256':sha(OUT/'train-keys.parquet')})
    best=ROOT/'Claude Version/outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
    bestsha='6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b';assert sha(best)==bestsha
    sources=[Path(__file__),HERE/'src/real_history_encoder.py',OUT/'test_encoder.py',HERE/'research/real_history_pretraining_protocol_20260912.md',OUT/'corpus-freeze.json',OUT/'corpus-receipt.json',OUT/'train-corpus.npz',OUT/'target-history.npz',OUT/'train-keys.parquet',initialpath,best]
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'sources':{str(p):sha(p) for p in sources},'design':details,'seed':SEED,'batch':BATCH,'epochs':6,'adamw':{'lr':.001,'weight_decay':.0001},'clip_norm':5,'device':'cpu','threads':2,'torch':torch.__version__,'numpy':np.__version__,'trainable_parameters':sum(p.numel() for p in initial.parameters()),'architecture':'GRU10->16,1layer,linear6; pretrained representation is hidden16, not action logits','checkpoint_selection':'fixed final epoch; no scored-pool pretext or evidence metrics','best_sha256':bestsha}
    write(OUT/'training-freeze.json',freeze)
    randomh=encode(initial,fx,fn).reshape(3240,2,16)
    replay=HistoryEncoder();replay.load_state_dict(torch.load(initialpath,weights_only=True,map_location='cpu'))
    assert np.array_equal(randomh,encode(replay,fx,fn).reshape(3240,2,16))
    randomf=pair_features(randomh,tn);np.save(OUT/'random_encoder.npy',randomf)
    receipts=[]
    for k in range(5):
        tick=time.perf_counter();ix=selected[k];xx=x[ix];nn=n[ix];yy=y[ix]
        model=HistoryEncoder();model.load_state_dict(torch.load(initialpath,weights_only=True,map_location='cpu'))
        for name,tensor in model.state_dict().items():assert torch.equal(tensor,initial.state_dict()[name])
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(SEED);losses=[]
        for epoch in range(6):
            model.train();order=rng.permutation(len(ix));total=0.;seen=0
            for offset in range(0,len(order),BATCH):
                j=order[offset:offset+BATCH];opt.zero_grad(set_to_none=True)
                _,logits=model(torch.from_numpy(xx[j]),torch.from_numpy(nn[j]))
                loss=behavioral_loss(logits,torch.from_numpy(yy[j]));assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
                total+=float(loss.detach())*len(j);seen+=len(j)
            assert seen==len(ix);losses.append(total/seen)
            print('Encoder',k,'epoch',epoch+1,'train_behavior_loss',round(losses[-1],6),'seconds',round(time.perf_counter()-tick,1),flush=True)
        mp=OUT/f'encoder-{k}.pt';torch.save(model.state_dict(),mp)
        h=encode(model,fx,fn).reshape(3240,2,16)
        restored=HistoryEncoder();restored.load_state_dict(torch.load(mp,weights_only=True,map_location='cpu'))
        assert np.array_equal(h,encode(restored,fx,fn).reshape(3240,2,16))
        features=pair_features(h,tn);np.save(OUT/f'pretrained-{k}.npy',features)
        rec={'fold':k,'training_rows':len(ix),'epoch_training_losses':losses,'model_sha256':sha(mp),'features_sha256':sha(OUT/f'pretrained-{k}.npy'),'reload_exact':True,'pair_swap_exact':True,'feature_shape':list(features.shape),'seconds':time.perf_counter()-tick,'training_detail':details[k]}
        write(OUT/f'encoder-{k}-receipt.json',rec);receipts.append(rec)
        del model,restored,xx,nn,yy
    for path,digest in freeze['sources'].items():assert sha(path)==digest
    for detail in freeze['design']:
        assert sha(OUT/f"encoder-{detail['fold']}-training-indices.npy")==detail['training_indices_sha256']
    assert sha(best)==bestsha
    result={'status':'complete','models':5,'training_freeze_sha256':sha(OUT/'training-freeze.json'),'random_features_sha256':sha(OUT/'random_encoder.npy'),'random_reload_exact':True,'receipts':receipts,'seconds':time.perf_counter()-begin,'protected_best_unchanged':True,'no_evidence_metrics_read':True}
    write(OUT/'training-receipt.json',result);print('Completed five encoders; seconds',round(result['seconds'],1),flush=True)

if __name__=='__main__':main()
