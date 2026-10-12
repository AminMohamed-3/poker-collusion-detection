"""Six fixed fits; held labels absent, no validation-selected checkpoints."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from attention_reranker_model import ContextReranker,ranking_loss


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=['cpu','cuda'],required=True); args=parser.parse_args()
    root=Path(__file__).resolve().parent
    planpath=root/'worker-plan.json'; plan=json.loads(planpath.read_text())
    assert all(sha(root/name)==digest for name,digest in plan['files'].items())
    out=root/'results'; out.mkdir(exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.use_deterministic_algorithms(True)
    if args.device=='cuda': assert torch.cuda.is_available()
    device=torch.device(args.device)
    arrays=np.load(root/'arrays.npz',allow_pickle=False)
    assert set(arrays.files)=={'train_x','train_y','train_base','train_mask','full_x','full_base','full_mask','shortened_x','shortened_base','shortened_mask'}
    valid=arrays['train_mask']; observed=arrays['train_x'][valid]
    center=np.median(observed,axis=0).astype(np.float32)
    low,high=np.quantile(observed,[.25,.75],axis=0); scale=np.maximum(high-low,.1).astype(np.float32)
    scaler=out/'normalization.npz'
    if scaler.exists():
        stored=np.load(scaler,allow_pickle=False); np.testing.assert_array_equal(center,stored['center']); np.testing.assert_array_equal(scale,stored['scale'])
    else: np.savez_compressed(scaler,center=center,scale=scale)
    tensors={}
    for group in ['train','full','shortened']:
        x=np.clip((arrays[group+'_x']-center)/scale,-8,8).astype(np.float32); x[~arrays[group+'_mask']]=0
        assert np.isfinite(x).all() and arrays[group+'_mask'].any(axis=1).all()
        tensors[group]={'x':torch.as_tensor(x,device=device),'baseline':torch.as_tensor(arrays[group+'_base'],device=device),
            'mask':torch.as_tensor(arrays[group+'_mask'],device=device)}
    targets=torch.as_tensor(arrays['train_y'],device=device)
    usable=np.flatnonzero((arrays['train_y']*valid).sum(axis=1)>0)
    opt=plan['optimization']; receipts=[]
    for objective in opt['objectives']:
        for seed in opt['seeds']:
            tag=f'{objective}-{seed}'; receiptpath=out/(tag+'.json')
            if receiptpath.exists():
                saved=json.loads(receiptpath.read_text()); assert saved['plan_sha256']==sha(planpath) and saved['device']==str(device)
                assert all(sha(out/name)==digest for name,digest in saved['files'].items()); receipts.append(saved); continue
            started=time.perf_counter(); torch.manual_seed(seed); np.random.seed(seed)
            model=ContextReranker(arrays['train_x'].shape[-1],plan['architecture']).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
            rng=np.random.default_rng(seed); history=[]
            for epoch in range(opt['epochs']):
                model.train(); order=rng.permutation(usable); losses=[]
                for start in range(0,len(order),opt['batch_size']):
                    indices=torch.as_tensor(order[start:start+opt['batch_size']],device=device)
                    batch={name:value[indices] for name,value in tensors['train'].items()}
                    score,correction=model(**batch)
                    loss=ranking_loss(score,targets[indices],batch['mask'],objective)
                    loss=loss+opt['correction_penalty']*(correction.square().sum(1)/batch['mask'].sum(1)).mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),opt['gradient_clip']); optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                history.append(float(np.mean(losses)))
                if (epoch+1)%10==0: print(json.dumps({'fit':tag,'epoch':epoch+1,'training_loss':history[-1],'seconds':time.perf_counter()-started}),flush=True)
            model.eval(); predictions={}
            with torch.no_grad():
                for group in ['full','shortened']:
                    parts=[]
                    for start in range(0,len(tensors[group]['x']),16):
                        batch={name:value[start:start+16] for name,value in tensors[group].items()}
                        score,_=model(**batch); parts.append(score.cpu().numpy())
                    predictions[group]=np.concatenate(parts); assert np.isfinite(predictions[group]).all()
            modelpath=out/(tag+'.pt'); torch.save({'state_dict':model.state_dict(),'features':arrays['train_x'].shape[-1],'architecture':plan['architecture']},modelpath)
            loaded=ContextReranker(arrays['train_x'].shape[-1],plan['architecture']).to(device)
            loaded.load_state_dict(torch.load(modelpath,map_location=device,weights_only=True)['state_dict']); loaded.eval()
            with torch.no_grad():
                probe={name:value[:4] for name,value in tensors['full'].items()}
                assert torch.equal(model(**probe)[0],loaded(**probe)[0])
            predictionpath=out/(tag+'.npz'); np.savez_compressed(predictionpath,**predictions)
            receipt={'plan_sha256':sha(planpath),'objective':objective,'seed':seed,'device':str(device),'torch':torch.__version__,
                'parameters':sum(p.numel() for p in model.parameters()),'training_queries':len(usable),'skipped_queries':len(valid)-len(usable),
                'fixed_epochs':opt['epochs'],'training_loss':history,'seconds':time.perf_counter()-started,'reload_exact':True,
                'files':{path.name:sha(path) for path in [modelpath,predictionpath,scaler]}}
            receiptpath.write_text(json.dumps(receipt,indent=2)+'\n'); receipts.append(receipt)
            print(json.dumps({'fit_complete':tag,'seconds':receipt['seconds'],'reload_exact':True}),flush=True)
            del model,loaded,optimizer
            if args.device=='cuda': torch.cuda.empty_cache()
    result={'plan_sha256':sha(planpath),'fits':len(receipts),'device':str(device),'held_labels_used':False,'receipts':receipts}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'complete':True,'fits':len(receipts),'device':str(device)}),flush=True)


if __name__=='__main__': main()
