"""Six fixed GPU fits, with no held labels or validation-selected checkpoints."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from mil_pair_model import PairMIL,objective


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def normalize(x):
    center=np.median(x,axis=0).astype(np.float32)
    low,high=np.quantile(x,[.25,.75],axis=0)
    return center,np.maximum(high-low,.1).astype(np.float32)


def batch(data,indices):
    offsets=data['offsets']; count=offsets[indices+1]-offsets[indices]
    position=torch.arange(int(count.max()),device=indices.device)[None,:]
    mask=position<count[:,None]
    gather=(offsets[indices,None]+position).clamp(max=len(data['x'])-1)
    result={'x':data['x'][gather].masked_fill(~mask[:,:,None],0),'mask':mask,'pair':data['pair'][indices]}
    return result,gather


def main():
    root=Path(__file__).resolve().parent; planpath=root/'worker-plan.json'; plan=json.loads(planpath.read_text())
    assert all(sha(root/n)==h for n,h in plan['files'].items())
    out=root/'results'; out.mkdir(exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(); device=torch.device('cuda')
    arrays=np.load(root/'arrays.npz',allow_pickle=False)
    assert set(arrays.files)=={'train_x','train_pair','train_offsets','train_y','train_relevance','full_x','full_pair','full_offsets','shortened_x','shortened_pair','shortened_offsets'}
    hc,hs=normalize(arrays['train_x']); pc,ps=normalize(arrays['train_pair'])
    scaler=out/'normalization.npz'
    if scaler.exists():
        z=np.load(scaler)
        for k,v in [('hc',hc),('hs',hs),('pc',pc),('ps',ps)]: np.testing.assert_array_equal(z[k],v)
    else: np.savez_compressed(scaler,hc=hc,hs=hs,pc=pc,ps=ps)
    groups={}
    for tag in ['train','full','shortened']:
        groups[tag]={'x':torch.tensor(np.clip((arrays[tag+'_x']-hc)/hs,-8,8),device=device),
                     'pair':torch.tensor(np.clip((arrays[tag+'_pair']-pc)/ps,-8,8),device=device),
                     'offsets':torch.tensor(arrays[tag+'_offsets'],device=device)}
        assert all(torch.isfinite(v).all() for v in groups[tag].values())
    targets=torch.tensor(arrays['train_y'],device=device); relevance=torch.tensor(arrays['train_relevance'],device=device)
    opt=plan['optimization']; receipts=[]
    for name in opt['objectives']:
        for seed in opt['seeds']:
            tag=f'{name}-{seed}'; path=out/(tag+'.json')
            if path.exists():
                saved=json.loads(path.read_text()); assert saved['plan_sha256']==sha(planpath)
                assert all(sha(out/n)==h for n,h in saved['files'].items()); receipts.append(saved); continue
            started=time.perf_counter(); torch.manual_seed(seed); rng=np.random.default_rng(seed)
            model=PairMIL(len(hc),len(pc)).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt['epochs'],eta_min=1e-5)
            average={}; average_count=0; losses=[]
            for epoch in range(opt['epochs']):
                model.train(); order=rng.permutation(len(targets)); epoch_losses=[]
                for start in range(0,len(order),opt['batch_size']):
                    ids=torch.tensor(order[start:start+opt['batch_size']],device=device)
                    inputs,gather=batch(groups['train'],ids)
                    logits,events=model(**inputs)
                    loss=objective(logits,events,inputs['mask'],targets[ids],relevance[gather],name=='joint')
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),opt['gradient_clip']); optimizer.step()
                    epoch_losses.append(float(loss.detach()))
                scheduler.step(); losses.append(float(np.mean(epoch_losses)))
                if epoch>=opt['epochs']-opt['average_last_epochs']:
                    average_count+=1
                    for key,value in model.state_dict().items():
                        if key not in average: average[key]=value.clone()
                        else: average[key]+=(value-average[key])/average_count
                if (epoch+1)%10==0: print(json.dumps({'fit':tag,'epoch':epoch+1,'loss':losses[-1],'seconds':time.perf_counter()-started}),flush=True)
            model.load_state_dict(average); model.eval(); predictions={}
            with torch.no_grad():
                for history in ['full','shortened']:
                    data=groups[history]; probabilities=[]; event_scores=[]
                    for start in range(0,len(data['pair']),16):
                        ids=torch.arange(start,min(start+16,len(data['pair'])),device=device)
                        inputs,_=batch(data,ids); logits,events=model(**inputs)
                        probabilities.append(logits.softmax(1).cpu().numpy()); event_scores.append(events[inputs['mask']].cpu().numpy())
                    predictions[history]=np.concatenate(probabilities); predictions[history+'_events']=np.concatenate(event_scores)
                    assert all(np.isfinite(v).all() for v in predictions.values())
            modelpath=out/(tag+'.pt'); torch.save({'state_dict':model.state_dict(),'hand_features':len(hc),'pair_features':len(pc)},modelpath)
            loaded=PairMIL(len(hc),len(pc)).to(device); loaded.load_state_dict(torch.load(modelpath,map_location=device,weights_only=True)['state_dict']); loaded.eval()
            with torch.no_grad():
                probe,_=batch(groups['full'],torch.arange(4,device=device)); a=model(**probe); b=loaded(**probe)
                assert all(torch.equal(v,w) for v,w in zip(a,b))
            predictionpath=out/(tag+'.npz'); np.savez_compressed(predictionpath,**predictions)
            receipt={'plan_sha256':sha(planpath),'name':name,'seed':seed,'device':'cuda','torch':torch.__version__,'held_labels_used':False,'fixed_epochs':opt['epochs'],'averaged_epochs':average_count,'training_losses':losses,'seconds':time.perf_counter()-started,'reload_exact':True,'files':{p.name:sha(p) for p in [modelpath,predictionpath,scaler]}}
            write(path,receipt); receipts.append(receipt); print(json.dumps({'fit_complete':tag,'seconds':receipt['seconds']}),flush=True)
            del model,loaded,optimizer; torch.cuda.empty_cache()
    write(out/'summary.json',{'plan_sha256':sha(planpath),'fits':len(receipts),'receipts':receipts})
    print(json.dumps({'complete':True,'fits':len(receipts)}),flush=True)


if __name__=='__main__': main()
