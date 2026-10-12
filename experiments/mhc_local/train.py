"""Local-only fixed training; no held labels or held-selected checkpoints."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import Config,TinyCED

HERE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--job',default='fold0'); parser.add_argument('--variant',required=True); parser.add_argument('--seed',type=int,required=True); parser.add_argument('--device',choices=['cpu','cuda'],default='cuda'); args=parser.parse_args()
    plan=json.loads((HERE/'plan.json').read_text()); assert args.variant in plan['variants'] and args.seed in plan['seeds']
    assert all(sha(HERE/p)==h for p,h in plan['source_hashes'].items()); root=HERE/args.job; audit=json.loads((root/'data-audit.json').read_text())
    assert audit['plan_sha256']==sha(HERE/'plan.json') and sha(root/'train-arrays.npz')==audit['train_arrays_sha256']
    dest=root/(args.variant+'-'+str(args.seed)); dest.mkdir(exist_ok=False)
    opt=plan['training']; torch.set_num_threads(opt['threads']); torch.set_num_interop_threads(opt['threads']); torch.use_deterministic_algorithms(True)
    if args.device=='cuda': assert torch.cuda.is_available(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed); device=torch.device(args.device); rng=np.random.default_rng(args.seed)
    arrays=np.load(root/'train-arrays.npz',allow_pickle=False); assert set(arrays.files)=={'train_pair','train_y','full_pair','shortened_pair'}
    train=arrays['train_pair']; center=np.median(train,axis=0).astype(np.float32); q=np.quantile(train,[.25,.75],axis=0); scale=np.maximum(q[1]-q[0],.1).astype(np.float32)
    groups={n:torch.tensor(np.clip((arrays[n+'_pair']-center)/scale,-8,8),device=device) for n in ['train','full','shortened']}; targets=torch.tensor(arrays['train_y'],device=device)
    model=TinyCED(Config(**plan['variants'][args.variant])).to(device); assert model.parameter_count()==plan['parameters'][args.variant]
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
    steps=math.ceil(len(targets)/opt['batch_size']); total=steps*opt['epochs']; elapsed=0; average={}; averaged=0; history=[]; start=time.perf_counter()
    for epoch in range(opt['epochs']):
        model.train(); permutation=rng.permutation(len(targets)); losses=[]; gradients=[]
        for offset in range(0,len(targets),opt['batch_size']):
            frac=elapsed/max(total-1,1); warm=(elapsed+1)/steps
            lr=opt['learning_rate']*warm if elapsed<steps else 1e-5+.5*(opt['learning_rate']-1e-5)*(1+math.cos(math.pi*(elapsed-steps)/max(total-steps-1,1)))
            for group in optimizer.param_groups: group['lr']=lr
            ids=torch.tensor(permutation[offset:offset+opt['batch_size']],device=device)
            optimizer.zero_grad(set_to_none=True); logits=model(groups['train'][ids]); loss=F.cross_entropy(logits,targets[ids]); assert torch.isfinite(loss)
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),opt['gradient_clip']); assert torch.isfinite(grad)
            optimizer.step(); model.update_routing(); losses.append(float(loss.detach())); gradients.append(float(grad)); elapsed+=1
        if epoch>=opt['epochs']-opt['average_last_epochs']:
            averaged+=1
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name not in average: average[name]=p.detach().clone()
                    else: average[name].add_((p.detach()-average[name])/averaged)
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'max_gradient_norm_before_clip':max(gradients),'seconds':time.perf_counter()-start}; history.append(row)
        if (epoch+1)%5==0 or epoch==0: print(json.dumps({'variant':args.variant,'seed':args.seed,**row}),flush=True)
    with torch.no_grad():
        for name,p in model.named_parameters(): p.copy_(average[name])
    model.eval(); probabilities={}
    with torch.no_grad():
        for tag in ['full','shortened']:
            probabilities[tag]=torch.cat([model(groups[tag][i:i+64]).softmax(-1).cpu() for i in range(0,len(groups[tag]),64)]).numpy()
            assert np.isfinite(probabilities[tag]).all(); np.testing.assert_allclose(probabilities[tag].sum(1),1,atol=1e-6)
    torch.save({'config':plan['variants'][args.variant],'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'center':torch.tensor(center),'scale':torch.tensor(scale)},dest/'model.pt')
    saved=torch.load(dest/'model.pt',map_location=device,weights_only=True); clone=TinyCED(Config(**saved['config'])).to(device); clone.load_state_dict(saved['state_dict']); clone.eval()
    with torch.no_grad(): assert torch.equal(model(groups['full'][:16]),clone(groups['full'][:16]))
    np.savez_compressed(dest/'predictions.npz',**probabilities)
    receipt={'completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(HERE/'plan.json'),'data_audit_sha256':sha(root/'data-audit.json'),'variant':args.variant,'seed':args.seed,'parameters':model.parameter_count(),'device':str(device),'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,'torch':torch.__version__,'numpy':np.__version__,'fixed_epochs':opt['epochs'],'steps':elapsed,'averaged_epochs':averaged,'history':history,'training_seconds':time.perf_counter()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,'reload_exact':True,'held_labels_used':False,'constraint_diagnostics':model.constraint_diagnostic(groups['full'][:16]),'model_sha256':sha(dest/'model.pt'),'predictions_sha256':sha(dest/'predictions.npz')}
    write(dest/'receipt.json',receipt); print(json.dumps({k:v for k,v in receipt.items() if k!='history'}),flush=True)

if __name__=='__main__': main()
