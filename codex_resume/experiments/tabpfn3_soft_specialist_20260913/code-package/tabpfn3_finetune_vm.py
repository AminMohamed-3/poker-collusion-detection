"""Offline fixed-epoch TabPFN3 adaptation; no query labels or metrics."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items():os.environ[k]=v
import argparse,gc,hashlib,json,subprocess,sys,time,traceback,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import tabpfn3_hardware_resume as baseline
import tabpfn_finetune_hardware_v3 as hardware
SHA=baseline.sha;READ=baseline.read;WRITE=baseline.write
GUARD=13.5*1024**3
EPOCHS=10

def check(root):
    f=READ(root/'study-freeze.json')
    for n,h in f['files_sha256'].items():assert SHA(root/n)==h,n
    assert SHA(Path(__file__))==f['files_sha256'][Path(__file__).name]
    assert SHA(root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==baseline.CP_SHA
    import tabpfn
    installed=Path(tabpfn.__file__).resolve().parent
    for rel,h in f['installed_tabpfn_source_sha256'].items():assert SHA(installed/rel)==h,rel
    env=baseline.runtime(root)
    assert env['gpu_capacity_bytes']>=15*1024**3
    assert baseline.PARAMS==f['inference_params']
    ctx=READ(root/'context-receipt.json')
    assert len(ctx['contexts'])==3
    assert {m['stem'] for m in ctx['contexts']}=={f'fold{i}-soft_play' for i in range(3)}
    assert all(m['family']=='soft_play' for m in ctx['contexts'])
    for m in ctx['contexts']:
        assert SHA(root/m['context_file'])==m['context_sha256']
    return f,ctx,env

def arrays(root,m):
    with np.load(root/m['context_file'],allow_pickle=False) as z:c={k:z[k] for k in z.files}
    assert set(c)=={'xtrain','ytrain','xtest','held_indices','held_pair_ids','groups_train','groups_held'}
    assert c['xtrain'].dtype==c['xtest'].dtype==np.float64
    assert np.isfinite(c['xtrain']).all() and np.isfinite(c['xtest']).all()
    assert set(c['ytrain'])=={0,1}
    assert not set(c['groups_train'])&set(c['groups_held'])
    return c

def state_digest(state):
    import torch
    h=hashlib.sha256()
    for key,t in sorted(state.items()):
        h.update(key.encode());h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def infer(checkpoint,c,fit_path=None,expected_config=None):
    import torch
    from tabpfn import TabPFNClassifier
    model=TabPFNClassifier(model_path=str(checkpoint.resolve()),**baseline.PARAMS)
    if expected_config is not None:assert hardware.json_value(asdict(model.get_inference_config())) == expected_config
    model.fit(c['xtrain'],c['ytrain'])
    if expected_config is not None:assert hardware.json_value(asdict(model.get_inference_config())) == expected_config
    assert model.n_estimators==model.n_estimators_==len(model.ensemble_configs_)==8
    assert tuple(model.executor_.X_train.shape)==tuple(c['xtrain'].shape)
    if fit_path is not None:
        assert not fit_path.exists();model.save_fit_state(fit_path)
    ix=int(np.flatnonzero(model.classes_==1)[0])
    result=np.concatenate([model.predict_proba(c['xtest'][i:i+256])[:,ix] for i in range(0,len(c['xtest']),256)])
    assert np.isfinite(result).all() and np.all((result>=0)&(result<=1))
    del model;gc.collect();torch.cuda.empty_cache()
    return result

def train_one(root,out,c):
    import torch
    from tabpfn.constants import ModelVersion
    from tabpfn import TabPFNClassifier
    from tabpfn.finetuning import FinetunedTabPFNClassifier
    import tabpfn.finetuning.finetuned_base as fb
    from tabpfn.model_loading import save_tabpfn_model
    assert not out.exists();out.mkdir(parents=True)
    started=time.monotonic();receipt={'status':'started','source_sha256':SHA(Path(__file__)),'epochs_required':EPOCHS,'validation_disabled':True,'automatic_selection_disabled':True,'train_shape':list(c['xtrain'].shape),'query_shape':list(c['xtest'].shape)}
    WRITE(out/'started.json',receipt)
    torch.set_num_threads(2);torch.manual_seed(7329);torch.cuda.manual_seed_all(7329);np.random.seed(7329)
    free,_=torch.cuda.mem_get_info(0);assert free>=14*1024**3,'GPU occupied; no concurrency'
    torch.cuda.reset_peak_memory_stats()
    original_factory=fb.get_and_init_optimizer
    original_scaler_factory=fb.GradScaler;original_amp_dtype=torch.get_autocast_dtype('cuda')
    assert torch.cuda.is_bf16_supported(), 'BF16 training unsupported'
    receipt['training_autocast_dtype']='bfloat16';receipt['gradient_scaling_enabled']=False
    trace={};losses=[];model=None
    receipt['gradient_trace']=trace;receipt['losses']=losses
    original_cp=root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt'
    try:
        probe=TabPFNClassifier(model_path=str(original_cp.resolve()),**baseline.PARAMS)
        original_resolved=asdict(probe.get_inference_config())
        receipt['original_resolved_inference_config']=hardware.json_value(original_resolved)
        del probe;gc.collect();torch.cuda.empty_cache()
        torch.manual_seed(7329);torch.cuda.manual_seed_all(7329)
        extra={k:v for k,v in baseline.PARAMS.items() if k not in ('n_estimators','fit_mode','device','random_state','ignore_pretraining_limits')}
        extra['model_path']=str(original_cp.resolve())
        model=FinetunedTabPFNClassifier(model_version=ModelVersion.V3,device='cuda:0',epochs=EPOCHS,random_state=7329,learning_rate=1e-5,weight_decay=.01,grad_clip_value=1.,validation_split_ratio=0.,early_stopping=False,use_lr_scheduler=False,n_estimators_finetune=3,n_estimators_validation=3,n_estimators_final_inference=8,use_activation_checkpointing=True,use_fixed_preprocessing_seed=False,save_checkpoint_interval=None,time_limit=None,n_finetune_ctx_plus_query_samples=50000,finetune_ctx_query_split_ratio=.2,extra_classifier_kwargs=extra)
        original_forward=model._forward_with_loss
        def factory(*args,**kwargs):
            opt=hardware.make_gradient_hook(torch,trace)(original_factory(*args,**kwargs));old=opt.step
            def step(*a,**kw):
                z=old(*a,**kw)
                assert trace['gradient_finite_before_step'] and trace['parameters_finite_after_step'],'nonfinite optimization'
                return z
            opt.step=step;return opt
        def forward(batch):
            loss=original_forward(batch);assert torch.isfinite(loss.detach()).all()
            losses.append(float(loss.detach().cpu()));return loss
        def forbidden_validation(*args,**kwargs):raise AssertionError('Unexpected validation-driven evaluation')
        fb.get_and_init_optimizer=factory;model._forward_with_loss=forward;model._evaluate_model=forbidden_validation
        torch.set_autocast_dtype('cuda',torch.bfloat16)
        fb.GradScaler=lambda *args,**kwargs: torch.amp.GradScaler('cuda',enabled=False)
        try:model.fit(c['xtrain'],c['ytrain'],X_val=None,y_val=None,output_dir=None)
        finally:
            fb.get_and_init_optimizer=original_factory;fb.GradScaler=original_scaler_factory
            torch.set_autocast_dtype('cuda',original_amp_dtype);model._forward_with_loss=original_forward
        assert torch.get_autocast_dtype('cuda')==original_amp_dtype
        assert len(losses)==EPOCHS and trace['optimizer_step_calls']==EPOCHS
        assert trace['changed_parameter_tensors_by_sample']>0
        assert np.array_equal(model.X_,c['xtrain']) and np.array_equal(model.y_,c['ytrain'])
        cp=out/'tabpfn-v3-classifier-finetuned-epoch10.ckpt';assert not cp.exists()
        live_digest=state_digest(model.finetuned_estimator_.models_[0].state_dict())
        receipt['training_resolved_inference_config']=hardware.json_value(asdict(model.finetuned_estimator_.get_inference_config()))
        save_tabpfn_model(model.finetuned_estimator_,cp,additional_fields={'inference_config':original_resolved})
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        assert live_digest==state_digest(saved['state_dict'])
        receipt['saved_state_equals_live']=True;receipt['weights_state_sha256']=live_digest
        receipt['weights_sha256']=SHA(cp);receipt['weights_bytes']=cp.stat().st_size
        del saved,original_forward,forward,model;model=None;gc.collect();torch.cuda.empty_cache()
        p=infer(cp,c,out/'matched_context.tabpfn_fit',receipt['original_resolved_inference_config']);receipt['inference_config_matches_original']=True;np.save(out/'probabilities.npy',p)
        receipt['fit_state_sha256']=SHA(out/'matched_context.tabpfn_fit');receipt['probabilities_sha256']=SHA(out/'probabilities.npy')
        receipt['original_checkpoint_unchanged']=SHA(original_cp)==baseline.CP_SHA
        assert receipt['original_checkpoint_unchanged']
        receipt['peak_reserved_bytes']=torch.cuda.max_memory_reserved();receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        assert receipt['peak_reserved_bytes']<=GUARD
        receipt['status']='passed'
    except BaseException as e:
        receipt['status']='failed';receipt['error']={'type':type(e).__name__,'message':str(e)};receipt['traceback']=traceback.format_exc();raise
    finally:
        fb.get_and_init_optimizer=original_factory;fb.GradScaler=original_scaler_factory
        torch.set_autocast_dtype('cuda',original_amp_dtype)
        receipt['seconds']=time.monotonic()-started
        receipt['final_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        WRITE(out/'receipt.json',receipt)
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()
    return receipt

def replay(root,out,c):
    import torch
    from tabpfn import TabPFNClassifier
    r=READ(out/'receipt.json');assert r['status']=='passed'
    assert SHA(out/'tabpfn-v3-classifier-finetuned-epoch10.ckpt')==r['weights_sha256']
    assert SHA(out/'matched_context.tabpfn_fit')==r['fit_state_sha256']
    assert SHA(out/'probabilities.npy')==r['probabilities_sha256']
    torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    model=TabPFNClassifier.load_from_fit_state(out/'matched_context.tabpfn_fit',device='cuda:0')
    ix=int(np.flatnonzero(model.classes_==1)[0])
    p=np.concatenate([model.predict_proba(c['xtest'][i:i+256])[:,ix] for i in range(0,len(c['xtest']),256)])
    assert hardware.json_value(asdict(model.get_inference_config())) == r['original_resolved_inference_config']
    del model;gc.collect();torch.cuda.empty_cache()
    q=infer(out/'tabpfn-v3-classifier-finetuned-epoch10.ckpt',c,expected_config=r['original_resolved_inference_config'])
    assert np.isfinite(p).all() and np.isfinite(q).all()
    original=np.load(out/'probabilities.npy',allow_pickle=False);ids=c['held_pair_ids']
    def exact_rank(x):return all(np.array_equal(np.argsort(-x[ids==k],kind='stable'),np.argsort(-original[ids==k],kind='stable')) for k in np.unique(ids))
    receipt={'saved_fit_rank_exact':exact_rank(p),'refitted_checkpoint_rank_exact':exact_rank(q),'saved_fit_probabilities_exact':bool(np.array_equal(p,original)),'refitted_checkpoint_probabilities_exact':bool(np.array_equal(q,original)),'max_diff_saved':float(np.max(np.abs(p-original))),'max_diff_refit':float(np.max(np.abs(q-original))),'weights_sha256':r['weights_sha256'],'fit_state_sha256':r['fit_state_sha256'],'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    np.save(out/'replay_saved.npy',p);np.save(out/'replay_refit.npy',q)
    WRITE(out/'replay.json',receipt)
    assert receipt['saved_fit_rank_exact'] and receipt['refitted_checkpoint_rank_exact'] and receipt['saved_fit_probabilities_exact'] and receipt['refitted_checkpoint_probabilities_exact'] and receipt['peak_reserved_bytes']<=GUARD
    return receipt

def synthetic(root,run,ctx):
    rng=np.random.default_rng(7329);n=max(m['train_shape'][0] for m in ctx['contexts']);q=max(m['query_shape'][0] for m in ctx['contexts']);w=453
    assert n==1980 and q==920 and w==453
    c={'xtrain':rng.normal(size=(n,w)),'ytrain':np.arange(n,dtype=np.int64)%2,'xtest':rng.normal(size=(q,w)),'held_pair_ids':np.array(['synthetic']*q)}
    rng.shuffle(c['ytrain'])
    cpath=run/'synthetic-context.npz';assert not cpath.exists();np.savez_compressed(cpath,**c)
    result=train_one(root,run/'synthetic',c)
    print(json.dumps({'synthetic':result['status'],'seconds':result['seconds'],'peak_reserved_bytes':result['peak_reserved_bytes']}),flush=True)

def child(root,run,mode,stem=None):
    cmd=[sys.executable,str(Path(__file__)),mode,'--root',str(root),'--out',str(run)]
    if stem is not None:cmd+=['--stem',stem]
    r=subprocess.run(cmd);assert r.returncode==0,(mode,stem,r.returncode)

def main(v):
    root=v.root.resolve();run=v.out.resolve();f,ctx,env=check(root);baseline.network_off()
    original=root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt'
    m=next((m for m in ctx['contexts'] if m['stem']==v.stem),None)
    if v.mode=='all':
        assert not run.exists();run.mkdir(parents=True)
        assert shutil.disk_usage(root).free>=2*1024**3,'Need room for3checkpoints plus rehearsal and reserve'
        WRITE(run/'run-freeze.json',{'study_freeze_sha256':SHA(root/'study-freeze.json'),'runtime':env,'epoch_count':EPOCHS,'started':time.time()})
        child(root,run,'synthetic');child(root,run,'synthetic-replay');child(root,run,'baseline')
        for m in ctx['contexts']:
            child(root,run,'context',m['stem'])
        child(root,run,'replay',f['replay_stem'])
        WRITE(run/'complete.json',{'study_freeze_sha256':SHA(root/'study-freeze.json'),'context_receipts':{m['stem']:SHA(run/'contexts'/m['stem']/'receipt.json') for m in ctx['contexts']},'baseline_sha256':SHA(run/'baseline.json'),'synthetic_replay_sha256':SHA(run/'synthetic/replay.json'),'real_replay_sha256':SHA(run/'contexts'/f['replay_stem']/'replay.json')})
    elif v.mode=='synthetic':synthetic(root,run,ctx)
    elif v.mode=='synthetic-replay':
        with np.load(run/'synthetic-context.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
        print(json.dumps(replay(root,run/'synthetic',c)),flush=True)
    elif v.mode=='baseline':
        import torch
        checks={}
        for m in ctx['contexts']:
            c=arrays(root,m);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
            p=infer(original,c);expected=np.load(root/('baseline_expected-'+m['stem']+'.npy'),allow_pickle=False)
            ids=c['held_pair_ids'];rank=all(np.array_equal(np.argsort(-p[ids==k],kind='stable'),np.argsort(-expected[ids==k],kind='stable')) for k in np.unique(ids))
            np.save(run/('baseline-'+m['stem']+'.npy'),p)
            checks[m['stem']]={'probabilities_exact':bool(np.array_equal(p,expected)),'ranking_exact':rank,'max_diff':float(np.max(np.abs(p-expected))),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            WRITE(run/('baseline-'+m['stem']+'.json'),checks[m['stem']])
            assert checks[m['stem']]['probabilities_exact'] and rank and checks[m['stem']]['peak_reserved_bytes']<=GUARD
        WRITE(run/'baseline.json',{'probabilities_exact':True,'ranking_exact':True,'contexts':checks})
        print(json.dumps({'baseline':checks}),flush=True)
    elif v.mode=='context':
        assert m is not None
        assert READ(run/'baseline.json')['probabilities_exact']
        rr=train_one(root,run/'contexts'/m['stem'],arrays(root,m));print(json.dumps({'context':m['stem'],'status':rr['status'],'seconds':rr['seconds'],'peak_reserved_bytes':rr['peak_reserved_bytes']}),flush=True)
    elif v.mode=='replay':
        assert m is not None;print(json.dumps(replay(root,run/'contexts'/m['stem'],arrays(root,m))),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['all','synthetic','synthetic-replay','baseline','context','replay']);a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--stem');main(a.parse_args())
