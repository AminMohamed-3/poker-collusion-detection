"""Hardware-only continuation of the frozen TabPFN-3 confirmation. No outcomes."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1'}.items():os.environ[k]=v
import sys,json,time,gc,hashlib,argparse,platform,importlib.metadata as md
from pathlib import Path
import numpy as np
CHUNK=256
CP_SHA='d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'
PARAMS=dict(n_estimators=8,auto_scale_n_estimators=False,softmax_temperature=.9,balance_probabilities=False,average_before_softmax=False,device='cuda:0',ignore_pretraining_limits=False,inference_precision='auto',fit_mode='low_memory',memory_saving_mode=True,random_state=7329,n_preprocessing_jobs=1,show_progress_bar=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);assert not p.exists(),f'write-once: {p}'
    p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def network_off():
    def audit(event,args):
        if event=='socket.connect':raise RuntimeError('Inference is offline')
    sys.addaudithook(audit)
def check_assets(root):
    assert sha(root/'transfer-inventory.json')=='0fecf96e729e304605197ee5efe59a19479cc2c1707b205432a756081ea10411'
    for n,m in read(root/'transfer-inventory.json').items():assert sha(root/n)==m['sha256'],n
    origin=read(root/'origin-freeze.json');context=read(root/'context-receipt.json')
    assert origin['params']==PARAMS
    assert origin['context_receipt_sha256']==sha(root/'context-receipt.json')
    assert origin['plan_sha256']==sha(root/'origin-plan.json')==context['plan_sha256']
    assert origin['pilot_source_sha256']==sha(root/'original-pilot.py')
    assert origin['inference_source_sha256']==sha(root/'original-confirm-infer.py')
    assert context['controls_exact']==9 and context['protected_unchanged'] and len(context['contexts'])==9
    assert sha(root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==CP_SHA
    for m in context['contexts']:
        assert sha(root/m['context_file'].replace('\\','/'))==m['context_sha256']
        with np.load(root/m['context_file'].replace('\\','/'),allow_pickle=False) as c:
            assert set(c.files)=={'xtrain','ytrain','xtest','held_indices','held_pair_ids','groups_train','groups_held'}
            assert c['xtrain'].dtype==np.float64 and c['xtest'].dtype==np.float64
            assert c['xtrain'].shape==tuple(m['train_shape'])
            assert c['xtest'].shape==tuple(m['query_shape'])
            assert not set(c['groups_train'])&set(c['groups_held'])
    return context

def runtime(root):
    import torch
    expected=read(root/'runtime-receipt.json')['required_versions']
    actual={k:md.version(k) for k in expected};assert actual==expected,(actual,expected)
    assert torch.cuda.is_available()
    prop=torch.cuda.get_device_properties(0);assert prop.total_memory>=8*1024**3
    return {'required_versions':actual,'python':sys.version,'platform':platform.platform(),'gpu_name':prop.name,'gpu_capacity_bytes':prop.total_memory,'memory_ceiling_bytes':int(.9*prop.total_memory),'cuda_build':torch.version.cuda,'capability':list(torch.cuda.get_device_capability(0))}

def freeze_check(root,out):
    f=read(out/'freeze.json');assert f['source_sha256']==sha(Path(__file__))
    assert f['transfer_inventory_sha256']==sha(root/'transfer-inventory.json')
    assert f['origin_freeze_sha256']==sha(root/'origin-freeze.json') and f['context_receipt_sha256']==sha(root/'context-receipt.json')
    assert f['checkpoint_sha256']==CP_SHA and f['params']==PARAMS and f['chunk']==CHUNK
    ctx=check_assets(root);assert runtime(root)==f['runtime']
    return f,ctx

def prepare(root,out):
    ctx=check_assets(root);env=runtime(root);out.mkdir(parents=True,exist_ok=True)
    write(out/'freeze.json',{'source_sha256':sha(Path(__file__)),'transfer_inventory_sha256':sha(root/'transfer-inventory.json'),'origin_freeze_sha256':sha(root/'origin-freeze.json'),'context_receipt_sha256':sha(root/'context-receipt.json'),'plan_sha256':sha(root/'origin-plan.json'),'checkpoint_sha256':CP_SHA,'params':PARAMS,'chunk':CHUNK,'runtime':env,'replay_stem':'fold0-directed_transfer','primary':'tabpfn3_default','changes':['Windows RTX3050Ti4GB to Linux RTX4060Ti16GB','memory ceiling now90percent of recorded physicalVRAM'],'unchanged':['all9originalcontextfiles','allmodelparameters','seed','querychunk','originalstatisticalprotocol'],'no_outcomes':True})
    print(json.dumps({'prepared':True,'runtime':env}),flush=True)

def infer(root,context,ceiling,diagnostic):
    import torch
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(2);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();info={};model=None
    try:
        model=TabPFNClassifier(model_path=str(root/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt'),**PARAMS)
        model.fit(context['xtrain'],context['ytrain'])
        assert model.n_estimators==model.n_estimators_==len(model.ensemble_configs_)==8
        assert tuple(model.executor_.X_train.shape)==tuple(context['xtrain'].shape)
        ix=int(np.flatnonzero(model.classes_==1)[0]);xs=context['xtest'];parts=[]
        for i in range(0,len(xs),CHUNK):parts.append(model.predict_proba(xs[i:i+CHUNK])[:,ix])
        p=np.concatenate(parts);assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        info.update(estimators=8,training_rows=len(context['xtrain']),query_rows=len(xs),width=xs.shape[1],class_order=model.classes_.tolist(),chunks=[min(CHUNK,len(xs)-i) for i in range(0,len(xs),CHUNK)])
    except BaseException as e:
        info['error']={'type':type(e).__name__,'message':str(e)}
        raise
    finally:
        info.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory_ceiling_bytes=ceiling)
        info['memory_pass']=info['peak_reserved_bytes']<=ceiling
        write(diagnostic,info)
        del model;gc.collect();torch.cuda.empty_cache()
    assert info['memory_pass'],'memory budget exceeded; diagnostic retained'
    return p,info

def benchmark(root,out):
    f,ctx=freeze_check(root,out);rng=np.random.default_rng(7329);n,q,w=ctx['max_context'],ctx['max_query'],ctx['max_width']
    x=rng.normal(size=(n,w));z=rng.normal(size=(q,w));y=np.arange(n,dtype=np.int64)%2;rng.shuffle(y);network_off()
    p,info=infer(root,{'xtrain':x,'ytrain':y,'xtest':z},f['runtime']['memory_ceiling_bytes'],out/'benchmark-diagnostic.json')
    write(out/'benchmark.json',{'passed':True,'synthetic_only':True,'freeze_sha256':sha(out/'freeze.json'),'runtime':info,'probabilities_sha256':hashlib.sha256(p.tobytes()).hexdigest()})
    print(json.dumps({'benchmark_pass':True,**info}),flush=True)

def fit(root,out):
    f,ctx=freeze_check(root,out);b=read(out/'benchmark.json');assert b['passed'] and b['freeze_sha256']==sha(out/'freeze.json');network_off();dest=out/'predictions';dest.mkdir(exist_ok=True)
    for m in ctx['contexts']:
        pp=dest/(m['stem']+'.npy');rr=dest/(m['stem']+'.json');dd=dest/(m['stem']+'-diagnostic.json')
        if pp.exists() or rr.exists():
            assert pp.exists() and rr.exists() and dd.exists(),'orphan result'
            r=read(rr);assert r['freeze_sha256']==sha(out/'freeze.json') and r['probabilities_sha256']==sha(pp) and r['diagnostic_sha256']==sha(dd) and r['context_sha256']==m['context_sha256'];continue
        assert not dd.exists(),'previous failed context diagnostic exists; stop for review'
        with np.load(root/m['context_file'].replace('\\','/'),allow_pickle=False) as c:p,info=infer(root,c,f['runtime']['memory_ceiling_bytes'],dd)
        np.save(pp,p);write(rr,{'freeze_sha256':sha(out/'freeze.json'),'context_sha256':m['context_sha256'],'probabilities_sha256':sha(pp),'diagnostic_sha256':sha(dd),'runtime':info})
        print(json.dumps({'context':m['stem'],**info}),flush=True)
    write(out/'fit-complete.json',{'freeze_sha256':sha(out/'freeze.json'),'receipts':{m['stem']:sha(dest/(m['stem']+'.json')) for m in ctx['contexts']}})

def replay(root,out):
    f,ctx=freeze_check(root,out);done=read(out/'fit-complete.json');assert done['freeze_sha256']==sha(out/'freeze.json');m=next(m for m in ctx['contexts'] if m['stem']==f['replay_stem']);network_off()
    rr=out/'predictions'/(m['stem']+'.json');assert sha(rr)==done['receipts'][m['stem']];originalpath=out/'predictions'/(m['stem']+'.npy');assert sha(originalpath)==read(rr)['probabilities_sha256']
    with np.load(root/m['context_file'].replace('\\','/'),allow_pickle=False) as c:p,info=infer(root,c,f['runtime']['memory_ceiling_bytes'],out/'replay-diagnostic.json');ids=c['held_pair_ids']
    original=np.load(originalpath,allow_pickle=False);exact=all(np.array_equal(np.argsort(-p[ids==k],kind='stable'),np.argsort(-original[ids==k],kind='stable')) for k in np.unique(ids))
    np.save(out/'replay.npy',p);write(out/'replay.json',{'freeze_sha256':sha(out/'freeze.json'),'replay_sha256':sha(out/'replay.npy'),'ranking_exact':exact,'probabilities_exact':bool(np.array_equal(p,original)),'maximum_difference':float(np.max(np.abs(p-original))),'runtime':info})
    assert exact,'fresh-process rank drift'
    print(json.dumps({'replay_ranking_exact':exact,'maximum_difference':float(np.max(np.abs(p-original)))}),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','benchmark','fit','replay']);a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();globals()[args.mode](args.root.resolve(),args.out.resolve())
