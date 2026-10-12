"""Fixed TabPFN-3 selection pilot; prepare/fit/replay/evaluate are separate stages."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1'}.items():os.environ[k]=v
import sys,json,time,gc,hashlib,argparse
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'experiments/tabpfn3_evidence_20260912'
RUNTIME=HERE/'research/tabpfn3_runtime_20260912'
CHECKPOINT=RUNTIME/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt'
CHECKPOINT_SHA='d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'
PARAMS=dict(n_estimators=8,auto_scale_n_estimators=False,softmax_temperature=.9,
    balance_probabilities=False,average_before_softmax=False,device='cuda:0',
    ignore_pretraining_limits=False,inference_precision='auto',fit_mode='low_memory',
    memory_saving_mode=True,random_state=7329,n_preprocessing_jobs=1,show_progress_bar=False)
CHUNK=256
PRIMARY='tabpfn3_default'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def freeze_check():
    f=read(OUT/'freeze.json')
    for p,h in f['sources'].items():assert sha(p)==h,p
    for m in f['folds']:
        assert sha(OUT/m['context_file'])==m['context_sha256']
    return f

def prepare():
    from audit_common import load_current,PREV,CLF,FAMILIES,check_protected
    import audit_common, lightgbm as lgb,numpy as np,polars as pl
    from sklearn.model_selection import GroupKFold
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'write-once experiment already prepared'
    benchmark=read(RUNTIME/'benchmark.json')
    assert benchmark['passed'],'synthetic hardware gate did not pass'
    assert benchmark['parameters']==PARAMS,'benchmark and pilot settings differ'
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    d,shared,rcols,saved=load_current()
    pred=d['oof'].to_numpy().copy()-1e6
    folds=[];covered=np.zeros(d.height,dtype=int)
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy())
        q=d[ix];cols=shared+(rcols if fam!='coordinated_isolation' else [])
        x=q.select(cols).to_numpy().astype(float);y=q['relevance'].to_numpy().astype(int);groups=q['table_id'].to_numpy()
        assert np.isfinite(x).all()
        for fold,(tr,te) in enumerate(GroupKFold(5).split(x,y,groups)):
            assert not set(groups[tr])&set(groups[te])
            model=lgb.LGBMClassifier(**CLF,random_state=11).fit(x[tr],y[tr])
            pr=model.predict_proba(x[te])[:,1]+10
            np.testing.assert_array_equal(pr,saved['incumbent'].to_numpy()[ix[te]])
            pred[ix[te]]=pr;covered[ix[te]]+=1
            assert len(tr)<=benchmark['shape']['training_rows'] and len(te)<=benchmark['shape']['query_rows'] and len(cols)<=benchmark['shape']['features']
            stem=f'{fam}-{fold}';path=OUT/(stem+'-context.npz')
            np.savez_compressed(path,xtrain=x[tr],ytrain=y[tr],xtest=x[te],ytest=y[te],indices=ix[te],pair_ids=q['pair_id'].to_numpy()[te].astype(str))
            folds.append({'stem':stem,'family':fam,'fold':fold,'columns':cols,'context_file':path.name,
              'context_sha256':sha(path),'training_rows':len(tr),'query_rows':len(te),'width':len(cols),
              'train_pools':sorted(set(groups[tr])),'held_pools':sorted(set(groups[te])),
              'training_class_counts':np.bincount(y[tr],minlength=2).tolist(),'control_exact':True})
    top=(d['oof_rank']<=20).to_numpy()
    assert int(top.sum())==3240 and np.all(covered[top]==1) and np.all(covered[~top]==0)
    np.testing.assert_array_equal(pred,saved['incumbent'].to_numpy())
    frame=d.select('pair_id','hand_id','table_id','behavior_family','relevance','oof','oof_rank').with_columns(pl.Series('incumbent',pred))
    frame.write_parquet(OUT/'evaluation-frame.parquet')
    sources=[Path(__file__),Path(audit_common.__file__),CHECKPOINT,RUNTIME/'benchmark.json',
             RUNTIME/'runtime-receipt.json',HERE/'research/tabpfn3_evidence_protocol_20260912.md',
             PREV/'scores.parquet',PREV/'predictions-frozen.json',OUT/'evaluation-frame.parquet',
             HERE/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv',
             HERE.parent/'Claude Version/outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv']
    write(OUT/'freeze.json',{'primary':PRIMARY,'params':PARAMS,'chunk':CHUNK,'checkpoint_sha256':CHECKPOINT_SHA,
       'sources':{str(p.resolve()):sha(p) for p in sources},'folds':folds,'protected':check_protected(),
       'poker_predictions_made':False,'controls_exact':15,'audit_reused_not_nested':True,
       'replay_stem':'directed_transfer-0','gate':{'map_gain':.01,'ci_lower_gt':0,'family_min':-.01,'half_min':0}})
    print(json.dumps({'prepared':True,'folds':len(folds),'shortlist_rows':int(top.sum())}),flush=True)

def network_off():
    def audit(event,args):
        if event=='socket.connect':raise RuntimeError('Offline study forbids outbound network connections')
    sys.addaudithook(audit)

def infer(context):
    import numpy as np,torch
    import importlib.metadata as md
    for pkg,version in read(RUNTIME/'runtime-receipt.json')['required_versions'].items():assert md.version(pkg)==version,(pkg,version)
    from tabpfn import TabPFNClassifier
    torch.set_num_threads(2);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();model=TabPFNClassifier(model_path=str(CHECKPOINT.resolve()),**PARAMS)
    model.fit(context['xtrain'],context['ytrain'])
    assert model.n_estimators==model.n_estimators_==8
    assert tuple(model.executor_.X_train.shape)==tuple(context['xtrain'].shape),'context changed'
    effective=len(model.ensemble_configs_) if hasattr(model,'ensemble_configs_') else None
    assert effective==8,f'effective estimator count {effective}'
    ix=int(np.flatnonzero(model.classes_==1)[0]);xs=context['xtest']
    parts=[model.predict_proba(xs[i:i+CHUNK])[:,ix] for i in range(0,len(xs),CHUNK)]
    p=np.concatenate(parts);assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    info={'seconds':time.perf_counter()-start,'estimators':effective,
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
          'training_rows':len(context['xtrain']),'query_rows':len(xs),'width':xs.shape[1],
          'class_order':model.classes_.tolist(),'chunks':[min(CHUNK,len(xs)-i) for i in range(0,len(xs),CHUNK)]}
    assert info['peak_reserved_bytes']<=3.6*1024**3,'frozen GPU memory ceiling exceeded'
    del model;gc.collect();torch.cuda.empty_cache()
    return p,info

def fit():
    import numpy as np
    f=freeze_check();network_off()
    for m in f['folds']:
        path=OUT/(m['stem']+'-probabilities.npy');receipt=OUT/(m['stem']+'-receipt.json')
        if path.exists() or receipt.exists():
            assert path.exists() and receipt.exists(),'orphan prediction file'
            r=read(receipt);assert r['freeze_sha256']==sha(OUT/'freeze.json') and r['probabilities_sha256']==sha(path)
            continue
        with np.load(OUT/m['context_file'],allow_pickle=False) as ctx:p,info=infer(ctx)
        np.save(path,p)
        write(receipt,{'freeze_sha256':sha(OUT/'freeze.json'),'context_sha256':m['context_sha256'],
                      'probabilities_sha256':sha(path),'runtime':info})
        print(json.dumps({'fold':m['stem'],**info}),flush=True)
    write(OUT/'fit-complete.json',{'freeze_sha256':sha(OUT/'freeze.json'),
        'receipts':{m['stem']:sha(OUT/(m['stem']+'-receipt.json')) for m in f['folds']}})

def replay():
    import numpy as np
    f=freeze_check();network_off();assert (OUT/'fit-complete.json').exists()
    m=next(x for x in f['folds'] if x['stem']==f['replay_stem'])
    with np.load(OUT/m['context_file'],allow_pickle=False) as ctx:
        p,info=infer(ctx);ids=ctx['pair_ids']
    original=np.load(OUT/(m['stem']+'-probabilities.npy'),allow_pickle=False)
    maxdiff=float(np.max(np.abs(p-original)))
    rank_exact=all(np.array_equal(np.argsort(-p[ids==k],kind='stable'),np.argsort(-original[ids==k],kind='stable')) for k in np.unique(ids))
    assert rank_exact,'fresh-process within-pair ranking drift'
    np.save(OUT/'replay-probabilities.npy',p)
    write(OUT/'replay.json',{'freeze_sha256':sha(OUT/'freeze.json'),'stem':m['stem'],'ranking_exact':rank_exact,
        'probabilities_exact':bool(np.array_equal(p,original)),'maximum_probability_difference':maxdiff,
        'replay_probabilities_sha256':sha(OUT/'replay-probabilities.npy'),'runtime':info})
    print(json.dumps({'replay':True,'ranking_exact':rank_exact,'maxdiff':maxdiff}),flush=True)

def evaluate_only():
    import numpy as np,polars as pl
    from audit_common import evaluate,rankmean,check_protected
    from sklearn.metrics import log_loss
    f=freeze_check();r=read(OUT/'replay.json');assert r['freeze_sha256']==sha(OUT/'freeze.json') and r['ranking_exact']
    done=read(OUT/'fit-complete.json');assert done['freeze_sha256']==sha(OUT/'freeze.json')
    d=pl.read_parquet(OUT/'evaluation-frame.parquet');pred=d['oof'].to_numpy().copy()-1e6;covered=np.zeros(d.height,int);diagnostics=[]
    for m in f['folds']:
        receipt=OUT/(m['stem']+'-receipt.json');assert sha(receipt)==done['receipts'][m['stem']]
        rr=read(receipt);path=OUT/(m['stem']+'-probabilities.npy')
        assert rr['freeze_sha256']==sha(OUT/'freeze.json') and rr['probabilities_sha256']==sha(path)
        p=np.load(path,allow_pickle=False)
        with np.load(OUT/m['context_file'],allow_pickle=False) as ctx:
            ix=ctx['indices'];pred[ix]=p+10;covered[ix]+=1
            diagnostics.append({'fold':m['stem'],'log_loss':float(log_loss(ctx['ytest'],p,labels=[0,1]))})
    top=(d['oof_rank']<=20).to_numpy();assert np.all(covered[top]==1) and np.all(covered[~top]==0)
    inc=d['incumbent'].to_numpy();blend=d['oof'].to_numpy().copy()-1e6
    blend[top]=rankmean(d['pair_id'].to_numpy()[top],inc[top],pred[top])
    preds={'incumbent':inc,PRIMARY:pred,'rank_blend_diagnostic':blend,'stage1_only':d['oof'].to_numpy()}
    sf=OUT/'scores.parquet';assert not sf.exists(),'write-once evaluation'
    d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(k,v) for k,v in preds.items()]).write_parquet(sf)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'fit_complete_sha256':sha(OUT/'fit-complete.json'),
        'replay_sha256':sha(OUT/'replay.json'),'scores_sha256':sha(sf),'all_predictions_before_metric':True})
    res=evaluate(d,preds);res['generic_gate_passed']=res['passed'];res['passed']=[PRIMARY] if res['comparisons'][PRIMARY]['gate_pass'] else []
    res['primary_eligible']=bool(res['passed']);res['negative_control_rejected']=not res['comparisons']['stage1_only']['gate_pass']
    res['diagnostics']=diagnostics;res['protected_unchanged']=check_protected()==f['protected'];assert res['protected_unchanged']
    res['outer_truth_read']=False;res['no_rescue']=True
    write(OUT/'results.json',res);print(json.dumps(res,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','replay','evaluate']);args=p.parse_args()
    {'prepare':prepare,'fit':fit,'replay':replay,'evaluate':evaluate_only}[args.mode]()
