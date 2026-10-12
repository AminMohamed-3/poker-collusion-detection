"""Freeze a bounded MIL experiment and export label-isolated worker packages."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
import shutil
import tarfile
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,thin,TRAIN_VIEWS,HELDOUT_VIEW,LABELS,assert_disjoint
from prepare_reserved_tabicl import data_views
from prepare_attention_reranker import expanded_views
from evaluate_conditional_confirmation import verified_policy
from diverse_next_day import locations
from rank_evidence import context

OUT=ROOT/'reports/mil-pair-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    protected=[ROOT/'outputs_v13_coverage/submission_conditional_coverage.csv',*sorted((ROOT/'outputs_v14_evidence_portfolio').glob('submission_*.csv'))]
    plan={'created_at':datetime.now(timezone.utc).isoformat(),
          'hypothesis':'Learn rare gameplay combinations by pooling full hand sets for pair detection; auxiliary evidence preferences may improve sample efficiency.',
          'sources':{p.name:sha(p) for p in [ROOT/'src/prepare_mil_pairs.py',ROOT/'src/mil_pair_model.py',ROOT/'src/run_mil_pairs.py']},
          'architecture':'285 gameplay hand features; MLP128/64, mean/max/gated attention and event logmeanexp pooling; 897 pair features via MLP64; combined64 -> four classes. No IDs, labels, roles or prior scores as input.',
          'optimization':{'objectives':['bag','joint'],'seeds':[7329,39117,55193],'epochs':60,'batch_size':32,'learning_rate':.001,'weight_decay':.01,'gradient_clip':2.,'average_last_epochs':10},
          'recipes':['bag_equal_blend','joint_equal_blend'],
          'recipe_definition':'Mean three fixed seed class probabilities, risk=1-P(none); equal arithmetic blend with incumbent risk. Incumbent behavior and evidence unchanged. Pure MIL is diagnostic only.',
          'screen_gate':{'mean_pair_behavior_delta_min':.001,'worst_pair_behavior_delta_min':-.003,'mean_sparse_ap_delta_min':-.01},
          'confirmation_gate':{'known_mean_pair_behavior_delta_min':.001,'known_worst_delta_min':-.003,'known_mean_sparse_delta_min':-.01,'omitted_mean_sparse_delta_min':0.,'omitted_worst_sparse_delta_min':-.05,'stress_mean_pair_behavior_delta_min':0.,'stress_worst_delta_min':-.003},
          'progression':'Six fixed fold0 GPU fits. Passing risk recipes only proceed to fold1/2, three omitted families, audit169, coherent65/40. Stop promotion on failed gates; no validation-selected epochs or subgroup rescue.',
          'validation_limit':'Every pool partition has been used previously. Screening and confirmations are repeated development diagnostics, not an untouched estimate.',
          'protected':{str(p.relative_to(ROOT)):sha(p) for p in protected},
          'constraints':['All normal-reference features verified pool-cross-fitted. Outer held pools excluded from all reference fits.','Only explicit development pair labels used. Annotation zeros are noisy preferences only within target pairs.','Held labels and held evidence references stay local; worker receives training labels only.','Shortened pair histories retain original phase style baselines; promotion also requires coherent rebuilt-history checks.','No public-score fitting, external data or pretrained weight download.']}
    write(OUT/'plan.json',plan)
    print(json.dumps({'plan_sha256':sha(OUT/'plan.json')}),flush=True)


def pack(frame, pairs, names, pairnames, labelled):
    keys=pairs.sort('pair_id')['pair_id'].to_list()
    frame=frame.join(pairs.select('pair_id'),on='pair_id',how='semi').sort('pair_id','hand_id')
    grouped=frame.group_by('pair_id',maintain_order=True).len()
    assert grouped['pair_id'].to_list()==keys
    lengths=grouped['len'].to_numpy().astype(np.int64)
    arrays={'x':matrix(frame,names),'offsets':np.r_[0,np.cumsum(lengths)],'pair':matrix(pairs.sort('pair_id'),pairnames)}
    if labelled:
        arrays['y']=np.array([LABELS[f] for f in pairs.sort('pair_id')['behavior_family']],dtype=np.int64)
        arrays['relevance']=frame['relevance'].to_numpy().astype(np.float32)
    meta={'pairs':pairs.sort('pair_id').select('pair_id','table_id','behavior_family').to_dicts(),'hands':frame.select('pair_id','hand_id','relevance').to_dicts()}
    return arrays,meta


def prepare(job):
    plan=json.loads((OUT/'plan.json').read_text())
    assert all(sha(ROOT/'src'/p)==h for p,h in plan['sources'].items())
    dest=OUT/job; dest.mkdir(exist_ok=False); package=dest/'package'; package.mkdir()
    heldpath,policydir,modeldir,_=locations(job)
    keys=pl.read_csv(heldpath); ca,reference=verified_policy(policydir)
    pools=reference['completed'][-1]['scored_pools']
    assert set(keys['table_id']).issubset(pools)
    raw,full,baseviews=data_views(); caf=[c for c in ca.columns if c.startswith('ca_')]
    expanded,views=expanded_views(raw,full,baseviews,ca,caf)
    train=expanded.filter(~pl.col('table_id').is_in(pools))
    if job.startswith('omit_'): train=train.filter(pl.col('behavior_family')!=job.removeprefix('omit_'))
    held=expanded.join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'); assert_disjoint(train,held)
    bundle=joblib.load(modeldir/'conditional-pair.joblib'); pairnames=bundle['features']
    raw=raw.join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1')
    base=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]
    contexts=[c for c in base if c.startswith(('v2_','ca_'))]
    names=base+['context_'+c for c in contexts]
    assert len(names)==285 and len(pairnames)==897
    evidence=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('relevance'))
    raw=raw.join(evidence,on=['pair_id','hand_id'],how='left').with_columns(pl.col('relevance').fill_null(0))
    arrays={}; metadata={}; training=[]
    for tag,key in [('train_full',None),('train_70',TRAIN_VIEWS[0]),('train_40',TRAIN_VIEWS[1]),('full',None),('shortened',HELDOUT_VIEW)]:
        labelled=tag.startswith('train_'); pf=train if labelled else held
        if key is not None: pf=views[key].join(pf.select('pair_id'),on='pair_id',how='semi').sort('pair_id')
        hf=context(raw if key is None else thin(raw,*key),contexts)
        part,meta=pack(hf,pf,names,pairnames,labelled); metadata[tag]=meta
        if labelled: training.append(part)
        else:
            arrays.update({tag+'_'+k:v for k,v in part.items()})
            probability=bundle['model'].predict_proba(matrix(pf,pairnames),num_threads=2)
            meta['control_risk']=(1-probability[:,0]).tolist()
            meta['control_roles']=bundle['model'].classes_[1:][np.argmax(probability[:,1:],axis=1)].tolist()
            old=pl.read_csv(ROOT/'reports/diverse-next-day'/job/'pair-predictions.csv').filter((pl.col('candidate')=='baseline')&(pl.col('history')==tag)).sort('pair_id')
            assert pf['pair_id'].to_list()==old['pair_id'].to_list()
            np.testing.assert_array_equal(1-probability[:,0],old['risk'].to_numpy())
    for k in ['x','pair','y','relevance']: arrays['train_'+k]=np.concatenate([p[k] for p in training])
    lengths=np.concatenate([np.diff(p['offsets']) for p in training]); arrays['train_offsets']=np.r_[0,np.cumsum(lengths)]
    assert arrays['train_offsets'][-1]==len(arrays['train_x'])
    np.savez_compressed(package/'arrays.npz',**arrays)
    write(dest/'metadata.json',metadata); write(dest/'features.json',{'hand':names,'pair':pairnames})
    for name in ['mil_pair_model.py','run_mil_pairs.py']: shutil.copy2(ROOT/'src'/name,package/name)
    write(package/'worker-plan.json',{'parent_plan_sha256':sha(OUT/'plan.json'),'job':job,'optimization':plan['optimization'],'files':{p.name:sha(p) for p in package.iterdir()}})
    inputs=[heldpath,policydir/'summary.json',policydir/'hand-features.parquet',modeldir/'conditional-pair.joblib',ROOT/'data/development_labels.csv',ROOT/'data/development_evidence.csv',*sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet'))]
    write(dest/'audit.json',{'parent_plan_sha256':sha(OUT/'plan.json'),'sources':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'worker_plan_sha256':sha(package/'worker-plan.json'),'metadata_sha256':sha(dest/'metadata.json'),'held_labels_in_package':False,'train_pairs':train.height,'held_pairs':held.height,'train_pool_count':train['table_id'].n_unique(),'held_pool_count':len(pools),'no_player_or_pool_overlap':True,'exact_incumbent_reproduction':True,'shapes':{k:list(v.shape) for k,v in arrays.items()}})
    archive=dest/'package.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in package.iterdir(): t.add(p,arcname=p.name)
    write(dest/'transfer.json',{'sha256':sha(archive),'bytes':archive.stat().st_size})
    print(json.dumps({'job':job,'prepared':True,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); p.add_argument('--job'); args=p.parse_args()
    freeze() if args.freeze else prepare(args.job)
