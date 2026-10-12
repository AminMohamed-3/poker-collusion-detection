"""One independent pretrained-model family; frozen known/nonknown checks."""
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
from robust_pair_experiment import ROOT,sha,matrix,TRAIN_VIEWS,HELDOUT_VIEW,LABELS,assert_disjoint
from prepare_reserved_tabicl import data_views
from prepare_attention_reranker import expanded_views
from evaluate_conditional_confirmation import verified_policy
from diverse_next_day import locations

OUT=ROOT/'reports/tabpfn-v2-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'sources':{p:sha(ROOT/'src'/p) for p in ['prepare_tabpfn_v2.py','run_tabpfn_v2.py']},
        'hypothesis':'A second independently pretrained tabular architecture may improve sparse and omitted-family generalization while preserving near-saturated known-pair accuracy.',
        'checkpoint_repo':'Prior-Labs/TabPFN-v2-clf','checkpoint_revision':'f851f2a3c941544733b712d8c0f96dfae9b28862','checkpoint_file':'tabpfn-v2-classifier-v2_default.ckpt','checkpoint_sha256':'cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49',
        'license':'Prior Labs License1.1, Apache-derived attribution terms; public ungated checkpoint. Retain license and Built with PriorLabs-TabPFN; published model names must begin TabPFN. No external inference API.',
        'variants':[128,256],'n_estimators':4,'random_state':7329,'parameters':{'device':'cuda','fit_mode':'fit_preprocessors','memory_saving_mode':True,'n_jobs':2,'inference_precision':'auto','ignore_pretraining_limits':False},
        'selection':'Feature gain order of the unchanged conditional classifier trained only on each job training pools; exclude omitted family. Context is full/.7/.4 equally represented train histories, no held labels.',
        'recipes':['TabPFN128_equal_blend','TabPFN256_equal_blend'],'risk_recipe':'Equal arithmetic risk blend with incumbent. Incumbent behavior/evidence fixed; pure pretrained risk is diagnostic.',
        'known_screen_gate':{'mean_pair_behavior_delta_min':-.0005,'worst_pair_behavior_delta_min':-.003,'mean_sparse_ap_delta_min':-.01},
        'confirmation_gate':{'known_mean_delta_min':-.0005,'known_worst_delta_min':-.003,'known_mean_sparse_delta_min':-.01,'omitted_mean_sparse_delta_min':.01,'omitted_worst_sparse_delta_min':-.03,'stress_mean_delta_min':0.,'stress_worst_delta_min':-.003},
        'progression':'First2 fold0 fits. Passing recipes only continue to other known folds and all omitted families, followed by coherent history and169-pool audit. Passing known nonregression alone is not improvement or admission.',
        'limits':['All held partitions reused; no untouched holdout claim.','No public-score choice of weights or variants.','No new model license access flow or remote inference service. Telemetry explicitly disabled.','No production inference before all confirmation gates pass.'],
        'protected':json.loads((ROOT/'reports/mil-pair-experiment/plan.json').read_text())['protected']}
    write(OUT/'plan.json',plan); print(json.dumps({'plan_sha256':sha(OUT/'plan.json')}))


def prepare(job):
    plan=json.loads((OUT/'plan.json').read_text()); assert all(sha(ROOT/'src'/p)==h for p,h in plan['sources'].items())
    dest=OUT/job; dest.mkdir(exist_ok=False); package=dest/'package'; package.mkdir()
    heldpath,policy,modeldir,_=locations(job); keys=pl.read_csv(heldpath); ca,summary=verified_policy(policy)
    raw,full,baseviews=data_views(); caf=[c for c in ca.columns if c.startswith('ca_')]; full,views=expanded_views(raw,full,baseviews,ca,caf)
    heldpools=summary['completed'][-1]['scored_pools']; train=full.filter(~pl.col('table_id').is_in(heldpools)); held=full.join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id')
    if job.startswith('omit_'): train=train.filter(pl.col('behavior_family')!=job.removeprefix('omit_'))
    assert_disjoint(train,held)
    augmented=pl.concat([train]+[views[k].join(train.select('pair_id'),on='pair_id',how='semi') for k in TRAIN_VIEWS])
    bundle=joblib.load(modeldir/'conditional-pair.joblib'); features=bundle['features']; gain=bundle['model'].booster_.feature_importance('gain'); order=np.argsort(-gain,kind='stable')[:256]; chosen=[features[i] for i in order]
    arrays={'x_train':matrix(augmented,chosen),'y_train':np.array([LABELS[f] for f in augmented['behavior_family']])}; metadata={}
    for history,frame in [('full',held),('shortened',views[HELDOUT_VIEW].join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'))]:
        probability=bundle['model'].predict_proba(matrix(frame,features),num_threads=2); control=1-probability[:,0]
        old=pl.read_csv(ROOT/'reports/diverse-next-day'/job/'pair-predictions.csv').filter((pl.col('history')==history)&(pl.col('candidate')=='baseline')).sort('pair_id')
        assert frame['pair_id'].to_list()==old['pair_id'].to_list(); np.testing.assert_array_equal(control,old['risk'].to_numpy())
        arrays['x_'+history]=matrix(frame,chosen)
        metadata[history]={'pairs':frame.select('pair_id','table_id','behavior_family').to_dicts(),'control_risk':control.tolist(),'control_roles':bundle['model'].classes_[1:][np.argmax(probability[:,1:],axis=1)].tolist()}
    np.savez_compressed(package/'arrays.npz',**arrays); write(dest/'metadata.json',metadata); write(dest/'features.json',chosen)
    shutil.copy2(ROOT/'src/run_tabpfn_v2.py',package/'run_tabpfn_v2.py')
    worker={k:plan[k] for k in ['checkpoint_file','checkpoint_sha256','variants','n_estimators','random_state','parameters']}; worker.update({'job':job,'parent_plan_sha256':sha(OUT/'plan.json'),'files':{p.name:sha(p) for p in package.iterdir()}}); write(package/'worker-plan.json',worker)
    write(dest/'audit.json',{'plan_sha256':sha(OUT/'plan.json'),'worker_plan_sha256':sha(package/'worker-plan.json'),'metadata_sha256':sha(dest/'metadata.json'),'feature_selector_sha256':sha(modeldir/'conditional-pair.joblib'),'policy_summary_sha256':sha(policy/'summary.json'),'no_player_or_pool_overlap':True,'exact_incumbent_reproduction':True,'held_labels_exported':False,'train_pairs':train.height,'held_pairs':held.height,'features_sha256':sha(dest/'features.json')})
    archive=dest/'package.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for path in package.iterdir(): t.add(path,arcname=path.name)
    write(dest/'transfer.json',{'sha256':sha(archive),'bytes':archive.stat().st_size}); print(json.dumps({'job':job,'sha256':sha(archive),'bytes':archive.stat().st_size}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); p.add_argument('--job'); args=p.parse_args(); freeze() if args.freeze else prepare(args.job)
