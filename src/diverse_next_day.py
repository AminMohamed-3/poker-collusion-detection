"""Fixed portfolio of gameplay-only risk/evidence models, with explicit admission gates."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,thin,aggregate,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS,assert_disjoint,stable_ap
from tail_pair_experiment import tails
from prepare_reserved_tabicl import data_views
from prepare_attention_reranker import expanded_views
from evidence_capacity_experiment import load_sequence,fit_variant,compare
from rank_evidence import context,family_columns,PARAMS
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from relative_evidence_experiment import records,summarize
from evaluate_conditional_action_screen import extra_aggregate
from evaluate_conditional_behavior import family_metrics
from evaluate_conditional_confirmation import verified_policy

OUT=ROOT/'reports/diverse-next-day'
DENSE=[(f,s) for f in [.8,.6,.4,.2] for s in [48113,90271]]
SEEDS=[7329,39117,55193]

def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')

def locations(job):
    omitted=job.startswith('omit_')
    if job=='audit':
        d=ROOT/'reports/conditional-reserved-audit'
        return d/'held-pairs.csv',d/'remote-returned/policy-screen',d/'results',d/'gpu-returned/predictions.npz'
    reference=ROOT/('reports/tabicl-screen' if job=='fold0' else 'reports/tabicl-confirmation/'+job)
    policy=ROOT/('reports/conditional-action-experiment/policy-screen' if job=='fold0' or omitted else 'reports/conditional-action-confirmation/conditional-'+job+'-v1/policy-screen')
    modeldir=ROOT/('reports/conditional-action-experiment/screen-results' if job=='fold0' else 'reports/conditional-action-confirmation/results/'+job)
    return reference/'held-pairs.csv',policy,modeldir,reference/'predictions.npz'

def prepare_plan():
    OUT.mkdir(exist_ok=False)
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'purpose':'Five prespecified diverse next-day candidates; portfolio admission is not a claim of improvement. Keep v13 as incumbent. No public score fitting, subgroup rescue, or inference from public ties.',
        'candidates':{'hybrid':'Equal conditional tree and existing pretrained TabICL risk; v13 behavior/evidence.',
            'seed_risk':'Mean conditional class-0 complement across seeds 7329,39117,55193; v13 behavior/evidence.',
            'history_risk':'Unchanged conditional tree trained on nine equally weighted full/dense-thinned views; v13 behavior/evidence.',
            'seed_evidence':'Mean conditional evidence scores across three seeds; v13 risk/behavior.',
            'mixed':'Equal risk average of hybrid, seed_risk, history_risk; seed_evidence ranks; v13 behavior.'},
        'research_only':'Evidence history augmentation, full/.7/.4, equal total weight per original pair. Not used by these five candidates.',
        'pair_parameters':MODEL_PARAMS,'evidence_parameters':PARAMS,'seeds':SEEDS,'dense_train_views':DENSE,
        'jobs':['fold0','fold1','fold2','omit_directed_transfer','omit_soft_play','omit_coordinated_isolation','audit'],
        'validation':'Same frozen pool-separated references; full/65% histories, all omitted families, less-reused 169-pool audit, and coherent 65%/40% histories. All partitions have been used before; no untouched estimate.',
        'gates':{'risk':'Known mean .7*PairAP+.1*BehaviorMAP delta >=-.002 and every delta>=-.006 vs v13; mean hypothetical-1%-AP delta>=-.02. Omitted mean sparse delta>=-.01 and every delta>=-.05. Must beat v10 ordinary PairAP in mean across full/short known folds.',
            'evidence':'Known pooled MAP delta>=-.005, each family>=-.025 vs v13. Omitted mean delta>=-.005 and every delta>=-.03. Coherent/audit mean delta>=-.005 and every delta>=-.02.',
            'incremental_improvement':'Separate label only: mean known composite gain>=.001, no known composite loss beyond .003, and confirmations nonnegative in mean. A portfolio candidate can pass admission without passing improvement.',
            'failed_candidates':'Retain complete diagnostics; do not silently relax thresholds or put failed candidates in the eligible upload queue.'},
        'protected':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'outputs_v13_coverage/submission_conditional_coverage.csv',ROOT/'outputs_v11_conditional/pair-model.joblib',ROOT/'outputs_v11_conditional/evidence-model.joblib']},
        'limits':['IDs only joins/groups/splits/ties; no simulator internals.','No evaluation labels. No public-score-trained weights or rescaling.','Unknown pairs are not normal training labels; unannotated hands are noisy ranking preferences.','Dense pair thinning retains original phase player baselines; coherent checks rebuild them.','Candidate selection never uses public score correlation.']}
    write(OUT/'plan.json',plan); print(json.dumps({'plan_sha256':sha(OUT/'plan.json')}),flush=True)

def augmented_ranker(data,features,contexts):
    pieces=[]
    for i,key in enumerate([None]+TRAIN_VIEWS):
        part=data if key is None else context(thin(data,*key),contexts)
        valid=part.group_by('pair_id').agg(pl.col('relevance').sum()).filter(pl.col('relevance')>0).select('pair_id')
        part=part.join(valid,on='pair_id',how='semi').with_columns(pl.lit(i).alias('view'))
        pieces.append(part)
    frame=family_columns(pl.concat(pieces).sort('pair_id','view','hand_id'),'behavior_family')
    counts=frame.group_by('pair_id').agg(pl.col('view').n_unique().alias('nviews'))
    frame=frame.join(counts,on='pair_id',validate='m:1',maintain_order='left')
    groups=frame.group_by('pair_id','view',maintain_order=True).len()['len'].to_numpy()
    model=lgb.LGBMRanker(**PARAMS).fit(matrix(frame,features),frame['relevance'].to_numpy(),group=groups,sample_weight=1/frame['nviews'].to_numpy())
    return model

def run(jobs):
    plan=json.loads((OUT/'plan.json').read_text()); assert plan['source_sha256']==sha(__file__)
    raw,full,baseviews=data_views(); labels=pl.read_csv(ROOT/'data/development_labels.csv')
    names=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    for key in DENSE:
        sample=thin(raw,*key)
        baseviews[key]=aggregate(sample,names).join(tails(sample,events),on=['pair_id','table_id']).join(labels,on='pair_id').select(full.columns).sort('pair_id')
    evidence,_,truths,_,_=load_sequence()
    evidence=context(evidence.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    for job in jobs:
        dest=OUT/job; dest.mkdir(exist_ok=True)
        if (dest/'metrics.json').exists(): print(json.dumps({'already_completed':job}),flush=True); continue
        heldpath,policy,modeldir,tabpath=locations(job)
        keys=pl.read_csv(heldpath); pools=sorted(keys['table_id'].unique()); omitted=job.removeprefix('omit_') if job.startswith('omit_') else None
        ca,_=verified_policy(policy); caf=[c for c in ca.columns if c.startswith('ca_')]
        expanded,views=expanded_views(raw,full,baseviews,ca,caf)
        train=expanded.filter(~pl.col('table_id').is_in(pools)); held=expanded.join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id')
        if omitted: train=train.filter(pl.col('behavior_family')!=omitted)
        assert_disjoint(train,held)
        trainkeys=train.select('pair_id'); original=pl.concat([train]+[views[k].join(trainkeys,on='pair_id',how='semi') for k in TRAIN_VIEWS])
        dense=pl.concat([train]+[views[k].join(trainkeys,on='pair_id',how='semi') for k in DENSE])
        baseline=joblib.load(modeldir/'conditional-pair.joblib'); features=baseline['features']
        bag=[baseline['model']]
        for seed in SEEDS[1:]:
            model=lgb.LGBMClassifier(**{**MODEL_PARAMS,'random_state':seed}).fit(matrix(original,features),np.array([LABELS[f] for f in original['behavior_family']]),sample_weight=np.full(original.height,1/3))
            joblib.dump({'model':model,'features':features},dest/f'pair-seed-{seed}.joblib'); bag.append(model)
        model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(dense,features),np.array([LABELS[f] for f in dense['behavior_family']]),sample_weight=np.full(dense.height,1/9))
        joblib.dump({'model':model,'features':features},dest/'pair-history.joblib')
        tab=np.load(tabpath,allow_pickle=False); pairrows=[]; predrows=[]; erows=[]
        eh=context(evidence.join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1'),caf)
        etrain=eh.join(trainkeys,on='pair_id',how='semi'); ranker=joblib.load(modeldir/'conditional-evidence.joblib'); ef=ranker['features']; rbag=[ranker['model']]
        for seed in SEEDS[1:]:
            r=fit_variant(etrain,ef,{**PARAMS,'random_state':seed}); rbag.append(r); joblib.dump({'model':r,'features':ef},dest/f'evidence-seed-{seed}.joblib')
        contexts=events+SEQ+EQ+caf
        history_rank=augmented_ranker(etrain,ef,contexts); joblib.dump({'model':history_rank,'features':ef},dest/'evidence-history.joblib')
        for history,frame in [('full',held),('shortened',views[HELDOUT_VIEW].join(keys.select('pair_id'),on='pair_id',how='semi').sort('pair_id'))]:
            assert frame['pair_id'].to_list()==keys['pair_id'].to_list()
            x=matrix(frame,features); prob=baseline['model'].predict_proba(x); risk=1-prob[:,0]
            roles=baseline['model'].classes_[1:][np.argmax(prob[:,1:],axis=1)]; y=np.array([LABELS[f] for f in frame['behavior_family']])
            candidates={'baseline':risk,'hybrid':.5*(risk+tab[history]),'seed_risk':np.mean([1-m.predict_proba(x)[:,0] for m in bag],axis=0),'history_risk':1-model.predict_proba(x)[:,0]}
            candidates['mixed']=np.mean([candidates[k] for k in ['hybrid','seed_risk','history_risk']],axis=0)
            old=pl.read_csv(modeldir/'pair-predictions.csv').filter((pl.col('candidate')=='conditional')&(pl.col('history')==history)).sort('pair_id') if job!='audit' else None
            if old is not None: np.testing.assert_array_equal(risk,old['risk'].to_numpy())
            weights=np.where(y>0,.01/np.mean(y>0),.99/np.mean(y==0))
            for candidate,values in candidates.items():
                bm=family_metrics(y,values,roles)['behavior_map']; ap=stable_ap(y>0,values)
                pairrows.append({'job':job,'history':history,'candidate':candidate,'pair_ap':ap,'sparse_ap':stable_ap(y>0,values,weights),'behavior_map':bm,'pair_behavior':.7*ap+.1*bm})
                predrows.extend({'pair_id':p,'history':history,'candidate':candidate,'risk':float(v),'class':int(c)} for p,v,c in zip(frame['pair_id'],values,roles))
            assigned=frame.select('pair_id').with_columns(pl.Series('predicted_behavior',[{v:k for k,v in LABELS.items()}[int(i)] for i in roles]))
            eheld=eh.join(keys.select('pair_id'),on='pair_id',how='semi')
            if history!='full': eheld=context(thin(eheld,*HELDOUT_VIEW),contexts)
            truth={k[0]:set(g.filter(pl.col('relevance')>0)['hand_id']) for k,g in eheld.group_by('pair_id') if g['relevance'].sum()>0}
            eheld=family_columns(eheld.filter(pl.col('pair_id').is_in(list(truth))).join(assigned,on='pair_id',validate='m:1'),'predicted_behavior').sort('pair_id','hand_id')
            z=matrix(eheld,ef); escores={'baseline':ranker['model'].predict(z,num_threads=2),'seed_evidence':np.mean([r.predict(z,num_threads=2) for r in rbag],axis=0),'history_evidence':history_rank.predict(z,num_threads=2)}
            for candidate,values in escores.items():
                erows.extend(records(eheld,values,truth,history,candidate))
                eheld.select('pair_id','hand_id').with_columns(pl.Series('score',values)).write_parquet(dest/(history+'-'+candidate+'-evidence.parquet'))
        pl.DataFrame(predrows).write_csv(dest/'pair-predictions.csv'); pl.DataFrame(erows).write_csv(dest/'evidence-pairs.csv')
        comparisons={h:{c:compare([r for r in erows if r['experiment']==h],c,baseline='baseline') for c in ['seed_evidence','history_evidence']} for h in ['full','shortened']}
        payload={'job':job,'plan_sha256':sha(OUT/'plan.json'),'pair_results':pairrows,'evidence_comparisons':comparisons,'models':{p.name:sha(p) for p in dest.glob('*.joblib')}}
        write(dest/'metrics.json',payload); print(json.dumps(payload),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); parser.add_argument('--jobs',nargs='+'); args=parser.parse_args()
    if args.freeze: prepare_plan()
    else: run(args.jobs)
