"""Frozen alternative learner/objective screen on the 162-pair training-only audit.

Current chronology383 shortlist and shipping hybrid feature routing remain fixed.
No outer evidence truth, evaluation examples or public outcomes select parameters.
"""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import time
from datetime import datetime,timezone
import warnings
warnings.filterwarnings('ignore',message='X does not have valid feature names')
HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parent; CV=ROOT/'Claude Version'
sys.path.insert(0,str(HERE/'vendor/catboost')); sys.path.insert(0,str(CV/'src'))
import numpy as np
import polars as pl
import lightgbm as lgb
import catboost
from catboost import CatBoostClassifier,CatBoostRanker,Pool
from sklearn.model_selection import GroupKFold
from evidence_aug.paths import check_protected,sha
from evidence_aug.fold import load_fold,fold_pools
from evidence_aug.frame import FAMILIES
from evidence_aug.step14_audit_truncation import OUT as AUDIT,CLF
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step5_evaluate import half

OUT=HERE/'experiments/catboost_evidence_20260912'
PREV=HERE/'experiments/symmetry_censor_20260912'
PARAMS=dict(iterations=300,depth=5,learning_rate=.05,l2_leaf_reg=5,
            random_seed=7329,thread_count=2,verbose=False,allow_writing_files=False,
            task_type='CPU')
ARMS=['incumbent','cat_logloss','cat_map','blend_logloss','blend_map']

def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def rankmean(pid,a,b):
    f=pl.DataFrame({'pair_id':pid,'a':a,'b':b})
    return f.select((-(pl.col('a').rank('average',descending=True).over('pair_id')+pl.col('b').rank('average',descending=True).over('pair_id'))/2).alias('s'))['s'].to_numpy()

def main():
    t=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Study already frozen'
    protected=check_protected()
    srcs=[Path(__file__),PREV/'scores.parquet',PREV/'predictions-frozen.json',AUDIT/'audit-oof.parquet',AUDIT/'audit-structural-features.parquet']
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'version':catboost.__version__,
      'arms':ARMS,'primary':'cat_logloss','secondary':'cat_map and two fixed equal within-pair rank blends',
      'source_hashes':{str(p):sha(p) for p in srcs},'protected':protected,
      'reference':'Exact saved incumbent of symmetry_censor audit: current383 chronology stage1 + r382DTSP/structCI stage2.',
      'catboost_params':PARAMS,'objectives':{'cat_logloss':'Logloss','cat_map':'YetiRank:mode=MAP;top=5;permutations=10'},
      'feature_set':'Identical to shipping per-family hybrid; no new feature selection.',
      'split':'Same per-family GroupKFold5 as previous audit;162 pairs entirely outside union outer0/1/2.',
      'gate':{'mean_map5_gain_min':.01,'pool_bootstrap_lower_gt':0,'family_floor':-.01,'both_pool_halves_nonnegative':True,'negative_control':'Saved stage1_only must fail versus incumbent.'},
      'budget':'30 CatBoost fits,15 LightGBM parity controls; two CPU threads; no early stopping on validation.',
      'limitations':['Adaptive reused training-only audit, not nested.', 'Audit uses known family; confirmation fixed predicted family only.', 'MAP surrogate is not guaranteed to optimize the official metric.'],
      'sources':['https://catboost.ai/docs/en/concepts/loss-functions-ranking','https://pypi.org/project/catboost/1.2.10/']}
    write(OUT/'freeze.json',freeze)
    saved=pl.read_parquet(PREV/'scores.parquet').sort('pair_id','hand_id')
    assert sha(PREV/'scores.parquet')==json.loads((PREV/'predictions-frozen.json').read_text(encoding='utf-8'))['scores_sha256']
    audit=pl.read_parquet(AUDIT/'audit-oof.parquet').sort('pair_id','hand_id')
    assert saved.select('pair_id','hand_id').equals(audit.select('pair_id','hand_id'))
    outer=set().union(*(set(fold_pools(f)) for f in (0,1,2)))
    assert not set(audit['table_id'])&outer and audit['pair_id'].n_unique()==162
    structf=pl.read_parquet(AUDIT/'audit-structural-features.parquet')
    struct=[c for c in structf.columns if c not in ('pair_id','hand_id')]
    data,features,_,identity=load_fold(0)
    data=data.join(audit.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi')
    rcols=[f for f in features if not f.startswith('family_') and f not in struct]
    d=audit.drop('oof','oof_rank').join(structf,on=['pair_id','hand_id'],validate='1:1').join(data.select(['pair_id','hand_id']+rcols),on=['pair_id','hand_id'],validate='1:1').sort('pair_id','hand_id')
    d=d.with_columns(pl.Series('oof',saved['stage1_only'].to_numpy()),pl.col(rcols).fill_null(-999.)).with_columns(pl.col('oof').rank('ordinal',descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    shared=struct+['chrono_rank','chrono_frac','n_hands','oof','oof_rank']
    pred={name:d['oof'].to_numpy().copy()-1e6 for name in ARMS}
    receipts=[]
    for fam in FAMILIES:
        ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy())
        rows=d[ix]; cols=shared+(rcols if fam!='coordinated_isolation' else [])
        x=rows.select(cols).to_numpy().astype(float); assert np.isfinite(x).all()
        y=rows['relevance'].to_numpy().astype(int); groups=rows['table_id'].to_numpy(); pid=rows['pair_id'].to_numpy()
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,groups)):
            assert not set(groups[tr])&set(groups[te])
            base=lgb.LGBMClassifier(**CLF,random_state=11).fit(x[tr],y[tr]); pp=base.predict_proba(x[te])[:,1]
            assert np.array_equal(pp+10,saved['incumbent'].to_numpy()[ix[te]]),'Shipping analogue did not reproduce'
            pred['incumbent'][ix[te]]=pp+10
            for name,ranker in [('cat_logloss',False),('cat_map',True)]:
                at=time.perf_counter()
                if ranker:
                    model=CatBoostRanker(**PARAMS,loss_function=freeze['objectives'][name])
                    model.fit(Pool(x[tr],label=y[tr],group_id=pid[tr].tolist()))
                    p=model.predict(x[te])
                else:
                    model=CatBoostClassifier(**PARAMS,loss_function='Logloss')
                    model.fit(x[tr],y[tr]);p=model.predict_proba(x[te])[:,1]
                path=OUT/f'{fam}-{fold}-{name}.cbm';model.save_model(str(path))
                reloaded=CatBoostRanker() if ranker else CatBoostClassifier();reloaded.load_model(str(path))
                reloadp=reloaded.predict(x[te]) if ranker else reloaded.predict_proba(x[te])[:,1]
                assert np.array_equal(p,reloadp)
                # +1000 makes the fixed top20 outrank every untouched tail row.
                pred[name][ix[te]]=p+1000
                blend='blend_map' if ranker else 'blend_logloss'
                pred[blend][ix[te]]=rankmean(pid[te],pp,p)+1000
                receipts.append({'family':fam,'fold':fold,'arm':name,'seconds':time.perf_counter()-at,'model_sha256':sha(path),'reload_exact':True})
            print('Finished',fam,'fold',fold,'seconds',round(time.perf_counter()-t,1),flush=True)
    sf=OUT/'scores.parquet'; key=d.select('pair_id','hand_id','table_id')
    key.with_columns(*[pl.Series(k,v) for k,v in pred.items()]).write_parquet(sf)
    write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(sf),'model_receipts':receipts,'outer_truth_read':False})
    truths={k[0]:set(g['hand_id']) for k,g in d.filter(pl.col('relevance')==1).group_by('pair_id')}
    fams=dict(d.select('pair_id','behavior_family').unique().iter_rows());ptab=dict(d.select('pair_id','table_id').unique().iter_rows());pools=sorted(set(ptab.values()))
    maps={};res={'audit_only':True,'sets':{},'comparisons':{},'passed':[]}
    for name,p in dict(pred,stage1_only=saved['stage1_only'].to_numpy()).items():
        summary,maps[name]=summarize(key.with_columns(pl.Series('score',p)),truths,fams);res['sets'][name]=summary
    for name in [x for x in maps if x!='incumbent']:
        a,b=maps['incumbent'],maps[name];c=boot(a,b,ptab,pools)
        c['by_family']={f:float(np.mean([b[p]-a[p] for p in a if fams[p]==f])) for f in FAMILIES}
        c['halves']={str(h):float(np.mean([b[p]-a[p] for p in a if half(ptab[p])==h])) for h in (0,1)}
        c['gate_pass']=c['mean_delta']>=.01 and c['ci95'][0]>0 and min(c['by_family'].values())>=-.01 and min(c['halves'].values())>=0
        res['comparisons'][name]=c
        if c['gate_pass'] and name!='stage1_only':res['passed'].append(name)
        print(name,json.dumps(c),flush=True)
    res['negative_control_rejected']=not res['comparisons']['stage1_only']['gate_pass']
    res['protected_unchanged']=check_protected()==protected;res['seconds']=time.perf_counter()-t
    write(OUT/'results.json',res);print('DONE',json.dumps({'passed':res['passed'],'seconds':res['seconds']}),flush=True)

if __name__=='__main__':main()
