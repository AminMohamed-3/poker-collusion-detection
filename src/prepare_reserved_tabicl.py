"""Prepare the unchanged v10 control on the frozen less-reused partition."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import json
import shutil
import tarfile
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,aggregate,thin,TRAIN_VIEWS,HELDOUT_VIEW,MODEL_PARAMS,LABELS,assert_disjoint
from tail_pair_experiment import tails

OUT=ROOT/'reports/conditional-reserved-audit'


def data_views():
    raw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    names=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    labels=pl.read_csv(ROOT/'data/development_labels.csv')
    full=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(tails(raw,events),on=['pair_id','table_id']).join(labels,on='pair_id').sort('pair_id')
    views={}
    for key in TRAIN_VIEWS+[HELDOUT_VIEW]:
        part=thin(raw,*key); views[key]=aggregate(part,names).join(tails(part,events),on=['pair_id','table_id']).join(labels,on='pair_id').select(full.columns).sort('pair_id')
    return raw,full,views


def main():
    out=OUT/'gpu-package'; out.mkdir(exist_ok=False)
    pools=json.loads((OUT/'plan.json').read_text())['held_pools']; raw,full,views=data_views()
    train=full.filter(~pl.col('table_id').is_in(pools)); held=full.filter(pl.col('table_id').is_in(pools)); short=views[HELDOUT_VIEW].filter(pl.col('table_id').is_in(pools))
    augmented=pl.concat([train]+[views[k].filter(~pl.col('table_id').is_in(pools)) for k in TRAIN_VIEWS]); assert_disjoint(augmented,held)
    meta=json.loads((ROOT/'reports/tail-pair-experiment/metrics.json').read_text()); features=meta['base_features']+meta['extra_features']
    model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(augmented,features),np.array([LABELS[f] for f in augmented['behavior_family']]),sample_weight=np.full(augmented.height,1/3))
    joblib.dump({'model':model,'features':features},OUT/'v5-control.joblib')
    chosen=[features[i] for i in np.argsort(-model.booster_.feature_importance('gain'),kind='stable')[:128]]
    arrays={'x_train':matrix(train,chosen),'y_train':np.array([LABELS[f] for f in train['behavior_family']]),'y_held':np.array([LABELS[f] for f in held['behavior_family']])}
    for history,frame in [('full',held),('shortened',short)]:
        assert frame['pair_id'].to_list()==pl.read_csv(OUT/'held-pairs.csv')['pair_id'].to_list()
        arrays['x_'+history]=matrix(frame,chosen); arrays['control_'+history]=1-model.predict_proba(matrix(frame,features))[:,0]
    np.savez_compressed(out/'arrays.npz',**arrays)
    (OUT/'features.json').write_text(json.dumps({'features':chosen,'train_pairs':train.height,'held_pairs':held.height,'source_sha256':sha(__file__),'v5_model_sha256':sha(OUT/'v5-control.joblib')},indent=2)+'\n')
    shutil.copy2(ROOT/'src/run_reserved_tabicl.py',out/'run_reserved_tabicl.py')
    previous=json.loads((ROOT/'reports/conditional-coherent-history/gpu-package/plan.json').read_text())
    plan={k:previous[k] for k in ['checkpoint_repo','checkpoint_file','checkpoint_revision','checkpoint_sha256','common_parameters']}
    plan.update({'parent_plan_sha256':sha(OUT/'plan.json'),'n_estimators':2,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    archive=OUT/'gpu-bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for path in out.iterdir(): t.add(path,arcname=path.name)
    receipt={'sha256':sha(archive),'bytes':archive.stat().st_size,'plan_sha256':sha(out/'plan.json')}
    (OUT/'gpu-bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
