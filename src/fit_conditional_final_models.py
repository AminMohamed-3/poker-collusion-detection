"""Refit only the candidates that passed the frozen conditional-action checks."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import json
from datetime import datetime, timezone
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT, LABELS, TRAIN_VIEWS, MODEL_PARAMS, thin, aggregate, matrix, sha
from tail_pair_experiment import tails
from evaluate_conditional_action_screen import extra_aggregate
from evidence_capacity_experiment import load_sequence, fit_variant
from rank_evidence import context, family_columns, PARAMS
from equity_evidence_features import FEATURES as EQ

OUT=ROOT/'outputs_v11_conditional'


def main():
    confirmation=ROOT/'reports/conditional-action-confirmation/results/summary.json'
    outcome=json.loads(confirmation.read_text()); assert outcome['selected_pair']=='conditional' and outcome['advance_evidence']
    ca_path=OUT/'policy/development-hand-features.parquet'
    policy=json.loads((OUT/'policy/summary.json').read_text()); assert policy['features_sha256']==sha(ca_path)
    identity={'source_sha256':sha(__file__),'confirmation_sha256':sha(confirmation),'policy_features_sha256':sha(ca_path),
        'pair_parameters':MODEL_PARAMS,'evidence_parameters':PARAMS,
        'source_submission_sha256':sha(ROOT/'outputs_v10_tabicl/submission_tabicl.csv'),
        'decision':'Train the selected pure conditional risk model and conditional evidence ranker. Preserve production v2 behavior assignments. Controlled risk-only and evidence-only candidates will be validated before any combined candidate.'}
    path=OUT/'fit-plan.json'
    if path.exists(): assert json.loads(path.read_text())==identity
    else: path.write_text(json.dumps(identity,indent=2)+'\n')
    if (OUT/'fit.json').exists():
        saved=json.loads((OUT/'fit.json').read_text()); assert saved['identity']==identity
        assert all(sha(OUT/p)==h for p,h in saved['models_sha256'].items()); print('Verified existing final models'); return
    ca=pl.read_parquet(ca_path); caf=policy['features']; extra=[c+s for s in ['_mean','_std','_max','_q95'] for c in caf]
    labels=pl.read_csv(ROOT/'data/development_labels.csv')
    raw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    names=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    meta=json.loads((ROOT/'reports/tail-pair-experiment/metrics.json').read_text()); base=meta['base_features']+meta['extra_features']
    combined=raw.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(pl.col(caf).fill_null(0))
    full=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(tails(raw,events),on=['pair_id','table_id']).join(labels,on='pair_id').join(extra_aggregate(combined,caf),on=['pair_id','table_id']).sort('pair_id')
    views=[full]
    for k in TRAIN_VIEWS:
        selected=thin(combined,*k)
        view=aggregate(selected,names).join(tails(selected,events),on=['pair_id','table_id']).join(labels,on='pair_id').join(extra_aggregate(selected,caf),on=['pair_id','table_id']).select(full.columns).sort('pair_id')
        views.append(view)
    train=pl.concat(views); features=base+extra
    assert train.height==5580 and full.height==1860 and len(features)==897
    model=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(train,features),np.array([LABELS[f] for f in train['behavior_family']]),sample_weight=np.full(train.height,1/3))
    joblib.dump({'model':model,'features':features,'conditional_features':caf,'events':events},OUT/'pair-model.joblib')
    check=matrix(train.head(64),features)
    np.testing.assert_array_equal(model.predict_proba(check),joblib.load(OUT/'pair-model.joblib')['model'].predict_proba(check))
    print('Final 897-feature pair model fitted; reload exact',flush=True)
    data,current,_,_,_=load_sequence()
    data=context(data.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
    data=context(data.join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0)),caf)
    ef=current+EQ+['context_'+c for c in EQ]+caf+['context_'+c for c in caf]
    assert len(ef)==382 and data['pair_id'].n_unique()==372
    ranker=fit_variant(data,ef,PARAMS)
    joblib.dump({'model':ranker,'features':ef,'conditional_features':caf},OUT/'evidence-model.joblib')
    x=matrix(family_columns(data.head(64),'behavior_family'),ef)
    np.testing.assert_array_equal(ranker.predict(x),joblib.load(OUT/'evidence-model.joblib')['model'].predict(x))
    receipt={'completed_at':datetime.now(timezone.utc).isoformat(),'identity':identity,'pair_rows':1860,'pair_training_views':5580,'pair_features':897,
        'evidence_pairs':372,'evidence_hands':data.height,'evidence_features':382,'reload_exact':True,
        'models_sha256':{p:sha(OUT/p) for p in ['pair-model.joblib','evidence-model.joblib']}}
    (OUT/'fit.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
