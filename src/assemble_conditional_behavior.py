"""Assemble the prevalidated fixed-model behavior change with exact v11 controls."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from datetime import datetime, timezone
import gc
import json
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,LABELS
from tail_pair_experiment import tails
from rank_evidence import context,family_columns,EVIDENCE_COLUMNS
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from evidence_public_calibration import rank
from validation import validate_submission

OLD=ROOT/'outputs_v11_conditional'
OUT=ROOT/'outputs_v12_behavior'
FEATURES=OLD/'inference-package/results'
SOURCE=OLD/'submission_conditional_combined.csv'


def main():
    OUT.mkdir(exist_ok=True)
    confirmation=ROOT/'reports/conditional-behavior-experiment/coherent-summary.json'
    assert json.loads(confirmation.read_text())['advance_behavior']
    ready=json.loads((OLD/'ready-audit.json').read_text())
    protected={e['file']:e['sha256'] for e in ready['entries']}
    assert all(sha(OLD/p)==h for p,h in protected.items())
    fit=json.loads((OLD/'fit.json').read_text())
    assert all(sha(OLD/p)==h for p,h in fit['models_sha256'].items())
    identity={'source_sha256':sha(__file__),'reference_sha256':sha(SOURCE),
        'confirmation_sha256':sha(confirmation),'ordinary_confirmation_sha256':sha(ROOT/'reports/conditional-behavior-experiment/summary.json'),
        'fit_sha256':sha(OLD/'fit.json'),'v11_ready_audit_sha256':sha(OLD/'ready-audit.json'),
        'scope':'Preserve every v11 combined risk. Assign supported target family by conditional model argmax excluding class zero. Rerank unchanged original 10000 selected pairs using unchanged v11 ranker and new family flags. Other evidence unchanged.'}
    planpath=OUT/'assembly-plan.json'
    if planpath.exists(): assert json.loads(planpath.read_text())==identity
    else: planpath.write_text(json.dumps(identity,indent=2)+'\n')
    source=pl.read_csv(SOURCE,infer_schema=False)
    pairmodel=joblib.load(OLD/'pair-model.joblib'); evidencemodel=joblib.load(OLD/'evidence-model.joblib')
    pairs=pl.read_parquet(ROOT/'outputs_v2/pairs_evaluation.parquet')
    selected=pl.read_parquet(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet')
    inverse={v:k for k,v in LABELS.items()}
    cache=OUT/'scored-batches'; cache.mkdir(exist_ok=True)
    for index,path in enumerate(sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet'))):
        name=path.stem; receipt=cache/(name+'.json')
        feature_receipt=FEATURES/(name+'.json'); original=json.loads(feature_receipt.read_text())
        assert all(sha(FEATURES/p)==h for p,h in original['files'].items())
        old_receipt=OLD/'scored-batches'/(name+'.json'); old=json.loads(old_receipt.read_text())
        assert all(sha(OLD/'scored-batches'/p)==h for p,h in old['files'].items())
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            assert saved['plan_sha256']==sha(planpath) and saved['feature_receipt_sha256']==sha(feature_receipt) and saved['v11_receipt_sha256']==sha(old_receipt)
            assert all(sha(cache/p)==h for p,h in saved['files'].items()); continue
        raw=pl.read_parquet(path)
        frame=pairs.join(tails(raw,pairmodel['events']),on=['pair_id','table_id'],validate='1:1').join(pl.read_parquet(FEATURES/(name+'-pairs.parquet')),on=['pair_id','table_id'],validate='1:1').sort('pair_id')
        probability=pairmodel['model'].predict_proba(matrix(frame,pairmodel['features']))
        oldrisk=source.join(frame.select('pair_id'),on='pair_id',how='semi').sort('pair_id')['risk_score'].cast(pl.Float64).to_numpy()
        np.testing.assert_array_equal(1-probability[:,0],oldrisk)
        assert list(pairmodel['model'].classes_)==[0,1,2,3]
        classes=pairmodel['model'].classes_[1:][np.argmax(probability[:,1:],axis=1)]
        assignments=frame.select('pair_id').with_columns(pl.Series('new_behavior',[inverse[int(x)] for x in classes]))
        pairpath=cache/(name+'-behavior.parquet'); assignments.write_parquet(pairpath)
        pools=sorted(raw['table_id'].unique()); chunks=[]
        for offset in range(0,len(pools),5):
            part=raw.filter(pl.col('table_id').is_in(pools[offset:offset+5])).sort('pair_id','hand_id')
            part=part.with_columns(pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(part.height)))
            chunks.append(part.join(selected,on='pair_id',how='semi'))
        rankdata=pl.concat(chunks).join(pl.read_parquet(ROOT/'reports/evidence-public-calibration'/(name+'-features.parquet')),on=['pair_id','hand_id'],validate='1:1')
        rankdata=rankdata.join(pl.read_parquet(FEATURES/(name+'-evidence.parquet')).drop('table_id'),on=['pair_id','hand_id'],validate='1:1').join(source.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id')
        rankdata=context(rankdata,[c for c in raw.columns if c.startswith('v2_')]+SEQ+EQ+pairmodel['conditional_features'])
        oldranking=rank(rankdata,evidencemodel['model'].predict(matrix(family_columns(rankdata,'predicted_behavior'),evidencemodel['features']),num_threads=2))
        assert oldranking.sort('pair_id').equals(pl.read_parquet(OLD/'scored-batches'/(name+'-rank.parquet')).sort('pair_id')),'v11 evidence changed'
        rankdata=rankdata.join(assignments,on='pair_id',validate='m:1',maintain_order='left')
        score=evidencemodel['model'].predict(matrix(family_columns(rankdata,'new_behavior'),evidencemodel['features']),num_threads=2)
        assert np.isfinite(score).all()
        ranking=rank(rankdata,score); rankpath=cache/(name+'-rank.parquet'); ranking.write_parquet(rankpath)
        saved={'plan_sha256':sha(planpath),'feature_receipt_sha256':sha(feature_receipt),'v11_receipt_sha256':sha(old_receipt),
            'v11_risks_exact':True,'v11_evidence_exact':True,'pair_rows':frame.height,'reranked_pairs':ranking.height,'files':{p.name:sha(p) for p in [pairpath,rankpath]}}
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); print(json.dumps({'batch':index,**saved}),flush=True)
        del raw,frame,chunks,rankdata; gc.collect()
    assignments=pl.read_parquet(sorted(cache.glob('batch_*-behavior.parquet')))
    ranking=pl.read_parquet(sorted(cache.glob('batch_*-rank.parquet')))
    assert assignments.height==assignments['pair_id'].n_unique()==source.height==112540
    assert ranking.height==ranking['pair_id'].n_unique()==selected.height==10000
    expanded=ranking.select('pair_id',*[pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(c+'_new') for i,c in enumerate(EVIDENCE_COLUMNS)])
    result=source.join(assignments,on='pair_id',validate='1:1',maintain_order='left').join(expanded,on='pair_id',how='left',validate='1:1',maintain_order='left').with_columns(pl.col('new_behavior').alias('predicted_behavior'),*[pl.coalesce(c+'_new',c).alias(c) for c in EVIDENCE_COLUMNS]).select(source.columns)
    assert result.select('pair_id','risk_score').equals(source.select('pair_id','risk_score'))
    assert result.join(selected,on='pair_id',how='anti').select(['pair_id']+EVIDENCE_COLUMNS).equals(source.join(selected,on='pair_id',how='anti').select(['pair_id']+EVIDENCE_COLUMNS))
    dest=OUT/'submission_conditional_behavior.csv'
    if dest.exists(): assert pl.read_csv(dest,infer_schema=False).equals(result)
    else: result.write_csv(dest)
    assert pl.read_csv(dest,infer_schema=False).equals(result)
    report=validate_submission(dest,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv','evaluation')
    assert report.valid,report.to_dict()
    (OUT/'validation.json').write_text(json.dumps(report.to_dict(),indent=2)+'\n')
    assert all(sha(OLD/p)==h for p,h in protected.items())
    audit={'created_at':datetime.now(timezone.utc).isoformat(),'file':dest.name,'sha256':sha(dest),'rows':result.height,
        'validation_sha256':sha(OUT/'validation.json'),'plan_sha256':sha(planpath),'validation':report.to_dict(),
        'all_v11_control_predictions_exact':True,'v11_candidate_files_unchanged':True,'risk_column_unchanged':True,'unselected_evidence_unchanged':True,
        'changed_behavior_rows':int((result['predicted_behavior']!=source['predicted_behavior']).sum()),
        'changed_evidence_rows':sum(a!=b for a,b in zip(result.select(EVIDENCE_COLUMNS).rows(),source.select(EVIDENCE_COLUMNS).rows())),
        'public_status':'Unsubmitted; compare with v11 combined to isolate behavior effect.'}
    (OUT/'ready-audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit),flush=True)


if __name__=='__main__': main()
