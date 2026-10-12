"""Reconstruct v13 from frozen models and verified gameplay feature caches.

No previous submission CSV supplies predictions. The original v13 CSV is opened
only after construction for an exact comparison. Raw-feature extraction/training
are outside this bounded replay and remain separately documented stages.
"""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
from datetime import datetime,timezone
import gc
import json
import time
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,LABELS
from tail_pair_experiment import tails
from rank_evidence import context,family_columns,EVIDENCE_COLUMNS
from sequence_evidence_fast import extract
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from sequence_evidence_candidate import verify_source_cache
from evidence_public_calibration import rank

OUT=ROOT/'reports/replay-v13'
CONDITIONAL=ROOT/'outputs_v11_conditional'
PACKAGE=CONDITIONAL/'inference-package'


def main():
    OUT.mkdir(exist_ok=True); cache=OUT/'batches'; cache.mkdir(exist_ok=True)
    verify_source_cache('evaluation')
    model_paths={'old_role':ROOT/'outputs_v2/pair_model_robust.joblib','v5':ROOT/'outputs_v5/pair_model_event_tails.joblib',
        'conditional':CONDITIONAL/'pair-model.joblib','sequence':ROOT/'outputs_v6/sequence_ranker.joblib','ranker':CONDITIONAL/'evidence-model.joblib'}
    assert sha(model_paths['old_role'])==json.loads((ROOT/'outputs_v2/submission_advanced_robust.csv.provenance.json').read_text())['pair_model_sha256']
    assert sha(model_paths['v5'])=='5e24d4f9252377366a8336c808fe36225e47872e46114d59a2aa0837a4dd1d11'
    assert sha(model_paths['sequence'])==json.loads((ROOT/'outputs_v6/fit.json').read_text())['model_sha256']
    fit=json.loads((CONDITIONAL/'fit.json').read_text()); assert all(sha(CONDITIONAL/p)==h for p,h in fit['models_sha256'].items())
    package_plan=json.loads((PACKAGE/'config.json').read_text())
    assert all(sha(PACKAGE/p)==package_plan['files'][p] for p in ['data/hands.parquet','data/seats.parquet','data/actions.parquet','data/evaluation_pairs.csv'])
    source=ROOT/'outputs_v13_coverage/submission_conditional_coverage.csv'
    expected=json.loads((source.parent/'ready-audit.json').read_text()); assert expected['validation']['valid']
    identity={'source_sha256':sha(__file__),'models':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for k,p in model_paths.items()},
        'pair_cache_sha256':sha(ROOT/'outputs_v2/pairs_evaluation.parquet'),'reference_csv_sha256':expected['sha256'],
        'scope':'Rebuild all pair scores/classes and both risk-based selections from models; recompute sequence features/ranks for all 9651820 pair-hands; reapply conditional ranker using verified EQ/CA caches. Read no predecessor prediction CSV. Compare original v13 only at the end. This is inference replay, not raw extraction or retraining.'}
    planpath=OUT/'plan.json'
    if planpath.exists(): assert json.loads(planpath.read_text())==identity
    else: planpath.write_text(json.dumps(identity,indent=2)+'\n')
    models={k:joblib.load(p) for k,p in model_paths.items()}; inverse={v:k for k,v in LABELS.items()}
    pair_cache=pl.read_parquet(ROOT/'outputs_v2/pairs_evaluation.parquet'); predictions=[]
    for path in sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet')):
        name=path.stem; out=cache/(name+'-pairs.parquet'); receipt=cache/(name+'-pairs.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath) and saved['sha256']==sha(out)
            predictions.append(pl.read_parquet(out)); continue
        feature_receipt=json.loads((PACKAGE/'results'/(name+'.json')).read_text())
        assert all(sha(PACKAGE/'results'/p)==h for p,h in feature_receipt['files'].items())
        raw=pl.read_parquet(path)
        frame=pair_cache.join(tails(raw,models['v5']['events']),on=['pair_id','table_id'],validate='1:1').join(pl.read_parquet(PACKAGE/'results'/(name+'-pairs.parquet')),on=['pair_id','table_id'],validate='1:1').sort('pair_id')
        probabilities={k:models[k]['model'].predict_proba(matrix(frame,models[k]['features'])) for k in ['old_role','v5','conditional']}
        for k in probabilities: assert list(models[k]['model'].classes_)==[0,1,2,3]
        result=frame.select('pair_id').with_columns(pl.Series('risk_score',1-probabilities['conditional'][:,0]),pl.Series('old_risk',1-probabilities['v5'][:,0]),
            pl.Series('old_behavior',[inverse[int(i)] for i in np.argmax(probabilities['old_role'][:,1:],axis=1)+1]),
            pl.Series('predicted_behavior',[inverse[int(i)] for i in np.argmax(probabilities['conditional'][:,1:],axis=1)+1]))
        result.write_parquet(out); predictions.append(result)
        receipt.write_text(json.dumps({'plan_sha256':sha(planpath),'sha256':sha(out),'rows':result.height},indent=2)+'\n')
        del raw,frame; gc.collect()
    predictions=pl.concat(predictions); assert predictions.height==predictions['pair_id'].n_unique()==112540
    old=predictions.sort(['old_risk','pair_id'],descending=[True,False]).head(10000).select('pair_id')
    new=predictions.sort(['risk_score','pair_id'],descending=[True,False]).head(10000).select('pair_id')
    added=new.join(old,on='pair_id',how='anti'); selected=pl.concat([old,added]).sort('pair_id')
    assert selected.height==16060 and added.height==6060; selected.write_parquet(OUT/'selected-pairs.parquet')
    pairs=pl.read_csv(PACKAGE/'data/evaluation_pairs.csv').select('pair_id','player_1','player_2'); allhands=pl.read_parquet(PACKAGE/'data/hands.parquet')
    caf=models['ranker']['conditional_features']; hand_count=0
    for index,path in enumerate(sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet'))):
        started=time.perf_counter(); name=path.stem; destination=cache/(name+'-rank.parquet'); receipt=cache/(name+'-rank.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath) and saved['sha256']==sha(destination); hand_count+=saved['hands']; continue
        raw=pl.read_parquet(path); ids=raw.select('hand_id').unique(); hands=allhands.join(ids,on='hand_id',how='semi')
        seats=pl.scan_parquet(PACKAGE/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect(); actions=pl.scan_parquet(PACKAGE/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
        eq=pl.read_parquet(ROOT/'reports/evidence-public-calibration'/(name+'-features.parquet'))
        ca=pl.read_parquet(PACKAGE/'results'/(name+'-evidence.parquet')).drop('table_id')
        added_receipt=json.loads((ROOT/'outputs_v13_coverage/batches'/(name+'.json')).read_text())
        assert all(sha(ROOT/'outputs_v13_coverage/batches'/p)==h for p,h in added_receipt['files'].items())
        extra=pl.concat([eq.join(ca,on=['pair_id','hand_id'],validate='1:1').select(['pair_id','hand_id']+SEQ+EQ+caf),
            pl.read_parquet(ROOT/'outputs_v13_coverage/batches'/(name+'-features.parquet')).join(added,on='pair_id',how='semi').select(['pair_id','hand_id']+SEQ+EQ+caf)])
        pieces=[]; pools=sorted(raw['table_id'].unique()); selected_rows=0
        for offset in range(0,len(pools),5):
            group=pools[offset:offset+5]; part=raw.filter(pl.col('table_id').is_in(group)).sort('pair_id','hand_id')
            part=part.with_columns(pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(part.height)))
            keys=part.select('pair_id','hand_id').join(pairs,on='pair_id',validate='m:1'); ph=hands.filter(pl.col('table_id').is_in(group)); ids=ph.select('hand_id')
            seq=extract(keys,ph,seats.join(ids,on='hand_id',how='semi'),actions.join(ids,on='hand_id',how='semi'))
            frame=part.join(seq,on=['pair_id','hand_id'],validate='1:1').join(predictions.select('pair_id','old_behavior','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id')
            frame=context(frame,[c for c in raw.columns if c.startswith('v2_')]+SEQ)
            base=rank(frame,models['sequence']['model'].predict(matrix(family_columns(frame,'old_behavior'),models['sequence']['features']),num_threads=2))
            use=frame.join(selected,on='pair_id',how='semi'); known=extra.join(use.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi').sort('pair_id','hand_id')
            assert use.select(['pair_id','hand_id']+SEQ).equals(known.select(['pair_id','hand_id']+SEQ)),'Fresh sequence features differ from selected caches'
            use=use.join(known.select(['pair_id','hand_id']+EQ+caf),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
            use=context(use,EQ+caf); selected_rows+=use.height
            enriched=rank(use,models['ranker']['model'].predict(matrix(family_columns(use,'predicted_behavior'),models['ranker']['features']),num_threads=2))
            pieces.append(pl.concat([base.join(selected,on='pair_id',how='anti'),enriched]))
        result=pl.concat(pieces).sort('pair_id'); assert result.height==raw['pair_id'].n_unique()
        result.write_parquet(destination); hand_count+=raw.height
        saved={'plan_sha256':sha(planpath),'sha256':sha(destination),'hands':raw.height,'selected_hands':selected_rows,'pairs':result.height,'selected_sequence_features_exact':True,'seconds':time.perf_counter()-started}
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); print(json.dumps({'batch':index,**saved}),flush=True)
        del raw,hands,seats,actions,extra,frame,use,known,pieces; gc.collect()
    ranking=pl.read_parquet(sorted(cache.glob('batch_*-rank.parquet'))); template=pl.read_csv(ROOT/'data/sample_submission.csv',infer_schema=False)
    expanded=ranking.select('pair_id',*[pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i,c in enumerate(EVIDENCE_COLUMNS)])
    output=template.select('pair_id').join(predictions.select('pair_id',pl.col('risk_score').cast(pl.String),'predicted_behavior'),on='pair_id',validate='1:1',maintain_order='left').join(expanded,on='pair_id',validate='1:1',maintain_order='left').select(template.columns)
    path=OUT/'submission_replay.csv'; output.write_csv(path)
    # First and only access to an existing prediction CSV, after model replay.
    assert sha(source)==expected['sha256']; reference=pl.read_csv(source,infer_schema=False)
    same=output.equals(reference); byte_equal=sha(path)==expected['sha256']
    report={'completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(planpath),'rows':output.height,'pair_hands_replayed':hand_count,'selected_pairs':selected.height,
        'identical_cells':same,'byte_identical':byte_equal,'sha256':sha(path),'expected_sha256':expected['sha256'],'previous_csv_used_only_for_final_comparison':True,
        'scope':'Inference replay from frozen models and verified feature caches; not a fresh raw-feature build or retraining run.'}
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
    assert same and byte_equal

if __name__=='__main__': main()
