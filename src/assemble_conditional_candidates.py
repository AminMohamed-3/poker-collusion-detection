"""Incrementally score verified batches, reproduce v5/v8, and assemble entries."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import argparse
from datetime import datetime, timezone
import gc
import json
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix
from tail_pair_experiment import tails
from rank_evidence import context,family_columns,EVIDENCE_COLUMNS
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from evidence_public_calibration import rank

OUT=ROOT/'outputs_v11_conditional'
FEATURES=OUT/'inference-package/results'
SOURCE=ROOT/'outputs_v10_tabicl/submission_tabicl.csv'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--available-only',action='store_true'); args=parser.parse_args()
    fit=json.loads((OUT/'fit.json').read_text())
    assert all(sha(OUT/p)==h for p,h in fit['models_sha256'].items())
    identity={'source_sha256':sha(__file__),'fit_sha256':sha(OUT/'fit.json'),'reference_sha256':sha(SOURCE),
        'inference_config_sha256':sha(OUT/'inference-package/config.json'),
        'evidence_selection_sha256':sha(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet'),
        'behavior':'Preserve every production v2 behavior assignment.',
        'candidates':['conditional risk only','conditional evidence only','both conditional changes'],
        'scope':'Rerank the unchanged original 10000-pair evidence selection; all pair risks receive new predictions. All other evidence remains v10.'}
    planpath=OUT/'assembly-plan.json'
    if planpath.exists(): assert json.loads(planpath.read_text())==identity
    else: planpath.write_text(json.dumps(identity,indent=2)+'\n')
    source=pl.read_csv(SOURCE,infer_schema=False)
    assert sha(SOURCE)==fit['identity']['source_submission_sha256']
    controlsource=pl.read_csv(ROOT/'outputs_v5/submission_event_tails.csv',infer_schema=False)
    pairmodel=joblib.load(OUT/'pair-model.joblib'); evidencemodel=joblib.load(OUT/'evidence-model.joblib')
    v5=joblib.load(ROOT/'outputs_v5/pair_model_event_tails.joblib'); v8=joblib.load(ROOT/'outputs_v8_equity/ranker.joblib')
    pairs=pl.read_parquet(ROOT/'outputs_v2/pairs_evaluation.parquet')
    selected=pl.read_parquet(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet')
    cache=OUT/'scored-batches'; cache.mkdir(exist_ok=True)
    for index,path in enumerate(sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet'))):
        name=path.stem; feature_receipt=FEATURES/(name+'.json'); receipt=cache/(name+'.json')
        if not feature_receipt.exists():
            if args.available_only: continue
            raise FileNotFoundError(feature_receipt)
        original=json.loads(feature_receipt.read_text()); assert original['plan_sha256']==identity['inference_config_sha256']
        assert all(sha(FEATURES/p)==h for p,h in original['files'].items())
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath) and saved['feature_receipt_sha256']==sha(feature_receipt)
            assert all(sha(cache/p)==h for p,h in saved['files'].items()); continue
        raw=pl.read_parquet(path); extra=tails(raw,v5['events'])
        conditional=pl.read_parquet(FEATURES/(name+'-pairs.parquet'))
        frame=pairs.join(extra,on=['pair_id','table_id'],validate='1:1').join(conditional,on=['pair_id','table_id'],validate='1:1').sort('pair_id')
        assert frame.height==extra.height==conditional.height==original['pairs']
        expected=controlsource.join(frame.select('pair_id'),on='pair_id',how='semi').sort('pair_id')['risk_score'].cast(pl.Float64).to_numpy()
        control=1-v5['model'].predict_proba(matrix(frame,v5['features']))[:,0]
        np.testing.assert_array_equal(control,expected)
        risk=1-pairmodel['model'].predict_proba(matrix(frame,pairmodel['features']))[:,0]
        assert np.isfinite(risk).all() and np.all((risk>=0)&(risk<=1))
        pairpath=cache/(name+'-risk.parquet'); frame.select('pair_id').with_columns(pl.Series('risk_score',risk)).write_parquet(pairpath)
        # Reproduce the original public-calibration tie stream, including all
        # unselected raw hands before subsetting the same 10000 pairs.
        pools=sorted(raw['table_id'].unique()); chunks=[]
        for offset in range(0,len(pools),5):
            part=raw.filter(pl.col('table_id').is_in(pools[offset:offset+5])).sort('pair_id','hand_id')
            part=part.with_columns(pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(part.height)))
            chunks.append(part.join(selected,on='pair_id',how='semi'))
        rankdata=pl.concat(chunks).join(pl.read_parquet(ROOT/'reports/evidence-public-calibration'/(name+'-features.parquet')),on=['pair_id','hand_id'],validate='1:1')
        ca=pl.read_parquet(FEATURES/(name+'-evidence.parquet'))
        assert rankdata.height==ca.height==original['evidence_hands']
        rankdata=rankdata.join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1').join(source.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id')
        rankdata=family_columns(context(rankdata,[c for c in raw.columns if c.startswith('v2_')]+SEQ+EQ+pairmodel['conditional_features']),'predicted_behavior')
        oldranking=rank(rankdata,v8['model'].predict(matrix(rankdata,v8['features']),num_threads=2))
        comparison=oldranking.join(source.select('pair_id',pl.concat_list(EVIDENCE_COLUMNS).alias('expected')),on='pair_id',validate='1:1')
        assert all(actual==[h for h in wanted if h!='NO_EVIDENCE'] for actual,wanted in comparison.select('hands','expected').iter_rows()),'v8 evidence control changed'
        score=evidencemodel['model'].predict(matrix(rankdata,evidencemodel['features']),num_threads=2)
        assert np.isfinite(score).all()
        newranking=rank(rankdata,score); rankpath=cache/(name+'-rank.parquet'); newranking.write_parquet(rankpath)
        saved={'plan_sha256':sha(planpath),'feature_receipt_sha256':sha(feature_receipt),'v5_risks_exact':True,'v8_evidence_exact':True,
            'pair_rows':frame.height,'reranked_pairs':newranking.height,'files':{p.name:sha(p) for p in [pairpath,rankpath]}}
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); print(json.dumps({'batch':index,**saved}),flush=True)
        del raw,extra,conditional,frame,chunks,rankdata,ca; gc.collect()
    complete=sorted(cache.glob('batch_*.json'))
    if len(complete)!=20:
        assert args.available_only; print(json.dumps({'scored_batches':len(complete),'waiting_for':20-len(complete)}),flush=True); return
    predictions=pl.read_parquet(sorted(cache.glob('batch_*-risk.parquet')))
    ranking=pl.read_parquet(sorted(cache.glob('batch_*-rank.parquet')))
    assert predictions.height==predictions['pair_id'].n_unique()==112540
    assert ranking.height==ranking['pair_id'].n_unique()==selected.height==10000
    assert set(predictions['pair_id'])==set(source['pair_id']) and set(ranking['pair_id'])==set(selected['pair_id'])
    riskonly=source.join(predictions.rename({'risk_score':'_new_risk'}),on='pair_id',validate='1:1',maintain_order='left').with_columns(pl.col('_new_risk').cast(pl.String).alias('risk_score')).select(source.columns)
    expanded=ranking.select('pair_id',*[pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(c+'_new') for i,c in enumerate(EVIDENCE_COLUMNS)])
    evidenceonly=source.join(expanded,on='pair_id',how='left',validate='1:1',maintain_order='left').with_columns(*[pl.coalesce(c+'_new',c).alias(c) for c in EVIDENCE_COLUMNS]).select(source.columns)
    combined=evidenceonly.with_columns(riskonly['risk_score'])
    assert riskonly.select([c for c in source.columns if c!='risk_score']).equals(source.select([c for c in source.columns if c!='risk_score']))
    assert evidenceonly.select('pair_id','risk_score','predicted_behavior').equals(source.select('pair_id','risk_score','predicted_behavior'))
    assert evidenceonly.join(selected,on='pair_id',how='anti').equals(source.join(selected,on='pair_id',how='anti'))
    assert combined.select('pair_id','predicted_behavior').equals(source.select('pair_id','predicted_behavior'))
    entries=[]
    for name,frame in [('risk',riskonly),('evidence',evidenceonly),('combined',combined)]:
        destination=OUT/('submission_conditional_'+name+'.csv')
        if destination.exists(): assert pl.read_csv(destination,infer_schema=False).equals(frame)
        else: frame.write_csv(destination)
        assert pl.read_csv(destination,infer_schema=False).equals(frame)
        entries.append({'candidate':name,'file':destination.name,'sha256':sha(destination),'rows':frame.height})
    receipt={'completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(planpath),'entries':entries,
        'v5_risk_and_v8_evidence_controls_exact_all_batches':True,'full_membership_validation_required':True,
        'changed_evidence_rows':sum(a!=b for a,b in zip(source.select(EVIDENCE_COLUMNS).rows(),evidenceonly.select(EVIDENCE_COLUMNS).rows())),
        'risk_summary':predictions.select(pl.col('risk_score').min().alias('min'),pl.col('risk_score').max().alias('max'),pl.col('risk_score').mean().alias('mean')).to_dicts()[0]}
    (OUT/'assembly.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
