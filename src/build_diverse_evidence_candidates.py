"""Fit and prepare five admitted evidence recipes, keeping v13 risk/behavior."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
from datetime import datetime,timezone
import gc,json,time
import joblib
import numpy as np
import polars as pl
from diverse_next_day import OUT as RESEARCH,write,SEEDS,augmented_ranker
from diverse_evidence_panel import NAMES,scores
from robust_pair_experiment import ROOT,sha,matrix
from evidence_capacity_experiment import load_sequence,fit_variant
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from rank_evidence import context,family_columns,PARAMS,EVIDENCE_COLUMNS
from evidence_public_calibration import rank
from validation import validate_submission
from sequence_evidence_candidate import verify_source_cache

OUT=ROOT/'outputs_v14_evidence_portfolio'
BASE=ROOT/'outputs_v13_coverage/submission_conditional_coverage.csv'
CONDITIONAL=ROOT/'outputs_v11_conditional'; PACKAGE=CONDITIONAL/'inference-package'

def main():
    OUT.mkdir(exist_ok=True); (OUT/'batches').mkdir(exist_ok=True)
    admission=json.loads((RESEARCH/'evidence-panel/admission.json').read_text()); assert set(admission['eligible'])==set(NAMES)
    identity={'source_sha256':sha(__file__),'admission_sha256':sha(RESEARCH/'evidence-panel/admission.json'),'panel_source_sha256':sha(ROOT/'src/diverse_evidence_panel.py'),
        'panel_plan_sha256':sha(RESEARCH/'evidence-panel-plan.json'),'source_csv_sha256':sha(BASE),'baseline_model_sha256':sha(CONDITIONAL/'evidence-model.joblib'),
        'final_reference_features_sha256':sha(CONDITIONAL/'policy/development-hand-features.parquet'),'selection_sha256':sha(ROOT/'reports/replay-v13/selected-pairs.parquet'),
        'candidate_names':NAMES,'purpose':'Five admitted portfolio alternatives, not automatic replacements; no upload today, quota exhausted. All risk/behavior and outside-selection evidence exactly retained.'}
    if (OUT/'plan.json').exists(): assert json.loads((OUT/'plan.json').read_text())==identity
    else: write(OUT/'plan.json',identity)
    verify_source_cache('development'); verify_source_cache('evaluation')
    fitted=joblib.load(CONDITIONAL/'evidence-model.joblib'); caf=fitted['conditional_features']; features=fitted['features']
    fitpath=OUT/'fit.json'
    if not fitpath.exists():
        data,current,_,_,_=load_sequence()
        data=context(data.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=['pair_id','hand_id'],validate='1:1'),EQ)
        data=context(data.join(pl.read_parquet(CONDITIONAL/'policy/development-hand-features.parquet').drop('table_id'),on=['pair_id','hand_id'],validate='1:1'),caf)
        assert features==current+EQ+['context_'+c for c in EQ]+caf+['context_'+c for c in caf]
        check=family_columns(data.sort('pair_id','hand_id'),'behavior_family'); x=matrix(check,features)
        for seed in SEEDS[1:]:
            model=fit_variant(data,features,{**PARAMS,'random_state':seed}); path=OUT/f'evidence-seed-{seed}.joblib'; joblib.dump({'model':model,'features':features},path)
            np.testing.assert_array_equal(model.predict(x,num_threads=2),joblib.load(path)['model'].predict(x,num_threads=2))
        contexts=[c for c in data.columns if c.startswith('v2_')]+SEQ+EQ+caf
        model=augmented_ranker(data,features,contexts); path=OUT/'evidence-history.joblib'; joblib.dump({'model':model,'features':features},path)
        np.testing.assert_array_equal(model.predict(x,num_threads=2),joblib.load(path)['model'].predict(x,num_threads=2))
        write(fitpath,{'plan_sha256':sha(OUT/'plan.json'),'training_pairs':data['pair_id'].n_unique(),'training_hands':data.height,'annotations':int(data['relevance'].sum()),'features':len(features),
            'models':{p.name:sha(p) for p in OUT.glob('*.joblib')},'reload_predictions_exact':True})
        del data,check,x,model; gc.collect()
    fit=json.loads(fitpath.read_text()); assert fit['plan_sha256']==sha(OUT/'plan.json') and all(sha(OUT/p)==h for p,h in fit['models'].items())
    bag=[fitted]+[joblib.load(OUT/f'evidence-seed-{s}.joblib') for s in SEEDS[1:]]; augmented=joblib.load(OUT/'evidence-history.joblib')
    source=pl.read_csv(BASE,infer_schema=False); selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet')
    old=pl.read_parquet(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id'); added=selected.join(old,on='pair_id',how='anti'); assert selected.height==16060 and added.height==6060
    for index,path in enumerate(sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet'))):
        name=path.stem; dest=OUT/'batches'; receipt=dest/(name+'.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(OUT/'plan.json') and all(sha(dest/p)==h for p,h in saved['files'].items()); continue
        start=time.perf_counter(); raw=pl.read_parquet(path); events=[c for c in raw.columns if c.startswith('v2_')]
        eq=pl.read_parquet(ROOT/'reports/evidence-public-calibration'/(name+'-features.parquet'))
        ca=pl.read_parquet(PACKAGE/'results'/(name+'-evidence.parquet')).drop('table_id')
        feature_receipt=json.loads((PACKAGE/'results'/(name+'.json')).read_text()); assert all(sha(PACKAGE/'results'/p)==h for p,h in feature_receipt['files'].items())
        extra_receipt=json.loads((ROOT/'outputs_v13_coverage/batches'/(name+'.json')).read_text()); assert all(sha(ROOT/'outputs_v13_coverage/batches'/p)==h for p,h in extra_receipt['files'].items())
        extra=pl.concat([eq.join(ca,on=['pair_id','hand_id'],validate='1:1').select(['pair_id','hand_id']+SEQ+EQ+caf),pl.read_parquet(ROOT/'outputs_v13_coverage/batches'/(name+'-features.parquet')).join(added,on='pair_id',how='semi').select(['pair_id','hand_id']+SEQ+EQ+caf)])
        outputs={k:[] for k in ['baseline']+NAMES}; pools=sorted(raw['table_id'].unique()); count=0
        for offset in range(0,len(pools),5):
            part=raw.filter(pl.col('table_id').is_in(pools[offset:offset+5])).sort('pair_id','hand_id')
            part=part.with_columns(pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(part.height))).join(selected,on='pair_id',how='semi')
            if not part.height: continue
            frame=part.join(extra,on=['pair_id','hand_id'],validate='1:1').join(source.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id')
            assert frame.height==part.height
            frame=family_columns(context(frame,events+SEQ+EQ+caf),'predicted_behavior'); x=matrix(frame,features); count+=frame.height
            base=fitted['model'].predict(x,num_threads=2); seed=np.mean([b['model'].predict(x,num_threads=2) for b in bag],axis=0); history=augmented['model'].predict(x,num_threads=2)
            for candidate,values in scores(frame,base,seed,history).items(): outputs[candidate].append(rank(frame,values))
        files={}
        for candidate,pieces in outputs.items():
            ranked=pl.concat(pieces).sort('pair_id'); expanded=ranked.select('pair_id',*[pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i,c in enumerate(EVIDENCE_COLUMNS)])
            if candidate=='baseline': assert expanded.equals(source.join(expanded.select('pair_id'),on='pair_id',how='semi').select(expanded.columns).sort('pair_id')),'Incumbent ranks differ'
            output=dest/(name+'-'+candidate+'.parquet'); expanded.write_parquet(output); files[output.name]=sha(output)
        write(receipt,{'plan_sha256':sha(OUT/'plan.json'),'files':files,'selected_hands':count,'incumbent_evidence_exact':True,'seconds':time.perf_counter()-start})
        print(json.dumps({'batch':index,'selected_hands':count,'seconds':time.perf_counter()-start}),flush=True)
        del raw,part,extra,frame,x,outputs; gc.collect()
    entries=[]
    for candidate in NAMES:
        replacement=pl.read_parquet(sorted((OUT/'batches').glob('batch_*-'+candidate+'.parquet'))).sort('pair_id'); assert replacement.height==16060
        result=pl.concat([source.join(selected,on='pair_id',how='anti'),source.join(selected,on='pair_id',how='semi').drop(EVIDENCE_COLUMNS).join(replacement,on='pair_id',validate='1:1')]).select(source.columns)
        result=source.select('pair_id').join(result,on='pair_id',validate='1:1',maintain_order='left')
        assert result.select('pair_id','risk_score','predicted_behavior').equals(source.select('pair_id','risk_score','predicted_behavior'))
        assert result.join(selected,on='pair_id',how='anti').equals(source.join(selected,on='pair_id',how='anti'))
        output=OUT/('submission_'+candidate+'.csv'); result.write_csv(output)
        validation=validate_submission(output,ROOT/'data/evaluation_pairs.csv',PACKAGE/'data/hands.parquet',PACKAGE/'data/seats.parquet',ROOT/'data/sample_submission.csv').to_dict()
        write(OUT/(candidate+'-validation.json'),validation); assert validation['valid']
        changed=int(result.select(pl.any_horizontal(*[pl.col(c)!=source[c] for c in EVIDENCE_COLUMNS]).sum()).item())
        entry={'candidate':candidate,'file':output.name,'sha256':sha(output),'validation':validation,'changed_evidence_lists':changed,'risk_behavior_exact':True,'outside_selection_exact':True,'eligible':True,'submitted':False}
        entries.append(entry); print(json.dumps({'candidate':candidate,'changed_evidence_lists':changed,'valid':True}),flush=True)
    assert len({r['sha256'] for r in entries})==5
    write(OUT/'ready-audit.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'plan.json'),'fit_sha256':sha(fitpath),'admission_sha256':sha(RESEARCH/'evidence-panel/admission.json'),'entries':entries,'twenty_incumbent_batch_controls_exact':True,'no_submissions_attempted':True})

if __name__=='__main__': main()
