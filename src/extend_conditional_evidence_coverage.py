"""Apply the frozen conditional ranker to newly prioritized pairs, without fitting."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import gc
import json
import time
import joblib
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix
from conditional_inference_worker import extract_chunk
from run_conditional_policy_screen import hand_features
from rank_evidence import context,family_columns,EVIDENCE_COLUMNS
from sequence_evidence_fast import extract
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ,_canonical_equity
from evidence_public_calibration import equity_frame,rank
from validation import validate_submission

OUT=ROOT/'outputs_v13_coverage'
OLD=ROOT/'outputs_v11_conditional'
PACKAGE=OLD/'inference-package'
SOURCE=ROOT/'outputs_v12_behavior/submission_conditional_behavior.csv'


def assert_ranking_matches(ranking,source):
    comparison=ranking.join(source.select('pair_id',pl.concat_list(EVIDENCE_COLUMNS).alias('expected')),on='pair_id',validate='1:1')
    assert comparison.height==ranking.height
    assert all(actual==[h for h in expected if h!='NO_EVIDENCE'] for actual,expected in comparison.select('hands','expected').iter_rows())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--batches',nargs='+',type=int,default=list(range(20))); args=parser.parse_args()
    OUT.mkdir(exist_ok=True); cache=OUT/'batches'; cache.mkdir(exist_ok=True)
    ready=json.loads((SOURCE.parent/'ready-audit.json').read_text()); assert sha(SOURCE)==ready['sha256']
    source=pl.read_csv(SOURCE,infer_schema=False)
    original=pl.read_parquet(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id')
    top=source.with_columns(pl.col('risk_score').cast(pl.Float64)).sort(['risk_score','pair_id'],descending=[True,False]).head(10000).select('pair_id')
    added=top.join(original,on='pair_id',how='anti').sort('pair_id'); union=pl.concat([original,added]).sort('pair_id')
    assert original.height==10000 and added.height==6060 and union.height==16060
    audit=ROOT/'reports/conditional-reserved-audit/results/summary.json'; assert json.loads(audit.read_text())['all_passed']
    package_plan=json.loads((PACKAGE/'config.json').read_text())
    used=['data/hands.parquet','data/seats.parquet','data/actions.parquet','data/evaluation_pairs.csv','style-counts.parquet','final-models.joblib']
    assert all(sha(PACKAGE/p)==package_plan['files'][p] for p in used)
    code=['extend_conditional_evidence_coverage.py','conditional_inference_worker.py','conditional_action_features.py','run_conditional_policy_screen.py','sequence_evidence_fast.py','equity_evidence_features.py','evidence_public_calibration.py']
    identity={'source_submission_sha256':sha(SOURCE),'source_ready_sha256':sha(SOURCE.parent/'ready-audit.json'),
        'reserved_audit_sha256':sha(audit),'package_config_sha256':sha(PACKAGE/'config.json'),
        'model_sha256':sha(OLD/'evidence-model.joblib'),'code':{p:sha(ROOT/'src'/p) for p in code},
        'selection':'Union of unchanged original v6 top 10000 and frozen conditional-risk top 10000; risk ties use pair_id for deterministic output only.',
        'added_pairs':added.height,'union_pairs':union.height,
        'rationale':'163 of the conditional top 1000 were outside the original ranker selection. Apply an already validated model to the newly prioritized full histories. No evaluation labels, new fit, weight, threshold search or row-level score probing.',
        'controls':'Reproduce the v6 sequence ranking for every newly processed pair. Recompute one original selected witness pair per batch; require exact sequence/equity/conditional feature parity and v12 ranking parity. Preserve every v12 risk, behavior and non-added evidence cell.',
        'validation_basis':'The fixed conditional ranker was evaluated on every eligible held-out target, including those outside any evaluation selection. This extends inference coverage without selecting a new model.'}
    planpath=OUT/'plan.json'
    if planpath.exists(): assert json.loads(planpath.read_text())==identity
    else:
        planpath.write_text(json.dumps(identity,indent=2)+'\n'); added.write_parquet(OUT/'added-pairs.parquet'); union.write_parquet(OUT/'selected-pairs.parquet')
    policy=joblib.load(PACKAGE/'final-models.joblib'); conditional=joblib.load(OLD/'evidence-model.joblib'); caf=conditional['conditional_features']
    control=joblib.load(ROOT/'outputs_v6/sequence_ranker.joblib'); oldsource=pl.read_csv(ROOT/'outputs_v6/submission_sequence_evidence.csv',infer_schema=False)
    pairs=pl.read_csv(PACKAGE/'data/evaluation_pairs.csv').select('pair_id','player_1','player_2')
    counts={r[0]:np.array(r[1:],dtype=np.int64) for r in pl.read_parquet(PACKAGE/'style-counts.parquet').iter_rows()}
    allhands=pl.read_parquet(PACKAGE/'data/hands.parquet')
    for index in args.batches:
        started=time.perf_counter(); name=f'batch_{index:03}'; receipt=cache/(name+'.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text()); assert saved['plan_sha256']==sha(planpath)
            assert all(sha(cache/p)==h for p,h in saved['files'].items()); continue
        raw=pl.read_parquet(ROOT/'outputs_v2/hands_evaluation'/(name+'.parquet'))
        witness=raw.select('pair_id').unique().join(original,on='pair_id',how='semi').sort('pair_id').head(1); assert witness.height==1
        selected=pl.concat([added,witness]).unique(); keys=raw.select('pair_id','hand_id','table_id').join(selected,on='pair_id',how='semi').join(pairs,on='pair_id',validate='m:1')
        ids=keys.select('hand_id').unique(); hands=allhands.join(ids,on='hand_id',how='semi')
        seats=pl.scan_parquet(PACKAGE/'data/seats.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
        actions=pl.scan_parquet(PACKAGE/'data/actions.parquet').join(ids.lazy(),on='hand_id',how='semi').collect()
        pools=sorted(raw['table_id'].unique()); rankparts=[]; featureparts=[]
        for offset in range(0,len(pools),5):
            group=pools[offset:offset+5]; part=raw.filter(pl.col('table_id').is_in(group)).sort('pair_id','hand_id')
            part=part.with_columns(pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(part.height))).join(selected,on='pair_id',how='semi')
            if not part.height: continue
            pk=keys.filter(pl.col('table_id').is_in(group)); ph=hands.filter(pl.col('table_id').is_in(group)); ids=ph.select('hand_id')
            ps=seats.join(ids,on='hand_id',how='semi'); pa=actions.join(ids,on='hand_id',how='semi')
            seq=extract(pk,ph,ps,pa); eq=equity_frame(pk,ph,ps,pa)
            action=extract_chunk(pk,ph,ps,pa,counts); x=action.select(policy['inputs']).to_numpy().astype(np.float32)
            ca=hand_features(action,policy['classifier'].predict_proba(x,num_threads=2),policy['regressor'].predict(x,num_threads=2))
            ca=pk.select('pair_id','hand_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0))
            extra=seq.join(eq,on=['pair_id','hand_id'],validate='1:1').join(ca,on=['pair_id','hand_id'],validate='1:1'); featureparts.append(extra)
            frame=part.join(extra,on=['pair_id','hand_id'],validate='1:1').join(oldsource.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id')
            frame=context(frame,[c for c in raw.columns if c.startswith('v2_')]+SEQ+EQ+caf)
            oldranking=rank(frame,control['model'].predict(matrix(family_columns(frame,'predicted_behavior'),control['features']),num_threads=2))
            assert_ranking_matches(oldranking,oldsource)
            frame=frame.drop('predicted_behavior').join(source.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1',maintain_order='left')
            scores=conditional['model'].predict(matrix(family_columns(frame,'predicted_behavior'),conditional['features']),num_threads=2); assert np.isfinite(scores).all()
            rankparts.append(rank(frame,scores))
        features=pl.concat(featureparts).sort('pair_id','hand_id'); rankings=pl.concat(rankparts).sort('pair_id')
        measured=features.join(witness,on='pair_id',how='semi')
        oldfeatures=pl.read_parquet(ROOT/'reports/evidence-public-calibration'/(name+'-features.parquet')).join(witness,on='pair_id',how='semi').sort('pair_id','hand_id')
        oldca=pl.read_parquet(PACKAGE/'results'/(name+'-evidence.parquet')).join(witness,on='pair_id',how='semi').sort('pair_id','hand_id')
        assert measured.select(['pair_id','hand_id']+SEQ+EQ).equals(oldfeatures.select(['pair_id','hand_id']+SEQ+EQ)),'Witness sequence/equity features changed'
        assert measured.select(['pair_id','hand_id']+caf).equals(oldca.select(['pair_id','hand_id']+caf)),'Witness conditional features changed'
        assert_ranking_matches(rankings.join(witness,on='pair_id',how='semi'),source)
        fpath=cache/(name+'-features.parquet'); rpath=cache/(name+'-rank.parquet')
        features.write_parquet(fpath); rankings.join(added,on='pair_id',how='semi').write_parquet(rpath)
        saved={'plan_sha256':sha(planpath),'rows':features.height,'new_pairs':rankings.height-1,'witness_pair':witness['pair_id'][0],
            'v6_rankings_exact':True,'witness_features_and_v12_ranking_exact':True,'seconds':time.perf_counter()-started,'files':{p.name:sha(p) for p in [fpath,rpath]}}
        receipt.write_text(json.dumps(saved,indent=2)+'\n'); print(json.dumps({'batch':index,**saved}),flush=True)
        del raw,keys,hands,seats,actions,part,frame,features,rankings,featureparts,rankparts; _canonical_equity.cache_clear(); gc.collect()
    receipts=sorted(cache.glob('batch_*.json'))
    if len(receipts)!=20: print(json.dumps({'complete_batches':len(receipts),'waiting':20-len(receipts)}),flush=True); return
    rankings=pl.read_parquet(sorted(cache.glob('batch_*-rank.parquet'))); assert rankings.height==rankings['pair_id'].n_unique()==added.height
    expanded=rankings.select('pair_id',*[pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(c+'_new') for i,c in enumerate(EVIDENCE_COLUMNS)])
    result=source.join(expanded,on='pair_id',how='left',validate='1:1',maintain_order='left').with_columns(*[pl.coalesce(c+'_new',c).alias(c) for c in EVIDENCE_COLUMNS]).select(source.columns)
    assert result.select('pair_id','risk_score','predicted_behavior').equals(source.select('pair_id','risk_score','predicted_behavior'))
    assert result.join(added,on='pair_id',how='anti').equals(source.join(added,on='pair_id',how='anti'))
    path=OUT/'submission_conditional_coverage.csv'
    if path.exists(): assert pl.read_csv(path,infer_schema=False).equals(result)
    else: result.write_csv(path)
    validation=validate_submission(path,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv','evaluation'); assert validation.valid,validation.to_dict()
    (OUT/'validation.json').write_text(json.dumps(validation.to_dict(),indent=2)+'\n')
    assert sha(SOURCE)==ready['sha256']
    audit={'created_at':datetime.now(timezone.utc).isoformat(),'file':path.name,'sha256':sha(path),'plan_sha256':sha(planpath),'validation':validation.to_dict(),'validation_sha256':sha(OUT/'validation.json'),
        'all_twenty_controls_exact':True,'v12_risk_behavior_and_non_added_evidence_unchanged':True,'added_pairs':added.height,'union_pairs':union.height,
        'changed_evidence_rows':sum(a!=b for a,b in zip(result.select(EVIDENCE_COLUMNS).rows(),source.select(EVIDENCE_COLUMNS).rows())),'public_status':'Unsubmitted, evidence coverage only; compare against v12 behavior candidate.'}
    (OUT/'ready-audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit),flush=True)

if __name__=='__main__': main()
