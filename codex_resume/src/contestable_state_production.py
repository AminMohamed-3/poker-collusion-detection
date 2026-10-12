"""Authorized exploratory production build for the frozen 29 contestable-state fields.

This is deliberately separate from the failed selection/confirmation gates.  It
only runs when root supplies the user-request exception authorization file; it
never submits anything.
"""
from __future__ import annotations
import os
for _thread_key in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_thread_key]='2'
os.environ['PYTHONUTF8']='1'
import gc, json, sys, time
sys.stdout.reconfigure(encoding="utf-8")
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import polars as pl

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parent; CV=ROOT/'Claude Version'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(CV/'src'))
from evidence_aug.paths import sha, check_protected, write_json
from validation import validate_submission
from evidence_aug.frame import FAMILIES
from evidence_aug.step65k_r382_candidate import (OUT as R382_OUT, PREV_OUT, BASE, M3, STAGE1, KEYS, TOP, CLF, SEEDS,
    CONDITIONAL, PACKAGE, FS2, CHRONO, S1, FF50, CI, PARAMS_BY_VARIANT, add_chronology, started_at, context,
    family_columns, matrix, fit_variant, split_of, SPLITS, with_rank, stage2_fit, stage2_predict, rank,
    load_sequence, EQ, SEQ, EVIDENCE_COLUMNS, verify_source_cache, PARAMS)
from contestable_state_confirmation_features import build_for_keys
from contestable_state_features import FEATURES as STATE_FEATURES

OUT=HERE/'outputs_contestable_exploratory_20260912'
PROTOCOL=HERE/'research/contestable_exploratory_protocol_20260912.md'
AUTH=OUT/'exploratory-authorization.json'
BEST=R382_OUT/'submission_M3_stage2_r382DTSP.csv'
BEST_SHA='6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b'


def require_authorization():
    auth=json.loads(AUTH.read_text(encoding='utf-8'))
    assert auth.get('user_new_request_exception') is True, 'missing explicit exploratory exception'
    return auth


def write_once(path, value):
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==value, f'changed existing {path.name}'
    else: write_json(path,value)


def development_frame():
    """Exact step65k production frame with full-development fixed inner OOF stage1."""
    fitted=joblib.load(CONDITIONAL/'evidence-model.joblib'); caf=fitted['conditional_features']; features=fitted['features']; assert len(features)==382
    chrono=joblib.load(STAGE1); data,current,_,_,_=load_sequence()
    data=context(data.join(pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet'),on=KEYS,validate='1:1'),EQ)
    data=context(data.join(pl.read_parquet(CONDITIONAL/'policy/development-hand-features.parquet').drop('table_id'),on=KEYS,validate='1:1'),caf)
    assert features==current+EQ+['context_'+x for x in EQ]+caf+['context_'+x for x in caf]
    data=add_chronology(data,started_at('development')).sort('pair_id','hand_id'); f1=chrono['features']
    refit=fit_variant(data,f1,PARAMS)
    np.testing.assert_array_equal(refit.predict(matrix(family_columns(data,'behavior_family'),f1),num_threads=2),chrono['model'].predict(matrix(family_columns(data,'behavior_family'),f1),num_threads=2))
    data=data.with_columns(pl.col('table_id').map_elements(split_of,return_dtype=pl.Int64).alias('split'))
    oof=np.full(data.height,np.nan)
    for split in range(SPLITS):
        ix=np.flatnonzero(data['split'].to_numpy()==split)
        model=fit_variant(data.filter(pl.col('split')!=split),f1,PARAMS)
        oof[ix]=model.predict(matrix(family_columns(data[ix],'behavior_family'),f1),num_threads=2)
    assert np.isfinite(oof).all()
    data=with_rank(data.with_columns(pl.Series('s1',oof)),'s1','s1_rank')
    structural=pl.read_parquet(FS2/'dev-structural-features.parquet'); struct=[c for c in structural.columns if c not in KEYS]
    overlap=sorted(set(struct)&set(data.columns)); data=data.join(structural,on=KEYS,how='left',validate='1:1')
    assert data.height==45129 and data.select(pl.sum_horizontal(pl.col(struct).null_count())).item()==0
    r382=[f for f in features if not f.startswith('family_') and f not in struct+CHRONO+S1]
    print('Production stage1 refit and all five inner OOF folds complete',flush=True)
    return data,chrono,struct,r382,overlap


def fit_candidate(data, struct, r382):
    players=pl.read_csv(ROOT/'data/development_labels.csv',columns=['pair_id','player_1','player_2'])
    requested=data.filter(pl.col('s1_rank')<=TOP).select('pair_id','hand_id').unique().sort('pair_id','hand_id')
    state, accounting=build_for_keys(requested,players); assert state.select(KEYS).equals(requested)
    data=data.join(state,on=KEYS,how='left',validate='1:1')
    assert data.filter(pl.col('s1_rank')<=TOP).select(STATE_FEATURES).null_count().sum_horizontal().item()==0
    r382_bundle=joblib.load(R382_OUT/'stage2-models.joblib'); prev=joblib.load(PREV_OUT/'stage2-models.joblib')
    models={}; receipts=[]
    for family in FAMILIES:
        base_cols=(prev['cols']['chrono3'] if family==CI else r382_bundle['cols']['r382'])
        refit,_,_=stage2_fit(data,base_cols,family,CLF)
        top=data.filter((pl.col('behavior_family')==family)&(pl.col('s1_rank')<=TOP))
        saved=(prev['models'][f'chrono3/{family}'] if family==CI else r382_bundle['models'][f'r382/{family}'])
        np.testing.assert_array_equal(stage2_predict(refit,top,base_cols),stage2_predict(saved,top,base_cols))
    print('All current production hybrid family models reproduced exactly',flush=True)
    for family in FAMILIES:
        base_cols=(prev['cols']['chrono3'] if family==CI else r382_bundle['cols']['r382'])
        cols=base_cols+STATE_FEATURES
        fitted,nq,npos=stage2_fit(data,cols,family,CLF)
        models[family]=fitted
        top=data.filter((pl.col('behavior_family')==family)&(pl.col('s1_rank')<=TOP))
        replay=stage2_predict(fitted,top,cols)
        assert np.isfinite(replay).all()
        receipts.append({'family':family,'columns':cols,'training_pairs':int(nq),'training_listed':int(npos),'seeds':list(SEEDS)})
    bundle={'models':models,'struct':struct,'r382':r382,'top':TOP,'state_features':STATE_FEATURES,
            'stage1_sha256':sha(STAGE1),'base_r382_sha256':sha(R382_OUT/'stage2-models.joblib'),
            'base_ci_sha256':sha(PREV_OUT/'stage2-models.joblib'),'receipts':receipts}
    joblib.dump(bundle,OUT/'stage2-models.joblib'); restored=joblib.load(OUT/'stage2-models.joblib')
    for family in FAMILIES:
        cols=next(row['columns'] for row in receipts if row['family']==family)
        top=data.filter((pl.col('behavior_family')==family)&(pl.col('s1_rank')<=TOP))
        np.testing.assert_array_equal(stage2_predict(models[family],top,cols),stage2_predict(restored['models'][family],top,cols))
    return bundle,accounting


def build_batches(bundle):
    """Mirror step65k evaluation assembly, then replace only hybrid top20 evidence."""
    source=pl.read_csv(BEST,infer_schema=False); assert sha(BEST)==BEST_SHA
    base=pl.read_csv(BASE,infer_schema=False); assert base.select('pair_id','predicted_behavior').equals(source.select('pair_id','predicted_behavior'))
    selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet'); assert selected.height==16060
    old10=pl.read_parquet(ROOT/'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id')
    added=selected.join(old10,on='pair_id',how='anti'); players=pl.read_csv(ROOT/'data/evaluation_pairs.csv',columns=['pair_id','player_1','player_2'])
    r382=joblib.load(R382_OUT/'stage2-models.joblib'); previous=joblib.load(PREV_OUT/'stage2-models.joblib'); chrono=joblib.load(STAGE1)
    structural_receipt=json.loads((FS2/'eval_structural/receipt.json').read_text()); (OUT/'batches').mkdir(exist_ok=True)
    for index,path in enumerate(sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet'))):
        name=path.stem; receipt=OUT/'batches'/(name+'.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text(encoding='utf-8')); assert saved['plan_sha256']==sha(OUT/'plan.json') and all(sha(OUT/'batches'/p)==h for p,h in saved['files'].items()); continue
        raw=pl.read_parquet(path); events=[c for c in raw.columns if c.startswith('v2_')]
        eq=pl.read_parquet(ROOT/'reports/evidence-public-calibration'/(name+'-features.parquet'))
        ca=pl.read_parquet(PACKAGE/'results'/(name+'-evidence.parquet')).drop('table_id')
        policy_receipt=json.loads((PACKAGE/'results'/(name+'.json')).read_text()); assert all(sha(PACKAGE/'results'/p)==h for p,h in policy_receipt['files'].items())
        coverage_receipt=json.loads(((ROOT / "outputs_v13_coverage" / "batches" / (name + ".json"))).read_text()); assert all(sha((ROOT / "outputs_v13_coverage" / "batches" / p))==h for p,h in coverage_receipt['files'].items())
        caf=joblib.load(CONDITIONAL/'evidence-model.joblib')['conditional_features']
        extra=pl.concat([eq.join(ca,on=KEYS,validate='1:1').select(KEYS+SEQ+EQ+caf),pl.read_parquet((ROOT / "outputs_v13_coverage" / "batches" / (name + "-features.parquet"))).join(added,on='pair_id',how='semi').select(KEYS+SEQ+EQ+caf)])
        sfile=FS2/'eval_structural'/(name+'.parquet'); assert sha(sfile)==structural_receipt['files'][sfile.name]; sfeat=pl.read_parquet(sfile)
        frames=[]; pools=sorted(raw['table_id'].unique())
        for offset in range(0,len(pools),5):
            part=raw.filter(pl.col('table_id').is_in(pools[offset:offset+5])).sort('pair_id','hand_id').with_columns(pl.Series('tie',np.random.default_rng(931207+index*100+offset).random(raw.filter(pl.col('table_id').is_in(pools[offset:offset+5])).height))).join(selected,on='pair_id',how='semi')
            if not part.height: continue
            frame=part.join(extra,on=KEYS,validate='1:1').join(source.select('pair_id','predicted_behavior'),on='pair_id',validate='m:1').sort('pair_id','hand_id'); assert frame.height==part.height
            frame=add_chronology(family_columns(context(frame,events+SEQ+EQ+caf),'predicted_behavior'),started_at('evaluation')).sort('pair_id','hand_id')
            s1=chrono['model'].predict(matrix(frame,chrono['features']),num_threads=2)
            frame=with_rank(frame.with_columns(pl.Series('s1',s1)),'s1','s1_rank').join(sfeat,on=KEYS,how='left',validate='1:1').sort('pair_id','hand_id')
            assert frame.select(pl.sum_horizontal(pl.col(bundle['struct']).null_count())).item()==0; frames.append(frame)
        requested=pl.concat(frames).filter(pl.col('s1_rank')<=TOP).select(KEYS).unique().sort(KEYS)
        state,accounting=build_for_keys(requested,players); assert state.select(KEYS).equals(requested)
        incumbent=[]; candidate=[]
        for frame in frames:
            frame=frame.join(state,on=KEYS,how='left',validate='1:1'); top=(frame['s1_rank']<=TOP).to_numpy(); assert frame.filter(pl.Series(top)).select(STATE_FEATURES).null_count().sum_horizontal().item()==0
            fams=frame['predicted_behavior'].to_numpy(); current=np.full(frame.height,np.nan); novel=np.full(frame.height,np.nan)
            for family in FAMILIES:
                mask=top&(fams==family)
                if mask.any():
                    sub=frame.filter(pl.Series(mask)); basecols=(previous['cols']['chrono3'] if family==CI else r382['cols']['r382']); basemodel=(previous['models'][f'chrono3/{family}'] if family==CI else r382['models'][f'r382/{family}'])
                    current[mask]=stage2_predict(basemodel,sub,basecols); novel[mask]=stage2_predict(bundle['models'][family],sub,next(x['columns'] for x in bundle['receipts'] if x['family']==family))
            hybrid=np.where(top,current+10,frame['s1'].to_numpy()-1e6); proposed=np.where(top,novel+10,frame['s1'].to_numpy()-1e6)
            assert np.isfinite(hybrid).all() and np.isfinite(proposed).all()
            incumbent.append(rank(frame,hybrid)); candidate.append(rank(frame,proposed))
        def expand(pieces):
            return pl.concat(pieces).sort('pair_id').select('pair_id',*[pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(c) for i,c in enumerate(EVIDENCE_COLUMNS)])
        base_lists, new_lists=expand(incumbent),expand(candidate)
        assert base_lists.equals(source.join(base_lists.select('pair_id'),on='pair_id',how='semi').select(base_lists.columns).sort('pair_id'))
        saved_r382=pl.read_parquet(R382_OUT/'batches'/(name+'-r382.parquet')).select(base_lists.columns).sort('pair_id'); assert base_lists.equals(saved_r382)
        out=OUT/'batches'/(name+'-contestable.parquet'); new_lists.write_parquet(out)
        write_json(receipt,{'plan_sha256':sha(OUT/'plan.json'),'files':{out.name:sha(out)},'selected_pairs':base_lists.height,'top20_pair_hands':requested.height,'incumbent_hybrid_exact':True,'saved_r382_lists_exact':True,'state_accounting':accounting,'all_raw_public_actions_accounted_for_requested_hands':True})
        print(json.dumps({'batch':name,'selected_pairs':base_lists.height,'top20_pair_hands':requested.height,'hybrid_replay_exact':True}),flush=True)
        del raw,frames,state; gc.collect()
def source_manifest():
    """Hashes all immutable production, raw-data, and cached feature receipts used by this build."""
    fixed=[Path(__file__),HERE/'src/contestable_state_features.py',HERE/'src/contestable_state_confirmation_features.py',
           PROTOCOL,AUTH,BEST,STAGE1,R382_OUT/'stage2-models.joblib',PREV_OUT/'stage2-models.joblib',
           ROOT/'reports/replay-v13/selected-pairs.parquet',ROOT/'data/evaluation_pairs.csv',
           ROOT/'data/development_labels.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/actions.parquet',
           CONDITIONAL/'evidence-model.joblib',CONDITIONAL/'policy/development-hand-features.parquet',
           FS2/'dev-structural-features.parquet',FS2/'eval_structural/receipt.json',
           CV/'src/evidence_aug/step65k_r382_candidate.py',CV/'src/evidence_aug/step65f_stage2_candidate.py',
           CV/'src/evidence_aug/step65i_r382_stage2_fit.py',ROOT/'src/rank_evidence.py',
           ROOT/'data/development_evidence.csv',ROOT/'data/sample_submission.csv',BASE,
           ROOT/'reports/sequence-evidence-experiment/development.parquet',
           ROOT/'reports/equity-evidence-experiment/development.parquet',
           ROOT/'reports/evidence-public-calibration/selected-pairs.parquet',
           ROOT/'outputs_v3/fit.json',ROOT/'outputs_v3/rank_batches/manifest.json']
    fixed += [Path(m.__file__).resolve() for m in list(sys.modules.values())
              if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT.resolve()))
              and Path(m.__file__).suffix=='.py' and 'site-packages' not in str(m.__file__)]
    fixed += sorted((ROOT/'reports/evidence-public-calibration').glob('batch_*-features.parquet'))
    fixed += sorted((R382_OUT/'batches').glob('batch_*-r382.parquet'))
    receipts=list((PACKAGE/'results').glob('batch_*.json'))+list((ROOT/'outputs_v13_coverage/batches').glob('batch_*.json'))
    return {'fixed':{str(p):sha(p) for p in fixed},'package_receipts':{str(p):sha(p) for p in receipts}}
def main():
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); auth=require_authorization(); protected=check_protected(); assert sha(BEST)==BEST_SHA
    identity={'purpose':'Authorized one-slot exploratory contestable-state production build; earlier strict confirmation failed and is not represented as passing.',
      'authorization_sha256':sha(AUTH),'protocol_sha256':sha(PROTOCOL),'source_m3_sha256':BEST_SHA,'stage1_sha256':sha(STAGE1),'code_sha256':sha(Path(__file__)),'extractor_sha256':sha(HERE / 'src' / 'contestable_state_features.py'),'wrapper_sha256':sha(HERE / 'src' / 'contestable_state_confirmation_features.py'),'r382_models_sha256':sha(R382_OUT / 'stage2-models.joblib'),'ci_models_sha256':sha(PREV_OUT / 'stage2-models.joblib'),'selection_sha256':sha(ROOT / 'reports' / 'replay-v13' / 'selected-pairs.parquet'),'evaluation_roster_sha256':sha(ROOT / 'data' / 'evaluation_pairs.csv'),
      'source_manifest':source_manifest(),'state_features':STATE_FEATURES,'candidate':'complete-current production hybrid plus all 29 fixed contestable-state fields for every family','no_submission_actions':True,'protected':protected}
    write_once(OUT/'plan.json',identity); write_once(OUT/'prediction.json',{'plan_sha256':sha(OUT/'plan.json'),'exploratory':True,'earlier_strict_confirmation':'failed','submission_not_attempted':True,'internal_composite_interval':[-.000028,.003830],'predeclared_descriptive_public_delta_band':[-.006,.010],'base_public_score':.89447,'note':'Planning range only; not a statistical confidence interval and not a passed gate.'})
    verify_source_cache('development'); verify_source_cache('evaluation')
    model_path=OUT/'stage2-models.joblib'; fit_path=OUT/'fit.json'
    if model_path.exists() or fit_path.exists():
        assert model_path.exists() and fit_path.exists(),'orphan model or fit receipt'
        fit=json.loads(fit_path.read_text(encoding='utf-8')); assert fit['plan_sha256']==sha(OUT/'plan.json') and fit['models_sha256']==sha(model_path),'invalid model resume chain'
    else:
        data,chrono,struct,r382,overlap=development_frame(); bundle,accounting=fit_candidate(data,struct,r382)
        write_json(fit_path,{'plan_sha256':sha(OUT/'plan.json'),'models_sha256':sha(model_path),'state_accounting':accounting,'struct_overlap':overlap,'reload_exact':True}); del data;gc.collect()
    bundle=joblib.load(model_path); build_batches(bundle)
    expected_batches=sorted((ROOT/'outputs_v2/hands_evaluation').glob('batch_*.parquet')); outputs=sorted((OUT/'batches').glob('batch_*-contestable.parquet'))
    assert len(expected_batches)==20 and [p.stem.replace('-contestable','') for p in outputs]==[p.stem for p in expected_batches],'missing or unexpected batch outputs'
    batch_receipts=[json.loads((OUT/'batches'/(p.stem+'.json')).read_text(encoding='utf-8')) for p in expected_batches]
    assert all(r['plan_sha256']==sha(OUT/'plan.json') and r['incumbent_hybrid_exact'] and r['saved_r382_lists_exact'] for r in batch_receipts) and sum(r['selected_pairs'] for r in batch_receipts)==16060
    source=pl.read_csv(BEST,infer_schema=False); replacements=pl.read_parquet(outputs).sort('pair_id')
    selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet').select('pair_id').sort('pair_id')
    assert replacements.height==16060 and replacements.select('pair_id').n_unique()==16060 and replacements.select('pair_id').equals(selected)
    result=pl.concat([source.join(selected,on='pair_id',how='anti'),source.join(selected,on='pair_id',how='semi').drop(EVIDENCE_COLUMNS).join(replacements,on='pair_id',validate='1:1')]).select(source.columns)
    result=source.select('pair_id').join(result,on='pair_id',validate='1:1',maintain_order='left')
    assert result.select('pair_id','risk_score','predicted_behavior').equals(source.select('pair_id','risk_score','predicted_behavior'))
    assert result.join(selected,on='pair_id',how='anti').equals(source.join(selected,on='pair_id',how='anti'))
    assert result.select('pair_id').equals(source.select('pair_id')) and check_protected()==protected and sha(BEST)==BEST_SHA
    output=OUT/'submission_contestable_exploratory.csv';result.write_csv(output)
    validation=validate_submission(output,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv').to_dict()
    assert validation['valid'],validation; write_json(OUT/'validation.json',validation)
    write_json(OUT/'receipt.json',{'plan_sha256':sha(OUT/'plan.json'),'fit_sha256':sha(OUT/'fit.json'),'output_sha256':sha(output),'validation_sha256':sha(OUT/'validation.json'),'risk_behavior_exact':True,'no_submission_actions':True,'protected_after':check_protected(),'seconds':time.perf_counter()-start})
if __name__=='__main__':main()







