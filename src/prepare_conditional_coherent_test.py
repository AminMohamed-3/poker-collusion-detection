"""Score fixed out-of-pool trees/rankers and package matching TabICL controls."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from pathlib import Path
import json
import shutil
import tarfile
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,matrix,aggregate,thin,TRAIN_VIEWS,MODEL_PARAMS,LABELS
from tail_pair_experiment import tails
from run_conditional_policy_screen import hand_features
from evaluate_conditional_action_screen import extra_aggregate,metrics
from evaluate_conditional_confirmation import behavior_metrics
from rank_evidence import context,family_columns
from sequence_evidence_experiment import FEATURES as SEQ
from equity_evidence_features import FEATURES as EQ
from relative_evidence_experiment import records,summarize
from evidence_capacity_experiment import compare

OUT=ROOT/'reports/conditional-coherent-history'


def main():
    plan=json.loads((OUT/'plan.json').read_text()); cacheid=json.loads((OUT/'cache-identity.json').read_text())
    identity={'plan_sha256':sha(OUT/'plan.json'),'cache_identity_sha256':sha(OUT/'cache-identity.json'),'source_sha256':sha(__file__)}
    if (OUT/'prepare-identity.json').exists(): assert json.loads((OUT/'prepare-identity.json').read_text())==identity
    else: (OUT/'prepare-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    allraw=pl.read_parquet(sorted((ROOT/'outputs_v2/hands_development').glob('batch_*.parquet')))
    names=[c for c in allraw.columns if c not in ['pair_id','hand_id','table_id']]; events=[c for c in names if c.startswith('v2_')]
    labels=pl.read_csv(ROOT/'data/development_labels.csv'); annotations=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id')
    full=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').join(labels,on='pair_id').sort('pair_id')
    metadata=full.select('pair_id','table_id','player_1','player_2','label','behavior_family')
    sequence=pl.read_parquet(ROOT/'reports/sequence-evidence-experiment/development.parquet')
    equity=pl.read_parquet(ROOT/'reports/equity-evidence-experiment/development.parquet')
    raw_by_fraction={}; actions_by_fraction={}
    for fraction in plan['fractions']:
        folder=OUT/str(fraction); receipts=sorted(folder.glob('batch_*.json')); assert len(receipts)==12
        for p in receipts:
            saved=json.loads(p.read_text()); assert saved['identity']==cacheid
            assert all(sha(folder/n)==h for n,h in saved['files'].items())
        raw_by_fraction[fraction]=pl.read_parquet(sorted(folder.glob('*-raw.parquet')))
        actions_by_fraction[fraction]=pl.read_parquet(sorted(folder.glob('*-actions.parquet')))
    pair_rows=[]; evidence_rows=[]; evidence_results=[]
    for fold in plan['folds']:
        name=f'fold{fold}'; dest=OUT/name; dest.mkdir(exist_ok=True)
        source=ROOT/('reports/tabicl-screen' if fold==0 else 'reports/tabicl-confirmation/'+name)
        prior=np.load(source/'arrays.npz',allow_pickle=False); old_tab=np.load(source/'predictions.npz',allow_pickle=False)
        heldkeys=pl.read_csv(source/'held-pairs.csv'); pools=heldkeys['table_id'].unique().to_list()
        pdir=ROOT/('reports/conditional-action-experiment/policy-screen' if fold==0 else 'reports/conditional-action-confirmation/conditional-'+name+'-v1/policy-screen')
        modeldir=ROOT/('reports/conditional-action-experiment/screen-results' if fold==0 else 'reports/conditional-action-confirmation/results/'+name)
        reference=joblib.load(pdir/'held-models.joblib')
        pref=json.loads((pdir/'held.json').read_text()); assert set(pref['scored_pools'])==set(pools) and not set(pref['training_pools'])&set(pools)
        pair=joblib.load(modeldir/'conditional-pair.joblib'); v5=joblib.load(modeldir/'v5_control-pair.joblib')
        erank=joblib.load(modeldir/'conditional-evidence.joblib')
        eq=joblib.load(ROOT/'reports/equity-evidence-experiment/equity.joblib') if fold==0 else joblib.load(modeldir/'equity-evidence.joblib')
        if fold==0:
            rolefeatures=json.loads((ROOT/'reports/tail-pair-experiment/metrics.json').read_text())['base_features']
            training=pl.concat([full.filter(~pl.col('table_id').is_in(pools))]+[
                aggregate(thin(allraw,*k),names).join(labels,on='pair_id').select(full.columns).filter(~pl.col('table_id').is_in(pools)).sort('pair_id') for k in TRAIN_VIEWS])
            rolemodel=lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(training,rolefeatures),np.array([LABELS[f] for f in training['behavior_family']]),sample_weight=np.full(training.height,1/3))
            role={'model':rolemodel,'features':rolefeatures}; joblib.dump(role,dest/'behavior-model.joblib')
        else: role=joblib.load(modeldir/'behavior.joblib')
        chosen=json.loads((source/'features.json').read_text())['features']
        arrays={k:prior[k] for k in ['x_train','y_train','y_held','x_full']}
        arrays['reference_full_tabicl']=old_tab['full']
        for fraction in plan['fractions']:
            history='coherent_'+str(round(fraction*100)); raw=raw_by_fraction[fraction].filter(pl.col('table_id').is_in(pools))
            actions=actions_by_fraction[fraction].filter(pl.col('table_id').is_in(pools))
            x=actions.select(reference['inputs']).to_numpy().astype(np.float32)
            ca=hand_features(actions,reference['classifier'].predict_proba(x,num_threads=2),reference['regressor'].predict(x,num_threads=2))
            caf=[c for c in ca.columns if c.startswith('ca_')]
            ca=raw.select('pair_id','hand_id','table_id').join(ca.drop('table_id'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.col(caf).fill_null(0))
            frame=metadata.filter(pl.col('table_id').is_in(pools)).join(aggregate(raw,names),on=['pair_id','table_id'],how='left',validate='1:1').join(tails(raw,events),on=['pair_id','table_id'],how='left',validate='1:1').join(extra_aggregate(ca,caf),on=['pair_id','table_id'],how='left',validate='1:1').sort('pair_id')
            assert frame['pair_id'].to_list()==heldkeys['pair_id'].to_list()
            y=np.array([LABELS[f] for f in frame['behavior_family']]); np.testing.assert_array_equal(y,prior['y_held'])
            roles=role['model'].classes_[1:][np.argmax(role['model'].predict_proba(matrix(frame,role['features']))[:,1:],axis=1)]
            risks={}
            for candidate,bundle in [('v5',v5),('conditional',pair)]:
                risk=1-bundle['model'].predict_proba(matrix(frame,bundle['features']))[:,0]; risks[candidate]=risk
                pair_rows.append({'fold':fold,'history':history,'candidate':candidate,**metrics(y,risk),**behavior_metrics(y,risk,roles)})
            frame.select('pair_id','table_id','behavior_family','shared_hands').with_columns(pl.Series('v5',risks['v5']),pl.Series('conditional',risks['conditional']),pl.Series('predicted_class',roles)).write_parquet(dest/(history+'-pairs.parquet'))
            arrays['x_'+history]=matrix(frame,chosen); arrays['v5_'+history]=risks['v5']; arrays['conditional_'+history]=risks['conditional']; arrays['roles_'+history]=roles
            inverse={v:k for k,v in LABELS.items()}
            assigned=frame.select('pair_id').with_columns(pl.Series('predicted_behavior',[inverse[int(i)] for i in roles]))
            data=raw.join(labels.filter(pl.col('label')==1).select('pair_id','behavior_family'),on='pair_id').join(sequence,on=['pair_id','hand_id'],validate='1:1').join(equity,on=['pair_id','hand_id'],validate='1:1').join(ca.drop('table_id'),on=['pair_id','hand_id'],validate='1:1')
            full_count=data['pair_id'].n_unique()
            observed=annotations.join(data.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi')
            truth={k[0]:set(g['hand_id']) for k,g in observed.group_by('pair_id')}
            data=family_columns(context(data,events+SEQ+EQ+caf).join(assigned,on='pair_id',validate='m:1'),'predicted_behavior').filter(pl.col('pair_id').is_in(list(truth))).sort('pair_id','hand_id')
            coverage={'target_pairs':full_count,'pairs_with_retained_annotation':len(truth),'retained_annotations':observed.height,'scored_hands':data.height}
            for candidate,bundle in [('equity',eq),('conditional',erank)]:
                score=bundle['model'].predict(matrix(data,bundle['features']),num_threads=2)
                rows=records(data,score,truth,name+'_'+history,candidate); evidence_rows.extend(rows)
                evidence_results.append({'fold':fold,'history':history,'candidate':candidate,**coverage,**summarize(rows)})
                data.select('pair_id','hand_id').with_columns(pl.Series('score',score)).write_parquet(dest/(history+'-'+candidate+'-evidence.parquet'))
            print(json.dumps({'fold':fold,'history':history,'pair_results':pair_rows[-2:],'evidence_results':evidence_results[-2:]}),flush=True)
        np.savez_compressed(dest/'arrays.npz',**arrays)
    pl.DataFrame(evidence_rows).write_csv(OUT/'evidence-pairs.csv')
    comparisons={}
    frame=pl.DataFrame(evidence_rows)
    for fraction in plan['fractions']:
        history='coherent_'+str(round(fraction*100)); relevant=frame.filter(pl.col('experiment').str.ends_with(history))
        pooled=relevant.group_by('pair_id','table_id','family','candidate').agg(pl.col('map5').mean())
        comparison=compare(pooled.to_dicts(),'conditional',baseline='equity')
        deltas=[compare(relevant.filter(pl.col('experiment')==f'fold{i}_'+history).to_dicts(),'conditional',baseline='equity')['delta'] for i in plan['folds']]
        comparisons[history]={'pooled':comparison,'fold_deltas':deltas,'passed':bool(comparison['delta']>=.005 and min(deltas)>=-.005 and min(comparison['by_family_delta'].values())>=-.025)}
    (OUT/'local-metrics.json').write_text(json.dumps({'pair_results':pair_rows,'evidence_results':evidence_results,'evidence_gates':comparisons,'evidence_pass':all(v['passed'] for v in comparisons.values())},indent=2)+'\n')
    package=OUT/'gpu-package'; package.mkdir(exist_ok=True)
    files={}
    for fold in plan['folds']:
        name=f'fold{fold}.npz'; shutil.copy2(OUT/f'fold{fold}/arrays.npz',package/name); files[name]=sha(package/name)
    runner=ROOT/'src/run_coherent_tabicl_control.py'; shutil.copy2(runner,package/runner.name); files[runner.name]=sha(runner)
    previous=json.loads((ROOT/'reports/gpu-objective-experiment/plan.json').read_text())
    gp={k:previous[k] for k in ['checkpoint_repo','checkpoint_file','checkpoint_revision','checkpoint_sha256','common_parameters']}
    gp.update(files=files,folds=plan['folds'],histories=['coherent_65','coherent_40'],source_plan_sha256=sha(OUT/'plan.json'))
    (package/'plan.json').write_text(json.dumps(gp,indent=2)+'\n')
    archive=OUT/'gpu-package.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in package.iterdir(): t.add(p,arcname=p.name)
    receipt={'sha256':sha(archive),'bytes':archive.stat().st_size}
    (OUT/'gpu-package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
