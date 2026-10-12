"""Frozen prediction composition and one developmental quality evaluation; never trains/uploads."""
import os, sys, json, hashlib, argparse, ast
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent; CR=HERE.parent.parent; ROOT=CR.parent; CV=ROOT/'Claude Version'
sys.path.insert(0,str(CV/'src'));sys.path.insert(0,str(CR/'src'))
from evidence_aug.step104_C_typed_decoder_build import chronology, align, decode, less_than_five
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step5_evaluate import half
K=['pair_id','hand_id']; SP='soft_play'; CI='coordinated_isolation'; ARMS=['plain_mlp','tabm']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path,value):
    assert not path.exists(),str(path)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def verify_inputs():
    f=json.loads((HERE/'evaluation-freeze.json').read_text())
    for p,h in f['sources'].items(): assert sha(p)==h,p
    return f

def decode_ci2(frame,typed,gamma=2.):
    score=frame['score'].to_numpy(); out=score.copy()
    ids=frame['pair_id'].to_numpy(); cr=frame['chrono_rank'].to_numpy(); fam=frame['behavior_family'].to_numpy()
    ix=np.flatnonzero(score>5); _,first=np.unique(ids[ix],return_index=True)
    for g in np.split(ix,first[1:]):
        p=np.clip(score[g]-10,0,1); order=np.lexsort((cr[g],~typed[g])); q=np.empty(len(g))
        q[order]=less_than_five(p[order]); exponent=gamma if fam[g[0]]==CI else 1.
        out[g]=10+p*(q**exponent)+1e-9*p
    return out

def reference():
    verify_inputs(); assert not (HERE/'reference-receipt.json').exists()
    old=pl.read_parquet(CR/'experiments/step277_composition_check_20260919/predictions.parquet').sort(K)
    fr=chronology(old.with_columns(pl.col('current_plus_rule_predecode').alias('score')))
    assert fr.select(K).equals(old.select(K)) and fr.height==19325
    fam=fr['behavior_family'].to_numpy(); typed=align(fr,pl.read_parquet(CV/'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))&(fam!=CI)
    assert np.array_equal(decode(fr,typed),old['current_plus_rule'].to_numpy())
    assert np.array_equal(decode_ci2(fr,typed,1.),old['current_plus_rule'].to_numpy())
    text=(CV/'src/evidence_aug/step276_G_ci_gamma_build.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='decode_gamma')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_step276_helper','exec'),ns)
    current=decode_ci2(fr,typed); official=ns['decode_gamma'](fr,typed,fam==CI,2.)
    assert np.array_equal(current,official)
    m=old['is_top20'].to_numpy()&(fam==SP)
    tree=np.load(CV/'experiments/step277_generator_rule_20260919/induced_rule_audit.npy')
    replay=old['current_reference_predecode'].to_numpy().copy()
    replay[m]=10+np.clip(replay[m]-10,1e-9,1.)**.70*np.clip(tree[m],1e-9,1.)**.30
    assert np.array_equal(replay,old['current_plus_rule_predecode'].to_numpy())
    assert np.array_equal(decode_ci2(fr.with_columns(pl.Series('score',replay)),typed),current)
    result=fr.with_columns(pl.Series('typed',typed),pl.Series('current',current))
    path=HERE/'reference.parquet';assert not path.exists();result.write_parquet(path)
    write(HERE/'reference-receipt.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(path),'evaluation_freeze_sha256':sha(HERE/'evaluation-freeze.json'),'controls':{'old_decoder_bit_exact':True,'gamma1_bit_exact':True,'gamma2_official_helper_bit_exact':True,'original_tree_source_replay_bit_exact':True},'rows':len(current),'queries':162,'pools':104,'top20_SP_rows':int(m.sum()),'quality_reads':0,'fits':0})
    print('Reference and neutral controls prepared; no quality labels read.')

def compose():
    verify_inputs(); assert not (HERE/'predictions-frozen.json').exists()
    ref=json.loads((HERE/'reference-receipt.json').read_text());assert ref['sha256']==sha(HERE/'reference.parquet')
    assert ref['evaluation_freeze_sha256']==sha(HERE/'evaluation-freeze.json')
    run=json.loads((HERE/'run/receipt.json').read_text()); assert run['status']=='complete'
    lock=json.loads((HERE/'run-freeze.json').read_text())
    context_receipt=json.loads((HERE/'context-receipt.json').read_text())
    assert lock['context_sha256']==context_receipt['context_sha256']==sha(HERE/'context.npz')
    assert context_receipt['audit_keys_sha256']==sha(HERE/'audit-keys.parquet')
    assert all(run['input_file_hashes'][k]==lock[k] for k in ['config_sha256','context_sha256','train_script_sha256'])
    fr=pl.read_parquet(HERE/'reference.parquet'); typed=fr['typed'].to_numpy(); current=fr['current'].to_numpy()
    m=fr['is_top20'].to_numpy()&(fr['behavior_family'].to_numpy()==SP)
    keys=pl.read_parquet(HERE/'audit-keys.parquet').select(K)
    assert keys.equals(fr.filter(pl.Series(m)).select(K)) and len(keys)==940
    result=fr.select(K+['table_id','behavior_family','is_top20','current']).with_columns(pl.col('current'))
    result=result.with_columns(fr['current_plus_rule_predecode'].alias('current_predecode'))
    hashes={}
    for arm in ARMS:
        r=json.loads((HERE/f'run/{arm}/arm-receipt.json').read_text()); f=HERE/f'run/{arm}/audit_probabilities.npy'
        assert r['exact_saved_fresh_prediction_replay'] and r['audit_prediction_sha256']==sha(f)
        assert r['gpu_peak_reserved_bytes']<2*1024**3
        p=np.load(f,allow_pickle=False);assert p.shape==(940,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        raw=fr['current_plus_rule_predecode'].to_numpy().copy(); event=fr['current_reference_predecode'].to_numpy()
        raw[m]=10+np.clip(event[m]-10,1e-9,1.)**.70*np.clip(p.astype(np.float64),1e-9,1.)**.30
        decoded=decode_ci2(fr.with_columns(pl.Series('score',raw)),typed)
        assert np.array_equal(decoded[~m],current[~m]) and np.array_equal(decoded>5,current>5)
        result=result.with_columns(pl.Series(arm,decoded),pl.Series(arm+'_predecode',raw))
        for name in ['audit_probabilities.npy','final_state_dict.pt','inner_best_state_dict.pt','arm-receipt.json']:
            path=HERE/f'run/{arm}/{name}';hashes[str(path)]=sha(path)
    path=HERE/'predictions.parquet'; assert not path.exists();result.write_parquet(path)
    write(HERE/'predictions-frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(path),'reference_sha256':ref['sha256'],'evaluation_freeze_sha256':sha(HERE/'evaluation-freeze.json'),'fit_artifacts':hashes,'non_SP_unchanged':True,'outside_top20_unchanged':True,'candidate_membership_unchanged':True,'quality_reads':0})
    print('Both fixed candidates composed and frozen; no quality labels read.')

def compare(delta,qids,pools):
    tabs=sorted(set(pools[q] for q in qids)); ss=np.array([sum(delta[i] for i,q in enumerate(qids) if pools[q]==t) for t in tabs]); nn=np.array([sum(pools[q]==t for q in qids) for t in tabs])
    ix=np.random.default_rng(19143).integers(0,len(tabs),(5000,len(tabs)));boot=ss[ix].sum(1)/nn[ix].sum(1)
    return {'mean_delta':float(delta.mean()),'CI95_pool_bootstrap':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_positive_fraction':float((boot>0).mean()),'halves':{str(h):float(np.mean([delta[i] for i,q in enumerate(qids) if half(pools[q])==h])) for h in [0,1]},'up_down_unchanged':[int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum())]}

def evaluate():
    verify_inputs(); f=json.loads((HERE/'predictions-frozen.json').read_text()); assert sha(HERE/'predictions.parquet')==f['sha256'] and sha(HERE/'evaluation-freeze.json')==f['evaluation_freeze_sha256']
    for p,h in f['fit_artifacts'].items(): assert sha(p)==h,p
    write(HERE/'evaluation-started.json',{'utc':datetime.now(timezone.utc).isoformat(),'single_fixed_quality_read':True})
    d=pl.read_parquet(HERE/'predictions.parquet');truth=pl.read_parquet(CV/'experiments/label_process_20260911/audit-oof.parquet',columns=K+['relevance']).sort(K)
    assert d.select(K).equals(truth.select(K));d=d.with_columns(truth['relevance'])
    truths={k[0]:set(g['hand_id']) for k,g in d.filter(pl.col('relevance')==1).group_by('pair_id')}
    fams=dict(d.select('pair_id','behavior_family').unique().iter_rows());pools=dict(d.select('pair_id','table_id').unique().iter_rows());qids=sorted(truths)
    assert len(qids)==162 and len(set(pools.values()))==104
    scores={};rank={}
    for name in ['current']+ARMS:
        _,scores[name]=summarize(d.select(K+['table_id',pl.col(name).alias('score')]),truths,fams)
        num=den=0.; aps=[]; both_classes=0; zero_positive=0
        for _,g in d.filter(pl.col('is_top20')&(pl.col('behavior_family')==SP)).group_by('pair_id'):
            y=g['relevance'].to_numpy();p=g[name+'_predecode'].to_numpy();pos=p[y==1];neg=p[y==0]
            num+=float((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum());den+=len(pos)*len(neg)
            both_classes += int(len(pos)>0 and len(neg)>0); zero_positive += int(len(pos)==0)
            aps.append(average_precision_score(y,p) if len(pos)>0 else 0.)
        assert len(aps)==47 and den>0
        rank[name]={'SP_within_pair_AP':float(np.mean(aps)),'SP_weighted_pairwise_AUC':num/den,'SP_queries_with_both_classes':both_classes,'SP_queries_without_listed_top20_hands':zero_positive,'positive_negative_comparisons':int(den)}
    comps={}
    for a,b in [('tabm','current'),('tabm','plain_mlp'),('plain_mlp','current')]:
        comps[a+'-'+b]=compare(np.array([scores[a][q]-scores[b][q] for q in qids]),qids,pools)
    gates={}
    for key in ['tabm-current','tabm-plain_mlp']:
        c=comps[key];gates[key+'_positive']=c['mean_delta']>0;gates[key+'_CI_lower_positive']=c['CI95_pool_bootstrap'][0]>0;gates[key+'_halves_nonnegative']=min(c['halves'].values())>=0
    for metric in ['SP_within_pair_AP','SP_weighted_pairwise_AUC']:gates[metric+'_vs_current']=rank['tabm'][metric]>=rank['current'][metric]
    means={n:float(np.mean([scores[n][q] for q in qids])) for n in scores}
    res={'utc':datetime.now(timezone.utc).isoformat(),'means':means,'comparisons':comps,'ranking_diagnostics':rank,'gate_conditions':gates,'verdict':'DEVELOPMENT_SCREEN_PASS' if all(gates.values()) else 'NOT_ESTABLISHED','reference':'Full current SP277+CI276 pipeline in OOF production form, fixed tree-source replacement; pair components unchanged','quality_evaluations':1,'max_model_fits':4,'audit_pairs':162,'audit_pools':104,'historical_outer_training_pairs':85,'new_submission':False,'limitations':['Audit reused heavily and SP scope/weight previously selected','Known-family routing differs from predicted-family deployment; not a full composite score','G4 production-form OOF proxy uses smaller training folds than deployment','Single seed, no rescue or settings search after audit; developmental evidence only']}
    path=HERE/'per-query.parquet';assert not path.exists();pl.DataFrame({'pair_id':qids,'table_id':[pools[q] for q in qids],**{n:[scores[n][q] for q in qids] for n in scores}}).write_parquet(path)
    write(HERE/'results.json',res)
    c=comps['tabm-current'];report=f"# TabM citation pilot\n\nVerdict: {res['verdict']}. Current {means['current']:.9f}; TabM {means['tabm']:.9f}; plain MLP {means['plain_mlp']:.9f} evidence MAP@5.\n\nTabM minus current {c['mean_delta']:+.9f}, pool interval {c['CI95_pool_bootstrap']}; halves {c['halves']}.\n\nOne reused-audit quality read. Train85 SP pairs/71 pools, audit47 SP pairs/43 pools; 162 full audit queries/104 pools. All historical outer training pairs are unavailable as independent confirmation. Current includes SP277 and CI276. Known-family routing and G4 train-capacity differences prevent a full deployment/composite claim. No submission or rescue.\n"
    assert not (HERE/'REPORT.md').exists();(HERE/'REPORT.md').write_text(report,encoding='utf-8');print(json.dumps(res,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['reference','compose','evaluate']);a=p.parse_args();globals()[a.mode]()
