"""Preparation only by default; owner hash release required for fitting/evaluation."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import sys,json,hashlib,ast,time,importlib.util
from pathlib import Path
import numpy as np
import polars as pl
import joblib
from scipy.stats import rankdata
H=Path(__file__).resolve().parent; P=H.parents[2]; CV=P/'Claude Version'; CR=P/'Codex Resume'
L=H.parent/'private_card_likelihood'; REF=CR/'experiments/tabm_citation_pilot_20260919'
K=['pair_id','hand_id']; AK=['pair_id','direction','hand_id','action_no']; SEED=35935
sys.path.insert(0,str(CV/'src')); sys.path.insert(0,str(CR/'src'))
from evidence_aug.step5_evaluate import half,per_pair_map,pool_bootstrap
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    p=H/name; assert not p.exists(),str(p); p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def helper():
    spec=importlib.util.spec_from_file_location('private_frozen_fit',L/'fit.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def verify(release=False):
    m=json.loads((H/'MANIFEST.json').read_text())
    for p,h in m['sources'].items(): assert sha(P/p)==h,p
    for p,h in m['artifacts'].items(): assert sha(H/p)==h,p
    if release: assert json.loads((H/'RELEASE.json').read_text())['manifest_sha256']==sha(H/'MANIFEST.json')
    return m
def prepare():
    assert not (H/'MANIFEST.json').exists()
    rr=json.loads((REF/'reference-receipt.json').read_text());assert rr['sha256']==sha(REF/'reference.parquet')
    r=pl.read_parquet(REF/'reference.parquet').sort(K)
    # Reconstruct the complete current decoded scores, without reading truth.
    source=CV/'src/evidence_aug/step276_G_ci_gamma_build.py'
    node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='decode_gamma')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_decoder','exec'),ns)
    replay=ns['decode_gamma'](r.with_columns(pl.col('current_plus_rule_predecode').alias('score')),r['typed'].to_numpy(),r['behavior_family'].to_numpy()=='coordinated_isolation',2.)
    assert np.array_equal(replay,r['current'].to_numpy())
    audit=sorted(r['table_id'].unique().to_list()); assert len(audit)==104 and r['pair_id'].n_unique()==162
    candidate=r.filter(pl.col('is_top20')).select(K+['table_id']); assert candidate.height==3240
    assert r.filter(~pl.col('is_top20'))['current'].max()<5
    mem=pl.read_parquet(L/'row_membership.parquet').select(*AK,'table_id','actor_id')
    alltab=sorted(mem['table_id'].unique().to_list());train=sorted(set(alltab)-set(audit));assert len(train)==293
    scored=mem.join(candidate,on=K+['table_id'],how='semi')
    assert scored.select(K).unique().height==3240
    trainmem=mem.filter(pl.col('table_id').is_in(train));heldmem=mem.filter(pl.col('table_id').is_in(audit))
    for col in ['table_id','actor_id','hand_id','pair_id']: assert not set(trainmem[col])&set(heldmem[col]),col
    pl.DataFrame({'table_id':train,'inner_fold':[int(hashlib.sha256(f'private-card-evidence-v1|{t}'.encode()).hexdigest(),16)%2 for t in train]}).write_parquet(H/'training_tables.parquet')
    r.select(K+['table_id','behavior_family','is_top20','current']).write_parquet(H/'reference.parquet')
    candidate.write_parquet(H/'candidate_keys.parquet')
    write('membership.json',{'audit_tables':audit,'train_tables':train,'baseline_replay_exact':True,'rows':r.height,'candidates':3240,'queries':162,'new_quality_reads':0})
    lm=json.loads((L/'MANIFEST.json').read_text()); paths=[P/k for k in lm['sources'] if k.startswith('reports/conditional-action-experiment/action-cache/')]
    deps=[*paths,L/'MANIFEST.json',L/'fit.py',L/'row_membership.parquet',L/'capped_training_keys.parquet',REF/'reference.parquet',REF/'reference-receipt.json',source,CV/'src/evidence_aug/step104_C_typed_decoder_build.py',CV/'src/evidence_aug/step5_evaluate.py',CR/'src/audit_common.py',CV/'experiments/label_process_20260911/audit-oof.parquet']
    write('MANIFEST.json',{'inputs':lm['inputs'],'sources':{str(p.relative_to(P)).replace('\\','/'):sha(p) for p in deps},'artifacts':{n:sha(H/n) for n in ['study.py','PROTOCOL.md','training_tables.parquet','reference.parquet','candidate_keys.parquet','membership.json']},'cache_paths':[str(p.relative_to(P)).replace('\\','/') for p in paths],'fits':5,'runtime_seconds_limit':900,'rank_weights':[9,1]})
    print('Prepared clean104 evidence experiment; zero fits/quality reads.')
def disjoint(a,b):
    assert a.height and b.height
    for c in ['table_id','actor_id','hand_id','pair_id']: assert not set(a[c])&set(b[c]),c
def fit():
    m=verify(True); assert not (H/'FIT_STARTED.json').exists();write('FIT_STARTED.json',{'manifest_sha256':sha(H/'MANIFEST.json')})
    start=time.monotonic(); f=helper();inputs=m['inputs']
    d=pl.read_parquet([P/k for k in m['cache_paths']]).join(pl.read_parquet(L/'row_membership.parquet').select(*AK,'actor_id'),on=AK,validate='1:1')
    tr=d.join(pl.read_parquet(H/'training_tables.parquet'),on='table_id',validate='m:1').join(pl.read_parquet(L/'capped_training_keys.parquet').select(AK),on=AK,how='semi')
    tr=f.weights(tr).with_row_index('_i'); he=d.join(pl.read_parquet(H/'candidate_keys.parquet'),on=K+['table_id'],how='semi');disjoint(tr,he)
    assert np.isfinite(tr.select(inputs).to_numpy()).all() and np.isfinite(he.select(inputs).to_numpy()).all()
    tr.select(*AK,'table_id','actor_id','inner_fold','_w').write_parquet(H/'actual_training_membership.parquet')
    assert set(tr['action_class'])=={0,1,2,3}; pred=np.empty(tr.height);models={};members=[]
    def timed(): assert time.monotonic()-start<m['runtime_seconds_limit']
    for j in [0,1]:
        a=f.weights(tr.filter(pl.col('inner_fold')!=j));b=tr.filter(pl.col('inner_fold')==j);disjoint(a,b);timed()
        q=f.model(True).fit(a.select(inputs).to_numpy(),a['partner_strength'].to_numpy(),sample_weight=a['_w'].to_numpy());pred[b['_i'].to_numpy()]=q.predict(b.select(inputs).to_numpy());models[f'strength_inner{j}']=q
        members.append({'inner':j,'train_tables':sorted(a['table_id'].unique().to_list()),'score_tables':sorted(b['table_id'].unique().to_list()),'train_key_sha256':hashlib.sha256(a.select(AK).write_csv().encode()).hexdigest()})
    X=tr.select(inputs).to_numpy();Z=he.select(inputs).to_numpy();w=tr['_w'].to_numpy();y=tr['action_class'].to_numpy();timed()
    q=f.model(True).fit(X,tr['partner_strength'].to_numpy(),sample_weight=w);models['strength_full']=q
    ts=tr['partner_strength'].to_numpy()-pred;hs=he['partner_strength'].to_numpy()-q.predict(Z)
    timed();p0=f.model().fit(X,y,sample_weight=w);timed();p1=f.model().fit(np.c_[X,ts],y,sample_weight=w);models.update(p0=p0,p1=p1)
    assert np.array_equal(p0.classes_,np.arange(4)) and np.array_equal(p1.classes_,np.arange(4))
    a=p0.predict_proba(Z);b=p1.predict_proba(np.c_[Z,hs]);assert np.isfinite(a).all() and np.isfinite(b).all();timed()
    joblib.dump(models,H/'models.joblib');reloaded=joblib.load(H/'models.joblib');assert np.array_equal(reloaded['p1'].predict_proba(np.c_[Z,hs]),b)
    target=he['action_class'].to_numpy();ix=np.arange(he.height);gain=np.log(np.maximum(b[ix,target],1e-12))-np.log(np.maximum(a[ix,target],1e-12))
    actions=he.select(*AK,'table_id').with_columns(pl.Series('gain',gain));actions.write_parquet(H/'action_predictions.parquet')
    hands=actions.group_by(K+['direction']).agg(pl.mean('gain')).group_by(K).agg(pl.mean('gain')); assert hands.height==3240
    r=pl.read_parquet(H/'reference.parquet').sort(K).join(hands,on=K,how='left',validate='1:1',maintain_order='left');candidate=r['current'].to_numpy().copy()
    for _,g in r.with_row_index('_row').filter(pl.col('is_top20')).group_by('pair_id'):
        assert g.height==20 and g['gain'].null_count()==0
        rb=np.rint(2*rankdata(g['current'].to_numpy(),method='average')).astype(np.int64);rp=np.rint(2*rankdata(g['gain'].to_numpy(),method='average')).astype(np.int64)
        candidate[g['_row'].to_numpy()]=10+(9*rb+rp)/400.
    outside=~r['is_top20'].to_numpy();assert np.array_equal(candidate[outside],r['current'].to_numpy()[outside])
    r.with_columns(pl.Series('candidate',candidate)).write_parquet(H/'predictions.parquet');timed()
    write('FIT_RECEIPT.json',{'manifest_sha256':sha(H/'MANIFEST.json'),'seconds':time.monotonic()-start,'fits':5,'P1_reload_exact':True,'membership':members,'artifacts':{n:sha(H/n) for n in ['models.joblib','action_predictions.parquet','predictions.parquet','actual_training_membership.parquet']},'quality_reads':0})
def evaluate():
    verify(True);rec=json.loads((H/'FIT_RECEIPT.json').read_text());assert rec['manifest_sha256']==sha(H/'MANIFEST.json')
    for n,h in rec['artifacts'].items():assert sha(H/n)==h
    write('EVALUATION_STARTED.json',{'single_quality_read':True});d=pl.read_parquet(H/'predictions.parquet').sort(K)
    truth=pl.read_parquet(CV/'experiments/label_process_20260911/audit-oof.parquet',columns=K+['relevance']).sort(K);assert d.select(K).equals(truth.select(K));t=d.select(K).with_columns(truth['relevance']);truths={k[0]:set(g['hand_id']) for k,g in t.filter(pl.col('relevance')==1).group_by('pair_id')}
    scores={n:per_pair_map(d.select(K+['table_id',pl.col(n).alias('score')]),truths) for n in ['current','candidate']}
    delta={k:scores['candidate'][k]-v for k,v in scores['current'].items()};tables=dict(d.select('pair_id','table_id').unique().iter_rows());means={n:float(np.mean(list(v.values()))) for n,v in scores.items()};halves={str(h):float(np.mean([v for k,v in delta.items() if half(tables[k])==h])) for h in [0,1]}
    write('RESULTS.json',{'means':means,'delta':float(np.mean(list(delta.values()))),'halves':halves,'CI95':pool_bootstrap(delta,tables),'gate':bool(np.mean(list(delta.values()))>0 and min(halves.values())>0),'limitations':['Adaptive reused-development hypothesis; all104 excluded from NEW upstream fits only.','Historical current baseline is reused OOF analogue, not fresh private confirmation.','Known-family baseline routing; no full-composite or public-gain claim.'],'new_quality_reads':1,'submission':False})
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','fit','evaluate'];globals()[sys.argv[1]]()
