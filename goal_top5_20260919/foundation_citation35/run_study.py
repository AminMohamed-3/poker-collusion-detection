import os,sys,json,hashlib,argparse,time,gc,site
from pathlib import Path
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1','POLARS_MAX_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
R35=Path(r'C:/Data/Projects/Money Making/work/poker/goal_top5_20260919/tabpfn35');sys.path.insert(0,str(R35));import runtime35
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score
ROOT=Path(r'C:/Data/Projects/Money Making/work/poker');CV=ROOT/'Claude Version';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(CV/'src'));sys.path.insert(0,str(ROOT/'Codex Resume/src'))
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology,align,less_than_five
from evidence_aug.step5_evaluate import half
K=S165.K;SP='soft_play';CI='coordinated_isolation';MP=CV/'experiments/step254_mechanical_tensor_20260917/step254_features.parquet'
CP=R35/'tabpfn-v3.5-fast-20260909.safetensors'
OOF=CV/'experiments/step254_mechanical_tensor_20260917/oof_mechplus.npy';TYPE=CV/'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet';V1=CV/'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet'
BARE=ROOT/'goal_top5_20260919/reproduction/inc_p_PRODbare.npy';LGBRULE=CV/'experiments/step277_generator_rule_20260919/induced_rule_audit.npy'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def hval(p,h):return int.from_bytes(hashlib.sha256(f'{p[0]}|{p[1]}|{h}'.encode()).digest()[:8],'big')
def sample(frame,y,cap=1024,seed=2773):
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);assert len(pos)<=768 and len(pos)<cap
 pids=frame['pair_id'].to_numpy();hids=frame['hand_id'].to_numpy();pairs=sorted(set(pids));need=cap-len(pos);cand=[]
 for p in pairs:
  group=[int(i) for i in neg if pids[i]==p];group.sort(key=lambda i:hval((pids[i],hids[i]),seed))
  cand.extend((rank,hval((pids[i],hids[i]),seed),i) for rank,i in enumerate(group))
 take=[x[2] for x in sorted(cand)[:need]];ix=np.sort(np.r_[pos,take]);assert len(ix)==cap and set(pairs)<=set(pids[ix]);return ix
def prepare():
 assert not (OUT/'context.npz').exists();old=OUT.parent/'foundation_citation';r=json.loads((old/'prepare-receipt.json').read_text());chosen=r['features'];assert len(chosen)==64
 ev=pl.read_csv(ROOT/'data/development_evidence.csv');fam=dict(ev.select('pair_id','behavior_family').unique().iter_rows());D=S165.load();d=D['d'];audit=set(d['pair_id'].unique())
 m=pl.read_parquet(MP).filter(pl.col('pair_id').is_in(list(fam)));m=m.join(ev.select('pair_id','hand_id','evidence_rank'),on=K,how='left',validate='1:1').with_columns(pl.col('pair_id').replace_strict(fam).alias('family'),pl.col('evidence_rank').is_not_null().cast(pl.Int8).alias('target'))
 sp=m.filter(~pl.col('pair_id').is_in(audit)&(pl.col('family')==SP)).sort(K);y=sp['target'].to_numpy();isp=sample(sp,y);yp=y.copy();rng=np.random.default_rng(401);pids=sp['pair_id'].to_numpy()
 for p in sorted(set(pids)):
  z=np.flatnonzero(pids==p);yp[z]=rng.permutation(y[z])
 pi=float(y.mean());pic=float(y[isp].mean());q=d.filter((pl.col('behavior_family')==SP)&(pl.col('oof_rank')<=20)).select(K).sort(K);qm=q.join(m.select(K+chosen),on=K,how='left',validate='1:1').sort(K);assert qm.select(K).equals(q)
 np.savez_compressed(OUT/'context.npz',xfull=sp.select(chosen).to_numpy().astype(np.float32),yfull=y,yperm=yp,train_pair_ids=pids.astype('U'),xcapped=sp[isp].select(chosen).to_numpy().astype(np.float32),ycapped=y[isp],xquery=qm.select(chosen).to_numpy().astype(np.float32),pair_ids=qm['pair_id'].to_numpy().astype('U'),hand_ids=qm['hand_id'].to_numpy().astype('U'),pi_full=pi,pi_capped=pic)
 rec={'features':chosen,'selector_source':str(old/'prepare-receipt.json'),'full_rows':sp.height,'full_positives':int(y.sum()),'full_prevalence':pi,'capped_rows':len(isp),'capped_positives':int(y[isp].sum()),'capped_prevalence':pic,'train_pairs':len(set(pids)),'permute':'within pair seed401','query_rows':qm.height,'query_pairs':qm['pair_id'].n_unique(),'sources':{str(p):sha(p) for p in [MP,CP,OOF,TYPE,V1,BARE,LGBRULE]},'context_sha256':sha(OUT/'context.npz')};assert rec['full_rows']==10438 and rec['full_positives']==408 and rec['train_pairs']==85 and rec['query_rows']==940;dump(OUT/'prepare-receipt.json',rec);print(json.dumps(rec,indent=2))
def correct(p,full,ctx):
 p=np.clip(p,1e-9,1-1e-9);o=p/(1-p)*(full/(1-full))/(ctx/(1-ctx));return o/(1+o)
def infer(replay=False):
 import torch;torch.cuda.set_per_process_memory_fraction(.70)
 from tabpfn import TabPFNClassifier
 z=np.load(OUT/'context.npz');outs={};meta={};torch.set_num_threads(2)
 for name,x,y,corr in [('primary',z['xfull'],z['yfull'],False),('perm',z['xfull'],z['yperm'],False),('capped',z['xcapped'],z['ycapped'],True)]:
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();m=TabPFNClassifier(model_path=str(CP),n_estimators=4,auto_scale_n_estimators=False,device='cuda',fit_mode='low_memory',memory_saving_mode=True,random_state=2773,n_preprocessing_jobs=1);m.fit(x,y);ix=int(np.flatnonzero(m.classes_==1)[0]);p=np.concatenate([m.predict_proba(z['xquery'][i:i+256])[:,ix] for i in range(0,len(z['xquery']),256)]);p=correct(p,float(z['pi_full']),float(z['pi_capped'])) if corr else p;outs[name]=p;meta[name]={'seconds':time.perf_counter()-t,'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'finite':bool(np.isfinite(p).all())};del m;gc.collect()
 path=OUT/('predictions-replay.npz' if replay else 'predictions.npz');np.savez_compressed(path,**outs);dump(OUT/('replay-receipt.json' if replay else 'inference-receipt.json'),{'predictions_sha256':sha(path),'context_sha256':sha(OUT/'context.npz'),'meta':meta});print(json.dumps(meta,indent=2))
 if replay:
  a=np.load(OUT/'predictions.npz');assert all(np.array_equal(a[k],outs[k]) for k in outs);dump(OUT/'replay-exact.json',{'exact':True,'original_sha256':sha(OUT/'predictions.npz'),'replay_sha256':sha(path)})
def evaluate():
 D=S165.load();d=D['d'];top=D['top'];pid=d['pair_id'].to_numpy();hids=d['hand_id'].to_numpy();rel=d['relevance'].to_numpy().astype(int);fams=dict(d.select('pair_id','behavior_family').unique().iter_rows());famrow=np.array([fams[p] for p in pid]);qall=sorted(set(pid[top]));GT={q:np.flatnonzero((pid==q)&top) for q in qall};truth={q:int(rel[pid==q].sum()) for q in qall};ptab=dict(d.select('pair_id','table_id').unique().iter_rows());fqa=np.array([fams[q] for q in qall]);v=pl.read_parquet(V1).sort(K);fr=chronology(d.select(K+['table_id']).with_columns(pl.Series('score',v['S_star'].to_numpy())));t1=align(fr,pl.read_parquet(TYPE));cr=fr['chrono_rank'].to_numpy();PRI=(~(t1&(famrow!=CI))).astype(int);pf=np.zeros(len(pid));pf[D['QIX']]=np.load(BARE)
 import evidence_aug.step254_eval as S254
 h,_=S254.load();oof=h.select(K).with_columns(pl.Series('p',np.load(OOF)[2]));joined=d.select(K).join(oof,on=K,how='left',validate='1:1');assert joined.select(K).equals(d.select(K));pev=joined['p'].to_numpy();sm=famrow==SP;q=np.clip(pev,1e-9,1-1e-9);o=q/(1-q)*10;pe=o/(1+o);event=pf.copy();event[sm]=np.clip(pf[sm],1e-9,1)**.75*pe[sm]**.25
 z=np.load(OUT/'context.npz');pred=np.load(OUT/'predictions.npz');lk={(a,b):i for i,(a,b) in enumerate(zip(z['pair_ids'],z['hand_ids']))};saved=np.load(LGBRULE)
 arms={'event':event,'primary':event.copy(),'perm':event.copy(),'capped':event.copy(),'matched64_lgb':event.copy(),'lightgbm':event.copy()}
 ix=np.flatnonzero(sm & top)
 for name in ['primary','perm','capped']:arms[name][ix]=np.clip(event[ix],1e-9,1)**.7*np.clip(np.array([pred[name][lk[(pid[i],hids[i])]] for i in ix]),1e-9,1)**.3
 matched=np.load(OUT/'matched64_lgb.npy');arms['matched64_lgb'][ix]=np.clip(event[ix],1e-9,1)**.7*np.clip(np.array([matched[lk[(pid[i],hids[i])]] for i in ix]),1e-9,1)**.3
 arms['lightgbm'][ix]=np.clip(event[ix],1e-9,1)**.7*np.clip(saved[ix],1e-9,1)**.3
 def run(p):
  out=[];sels={}
  for qid in qall:
   g=GT[qid];x=np.clip(p[g],0,1);order=np.lexsort((cr[g],PRI[g]));w=np.empty(len(g));w[order]=less_than_five(x[order]);s=g[np.argsort(-(10+x*w+1e-9*x),kind='stable')[:5]];n=0;v=0
   for j,r in enumerate(rel[s],1):
    if r:n+=1;v+=n/j
   out.append(v/min(5,truth[qid]));sels[qid]=tuple(s)
  return np.array(out),sels
 bare,_=run(pf);base,s0=run(event);assert abs(float(bare.mean())-.7337397119341563)<1e-12;rng=np.random.default_rng(27730);pools=sorted(set(ptab.values()));qbp={p:[i for i,q in enumerate(qall) if ptab[q]==p] for p in pools};IDX=[np.concatenate([qbp[pools[i]] for i in rng.choice(len(pools),len(pools),replace=True)]) for _ in range(4000)];res={};vals={}
 from sklearn.metrics import roc_auc_score
 signals={name:np.array([pred[name][lk[(pid[i],hids[i])]] for i in ix]) for name in ['primary','perm','capped']};signals['matched64_lgb']=np.array([matched[lk[(pid[i],hids[i])]] for i in ix]);signals['lightgbm']=saved[ix]
 for name in ['primary','lightgbm','matched64_lgb','perm','capped']:
  v,s=run(arms[name]);vals[name]=v;dl=v-base;bs=np.array([dl[j].mean() for j in IDX]);res[name]={'delta_all':float(dl.mean()),'delta_sp':float(dl[fqa==SP].mean()),'P_gt_0':float((bs>0).mean()),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'halves':[float(dl[np.array([half(ptab[q]) for q in qall])==k].mean()) for k in [0,1]],'lists_changed':sum(s[q]!=s0[q] for q in qall),'sets_changed':sum(set(s[q])!=set(s0[q]) for q in qall),'citation_auc_sp_top20':float(roc_auc_score(rel[ix],signals[name]))}
 inc=vals['lightgbm'];comp={}
 for name in ['primary','matched64_lgb','perm','capped']:
  dl=vals[name]-inc;bs=np.array([dl[j].mean() for j in IDX]);comp[name]={'delta_all':float(dl.mean()),'delta_sp':float(dl[fqa==SP].mean()),'P_gt_0':float((bs>0).mean()),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'halves':[float(dl[np.array([half(ptab[q]) for q in qall])==k].mean()) for k in [0,1]]}
 assert abs(float(base.mean())-.7373662551440329)<1e-12;assert abs(float(res['lightgbm']['delta_all'])-.003539094650205762)<1e-12;res['capped']['gain_over_v3_capped']=res['capped']['delta_all']-.00046296296296296303;dump(OUT/'result.json',{'baseline_bare_map5':float(bare.mean()),'baseline_event_map5':float(base.mean()),'fixed_weights':{'event':.25,'F':10,'citation':.30},'results_vs_event':res,'comparisons_vs_lightgbm':comp,'audit_reused':True,'fresh_replay_exact':json.loads((OUT/'replay-exact.json').read_text())['exact']});print(json.dumps({'results_vs_event':res,'comparisons_vs_lightgbm':comp},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','infer','replay','evaluate']);x=a.parse_args();{'prepare':prepare,'infer':lambda:infer(False),'replay':lambda:infer(True),'evaluate':evaluate}[x.mode]()



