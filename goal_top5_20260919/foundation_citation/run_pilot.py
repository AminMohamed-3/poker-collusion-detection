import os,sys,json,hashlib,argparse,time,gc,site
from pathlib import Path
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1','POLARS_MAX_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/Codex Resume/.venv_tabpfn3/Lib/site-packages')
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score
ROOT=Path(r'C:/Data/Projects/Money Making/work/poker');CV=ROOT/'Claude Version';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(CV/'src'));sys.path.insert(0,str(ROOT/'Codex Resume/src'))
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step104_C_typed_decoder_build import chronology,align,less_than_five
from evidence_aug.step5_evaluate import half
K=S165.K;SP='soft_play';CI='coordinated_isolation';MP=CV/'experiments/step254_mechanical_tensor_20260917/step254_features.parquet'
CP=ROOT/'Codex Resume/research/tabpfn3_runtime_20260912/checkpoint/tabpfn-v3-classifier-v3_default.ckpt'
OOF=CV/'experiments/step254_mechanical_tensor_20260917/oof_mechplus.npy';TYPE=CV/'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet';V1=CV/'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet'
BARE=Path(r'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/074de80d-a4a8-4cac-89d6-7754b7d2c154/scratchpad/inc_p_PRODbare.npy');LGBRULE=CV/'experiments/step277_generator_rule_20260919/induced_rule_audit.npy'
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
 assert not (OUT/'context.npz').exists();ev=pl.read_csv(ROOT/'data/development_evidence.csv');fam=dict(ev.select('pair_id','behavior_family').unique().iter_rows());D=S165.load();d=D['d'];audit=set(d['pair_id'].unique());
 m=pl.read_parquet(MP).filter(pl.col('pair_id').is_in(list(fam)));m=m.join(ev.select('pair_id','hand_id','evidence_rank'),on=K,how='left').with_columns(pl.col('pair_id').replace_strict(fam).alias('family'),pl.col('evidence_rank').is_not_null().cast(pl.Int8).alias('target'))
 train=m.filter(~pl.col('pair_id').is_in(audit));sp=train.filter(pl.col('family')==SP).sort(K);other=train.filter(pl.col('family')!=SP).sort(K);cols=[c for c in m.columns if c.startswith(('t1_','t2_','t3_','t4_','t5_'))];assert len(cols)==662
 X=sp.select(cols).to_numpy().astype(np.float32);y=sp['target'].to_numpy();valid=[]
 for j,c in enumerate(cols):
  x=X[:,j];finite=np.isfinite(x);x=np.where(finite,x,np.nanmedian(x[finite]) if finite.any() else 0);sd=x.std()
  if sd==0:continue
  corr=abs(float(np.corrcoef(x,y)[0,1]));ap=float(average_precision_score(y,x));ap2=float(average_precision_score(y,-x));valid.append((c,corr,max(ap,ap2)))
 cr={c:i for i,(c,_,_) in enumerate(sorted(valid,key=lambda z:(-z[1],z[0])))};ar={c:i for i,(c,_,_) in enumerate(sorted(valid,key=lambda z:(-z[2],z[0])))}
 chosen=[c for c,_,_ in sorted(valid,key=lambda z:(cr[z[0]]+ar[z[0]],z[0]))[:64]]
 isp=sample(sp,y);yo=other['target'].to_numpy();io=sample(other,yo);yp=y.copy();rng=np.random.default_rng(401);pids=sp['pair_id'].to_numpy()
 for p in sorted(set(pids)):
  z=np.flatnonzero(pids==p);yp[z]=rng.permutation(y[z])
 pi=float(y.mean());pic=float(y[isp].mean());pio=float(yo.mean());pioc=float(yo[io].mean())
 q=d.filter((pl.col('behavior_family')==SP)&(pl.col('oof_rank')<=20)).select(K);qm=q.join(m.select(K+chosen),on=K,validate='1:1').sort(K)
 np.savez_compressed(OUT/'context.npz',xreal=sp[isp].select(chosen).to_numpy().astype(np.float32),yreal=y[isp],yperm=yp[isp],xwrong=other[io].select(chosen).to_numpy().astype(np.float32),ywrong=yo[io],xquery=qm.select(chosen).to_numpy().astype(np.float32),pair_ids=qm['pair_id'].to_numpy().astype('U'),hand_ids=qm['hand_id'].to_numpy().astype('U'),pi_full=pi,pi_context=pic,pi_wrong_full=pio,pi_wrong_context=pioc)
 receipt={'feature_count':64,'features':chosen,'full_sp_rows':sp.height,'full_sp_positives':int(y.sum()),'context_rows':len(isp),'context_positives':int(y[isp].sum()),'full_sp_prevalence':pi,'context_prevalence':pic,'wrong_full_rows':other.height,'wrong_full_positives':int(yo.sum()),'wrong_context_rows':len(io),'wrong_context_positives':int(yo[io].sum()),'wrong_full_prevalence':pio,'wrong_context_prevalence':pioc,'audit_query_rows':qm.height,'audit_query_pairs':qm['pair_id'].n_unique(),'selector':'training-only mean of ranks: absolute point-biserial correlation and max(AP(x),AP(-x)); ties column name','sampling':'all positives plus negatives evenly by within-pair SHA256 rank; cap1024; seed2773','prior_correction':'odds_corrected = odds_model * odds(full eligible training prevalence) / odds(capped context prevalence)','sources':{str(p):sha(p) for p in [MP,CP,OOF,TYPE,V1,BARE,LGBRULE]},'context_sha256':sha(OUT/'context.npz')};dump(OUT/'prepare-receipt.json',receipt);print(json.dumps(receipt,indent=2))
def correct(p,full,ctx):
 p=np.clip(p,1e-9,1-1e-9);o=p/(1-p)*(full/(1-full))/(ctx/(1-ctx));return o/(1+o)
def infer(replay=False):
 import torch
 from tabpfn import TabPFNClassifier
 z=np.load(OUT/'context.npz');outs={};meta={};torch.set_num_threads(2)
 for name,x,y,pf,pc in [('real',z['xreal'],z['yreal'],z['pi_full'],z['pi_context']),('perm',z['xreal'],z['yperm'],z['pi_full'],z['pi_context']),('wrongfam',z['xwrong'],z['ywrong'],z['pi_wrong_full'],z['pi_wrong_context'])]:
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();m=TabPFNClassifier(model_path=str(CP),n_estimators=4,auto_scale_n_estimators=False,device='cuda:0',fit_mode='low_memory',memory_saving_mode=True,random_state=2773,n_preprocessing_jobs=1,show_progress_bar=False,balance_probabilities=False,average_before_softmax=False);m.fit(x,y);ix=int(np.flatnonzero(m.classes_==1)[0]);p=np.concatenate([m.predict_proba(z['xquery'][i:i+256])[:,ix] for i in range(0,len(z['xquery']),256)]);p=correct(p,float(pf),float(pc));outs[name]=p;meta[name]={'seconds':time.perf_counter()-t,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'finite':bool(np.isfinite(p).all())};del m;gc.collect()
 path=OUT/('predictions-replay.npz' if replay else 'predictions.npz');np.savez_compressed(path,**outs);dump(OUT/('replay-receipt.json' if replay else 'inference-receipt.json'),{'predictions_sha256':sha(path),'context_sha256':sha(OUT/'context.npz'),'meta':meta});print(json.dumps(meta,indent=2))
 if replay:
  a=np.load(OUT/'predictions.npz');assert all(np.array_equal(a[k],outs[k]) for k in outs);dump(OUT/'replay-exact.json',{'exact':True,'original_sha256':sha(OUT/'predictions.npz'),'replay_sha256':sha(path)})
def evaluate():
 D=S165.load();d=D['d'];top=D['top'];pid=d['pair_id'].to_numpy();hids=d['hand_id'].to_numpy();rel=d['relevance'].to_numpy().astype(int);fams=dict(d.select('pair_id','behavior_family').unique().iter_rows());famrow=np.array([fams[p] for p in pid]);qall=sorted(set(pid[top]));GT={q:np.flatnonzero((pid==q)&top) for q in qall};truth={q:int(rel[pid==q].sum()) for q in qall};ptab=dict(d.select('pair_id','table_id').unique().iter_rows());fqa=np.array([fams[q] for q in qall]);v=pl.read_parquet(V1).sort(K);fr=chronology(d.select(K+['table_id']).with_columns(pl.Series('score',v['S_star'].to_numpy())));t1=align(fr,pl.read_parquet(TYPE));cr=fr['chrono_rank'].to_numpy();PRI=(~(t1&(famrow!=CI))).astype(int);pf=np.zeros(len(pid));pf[D['QIX']]=np.load(BARE)
 import evidence_aug.step254_eval as S254
 h,_=S254.load();oof=h.select(K).with_columns(pl.Series('p',np.load(OOF)[2]));joined=d.select(K).join(oof,on=K,how='left',validate='1:1');assert joined.select(K).equals(d.select(K));pev=joined['p'].to_numpy();sm=famrow==SP;q=np.clip(pev,1e-9,1-1e-9);o=q/(1-q)*10;pe=o/(1+o);event=pf.copy();event[sm]=np.clip(pf[sm],1e-9,1)**.75*pe[sm]**.25
 z=np.load(OUT/'context.npz');pred=np.load(OUT/'predictions.npz');lk={(a,b):i for i,(a,b) in enumerate(zip(z['pair_ids'],z['hand_ids']))};saved=np.load(LGBRULE)
 arms={'event':event,'tabpfn':event.copy(),'perm':event.copy(),'wrongfam':event.copy(),'lightgbm':event.copy()}
 ix=np.flatnonzero(sm & top)
 for name in ['tabpfn','perm','wrongfam']:arms[name][ix]=np.clip(event[ix],1e-9,1)**.7*np.clip(np.array([pred['real' if name=='tabpfn' else name][lk[(pid[i],hids[i])]] for i in ix]),1e-9,1)**.3
 arms['lightgbm'][ix]=np.clip(event[ix],1e-9,1)**.7*np.clip(saved[ix],1e-9,1)**.3
 def run(p):
  out=[];sels={}
  for qid in qall:
   g=GT[qid];x=np.clip(p[g],0,1);order=np.lexsort((cr[g],PRI[g]));w=np.empty(len(g));w[order]=less_than_five(x[order]);s=g[np.argsort(-(10+x*w+1e-9*x),kind='stable')[:5]];n=0;v=0
   for j,r in enumerate(rel[s],1):
    if r:n+=1;v+=n/j
   out.append(v/min(5,truth[qid]));sels[qid]=tuple(s)
  return np.array(out),sels
 base,s0=run(event);rng=np.random.default_rng(27730);pools=sorted(set(ptab.values()));qbp={p:[i for i,q in enumerate(qall) if ptab[q]==p] for p in pools};IDX=[np.concatenate([qbp[pools[i]] for i in rng.choice(len(pools),len(pools),replace=True)]) for _ in range(4000)];res={}
 from sklearn.metrics import roc_auc_score
 signals={name:np.array([pred['real' if name=='tabpfn' else name][lk[(pid[i],hids[i])]] for i in ix]) for name in ['tabpfn','perm','wrongfam']};signals['lightgbm']=saved[ix]
 for name in ['tabpfn','lightgbm','perm','wrongfam']:
  v,s=run(arms[name]);dl=v-base;bs=np.array([dl[j].mean() for j in IDX]);res[name]={'delta_all':float(dl.mean()),'delta_sp':float(dl[fqa==SP].mean()),'P_gt_0':float((bs>0).mean()),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'halves':[float(dl[np.array([half(ptab[q]) for q in qall])==k].mean()) for k in [0,1]],'lists_changed':sum(s[q]!=s0[q] for q in qall),'sets_changed':sum(set(s[q])!=set(s0[q]) for q in qall),'citation_auc_sp_top20':float(roc_auc_score(rel[ix],signals[name]))}
 dump(OUT/'result.json',{'baseline_event_map5':float(base.mean()),'fixed_weights':{'event':.25,'F':10,'citation':.30},'results':res,'audit_reused':True,'fresh_replay_exact':json.loads((OUT/'replay-exact.json').read_text())['exact']});print(json.dumps(res,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','infer','replay','evaluate']);x=a.parse_args();{'prepare':prepare,'infer':lambda:infer(False),'replay':lambda:infer(True),'evaluate':evaluate}[x.mode]()
