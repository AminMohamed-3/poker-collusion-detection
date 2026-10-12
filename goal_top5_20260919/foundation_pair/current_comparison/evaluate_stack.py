import os,sys,site,json,hashlib
from pathlib import Path
os.environ['POLARS_MAX_THREADS']='2'; ROOT=Path(r'C:\Data\Projects\Money Making'); C=ROOT/'work/poker/Claude Version'; OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/current_comparison'
site.addsitedir(str(ROOT/'work/poker/.venv/Lib/site-packages'))
sys.path.insert(0,str(C/'src'))
import numpy as np, polars as pl
from pu_pairs.step60_multiview_blend import ap,logreg,predict,lg
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step64f_M3_candidate import FS_M3
from pu_pairs.step62_multiview_candidate import DEFS
SALT='foundation-pf35-screen-v1:'; F10=['SXN3_SHORT3','SXN3_ALL5','SXN3_LIN30','SXN3_LIN300','SXN3_FAMMAX','ARCH_XGB3','ARCH_ET','ARCH_MLP','ARCH2_MLP2','ARCH2_HGB5']
def fold(t): return int(hashlib.sha256((SALT+t).encode()).hexdigest()[:16],16)%2
def main():
 vpath=ROOT/'work/poker/goal_top5_20260919/reproduction/recovered_step272/V_arch3.parquet'; assert hashlib.sha256(vpath.read_bytes()).hexdigest()=='914ec304ede0d702f11c4db9898557632987e43f02f0dc06930a8fcfe9ecb39d'
 d=pl.read_parquet(C/'experiments/step258_mech_block_view_20260918/testbed32.parquet').join(pl.read_parquet(vpath).filter(pl.col('surface')=='prefix60').drop('surface'),on=['pair_id','fold'],validate='1:1')
 S=pl.concat([pl.read_parquet(C/f'experiments/pu_pairs_20260911/fold{f}/scores.parquet',columns=['pair_id','shared_hands']) for f in (1,2)]).unique('pair_id',keep='first')
 d=d.join(S.rename({'shared_hands':'sh_full'}),on='pair_id',validate='1:1'); X0=X_of(d,FS_M3,'risk_prefix','R1_prefix','risk_noT'); key=d.select(F10).to_numpy().mean(1); m2=d['ARCH2_MLP2'].to_numpy(); X=np.column_stack([X0,key,m2])
 pf=np.load(OUT/'fast_predictions.npz'); outer_map={str(i):p for f in (0,1) for i,p in zip(pf[f'instrument_id{f}'],pf[f'instrument_p{f}'])}; outer_pp=np.array([outer_map[i] for i in d['pair_id']])
 hf=np.array([fold(t) for t in d['table_id']],dtype=int); posA,negA=DEFS['A']; certA=d['cls'].is_in(posA+negA).to_numpy(); yA=d['cls'].is_in(posA).to_numpy().astype(float)
 def cf_base(Z):
  o=np.empty(len(d))
  for f in (0,1):
   tr=(hf!=f)&certA; va=hf==f; m=logreg(Z[tr],yA[tr],10.0); o[va]=predict(m,Z[va])
  return o
 base=cf_base(X); arm=np.empty(len(d)); nested_pp=np.empty((len(d),4))
 for f in (0,1):
  tr=hf!=f; va=hf==f; imap={str(i):p for inn in (0,1) for i,p in zip(pf[f'outer{f}_inner{inn}_id'],pf[f'outer{f}_inner{inn}_p'])}; tp=np.array([imap[i] for i in np.array(d['pair_id'].to_list())[tr]]); hp=outer_pp[va]; nested_pp[tr]=tp; nested_pp[va]=hp
  Ztr=np.column_stack([X[tr],lg(np.clip(1-tp[:,0],1e-9,1-1e-9))]); Zh=np.column_stack([X[va],lg(np.clip(1-hp[:,0],1e-9,1-1e-9))]); m=logreg(Ztr[certA[tr]],yA[tr][certA[tr]],10.0); arm[va]=predict(m,Zh)
 pr=np.clip(1-outer_pp[:,0],1e-9,1-1e-9); elig=d['sh_full'].to_numpy()>=57; DEFB=(['pos','undisclosed'],['FP','undisclosed_R1','ambiguous','mid','neg'])
 res={'scope':'proxy instrument compatibility; not public forecast','arm':'STEP272 14 inputs + standardized PF logit, l2=10','definitions':{},'pf_risk_correlation_with_base':float(np.corrcoef(pr,base)[0,1])}
 for name,(ps,ns) in [('A',DEFS['A']),('C',DEFS['C']),('B',DEFB)]:
  cert=d['cls'].is_in(ps+ns).to_numpy(); y=d['cls'].is_in(ps).to_numpy().astype(float); rr={}
  for pop,mask in [('all',np.ones(len(d),bool)),('eligible',elig),('fold0',hf==0),('fold1',hf==1)]:
   z=cert&mask; rr[pop]={'rows':int(z.sum()),'positives':int(y[z].sum()),'base':float(ap(y[z],base[z])),'arm':float(ap(y[z],arm[z])),'delta':float(ap(y[z],arm[z])-ap(y[z],base[z]))}
  # Paired table bootstrap on the two prespecified populations.
  rng=np.random.default_rng(35935); tabs=np.array(d['table_id'].to_list()); uniq=np.unique(tabs)
  for pop,mask in [('all',np.ones(len(d),bool)),('eligible',elig)]:
   ds=[]
   for _ in range(1000):
    take=rng.choice(uniq,len(uniq),replace=True); idx=np.concatenate([np.where((tabs==t)&cert&mask)[0] for t in take]);
    if y[idx].sum()>0 and y[idx].sum()<len(idx): ds.append(ap(y[idx],arm[idx])-ap(y[idx],base[idx]))
   rr[pop]['table_bootstrap']={'n':len(ds),'mean':float(np.mean(ds)),'q025':float(np.quantile(ds,.025)),'q975':float(np.quantile(ds,.975)),'p_gt0':float(np.mean(np.array(ds)>0))}
  res['definitions'][name]=rr
 lab=pl.read_csv(ROOT/'work/poker/data/development_labels.csv').select('pair_id','label'); dl=d.select('pair_id').join(lab,on='pair_id',how='left'); known=dl['label'].is_not_null().to_numpy(); yy=dl['label'].fill_null(0).to_numpy()
 res['confirmed_tiny_negative_diagnostic']={'rows':int(known.sum()),'positives':int(yy[known].sum()),'negatives':int((1-yy[known]).sum()),'base_ap':float(ap(yy[known],base[known])),'arm_ap':float(ap(yy[known],arm[known])),'warning':'Only three negatives; descriptive, not reliable truth estimate.'}
 rb=np.empty(len(d),int);ra=np.empty(len(d),int);rb[np.argsort(-base,kind='stable')]=np.arange(len(d));ra[np.argsort(-arm,kind='stable')]=np.arange(len(d));res['rank_change']={'spearman':float(np.corrcoef(rb,ra)[0,1]),'mean_abs':float(np.abs(rb-ra).mean()),'max_abs':int(np.abs(rb-ra).max()),'gt50':int((np.abs(rb-ra)>50).sum())}
 (OUT/'stack_results.json').write_text(json.dumps(res,indent=2))
 np.savez_compressed(OUT/'stack_predictions.npz',pair_id=np.array(d['pair_id'].to_list(),dtype='U'),hash_fold=hf,outer_pf_probability=outer_pp,nested_pf_probability=nested_pp,base=base,arm=arm)
if __name__=='__main__': main()
