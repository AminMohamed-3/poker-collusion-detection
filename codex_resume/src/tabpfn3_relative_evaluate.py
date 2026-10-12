"""One-shot local paired selection evaluation after offline predictions and replay freeze."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,argparse
from pathlib import Path
import numpy as np
import polars as pl
import tabpfn3_relative_prepare as prep
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step5_evaluate import half
sha=prep.sha;read=prep.read;write=prep.write;OUT=prep.OUT

def main(run):
 assert not any((OUT/n).exists() for n in ['results.json','evaluation-started.json','predictions-frozen.json','new-scores.parquet']),'one-shot evaluation already started'
 pf=prep.check();ctx=read(OUT/'context-receipt.json');f=read(run/'freeze.json');done=read(run/'fit-complete.json');re=read(run/'replay.json')
 assert ctx['preflight_sha256']==sha(OUT/'preflight.json') and ctx['normal_reference_receipt_sha256']==sha(OUT/'normal-reference-receipt.json')
 for n,h in {**ctx['artifact_hashes'],**read(OUT/'normal-reference-receipt.json')['outputs']}.items():assert sha(OUT/n)==h,n
 expected={m['stem'] for m in ctx['contexts']};assert len(expected)==90 and set(done['receipts'])==expected
 assert len(ctx['frames'])==15 and len({x['stem'] for x in ctx['frames']})==15
 assert {(x['salt'],x['fold']) for x in ctx['frames']}=={(s,i) for s in range(3) for i in range(5)}
 frame_map={x['stem']:(x['salt'],x['fold']) for x in ctx['frames']}
 assert all(frame_map[m['frame']]==(m['salt'],m['fold']) for m in ctx['contexts'])
 assert {(m['salt'],m['fold'],m['family'],m['arm']) for m in ctx['contexts']}=={(s,i,f,a) for s in range(3) for i in range(5) for f in prep.FAMILIES for a in ['T','RZ']}
 assert f['source_sha256']==sha(prep.HERE/'src/tabpfn3_relative_vm.py') and f['context_receipt_sha256']==sha(OUT/'context-receipt.json')
 assert done['freeze_sha256']==re['freeze_sha256']==sha(run/'freeze.json') and re['ranking_exact'] and re['replay_sha256']==sha(run/'replay.npy')
 assert f['checkpoint_sha256']=='d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'
 import tabpfn3_hardware_resume as helper
 assert f['helper_sha256']==sha(Path(helper.__file__)) and f['params']==helper.PARAMS and f['query_chunk']==256
 scores=[]
 for fr in ctx['frames']:
  hp=OUT/fr['file'];assert sha(hp)==fr['sha256'];d=pl.read_parquet(hp);pred={k:d['oof'].to_numpy().copy()-1e6 for k in ['T','RZ']};coverage={k:np.zeros(d.height,dtype=int) for k in pred}
  for m in [v for v in ctx['contexts'] if v['frame']==fr['stem']]:
   rr=run/'predictions'/(m['stem']+'.json');pp=run/'predictions'/(m['stem']+'.npy');dd=run/'predictions'/(m['stem']+'-diagnostic.json');r=read(rr)
   assert sha(rr)==done['receipts'][m['stem']] and r['probabilities_sha256']==sha(pp) and r['diagnostic_sha256']==sha(dd) and r['context_sha256']==m['context_sha256'] and r['freeze_sha256']==sha(run/'freeze.json')
   cp=OUT/m['context_file'];assert sha(cp)==m['context_sha256']
   with np.load(cp,allow_pickle=False) as c:ix=c['held_indices'];assert np.array_equal(c['held_pair_ids'],d['pair_id'].to_numpy()[ix])
   x=np.load(pp,allow_pickle=False);assert x.shape==(len(ix),) and np.isfinite(x).all() and np.all((x>=0)&(x<=1));pred[m['arm']][ix]=x+10;coverage[m['arm']][ix]+=1
  top=(d['oof_rank']<=20).to_numpy()
  for arm in pred:assert np.all(coverage[arm][top]==1) and np.all(coverage[arm][~top]==0)
  scores.append(d.with_columns(*[pl.Series(k,v) for k,v in pred.items()],pl.lit(fr['salt']).alias('salt'),pl.lit(fr['fold']).alias('fold')))
 scored=pl.concat(scores);assert scored.height==19325*3
 path=OUT/'new-scores.parquet';assert not path.exists();scored.write_parquet(path)
 write(OUT/'predictions-frozen.json',{'scores_sha256':sha(path),'inference_freeze_sha256':sha(run/'freeze.json'),'fit_complete_sha256':sha(run/'fit-complete.json'),'replay_sha256':sha(run/'replay.json'),'context_receipt_sha256':sha(OUT/'context-receipt.json'),'before_evaluation':True})
 # This marker precedes the only metric read; a crash does not authorize a silent second read.
 write(OUT/'evaluation-started.json',{'predictions_frozen_sha256':sha(OUT/'predictions-frozen.json'),'invocation':1})
 evidence=prep.training_relevance(pf['audit_pools']);truths={k[0]:set(g['hand_id']) for k,g in evidence.group_by('pair_id')}
 names=['T','RZ','lgbm','oof'];maps={k:[] for k in names};summaries={};fold_deltas={}
 fams=dict(scored.select('pair_id','behavior_family').unique().iter_rows());ptab=dict(scored.select('pair_id','table_id').unique().iter_rows());pairs=sorted(fams);pools=sorted(set(ptab.values()))
 assert len(pairs)==162 and len(pools)==104 and set(truths)==set(pairs)
 for salt in range(3):
  d=scored.filter(pl.col('salt')==salt);summaries[str(salt)]={}
  for name in names:
   summ,pm=summarize(d.select('pair_id','hand_id','table_id',pl.col(name).alias('score')),truths,fams);maps[name].append(pm);summaries[str(salt)][name]=summ
  fold_deltas[str(salt)]={str(i):float(np.mean([maps['RZ'][-1][p]-maps['T'][-1][p] for p in d.filter(pl.col('fold')==i)['pair_id'].unique()])) for i in range(5)}
 avg={k:{p:float(np.mean([m[p] for m in maps[k]])) for p in pairs} for k in names}
 def compare(base,candidate):
  res=boot(avg[base],avg[candidate],ptab,pools,n=2000,seed=652)
  res['by_family']={f:float(np.mean([avg[candidate][p]-avg[base][p] for p in pairs if fams[p]==f])) for f in prep.FAMILIES}
  res['halves']={str(h):float(np.mean([avg[candidate][p]-avg[base][p] for p in pairs if half(ptab[p])==h])) for h in [0,1]}
  res['salt_means']=[float(np.mean([maps[candidate][s][p]-maps[base][s][p] for p in pairs])) for s in range(3)]
  res['gate_pass']=res['mean_delta']>=.01 and res['ci95'][0]>0 and min(res['by_family'].values())>=-.01 and min(res['halves'].values())>=0 and min(res['salt_means'])>=0
  return res
 primary=compare('T','RZ');lgbmprotect=compare('lgbm','RZ');negative=compare('T','oof')
 result={'study':'T plus relative encoding under common65 normal reference','selection_reused':True,'audit_queries':162,'pools':104,'appearances':486,'outer_confirmation_metrics_read':False,'summaries':summaries,'mean_map':{k:float(np.mean(list(v.values()))) for k,v in avg.items()},'primary_RZ_minus_T':primary,'additional_RZ_minus_LGBM':lgbmprotect,'stage1_negative_control':negative,'T_minus_LGBM_diagnostic':compare('lgbm','T'),'primary_fold_deltas':fold_deltas,'primary_eligible':bool(primary['gate_pass'] and lgbmprotect['gate_pass'] and not negative['gate_pass']),'protected_unchanged':prep.protected()==pf['protected'],'evaluation_invocation':1,'no_posthoc_rescue':True}
 assert result['protected_unchanged'];write(OUT/'results.json',result);write(OUT/'per-pair-maps.json',{'maps':avg,'pool_of_pair':ptab,'family_of_pair':fams});print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);v=a.parse_args();main(v.run.resolve())
