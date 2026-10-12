"""Fixed TabPFN3 confirmation capacity, inference, replay and assembly stages."""
import sys,json,time,gc,argparse
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import tabpfn3_evidence_pilot as pilot
import numpy as np
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'experiments/tabpfn3_confirmation_20260912'
PRIMARY='tabpfn3_default'
sha,read,write=pilot.sha,pilot.read,pilot.write

def check():
 plan=read(OUT/'plan.json');ctx=read(OUT/'context-receipt.json');assert ctx['plan_sha256']==sha(OUT/'plan.json')
 for p,h in plan['sources'].items():assert sha(p)==h,p
 for p,h in ctx['context_hashes'].items():assert sha(p)==h,p
 for p,h in ctx['artifact_hashes'].items():assert sha(p)==h,p
 assert ctx['protected_unchanged'] and ctx['controls_exact']==9
 return plan,ctx

def benchmark():
 plan,ctx=check();output=OUT/'benchmark.json';assert not output.exists()
 n,q,w=ctx['max_context'],ctx['max_query'],ctx['max_width'];rng=np.random.default_rng(7329)
 x=rng.normal(size=(n,w));z=rng.normal(size=(q,w));y=np.arange(n,dtype=np.int64)%2;rng.shuffle(y)
 result={'passed':False,'synthetic_only':True,'no_poker_outcomes':True,'shape':{'training_rows':n,'query_rows':q,'features':w},'params':pilot.PARAMS,'source_sha256':sha(Path(__file__)),'context_receipt_sha256':sha(OUT/'context-receipt.json')}
 pilot.network_off()
 try:
  pr,info=pilot.infer({'xtrain':x,'ytrain':y,'xtest':z});result.update({'passed':True,'runtime':info,'probabilities_sha256':__import__('hashlib').sha256(pr.tobytes()).hexdigest()})
 except BaseException as e:
  result['error']={'type':type(e).__name__,'message':str(e)}
 finally:write(output,result)
 assert result['passed'],result
 print(json.dumps(result),flush=True)

def freeze():
 plan,ctx=check();b=read(OUT/'benchmark.json');assert b['passed'] and b['params']==pilot.PARAMS
 expected={'plan_sha256':sha(OUT/'plan.json'),'context_receipt_sha256':sha(OUT/'context-receipt.json'),'benchmark_sha256':sha(OUT/'benchmark.json'),'inference_source_sha256':sha(Path(__file__)),'pilot_source_sha256':sha(Path(pilot.__file__)),'params':pilot.PARAMS,'replay_stem':'fold0-directed_transfer'}
 f=OUT/'freeze.json'
 if f.exists():assert read(f)==expected
 else:write(f,expected)
 return ctx

def fit():
 ctx=freeze();pilot.network_off();dest=OUT/'predictions';dest.mkdir(exist_ok=True)
 for m in ctx['contexts']:
  pp=dest/(m['stem']+'.npy');rr=dest/(m['stem']+'.json')
  if pp.exists() or rr.exists():
   assert pp.exists() and rr.exists();r=read(rr);assert r['freeze_sha256']==sha(OUT/'freeze.json') and r['probabilities_sha256']==sha(pp);continue
  with np.load(OUT/m['context_file'],allow_pickle=False) as c:pr,info=pilot.infer(c)
  np.save(pp,pr);write(rr,{'freeze_sha256':sha(OUT/'freeze.json'),'context_sha256':m['context_sha256'],'probabilities_sha256':sha(pp),'runtime':info})
  print(json.dumps({'context':m['stem'],**info}),flush=True)
 write(OUT/'fit-complete.json',{'freeze_sha256':sha(OUT/'freeze.json'),'receipts':{m['stem']:sha(dest/(m['stem']+'.json')) for m in ctx['contexts']}})

def replay():
 ctx=freeze();pilot.network_off();assert (OUT/'fit-complete.json').exists();m=next(m for m in ctx['contexts'] if m['stem']=='fold0-directed_transfer')
 with np.load(OUT/m['context_file'],allow_pickle=False) as c:pr,info=pilot.infer(c);ids=c['held_pair_ids']
 original=np.load(OUT/'predictions'/(m['stem']+'.npy'),allow_pickle=False)
 exact=all(np.array_equal(np.argsort(-pr[ids==k],kind='stable'),np.argsort(-original[ids==k],kind='stable')) for k in np.unique(ids));assert exact
 np.save(OUT/'replay.npy',pr);write(OUT/'replay.json',{'freeze_sha256':sha(OUT/'freeze.json'),'replay_sha256':sha(OUT/'replay.npy'),'ranking_exact':exact,'probabilities_exact':bool(np.array_equal(pr,original)),'maximum_difference':float(np.max(np.abs(pr-original))),'runtime':info})
 print(json.dumps({'ranking_exact':exact,'maximum_difference':float(np.max(np.abs(pr-original)))}),flush=True)

def assemble():
 import polars as pl
 ctx=freeze();r=read(OUT/'replay.json');assert r['freeze_sha256']==sha(OUT/'freeze.json') and r['ranking_exact'] and r['replay_sha256']==sha(OUT/'replay.npy')
 done=read(OUT/'fit-complete.json');assert done['freeze_sha256']==sha(OUT/'freeze.json');sf=OUT/'predictions-frozen.json';assert not sf.exists()
 files={}
 for fold in (0,1,2):
  d=pl.read_parquet(OUT/f'fold{fold}-held.parquet');p=d['oof'].to_numpy().copy();covered=np.zeros(d.height,int)
  for m in (m for m in ctx['contexts'] if m['fold']==fold):
   rr=OUT/'predictions'/(m['stem']+'.json');pp=OUT/'predictions'/(m['stem']+'.npy')
   assert sha(rr)==done['receipts'][m['stem']];rec=read(rr);assert rec['freeze_sha256']==sha(OUT/'freeze.json') and rec['probabilities_sha256']==sha(pp)
   with np.load(OUT/m['context_file'],allow_pickle=False) as c:
    ix=c['held_indices'];assert np.array_equal(c['held_pair_ids'],d['pair_id'].to_numpy()[ix]);p[ix]=np.load(pp,allow_pickle=False)+10;covered[ix]+=1
   tail=((d['predicted_behavior']==m['family'])&(d['oof_rank']>20)).to_numpy();p[tail]=d['oof'].to_numpy()[tail]-1e6
  recognized=d['predicted_behavior'].is_in(['directed_transfer','soft_play','coordinated_isolation']).to_numpy();top=recognized&(d['oof_rank']<=20).to_numpy()
  assert np.all(covered[top]==1) and np.all(covered[~top]==0) and np.isfinite(p).all()
  np.testing.assert_array_equal(p[~recognized],d['oof'].to_numpy()[~recognized])
  for name,values in [('incumbent',d['currentcontrol'].to_numpy()),(PRIMARY,p),('stage1_only',d['oof'].to_numpy())]:
   path=OUT/'scores'/f'fold{fold}-{name}.parquet';assert not path.exists()
   d.select('pair_id','hand_id','table_id').with_columns(pl.Series('score',values)).write_parquet(path);files[path.name]=sha(path)
 write(sf,{'plan_sha256':sha(OUT/'plan.json'),'context_receipt_sha256':sha(OUT/'context-receipt.json'),'freeze_sha256':sha(OUT/'freeze.json'),'replay_sha256':sha(OUT/'replay.json'),'fit_complete_sha256':sha(OUT/'fit-complete.json'),'scores':files,'all_predictions_before_held_metrics':True})
 print('All confirmation predictions frozen before held metrics',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['benchmark','fit','replay','assemble']);args=a.parse_args();{'benchmark':benchmark,'fit':fit,'replay':replay,'assemble':assemble}[args.mode]()
