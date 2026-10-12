"""Evaluate only frozen prediction files; this is the sole label-reading stage."""
import site; site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import json,hashlib
from pathlib import Path
import numpy as np, polars as pl
H=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (H/'RESULTS.json').exists()
 man=json.loads((H/'MANIFEST.json').read_text()); assert all(sha(H/k)==v for k,v in man['artifacts'].items()); rec=json.loads((H/'FIT_RECEIPT.json').read_text()); assert rec['manifest_sha256']==sha(H/'MANIFEST.json'); ps=[]
 for x in rec['folds']:
  p=H/f'predictions_outer{x["outer"]}.parquet'; assert sha(p)==x['prediction_sha256']; ps.append(pl.read_parquet(p))
 d=pl.concat(ps); assert d.height==man['stats']['cache_rows'] and d.select(pl.struct('pair_id','direction','hand_id','action_no').n_unique()).item()==d.height; y=d['action_class'].to_numpy(); ix=np.arange(d.height)
 for a,b,n in [('p1','placebo','gain_primary'),('p1','p0','gain_secondary')]:
  pa=np.column_stack([d[f'{a}_{k}'].to_numpy() for k in range(4)])[ix,y]; pb=np.column_stack([d[f'{b}_{k}'].to_numpy() for k in range(4)])[ix,y]
  d=d.with_columns(pl.Series(n,np.log(np.maximum(pa,1e-12))-np.log(np.maximum(pb,1e-12))))
 pair=d.group_by('pair_id','table_id','outer_fold','direction').agg(pl.mean('gain_primary'),pl.mean('gain_secondary'),pl.len().alias('actions')).group_by('pair_id','table_id','outer_fold').agg(pl.mean('gain_primary'),pl.mean('gain_secondary'),pl.sum('actions'))
 labels=pl.read_parquet(H/'evaluator_labels.parquet'); pair=pair.join(labels,on=['pair_id','table_id'],validate='1:1'); assert pair.height==1860 and pair['pair_id'].n_unique()==1860
 def excess(z,c): return float(z.filter(pl.col('label')==1)[c].mean()-z.filter(pl.col('label')==0)[c].mean())
 result={'primary_pooled':excess(pair,'gain_primary'),'secondary_pooled':excess(pair,'gain_secondary'),'primary_by_fold':{str(o):excess(pair.filter(pl.col('outer_fold')==o),'gain_primary') for o in [0,1]},'pairs':pair.height,'label_sha256':sha(H/'evaluator_labels.parquet'),'prediction_hashes':{f'outer{x["outer"]}':x['prediction_sha256'] for x in rec['folds']}}
 result['gains']={c:{'global':float(pair[c].mean()),'confirmed_normal':float(pair.filter(pl.col('label')==0)[c].mean()),'target':float(pair.filter(pl.col('label')==1)[c].mean()),'by_family':{str(r['behavior_family']):float(r[c]) for r in pair.group_by('behavior_family').agg(pl.mean(c)).to_dicts()}} for c in ['gain_primary','gain_secondary']}
 result['coverage']={'directions':d.group_by('direction').len().sort('direction').to_dicts(),'actions':d.height,'private_surprise_quantiles':[float(x) for x in np.quantile(d['private_surprise'].to_numpy(),[0,.01,.1,.5,.9,.99,1])],'placebo_surprise_quantiles':[float(x) for x in np.quantile(d['placebo_surprise'].to_numpy(),[0,.01,.1,.5,.9,.99,1])]}
 result['action_class_gains']={str(k):{'primary':float(d.filter(pl.col('action_class')==k)['gain_primary'].mean()),'secondary':float(d.filter(pl.col('action_class')==k)['gain_secondary'].mean()),'actions':d.filter(pl.col('action_class')==k).height} for k in range(4)}
 result['pair_action_coverage']=pair.select(pl.col('actions').min().alias('min_actions'),pl.col('actions').median().alias('median_actions'),pl.col('actions').max().alias('max_actions')).to_dicts()[0]
 rng=np.random.default_rng(35935); tabs=sorted(pair['table_id'].unique().to_list()); rows=pair.to_dicts(); by={t:[r for r in rows if r['table_id']==t] for t in tabs}; vals=[]
 for _ in range(1000):
  take=rng.integers(0,len(tabs),len(tabs)); rr=[r for q in take for r in by[tabs[q]]]; a=[r['gain_primary'] for r in rr if r['label']==1]; b=[r['gain_primary'] for r in rr if r['label']==0]; vals.append(float(np.mean(a)-np.mean(b)))
 result['table_bootstrap_95']=[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]; result['gate']=result['primary_pooled']>0 and all(v>0 for v in result['primary_by_fold'].values())
 (H/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
