"""Label-reading evaluator for frozen reduced integration predictions."""
import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json
from pathlib import Path
import numpy as np,polars as pl
H=Path(__file__).resolve().parent;P=H.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ap(y,s,k):
 y=np.asarray(y,dtype=bool);s=np.asarray(s,dtype=float);z=np.argsort(k,kind='stable');o=z[np.argsort(-s[z],kind='stable')];yy=y[o];return float((np.cumsum(yy)/(np.arange(len(y))+1))[yy].mean()) if yy.any() else 0.0
def met(d,c):
 y=d['label'].to_numpy();pr=np.column_stack([d[f'{c}_{i}'] for i in range(4)]);s=1-pr[:,0];k=d['pair_id'].to_numpy();pa=ap(y,s,k);fam=d['behavior_family'].to_numpy();route=np.argmax(pr[:,1:],axis=1);bm=np.mean([ap(fam==x,s*(route==i),k) for i,x in enumerate(['directed_transfer','soft_play','coordinated_isolation'])]);return {'pair_ap':pa,'behavior_map':float(bm),'joint':.7*pa+.1*bm}
def main():
 assert not(H/'RESULTS.json').exists();m=json.loads((H/'MANIFEST.json').read_text());assert all(sha(Path(k))==v for k,v in m['sources'].items()) and all(sha(H/k)==v for k,v in m['artifacts'].items());r=json.loads((H/'RUN_RECEIPT.json').read_text());assert r['manifest_sha256']==sha(H/'MANIFEST.json');parts=[]
 for x in r['folds']:
  p=H/f'pair_predictions_outer{x["outer"]}.parquet';assert sha(p)==x['prediction_sha256'];parts.append(pl.read_parquet(p))
 d=pl.concat(parts);assert d.height==1860 and d['pair_id'].n_unique()==1860
 for c in ['baseline','augmented']:
  q=np.column_stack([d[f'{c}_{i}'] for i in range(4)]);assert np.isfinite(q).all() and (q>=0).all() and np.allclose(q.sum(1),1,atol=1e-6)
 d=d.join(pl.read_csv(P/'data/development_labels.csv').select('pair_id','label','behavior_family'),on='pair_id',validate='1:1');assert abs(ap([1,0,1],[.5,.5,.2],np.array(['b','a','c']))-7/12)<1e-12 and ap([0,0],[.2,.1],np.array(['a','b']))==0
 sy=pl.DataFrame({'pair_id':['a','b','c','d'],'label':[0,1,1,1],'behavior_family':['none','directed_transfer','soft_play','coordinated_isolation'],'baseline_0':[.8,.3,.4,.2],'baseline_1':[.1,.5,.1,.1],'baseline_2':[.05,.1,.4,.1],'baseline_3':[.05,.1,.1,.6],'augmented_0':[.75,.25,.35,.15],'augmented_1':[.1,.55,.1,.1],'augmented_2':[.1,.1,.45,.1],'augmented_3':[.05,.1,.1,.65]});six=np.array([0,1,2,3,0,1]);sl=pl.concat([sy,sy[:2]]);assert abs((met(sy[six],'augmented')['joint']-met(sy[six],'baseline')['joint'])-(met(sl,'augmented')['joint']-met(sl,'baseline')['joint']))<1e-15
 b=met(d,'baseline');a=met(d,'augmented');fold={str(o):met(d.filter(pl.col('outer_fold')==o),'augmented')['joint']-met(d.filter(pl.col('outer_fold')==o),'baseline')['joint'] for o in [0,1]};tabs=sorted(d['table_id'].unique().to_list());ta=d['table_id'].to_numpy();by=[np.flatnonzero(ta==t) for t in tabs];rng=np.random.default_rng(35935);boot=[]
 for _ in range(1000):
  ix=np.concatenate([by[i] for i in rng.integers(0,len(tabs),len(tabs))]);z=d[ix];boot.append(met(z,'augmented')['joint']-met(z,'baseline')['joint'])
 fam={f:{'baseline_pair_ap':ap(d['behavior_family'].to_numpy()==f,1-np.array(d['baseline_0']),d['pair_id'].to_numpy()),'augmented_pair_ap':ap(d['behavior_family'].to_numpy()==f,1-np.array(d['augmented_0']),d['pair_id'].to_numpy())} for f in ['directed_transfer','soft_play','coordinated_isolation']};cls=np.array([{'none':0,'directed_transfer':1,'soft_play':2,'coordinated_isolation':3}[x] for x in d['behavior_family']]);ll={c:float(-np.log(np.maximum(np.column_stack([d[f'{c}_{i}'] for i in range(4)])[np.arange(d.height),cls],1e-12)).mean()) for c in ['baseline','augmented']}
 res={'baseline':b,'augmented':a,'delta_joint':a['joint']-b['joint'],'delta_by_fold':fold,'table_bootstrap_95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'per_family':fam,'multiclass_logloss':ll,'gate':a['joint']>b['joint'] and all(v>0 for v in fold.values()),'run_receipt_sha256':sha(H/'RUN_RECEIPT.json')};(H/'RESULTS.json').write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':main()
