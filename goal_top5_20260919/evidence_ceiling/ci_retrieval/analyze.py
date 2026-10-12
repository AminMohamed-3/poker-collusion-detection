import hashlib, json, site
site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
from pathlib import Path
import numpy as np
import polars as pl

P=Path(r'C:/Data/Projects/Money Making/work/poker')
REF=P/'Codex Resume/experiments/tabm_citation_pilot_20260919/reference.parquet'
TRUTH=P/'Claude Version/experiments/label_process_20260911/audit-oof.parquet'
STRUCT=P/'Claude Version/experiments/label_process_20260911/audit-structural-features.parquet'
OUT=P/'goal_top5_20260919/evidence_ceiling/ci_retrieval'
K=['pair_id','hand_id']
def sha(x): return hashlib.sha256(Path(x).read_bytes()).hexdigest()
def cliffs(a,b):
 a=np.asarray(a,float); b=np.asarray(b,float)
 return float(((a[:,None]>b).sum()-(a[:,None]<b).sum())/(len(a)*len(b)))

r=pl.read_parquet(REF).sort(K)
t=pl.read_parquet(TRUTH).select(K+['relevance','oof','oof_rank','chrono_rank','chrono_frac','n_hands']).sort(K)
s=pl.read_parquet(STRUCT).sort(K)
assert r.select(K).equals(t.select(K)) and r.select(K).equals(s.select(K))
d=r.join(t,on=K,validate='1:1').join(s,on=K,validate='1:1')
d=d.with_columns(pl.col('score').rank('ordinal',descending=True).over('pair_id').alias('current_stage1_rank'))
assert d.select(((pl.col('current_stage1_rank')<=20)==pl.col('is_top20')).all()).item()
c=d.filter(pl.col('behavior_family')=='coordinated_isolation')
miss=c.filter((pl.col('relevance')==1)&~pl.col('is_top20'))
got=c.filter((pl.col('relevance')==1)&pl.col('is_top20'))
neg=c.filter((pl.col('relevance')==0)&pl.col('is_top20'))
num=[x for x,v in s.schema.items() if x not in K and v.is_numeric()]
rows=[]
for x in num:
 a=miss[x].to_numpy(); b=got[x].to_numpy(); n=neg[x].to_numpy()
 if len(a) and np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(n).all():
  rows.append({'feature':x,'miss_median':float(np.median(a)),'retrieved_relevant_median':float(np.median(b)),
               'selected_negative_median':float(np.median(n)),'cliff_miss_vs_retrieved':cliffs(a,b),
               'cliff_miss_vs_selected_negative':cliffs(a,n)})
rows.sort(key=lambda z:-abs(z['cliff_miss_vs_retrieved']))
oldmiss=set(map(tuple,d.filter((pl.col('relevance')==1)&(pl.col('behavior_family')=='coordinated_isolation')&(pl.col('oof_rank')>20)).select(K).iter_rows()))
newmiss=set(map(tuple,miss.select(K).iter_rows()))
rankhist=miss.select(pl.min('current_stage1_rank').alias('min'),pl.median('current_stage1_rank').alias('median'),pl.max('current_stage1_rank').alias('max'),(pl.col('current_stage1_rank')<=25).sum().alias('le25'),(pl.col('current_stage1_rank')<=30).sum().alias('le30')).row(0,named=True)
out={'scope':'reused 43-query CI development diagnostic; no fitted models or parameter selection',
 'candidate_definition':{'formula':'within-pair ordinal descending rank of reference.score <= 20','identity_assertion':True},
 'counts':{'ci_relevant':int(c['relevance'].sum()),'current_in20':got.height,'current_missed':miss.height,'pairs_with_miss':miss['pair_id'].n_unique(),
           'old_oof_missed':len(oldmiss),'still_missed_from_old':len(newmiss&oldmiss),'recovered_old_misses':len(oldmiss-newmiss),'newly_missed_vs_old':len(newmiss-oldmiss)},
 'miss_current_rank':rankhist,
 'top_raw_descriptors_unadjusted':rows[:20],
 'warning':'Descriptors are automated univariate reused-label contrasts over n=20 misses, uncorrected and not a selection license.',
 'provenance':{str(x):sha(x) for x in [REF,TRUTH,STRUCT]}}
(OUT/'diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['counts']|{'rank':rankhist,'top5':rows[:5]},indent=2))
