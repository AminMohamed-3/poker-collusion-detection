"""Synthetic-only verification of preordered repeated-row table-bootstrap AP."""
from pathlib import Path
import json,time
import numpy as np

def official(y,s,k):
 order=np.argsort(k,kind='stable'); order=order[np.argsort(-s[order],kind='stable')]; yy=y[order].astype(float)
 return 0. if yy.sum()==0 else float(np.sum(np.cumsum(yy)/np.arange(1,len(yy)+1)*yy)/yy.sum())
def prepare(y,s,k,t):
 order=np.argsort(k,kind='stable');order=order[np.argsort(-s[order],kind='stable')]
 return y[order].astype(float),t[order]
def repeated(prepared,multiplicity):
 y,t=prepared; yy=np.repeat(y,multiplicity[t])
 return 0. if yy.sum()==0 else float(np.sum(np.cumsum(yy)/np.arange(1,len(yy)+1)*yy)/yy.sum())
def main():
 rng=np.random.default_rng(19143);worst=0.; checks=0
 for case in range(120):
  n=int(rng.integers(8,250));nt=7;t=rng.integers(0,nt,n);keys=np.array([f'pair{i:04}' for i in rng.permutation(n)]);y=rng.integers(0,4,n);route=rng.integers(1,4,n);s=rng.choice([0.,.1,.2,.8,1.],n)
  if case%6==0:s[:]=0
  if case%6==1:y[:]=0
  if case%6==2:y[:]=1
  prepared=[prepare(y>0,s,keys,t)]+[prepare(y==c,s*(route==c),keys,t) for c in (1,2,3)]
  for draw in range(12):
   chosen=rng.integers(0,nt,nt);mult=np.bincount(chosen,minlength=nt);idx=np.concatenate([np.flatnonzero(t==v) for v in chosen]);expected=[official(y[idx]>0,s[idx],keys[idx])]+[official(y[idx]==c,s[idx]*(route[idx]==c),keys[idx]) for c in (1,2,3)];actual=[repeated(z,mult) for z in prepared]
   err=float(np.max(np.abs(np.array(expected)-actual)));worst=max(worst,err);assert err==0,(case,draw,err)
   assert .7*expected[0]+.1*np.mean(expected[1:])==.7*actual[0]+.1*np.mean(actual[1:]);checks+=5
 result={'synthetic_metric_checks':checks,'max_absolute_error':worst,'exact':True,'cases':'table repetition, tied/zero scores, route masks, shuffled unique IDs, absent classes, allnegative/allpositive','real_data_or_quality_read':False,'scope':'precomputed per-metric official sort plus actual repeated labels; not endpoint-weighted AP'}
 Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
