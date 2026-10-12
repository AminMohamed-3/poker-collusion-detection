"""Frozen STEP277 composition check using leakage-free saved OOF event scores; no fits."""
import os, sys, json, hashlib
from pathlib import Path
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import numpy as np
import polars as pl

POKER=Path(r'C:/Data/Projects/Money Making/work/poker'); CV=POKER/'Claude Version'
sys.path.insert(0,str(CV/'src'));sys.path.insert(0,str(POKER/'Codex Resume/src'))
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug import step254_eval as S254
from evidence_aug.step104_C_typed_decoder_build import chronology,align,less_than_five
from evidence_aug.step5_evaluate import half

OUT=Path(__file__).resolve().parent; K=S165.K; SP='soft_play'; CI='coordinated_isolation'
BARE=Path(r'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/074de80d-a4a8-4cac-89d6-7754b7d2c154/scratchpad/inc_p_PRODbare.npy')
OOF=CV/'experiments/step254_mechanical_tensor_20260917/oof_mechplus.npy'
RULE=CV/'experiments/step277_generator_rule_20260919/induced_rule_audit.npy'
TYPE=CV/'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'
V1=CV/'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

D=S165.load();d=D['d'];QIX=D['QIX'];top=D['top'];pid=d['pair_id'].to_numpy();rel=d['relevance'].to_numpy().astype(int)
fams=dict(d.select('pair_id','behavior_family').unique().iter_rows()); famrow=np.array([fams[p] for p in pid])
ptab=dict(d.select('pair_id','table_id').unique().iter_rows()); qall=sorted(set(pid[top])); fqa=np.array([fams[q] for q in qall])
truths={k[0]:set(g['hand_id']) for k,g in d.filter(pl.col('relevance')==1).group_by('pair_id')}
NREL={q:len(truths[q]) for q in qall};GT={q:np.flatnonzero((pid==q)&top) for q in qall}
v=pl.read_parquet(V1).sort(K);fr=chronology(d.select(K+['table_id']).with_columns(pl.Series('score',v['S_star'].to_numpy())))
t1=align(fr,pl.read_parquet(TYPE));cr=fr['chrono_rank'].to_numpy();PRI=(~(t1&(famrow!=CI))).astype(np.int64)
pf=np.zeros(len(pid));pf[QIX]=np.load(BARE);rule=np.load(RULE);assert len(rule)==len(pid)

# Saved OOF mechplus predictions are aligned to step254_eval.load(), whose fold assignment is asserted
# row-by-row against step232. Select the SP head (stack order any, DT, SP, CI) and join by exact keys.
h,_=S254.load();pred=np.load(OOF);assert pred.shape==(4,h.height)
oof=h.select(K+['fold']).with_columns(pl.Series('p_SP_oof',pred[2]))
joined=d.select(K).join(oof,on=K,how='left',validate='1:1');assert joined['p_SP_oof'].null_count()==0
pev=joined['p_SP_oof'].to_numpy();assert np.isfinite(pev).all()

def prior_shift(p,F=10.):
 q=np.clip(p,1e-9,1-1e-9);o=q/(1-q)*F;return o/(1+o)
event=pf.copy();sm=famrow==SP
event[sm]=np.clip(pf[sm],1e-9,1.)**.75*prior_shift(pev[sm])**.25
composed=event.copy();composed[sm]=np.clip(event[sm],1e-9,1.)**.70*np.clip(rule[sm],1e-9,1.)**.30

def ap5(hits,q):
 n=0;s=0.
 for i,x in enumerate(hits,1):
  if x:n+=1;s+=n/i
 return s/min(5,NREL[q])
def run(p):
 vals=[];sels={}
 for q in qall:
  g=GT[q];z=np.clip(p[g],0,1);o=np.lexsort((cr[g],PRI[g]));w=np.empty(len(g));w[o]=less_than_five(z[o])
  score=10+z*w+1e-9*z;s5=g[np.argsort(-score,kind='stable')[:5]];sels[q]=tuple(s5);vals.append(ap5(rel[s5],q))
 return np.array(vals),sels
B,SB=run(pf);E,SE=run(event);C,SC=run(composed);delta=C-E
pools=sorted(set(ptab.values()));qbp={p:[i for i,q in enumerate(qall) if ptab[q]==p] for p in pools};rng=np.random.default_rng(27730)
boot=np.array([delta[np.concatenate([qbp[pools[i]] for i in rng.choice(len(pools),len(pools),replace=True)])].mean() for _ in range(4000)])
hva=np.array([half(ptab[q]) for q in qall]);spq=fqa==SP
result={'status':'COMPLETE_LEAKAGE_FREE_OOF_PROXY_NOT_EXACT_PRODUCTION_ENSEMBLE','fixed':{'w_rule':.30,'w_event':.25,'event_prior_F':10.,'typed_decoder':'current'},
 'rows':len(pid),'queries':len(qall),'soft_play_queries':int(spq.sum()),'baseline_bare_map5':float(B.mean()),'event_oof_baseline_map5':float(E.mean()),'event_plus_rule_map5':float(C.mean()),
 'primary_delta_all':float(delta.mean()),'primary_delta_soft_play':float(delta[spq].mean()),'bootstrap_P_gt_0':float((boot>0).mean()),'bootstrap_ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
 'half_deltas':[float(delta[hva==0].mean()),float(delta[hva==1].mean())],
 'lists_changed_vs_event':sum(SC[q]!=SE[q] for q in qall),'sets_changed_vs_event':sum(set(SC[q])!=set(SE[q]) for q in qall),
 'event_effect_vs_bare_all':float((E-B).mean()),'event_effect_vs_bare_soft_play':float((E-B)[spq].mean()),
 'provenance':{'bare_sha256':sha(BARE),'oof_mechplus_sha256':sha(OOF),'rule_sha256':sha(RULE),'type_sha256':sha(TYPE),'v1_sha256':sha(V1)},
 'validity':['Event score is the saved fold-held-out STEP254 mechplus SP prediction; no audit row was scored by a model trained on its table.','Citation rule is the already saved disjoint-210/162 prediction; no model was fitted in this audit.'],
 'limitation':'Production uses the geometric mean of five full-production fold models. A leakage-free audit can use only each row table fold held-out model, so this resolves pipeline composition/sign under proper exclusions but cannot exactly reproduce production event-score geometry.',
 'controls':'No saved PERM/SHUF_FAM per-row predictions exist. Their bare-baseline aggregate results cannot be recomposed exactly; rerunning their fits was intentionally avoided.'}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
