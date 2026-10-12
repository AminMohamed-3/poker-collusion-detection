"""Compose both frozen arms, then perform one quality read via peer helper semantics."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np,polars as pl
O=Path(__file__).resolve().parent;PEER=Path(r'C:/Data/Projects/Money Making/work/poker/Codex Resume/experiments/tabm_citation_pilot_20260919');CV=Path(r'C:/Data/Projects/Money Making/work/poker/Claude Version');K=['pair_id','hand_id'];ARMS=['support_control','conditional_candidate'];SP='soft_play'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def helper():
 spec=importlib.util.spec_from_file_location('peer_eval_readonly',PEER/'evaluate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def compose():
 assert (O/'predictions_frozen.json').exists() and not (O/'composed_frozen.json').exists();m=helper();pr=np.load(O/'predictions.npz',allow_pickle=False);keys=pl.read_parquet(PEER/'audit-keys.parquet').select(K);assert np.array_equal(pr['pair_id'],keys['pair_id']) and np.array_equal(pr['hand_id'],keys['hand_id'])
 fr=pl.read_parquet(PEER/'reference.parquet');refrec=json.loads((PEER/'reference-receipt.json').read_text());assert sha(PEER/'reference.parquet')==refrec['sha256'];mask=fr['is_top20'].to_numpy()&(fr['behavior_family'].to_numpy()==SP);assert keys.equals(fr.filter(pl.Series(mask)).select(K));typed=fr['typed'].to_numpy();current=fr['current'].to_numpy();result=fr.select(K+['table_id','behavior_family','is_top20','current']).with_columns(fr['current_plus_rule_predecode'].alias('current_predecode'))
 for arm in ARMS:
  raw=fr['current_plus_rule_predecode'].to_numpy().copy();event=fr['current_reference_predecode'].to_numpy();p=pr[arm].astype(np.float64);raw[mask]=10+np.clip(event[mask]-10,1e-9,1)**.70*np.clip(p,1e-9,1)**.30;decoded=m.decode_ci2(fr.with_columns(pl.Series('score',raw)),typed);assert np.array_equal(decoded[~mask],current[~mask]) and np.array_equal(decoded>5,current>5);result=result.with_columns(pl.Series(arm,decoded),pl.Series(arm+'_predecode',raw))
 result.write_parquet(O/'composed.parquet');mfest={'sha256':sha(O/'composed.parquet'),'fit_freeze_sha256':sha(O/'predictions_frozen.json'),'peer_evaluator_sha256':sha(PEER/'evaluate.py'),'peer_reference_sha256':sha(PEER/'reference.parquet'),'quality_reads':0};(O/'composed_frozen.json').write_text(json.dumps(mfest,indent=2)+'\n')
def evaluate():
 assert (O/'composed_frozen.json').exists() and not (O/'evaluation_started.json').exists() and not (O/'results.json').exists();m=helper();lock=json.loads((O/'composed_frozen.json').read_text());assert sha(O/'composed.parquet')==lock['sha256'] and sha(PEER/'evaluate.py')==lock['peer_evaluator_sha256'];(O/'evaluation_started.json').write_text('{"single_fixed_quality_read":true}\n')
 d=pl.read_parquet(O/'composed.parquet');truth=pl.read_parquet(CV/'experiments/label_process_20260911/audit-oof.parquet',columns=K+['relevance']).sort(K);assert d.select(K).equals(truth.select(K));d=d.with_columns(truth['relevance']);truths={k[0]:set(g['hand_id']) for k,g in d.filter(pl.col('relevance')==1).group_by('pair_id')};fams=dict(d.select('pair_id','behavior_family').unique().iter_rows());pools=dict(d.select('pair_id','table_id').unique().iter_rows());qids=sorted(truths);scores={};rank={}
 for name in ['current']+ARMS:
  _,scores[name]=m.summarize(d.select(K+['table_id',pl.col(name).alias('score')]),truths,fams);num=den=0.;aps=[]
  for _,g in d.filter(pl.col('is_top20')&(pl.col('behavior_family')==SP)).group_by('pair_id'):
   y=g['relevance'].to_numpy();p=g[name+'_predecode'].to_numpy();pos=p[y==1];neg=p[y==0];num+=float((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum());den+=len(pos)*len(neg);aps.append(m.average_precision_score(y,p) if len(pos) else 0.)
  rank[name]={'SP_within_pair_AP':float(np.mean(aps)),'SP_weighted_pairwise_AUC':num/den}
 means={n:float(np.mean([scores[n][q] for q in qids])) for n in scores};assert abs(means['current']-.7502623456790123)<1e-15;comps={}
 for a,b in [('conditional_candidate','current'),('conditional_candidate','support_control'),('support_control','current')]:comps[a+'-'+b]=m.compare(np.array([scores[a][q]-scores[b][q] for q in qids]),qids,pools)
 gates={}
 for key in ('conditional_candidate-current','conditional_candidate-support_control'):
  c=comps[key];gates[key+'_positive']=c['mean_delta']>0;gates[key+'_CI_lower_positive']=c['CI95_pool_bootstrap'][0]>0;gates[key+'_halves_nonnegative']=min(c['halves'].values())>=0
 for metric in ('SP_within_pair_AP','SP_weighted_pairwise_AUC'):gates[metric+'_nondegraded_vs_current']=rank['conditional_candidate'][metric]>=rank['current'][metric]
 out={'means':means,'comparisons':comps,'ranking_diagnostics':rank,'gates':gates,'verdict':'DEVELOPMENT_SCREEN_PASS' if all(gates.values()) else 'NOT_ESTABLISHED','quality_reads':1,'bootstrap':{'draws':5000,'seed':19143,'unit':'original audit pool'},'limitations':['heavily reused oracle-family audit','single seed','no rescue or submission authorization']};(O/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['compose','evaluate']);z=a.parse_args();compose() if z.mode=='compose' else evaluate()
