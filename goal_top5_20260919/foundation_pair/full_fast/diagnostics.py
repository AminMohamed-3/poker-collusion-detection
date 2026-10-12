import sys, site, json
from pathlib import Path
ROOT = Path(r'C:\Data\Projects\Money Making')
site.addsitedir(str(ROOT/'work/poker/.venv/Lib/site-packages'))
import numpy as np
import polars as pl
from scipy.stats import spearmanr
O = ROOT/'work/poker/goal_top5_20260919/foundation_pair/full_fast'
CUR = ROOT/'work/poker/goal_top5_20260919/outputs/submission_sp277_ci276_on_step272.csv'
STEP = ROOT/'work/poker/Claude Version/outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv'
p = pl.read_parquet(O/'fast_probabilities.parquet')
c = pl.read_csv(CUR, columns=['pair_id','risk_score','predicted_behavior'])
assert p['pair_id'].to_list() == c['pair_id'].to_list() and p.height == 112540
replay = np.load(O/'replay_first1024.npz')['p']
first = p.head(1024).select('p_none','p_dt','p_sp','p_ci').to_numpy()
exact = np.array_equal(replay, first)
assert exact
risk = 1-p['p_none'].to_numpy(); cr = c['risk_score'].to_numpy()
probs = p.select('p_dt','p_sp','p_ci').to_numpy()
labels = np.array(['directed_transfer','soft_play','coordinated_isolation'])
route = labels[np.argmax(probs,axis=1)]; curroute = c['predicted_behavior'].to_numpy()
oldblock = set(pl.read_csv(STEP).filter(pl.col('risk_score')>=.5)['pair_id'].to_list())
outside = np.array([x not in oldblock for x in p['pair_id']])
def tops(x,n): return set(np.array(p['pair_id'])[np.argsort(-x,kind='stable')[:n]])
over = {str(n):len(tops(risk,n)&tops(cr,n)) for n in (100,1000,5000)}
bands = [('ge_.99',cr>=.99),('.9_.99',(cr>=.9)&(cr<.99)),('.5_.9',(cr>=.5)&(cr<.9)),('.1_.5',(cr>=.1)&(cr<.5)),('lt_.1',cr<.1)]
bd = {k:{'n':int(m.sum()),'pf_risk_mean':float(risk[m].mean()),'pf_risk_q50':float(np.median(risk[m])),'route_disagreement':float((route[m]!=curroute[m]).mean())} for k,m in bands}
out = {'rows':len(risk),'fresh_replay_exact':bool(exact),'fresh_replay_max_abs':float(np.max(np.abs(replay-first))),'spearman_pf_vs_current':float(spearmanr(risk,cr).correlation),'top_overlap_counts':over,'pf_risk_quantiles':{str(q):float(np.quantile(risk,q)) for q in (0,.01,.1,.5,.9,.99,1)},'pf_route_counts':{x:int((route==x).sum()) for x in labels},'current_route_counts':{x:int((curroute==x).sum()) for x in labels},'route_disagreement_rate':float((route!=curroute).mean()),'current_risk_bands':bd,'old_block_size':len(oldblock),'outside_old_block':{'rows':int(outside.sum()),'top100_pf_outside':int(outside[np.argsort(-risk)[:100]].sum()),'top1000_pf_outside':int(outside[np.argsort(-risk)[:1000]].sum()),'top5000_pf_outside':int(outside[np.argsort(-risk)[:5000]].sum())},'warning':'All diagnostics are label-free prediction disagreement, not evaluation truth or measured recall.'}
(O/'diagnostics.json').write_text(json.dumps(out,indent=2))
pl.DataFrame({'pair_id':p['pair_id'],'pf_risk':risk,'pf_route':route,'current_risk':cr,'current_route':curroute,'outside_old909':outside}).filter(pl.col('outside_old909')).sort('pf_risk',descending=True).head(500).write_parquet(O/'top500_outside_old_block.parquet')
