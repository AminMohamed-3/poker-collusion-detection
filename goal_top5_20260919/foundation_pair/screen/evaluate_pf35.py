import sys,site,json
from pathlib import Path
site.addsitedir(r'C:\Data\Projects\Money Making\work\poker\.venv\Lib\site-packages')
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
ROOT=Path(r'C:\Data\Projects\Money Making'); OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/screen'
def ap(y,s): return float(average_precision_score(y,s))
def met(y,risk,route):
 bm=float(np.mean([ap(y==k,np.where(route==k,risk,0)) for k in (1,2,3)])); pa=ap(y>0,risk)
 return {'n':len(y),'positives':int((y>0).sum()),'pair_ap':pa,'behavior_map_predicted':bm,'pair_behavior':.7*pa+.1*bm}
def rk(x): return rankdata(x,method='average')/len(x)
def main():
 pf=np.load(OUT/'pf35_predictions.npz'); tr=np.load(OUT/'tree_oof_predictions.npz'); rows=[]; known=[]
 for f in (0,1):
  e=np.load(OUT/f'evaluator_fold{f}.npz'); y=e['y']; pp=pf[f'known_fold{f}']; lp=tr['lgb128'][tr['fold']==f]
  assert np.array_equal(e['pair_id'],tr['pair_id'][tr['fold']==f])
  lr=1-lp[:,0]; pr=1-pp[:,0]; route=1+np.argmax(lp[:,1:],axis=1); blend=.8*rk(lr)+.2*rk(pr)
  known.append((y,lr,pr,route,1+np.argmax(pp[:,1:],axis=1),pp,lp))
  rows.append({'context':f'known_fold{f}','LGB128':met(y,lr,route),'PF35_own_route':met(y,pr,1+np.argmax(pp[:,1:],axis=1)),'primary_blend_LGB_route':met(y,blend,route),'equal_probability_diagnostic':met(y,1-(.5*lp+.5*pp)[:,0],1+np.argmax((.5*lp+.5*pp)[:,1:],axis=1))})
  pe=np.load(OUT/f'evaluator_fold{f}_prefix60.npz'); ppf=pf[f'prefix60_fold{f}']; rows.append({'context':f'prefix60_fold{f}','PF35_own_route':met(pe['y'],1-ppf[:,0],1+np.argmax(ppf[:,1:],axis=1))})
  for omit in (1,2,3):
   oe=np.load(OUT/f'evaluator_fold{f}_omit{omit}.npz'); po=pf[f'omit{omit}_fold{f}']; rows.append({'context':f'omit{omit}_fold{f}','n':len(oe['y']),'positives':int((oe['y']==omit).sum()),'omitted_vs_confirmed_negative_ap':ap(oe['y']==omit,1-po[:,0]),'known_family_behavior_only_note':'omitted class probability fixed zero; no omitted BehaviorMAP claim'})
  sp=pf[f'shuffle_fold{f}']; rows.append({'context':f'shuffle_fold{f}','PF35_own_route':met(y,1-sp[:,0],1+np.argmax(sp[:,1:],axis=1))})
 y=np.concatenate([z[0] for z in known]); lr=np.concatenate([z[1] for z in known]); pr=np.concatenate([z[2] for z in known]); route=np.concatenate([z[3] for z in known]); routep=np.concatenate([z[4] for z in known]); blend=.8*rk(lr)+.2*rk(pr)
 result={'scope':'candidate-generation screen only; no STEP272 or public-gain claim','pooled_known':{'LGB128':met(y,lr,route),'PF35_own_route':met(y,pr,routep),'primary_blend_LGB_route':met(y,blend,route)},'contexts':rows}
 (OUT/'pf35_results.json').write_text(json.dumps(result,indent=2))
 r=np.load(OUT/'replay_fold0.npz'); exact=np.array_equal(r['p'],pf['known_fold0']); (OUT/'replay_check.json').write_text(json.dumps({'fresh_process_exact':bool(exact),'max_abs':float(np.max(np.abs(r['p']-pf['known_fold0'])))},indent=2)); assert exact
if __name__=='__main__': main()
