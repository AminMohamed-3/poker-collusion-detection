"""Frozen full-hybrid evidence screen: ordered prefix versus two controls."""
from audit_common import *
from datetime import datetime,timezone
import time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
OUT=HERE/'experiments/within_hand_sequence_20260912';FDIR=HERE/'research/within_hand_sequence_20260912'
ARMS=['incumbent','query_only','shuffled','ordered']
def main():
 t=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'freeze.json').exists()
 cr=json.loads((FDIR/'corpus-receipt.json').read_text());cf=json.loads((FDIR/'corpus-freeze.json').read_text());tr=json.loads((FDIR/'training-receipt.json').read_text());tf=json.loads((FDIR/'training-freeze.json').read_text());fr=json.loads((FDIR/'features-receipt.json').read_text());sr=json.loads((FDIR/'shuffle-receipt.json').read_text())
 assert cr['status']==tr['status']==fr['status']==sr['status']=='complete' and tr['models']==15 and fr['pair_swap_exact']
 assert sha(FDIR/'corpus-freeze.json')==cr['corpus_freeze_sha256'] and sha(FDIR/'training-freeze.json')==tr['training_freeze_sha256']
 assert sha(FDIR/'resume-freeze.json')==tr['resume_freeze_sha256']
 resume=json.loads((FDIR/'resume-freeze.json').read_text());assert sha(HERE/'src/within_hand_train_resume.py')==resume['source_sha256']
 for path,digest in resume['reused_completed_model'].items():assert sha(path)==digest
 for obj,field in [(cf,'source_hashes'),(tf,'sources'),(fr,'source_hashes'),(sr,'source_hashes')]:
  for path,digest in obj[field].items():assert sha(path)==digest
 for obj in [cr,fr,sr]:
  for name,digest in obj['artifacts'].items():assert sha(FDIR/name)==digest
 for r in tr['receipts']:
  a,k=r['arm'],r['fold'];assert r['reload_exact'] and sha(FDIR/f'{a}-{k}.pt')==r['model_sha256'] and sha(FDIR/f'{a}-{k}-normalizer.npz')==r['normalizer_sha256'] and sha(FDIR/f'{a}-{k}-probabilities.npy')==r['probabilities_sha256']
 for design in tf['design']:
  k=design['fold'];assert sha(FDIR/f'train-{k}-indices.npy')==design['train_indices_sha256'] and not set(design['training_pools'])&set(cf['encoder_exclusions'][str(k)]) and not set(design['training_pools'])&set(cf['outer_pools'])
 keys=pl.read_parquet(FDIR/'target-keys.parquet').sort('pair_id','hand_id');fcols=fr['columns'];assert len(fcols)==26
 best=CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
 bestsha='6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b';assert sha(best)==bestsha;protected=check_protected()
 sources=[Path(__file__),HERE/'src/audit_common.py',HERE/'research/within_hand_sequence_protocol_20260912.md',*[FDIR/n for n in ['prefit-shuffle-correction.md','corpus-freeze.json','corpus-receipt.json','training-freeze.json','training-receipt.json','features-receipt.json','shuffle-receipt.json','resume-freeze.json']],*[FDIR/n for n in fr['artifacts']],PREV/'scores.parquet',PREV/'predictions-frozen.json',best]
 write(OUT/'freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'source_hashes':{str(p):sha(p) for p in sources},'arms':ARMS,'eligible_arms':['ordered'],'additional_gate':'ordered-minus-query_only and ordered-minus-shuffled pool bootstrap lower bounds both positive','parameters':dict(CLF,random_state=11),'fits':60,'appended_features':fcols,'original_outer_pools_excluded':True,'protected':protected,'limits':['Fixed current upstream cache is not fully nested.','Reused selection audit, not fresh confirmation.','Same fold-specific reference for both downstream train and held rows.','Only ordered eligible; no rescue or public/outer reads.']})
 d,shared,rcols,saved=load_current();expected=d.filter(pl.col('oof_rank')<=20).select('pair_id','hand_id');assert keys.equals(expected)
 extra={}
 for arm in ARMS[1:]:
  for k in range(5):
   f=pl.read_parquet(FDIR/f'{arm}-{k}-features.parquet').sort('pair_id','hand_id');assert f.select('pair_id','hand_id').equals(keys)
   z=d.select('pair_id','hand_id').join(f,on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','hand_id');assert z.select('pair_id','hand_id').equals(d.select('pair_id','hand_id'))
   extra[(arm,k)]=z.select(fcols).to_numpy()
 pred={n:d['oof'].to_numpy().copy()-1e6 for n in ARMS};fills={n:np.zeros(d.height,dtype=int) for n in ARMS};controls=[]
 for fam in FAMILIES:
  ix=np.flatnonzero(((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy());q=d[ix];cols=shared+(rcols if fam!='coordinated_isolation' else []);x=q.select(cols).to_numpy().astype(float);y=q['relevance'].to_numpy().astype(int);g=q['table_id'].to_numpy()
  for k,(train,test) in enumerate(GroupKFold(n_splits=5).split(x,y,g)):
   design=next(row for row in cf['detector_design'] if row['family']==fam and row['fold']==k);assert set(g[train])==set(design['training_pools']) and set(g[test])==set(design['held_pools']) and not set(g[test])&set(tf['design'][k]['training_pools'])
   for arm in ARMS:
    z=x if arm=='incumbent' else np.column_stack([x,extra[(arm,k)][ix]]);assert np.isfinite(z).all()
    model=lgb.LGBMClassifier(**CLF,random_state=11).fit(z[train],y[train]);p=model.predict_proba(z[test])[:,1];path=OUT/f'{arm}-{fam}-{k}.txt';model.booster_.save_model(str(path));rp=lgb.Booster(model_file=str(path)).predict(z[test],num_threads=2);assert np.array_equal(p,rp)
    c={'arm':arm,'family':fam,'fold':k,'reload_exact':True,'model_sha256':sha(path)}
    if arm=='incumbent':assert np.array_equal(p+10,saved['incumbent'].to_numpy()[ix[test]]);c['reference_exact']=True
    pred[arm][ix[test]]=p+10;fills[arm][ix[test]]+=1;controls.append(c)
  print('Completed',fam,round(time.perf_counter()-t,1),flush=True)
 top=(d['oof_rank']<=20).to_numpy();assert all(np.all(v[top]==1) and np.all(v[~top]==0) for v in fills.values());assert all(np.array_equal(v[~top],d['oof'].to_numpy()[~top]-1e6) for v in pred.values())
 pred['stage1_only']=d['oof'].to_numpy();scores=OUT/'scores.parquet';d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(n,v) for n,v in pred.items()]).write_parquet(scores)
 write(OUT/'predictions-frozen.json',{'freeze_sha256':sha(OUT/'freeze.json'),'scores_sha256':sha(scores),'controls':controls,'top20_all_once':True,'tail_exact':True,'outer_truth_read':False})
 result=evaluate(d,pred);result['contrasts']={f'ordered_minus_{a}':evaluate(d,{'incumbent':pred[a],'ordered':pred['ordered']})['comparisons']['ordered'] for a in ['query_only','shuffled']}
 passed=result['comparisons']['ordered']['gate_pass'] and all(c['ci95'][0]>0 for c in result['contrasts'].values());result['passed']=['ordered'] if passed else [];result['selected_for_confirmation']='ordered' if passed else None
 result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass'];assert result['negative_control_rejected']
 result['protected_unchanged']=check_protected()==protected and sha(best)==bestsha;assert result['protected_unchanged']
 for path,digest in json.loads((OUT/'freeze.json').read_text())['source_hashes'].items():assert sha(path)==digest
 result['seconds']=time.perf_counter()-t;result['no_outer_evidence_truth_read']=True;write(OUT/'results.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
