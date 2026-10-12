"""One metrics pass for frozen TabPFN-3 confirmation predictions."""
from audit_common import *
from evidence_aug.paths import ORIGINAL
from evidence_aug.step5_evaluate import per_pair_map,pool_bootstrap
from datetime import datetime,timezone
OUT=HERE/'experiments/tabpfn3_confirmation_20260912';PRIMARY='tabpfn3_default'
def summary(q):
 d=dict(q.select('pair_id','delta').iter_rows());p=dict(q.select('pair_id','table_id').iter_rows());return {'n_pairs':q.height,'mean_delta':float(q['delta'].mean()),'ci95':pool_bootstrap(d,p),'by_family':{f:float(q.filter(pl.col('family')==f)['delta'].mean()) for f in FAMILIES},'halves':{str(h):float(q.filter(pl.col('table_id').map_elements(half,return_dtype=pl.Int64)==h)['delta'].mean()) for h in (0,1)}}
def main():
 assert not (OUT/'results.json').exists();plan=json.loads((OUT/'plan.json').read_text());ctx=json.loads((OUT/'context-receipt.json').read_text());pred=json.loads((OUT/'predictions-frozen.json').read_text());assert pred['plan_sha256']==sha(OUT/'plan.json') and pred['context_receipt_sha256']==sha(OUT/'context-receipt.json')
 for p,h in plan['sources'].items():assert sha(p)==h
 assert ctx['plan_sha256']==sha(OUT/'plan.json') and ctx['protected_unchanged']
 for p,h in ctx['context_hashes'].items():assert sha(p)==h
 for p,h in ctx['artifact_hashes'].items():assert sha(p)==h
 assert pred['freeze_sha256']==sha(OUT/'freeze.json') and pred['replay_sha256']==sha(OUT/'replay.json') and pred['fit_complete_sha256']==sha(OUT/'fit-complete.json')
 assert len(pred['scores'])==9
 for n,h in pred['scores'].items():assert sha(OUT/'scores'/n)==h
 evidence=pl.read_csv(ORIGINAL/'data/development_evidence.csv').select('pair_id','hand_id');truths={k[0]:set(g['hand_id']) for k,g in evidence.group_by('pair_id')};families=dict(pl.read_csv(ORIGINAL/'data/development_labels.csv').select('pair_id','behavior_family').iter_rows());rows=[]
 for fold in (0,1,2):
  ref=pl.read_parquet(OUT/'scores'/f'fold{fold}-incumbent.parquet');base=per_pair_map(ref,truths);tables=dict(ref.select('pair_id','table_id').unique().iter_rows());roles=dict(pl.read_csv(OUT/f'fold{fold}-roles.csv').iter_rows())
  for arm in (PRIMARY,'stage1_only'):
   armframe=pl.read_parquet(OUT/'scores'/f'fold{fold}-{arm}.parquet');assert ref.select('pair_id','hand_id','table_id').equals(armframe.select('pair_id','hand_id','table_id'));m=per_pair_map(armframe,truths)
   rows.extend({'fold':fold,'arm':arm,'pair_id':x,'table_id':tables[x],'family':families[x],'predicted':roles[x],'base_map5':base[x],'map5':m[x],'delta':m[x]-base[x]} for x in base)
 frame=pl.DataFrame(rows);frame.write_parquet(OUT/'pair_metrics.parquet');res={'evaluated_at':datetime.now(timezone.utc).isoformat(),'reused_outer':True,'comparisons':{},'protected_unchanged':check_protected()==plan['protected']}
 for arm in (PRIMARY,'stage1_only'):
  q=frame.filter(pl.col('arm')==arm);byfold={str(f):summary(q.filter(pl.col('fold')==f)) for f in (0,1,2)};u=q.group_by('pair_id','table_id','family').agg(pl.col('delta').mean(),pl.col('base_map5').mean(),pl.col('map5').mean());assert u.height==210 and u['table_id'].n_unique()==141 and q.height==284
  r=summary(u);r['unique_pair_base_map5']=float(u['base_map5'].mean());r['unique_pair_map5']=float(u['map5'].mean());r.update({'unique_pool_count':u['table_id'].n_unique(),'fold_appearance_count':q.height,'appearance_weighted_delta':float(q['delta'].mean()),'by_fold':byfold});checks={'mean_delta_min':r['mean_delta']>=.01,'ci_lower_positive':r['ci95'][0]>0,'all_fold_nonnegative':min(x['mean_delta'] for x in byfold.values())>=0,'family_floor':min(r['by_family'].values())>=-.01,'every_fold_family_floor':min(x for y in byfold.values() for x in y['by_family'].values())>=-.01,'halves_nonnegative':min(r['halves'].values())>=0};r['checks']=checks;r['gate_pass']=all(checks.values());res['comparisons'][arm]=r
 res['passed']=res['comparisons'][PRIMARY]['gate_pass'];res['negative_control_rejected']=not res['comparisons']['stage1_only']['gate_pass'];assert res['protected_unchanged'];write(OUT/'results.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
