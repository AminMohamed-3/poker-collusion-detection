"""Add three random history views to class-matched training pairs only."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,sys,hashlib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import thin,aggregate,LABELS
from tail_pair_experiment import tails
from conditional_inference_worker import pair_aggregate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((H/'study-plan.json').read_text());rng=np.random.default_rng(109271)
    meta=json.loads((H.parent/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];m=pl.DataFrame(meta)
    selected=[]
    for label,count in plan['added_pair_counts'].items():
        indices=np.flatnonzero(np.array([r['behavior_family'] for r in meta])==label)
        selected.extend(rng.choice(indices,count,replace=False).tolist())
    chosen=m[selected].sort('pair_id');ids=chosen.select('pair_id');keys=chosen['pair_id'].to_list();idx={r['pair_id']:i for i,r in enumerate(meta)};indices=np.array([idx[k] for k in keys])
    np.save(H/'selected-original-indices.npy',indices)
    assert chosen.height==sum(plan['added_pair_counts'].values())
    raw=pl.read_parquet(sorted((R/'outputs_v2/hands_development').glob('batch_*.parquet'))).join(ids,on='pair_id',how='semi')
    ca=pl.read_parquet(R/'reports/conditional-action-experiment/policy-screen/hand-features.parquet').join(ids,on='pair_id',how='semi')
    cb=pl.read_parquet(R/'reports/conditional-bucket-experiment/policy-screen/hand-features.parquet').join(ids,on='pair_id',how='semi')
    cols=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']];events=[c for c in cols if c.startswith('v2_')];caf=[c for c in ca.columns if c.startswith('ca_')];cbf=[c for c in cb.columns if c.startswith('cb_')]
    # Reconstruct selected full histories and require exact agreement with the frozen inputs.
    names=json.loads((H.parent/'history_realism_20260910/persistent_sessions/features/metadata.json').read_text())['features']
    def build(rr,cc,bb):return aggregate(rr,cols).join(tails(rr,events),on=['pair_id','table_id']).join(pair_aggregate(cc,caf),on=['pair_id','table_id']).join(pair_aggregate(bb,cbf),on=['pair_id','table_id']).sort('pair_id')
    full=build(raw,ca,cb);base=np.load(H.parent/'data_expansion/augmentation/synthetic/tree-arrays.npz')
    np.testing.assert_allclose(full.select(names).to_numpy().astype(np.float32),base['train_pair'][indices],rtol=1e-5,atol=1e-6)
    ev=pl.read_csv(R/'data/development_evidence.csv').select('pair_id','hand_id').unique().join(ids,on='pair_id',how='semi')
    positives=chosen.filter(pl.col('behavior_family')!='none').select('pair_id');rows=[];receipts=[]
    for frac,seed in zip(plan['real_history_views']['fractions'],plan['real_history_views']['seeds']):
        rr=thin(raw,frac,seed);kk=rr.select('pair_id','hand_id');supported=kk.join(ev,on=['pair_id','hand_id'],how='semi').select('pair_id').unique();fallback=positives.join(supported,on='pair_id',how='anti')
        if fallback.height:rr=pl.concat([rr.join(fallback,on='pair_id',how='anti'),raw.join(fallback,on='pair_id',how='semi')])
        kk=rr.select('pair_id','hand_id');cc=ca.join(kk,on=['pair_id','hand_id'],how='semi');bb=cb.join(kk,on=['pair_id','hand_id'],how='semi')
        x=build(rr,cc,bb);assert x['pair_id'].to_list()==keys
        rows.append(x.select(names).to_numpy().astype(np.float32));receipts.append({'fraction':frac,'seed':seed,'full_history_fallback_pairs':fallback['pair_id'].to_list(),'hands':rr.height})
    y=np.array([LABELS[v] for v in chosen['behavior_family']],dtype=np.int64)
    np.savez_compressed(H/'real-views.npz',x=np.concatenate(rows),y=np.tile(y,3))
    (H/'real-views.json').write_text(json.dumps({'selected_pairs':chosen.to_dicts(),'views':receipts,'full_reconstruction_agrees':True,'source_sha256':sha(__file__),'selected_indices_sha256':sha(H/'selected-original-indices.npy'),'arrays_sha256':sha(H/'real-views.npz'),'no_held_pool_used':True,'unique_original_pairs':len(keys),'extra_rows':3*len(keys)},indent=2));print('Built original history views.',flush=True)
if __name__=='__main__':main()
