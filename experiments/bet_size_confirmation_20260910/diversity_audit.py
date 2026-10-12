"""Training-only descriptive coverage audit; never a utility gate or sample filter."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
H=Path(__file__).resolve().parent;E=H.parent;U=E/'bet_size_utility_20260910';N=E/'augmentation_diversity_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(indices,distances,n):
    counts=np.bincount(indices,minlength=n);p=counts[counts>0]/len(indices)
    return {'examples':len(indices),'distinct_nearest_training_targets':int((counts>0).sum()),'effective_nearest_targets':float(np.exp(-np.sum(p*np.log(p)))),'top5_neighbor_share':float(np.sort(counts)[-5:].sum()/len(indices)),'nearest_distance_q10_50_90':np.quantile(distances,[.1,.5,.9]).tolist(),'neighbor_counts':counts.tolist()}
def main():
    N.mkdir(exist_ok=False);plan={'at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'recipe':'Original training full-view765 nonlearned gameplay features only. Scale by max(IQR,.1*std,1e-6), center median, clip[-10,10]; fit16-component whitened PCA on1403 original training pairs. Compare nearest real-soft target coverage in this diagnostic embedding. Original reference excludes itself and all same-pool targets. Report all selected78 synthetic targets, by population and pooled.','warning':'Nearest-neighbor coverage depends on representation; it is not a proof of independence, realism or utility. Do not select/filter generated samples from this audit or change frozen gates. No outer-held data or scores.'};(N/'plan.json').write_text(json.dumps(plan,indent=2))
    f=json.loads((U/'trial/freeze.json').read_text());p=json.loads((U/'study-plan.json').read_text());meta=json.loads((E/'history_augmentation_20260910/analysis_package/work/poker/reports/mil-pair-experiment/fold0/metadata.json').read_text())['train_full']['pairs'];x=np.load(E/'data_expansion/augmentation/synthetic/tree-arrays.npz')['train_pair'][:1403,:765].astype(float)
    median=np.median(x,axis=0);iqr=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0);scale=np.maximum.reduce([iqr,.1*x.std(axis=0),np.full(765,1e-6)]);transform=lambda v:np.clip((v-median)/scale,-10,10);pca=PCA(16,whiten=True,svd_solver='full').fit(transform(x));z=pca.transform(transform(x));take=np.array([r['behavior_family']=='soft_play' for r in meta]);rz=z[take];real=[r for r in meta if r['behavior_family']=='soft_play'];groups=np.array([r['table_id'] for r in real]);dd=cdist(rz,rz);dd[groups[:,None]==groups[None,:]]=np.inf;ii=dd.argmin(axis=1);result={'original_leave_pool_out':stats(ii,dd[np.arange(len(ii)),ii],len(real))};zs=[]
    for batch in p['batches']:
        arm=batch['arms'][0];m=json.loads((U/arm/'features/metadata.json').read_text())['views'][0];mapping={r['pair_id']:j for j,r in enumerate(m)};selected=f['selected_target_pairs'][str(batch['batch'])];sx=np.load(U/arm/'features/synthetic-pairs.npz')['x'][[mapping[key] for key in selected],:765];sz=pca.transform(transform(sx));zs.append(sz);dd=cdist(sz,rz);ii=dd.argmin(axis=1);result[arm]=stats(ii,dd[np.arange(len(ii)),ii],len(real))
    sz=np.concatenate(zs);dd=cdist(sz,rz);ii=dd.argmin(axis=1);result['pooled']=stats(ii,dd[np.arange(len(ii)),ii],len(real));out={'training_only':True,'outer_scores_read':False,'results':result,'nearest_target_order':[r['pair_id'] for r in real],'explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'plan_sha256':sha(N/'plan.json'),'not_a_new_admission_gate':True};(N/'result.json').write_text(json.dumps(out,indent=2));np.savez_compressed(N/'embedding.npz',real=rz,synthetic=sz);print(json.dumps({k:{n:a for n,a in v.items() if n!='neighbor_counts'} for k,v in result.items()},indent=2))
if __name__=='__main__':main()
