"""Verify saved evidence predictions without fitting or rereading truth."""
import json, hashlib
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
H=HERE.parents[2]/'goal_top5_20260919/foundation_pair/action_policy_evidence'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
K=['pair_id','hand_id']
def main():
    receipt=json.loads((H/'FIT_RECEIPT.json').read_text())
    assert receipt['manifest_sha256']==sha(H/'MANIFEST.json')
    for p,v in receipt['artifacts'].items():assert sha(H/p)==v
    audit=set(json.loads((H/'membership.json').read_text())['audit_tables'])
    for name in ['target','normal']:
        d=pl.read_parquet(H/f'train_membership_{name}.parquet')
        assert not audit.intersection(d['table_id'])
        w=d['fit_weight'].to_numpy();assert np.isfinite(w).all() and (w>0).all() and np.isclose(w.mean(),1)
        mass=d.group_by('pair_id','direction').agg(pl.sum('fit_weight'))['fit_weight'].to_numpy()
        assert np.allclose(mass,mass[0])
    actions=pl.read_parquet(H/'action_predictions.parquet')
    obs=[]
    for name in ['target','normal']:
        q=actions.select([f'{name}_p{k}' for k in range(4)]).to_numpy()
        assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-6)
        obs.append(q[np.arange(len(q)),actions['action_class'].to_numpy()])
    gain=np.log(np.maximum(obs[0],1e-12))-np.log(np.maximum(obs[1],1e-12))
    assert np.array_equal(gain,actions['gain'].to_numpy())
    rebuilt=actions.select(K+['direction']).with_columns(pl.Series('gain',gain)).group_by(K+['direction']).agg(pl.mean('gain')).group_by(K).agg(pl.mean('gain')).sort(K)
    saved=pl.read_parquet(H/'hand_predictions.parquet').sort(K)
    assert rebuilt.select(K).equals(saved.select(K)) and np.allclose(rebuilt['gain'],saved['gain'],atol=1e-14,rtol=0)
    ref=pl.read_parquet(H/'reference.parquet').sort(K)
    d=pl.read_parquet(H/'predictions.parquet').sort(K)
    assert ref.equals(d.select(ref.columns)) and d.height==19325
    expected=d['current'].to_numpy().copy()
    for _,g in d.with_row_index('ix').filter(pl.col('is_top20')).group_by('pair_id'):
        assert g.height==20
        rb=np.rint(2*rankdata(g['current'].to_numpy())).astype(np.int64)
        rg=np.rint(2*rankdata(g['gain'].to_numpy())).astype(np.int64)
        expected[g['ix'].to_numpy()]=10+(9*rb+rg)/400
    assert np.array_equal(expected,d['candidate'].to_numpy())
    out=HERE/'action_policy_evidence_output_review.json';assert not out.exists()
    result={'rows':d.height,'top20_hands':saved.height,'audit_tables_excluded':104,'action_llr_exact':True,'hand_aggregation_verified':True,'rank_formula_exact':True,'outside_scores_unchanged':True,'quality_labels_read':False}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
