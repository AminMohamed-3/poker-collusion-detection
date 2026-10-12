"""Independent final artifact review. No fitting, metrics, upload or input mutation."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import polars as pl
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
P=HERE.parents[2]
G=P/'goal_top5_20260919'
H=G/'foundation_pair/tabldm_full_candidate'
BASE=G/'outputs/submission_sp277_ci276_on_step272.csv'
INC=G/'foundation_pair/full_fast/submission_fast_rank80_20_on_current.csv'
CAND=H/'submission_tabldm_equal_rank80_20_on_pre_fast.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f:
        d=csv.DictReader(f);return d.fieldnames,list(d)
def main():
    assert CAND.exists(),'Candidate not built; wait for owner.'
    assert sha(BASE)=='0493a807d81b7f3cc5ec54a74c86d08c362b72305fceee8e6af9044740990fc9'
    assert sha(INC)=='e39785db83a521e7acf670844456abc16881ab02916b6b88b8e75b2e5c4c0906'
    manifest=json.loads((H/'MANIFEST.json').read_text())
    assert all(sha(G/k)==v for k,v in manifest['inputs'].items())
    assert all(sha(H/k)==v for k,v in manifest['code_files'].items())
    fields,base=read(BASE);ifields,inc=read(INC);cfields,cand=read(CAND)
    assert fields==ifields==cfields and len(base)==len(inc)==len(cand)==112540
    keys=np.array([r['pair_id'] for r in base]);assert len(np.unique(keys))==len(keys)
    assert all(a['pair_id']==b['pair_id']==c['pair_id'] for a,b,c in zip(base,inc,cand))
    fixed=[k for k in fields if k!='risk_score']
    assert all(all(a[k]==b[k]==c[k] for k in fixed) for a,b,c in zip(base,inc,cand))
    probs=[]
    for path in [G/'foundation_pair/full_fast/fast_probabilities.parquet',H/'tabldm_probabilities.parquet']:
        frame=pl.read_parquet(path);assert np.array_equal(frame['pair_id'].to_numpy(),keys)
        pr=frame.select('p_none','p_dt','p_sp','p_ci').to_numpy().astype(np.float64)
        assert pr.shape==(112540,4) and np.isfinite(pr).all() and (pr>=0).all() and (pr<=1).all()
        assert np.allclose(pr.sum(1),1,atol=2e-6);probs.append(pr)
    fast,tab=probs
    runtime=json.loads((H/'runtime.json').read_text());replay=json.loads((H/'replay_receipt.json').read_text())
    assert runtime['probabilities_sha256']==replay['probabilities_sha256']==sha(H/'tabldm_probabilities.parquet')
    assert replay['replay_sha256']==sha(H/'replay_first1024.npz')
    r=np.load(H/'replay_first1024.npz',allow_pickle=False)
    assert np.array_equal(r['pair_id'],keys[:1024]) and np.array_equal(r['p'],tab[:1024])
    twice=lambda x:np.rint(2*rankdata(x,method='average')).astype(np.int64)
    rb=twice(np.array([float(v['risk_score']) for v in base]));rf=twice(1-fast[:,0]);rt=twice(1-tab[:,0])
    expected=(4*rb+twice(rf+rt))/(10*len(keys));actual=np.array([float(v['risk_score']) for v in cand])
    assert np.array_equal(actual,expected) and np.isfinite(actual).all() and (actual>=0).all() and (actual<=1).all()
    old=np.array([float(v['risk_score']) for v in inc]);assert np.array_equal(old,(4*rb+rf)/(10*len(keys)))
    receipt=json.loads((H/'candidate_receipt.json').read_text());assert receipt['candidate_sha256']==sha(CAND)
    result={'candidate_sha256':sha(CAND),'rows':len(keys),'global_formula_exact':True,'all_nonrisk_fields_unchanged':True,'protected_best_hash_unchanged':True,'replay1024_exact':True,'changed_risk_rows':int((old!=actual).sum()),'quality_evaluation':False,'submission':False}
    output=HERE/'tabldm_full_final_artifact_review.json';assert not output.exists()
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
