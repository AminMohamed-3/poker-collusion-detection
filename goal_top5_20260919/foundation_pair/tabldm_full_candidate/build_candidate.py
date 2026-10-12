import argparse,csv,hashlib,io,json,os,sys
from pathlib import Path
H=Path(__file__).resolve().parent;G=H.parents[1];P=H.parents[2];sys.path.insert(0,str(P/'.venv/Lib/site-packages'))
import numpy as np,polars as pl
from scipy.stats import rankdata
BASE=G/'outputs/submission_sp277_ci276_on_step272.csv';FAST=H.parent/'full_fast/fast_probabilities.parquet';INC=H.parent/'full_fast/submission_fast_rank80_20_on_current.csv';TAB=H/'tabldm_probabilities.parquet';DEST=H/'submission_tabldm_equal_rank80_20_on_pre_fast.csv'
BASE_SHA='0493a807d81b7f3cc5ec54a74c86d08c362b72305fceee8e6af9044740990fc9';INC_SHA='e39785db83a521e7acf670844456abc16881ab02916b6b88b8e75b2e5c4c0906'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def r2(x):r=np.rint(2*rankdata(x,method='average')).astype(np.int64);assert np.allclose(r/2,rankdata(x,method='average'));return r
def render(rows,fields,score):
 b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=fields,lineterminator='\n');w.writeheader()
 for row,v in zip(rows,score):w.writerow({**row,'risk_score':format(float(v),'.17g')})
 text=b.getvalue();rest=list(csv.DictReader(io.StringIO(text)));assert len(rest)==len(rows) and np.array_equal(np.array([float(x['risk_score']) for x in rest]),score);fixed=[x for x in fields if x!='risk_score'];assert all(all(a[x]==b[x] for x in fixed) for a,b in zip(rows,rest));return text
def main():
 a=argparse.ArgumentParser();a.add_argument('--identity-test',action='store_true');a.add_argument('--build',action='store_true');z=a.parse_args();assert z.identity_test != z.build;man=json.loads((H/'MANIFEST.json').read_text())
 for n,v in man['inputs'].items():assert sha(G/n)==v,(n,'input drift')
 for n,v in man['code_files'].items():assert sha(H/n)==v,(n,'code drift')
 assert sha(BASE)==BASE_SHA and sha(INC)==INC_SHA
 with BASE.open(newline='',encoding='utf-8-sig') as f:q=csv.DictReader(f);fields=q.fieldnames;rows=list(q)
 keys=np.array([x['pair_id'] for x in rows]);assert len(keys)==112540 and len(np.unique(keys))==len(keys) and np.array_equal(keys,np.sort(keys));base=np.array([float(x['risk_score']) for x in rows]);fp=pl.read_parquet(FAST);assert np.array_equal(fp['pair_id'].to_numpy(),keys);pf=fp.select('p_none','p_dt','p_sp','p_ci').to_numpy().astype(np.float64);assert np.isfinite(pf).all() and (pf>=0).all() and (pf<=1).all() and np.allclose(pf.sum(1),1,atol=1e-5);rb=r2(base);rf=r2(1-pf[:,0]);N=len(keys)
 if z.identity_test:
  assert not (H/'IDENTITY_RECEIPT.json').exists();score=(4*rb+r2(rf))/(10*N);text=render(rows,fields,score);actual=hashlib.sha256(text.encode()).hexdigest();assert actual==INC_SHA and text.encode()==INC.read_bytes();(H/'IDENTITY_RECEIPT.json').write_text(json.dumps({'byte_exact':True,'sha256':actual,'rows':N,'base_sha256':BASE_SHA,'fast_sha256':sha(FAST),'incumbent_sha256':INC_SHA},indent=2)+'\n');return
 assert not DEST.exists() and not (H/'candidate_receipt.json').exists();release=json.loads((H/'BUILD_RELEASE.json').read_text()) if (H/'BUILD_RELEASE.json').exists() else {};assert release.get('manifest_sha256')==sha(H/'MANIFEST.json') and release.get('tabldm_probabilities_sha256')==sha(TAB),'root build release absent';tp=pl.read_parquet(TAB);assert np.array_equal(tp['pair_id'].to_numpy(),keys);pt=tp.select('p_none','p_dt','p_sp','p_ci').to_numpy().astype(np.float64);assert np.isfinite(pt).all() and (pt>=0).all() and (pt<=1).all() and np.allclose(pt.sum(1),1,atol=2e-6);rep=np.load(H/'replay_first1024.npz',allow_pickle=False);assert np.array_equal(rep['pair_id'],keys[:1024]) and np.array_equal(rep['p'],pt[:1024]);num=4*rb+r2(rf+r2(1-pt[:,0]));score=num/(10*N);assert len(np.unique(score))==len(np.unique(num));text=render(rows,fields,score);DEST.write_text(text,encoding='utf-8',newline='');rec={'candidate_sha256':sha(DEST),'rows':N,'formula':'(4*r2(pre-Fast .91411 base)+r2(r2(Fast risk)+r2(TabLDM risk)))/(10*N)','base_sha256':BASE_SHA,'fast_sha256':sha(FAST),'tabldm_sha256':sha(TAB),'replay_sha256':sha(H/'replay_first1024.npz'),'identity_receipt_sha256':sha(H/'IDENTITY_RECEIPT.json'),'behavior_evidence_exact':True,'float17g_roundtrip_exact':True,'upload_performed':False};(H/'candidate_receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
if __name__=='__main__':main()
