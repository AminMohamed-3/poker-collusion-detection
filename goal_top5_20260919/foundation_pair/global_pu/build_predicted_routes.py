import hashlib,json,os
from pathlib import Path
for k in ('POLARS_MAX_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='2'
import numpy as np,polars as pl
O=Path(__file__).resolve().parent;E=Path(r'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/pu_pairs_20260911')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parts=[];src={}
for f in (1,2):
 p=E/f'fold{f}-prefix60/scores.parquet';src[str(p)]=sha(p);parts.append(pl.read_parquet(p,columns=['pair_id','table_id','shared_hands','predicted_class']).filter(pl.col('shared_hands')>=38).with_columns(pl.lit(f).alias('fold')))
d=pl.concat(parts).unique('pair_id',keep='first',maintain_order=True);base=np.load(O/'final2_baseline.npz',allow_pickle=False);assert d.height==50761 and d['pair_id'].n_unique()==50761 and np.array_equal(d['pair_id'].to_numpy(),base['pair_id']) and np.array_equal(d['table_id'].to_numpy(),base['table_id']);route=d['predicted_class'].to_numpy().astype(np.int8);assert set(np.unique(route))<=set(range(4))
p=O/'predicted_routes.npz';np.savez_compressed(p,pair_id=d['pair_id'].to_numpy().astype('U'),table_id=d['table_id'].to_numpy().astype('U'),predicted_class=route,canonical_source_fold=d['fold'].to_numpy().astype(np.int8));rec={'rows':d.height,'canonical_rule':'fold1-first dedup after shared_hands>=38','class_map':{'0':'none','1':'directed_transfer','2':'soft_play','3':'coordinated_isolation'},'source_hashes':src,'final2_baseline_sha256':sha(O/'final2_baseline.npz'),'routes_sha256':sha(p),'key_order_exact':True,'table_order_exact':True,'counts':{str(int(v)):int((route==v).sum()) for v in np.unique(route)}};(O/'predicted_routes_receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
