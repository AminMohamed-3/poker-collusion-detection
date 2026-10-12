"""Portable bounded batch worker for the predeclared multiway experiment."""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import polars as pl
from phevaluator import Card
from multiway_evidence_features import FEATURES,HandShares,extract_pair_hand

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def batch(args):
    root,index=args; root=Path(root); started=time.perf_counter(); name=f'batch_{index:03}'; out=root/'results'; out.mkdir(exist_ok=True)
    receipt=out/(name+'.json')
    if receipt.exists():
        old=json.loads(receipt.read_text()); assert old['config_sha256']==sha(root/'config.json') and old['output_sha256']==sha(out/(name+'.parquet'))
        return old
    keys=pl.read_parquet(root/'keys'/(name+'.parquet')); ids=keys.select('hand_id').unique()
    hands=pl.read_parquet(root/'hands.parquet').join(ids,on='hand_id',how='semi').sort('hand_id')
    seats=pl.read_parquet(root/'seats.parquet').join(ids,on='hand_id',how='semi')
    actions=pl.read_parquet(root/'actions.parquet').join(ids,on='hand_id',how='semi')
    handpairs=defaultdict(list); handseats=defaultdict(dict); handactions=defaultdict(list)
    for p,h,a,b in keys.select('pair_id','hand_id','player_1','player_2').iter_rows(): handpairs[h].append((p,a,b))
    for h,p,a,b in seats.iter_rows(): handseats[h][p]=(Card.to_id(a),Card.to_id(b))
    for a in actions.iter_rows(named=True): handactions[a['hand_id']].append(a)
    rows=[]
    for count,(h,blind,board) in enumerate(hands.iter_rows(),1):
        board=[Card.to_id(c) for c in (board or '').split()]; engine=HandShares(handseats[h],board)
        for pair,a,b in handpairs[h]: rows.append((pair,h,*extract_pair_hand(a,b,blind,handseats[h],board,handactions[h],engine)))
        if count%1000==0: print(json.dumps({'batch':index,'hands':count,'total':hands.height,'seconds':time.perf_counter()-started}),flush=True)
    result=pl.DataFrame(rows,schema=['pair_id','hand_id',*FEATURES],orient='row').with_columns(pl.col(FEATURES).cast(pl.Float32)).sort('pair_id','hand_id')
    assert result.height==keys.height and result.select(pl.struct('pair_id','hand_id').n_unique()).item()==keys.height
    path=out/(name+'.parquet'); result.write_parquet(path)
    payload={'batch':index,'config_sha256':sha(root/'config.json'),'rows':result.height,'hands':hands.height,'seconds':time.perf_counter()-started,'output_sha256':sha(path)}
    receipt.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload),flush=True); return payload

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--jobs',type=int,default=1); parser.add_argument('--batches',type=int,nargs='+',default=list(range(8))); args=parser.parse_args()
    root=args.root.resolve(); plan=json.loads((root/'config.json').read_text())
    assert all(sha(root/p)==h for p,h in plan['files'].items())
    work=[(str(root),i) for i in args.batches]
    if args.jobs==1: results=[batch(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool: results=list(pool.map(batch,work))
    print(json.dumps({'completed':len(results),'rows':sum(r['rows'] for r in results)}),flush=True)

if __name__=='__main__': main()
