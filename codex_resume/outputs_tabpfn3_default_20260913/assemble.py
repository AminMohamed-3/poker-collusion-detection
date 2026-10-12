"""Assemble only a completed, replayed and hash-verified pretrained production run."""
import os
os.environ['POLARS_MAX_THREADS']='2'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
O=Path(__file__).resolve().parent;HERE=O.parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE/'src'));import contestable_state_production as prod
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
D=O/'returned';run=D/'run';recipe=read(O/'vm-package/production-recipe.json');freeze=read(D/'production-freeze.json');done=read(run/'complete.json');integrity=read(D/'remote-final-integrity.json');plan=read(O/'plan.json')
assert sha(D/'production-freeze.json')==sha(O/'vm-package/production-freeze.json')==done['freeze_sha256']
assert integrity['complete_sha256']==sha(run/'complete.json') and integrity['all_inputs_and_predictions_and_chunks_verified']
assert read(run/'replay.json')['passed'] and read(run/'benchmark.json')['passed']
assert prod.check_protected()==plan['protected']
for n,h in plan['sources'].items():assert sha(n)==h,n
assert done['completed_batches']==60 and done['query_rows']==321200
rankings=[]
for batch in sorted(read(O/'queries-complete.json')['batch_receipts']):
 receipt=read(O/'queries'/batch/'receipt.json');keys=O/'queries'/batch/'keys.parquet';assert sha(keys)==receipt['files']['keys.parquet']
 frame=pl.read_parquet(keys);scores=np.full(frame.height,np.nan,dtype=np.float64);covered=np.zeros(frame.height,dtype=np.int64)
 for item in receipt['contexts']:
  fam=item['family'];rel=batch+'/'+fam;rp=run/'predictions'/rel/'receipt.json';assert sha(rp)==done['receipts'][rel]
  rec=read(rp);pp=rp.parent/'probabilities.npy';assert sha(pp)==rec['probabilities_sha256'] and rec['query_sha256']==item['sha256'] and rec['freeze_sha256']==done['freeze_sha256']
  qp=O/item['file'];assert sha(qp)==item['sha256']
  with np.load(qp,allow_pickle=False) as z:
   ix=z['indices'];assert np.array_equal(z['pair_ids'],frame['pair_id'].to_numpy()[ix]);assert np.all(frame['predicted_behavior'].to_numpy()[ix]==fam)
  probs=np.load(pp,allow_pickle=False);assert len(probs)==len(ix) and np.isfinite(probs).all()
  scores[ix]=probs+10;covered[ix]+=1
 assert np.all(covered==1) and np.isfinite(scores).all()
 assert frame.group_by('pair_id').len()['len'].min()==20
 ranking=prod.rank(frame,scores).select('pair_id',*[pl.col('hands').list.get(i).alias(c) for i,c in enumerate(prod.EVIDENCE_COLUMNS)])
 rankings.append(ranking)
ranks=pl.concat(rankings);assert ranks.height==ranks['pair_id'].n_unique()==16060
selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet').select('pair_id')
assert ranks.select('pair_id').sort('pair_id').equals(selected.sort('pair_id'))
source=HERE/'outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv'
assert sha(source)=='9d0f0d2f698a5976b6aeff818156a5e3373b52aba433d81d313eb99c4cb2c315'
lookup={r[0]:r[1:] for r in ranks.iter_rows()};raw=source.read_bytes();lines=raw.splitlines(keepends=True);new=[lines[0]]
for line in lines[1:]:
 fields=line.rstrip(b'\r\n').split(b',');assert len(fields)==8
 pid=fields[0].decode()
 if pid in lookup:
  ending=b'\r\n' if line.endswith(b'\r\n') else b'\n'
  fields[3:]=[s.encode() for s in lookup[pid]];new.append(b','.join(fields)+ending)
 else:new.append(line)
f=O/'submission_tabpfn3_default.csv';assert not f.exists();f.write_bytes(b''.join(new))
a=pl.read_csv(source,infer_schema=False);b=pl.read_csv(f,infer_schema=False);assert a.height==b.height==112540
assert a.select('pair_id','risk_score','predicted_behavior').equals(b.select('pair_id','risk_score','predicted_behavior'))
assert a.join(selected,on='pair_id',how='anti').equals(b.join(selected,on='pair_id',how='anti'))
changes=int(np.any(a.select(prod.EVIDENCE_COLUMNS).to_numpy()!=b.select(prod.EVIDENCE_COLUMNS).to_numpy(),axis=1).sum())
sys.path.insert(0,str(ROOT/'src'));from validation import validate_submission
validation=validate_submission(f,ROOT/'data/evaluation_pairs.csv',ROOT/'data/hands.parquet',ROOT/'data/seats.parquet',ROOT/'data/sample_submission.csv').to_dict()
(O/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8');assert validation['valid']
assert prod.check_protected()==plan['protected']
result={'candidate_sha256':sha(f),'source_sha256':sha(source),'rows':112540,'reranked_pairs':16060,'changed_evidence_rows_vs_best':changes,'all_risk_and_behavior_unchanged':True,'all_outside_coverage_unchanged':True,'validation':validation,'production_freeze_sha256':done['freeze_sha256'],'production_complete_sha256':sha(run/'complete.json'),'assembler_sha256':sha(Path(__file__)),'submitted':False,'qualified_model_local_MAP_delta':0.014087301587301588,'public_gain_unproven':True}
(O/'build-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
