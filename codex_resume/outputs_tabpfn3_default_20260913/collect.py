import json,subprocess,hashlib,base64,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');O=Path(__file__).resolve().parent
remote=r"""from pathlib import Path
import json,hashlib,base64,subprocess,shutil,time
r=Path('/workspace/poker/codex/tabpfn3_production_20260913');run=r/'run';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=subprocess.run(['supervisorctl','status','codex_tabpfn_production'],capture_output=True,text=True)
files={}
paths=[r/'production-freeze.json',r/'launch-receipt.json',*run.glob('*.json'),*run.glob('benchmark/*/*.json'),*run.glob('replay/*.json'),*run.glob('predictions/*/*/receipt.json'),*run.glob('predictions/*/*/probabilities.npy')]
for p in paths:
 if not p.is_file():continue
 raw=p.read_bytes()
 if p.suffix=='.json':json.loads(raw)
 files[str(p.relative_to(r))]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
done=(run/'complete.json').exists();integrity=None
if done:
 f=json.loads((r/'production-freeze.json').read_text())
 for n,d in f['files'].items():assert h(r/n)==d,n
 assert h(r/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==f['checkpoint_sha256']
 c=json.loads((run/'complete.json').read_text())
 for rel,d in c['receipts'].items():
  p=run/'predictions'/rel/'receipt.json';assert h(p)==d
  v=json.loads(p.read_text());assert h(p.parent/'probabilities.npy')==v['probabilities_sha256']
  for n,d in v['chunks'].items():assert h(p.parent/'chunks'/n)==d
 bench=json.loads((run/'benchmark.json').read_text());fit={}
 for fam,v in bench['families'].items():
  p=run/'benchmark'/fam/'context.tabpfn_fit';assert h(p)==v['fit_state_sha256'];fit[str(p.relative_to(r))]=h(p)
 integrity={'all_inputs_and_predictions_and_chunks_verified':True,'checkpoint_sha256':f['checkpoint_sha256'],'fit_states':fit,'complete_sha256':h(run/'complete.json')}
print(json.dumps({'supervisor':q.stdout.strip(),'complete':done,'completed':len(list(run.glob('predictions/*/*/receipt.json'))),'disk_free':shutil.disk_usage(r).free,'gpu':subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),'log_tail':(r/'production.log').read_text()[-2000:] if (r/'production.log').exists() else '', 'files':files,'integrity':integrity,'observed_unix':time.time()}))
"""
q=subprocess.run(['ssh','-T','-i','C:/Users/USER/.ssh/codex_kaggle_vm_ed25519','-p','30281','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid','/venv/main/bin/python -'],input=remote,capture_output=True,text=True,encoding='utf-8',timeout=60)
assert q.returncode==0,(q.stdout,q.stderr);j=json.loads(q.stdout);dest=O/'returned';dest.mkdir(exist_ok=True)
for rel,v in j['files'].items():
 p=dest/rel;assert p.resolve().is_relative_to(dest.resolve());raw=base64.b64decode(v['base64']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
 if p.exists():assert p.read_bytes()==raw,rel
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
if j['integrity']:
 p=dest/'remote-final-integrity.json'
 if p.exists():assert json.loads(p.read_text())==j['integrity']
 else:p.write_text(json.dumps(j['integrity'],indent=2)+'\n',encoding='utf-8')
summary={k:v for k,v in j.items() if k not in ['files','integrity']};summary['verified_small_files']=len(j['files']);summary['final_integrity_verified']=bool(j['integrity'])
(O/'latest-observation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
