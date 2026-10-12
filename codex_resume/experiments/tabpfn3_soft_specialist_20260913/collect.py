import json,subprocess,sys,hashlib,base64
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
remote=r"""from pathlib import Path
import json,base64,hashlib,subprocess,shutil,time
r=Path('/workspace/poker/codex/tabpfn3_soft_specialist_20260913');run=r/'run';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=subprocess.run(['supervisorctl','status','codex_tabpfn_soft_specialist'],capture_output=True,text=True)
files={}
for p in [r/'study-freeze.json']+list(run.rglob('*.json'))+list(run.rglob('*.npy')):
 if p.is_file():
  raw=p.read_bytes()
  if p.suffix=='.json':json.loads(raw)
  files[str(p.relative_to(r))]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
receipts={p.parent.name:json.loads(p.read_text()) for p in (run/'contexts').glob('*/receipt.json')}
done=(run/'complete.json').exists();integrity=None
if done:
 f=json.loads((r/'study-freeze.json').read_text())
 for n,d in f['files_sha256'].items():assert h(r/n)==d,n
 assert h(r/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==f['original_checkpoint_sha256']
 digests={}
 for stem,v in receipts.items():
  x=run/'contexts'/stem;z={'receipt_sha256':h(x/'receipt.json'),'weights_sha256':h(x/'tabpfn-v3-classifier-finetuned-epoch10.ckpt'),'fit_state_sha256':h(x/'matched_context.tabpfn_fit'),'probabilities_sha256':h(x/'probabilities.npy')}
  for k in ['weights_sha256','fit_state_sha256','probabilities_sha256']:assert z[k]==v[k]
  assert v['status']=='passed';digests[stem]=z
 assert len(digests)==3
 integrity={'checked_contexts':3,'all_live_artifact_hashes_match':True,'study_freeze_sha256':h(r/'study-freeze.json'),'contexts':digests,'supervisor':q.stdout.strip()}
print(json.dumps({'supervisor':q.stdout.strip(),'complete':done,'completed_contexts':len(receipts),'contexts':{k:{a:v.get(a) for a in ['status','seconds','peak_reserved_bytes','error']} for k,v in receipts.items()},'disk_free':shutil.disk_usage(r).free,'gpu':subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),'log_tail':(r/'quality.log').read_text()[-1400:],'files':files,'integrity':integrity,'observed_unix':time.time()}))
"""
q=subprocess.run(['ssh','-T','-i','C:/Users/USER/.ssh/codex_kaggle_vm_ed25519','-p','30281','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid','/venv/main/bin/python -'],input=remote,capture_output=True,text=True,encoding='utf-8',timeout=45)
assert q.returncode==0,(q.stdout,q.stderr)
d=json.loads(q.stdout);returned=OUT/'returned';returned.mkdir(exist_ok=True)
for name,v in d['files'].items():
 p=returned/name;assert p.resolve().is_relative_to(returned.resolve());raw=base64.b64decode(v['base64']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
 if p.exists():assert p.read_bytes()==raw,name
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
if d['integrity'] is not None:
 p=returned/'remote-final-integrity.json'
 if p.exists():
  old=json.loads(p.read_text());assert old['contexts']==d['integrity']['contexts']
 else:p.write_text(json.dumps(d['integrity'],indent=2)+'\n',encoding='utf-8')
summary={k:v for k,v in d.items() if k not in ['files','integrity']};summary['locally_verified_small_files']=len(d['files']);summary['remote_final_integrity_pass']=d['integrity'] is not None
(OUT/'latest-observation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
