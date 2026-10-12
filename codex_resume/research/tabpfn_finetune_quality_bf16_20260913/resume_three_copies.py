from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,subprocess,hashlib,time
R=Path(__file__).resolve().parent;D=R/'returned';audit=json.loads((R/'final-local-weight-audit.json').read_text());assert len(audit['verified'])==89 and len(audit['failures'])==3
KEY='C:/Users/USER/.ssh/codex_kaggle_vm_ed25519';remote='/workspace/poker/codex/tabpfn_finetune_quality_bf16_20260913'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def resume(item):
 rel,v=item;p=(D/rel).resolve();assert p.is_relative_to(D.resolve());t=time.monotonic()
 cmd=['sftp','-q','-b','-','-i',KEY,'-P','30281','-R','64','-B','262144','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid']
 q=subprocess.run(cmd,input='reget "'+remote+'/'+rel+'" "'+p.as_posix()+'"\n',capture_output=True,text=True,timeout=900)
 assert q.returncode==0,(rel,q.stderr[-1000:]);assert sha(p)==v['expected'],rel
 print(json.dumps({'repaired':rel,'seconds':time.monotonic()-t}),flush=True);return rel,v['expected']
with ThreadPoolExecutor(max_workers=2) as pool:repaired=dict(pool.map(resume,audit['failures'].items()))
assert len(repaired)==3
j={'observed_utc':datetime.now(timezone.utc).isoformat(),'verified':{**audit['verified'],**repaired},'failures':{},'queued_or_active':0,'prior_verified_audit':'final-local-weight-audit.json','repaired_files':list(repaired),'complete':True}
assert len(j['verified'])==92
(R/'weight-backup-complete.json').write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8');print('ALL92_FILES_VERIFIED',flush=True)
