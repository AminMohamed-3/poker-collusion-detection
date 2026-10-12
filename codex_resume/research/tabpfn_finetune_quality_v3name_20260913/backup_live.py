"""Copy only finalized experiment artifacts from the authorized VM; no training changes."""
from pathlib import Path
import subprocess,json,hashlib,time,sys,re,datetime
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(__file__).resolve().parent;DEST=BASE/'returned';DEST.mkdir(exist_ok=True)
ROOT='/workspace/poker/codex/tabpfn_finetune_quality_v3name_20260913'
KEY='C:/Users/USER/.ssh/codex_kaggle_vm_ed25519';HOST='root@vm-host.invalid'
SSH=['ssh','-T','-i',KEY,'-p','30281','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'/venv/main/bin/python -']
SCP=['scp','-i',KEY,'-P','30281','-o','BatchMode=yes','-o','ConnectTimeout=15']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
remote=r"""from pathlib import Path
import json,hashlib,subprocess
r=Path('/workspace/poker/codex/tabpfn_finetune_quality_v3name_20260913');run=r/'run'
q=subprocess.run(['supervisorctl','status','codex_tabpfn_finetune_quality_v3name'],capture_output=True,text=True)
manifest={}
def add(p,h=None):
    assert p.is_file() and p.is_relative_to(r)
    manifest[str(p.relative_to(r))]={'sha256':h or hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
for p in [run/'synthetic/receipt.json',*sorted((run/'contexts').glob('*/receipt.json'))]:
    z=json.loads(p.read_text());add(p)
    if z['status']=='passed':
        for name,key in [('tabpfn-v3-classifier-finetuned-epoch10.ckpt','weights_sha256'),('matched_context.tabpfn_fit','fit_state_sha256'),('probabilities.npy','probabilities_sha256')]:add(p.parent/name,z[key])
        add(p.parent/'started.json')
    for extra in ['replay.json','replay_saved.npy','replay_refit.npy']:
        if (p.parent/extra).exists():add(p.parent/extra)
if (run/'complete.json').exists() or 'RUNNING' not in q.stdout:
    for p in run.iterdir():
        if p.is_file():add(p)
    for name in ['study-freeze.json','launch-receipt.json','quality.log']:add(r/name)
print(json.dumps({'supervisor':q.stdout.strip(),'complete':(run/'complete.json').exists(),'files':manifest}))
"""
verified={};started=time.monotonic();last_count=-1
while time.monotonic()-started<4500:
    q=subprocess.run(SSH,input=remote,capture_output=True,text=True,encoding='utf-8',timeout=45)
    if q.returncode:
        print(json.dumps({'observation_error':q.returncode,'retry_same_handle':True}),flush=True);time.sleep(20);continue
    snap=json.loads(q.stdout)
    for rel,m in snap['files'].items():
        assert re.fullmatch(r'[A-Za-z0-9_./-]+',rel) and '..' not in Path(rel).parts
        target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if verified.get(rel)==m['sha256']:continue
        if target.exists() and target.stat().st_size==m['bytes'] and sha(target)==m['sha256']:
            verified[rel]=m['sha256'];continue
        cp=subprocess.run(SCP+[HOST+':'+ROOT+'/'+rel,str(target)],capture_output=True,text=True,encoding='utf-8',timeout=180)
        if cp.returncode:raise RuntimeError('artifact copy failed '+rel+'; preserve partial file, revalidate before retry')
        assert target.stat().st_size==m['bytes'] and sha(target)==m['sha256'],rel
        verified[rel]=m['sha256']
    contexts=sum(1 for n in verified if n.startswith('run/contexts/') and n.endswith('/receipt.json'))
    status={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_state':snap['supervisor'],'remote_complete':snap['complete'],'verified_files':len(verified),'verified_contexts':contexts,'total_contexts':45,'files_sha256':verified,'seconds':time.monotonic()-started}
    (BASE/'backup-progress.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    if contexts!=last_count:print(json.dumps({k:v for k,v in status.items() if k!='files_sha256'}),flush=True);last_count=contexts
    if snap['complete']:
        assert contexts==45
        (BASE/'backup-complete.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        print('ALL_FINAL_ARTIFACTS_VERIFIED',flush=True);break
    if 'RUNNING' not in snap['supervisor']:
        print('REMOTE_TERMINAL_WITHOUT_COMPLETE',flush=True);break
    time.sleep(15)
else:raise TimeoutError('Backup observer cap reached; job not restarted or declared terminal')
