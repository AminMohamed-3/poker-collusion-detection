from pathlib import Path
import subprocess,json,base64,hashlib,sys,datetime
sys.stdout.reconfigure(encoding='utf-8')
base=Path(__file__).resolve().parent;dest=base/'returned';dest.mkdir(exist_ok=True)
remote=r"""from pathlib import Path
import json,base64,hashlib,subprocess
r=Path('/workspace/poker/codex/tabpfn_finetune_quality_v3name_20260913');run=r/'run';files={}
def add(p):
    b=p.read_bytes();assert len(b)<2000000
    files[str(p.relative_to(r))]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode(),'bytes':len(b)}
for p in [run/'synthetic/receipt.json',*sorted((run/'contexts').glob('*/receipt.json'))]:
    if not p.exists():continue
    z=json.loads(p.read_text());add(p)
    if (p.parent/'started.json').exists():add(p.parent/'started.json')
    for name in ['probabilities.npy','replay.json','replay_saved.npy','replay_refit.npy']:
        if (p.parent/name).exists():add(p.parent/name)
for p in [run/'baseline.json',run/'baseline.npy',run/'run-freeze.json',run/'complete.json',r/'study-freeze.json',r/'launch-receipt.json',r/'quality.log']:
    if p.exists():add(p)
q=subprocess.run(['supervisorctl','status','codex_tabpfn_finetune_quality_v3name'],capture_output=True,text=True)
print(json.dumps({'supervisor':q.stdout.strip(),'complete':(run/'complete.json').exists(),'files':files}))
"""
q=subprocess.run(['ssh','-T','-i','C:/Users/USER/.ssh/codex_kaggle_vm_ed25519','-p','30281','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid','/venv/main/bin/python -'],input=remote,capture_output=True,text=True,encoding='utf-8',timeout=45)
assert q.returncode==0,q.stderr
snap=json.loads(q.stdout);manifest={}
for rel,m in snap['files'].items():
    p=(dest/rel).resolve();assert p.is_relative_to(dest.resolve())
    b=base64.b64decode(m['base64']);assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256']
    if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'],rel
    else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    manifest[rel]={k:v for k,v in m.items() if k!='base64'}
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor':snap['supervisor'],'remote_complete':snap['complete'],'files':manifest,'full_weights_backed_up':False}
(base/'small-backup-latest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'supervisor':snap['supervisor'],'complete':snap['complete'],'verified_small_files':len(manifest),'contexts':sum(n.startswith('run/contexts/') and n.endswith('/receipt.json') for n in manifest),'bytes':sum(v['bytes'] for v in manifest.values())}))
