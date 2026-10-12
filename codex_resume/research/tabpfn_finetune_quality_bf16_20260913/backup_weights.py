"""Resume finalized weight/context files; three independent SFTP streams."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,subprocess,sys,time,datetime
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(__file__).resolve().parent;DEST=BASE/'returned';REMOTE='/workspace/poker/codex/tabpfn_finetune_quality_bf16_20260913'
KEY='C:/Users/USER/.ssh/codex_kaggle_vm_ed25519'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def download(rel,digest):
    p=(DEST/rel).resolve();assert p.is_relative_to(DEST.resolve());p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and sha(p)==digest:return rel,digest
    command=['sftp','-q','-b','-','-i',KEY,'-P','30281','-R','64','-B','262144','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid']
    batch='reget "'+REMOTE+'/'+rel+'" "'+p.as_posix()+'"\n'
    result=subprocess.run(command,input=batch,capture_output=True,text=True,encoding='utf-8',timeout=900)
    assert result.returncode==0,(rel,result.stderr[-1000:])
    assert sha(p)==digest,rel
    return rel,digest
start=time.monotonic();futures={};verified={};failures={}
with ThreadPoolExecutor(max_workers=3) as pool:
    while time.monotonic()-start<4500:
        for rp in [DEST/'run/synthetic/receipt.json',*sorted((DEST/'run/contexts').glob('*/receipt.json'))]:
            if not rp.exists():continue
            r=json.loads(rp.read_text(encoding='utf-8'))
            if r['status']!='passed':continue
            for name,key in [('tabpfn-v3-classifier-finetuned-epoch10.ckpt','weights_sha256'),('matched_context.tabpfn_fit','fit_state_sha256')]:
                rel=str((rp.parent/name).relative_to(DEST)).replace('\\','/')
                if rel not in futures and rel not in verified and rel not in failures:futures[rel]=pool.submit(download,rel,r[key])
        changed=False
        for rel,f in list(futures.items()):
            if not f.done():continue
            try:n,h=f.result();verified[n]=h
            except Exception as e:failures[rel]=repr(e)
            del futures[rel];changed=True
        state={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':verified,'failures':failures,'queued_or_active':len(futures),'elapsed_seconds':time.monotonic()-start}
        (BASE/'weight-backup-progress.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        if changed:print(json.dumps({'verified':len(verified),'failures':len(failures),'queued_or_active':len(futures)}),flush=True)
        if (DEST/'run/complete.json').exists() and not futures:
            assert len(verified)==92 and not failures
            (BASE/'weight-backup-complete.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print('ALL_46_WEIGHT_AND_CONTEXT_FILES_VERIFIED',flush=True);break
        if failures and not futures:break
        time.sleep(10)
