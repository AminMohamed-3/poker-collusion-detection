"""Wait on the named live job, verify completed artifacts, invoke frozen evaluation once."""
from pathlib import Path
import subprocess,json,sys,os,time,hashlib,datetime
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(__file__).resolve().parent;PROJECT=BASE.parents[1];RUN=BASE/'returned/run';PACKAGE=BASE/'code-package'
def write(n,d):(BASE/n).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
start=time.monotonic()
while time.monotonic()-start<3600:
    p=subprocess.run([sys.executable,str(BASE/'collect_small.py')],capture_output=True,text=True,encoding='utf-8',timeout=60)
    if p.returncode:
        write('collector-error.json',{'stderr':p.stderr,'stdout':p.stdout,'same_job_retained':True});time.sleep(20);continue
    snap=json.loads(p.stdout);print(json.dumps(snap),flush=True)
    if snap['complete']:break
    if 'RUNNING' not in snap['supervisor']:raise RuntimeError('Named job terminal without complete receipt; no evaluation')
    time.sleep(30)
else:raise TimeoutError('Observation cap reached; do not restart the VM job')
remote=r"""from pathlib import Path
import json,hashlib,subprocess
r=Path('/workspace/poker/codex/tabpfn_finetune_quality_bf16_20260913');run=r/'run'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
f=json.loads((r/'study-freeze.json').read_text());done=json.loads((run/'complete.json').read_text());ctx=json.loads((r/'context-receipt.json').read_text())
assert sha(r/'study-freeze.json')==done['study_freeze_sha256']=='d9cbe9c4a3653776e6200025f4a37da067c64134de655c9f22ebb4e914dd0e27'
for n,h in f['files_sha256'].items():assert sha(r/n)==h,n
assert sha(r/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==f['original_checkpoint_sha256']
checked={}
for m in ctx['contexts']:
    p=run/'contexts'/m['stem'];z=json.loads((p/'receipt.json').read_text())
    assert sha(p/'receipt.json')==done['context_receipts'][m['stem']]
    assert z['status']=='passed' and z['gradient_trace']['optimizer_step_calls']==10 and len(z['losses'])==10
    assert z['training_autocast_dtype']=='bfloat16' and z['gradient_scaling_enabled'] is False
    assert z['saved_state_equals_live'] and z['inference_config_matches_original'] and z['original_checkpoint_unchanged']
    for n,k in [('tabpfn-v3-classifier-finetuned-epoch10.ckpt','weights_sha256'),('matched_context.tabpfn_fit','fit_state_sha256'),('probabilities.npy','probabilities_sha256')]:assert sha(p/n)==z[k],str(p/n)
    checked[m['stem']]={'receipt_sha256':sha(p/'receipt.json'),'weights_sha256':z['weights_sha256'],'fit_state_sha256':z['fit_state_sha256'],'probabilities_sha256':z['probabilities_sha256']}
print(json.dumps({'checked_contexts':len(checked),'all_live_artifact_hashes_match':True,'study_freeze_sha256':sha(r/'study-freeze.json'),'contexts':checked,'supervisor':subprocess.run(['supervisorctl','status','codex_tabpfn_finetune_quality_bf16'],capture_output=True,text=True).stdout.strip()}))
"""
q=subprocess.run(['ssh','-T','-i','C:/Users/USER/.ssh/codex_kaggle_vm_ed25519','-p','30281','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid','/venv/main/bin/python -'],input=remote,capture_output=True,text=True,encoding='utf-8',timeout=120)
assert q.returncode==0,q.stderr
integrity=json.loads(q.stdout);assert integrity['checked_contexts']==45 and integrity['all_live_artifact_hashes_match'];write('remote-final-integrity.json',integrity)
script=PROJECT/'src/tabpfn3_finetune_evaluate.py';f=json.loads((PACKAGE/'study-freeze.json').read_text(encoding='utf-8'))
assert hashlib.sha256(script.read_bytes()).hexdigest()==f['files_sha256'][script.name]
assert not (PROJECT/'experiments/tabpfn3_finetune_20260913/evaluation-started.json').exists()
env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(PROJECT/'src'),str(PROJECT.parent/'Claude Version/src')]);env['PYTHONUTF8']='1'
p=subprocess.run([sys.executable,str(script),'--run',str(RUN),'--package',str(PACKAGE)],capture_output=True,text=True,encoding='utf-8',env=env,timeout=180)
write('evaluation-invocation-result.json',{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'full_weight_backup_may_still_be_running':True})
print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
