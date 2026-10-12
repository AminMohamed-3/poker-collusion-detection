"""Run a fixed serial confirmation batch on the private instance."""
import argparse
import json
from pathlib import Path
import subprocess
import time
root=Path(__file__).resolve().parents[1]; key=Path.home()/'.ssh/codex_kaggle_vm_ed25519'; options=['-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=15']
p=argparse.ArgumentParser(); p.add_argument('--jobs',nargs='+',required=True); args=p.parse_args()

def ssh(command):
    return subprocess.run(['ssh',*options,'-p','23794','root@vm-host.invalid',command],capture_output=True,text=True,check=True).stdout


for job in args.jobs:
    assert job.replace('_','').isalnum()
    dest=root/'reports/tabpfn-v2-experiment'/job; transfer=json.loads((dest/'transfer.json').read_text())
    subprocess.run(['scp','-O',*options,'-P','23794',str(dest/'package.tar.gz'),'root@vm-host.invalid:/workspace/poker-research/tabpfn-'+job+'.tar.gz'],check=True)
    print(ssh('/venv/main/bin/python /workspace/poker-research/deploy_tabpfn_worker.py --job '+job+' --sha256 '+transfer['sha256']),flush=True)
    print(ssh('supervisorctl start poker-tabpfn-'+job),flush=True)
    while True:
        time.sleep(15)
        r=subprocess.run(['ssh',*options,'-p','23794','root@vm-host.invalid','supervisorctl status poker-tabpfn-'+job],capture_output=True,text=True)
        if ' RUNNING ' in r.stdout: continue
        if ' EXITED ' in r.stdout: break
        raise RuntimeError('Unexpected worker status: '+r.stdout+r.stderr)
    subprocess.run([str(root/'.venv/Scripts/python.exe'),str(root/'src/collect_resume_results.py'),'--family','tabpfn','--job',job],check=True)
    subprocess.run([str(root/'.venv/Scripts/python.exe'),str(root/'src/evaluate_tabpfn_v2.py'),'--job',job],check=True)
    print(json.dumps({'job_complete':job}),flush=True)
