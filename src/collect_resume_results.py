"""Retrieve completed archives, verify hash, and unpack into a new directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
p=argparse.ArgumentParser(); p.add_argument('--family',choices=['mil','tabpfn'],required=True); p.add_argument('--job',required=True); p.add_argument('--proxy',action='store_true'); args=p.parse_args()
assert args.job.replace('_','').isalnum()
root=Path(__file__).resolve().parents[1]; report='mil-pair-experiment' if args.family=='mil' else 'tabpfn-v2-experiment'; dest=root/'reports'/report/args.job
returned=dest/'returned'; assert not returned.exists()
key=Path.home()/'.ssh/codex_kaggle_vm_ed25519'; options=['-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=15']
command='/venv/main/bin/python /workspace/poker-research/bundle_resume_results.py --family '+args.family+' --job '+args.job
host='root@ssh9.vast.ai' if args.proxy else 'root@vm-host.invalid'; port='28519' if args.proxy else '23794'
r=subprocess.run(['ssh',*options,'-p',port,host,command],capture_output=True,text=True,check=True)
receipt=json.loads(r.stdout.strip().splitlines()[-1]); archive=dest/'returned.tar.gz'
subprocess.run(['scp','-O',*options,'-P',port,host+':'+receipt['archive'],str(archive)],check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
returned.mkdir()
with tarfile.open(archive) as t: t.extractall(returned,filter='data')
(dest/'return-transfer.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'imported':True,**receipt}),flush=True)
