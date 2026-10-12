"""Accept identical uncompressed NPY members despite ZIP container differences."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
p=argparse.ArgumentParser(); p.add_argument('--job',required=True); args=p.parse_args(); assert args.job.startswith('omit_') and args.job.replace('_','').isalnum()
root=Path('/workspace/poker-research')/('mil-'+args.job+'-v1')
manifest=json.loads((root/'expected-npy-members.json').read_text()); archive=root/'arrays.npz'
with zipfile.ZipFile(archive) as z:
    actual={name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
assert actual==manifest['members'], 'Array serialization changed, refuse activation'
planpath=root/'worker-plan.json'; original=planpath.read_bytes(); original_sha=hashlib.sha256(original).hexdigest(); assert original_sha==manifest['original_worker_plan_sha256']
plan=json.loads(original); assert plan['files']['arrays.npz']==manifest['original_array_archive_sha256']
actualsha=hashlib.sha256(archive.read_bytes()).hexdigest()
(root/'worker-plan-original.json').write_bytes(original)
plan['files']['arrays.npz']=actualsha; planpath.write_text(json.dumps(plan,indent=2)+'\n')
receipt={'original_worker_plan_sha256':original_sha,'worker_plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'original_array_archive_sha256':manifest['original_array_archive_sha256'],'repacked_array_archive_sha256':actualsha,'all_uncompressed_npy_members_identical':True,'members':actual,'explanation':'Every uncompressed NPY member, including shape/dtype/order and all values, matches. Only the ZIP container encoding differs across platforms. No experiment/model/data change.'}
(root/'serialization-equivalence.json').write_text(json.dumps(receipt,indent=2)+'\n')
name='poker-mil-'+args.job; config=Path('/etc/supervisor/conf.d')/(name+'.conf'); assert not config.exists()
config.write_text(f'''[program:{name}]
command=/venv/main/bin/python -u {root}/run_mil_pairs.py
directory={root}
autostart=false
autorestart=false
startsecs=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={root}/run.log
stdout_logfile_maxbytes=0
environment=PYTHONUNBUFFERED="1"
''')
subprocess.run(['supervisorctl','reread'],check=True); subprocess.run(['supervisorctl','update',name],check=True); subprocess.run(['supervisorctl','start',name],check=True)
print(json.dumps(receipt))
