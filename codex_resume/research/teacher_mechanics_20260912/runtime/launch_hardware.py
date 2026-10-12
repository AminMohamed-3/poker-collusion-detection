from pathlib import Path
import hashlib,tarfile,subprocess,json,py_compile
r=Path('/workspace/poker/codex/teacher_mechanics_20260912');a=r/'hardware-code.tar.gz';assert hashlib.sha256(a.read_bytes()).hexdigest()=='c256262256ab78f3dae4b9ee58beafb88fb91d53fb73d771314f96dda6c2173b'
with tarfile.open(a) as t:
 assert set(t.getnames())==set({'teacher_mechanics_infer_v2.py': '3fc4b8514741578e5c8955c4ea9859b57208006826a8ef2a6e2b9b52369a14ff', 'hardware-requests.jsonl': 'b96426baae2c33f7ec2374e5c679642d33b7ff11df5dea7ce73d62ca5f9c5946', 'official-model.json': '5d4e7512c9203ac15826f7eeab24336044763555863e3c3e8297b7c95f25ee04'})
 for m in t.getmembers():
  assert m.isfile() and not (r/m.name).exists()
 t.extractall(r,filter='data')
for name,digest in {'teacher_mechanics_infer_v2.py': '3fc4b8514741578e5c8955c4ea9859b57208006826a8ef2a6e2b9b52369a14ff', 'hardware-requests.jsonl': 'b96426baae2c33f7ec2374e5c679642d33b7ff11df5dea7ce73d62ca5f9c5946', 'official-model.json': '5d4e7512c9203ac15826f7eeab24336044763555863e3c3e8297b7c95f25ee04'}.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
py_compile.compile(str(r/'teacher_mechanics_infer_v2.py'),doraise=True)
c=Path('/etc/supervisor/conf.d/codex-teacher-hardware.conf');assert not c.exists()
c.write_text('[program:codex_teacher_hardware]\ncommand='+str(r/'venv/bin/python')+' -u '+str(r/'teacher_mechanics_infer_v2.py')+' --root '+str(r)+' --requests '+str(r/'hardware-requests.jsonl')+' --output '+str(r/'hardware')+' --mode hardware\ndirectory='+str(r)+'\nautostart=false\nautorestart=false\nstartsecs=0\nstdout_logfile='+str(r/'hardware.log')+'\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
(r/'hardware-launch-receipt.json').write_text(json.dumps({'archive_sha256':'c256262256ab78f3dae4b9ee58beafb88fb91d53fb73d771314f96dda6c2173b','files':{'teacher_mechanics_infer_v2.py': '3fc4b8514741578e5c8955c4ea9859b57208006826a8ef2a6e2b9b52369a14ff', 'hardware-requests.jsonl': 'b96426baae2c33f7ec2374e5c679642d33b7ff11df5dea7ce73d62ca5f9c5946', 'official-model.json': '5d4e7512c9203ac15826f7eeab24336044763555863e3c3e8297b7c95f25ee04'},'config_sha256':hashlib.sha256(c.read_bytes()).hexdigest()},indent=2)+'\n')
subprocess.run(['supervisorctl','reread'],check=True);subprocess.run(['supervisorctl','update','codex_teacher_hardware'],check=True);subprocess.run(['supervisorctl','start','codex_teacher_hardware'],check=True)
