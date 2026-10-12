from pathlib import Path
import json,hashlib,subprocess
r=Path('/workspace/poker/codex/teacher_mechanics_20260912');h=r/'hardware_v3';comp=json.loads((h/'complete.json').read_text());assert comp['requests']==2
for name,digest in comp['files_sha256'].items():assert hashlib.sha256((h/name).read_bytes()).hexdigest()==digest
short=json.loads((h/'000.json').read_text());assert short['ended_with_eos'] and json.loads(short['raw_text'])=={'ready':True}
assert all(json.loads(p.read_text())['guard_pass'] for p in h.glob('[0-9][0-9][0-9].json'))
assert hashlib.sha256((r/'pilot-requests.jsonl').read_bytes()).hexdigest()=='f13e82bab1f36fbba9e336723cefe8d889f209451832b41047db4403421199e4'
assert hashlib.sha256((r/'teacher_mechanics_infer_v3.py').read_bytes()).hexdigest()=='9badf6c8312bd0b9d0e8e45d844b190b4f8b0d402781e470ad44c90491a7d457'
c=Path('/etc/supervisor/conf.d/codex-teacher-pilot.conf');assert not c.exists()
c.write_text('[program:codex_teacher_pilot]\ncommand='+str(r/'venv/bin/python')+' -u '+str(r/'teacher_mechanics_infer_v3.py')+' --root '+str(r)+' --requests '+str(r/'pilot-requests.jsonl')+' --output '+str(r/'pilot')+' --mode pilot\ndirectory='+str(r)+'\nautostart=false\nautorestart=false\nstartsecs=0\nstdout_logfile='+str(r/'pilot.log')+'\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
(r/'pilot-launch-receipt.json').write_text(json.dumps({'requests_sha256':'f13e82bab1f36fbba9e336723cefe8d889f209451832b41047db4403421199e4','source_sha256':'9badf6c8312bd0b9d0e8e45d844b190b4f8b0d402781e470ad44c90491a7d457','config_sha256':hashlib.sha256(c.read_bytes()).hexdigest()},indent=2)+'\n')
subprocess.run(['supervisorctl','reread'],check=True);subprocess.run(['supervisorctl','update','codex_teacher_pilot'],check=True);subprocess.run(['supervisorctl','start','codex_teacher_pilot'],check=True)
