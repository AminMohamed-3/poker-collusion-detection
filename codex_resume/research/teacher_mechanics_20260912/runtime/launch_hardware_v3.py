from pathlib import Path
import hashlib,subprocess,json,py_compile
r=Path('/workspace/poker/codex/teacher_mechanics_20260912');p=r/'teacher_mechanics_infer_v3.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='9badf6c8312bd0b9d0e8e45d844b190b4f8b0d402781e470ad44c90491a7d457';py_compile.compile(str(p),doraise=True)
c=Path('/etc/supervisor/conf.d/codex-teacher-hardware-v3.conf');assert not c.exists()
c.write_text('[program:codex_teacher_hardware_v3]\ncommand='+str(r/'venv/bin/python')+' -u '+str(p)+' --root '+str(r)+' --requests '+str(r/'hardware-requests.jsonl')+' --output '+str(r/'hardware_v3')+' --mode hardware\ndirectory='+str(r)+'\nautostart=false\nautorestart=false\nstartsecs=0\nstdout_logfile='+str(r/'hardware_v3.log')+'\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
subprocess.run(['supervisorctl','reread'],check=True);subprocess.run(['supervisorctl','update','codex_teacher_hardware_v3'],check=True);subprocess.run(['supervisorctl','start','codex_teacher_hardware_v3'],check=True)
