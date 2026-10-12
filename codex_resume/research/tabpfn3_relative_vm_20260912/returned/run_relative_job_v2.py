import subprocess,json,sys,time,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
assert not (r/'job-result.json').exists()
expected='0231b3ae2fb6cd6e7ac1ee792580be797632bff3dda594d4e4ae9d929e48e7b5'
assert hashlib.sha256((r/'tabpfn3_relative_vm.py').read_bytes()).hexdigest()==expected
assert hashlib.sha256((r/'tabpfn3_hardware_resume.py').read_bytes()).hexdigest()=='61738e64328370116de7f19686731704fe921d3fbea91164f49a61967dd44603'
started=time.time();results=[]
for mode in ('freeze','benchmark','fit','replay'):
 x=subprocess.run([str(r/'venv/bin/python'),'-u',str(r/'tabpfn3_relative_vm.py'),mode,'--root',str(r),'--out',str(r/'run')])
 results.append({'mode':mode,'exit_code':x.returncode})
 if x.returncode:break
(r/'job-result.json').write_text(json.dumps({'stages':results,'seconds':time.time()-started,'success':len(results)==4 and all(x['exit_code']==0 for x in results)},indent=2))
sys.exit(results[-1]['exit_code'])
