import subprocess,json,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
assert not (r/'job-result.json').exists()
import hashlib
assert (r/'hardware-run').resolve()!=r.resolve()
assert hashlib.sha256((r/'tabpfn3_hardware_resume.py').read_bytes()).hexdigest()=='61738e64328370116de7f19686731704fe921d3fbea91164f49a61967dd44603'
assert json.loads((r/'hardware-run/freeze.json').read_text())['source_sha256']=='61738e64328370116de7f19686731704fe921d3fbea91164f49a61967dd44603'
started=time.time();results=[]
for mode in ('fit','replay'):
    x=subprocess.run([str(r/'venv/bin/python'),str(r/'tabpfn3_hardware_resume.py'),mode,'--root',str(r),'--out',str(r/'hardware-run')])
    results.append({'mode':mode,'exit_code':x.returncode})
    if x.returncode:break
(r/'job-result.json').write_text(json.dumps({'stages':results,'seconds':time.time()-started,'success':len(results)==2 and all(x['exit_code']==0 for x in results)},indent=2)+'\n')
sys.exit(results[-1]['exit_code'])
