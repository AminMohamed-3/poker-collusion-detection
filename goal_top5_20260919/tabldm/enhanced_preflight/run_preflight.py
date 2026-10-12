import os,sys,json,time,hashlib,subprocess,threading
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making');OUT=ROOT/'work/poker/goal_top5_20260919/tabldm/enhanced_preflight'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
man=json.loads((OUT/'MANIFEST.json').read_text());rel=OUT/'GPU_RELEASE.json';assert rel.exists() and json.loads(rel.read_text())['manifest_sha256']==sha(OUT/'MANIFEST.json');assert not (OUT/'RESULT.json').exists() and not (OUT/'worker_result.json').exists() and not (OUT/'stage_milestones.log').exists()
for n,h in man['files'].items():assert sha(OUT/n)==h,(n,'hash mismatch')
env=os.environ.copy();env['PYTHONPATH']=man['pythonpath_overlay'];env['PYTHONUNBUFFERED']='1';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2';cmd=[man['python_executable'],'-B',str(OUT/'worker.py')];start=time.monotonic();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env);lines=[]
def read():
 for line in p.stdout:
  z=f'{time.monotonic()-start:.3f}s {line.rstrip()}';lines.append(z);print(z,flush=True)
t=threading.Thread(target=read,daemon=True);t.start()
try: code=p.wait(timeout=1200)
except subprocess.TimeoutExpired:
 p.terminate()
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:p.kill();p.wait()
 code=124
t.join(timeout=5);(OUT/'stage_milestones.log').write_text('\n'.join(lines)+'\n');result={'status':'PASS' if code==0 else ('TIMEOUT' if code==124 else 'FAILED'),'exit_code':code,'wall_seconds':time.monotonic()-start,'manifest_sha256':sha(OUT/'MANIFEST.json'),'stage_log_sha256':sha(OUT/'stage_milestones.log'),'worker_receipt_present':(OUT/'worker_result.json').exists()}
if code==0:
 assert result['worker_receipt_present'];w=json.loads((OUT/'worker_result.json').read_text());assert w['status']=='PASS' and w['hashes']['manifest']==result['manifest_sha256'];result['worker_result_sha256']=sha(OUT/'worker_result.json')
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');sys.exit(code)
