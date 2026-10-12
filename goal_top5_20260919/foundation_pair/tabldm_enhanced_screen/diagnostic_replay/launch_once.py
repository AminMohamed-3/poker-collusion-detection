import os,subprocess,time,json
from pathlib import Path
R=Path(r'C:/Data/Projects/Money Making');O=R/'work/poker/goal_top5_20260919/foundation_pair/tabldm_enhanced_screen/diagnostic_replay'
assert not (O/'launcher_receipt.json').exists() and not (O/'terminal.log').exists()
env=os.environ.copy();env['PYTHONPATH']=str(R/'work/poker/goal_top5_20260919/tabldm/overlay');env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2';t=time.monotonic()
try:r=subprocess.run([str(R/'work/poker/Codex Resume/.venv_tabpfn3/Scripts/python.exe'),'-B',str(O/'run_diagnostic.py')],env=env,capture_output=True,text=True,timeout=900);status='PASS' if r.returncode==0 else 'FAILED';text=(r.stdout or '')+(r.stderr or '')
except subprocess.TimeoutExpired as e:
 def s(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
 status='TIMEOUT';r=None;text=s(e.stdout)+s(e.stderr)
(O/'terminal.log').write_text(text);rec={'status':status,'returncode':None if r is None else r.returncode,'wall_seconds':time.monotonic()-t,'timeout_seconds':900};(O/'launcher_receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
assert status=='PASS',rec
