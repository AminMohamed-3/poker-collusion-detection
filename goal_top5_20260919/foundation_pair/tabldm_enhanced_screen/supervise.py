import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making'); OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_enhanced_screen'; LIMIT=1800
assert not (OUT/'supervisor_receipt.json').exists() and not (OUT/'terminal_fit.log').exists() and not (OUT/'terminal_replay.log').exists()
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'work/poker/goal_top5_20260919/tabldm/overlay');env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
py=str(ROOT/'work/poker/Codex Resume/.venv_tabpfn3/Scripts/python.exe'); start=time.monotonic(); stages=[]
for name,args in [('fit',[]),('replay',['--replay'])]:
 remaining=LIMIT-(time.monotonic()-start);assert remaining>0,'cumulative 30-minute budget exhausted'
 t=time.monotonic()
 try:r=subprocess.run([py,'-B',str(OUT/'run_tabldm.py'),*args],env=env,capture_output=True,text=True,timeout=remaining)
 except subprocess.TimeoutExpired as e:
  def txt(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
  (OUT/f'terminal_{name}.log').write_text(txt(e.stdout)+txt(e.stderr));raise
 (OUT/f'terminal_{name}.log').write_text(r.stdout+r.stderr);stages.append({'stage':name,'returncode':r.returncode,'seconds':time.monotonic()-t});assert r.returncode==0,(name,r.returncode)
receipt={'status':'PASS','wall_seconds':time.monotonic()-start,'limit_seconds':LIMIT,'stages':stages}
(OUT/'supervisor_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
