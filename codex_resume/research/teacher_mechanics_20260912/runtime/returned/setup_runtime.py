from pathlib import Path
import subprocess,json,os,time,sys,hashlib
r=Path('/workspace/poker/codex/teacher_mechanics_20260912')
r.mkdir(parents=True,exist_ok=True)
assert not (r/'setup-result.json').exists(), 'setup is write-once'
started=time.time(); env=dict(os.environ)
for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN']:env.pop(key,None)
env.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
commands=[['/venv/main/bin/python','-m','venv','--copies',str(r/'venv')]]
try:
 for cmd in commands:subprocess.run(cmd,check=True,env=env)
 site=r/'venv/lib/python3.12/site-packages'
 (site/'borrow-main-read-only.pth').write_text('/venv/main/lib/python3.12/site-packages\n')
 (r/'constraints.txt').write_text('torch==2.11.0+cu128\n')
 command=[str(r/'venv/bin/python'),'-m','pip','install','--no-cache-dir','--constraint',str(r/'constraints.txt'),'--report',str(r/'pip-report.json'),'transformers==5.17.0','bitsandbytes==0.50.2','accelerate==1.15.0']
 subprocess.run(command,check=True,env=env)
 code='import transformers,bitsandbytes,accelerate,torch; from transformers import Qwen3_5ForCausalLM,Qwen3_5ForConditionalGeneration,Qwen3_5TextConfig,BitsAndBytesConfig; print(transformers.__version__,bitsandbytes.__version__,accelerate.__version__,torch.__version__); print(torch.cuda.get_device_name(0))'
 check=subprocess.run([str(r/'venv/bin/python'),'-c',code],check=True,env=env,text=True,capture_output=True)
 print(check.stdout,flush=True)
 result={'exit_code':0,'seconds':time.time()-started,'import_check':check.stdout,'model_weights_downloaded':False}
except Exception as exc:
 result={'exit_code':1,'seconds':time.time()-started,'error':repr(exc)}
(r/'setup-result.json').write_text(json.dumps(result,indent=2)+'\n')
sys.exit(result['exit_code'])
