from pathlib import Path
import json,subprocess,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;P=O/'vm-package';R='/workspace/poker/codex/tabpfn3_production_20260913';KEY='C:/Users/USER/.ssh/codex_kaggle_vm_ed25519'
inv=json.loads((O/'transfer-plan.json').read_text());assert inv['file_count']==71 and inv['bytes']==207050194
ssh=['ssh','-T','-i',KEY,'-p','30281','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid','/venv/main/bin/python -']
def remote(code):
 q=subprocess.run(ssh,input=code,capture_output=True,text=True,encoding='utf-8',timeout=60);assert q.returncode==0,(q.stdout,q.stderr);return q.stdout
for n,h in inv['files'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h
meta=json.dumps({'root':R,'files':inv['files'],'bytes':inv['bytes']})
print(remote("import json,shutil\nfrom pathlib import Path\nm=json.loads("+repr(meta)+")\nr=Path(m['root']);assert not r.exists(), 'Existing directory: inspect before retry'\nassert shutil.disk_usage(r.parent).free>=m['bytes']+1024**3\nr.mkdir()\nfor n in m['files']:\n p=r/n;assert p.resolve().is_relative_to(r.resolve());p.parent.mkdir(parents=True,exist_ok=True)\nprint('Remote directories prepared')\n"),flush=True)
cmd=['sftp','-q','-b','-','-i',KEY,'-P','30281','-R','64','-B','262144','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid']
batch=''.join('put "'+(P/n).as_posix()+'" "'+R+'/'+n+'"\n' for n in inv['files'])
q=subprocess.run(cmd,input=batch,capture_output=True,text=True,encoding='utf-8',timeout=900)
assert q.returncode==0,q.stderr[-2000:]
code=r"""from pathlib import Path
import json,hashlib,subprocess,shutil
r=Path('/workspace/poker/codex/tabpfn3_production_20260913');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((r/'production-freeze.json').read_text())
for n,d in f['files'].items():assert h(r/n)==d,n
assert h(r/'production-freeze.json')=='c4f9527f1dbe527f447c267c2f711931f13de73ac6407730515ef544ef0589b3'
(r/'checkpoint').symlink_to((r.parent/'tabpfn3_confirmation_20260912/checkpoint').resolve(),target_is_directory=True)
assert h(r/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==f['checkpoint_sha256']
assert shutil.disk_usage(r).free>=1024**3
name='codex_tabpfn_production';config=Path('/etc/supervisor/conf.d/codex-tabpfn-production.conf');assert not config.exists()
python=r.parent/'tabpfn3_relative_20260912/venv/bin/python'
command='/usr/bin/timeout --signal=TERM --kill-after=15s 10800 '+str(python)+' -u '+str(r/'tabpfn3_production_vm.py')+' all --root '+str(r)
config.write_text('[program:'+name+']\ncommand='+command+'\ndirectory='+str(r)+'\nautostart=false\nautorestart=false\nstartsecs=0\nstdout_logfile='+str(r/'production.log')+'\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
receipt={'production_freeze_sha256':h(r/'production-freeze.json'),'supervisor_config_sha256':h(config),'verified_files':len(f['files'])+1,'disk_free':shutil.disk_usage(r).free,'wallclock_limit_seconds':10800,'name':name}
(r/'launch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for c in [['supervisorctl','reread'],['supervisorctl','update',name],['supervisorctl','start',name]]:
 q=subprocess.run(c,capture_output=True,text=True);assert q.returncode==0,(q.stdout,q.stderr)
receipt['supervisor_status']=subprocess.run(['supervisorctl','status',name],capture_output=True,text=True).stdout.strip()
print(json.dumps(receipt))
"""
result=json.loads(remote(code));(O/'launch-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
