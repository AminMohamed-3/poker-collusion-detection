from pathlib import Path
import json,hashlib,shutil,subprocess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=Path(__file__).resolve().parent;base=Path('/workspace/poker/codex');parent=base/'tabpfn3_confirmation_20260912'
assert r==base/'tabpfn3_soft_specialist_20260913'
f=json.loads((r/'study-freeze.json').read_text());ctx=json.loads((r/'context-receipt.json').read_text())
assert sha(parent/'context-receipt.json')==f['source_context_receipt_sha256']
assert shutil.disk_usage(r).free>=2*1024**3
(r/'contexts').mkdir(exist_ok=False)
for m in ctx['contexts']:
 src=parent/m['context_file'];dst=r/m['context_file'];assert sha(src)==m['context_sha256'];shutil.copy2(src,dst)
 stem=m['stem'];receipt=parent/'hardware-run/predictions'/(stem+'.json');expected=parent/'hardware-run/predictions'/(stem+'.npy')
 assert sha(receipt)==f['baseline_source_receipts'][stem]
 assert sha(expected)==json.loads(receipt.read_text())['probabilities_sha256']==f['baseline_probabilities'][stem]
 shutil.copy2(expected,r/('baseline_expected-'+stem+'.npy'))
(r/'checkpoint').symlink_to((parent/'checkpoint').resolve(),target_is_directory=True)
assert sha(r/'checkpoint/tabpfn-v3-classifier-v3_default.ckpt')==f['original_checkpoint_sha256']
for name,digest in f['files_sha256'].items():assert sha(r/name)==digest,name
runtime=base/'tabpfn3_relative_20260912/venv/bin/python';config=Path('/etc/supervisor/conf.d/codex-tabpfn-soft-specialist.conf')
assert not config.exists() and not (r/'run').exists()
name='codex_tabpfn_soft_specialist'
command='/usr/bin/timeout --signal=TERM --kill-after=15s 1200 '+str(runtime)+' -u '+str(r/'tabpfn3_finetune_vm.py')+' all --root '+str(r)+' --out '+str(r/'run')
config.write_text('[program:'+name+']\ncommand='+command+'\ndirectory='+str(r)+'\nautostart=false\nautorestart=false\nstartsecs=0\nstdout_logfile='+str(r/'quality.log')+'\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
(r/'launch-receipt.json').write_text(json.dumps({'study_freeze_sha256':sha(r/'study-freeze.json'),'supervisor_config_sha256':sha(config),'data_already_on_VM':True,'copied_contexts':3,'new_data_uploaded':False,'wallclock_limit_seconds':1200}))
for command in [['supervisorctl','reread'],['supervisorctl','update',name],['supervisorctl','start',name]]:
 q=subprocess.run(command,capture_output=True,text=True);assert q.returncode==0,(q.stdout,q.stderr)
print(subprocess.run(['supervisorctl','status',name],capture_output=True,text=True).stdout)
