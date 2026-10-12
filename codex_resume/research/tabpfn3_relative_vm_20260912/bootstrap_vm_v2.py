from pathlib import Path
import tarfile,json,hashlib,subprocess
r=Path('/workspace/poker/codex/tabpfn3_relative_20260912')
old=Path('/workspace/poker/codex/tabpfn3_confirmation_20260912')
assert r.resolve()==r and r.parent==Path('/workspace/poker/codex')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=r/'relative-contexts.tar.gz'
assert sha(archive)=='848b9029d0cdac2c3cdbc2081cfa81edf0f71ef0207a779f8228343eecf68f4d'
assert (r/'transfer-inventory.json').exists() and not (r/'run').exists()
assert sha(r/'transfer-inventory.json')=='762e462f03f9366a7d572bdce3a5f2dbfa22553f2040582a51b6676fd6b4e2b0'
for n,h in json.loads((r/'transfer-inventory.json').read_text()).items():assert sha(r/n)==h,n
assert sha(r/'tabpfn3_relative_vm.py')=='0231b3ae2fb6cd6e7ac1ee792580be797632bff3dda594d4e4ae9d929e48e7b5'
assert sha(r/'tabpfn3_hardware_resume.py')=='61738e64328370116de7f19686731704fe921d3fbea91164f49a61967dd44603'
assert sha(r/'run_relative_job.py')=='e7bf61a5c3e31eb2288aa41ae21a9c4d4e27a4ef23922f3a35b3c49cb3b2a10c'
assert sha(r/'run_relative_job_v2.py')=='984c354f2c1ff13fac0307827de355cbf7fab924085c90dcdb100b9b543251b3'
compile((r/'run_relative_job_v2.py').read_text(),str(r/'run_relative_job_v2.py'),'exec')
for name in ['venv','checkpoint']:
 assert (old/name).exists() and not (r/name).exists()
 (r/name).symlink_to(old/name,target_is_directory=True)
config=Path('/etc/supervisor/conf.d/codex-tabpfn-relative.conf');assert config.parent.is_dir() and not config.exists()
text=f'[program:codex_tabpfn_relative]\ncommand={r}/venv/bin/python -u {r}/run_relative_job_v2.py\ndirectory={r}\nautostart=false\nautorestart=false\nstartsecs=1\nstopasgroup=true\nkillasgroup=true\nredirect_stderr=true\nstdout_logfile={r}/job.log\nstdout_logfile_maxbytes=20MB\nstdout_logfile_backups=1\n'
config.write_text(text)
receipt={'archive_sha256':sha(archive),'inventory_sha256':sha(r/'transfer-inventory.json'),'runner_sha256':sha(r/'tabpfn3_relative_vm.py'),'wrapper_sha256':sha(r/'run_relative_job_v2.py'),'original_wrapper_preserved_sha256':sha(r/'run_relative_job.py'),'wrapper_revision_reason':'Original wrapper failed syntax check before any inference; v2 only corrects result JSON formatting','config_sha256':sha(config),'runtime_reused':str(old/'venv'),'stages':['freeze','synthetic599benchmark','fixed90contexts','freshprocess599replay']}
(r/'supervisor-launch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for args in [['supervisorctl','reread'],['supervisorctl','update','codex_tabpfn_relative'],['supervisorctl','start','codex_tabpfn_relative']]:subprocess.run(args,check=True)
print(json.dumps({'launched':True,**receipt}),flush=True)
