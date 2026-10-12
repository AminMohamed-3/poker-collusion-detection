from pathlib import Path
import hashlib,json,tarfile,subprocess
base=Path("/workspace/poker/codex");a=base/"tabpfn-finetune-hardware-v3-package.tar.gz"
assert hashlib.sha256(a.read_bytes()).hexdigest()=="c9f76add1d98b54defd0aca9a5df35c5f48aa59e2bb82e1b0d169e3aa2b28114"
r=base/"tabpfn_finetune_hardware_v3_20260913";r.mkdir(exist_ok=False)
with tarfile.open(a) as t:t.extractall(r,filter="data")
f=json.loads((r/"freeze.json").read_text())
for n,d in f["files_sha256"].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==d,n
runtime=base/"tabpfn3_relative_20260912";cp=runtime/"checkpoint/tabpfn-v3-classifier-v3_default.ckpt"
cmd=[str(runtime/"venv/bin/python"),str(r/"tabpfn_finetune_hardware_v3.py"),"--root",str(r),"--checkpoint",str(cp),"--output",str(r/"remote-metadata"),"--dry-run"]
x=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
(r/"remote-metadata.log").write_text(x.stdout+x.stderr)
assert x.returncode==0,(x.stdout,x.stderr)
assert json.loads((r/"remote-metadata/summary.json").read_text())["status"]=="dry-run-passed"
print(json.dumps({"remote_dry_run":"passed","root":str(r),"source_sha256":f["source_sha256"]}))
c=Path("/etc/supervisor/conf.d/codex-tabpfn-finetune-hardware-v3.conf");assert not c.exists()
command="/usr/bin/timeout --signal=TERM --kill-after=15s 900 "+str(runtime/"venv/bin/python")+" -u "+str(r/"tabpfn_finetune_hardware_v3.py")+" --root "+str(r)+" --checkpoint "+str(cp)+" --output "+str(r/"run")
c.write_text("[program:codex_tabpfn_finetune_hardware_v3]\ncommand="+command+"\ndirectory="+str(r)+"\nautostart=false\nautorestart=false\nstartsecs=0\nstdout_logfile="+str(r/"hardware.log")+"\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n")
(r/"launch-receipt.json").write_text(json.dumps({"source_sha256":f["source_sha256"],"config_sha256":hashlib.sha256(c.read_bytes()).hexdigest(),"wallclock_limit_seconds":900,"existing_runtime":str(runtime)},indent=2))
for cmd in [["supervisorctl","reread"],["supervisorctl","update","codex_tabpfn_finetune_hardware_v3"],["supervisorctl","start","codex_tabpfn_finetune_hardware_v3"]]:
 q=subprocess.run(cmd,capture_output=True,text=True);assert q.returncode==0,(q.stdout,q.stderr)
print(subprocess.run(["supervisorctl","status","codex_tabpfn_finetune_hardware_v3"],capture_output=True,text=True).stdout)
