"""Package only the frozen feature contexts and verification records for the user's existing VM."""
from pathlib import Path
import json,tarfile,hashlib,shutil
import tabpfn3_relative_prepare as prep
ROOT=prep.HERE;OUT=prep.OUT;DEST=ROOT/'research/tabpfn3_relative_vm_20260912'

def main():
 prep.check();ctx=prep.read(OUT/'context-receipt.json');assert len(ctx['contexts'])==90 and ctx['preflight_sha256']==prep.sha(OUT/'preflight.json')
 DEST.mkdir(exist_ok=True);stage=DEST/'package';stage.mkdir();(stage/'contexts').mkdir()
 files={'preflight.json':OUT/'preflight.json','context-receipt.json':OUT/'context-receipt.json','normal-reference-receipt.json':OUT/'normal-reference-receipt.json','runtime-receipt.json':ROOT/'research/tabpfn3_runtime_20260912/runtime-receipt.json','protocol.md':prep.PROTOCOL,'source-review.json':OUT/'source-review.json'}
 for n in prep.read(OUT/'normal-reference-receipt.json')['outputs']:files[n]=OUT/n
 for n in ['tabpfn3_relative_vm.py','tabpfn3_hardware_resume.py']:files[n]=ROOT/'src'/n
 for m in ctx['contexts']:
  assert prep.sha(OUT/m['context_file'])==m['context_sha256'];files[m['context_file']]=OUT/m['context_file']
 for n,src in files.items():shutil.copyfile(src,stage/n)
 vh=prep.sha(stage/'tabpfn3_relative_vm.py');hh=prep.sha(stage/'tabpfn3_hardware_resume.py')
 wrapper="import subprocess,json,sys,time,hashlib\nfrom pathlib import Path\nr=Path(__file__).resolve().parent\nassert not (r/'job-result.json').exists()\nexpected=VM_SHA_PLACEHOLDER\nassert hashlib.sha256((r/'tabpfn3_relative_vm.py').read_bytes()).hexdigest()==expected\nassert hashlib.sha256((r/'tabpfn3_hardware_resume.py').read_bytes()).hexdigest()==HELPER_SHA_PLACEHOLDER\nstarted=time.time();results=[]\nfor mode in ('freeze','benchmark','fit','replay'):\n x=subprocess.run([str(r/'venv/bin/python'),'-u',str(r/'tabpfn3_relative_vm.py'),mode,'--root',str(r),'--out',str(r/'run')])\n results.append({'mode':mode,'exit_code':x.returncode})\n if x.returncode:break\n(r/'job-result.json').write_text(json.dumps({'stages':results,'seconds':time.time()-started,'success':len(results)==4 and all(x['exit_code']==0 for x in results)},indent=2)+'\n')\nsys.exit(results[-1]['exit_code'])\n"
 wrapper=wrapper.replace('VM_SHA_PLACEHOLDER',repr(vh)).replace('HELPER_SHA_PLACEHOLDER',repr(hh));(stage/'run_relative_job.py').write_text(wrapper,encoding='utf-8')
 inventory={p.relative_to(stage).as_posix():prep.sha(p) for p in stage.rglob('*') if p.is_file()}
 prep.write(stage/'transfer-inventory.json',inventory)
 archive=DEST/'relative-contexts.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(stage.rglob('*')):
   if p.is_file():t.add(p,arcname=p.relative_to(stage).as_posix())
 prep.write(DEST/'transfer-receipt.json',{'archive_sha256':prep.sha(archive),'bytes':archive.stat().st_size,'file_count':len(inventory)+1,'inventory_sha256':prep.sha(stage/'transfer-inventory.json'),'vm_source_sha256':vh,'helper_sha256':hh,'wrapper_sha256':prep.sha(stage/'run_relative_job.py'),'contents':'90 frozen feature/label contexts, common-reference models/features, protocol and verification receipts; no account tokens, passwords or SSH keys','destination':'root@vm-host.invalid:30281 /workspace/poker/codex/tabpfn3_relative_20260912; fallback root@ssh5.vast.ai:22543','previous_transfer_authorization':'User explicitly approved poker experiment feature/label transfer to this VM; this is the next matched experiment on the same data and destination.'})
 print(json.dumps(prep.read(DEST/'transfer-receipt.json'),indent=2),flush=True)
if __name__=='__main__':main()
