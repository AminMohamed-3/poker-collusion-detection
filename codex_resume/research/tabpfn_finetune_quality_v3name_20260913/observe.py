from pathlib import Path
import json,subprocess,datetime,sys
sys.stdout.reconfigure(encoding='utf-8')
remote=r"""from pathlib import Path
import json,subprocess,time,shutil
r=Path('/workspace/poker/codex/tabpfn_finetune_quality_v3name_20260913');run=r/'run'
q=subprocess.run(['supervisorctl','status','codex_tabpfn_finetune_quality_v3name'],capture_output=True,text=True)
receipts={p.parent.name:json.loads(p.read_text()) for p in (run/'contexts').glob('*/receipt.json')}
started=[p.parent.name for p in (run/'contexts').glob('*/started.json')]
special={str(p.relative_to(run)):json.loads(p.read_text()) for p in [run/'synthetic/receipt.json',run/'synthetic/replay.json',run/'baseline.json',run/'complete.json'] if p.exists()}
summary={'observed_unix':time.time(),'supervisor':q.stdout.strip(),'supervisor_query_exit':q.returncode,'gpu':subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),'disk_free':shutil.disk_usage(r).free,'started_contexts':len(started),'completed_contexts':len(receipts),'active_contexts':sorted(set(started)-set(receipts)),'contexts':{n:{k:v.get(k) for k in ['status','seconds','peak_reserved_bytes','error']} for n,v in receipts.items()},'checks':special,'complete':(run/'complete.json').exists(),'log_tail':(r/'quality.log').read_text()[-2400:]}
print(json.dumps(summary))
"""
q=subprocess.run(['ssh','-T','-i','C:/Users/USER/.ssh/codex_kaggle_vm_ed25519','-p','30281','-o','BatchMode=yes','-o','ConnectTimeout=15','root@vm-host.invalid','/venv/main/bin/python -'],input=remote,capture_output=True,text=True,encoding='utf-8',timeout=35)
base=Path(__file__).resolve().parent
if q.returncode==0:
    d=json.loads(q.stdout);(base/'latest-observation.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
    compact={k:v for k,v in d.items() if k not in ['contexts','checks']}
    compact['checks']={k:({kk:vv for kk,vv in v.items() if kk not in ['original_resolved_inference_config','training_resolved_inference_config','losses','gradient_trace','context_receipts','traceback']} if isinstance(v,dict) else v) for k,v in d['checks'].items()}
    print(json.dumps(compact,indent=2))
else:print(q.stdout);print(q.stderr)
raise SystemExit(q.returncode)
