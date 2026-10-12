"""User-authorized stop of instance 50428518, preserving its disk. Never destroy."""
import json,os,shlex,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import requests
from robust_pair_experiment import ROOT,sha

INSTANCE=50428518
OUT=ROOT/'reports/diverse-next-day/vast-stop.json'
SSH=['ssh','-i',str(Path.home()/'.ssh/codex_kaggle_vm_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=10','-p','23794','root@vm-host.invalid']

def main():
    ready=ROOT/'outputs_v14_evidence_portfolio/ready-audit.json'; report=json.loads(ready.read_text())
    assert len(report['entries'])==5 and all(x['eligible'] and x['validation']['valid'] and sha(ready.parent/x['file'])==x['sha256'] for x in report['entries'])
    assert (ROOT/'reports/internal-public-20260910/comparison.json').exists()
    assert not OUT.exists(),'Inspect existing receipt rather than repeating stop'
    status=subprocess.run(SSH+['supervisorctl status'],capture_output=True,text=True,timeout=25)
    workers=[line.strip() for line in status.stdout.splitlines() if line.startswith('poker-')]
    assert workers and all('EXITED' in line or 'STOPPED' in line for line in workers),'Research worker still active or status unavailable'
    # Keep the scoped provider credential in process memory only; never write or print it.
    code='import json,os; print(json.dumps({"id":os.environ.get("CONTAINER_ID"),"key":os.environ.get("CONTAINER_API_KEY")}))'
    captured=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=25)
    assert captured.returncode==0,'Unable to read scoped instance credentials'
    data=json.loads(captured.stdout.strip().splitlines()[-1]); assert int(data['id'])==INSTANCE and data['key']
    session=requests.Session(); session.headers.update({'Authorization':'Bearer '+data['key']}); del captured,data
    url=f'https://console.vast.ai/api/v0/instances/{INSTANCE}/'
    def observe():
        response=session.get(url,params={'owner':'me'},timeout=30); response.raise_for_status(); body=response.json()['instances']
        assert body and int(body['id'])==INSTANCE
        return {'observed_at':datetime.now(timezone.utc).isoformat(),**{k:body.get(k) for k in ['id','actual_status','intended_status','cur_state','next_state']}}
    before=observe(); audit={'requested_by_user':True,'operation':'stop','instance_id':INSTANCE,'destroy_requested':False,'disk_preserved_by_stop':True,'storage_billing_continues':True,
        'five_candidates_ready_sha256':sha(ready),'source_sha256':sha(__file__),'all_research_workers_exited':workers,'observations':[before]}
    OUT.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'before':before,'research_workers_idle':True}),flush=True)
    if before['intended_status']!='stopped':
        response=session.put(url,json={'state':'stopped'},timeout=30); response.raise_for_status(); body=response.json(); assert body.get('success') is True,'Provider did not confirm stop request'
        audit['request_accepted_at']=datetime.now(timezone.utc).isoformat(); OUT.write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({'stop_request_accepted':True,'instance_id':INSTANCE}),flush=True)
    for _ in range(24):
        current=observe(); audit['observations'].append(current)
        verified=current['intended_status']=='stopped' and current['actual_status'] in ['stopped','exited']
        audit['verified_stopped']=verified; OUT.write_text(json.dumps(audit,indent=2)+'\n')
        if verified:
            print(json.dumps({'verified_stopped':True,'state':current,'receipt_sha256':sha(OUT)}),flush=True); return
        time.sleep(5)
    print(json.dumps({'verified_stopped':False,'last_observation':audit['observations'][-1],'request_accepted':bool(audit.get('request_accepted_at'))}),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        # Do not print HTTP bodies, request headers, captured SSH output or secrets.
        print(json.dumps({'stop_script_error':type(error).__name__,'http_status':error.response.status_code if isinstance(error,requests.HTTPError) and error.response is not None else None,'inspect_sanitized_receipt':str(OUT)}),flush=True)
        raise SystemExit(1)
