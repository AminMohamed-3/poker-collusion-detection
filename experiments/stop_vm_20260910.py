"""Stop the supplied VM and verify provider state after SSH becomes unavailable.

The scoped instance credential is transferred through a captured SSH pipe into
process memory only. It is never printed, written to disk, or sent to another
service. Endpoints/authentication follow the installed official Vast CLI.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

H = Path(__file__).resolve().parent
out = H/'vm-stop-20260910.json'
assert not out.exists(), 'Do not repeat a stop operation without reviewing its receipt'
backup = json.loads((H/'research-backup-20260910/receipt.json').read_text())
assert backup['verified_members'] == 2229
assert hashlib.sha256(Path(backup['archive']).read_bytes()).hexdigest() == backup['sha256']
remote = r'''
import json, os, subprocess
assert os.environ['CONTAINER_ID']=='50489879'
r=subprocess.run(['supervisorctl','status'],capture_output=True,text=True)
workers=[x for x in r.stdout.splitlines() if x.startswith('poker_')]
assert len(workers)==7 and all('EXITED' in x or 'STOPPED' in x for x in workers)
print(json.dumps({'key':os.environ['CONTAINER_API_KEY'],'instance_id':os.environ['CONTAINER_ID'],'workers':workers}))
'''
ssh = subprocess.run(['ssh','-i',r'C:\Users\USER\.ssh\codex_kaggle_vm_ed25519','-p','7368','-o','BatchMode=yes','-o','ConnectTimeout=12','root@vm-host.invalid','/venv/main/bin/python -'], input=remote, capture_output=True, text=True, timeout=30)
if ssh.returncode != 0:
    # A failed preflight has not issued an API request. SSH stderr contains
    # connection errors or Python tracebacks, never the captured credential.
    print(json.dumps({'ssh_returncode':ssh.returncode,'preflight_error':ssh.stderr[-1500:]}),flush=True)
    raise SystemExit('Could not check VM workers; no stop request sent')
credential = json.loads(ssh.stdout)
assert credential['instance_id'] == '50489879'
url = 'https://console.vast.ai/api/v0/instances/50489879/'
headers = {'Authorization':'Bearer '+credential['key'],'Content-Type':'application/json'}
result = {'instance_id':50489879,'started_at':datetime.now(timezone.utc).isoformat(),'backup_sha256':backup['sha256'],'research_workers':credential['workers'],'stop_accepted':False,'provider_stopped_confirmed':False,'states':[]}

def save():
    out.write_text(json.dumps(result,indent=2))

def request(method, data=None):
    req = Request(url + ('?owner=me' if method=='GET' else ''), data=None if data is None else json.dumps(data).encode(), headers=headers, method=method)
    with urlopen(req, timeout=15) as response:
        return json.load(response)

def state():
    response = request('GET')
    row = response['instances']
    assert row['id'] == 50489879
    item = {k:row[k] for k in ['id','actual_status','intended_status','cur_state'] if k in row}
    item['at'] = datetime.now(timezone.utc).isoformat()
    result['states'].append(item)
    result['provider_stopped_confirmed'] = row.get('actual_status') == 'stopped'
    save()
    print(json.dumps(item),flush=True)

try:
    state()
    if not result['provider_stopped_confirmed']:
        response = request('PUT', {'state':'stopped'})
        result['stop_accepted'] = response.get('success') is True
        save()
        assert result['stop_accepted'], 'Provider did not accept stop'
        print('Provider accepted stop request.',flush=True)
        for i in range(12):
            state()
            if result['provider_stopped_confirmed']:
                break
            time.sleep(5)
except HTTPError as e:
    result['error'] = {'type':'HTTPError','status':e.code}
except (URLError, TimeoutError) as e:
    result['error'] = {'type':type(e).__name__}
finally:
    headers.clear()
    credential.clear()
    del ssh
    result['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({k:v for k,v in result.items() if k not in ['research_workers','states']},indent=2),flush=True)
