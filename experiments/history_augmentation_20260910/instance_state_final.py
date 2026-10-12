"""Use only this instance's scoped Vast credential; emit filtered state, never keys."""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--stop', action='store_true')
args = p.parse_args()
instance = os.environ.get('CONTAINER_ID')
assert instance == '50489879', 'Unexpected instance'
key = os.environ['CONTAINER_API_KEY']

def call(action):
    r = subprocess.run(['vastai', action, 'instance', instance, '--api-key', key, '--raw'], capture_output=True, text=True, timeout=45)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        data = {}
    out = {'action': action, 'instance_id': instance, 'at': datetime.now(timezone.utc).isoformat(), 'returncode': r.returncode, 'parsed_json': bool(data)}
    if action == 'stop':
        out['stop_accepted'] = ('stopping instance '+instance+'.') in r.stdout
    rows = data.get('instances', data) if isinstance(data, dict) else data
    if isinstance(rows, list):
        assert len(rows) <= 1
        rows = rows[0] if rows else {}
    if isinstance(rows, dict):
        for name in ['id','actual_status','intended_status','cur_state','state','success']:
            if name in rows:
                out[name] = rows[name]
    print(json.dumps(out), flush=True)
    return out

state = call('show')
assert state['returncode'] == 0 and state['parsed_json'], 'State request failed'
if args.stop:
    status = subprocess.run(['supervisorctl', 'status'], capture_output=True, text=True)
    research = [line for line in status.stdout.splitlines() if line.startswith('poker_')]
    q=json.loads(Path('/workspace/history-augmentation-20260910/trial/queue.json').read_text())
    assert len(q['completed'])==q['total']==18
    assert Path('/workspace/history-augmentation-20260910/trial-results.tar.gz').is_file()
    assert len(research) == 10 and all('EXITED' in line or 'STOPPED' in line for line in research), 'Research still running'
    stopped = call('stop')
    assert stopped['returncode'] == 0 and stopped['stop_accepted'], 'Stop request failed'
    for i in range(8):
        final = call('show')
        if final.get('actual_status') == 'stopped':
            break
        time.sleep(2)
