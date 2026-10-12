"""Inspect or explicitly destroy only this supplied Vast instance; never print secrets."""
import json
import os
import subprocess
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else 'inspect'
assert mode in ('inspect', 'destroy')
identity = os.environ.get('CONTAINER_ID', '')
key = os.environ.get('CONTAINER_API_KEY', '')
assert identity.isdigit() and key, 'This instance must supply its identity and scoped credential'
assert os.environ.get('PUBLIC_IPADDR') == 'vm-host.invalid', 'Wrong supplied host'

def call(action):
    result = subprocess.run(['vastai', action, 'instance', identity, '--api-key', key, '--raw'],
                            capture_output=True, text=True, timeout=90)
    # Keep arbitrary provider responses and stderr private; return only an allowlist.
    try:
        body = json.loads(result.stdout)
    except json.JSONDecodeError:
        body = None
    if isinstance(body, list):
        body = next((x for x in body if isinstance(x, dict) and str(x.get('id')) == identity), None)
    if isinstance(body, dict) and isinstance(body.get('instances'), list):
        body = next((x for x in body['instances'] if str(x.get('id')) == identity), None)
    fields = ('id', 'actual_status', 'cur_state', 'next_state', 'status', 'success',
              'public_ipaddr', 'ssh_port', 'gpu_name')
    safe = {name: body[name] for name in fields if isinstance(body, dict) and name in body}
    return {'instance_id': identity, 'action': action, 'exit_code': result.returncode,
            'response_is_json': body is not None, 'response': safe}

before = call('show')
print(json.dumps(before), flush=True)
if mode == 'destroy':
    assert before['exit_code'] == 0
    assert str(before['response'].get('id')) == identity, 'Provider identity not verified'
    assert before['response'].get('public_ipaddr') == 'vm-host.invalid'
    print(json.dumps(call('destroy')), flush=True)
