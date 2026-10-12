from __future__ import annotations
import gc, hashlib, json, os, time, traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_NO_BROWSER'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from tabpfn.checkpoint import Checkpoint, save_as_safetensors

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'checkpoint' / 'tabpfn-v3-classifier-v3_default.ckpt'
OUTPUT = HERE / 'checkpoint' / 'tabpfn-v3-classifier-v3_default.safetensors'
RECEIPT = HERE / 'safe-checkpoint-receipt.json'
EXPECTED = 'd0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    if RECEIPT.exists() or OUTPUT.exists():
        raise RuntimeError('write-once safe-checkpoint output already exists')
    started = time.perf_counter()
    result = {'created_at':datetime.now(timezone.utc).isoformat(),'passed':False,'source':str(SOURCE.resolve()),'source_sha256':sha(SOURCE),'load_call':'torch.load(map_location="cpu", weights_only=True, mmap=True)','gpu_used':False}
    tmp = OUTPUT.with_suffix('.safetensors.partial')
    checkpoint = reloaded = None
    try:
        assert result['source_sha256'] == EXPECTED
        checkpoint = torch.load(str(SOURCE), map_location='cpu', weights_only=True, mmap=True)
        assert isinstance(checkpoint, dict)
        state = checkpoint.get('state_dict')
        assert isinstance(state, dict) and state
        assert all(isinstance(v, torch.Tensor) for v in state.values())
        save_as_safetensors(checkpoint, tmp)
        reloaded = Checkpoint(tmp).load()
        state2 = reloaded['state_dict']
        assert set(state) == set(state2)
        assert all(state[k].shape == state2[k].shape and state[k].dtype == state2[k].dtype and torch.equal(state[k], state2[k]) for k in state)
        metadata_keys = sorted(k for k in checkpoint if k != 'state_dict')
        tmp.replace(OUTPUT)
        result.update({'passed':True,'output':str(OUTPUT.resolve()),'output_sha256':sha(OUTPUT),'output_bytes':OUTPUT.stat().st_size,'checkpoint_keys':sorted(checkpoint),'metadata_keys':metadata_keys,'state_dict_tensors':len(state),'state_dict_all_tensors':True,'safe_reload_exact':True})
    except BaseException as exc:
        result['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        if tmp.exists():
            tmp.unlink()
    finally:
        result['seconds']=time.perf_counter()-started
        del checkpoint,reloaded
        gc.collect()
        RECEIPT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if not result['passed']:
        raise RuntimeError(f'safe conversion failed; see {RECEIPT}')
    print(json.dumps({k:result[k] for k in ('passed','output_sha256','output_bytes','state_dict_tensors','safe_reload_exact','seconds')},indent=2))

if __name__=='__main__':
    main()
