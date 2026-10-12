"""Fixed-model predictions on the rejected-for-training variation batch."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
from model import TinyCED,Config
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((H/'plan.json').read_text())
assert 'varied' not in plan['arms']
root=Path('/workspace/history-augmentation-20260910/varied')
raw=np.load(root/'features/synthetic-pairs.npz')['x'];x=raw[:len(raw)//3,:897]
torch.set_num_threads(2);torch.set_num_interop_threads(2)
device='cuda';done=[]
for arm in plan['arms']:
    for variant in plan['variants']:
        for seed in plan['seeds']:
            d=H/arm/f'{variant}-{seed}';r=json.loads((d/'receipt.json').read_text())
            assert r['plan_sha256']==sha(H/'plan.json') and r['model_sha256']==sha(d/'model.pt')
            saved=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
            m=TinyCED(Config(**saved['config'])).to(device);m.load_state_dict(saved['state_dict']);m.eval()
            z=torch.tensor(np.clip((x-saved['center'].numpy())/saved['scale'].numpy(),-8,8),device=device)
            with torch.no_grad():p=torch.cat([m(z[i:i+64]).softmax(-1).cpu() for i in range(0,len(z),64)]).numpy()
            np.savez_compressed(d/'stress-predictions.npz',full=p)
            done.append({'arm':arm,'variant':variant,'seed':seed,'predictions_sha256':sha(d/'stress-predictions.npz'),'model_sha256':r['model_sha256']})
(H/'stress-inference-receipt.json').write_text(json.dumps({'plan_sha256':sha(H/'plan.json'),'source_sha256':sha(__file__),'stress_features_sha256':sha(root/'features/synthetic-pairs.npz'),'no_fitting':True,'no_checkpoint_or_weight_selection':True,'fits':done},indent=2))
print('Stress predictions complete.',flush=True)
