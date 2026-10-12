"""Reconstruct omitted-family package byte-exactly from its existing parent."""
from pathlib import Path
import hashlib
import json
import numpy as np

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def select(data,group,indices):
    offsets=data[group+'_offsets']; lengths=np.diff(offsets)[indices]
    handix=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in indices])
    result={'x':data[group+'_x'][handix],'offsets':np.r_[0,np.cumsum(lengths)],'pair':data[group+'_pair'][indices]}
    if group=='train': result.update({'y':data['train_y'][indices],'relevance':data['train_relevance'][handix]})
    return {group+'_'+k:v for k,v in result.items()}


def main():
    root=Path(__file__).resolve().parent; plan=json.loads((root/'delta-plan.json').read_text())
    parent=Path('/workspace/poker-research/mil-fold0-v1/arrays.npz'); assert digest(parent)==plan['parent_sha256']
    data=np.load(parent,allow_pickle=False); delta=np.load(root/'delta.npz',allow_pickle=False); arrays={}
    for group in ['full','shortened','train']: arrays.update(select(data,group,delta[group]))
    for name,expected in plan['array_sha256'].items(): assert hashlib.sha256(arrays[name].tobytes()).hexdigest()==expected
    np.savez_compressed(root/'arrays.npz',**{k:arrays[k] for k in plan['key_order']})
    actual=digest(root/'arrays.npz'); assert actual==plan['expected_file_sha256'],(actual,plan['expected_file_sha256'])
    print(json.dumps({'exact_reconstruction':True,'sha256':actual}))


if __name__=='__main__': main()
