import hashlib,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent/'tabpfn-v3.5-fast-20260909.safetensors';url='https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-fast-20260909.safetensors';expected='14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
assert not p.exists();urllib.request.urlretrieve(url,p)
with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected;(p.parent/'checkpoint_receipt.json').write_text(__import__('json').dumps({'sha256':actual,'bytes':p.stat().st_size,'url':url},indent=2)+'\n')
