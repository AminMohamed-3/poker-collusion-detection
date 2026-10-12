import hashlib,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent/'tabpfn-v3.5-20260909_multiclass.safetensors';url='https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-20260909_multiclass.safetensors';expected='4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149'
assert not p.exists();urllib.request.urlretrieve(url,p)
with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected
print({'sha256':actual,'bytes':p.stat().st_size})
