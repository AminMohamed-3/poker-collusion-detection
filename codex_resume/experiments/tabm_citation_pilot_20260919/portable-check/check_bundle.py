import hashlib,json,importlib.metadata as m
from pathlib import Path
p=Path(__file__).resolve().parent
r=json.loads((p/'MANIFEST.json').read_text())
for name,h in r['files'].items():
    f=(p/name).resolve()
    assert f.is_relative_to(p) and f.is_file(),name
    assert hashlib.sha256(f.read_bytes()).hexdigest()==h,name
for name,version in r['runtime'].items():
    assert m.version(name)==version,(name,m.version(name),version)
print('All package hashes and runtime versions verified; no training performed.')
