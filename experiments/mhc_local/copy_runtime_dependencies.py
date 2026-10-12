"""Reuse verified local Python packages when the network pip transport stalls.

Run with .venv_tabicl, the existing same-Python-version environment. Only copy
the named distributions into the new isolated environment, never torch.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[1] / '.venv_mhc/Lib/site-packages'
NAMES = ['numpy', 'filelock', 'typing-extensions', 'setuptools', 'sympy',
         'networkx', 'jinja2', 'fsspec', 'mpmath', 'MarkupSafe']
records = []
for name in NAMES:
    dist = metadata.distribution(name)
    source_root = Path(dist.locate_file('')).resolve()
    files = []
    for relative in dist.files:
        source = Path(dist.locate_file(relative)).resolve()
        if not source.is_relative_to(source_root) or not source.is_file():
            continue  # External console entry points are not needed by training.
        target = (TARGET / str(relative)).resolve()
        assert target.is_relative_to(TARGET.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert source.read_bytes() == target.read_bytes(), str(target)
        else:
            shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        files.append({'path': str(relative), 'sha256': digest})
    records.append({'distribution': name, 'version': dist.version,
                    'source': str(source_root), 'files': files})
(HERE / 'runtime-dependencies.json').write_text(json.dumps(records, indent=2))
print(json.dumps({r['distribution']: {'version': r['version'], 'files': len(r['files'])} for r in records}))
