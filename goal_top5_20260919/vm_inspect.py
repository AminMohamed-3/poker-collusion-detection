import importlib.metadata
import json
import pathlib
import shutil
import sys
import torch

mem = {}
for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
    key, value = line.split(':', 1)
    if key in ('MemTotal', 'MemAvailable'):
        mem[key + '_kB'] = int(value.strip().split()[0])
packages = {}
for name in ['torch', 'tabpfn', 'skrub', 'numpy', 'scikit-learn', 'scipy', 'safetensors']:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'python': sys.version, 'executable': sys.executable,
                  'packages': packages, 'torch_cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0),
                  'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                  'memory': mem, 'workspace_free_bytes': shutil.disk_usage('/workspace').free}, indent=2))
