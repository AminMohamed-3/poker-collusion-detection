from __future__ import annotations
import hashlib, importlib.metadata as md, json, os, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODEX = HERE.parents[1]
VENV = CODEX / '.venv_tabpfn3'
BORROWED = CODEX.parent / '.venv_mhc' / 'Lib' / 'site-packages'
WHEEL = HERE / 'tabpfn-8.2.0-py3-none-any.whl'
CHECKPOINT = HERE / 'checkpoint' / 'tabpfn-v3-classifier-v3_default.ckpt'
EXPECTED_WHEEL = 'c9432bd301038764db210723e1a49569c4285310a2ecbecd1972241c9d6df290'
EXPECTED_CHECKPOINT = 'd0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def norm(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()

assert Path(sys.prefix).resolve() == VENV.resolve(), (sys.prefix, VENV)
assert sha(WHEEL) == EXPECTED_WHEEL
assert sha(CHECKPOINT) == EXPECTED_CHECKPOINT
packages = []
for dist in md.distributions():
    name = dist.metadata.get('Name') or dist.name
    location = Path(dist.locate_file('')).resolve()
    packages.append({
        'name': name,
        'normalized_name': norm(name),
        'version': dist.version,
        'location': str(location),
        'source': 'borrowed_read_only' if str(location).lower().startswith(str(BORROWED.resolve()).lower()) else 'isolated_environment',
    })
packages.sort(key=lambda x: (x['normalized_name'], x['location']))
by_name = {p['normalized_name']: p['version'] for p in packages}
required = ['tabpfn','torch','numpy','scipy','scikit-learn','pandas','einops','huggingface-hub','pydantic','pydantic-settings','joblib','tqdm','filelock','lightgbm','safetensors','typing-extensions']
assert by_name['tabpfn'] == '8.2.0'
assert by_name['torch'] == '2.11.0+cu128'
assert all(norm(x) in by_name for x in required)
installed = VENV / 'Lib' / 'site-packages'
files = {
    str(WHEEL): sha(WHEEL),
    str(CHECKPOINT): sha(CHECKPOINT),
    str(installed / 'tabpfn' / 'classifier.py'): sha(installed / 'tabpfn' / 'classifier.py'),
    str(installed / 'tabpfn' / 'checkpoint.py'): sha(installed / 'tabpfn' / 'checkpoint.py'),
    str(installed / 'tabpfn' / 'model_loading.py'): sha(installed / 'tabpfn' / 'model_loading.py'),
    str(BORROWED / 'torch-2.11.0+cu128.dist-info' / 'METADATA'): sha(BORROWED / 'torch-2.11.0+cu128.dist-info' / 'METADATA'),
    str(BORROWED / 'numpy-2.5.3.dist-info' / 'METADATA'): sha(BORROWED / 'numpy-2.5.3.dist-info' / 'METADATA'),
}
download_receipt = HERE / 'checkpoint' / 'download-receipt.json'
if download_receipt.exists():
    files[str(download_receipt)] = sha(download_receipt)
for extra_name in ('pypi-wheel-receipt.json','dependency-install-report.json','tabpfn-install-report.json','pip-upgrade-report.json','pip-freeze.txt','synthetic_benchmark.py','prepare_safe_checkpoint.py'):
    extra = HERE / extra_name
    if extra.exists():
        files[str(extra)] = sha(extra)
pip_check = subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, capture_output=True, text=True).stdout.strip()
receipt = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'environment': str(VENV.resolve()),
    'python': sys.version,
    'platform': platform.platform(),
    'pip': md.version('pip'),
    'tabpfn': md.version('tabpfn'),
    'torch': md.version('torch'),
    'cuda_build': '12.8',
    'read_through_path': str(BORROWED.resolve()),
    'existing_runtime_modified': False,
    'existing_runtime_metadata_unchanged': files[str(BORROWED / 'torch-2.11.0+cu128.dist-info' / 'METADATA')] == 'bb53a1e8e258707e738e108afaf1041304fa085a3775206876232345257e00b7',
    'direct_requirements': md.requires('tabpfn'),
    'required_versions': {norm(x): by_name[norm(x)] for x in required},
    'packages': packages,
    'files': files,
    'wheel_sha256': EXPECTED_WHEEL,
    'checkpoint_sha256': EXPECTED_CHECKPOINT,
    'checkpoint_bytes': CHECKPOINT.stat().st_size,
    'checkpoint_loading': {
        'format': 'PyTorch pickle .ckpt',
        'official_8_2_0_loader': 'torch.load(map_location="cpu", weights_only=None)',
        'weights_only_probe': 'not_run; conversion not required for present study',
        'safe_conversion_script': str(HERE / 'prepare_safe_checkpoint.py'),
    },
    'offline_benchmark_environment': {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1','HF_HUB_OFFLINE':'1'},
    'pip_check': pip_check or 'No broken requirements found.',
}
(HERE / 'runtime-receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: receipt[k] for k in ('environment','tabpfn','torch','checkpoint_bytes','pip_check')}, indent=2))

