import sys,json,hashlib
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making');RT=ROOT/'work/poker/goal_top5_20260919/tabldm';OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_screen';EXPECTED=(ROOT/'work/poker/Codex Resume/.venv_tabpfn3/Scripts/python.exe').resolve()
assert Path(sys.executable).resolve()==EXPECTED
sys.path.insert(0,str(RT/'overlay'))
import tabldm,torch,numpy,sklearn,scipy,einops
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prov=json.loads((RT/'receipts/provenance_manifest.json').read_text());checked=[]
for rel,h in prov['official_source']['tracked_files_sha256'].items():
 if rel.startswith('tabldm/') and rel.endswith('.py'):assert sha(RT/'overlay'/rel)==h;checked.append(rel)
mod=Path(tabldm.__file__).resolve();assert RT/'overlay' in mod.parents
ein=Path(einops.__file__).resolve();assert ROOT/'work/poker/Codex Resume/.venv_tabpfn3/Lib/site-packages' in ein.parents
rec={'status':'IMPORT_ONLY_PASS_NO_MODEL_INSTANTIATED','python_executable':str(EXPECTED),'pythonpath_overlay':str((RT/'overlay').resolve()),'tabldm_module':str(mod),'einops_module':str(ein),'versions':{'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'einops':einops.__version__},'pinned_tabldm_python_files_verified':len(checked)}
(OUT/'import_preflight_receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
