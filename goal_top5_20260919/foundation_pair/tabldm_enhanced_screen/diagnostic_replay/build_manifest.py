import hashlib,json
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making');P=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_enhanced_screen';O=P/'diagnostic_replay'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[P/'freeze.json',P/'run_tabldm.py',P/'predictions.npz',P/'runtime.json',P/'context_fold0.npz',P/'context_fold0_prefix60.npz',ROOT/'work/poker/goal_top5_20260919/tabldm/overlay/tabldm/_sklearn/classifier.py',ROOT/'work/poker/goal_top5_20260919/tabldm/checkpoint/clf_default.ckpt',O/'PROTOCOL.md',O/'run_diagnostic.py']
man={'status':'PREPARED_AWAITING_DIAGNOSTIC_RELEASE','scope':'one fold0 technical replay; no labels/quality','original_strict_failure_preserved':True,'bound_files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files},'thresholds':{'max_abs_probability':1e-5,'mean_abs_probability':1e-6,'argmax_changes':0}}
(O/'MANIFEST.json').write_text(json.dumps(man,indent=2)+'\n');print(sha(O/'MANIFEST.json'))
