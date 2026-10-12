"""Secondary training-only mismatch audit across all four populations."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).resolve().parent;N=H.parent/'softplay_capacity_diagnosis_20260910'
assert json.loads((H/'summary.json').read_text())['all_three_realism_gates_passed']
assert json.loads((H.parent/'softplay_capacity_confirmation_20260910/summary.json').read_text())['passes_realism']
N.mkdir(exist_ok=False)
s=(H.parent/'softplay_coupling_diagnosis_20260910/diagnose.py').read_text()
s=s.replace("P=H.parent/'softplay_coupling_20260910'","P=H.parent/'softplay_capacity_20260910';C=H.parent/'softplay_capacity_confirmation_20260910'")
s=s.replace('b0_coupled','b0_capacity').replace("for arm in ['b0_capacity','b1_coupled']:","parents={'b0_capacity':P/'b0_capacity','b1_capacity':P/'b1_capacity','b2_capacity':P/'b2_capacity','fresh':C/'fresh'}\n    for arm in parents:")
s=s.replace("P/arm/'features/", "parents[arm]/'features/")
s=s.replace("P/arm/'soft-actions.parquet'", "parents[arm]/'soft-actions.parquet'")
s=s.replace("P/a/'soft-actions.parquet'", "parents[a]/'soft-actions.parquet'")
(N/'diagnose.py').write_text(s,encoding='utf-8')
plan={'at':datetime.now(timezone.utc).isoformat(),'purpose':'Secondary training-only source audit pooling all four qualified populations. Does not change frozen utility gates or select a favorable source classifier.','source_sha256':hashlib.sha256((N/'diagnose.py').read_bytes()).hexdigest(),'fixed_split_seeds':[19083,27031,35027],'original_pools':297,'outer_held_access':False}
(N/'plan.json').write_text(json.dumps(plan,indent=2));print(str(N))
