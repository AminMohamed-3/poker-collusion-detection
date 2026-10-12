from datetime import datetime,timezone
import json
from robust_pair_experiment import ROOT,sha
out=ROOT/'reports/conditional-bucket-experiment'; plan=json.loads((out/'plan.json').read_text()); assert not (out/'policy-screen').exists() and not (out/'input-path-repair.json').exists()
value={'created_at':datetime.now(timezone.utc).isoformat(),'original_plan_sha256':sha(out/'plan.json'),'reason':'Original policy-plan refers to its remote bundle layout. Resolve local label, held-pair and worker-source paths to existing originals, retaining exact original hash checks. Failure occurred before any new reference fit or held result. No features/parameters/gates changed.','sources':{p:sha(ROOT/'src'/p) for p in plan['sources']}}
(out/'input-path-repair.json').write_text(json.dumps(value,indent=2)+'\n'); print(json.dumps(value))
