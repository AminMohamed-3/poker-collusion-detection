"""Persist the current stage and next action for autonomous continuation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).resolve().parent;E=H.parent;W=H.parents[3]
def main():
    c=E/'softplay_coupling_20260910';complete=(c/'summary.json').exists()
    record={'at':datetime.now(timezone.utc).isoformat(),'stage':'coupling_development_complete' if complete else 'coupling_development_running','directory':str(c.resolve()),'automation_id':'improve-poker-augmentation','vm_required':False,'outer_validation_closed':True}
    if complete:
        s=json.loads((c/'summary.json').read_text());record['result']=s['comparisons'];record['next_step']=s['next_step']
    else:record['next_step']='Finish existing generation and run_analysis processes; do not duplicate them. Run compare.py only after diagnostics.json exists.'
    (H/'continuation-checkpoint.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
