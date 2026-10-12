"""Record local software and external helper inputs for reproducibility."""
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import hashlib,json,platform,shutil
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=H/'dependency-snapshot';out.mkdir(exist_ok=False)
    paths=[R/'src'/n for n in ['features.py','advanced_features.py','conditional_action_features.py','conditional_inference_worker.py','run_conditional_policy_screen.py','conditional_bucket_features.py','robust_pair_experiment.py','tail_pair_experiment.py']]
    paths += [H.parent/'softplay_fast_20260910'/n for n in ['evaluate_trial.py','diagnose_actions.py','action-diagnosis.json','remaining-mismatch.json']]
    paths += [H.parent/'softplay_conditional_20260910/controller-fit.json']
    receipt={}
    for p in paths:
        dest=out/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(p)==sha(dest)
        receipt[p.relative_to(R).as_posix()]={'sha256':sha(p),'snapshot':dest.relative_to(H).as_posix()}
    for folder in ['conditional-action-experiment','conditional-bucket-experiment']:
        p=R/'reports'/folder/'policy-screen/held.json';dest=out/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);receipt[p.relative_to(R).as_posix()]={'sha256':sha(p),'snapshot':dest.relative_to(H).as_posix()}
    data={'created_at':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'platform':platform.platform(),'packages':{n:version(n) for n in ['numpy','polars','scipy','lightgbm','scikit-learn','joblib','phevaluator','matplotlib']},'external_helpers':receipt,'pokerkit_vendor':'../history_realism_20260910/vendor','pokerkit_version':'0.7.5','note':'Snapshots preserve the helper bytes used by this run; all actual fit arrays and models are retained separately. Parent raw original data and model references remain in existing verified archives.'}
    (H/'dependencies.json').write_text(json.dumps(data,indent=2));print(json.dumps(data['packages']))
if __name__=='__main__':main()
