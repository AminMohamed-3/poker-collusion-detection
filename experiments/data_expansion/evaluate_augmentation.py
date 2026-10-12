"""Score every complete predeclared trial arm locally, once, without selection."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'augmentation';sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import LABELS
from evaluate_mil_pairs import scorecard
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ce(y,p):return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
def main():
    out=O/'evaluation.json';assert not out.exists()
    plan=json.loads((O/'plan.json').read_text());metadata=json.loads((R/'experiments/mhc_local/fold0/held-metadata.json').read_text())
    audit=json.loads((R/'experiments/mhc_local/fold0/data-audit.json').read_text());assert sha(R/'experiments/mhc_local/fold0/held-metadata.json')==audit['held_metadata_sha256']
    assert all(sha(R/p)==h for p,h in plan['protected'].items())
    data={};receipts=[]
    for arm in plan['arms']:
        for variant in plan['variants']:
            for seed in plan['seeds']:
                d=O/arm/f'{variant}-{seed}';r=json.loads((d/'receipt.json').read_text())
                assert r['plan_sha256']==sha(O/'plan.json') and not r['held_labels_used'] and r['reload_exact']
                assert sha(d/'model.pt')==r['model_sha256'] and sha(d/'predictions.npz')==r['predictions_sha256']
                data[(arm,variant,seed)]=dict(np.load(d/'predictions.npz'));receipts.append({k:v for k,v in r.items() if k!='history'})
        for variant in ['tree897','tree1217']:
            d=O/arm/variant;r=json.loads((d/'receipt.json').read_text());assert r['plan_sha256']==sha(O/'plan.json') and not r['held_labels_used']
            assert sha(d/'model.joblib')==r['model_sha256'] and sha(d/'predictions.npz')==r['predictions_sha256']
            data[(arm,variant,0)]=dict(np.load(d/'predictions.npz'));receipts.append(r)
    rows=[];seedrows=[]
    for history in ['full','shortened']:
        m=metadata[history];y=np.array([LABELS[p['behavior_family']] for p in m['pairs']]);roles=np.array(m['control_roles']);base=scorecard(y,np.array(m['control_risk']),roles)
        rows.append({'history':history,'arm':'existing_incumbent','variant':'baseline',**base})
        for variant in plan['primary_data_endpoints']:
            seeds=plan['seeds'] if variant in plan['variants'] else [0]
            for arm in plan['arms']:
                probabilities=np.stack([data[(arm,variant,s)][history] for s in seeds]);p=probabilities.mean(0)
                for seed,prob in zip(seeds,probabilities):seedrows.append({'history':history,'arm':arm,'variant':variant,'seed':seed,'log_loss':ce(y,prob),**scorecard(y,1-prob[:,0],roles)})
                rows.append({'history':history,'arm':arm,'variant':variant,'log_loss':ce(y,p),**scorecard(y,1-p[:,0],roles)})
    decisions={}
    for variant in plan['primary_data_endpoints']:
        diffs=[]
        for history in ['full','shortened']:
            for seed in plan['seeds'] if variant in plan['variants'] else [0]:
                b=next(r for r in seedrows if r['variant']==variant and r['history']==history and r['seed']==seed and r['arm']=='resampled_control');t=next(r for r in seedrows if r['variant']==variant and r['history']==history and r['seed']==seed and r['arm']=='synthetic')
                diffs.append({'history':history,'seed':seed,'logloss_improvement':b['log_loss']-t['log_loss'],**{k:t[k]-b[k] for k in ['pair_ap','pair_behavior','sparse_ap']}})
        mean={k:float(np.mean([r[k] for r in diffs])) for k in ['logloss_improvement','pair_ap','pair_behavior','sparse_ap']}
        worst_ap=min(np.mean([r['pair_ap'] for r in diffs if r['history']==h]) for h in ['full','shortened']);worst_pb=min(r['pair_behavior'] for r in diffs)
        passed=(mean['logloss_improvement']>=.001 and mean['pair_ap']>=0 and worst_ap>=-.001) if variant in plan['variants'] else (mean['pair_behavior']>=.0001 and worst_pb>=-.003 and mean['sparse_ap']>=0)
        decisions[variant]={'passed_data_screen':bool(passed),'mean_deltas':mean,'worst_history_mean_ap_delta':float(worst_ap),'worst_pair_behavior_delta':float(worst_pb),'cases':diffs}
    result={'plan_sha256':sha(O/'plan.json'),'evaluator_sha256':sha(__file__),'decisions':decisions,'metrics':rows,'seed_metrics':seedrows,'receipts':receipts,'production_admitted':False,'all_protected_submissions_unchanged':True,'not_an_untouched_holdout':True}
    out.write_text(json.dumps(result,indent=2));print(json.dumps(decisions,indent=2))
if __name__=='__main__':main()
