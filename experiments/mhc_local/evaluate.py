"""Evaluate complete predeclared local models, preserving every outcome."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(ROOT/'src'))
from robust_pair_experiment import sha,LABELS
from evaluate_mil_pairs import scorecard

def ce(y,prob): return float(-np.log(np.maximum(prob[np.arange(len(y)),y],1e-12)).mean())
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--job',default='fold0'); parser.add_argument('--variants',nargs='+'); args=parser.parse_args()
    plan=json.loads((HERE/'plan.json').read_text()); variants=args.variants or list(plan['variants']); root=HERE/args.job; output=root/'evaluation.json'; assert not output.exists()
    assert all(sha(ROOT/p)==h for p,h in plan['protected'].items())
    audit=json.loads((root/'data-audit.json').read_text()); assert sha(root/'held-metadata.json')==audit['held_metadata_sha256']; meta=json.loads((root/'held-metadata.json').read_text())
    arrays={}; receipts=[]
    for variant in variants:
        for seed in plan['seeds']:
            dest=root/(variant+'-'+str(seed)); r=json.loads((dest/'receipt.json').read_text())
            assert r['plan_sha256']==sha(HERE/'plan.json') and r['data_audit_sha256']==sha(root/'data-audit.json') and r['reload_exact'] and not r['held_labels_used']
            assert sha(dest/'model.pt')==r['model_sha256'] and sha(dest/'predictions.npz')==r['predictions_sha256']
            arrays[(variant,seed)]=dict(np.load(dest/'predictions.npz')); receipts.append({k:v for k,v in r.items() if k!='history'})
    metrics=[]; seed_metrics=[]; predictions=[]
    for history in ['full','shortened']:
        m=meta[history]; y=np.array([LABELS[p['behavior_family']] for p in m['pairs']]); base=np.array(m['control_risk']); roles=np.array(m['control_roles']); control=scorecard(y,base,roles)
        metrics.append({'history':history,'candidate':'baseline',**control,'delta':{k:0. for k in control}})
        for variant in variants:
            probabilities=np.stack([arrays[(variant,s)][history] for s in plan['seeds']]); prob=probabilities.mean(0)
            for s,p in zip(plan['seeds'],probabilities): seed_metrics.append({'history':history,'variant':variant,'seed':s,'log_loss':ce(y,p),'accuracy':float((p.argmax(1)==y).mean()),**scorecard(y,1-p[:,0],roles)})
            for name,risk in [(variant+'_diagnostic',1-prob[:,0]),(variant+'_equal_blend',(base+1-prob[:,0])/2)]:
                score=scorecard(y,risk,roles); metrics.append({'history':history,'candidate':name,**score,'delta':{k:score[k]-control[k] for k in score}})
                predictions.extend({'history':history,'candidate':name,'pair_id':pair['pair_id'],'table_id':pair['table_id'],'risk':float(v)} for pair,v in zip(m['pairs'],risk))
    comparisons={}
    for variant in ['mhc','single_pass']:
        for baseline in ['residual','residual_budget']:
            if variant not in variants or baseline not in variants: continue
            differences=[]
            for h in ['full','shortened']:
                for s in plan['seeds']:
                    v=next(r for r in seed_metrics if r['history']==h and r['seed']==s and r['variant']==variant); b=next(r for r in seed_metrics if r['history']==h and r['seed']==s and r['variant']==baseline)
                    differences.append({'history':h,'seed':s,'logloss_improvement':b['log_loss']-v['log_loss'],'pair_ap_delta':v['pair_ap']-b['pair_ap']})
            comparisons[variant+'_vs_'+baseline]={'cases':differences,'mean_logloss_improvement':float(np.mean([r['logloss_improvement'] for r in differences])),'mean_pair_ap_delta':float(np.mean([r['pair_ap_delta'] for r in differences]))}
    gate=plan['kaggle_gate']; base_mean=np.mean([r['pair_behavior'] for r in metrics if r['candidate']=='baseline']); required=min(.001,.2*max(0,.8-base_mean)); decisions={}
    for variant in variants:
        rows=[r for r in metrics if r['candidate']==variant+'_equal_blend']; mean=float(np.mean([r['delta']['pair_behavior'] for r in rows])); worst=min(r['delta']['pair_behavior'] for r in rows); sparse=float(np.mean([r['delta']['sparse_ap'] for r in rows]))
        decisions[variant]={'screen_passed':mean>=required and worst>=gate['worst_pair_behavior_delta_min'] and sparse>=gate['mean_sparse_ap_delta_min'],'mean_pair_behavior_delta':mean,'required_mean_gain':required,'worst_pair_behavior_delta':worst,'mean_sparse_delta':sparse,'production_admitted':False}
    arch=comparisons.get('single_pass_vs_residual_budget'); primary=None
    if arch: primary=arch['mean_logloss_improvement']>=.005 and arch['mean_pair_ap_delta']>=-.002
    value={'plan_sha256':sha(HERE/'plan.json'),'job':args.job,'primary_architecture_test_passed':primary,'seed_metrics':seed_metrics,'comparisons':comparisons,'metrics':metrics,'kaggle_screen':decisions,'receipts':receipts,'all_protected_submissions_unchanged':True,'not_an_untouched_holdout':True}
    output.write_text(json.dumps(value,indent=2)+'\n'); (root/'pair-predictions.json').write_text(json.dumps(predictions)+'\n')
    print(json.dumps({'job':args.job,'primary_architecture_test_passed':primary,'comparisons':{k:{a:b for a,b in v.items() if a!='cases'} for k,v in comparisons.items()},'kaggle_screen':decisions}),flush=True)

if __name__=='__main__': main()
