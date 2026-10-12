"""A separately frozen global union of complementary detectors; no family gates."""
import argparse
from datetime import datetime,timezone
import json
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha,LABELS
from evaluate_mil_pairs import scorecard

OUT=ROOT/'reports/tabpfn-v2-experiment'
RECIPES=[f'TabPFN{w}_{op}' for w in [128,256] for op in ['max','noisyor']]
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def freeze():
    path=OUT/'union-plan.json'; assert not path.exists()
    value={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'parent_plan_sha256':sha(OUT/'plan.json'),
        'adaptation':'Equal blends failed due to a large omitted-soft-play loss despite positive mean omitted gains. New hypothesis: globally combine complementary detections by maximum or noisy OR rather than downweight a confident detector. This trial was proposed after those outcomes; earlier equal blends remain rejected.',
        'recipes':RECIPES,'formulas':{'max':'max(incumbent_risk,TabPFN_risk)','noisyor':'1-(1-incumbent_risk)*(1-TabPFN_risk)'},
        'interpretation':'These are ranking scores, not claims of independent or calibrated posterior probabilities. Apply the same formula to every pair; no family, score-threshold or ID gates.',
        'gate':{'known_mean_delta_min':-.0005,'known_worst_delta_min':-.003,'known_mean_sparse_delta_min':-.01,'omitted_mean_sparse_delta_min':.01,'omitted_worst_sparse_delta_min':-.03,'stress_mean_delta_min':0.,'stress_worst_delta_min':-.003},
        'progression':'Evaluate frozen four recipes on existing fold0 and three omitted tests. Passing ones require fold1/2, audit169 and all six coherent-history checks before any production fit. All partitions reused, no untouched estimate or public fitting.',
        'initial_inputs':{str(p.relative_to(OUT)):sha(p) for job in ['fold0','omit_soft_play','omit_coordinated_isolation','omit_directed_transfer'] for p in [OUT/job/'pair-predictions.csv',OUT/job/'metadata.json',OUT/job/'screen.json']}}
    write(path,value); print(json.dumps({'union_plan_sha256':sha(path)}))


def evaluate():
    plan=json.loads((OUT/'union-plan.json').read_text()); assert plan['source_sha256']==sha(__file__)
    assert all(sha(OUT/p)==h for p,h in plan['initial_inputs'].items())
    assert not (OUT/'union-screen.json').exists(); rows=[]; allpred=[]
    for job in ['fold0','omit_soft_play','omit_coordinated_isolation','omit_directed_transfer']:
        data=pl.read_csv(OUT/job/'pair-predictions.csv'); meta=json.loads((OUT/job/'metadata.json').read_text())
        for history,m in meta.items():
            frame=data.filter(pl.col('history')==history).pivot(on='candidate',index=['pair_id','table_id','role'],values='risk').sort('pair_id')
            assert frame['pair_id'].to_list()==[p['pair_id'] for p in m['pairs']]
            y=np.array([LABELS[p['behavior_family']] for p in m['pairs']]); control=frame['baseline'].to_numpy(); roles=frame['role'].to_numpy(); baseline=scorecard(y,control,roles)
            for width in [128,256]:
                second=frame[f'TabPFN{width}_diagnostic'].to_numpy()
                for method,risk in [('max',np.maximum(control,second)),('noisyor',1-(1-control)*(1-second))]:
                    candidate=f'TabPFN{width}_{method}'; result=scorecard(y,risk,roles); rows.append({'job':job,'history':history,'candidate':candidate,**result,'delta':{k:result[k]-baseline[k] for k in result}})
                    allpred.extend({'pair_id':pid,'job':job,'history':history,'candidate':candidate,'risk':float(v)} for pid,v in zip(frame['pair_id'],risk))
    gate=plan['gate']; decisions={}
    for candidate in RECIPES:
        known=[r for r in rows if r['candidate']==candidate and r['job']=='fold0']; omitted=[r for r in rows if r['candidate']==candidate and r['job']!='fold0']
        kmean=float(np.mean([r['delta']['pair_behavior'] for r in known])); kworst=min(r['delta']['pair_behavior'] for r in known); ksparse=float(np.mean([r['delta']['sparse_ap'] for r in known])); omean=float(np.mean([r['delta']['sparse_ap'] for r in omitted])); oworst=min(r['delta']['sparse_ap'] for r in omitted)
        passed=kmean>=gate['known_mean_delta_min'] and kworst>=gate['known_worst_delta_min'] and ksparse>=gate['known_mean_sparse_delta_min'] and omean>=gate['omitted_mean_sparse_delta_min'] and oworst>=gate['omitted_worst_sparse_delta_min']
        decisions[candidate]={'passed_initial_checks':passed,'known_mean_delta':kmean,'known_worst_delta':kworst,'known_sparse_mean_delta':ksparse,'omitted_sparse_mean_delta':omean,'omitted_sparse_worst_delta':oworst}
    pl.DataFrame(allpred).write_csv(OUT/'union-predictions.csv'); result={'plan_sha256':sha(OUT/'union-plan.json'),'decisions':decisions,'metrics':rows}; write(OUT/'union-screen.json',result); print(json.dumps(decisions,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); args=p.parse_args(); freeze() if args.freeze else evaluate()
