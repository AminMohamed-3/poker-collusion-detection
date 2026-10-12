"""Fixed evidence-only portfolio after risk variants fail omitted-family gates."""
import os
os.environ['POLARS_MAX_THREADS']='2'
import argparse,json
from datetime import datetime,timezone
import numpy as np
import polars as pl
from diverse_next_day import OUT,write
from robust_pair_experiment import ROOT,sha
from relative_evidence_experiment import records
from evidence_capacity_experiment import compare

NAMES=['seed_evidence','history_evidence','seed_history_mean','history_rank_fusion','three_way_mean']

def scores(frame,base,seed,history):
    ranks=frame.select('pair_id').with_columns(pl.Series('base',base),pl.Series('history',history))
    fusion=ranks.select((.5/(10+pl.col('base').rank(method='average',descending=True).over('pair_id'))+.5/(10+pl.col('history').rank(method='average',descending=True).over('pair_id'))).alias('v'))['v'].to_numpy()
    return {'baseline':base,'seed_evidence':seed,'history_evidence':history,'seed_history_mean':.5*(seed+history),'history_rank_fusion':fusion,'three_way_mean':(base+seed+history)/3}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); args=parser.parse_args()
    path=OUT/'evidence-panel-plan.json'
    if args.freeze:
        assert not path.exists()
        admission=json.loads((OUT/'admission.json').read_text())
        assert all(admission['evidence_gates'][k]['passed'] for k in NAMES[:2])
        write(path,{'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'parent_admission_sha256':sha(OUT/'admission.json'),
            'reason':'All proposed risk variants fail frozen omitted-family checks; risk and family assignment remain exactly v13. Both evidence recipes passed the frozen evidence admission gates.',
            'candidates':{'seed_evidence':'Mean scores from seeds 7329,39117,55193, each trained on full histories.',
                'history_evidence':'One ranker trained on full/.7/.4 histories; pair context recomputed and total query-view weight balanced per pair.',
                'seed_history_mean':'Equal score mean of seed_evidence and history_evidence.',
                'history_rank_fusion':'Equal reciprocal-rank fusion, fixed offset 10, of incumbent and history_evidence.',
                'three_way_mean':'Equal score mean of incumbent, seed_evidence and history_evidence.'},
            'gates':'Unchanged evidence admission gates from parent plan, full and short checked separately, all three omitted families plus coherent/audit confirmations. Do not rescue a failed mixture.',
            'diversity_limit':'Five ranking recipes using related LightGBM models and two training strategies, not five independent algorithms. Portfolio exploration, not a claim each improves v13.',
            'production':'Use unchanged v13 selected 16060 pairs and identical risk/behavior; preserve all evidence outside selection. No public-score changes to weights or coverage.',
            'adaptive_disclosure':'This portfolio is chosen after earlier risk failures and observed component diagnostics. New mixture formulas are fixed before their own outcomes. All local validation is reused.'}); return
    assert sha(__file__)==json.loads(path.read_text())['source_sha256']
    dest=OUT/'evidence-panel'; dest.mkdir(exist_ok=False)
    metadata=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').select('pair_id','table_id').join(pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','behavior_family'),on='pair_id',validate='1:1')
    annotation=pl.read_csv(ROOT/'data/development_evidence.csv').select('pair_id','hand_id'); allrows=[]
    jobs=json.loads((OUT/'plan.json').read_text())['jobs']
    views=[(j,h,OUT/j,h+'-') for j in jobs for h in ['full','shortened']]+[(j,h,OUT/'coherent',j+'_'+h+'-') for j in ['fold0','fold1','fold2'] for h in ['coherent_65','coherent_40']]
    for job,history,folder,prefix in views:
        base=pl.read_parquet(folder/(prefix+'baseline-evidence.parquet')); seed=pl.read_parquet(folder/(prefix+'seed_evidence-evidence.parquet')); aug=pl.read_parquet(folder/(prefix+'history_evidence-evidence.parquet'))
        assert base.select('pair_id','hand_id').equals(seed.select('pair_id','hand_id')) and base.select('pair_id','hand_id').equals(aug.select('pair_id','hand_id'))
        frame=base.select('pair_id','hand_id').join(metadata,on='pair_id',validate='m:1',maintain_order='left')
        truth={k[0]:set(g['hand_id']) for k,g in annotation.join(frame.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi').group_by('pair_id')}
        assert set(frame['pair_id'])==set(truth)
        computed=scores(frame,base['score'].to_numpy(),seed['score'].to_numpy(),aug['score'].to_numpy())
        for candidate,prediction in computed.items():
            rows=records(frame,prediction,truth,history,candidate)
            allrows.extend([{**r,'job':job} for r in rows])
        frame.select('pair_id','hand_id').with_columns(*[pl.Series(k,v) for k,v in computed.items()]).write_parquet(dest/(job+'_'+history+'-scores.parquet'))
    data=pl.DataFrame(allrows); data.write_csv(dest/'evidence-pairs.csv'); gates={}
    for candidate in NAMES:
        known={}
        for history in ['full','shortened']:
            rows=data.filter(pl.col('job').is_in(['fold0','fold1','fold2'])&(pl.col('experiment')==history)).group_by('pair_id','table_id','family','candidate').agg(pl.col('map5').mean()).to_dicts()
            known[history]=compare(rows,candidate,baseline='baseline')
        comparisons={job+'_'+history:compare(data.filter((pl.col('job')==job)&(pl.col('experiment')==history)).to_dicts(),candidate,baseline='baseline') for job,history,_,_ in views}
        omitted=[v['delta'] for k,v in comparisons.items() if k.startswith('omit_')]
        stress=[v['delta'] for k,v in comparisons.items() if k.startswith('audit_') or 'coherent_' in k]
        tests={'known_pooled_mean':min(v['delta'] for v in known.values())>=-.005,'known_family':min(d for v in known.values() for d in v['by_family_delta'].values())>=-.025,
            'omitted_mean':np.mean(omitted)>=-.005,'omitted_worst':min(omitted)>=-.03,'stress_mean':np.mean(stress)>=-.005,'stress_worst':min(stress)>=-.02}
        gates[candidate]={'passed':bool(all(tests.values())),'tests':{k:bool(v) for k,v in tests.items()},'known_pooled':known,'omitted_mean':float(np.mean(omitted)),'omitted_worst':min(omitted),'stress_mean':float(np.mean(stress)),'stress_worst':min(stress),'comparisons':comparisons}
    write(dest/'admission.json',{'plan_sha256':sha(path),'gates':gates,'eligible':[k for k,v in gates.items() if v['passed']]})
    print(json.dumps({k:{'passed':v['passed'],'full':v['known_pooled']['full']['delta'],'short':v['known_pooled']['shortened']['delta'],'omitted_mean':v['omitted_mean'],'stress_mean':v['stress_mean']} for k,v in gates.items()}),flush=True)

if __name__=='__main__': main()
