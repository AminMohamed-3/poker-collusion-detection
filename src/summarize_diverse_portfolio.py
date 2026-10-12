"""Apply frozen portfolio gates; expose rejected candidates without rescuing them."""
import json
import numpy as np
import polars as pl
from diverse_next_day import OUT,write,locations
from robust_pair_experiment import ROOT,sha
from evidence_capacity_experiment import compare

def main():
    assert not (OUT/'admission.json').exists()
    jobs=json.loads((OUT/'plan.json').read_text())['jobs']; results={j:json.loads((OUT/j/'metrics.json').read_text()) for j in jobs}
    binaries={j:json.loads((OUT/j/'binary-metrics.json').read_text()) for j in jobs}
    coherent=json.loads((OUT/'coherent/metrics.json').read_text())
    risk={}; evidence={}
    for candidate in ['hybrid','seed_risk','history_risk','mixed','binary','binary_mixed']:
        source=binaries if candidate.startswith('binary') else results
        deltas=[]
        for job,r in source.items():
            rows={(x['history'],x['candidate']):x for x in r['pair_results']}
            for h in ['full','shortened']:
                deltas.append({'job':job,'history':h,**{k:rows[h,candidate][k]-rows[h,'baseline'][k] for k in ['pair_ap','sparse_ap','pair_behavior']}})
        cr={(x['job'],x['history'],x['candidate']):x for x in coherent['pair_results']}
        for j,h,c in cr:
            if c==candidate: deltas.append({'job':j,'history':h,**{k:cr[j,h,c][k]-cr[j,h,'baseline'][k] for k in ['pair_ap','sparse_ap','pair_behavior']}})
        known=[x for x in deltas if x['job'].startswith('fold') and not x['history'].startswith('coherent')]
        omit=[x for x in deltas if x['job'].startswith('omit_')]; stress=[x for x in deltas if x['job']=='audit' or x['history'].startswith('coherent')]
        def mean(rows,k): return float(np.mean([x[k] for x in rows]))
        def minimum(rows,k): return float(min(x[k] for x in rows))
        v10=[]
        for j in ['fold0','fold1','fold2']:
            _,_,md,_=locations(j); old=pl.read_csv(md/'pair-predictions.csv')
            label=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','label')
            from robust_pair_experiment import stable_ap
            for h in ['full','shortened']:
                original=old.filter((pl.col('candidate')=='v10_control')&(pl.col('history')==h)).join(label,on='pair_id',validate='1:1').sort('pair_id')
                if not original.height: original=old.filter((pl.col('candidate')=='v10')&(pl.col('history')==h)).join(label,on='pair_id',validate='1:1').sort('pair_id')
                assert original.height
                row=next(x for x in source[j]['pair_results'] if x['history']==h and x['candidate']==candidate)
                v10.append(row['pair_ap']-stable_ap(original['label'].to_numpy()>0,original['risk'].to_numpy()))
        tests={'known_mean_composite':mean(known,'pair_behavior')>=-.002,'known_worst_composite':minimum(known,'pair_behavior')>=-.006,'known_mean_sparse':mean(known,'sparse_ap')>=-.02,
            'omitted_mean_sparse':mean(omit,'sparse_ap')>=-.01,'omitted_worst_sparse':minimum(omit,'sparse_ap')>=-.05,'beats_v10_mean_pair_ap':np.mean(v10)>0,
            'stress_mean_composite':mean(stress,'pair_behavior')>=-.002,'stress_worst_composite':minimum(stress,'pair_behavior')>=-.006,'stress_mean_sparse':mean(stress,'sparse_ap')>=-.02}
        risk[candidate]={'passed':bool(all(tests.values())),'tests':{k:bool(v) for k,v in tests.items()},'known_mean_pair_behavior_delta':mean(known,'pair_behavior'),'omitted_mean_sparse_delta':mean(omit,'sparse_ap'),'omitted_worst_sparse_delta':minimum(omit,'sparse_ap'),'mean_pair_ap_vs_v10':float(np.mean(v10)),'stress_mean_pair_behavior_delta':mean(stress,'pair_behavior'),'deltas':deltas}
    for candidate in ['seed_evidence','history_evidence']:
        allknown=pl.concat([pl.read_csv(OUT/j/'evidence-pairs.csv') for j in ['fold0','fold1','fold2']])
        pooled={}
        for h in ['full','shortened']:
            rows=allknown.filter(pl.col('experiment')==h).group_by('pair_id','table_id','family','candidate').agg(pl.col('map5').mean()).to_dicts()
            pooled[h]=compare(rows,candidate,baseline='baseline')
        omitted=[results[j]['evidence_comparisons'][h][candidate]['delta'] for j in jobs if j.startswith('omit_') for h in ['full','shortened']]
        stress=[results['audit']['evidence_comparisons'][h][candidate]['delta'] for h in ['full','shortened']]+[v[candidate]['delta'] for v in coherent['evidence_comparisons'].values()]
        tests={'known_pooled_mean':min(v['delta'] for v in pooled.values())>=-.005,'known_family':min(d for v in pooled.values() for d in v['by_family_delta'].values())>=-.025,
            'omitted_mean':np.mean(omitted)>=-.005,'omitted_worst':min(omitted)>=-.03,'stress_mean':np.mean(stress)>=-.005,'stress_worst':min(stress)>=-.02}
        evidence[candidate]={'passed':bool(all(tests.values())),'tests':{k:bool(v) for k,v in tests.items()},'known_pooled':pooled,'omitted_deltas':omitted,'stress_deltas':stress}
    candidates={c:{'risk_method':c if c!='seed_evidence' else 'baseline','evidence_method':'seed_evidence' if c in ['mixed','seed_evidence'] else 'baseline',
        'eligible':bool((risk[c]['passed'] if c!='seed_evidence' else True) and (evidence['seed_evidence']['passed'] if c in ['mixed','seed_evidence'] else True))} for c in ['hybrid','seed_risk','history_risk','seed_evidence','mixed']}
    payload={'plan_sha256':sha(OUT/'plan.json'),'binary_plan_sha256':sha(OUT/'binary-plan.json'),'risk_gates':risk,'evidence_gates':evidence,'five_prespecified_candidates':candidates,
        'incumbent_unchanged':True,'selection_policy':'Do not fill the upload quota with rejected candidates. Prepare requested diagnostic files separately from eligible submissions. No gate was relaxed.',
        'stress_risk_gate':'Uses the same conservative known-fold tolerances for coherent/audit checks; this operational interpretation does not relax any parent gate.'}
    write(OUT/'admission.json',payload)
    print(json.dumps({'risk':{k:v['passed'] for k,v in risk.items()},'evidence':{k:v['passed'] for k,v in evidence.items()},'eligible':[c for c,v in candidates.items() if v['eligible']]}),flush=True)

if __name__=='__main__': main()
