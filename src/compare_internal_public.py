"""Retrospective matched-component comparison. Never fits a prediction model."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from datetime import datetime,timezone
import json
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from robust_pair_experiment import ROOT,sha

OUT=ROOT/'reports/internal-public-20260910'
def read(p): return json.loads((ROOT/p).read_text())
def main():
    OUT.mkdir(exist_ok=False)
    official=read('reports/evidence-public-calibration/submissions-observed.json')
    scored={x['fileName']:x for x in official['submissions']}
    files={p.name:p for d in ROOT.glob('outputs*') if d.is_dir() for p in d.glob('*.csv') if p.name in scored}
    assert len(scored)==len(files)==15
    seq=read('reports/sequence-evidence-experiment/screen.json'); sr={x['candidate']:x['map5'] for x in seq['results'] if x['experiment']=='known'}
    cap=read('reports/evidence-capacity-experiment/metrics.json'); eq=read('reports/equity-evidence-experiment/metrics.json')
    rrf=read('reports/evidence-ensemble-experiment/metrics.json'); ca=read('reports/conditional-action-experiment/screen-results/metrics.json')
    panel=[('sequence','submission_event_tails.csv','submission_sequence_evidence.csv',sr['sequence']-sr['current']),
        ('capacity','submission_sequence_evidence.csv','submission_capacity.csv',cap['comparisons']['larger_regularized']['delta']),
        ('equity','submission_sequence_evidence.csv','submission_equity.csv',eq['comparison']['delta']),
        ('rank_fusion','submission_sequence_evidence.csv','submission_ensemble.csv',rrf['split_gains']['rrf3']['screen']),
        ('conditional_evidence','submission_tabicl.csv','submission_conditional_evidence.csv',ca['evidence_comparison']['delta'])]
    rows=[]
    for name,before,after,local in panel:
        a=pl.read_csv(files[before],infer_schema=False).sort('pair_id'); b=pl.read_csv(files[after],infer_schema=False).sort('pair_id')
        assert a.select('pair_id','risk_score','predicted_behavior').equals(b.select('pair_id','risk_score','predicted_behavior')),(before,after)
        delta=float(scored[after]['publicScore'])-float(scored[before]['publicScore'])
        rows.append({'change':name,'before':before,'after':after,'internal_fold0_evidence_delta':local,'public_composite_delta':delta,'implied_public_evidence_delta':delta/.2,
            'unchanged_risk_behavior_verified':True,'before_sha256':sha(files[before]),'after_sha256':sha(files[after]),'direction_agrees':bool(np.sign(local)==np.sign(delta))})
    x=np.array([r['internal_fold0_evidence_delta'] for r in rows]); y=np.array([r['implied_public_evidence_delta'] for r in rows])
    correlations={'n':len(rows),'pearson':float(np.corrcoef(x,y)[0,1]),'spearman':float(spearmanr(x,y).statistic),
        'direction_agreement':sum(r['direction_agrees'] for r in rows),'leave_one_change_out_pearson':[float(np.corrcoef(np.delete(x,i),np.delete(y,i))[0,1]) for i in range(len(rows))]}
    allscores=[{'file':r['fileName'],'ref':r['ref'],'score':float(r['publicScore']),'sha256':sha(files[r['fileName']])} for r in sorted(scored.values(),key=lambda x:x['date'])]
    writeup=['# Internal/public comparison — descriptive only','',
        'Fifteen official entries are complete; best .86005. Daily quota is exhausted. These data are a record of selected, related experiments, not an independent sample for fitting blend weights or predicting the private leaderboard.','',
        'The table uses the same reused screening partition for five evidence-only changes. Every pair ID, risk score and behavior cell was checked unchanged in each before/after CSV. Public differences can therefore be divided by the evidence weight .20; display rounding gives approximately ±.00005 uncertainty in each implied difference.','',
        '| Change | Internal evidence Δ | Public composite Δ | Implied public evidence Δ |','|---|---:|---:|---:|']
    for r in rows: writeup.append(f"| {r['change']} | {r['internal_fold0_evidence_delta']:+.6f} | {r['public_composite_delta']:+.5f} | {r['implied_public_evidence_delta']:+.5f} |")
    writeup+=['',f"Exploratory Pearson r = {correlations['pearson']:.3f}; Spearman = {correlations['spearman']:.3f}; directions agree in {correlations['direction_agreement']}/5 changes. Leaving out one change gives Pearson from {min(correlations['leave_one_change_out_pearson']):.3f} to {max(correlations['leave_one_change_out_pearson']):.3f}. These are descriptive numbers, with no claimed significance or forecast.",'',
        'Capacity improved slightly internally but declined publicly; equity declined on this screen yet improved publicly. Their additional reused confirmation gains were +.006090 and +.012951, respectively. The conditional evidence gain was +.039204 after averaging overlapping known-fold results per unique pair, versus about +.01590 publicly. These discrepancies argue against treating tiny held-out changes as reliable improvements.','',
        'The five contrasts share data, baselines and model-selection history. Some production rerankers cover only selected high-risk pairs while local checks score all annotated targets. Development/evaluation history length and class prevalence differ. None of these correlations corrects those differences. Raw public and internal composite levels are therefore not regressed against each other.','',
        'Risk-only v11 gained +.01263 publicly; it changes both PairAP and BehaviorMAP, so the gain cannot be divided by .70 alone. The separate evidence-only gain was +.00318. Their sum predicts .85932 for the combined entry, agreeing with .85933 within rounding. Behavior added +.00072. Coverage tied at displayed precision; no claim about hidden labels or private equivalence follows.','',
        'The next-day recipes, blend weights and admission gates were frozen separately before their held-out results. This report does not modify them. Preserve all outcomes, including losses; submit only whole-model predictions, never label probes; retain pool separation, omitted-family checks and coherent-history stress tests.','',
        '## Official history','', '| File | Public score |', '|---|---:|']
    writeup += [f"| {r['file']} | {r['score']:.5f} |" for r in allscores]
    (OUT/'REPORT.md').write_text('\n'.join(writeup)+'\n',encoding='utf-8')
    payload={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'official_sha256':sha(ROOT/'reports/evidence-public-calibration/submissions-observed.json'),'matched_evidence_changes':rows,'correlations':correlations,'official_history':allscores,'no_weights_fitted':True,'next_day_plan_sha256':sha(ROOT/'reports/diverse-next-day/plan.json')}
    (OUT/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n'); pl.DataFrame(rows).write_csv(OUT/'matched_changes.csv')
    print(json.dumps(correlations),flush=True)

if __name__=='__main__': main()
