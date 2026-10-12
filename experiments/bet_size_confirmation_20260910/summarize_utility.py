"""Report both predeclared doses, preserving every failed test."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;N=H.parent/'bet_size_utility_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    e=json.loads((N/'trial/evaluation.json').read_text());p=e['primary']['sized'];pool=e['secondary_pooling'];f=json.loads((N/'trial/freeze.json').read_text());plan=json.loads((N/'study-plan.json').read_text());assert all(sha(H.parents[1]/n)==v for n,v in plan['protected'].items())
    def compact(c,key):return {b:{'mean_deltas':c[b]['mean_deltas'],'loss_interval':c[b]['pool_bootstrap_logloss_improvement_95pct_interval'],'gate':c[b][key]} for b in ['original','mean_resampling']}
    result={'at':datetime.now(timezone.utc).isoformat(),'primary_passes':p['passes_utility_gate'],'secondary_pooling_passes':pool['passes_screen'],'low_dose_unique_targets':f['common_dose'],'pooled_unique_targets':pool['unique_generated_targets'],'primary':compact(p,'utility_gate'),'secondary_pooling':compact(pool,'screen'),'incremental_loss_gain_beyond_matched_resampling':pool['incremental_loss_gain_beyond_matched_resampling'],'fits':e['fit_count'],'fit_seconds':e['fit_seconds'],'reused_validation':True,'production_admitted':False,'protected_submissions_unchanged':True,'evaluation_sha256':sha(N/'trial/evaluation.json')};(N/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Conditional-size augmentation: fixed small and pooled doses','',f"The primary per-population test {'passes' if p['passes_utility_gate'] else 'fails'}. The secondary pooled-dose screen {'passes' if pool['passes_screen'] else 'fails'}. Both were prescribed before confirmation outcomes and before any new utility score. Neither is production admission.",'',f"All33 CPU tree fits took{e['fit_seconds']:.1f}s. Small dose: {f['common_dose']} unique generated targets per population. Pooled dose: {pool['unique_generated_targets']} unique targets across all3 populations. Each target contributes full/70%/40% views. Each dose has3 predetermined equal-size real-resampling controls. Every model uses all original4,209 rows;3 fixed model seeds are averaged without selection.",'', '| Dose | Comparison | Loss gain | Pair AP change | Sparse AP change | 95% pool interval | Gate |','|---|---|---:|---:|---:|---|---|']
    for dose,c,key in [('Small',p,'utility_gate'),('Pooled',pool,'screen')]:
        for baseline in ['original','mean_resampling']:
            v=c[baseline];m=v['mean_deltas'];ci=v['pool_bootstrap_logloss_improvement_95pct_interval'];lines.append(f"| {dose} | {baseline} | {m['logloss_improvement']:+.7f} | {m['pair_ap']:+.7f} | {m['sparse_ap']:+.7f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] | {'PASS' if v[key]['passed'] else 'FAIL'} |")
    lines+=['','| Small-dose population | Loss gain vs original | Loss gain vs matched resampling |','|---|---:|---:|']
    for a in plan['generator_arms']:lines.append(f"| {a} | {p['original']['by_batch'][a]['mean_deltas']['logloss_improvement']:+.7f} | {p['mean_resampling']['by_batch'][a]['mean_deltas']['logloss_improvement']:+.7f} |")
    lines+=['','All fixed gate checks:','']
    for dose,c,key in [('Small',p,'utility_gate'),('Pooled',pool,'screen')]:
        for baseline in ['original','mean_resampling']:lines.append('- '+dose+'/'+baseline+': '+'; '.join(k+'='+str(v) for k,v in c[baseline][key]['checks'].items()))
    lines+=['',f"The descriptive increase in loss gain beyond dose-matched resampling is {pool['incremental_loss_gain_beyond_matched_resampling']:+.7f}. Pooling changes both training amount and population diversity, so this is not an isolated causal estimate of either one.",'','The100-pool/457-pair outer validation has been repeatedly reused. The intervals condition on fitted models. Seeds, thin views and counterfactual simulations are not independent validation observations. Thin views retain full-history style references. The pooled dose is one combined sample; a favorable pooled screen would require prospective replication rather than selecting it after results. All generator changes used original training pools, and every prediction was frozen before this single evaluation. Earlier capacity-only utility failure remains unchanged.','', 'No VM, neural training, new submission or production change. All protected submissions remain unchanged.','']
    (N/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4))
    for baseline,label in [('original','Versus original'),('mean_resampling','Versus matched real resampling')]:
        values=[p[baseline]['mean_deltas']['logloss_improvement'],pool[baseline]['mean_deltas']['logloss_improvement']];cis=[p[baseline]['pool_bootstrap_logloss_improvement_95pct_interval'],pool[baseline]['pool_bootstrap_logloss_improvement_95pct_interval']];ax.errorbar([0,1],values,yerr=[[v-c[0] for v,c in zip(values,cis)],[c[1]-v for v,c in zip(values,cis)]],fmt='o-',capsize=4,label=label)
    ax.axhline(0,color='gray',linewidth=1);ax.set_xticks([0,1],[f"Small: {f['common_dose']} targets",f"Pooled: {pool['unique_generated_targets']} targets"]);ax.set_ylabel('Logloss improvement');ax.set_title('Does a larger, more varied augmentation dose help?');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(N/'dose-screen.png',dpi=160);plt.close(fig);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
