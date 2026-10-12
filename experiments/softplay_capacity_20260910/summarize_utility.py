"""Report every prespecified capacity-augmentation comparison without selection."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;N=H.parent/'softplay_capacity_utility_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    e=json.loads((N/'trial/evaluation.json').read_text());p=e['primary']['capacity'];f=json.loads((N/'trial/freeze.json').read_text());plan=json.loads((N/'study-plan.json').read_text())
    assert all(sha(H.parents[1]/path)==v for path,v in plan['protected'].items())
    out={'at':datetime.now(timezone.utc).isoformat(),'passes_utility_gate':p['passes_utility_gate'],'mean_against_original':p['original']['mean_deltas'],'mean_against_resampling':p['mean_resampling']['mean_deltas'],'original_loss_interval':p['original']['pool_bootstrap_logloss_improvement_95pct_interval'],'resampling_loss_interval':p['mean_resampling']['pool_bootstrap_logloss_improvement_95pct_interval'],'common_unique_targets_per_population':f['common_dose'],'fits':e['fit_count'],'fit_seconds':e['fit_seconds'],'protected_submissions_unchanged':True,'reused_validation':True,'production_admitted':False,'evaluation_sha256':sha(N/'trial/evaluation.json')}
    (N/'summary.json').write_text(json.dumps(out,indent=2))
    lines=['# Capacity-policy augmentation utility','',('The prespecified utility screen passes.' if p['passes_utility_gate'] else 'The prespecified utility screen fails; retain the realism gain without claiming reliable detector benefit.'),'',
      f"All four generated populations passed the unchanged realism screen before this test. {e['fit_count']} fixed CPU fits took {e['fit_seconds']:.1f}s. Each augmented arm adds {f['common_dose']} unique target pairs, with full/70%/40% history views, to the original4,209 training rows. Three fixed real-resampling controls have the same additional target count. Three model seeds and all four populations contribute equally; no favorable subset is selected.",
      '', '| Baseline | Mean logloss gain | Pair AP change | Sparse AP change | 95% pool interval for loss gain | Gate |','|---|---:|---:|---:|---|---|']
    for b in ['original','mean_resampling']:
        v=p[b];m=v['mean_deltas'];ci=v['pool_bootstrap_logloss_improvement_95pct_interval'];lines.append(f"| {b} | {m['logloss_improvement']:+.7f} | {m['pair_ap']:+.7f} | {m['sparse_ap']:+.7f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] | {'PASS' if v['utility_gate']['passed'] else 'FAIL'} |")
    lines+=['','| Population | Loss gain vs original | Loss gain vs mean resampling |','|---|---:|---:|']
    for a in plan['generator_arms']:lines.append(f"| {a} | {p['original']['by_batch'][a]['mean_deltas']['logloss_improvement']:+.7f} | {p['mean_resampling']['by_batch'][a]['mean_deltas']['logloss_improvement']:+.7f} |")
    lines+=['','All gate checks:','']
    for b in ['original','mean_resampling']:lines.append(f"- {b}: "+'; '.join(k+'='+str(v) for k,v in p[b]['utility_gate']['checks'].items()))
    lines+=['','Real-resampling controls versus original-only:','']
    for b,v in e['resampling_vs_original'].items():lines.append(f"- {b}: loss gain {v['mean_deltas']['logloss_improvement']:+.7f}.")
    lines+=['','This uses the repeatedly examined100-pool/457-pair outer validation set. It is not an untouched confirmation. Pool intervals condition on the fitted models; repeated histories, seeds and counterfactuals are not independent real-world observations. All generator choices were confined to the original training pools, and all24 fits/predictions were frozen before this evaluation. Thin-history features retain full-history style references, as in prior studies.',
      '', 'The four ordinary-policy families share a learned source; fresh simulator seeds are not external data. Changed legal actions provide traceable intervention evidence, but realistic simulation and actual predictive utility are separate requirements. No neural run, VM, upload or production replacement occurred. All seven protected submission files are unchanged.','']
    (N/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4));labels=plan['generator_arms'];x=list(range(len(labels)))
    for b,label in [('original','Versus original'),('mean_resampling','Versus real resampling')]:ax.plot(x,[p[b]['by_batch'][a]['mean_deltas']['logloss_improvement'] for a in labels],'o-',label=label)
    ax.axhline(0,color='gray',linewidth=1);ax.set_xticks(x,['Development0','Development1','Development2','Fresh confirmation']);ax.set_ylabel('Logloss improvement');ax.set_title('Does more realistic generated data improve the detector?');ax.legend();fig.tight_layout();fig.savefig(N/'utility-screen.png',dpi=160);plt.close(fig);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
