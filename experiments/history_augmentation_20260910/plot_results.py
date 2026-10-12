from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
H=Path(__file__).resolve().parent
s=json.loads((H/'summary.json').read_text());d=s['decisions'];names=list(d)
labels={'residual_budget':'Residual','mhc':'mHC','single_pass':'SinglePass','tree897':'Tree (897)','tree1217':'Tree (1,217)'}
interval={r['variant']:r['pool_bootstrap_95_interval'] for r in s['uncertainty']}
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for j,(ax,key,title) in enumerate(zip(axes,['logloss_improvement','pair_ap'],['Held-data log-loss reduction','Held-data pair AP change (thousandths)'])):
    values=[d[n]['mean_deltas'][key]*(1000 if j else 1) for n in names]
    ax.barh([labels[n] for n in names],values,color=['#24765f' if v>=0 else '#ac4c3b' for v in values])
    ax.invert_yaxis();ax.axvline(0,color='#444',lw=.8);ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(title,loc='left',fontsize=12);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    if not j:
        for i,n in enumerate(names):ax.plot(interval[n],[i,i],color='#333',lw=1.4)
    ax.set_xlabel('Positive favors calibrated augmentation')
fig.suptitle('Repaired synthetic histories versus matched real-data resampling',x=.08,ha='left',fontsize=14)
fig.text(.08,.015,'All seeds and both histories included. Loss intervals resample whole pools; reused validation data.',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.045,1,.93]);fig.savefig(H/'augmentation-results.png',dpi=180);plt.close(fig)
