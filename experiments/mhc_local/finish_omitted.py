from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
plan_sha=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest();rows=[]
for family in ['directed_transfer','soft_play','coordinated_isolation']:
    r=json.loads((H/('omit_'+family)/'evaluation.json').read_text());assert r['plan_sha256']==plan_sha and len(r['receipts'])==6
    for row in r['metrics']:
        if row['candidate'] in ['mhc_equal_blend','single_pass_equal_blend']:
            rows.append({'omitted_family':family,'history':row['history'],'variant':row['candidate'].removesuffix('_equal_blend'),**row['delta']})
decisions={}
for variant in ['mhc','single_pass']:
    rr=[r for r in rows if r['variant']==variant];mean=float(np.mean([r['sparse_ap'] for r in rr]));worst=min(r['sparse_ap'] for r in rr)
    decisions[variant]={'mean_sparse_ap_delta':mean,'worst_sparse_ap_delta':worst,'passed_omitted_gate':mean>=0 and worst>=-.05,'production_admitted':False}
result={'plan_sha256':plan_sha,'fits':18,'rows':rows,'decisions':decisions,'reused_development_splits':True}
out=H/'omitted-summary.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
lines=['# Omitted-family confirmation','','Eighteen fixed GPU fits: two mHC modes, three fixed seeds and three omitted behavior families. Reported risks are the frozen equal blend of the three-seed neural mean and incumbent; incumbent behavior/evidence remain unchanged.','','| Omitted family | History | Variant | Sparse AP change | Pair AP change |','|---|---|---|---:|---:|']
for r in rows:lines.append(f"| {r['omitted_family']} | {r['history']} | {r['variant']} | {r['sparse_ap']:+.6f} | {r['pair_ap']:+.6f} |")
lines+=['','Required: mean sparse AP change ≥ 0 and worst change ≥ −.05.','','| Variant | Mean | Worst | Passed |','|---|---:|---:|---|']
for v,d in decisions.items():lines.append(f"| {v} | {d['mean_sparse_ap_delta']:+.6f} | {d['worst_sparse_ap_delta']:+.6f} | {d['passed_omitted_gate']} |")
lines+=['','Passing would require further known-fold/history/audit confirmation. A failure rejects promotion; it does not justify changing the thresholds or choosing a favorable seed. These are reused development splits, not an untouched estimate. No production file or submission changed.','']
(H/'OMITTED.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(decisions,indent=2))
