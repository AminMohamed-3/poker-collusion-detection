"""Explain already-frozen training-only OOF action gains; no new fit or gate."""
from pathlib import Path
import json,hashlib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent;S=H.parent/'softplay_causal_style_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    f=pl.read_parquet(S/'training-actions-causal.parquet').filter(pl.col('behavior_family')=='soft_play')
    old=np.load(S/'own_style-oof.npz');new=np.load(S/'depth5_300-oof.npz')
    y=f['action_class'].to_numpy();g=f['table_id'].to_numpy().astype(str)
    for v in [old,new]:np.testing.assert_array_equal(y,v['y']);np.testing.assert_array_equal(g,v['groups'])
    counts=f.group_by('pair_id').len();w=1/f.select('pair_id').join(counts,on='pair_id',how='left')['len'].to_numpy()
    i=np.arange(len(y));loss0=-np.log(np.maximum(old['p'][i,y],1e-12));loss1=-np.log(np.maximum(new['p'][i,y],1e-12))
    conditions={'all':np.ones(len(y),dtype=bool)}
    for name in ['street_index','action_class','facing_partner']:
        for value in sorted(f[name].unique()):conditions[f'{name}={value}']=f[name].to_numpy()==value
    for name in ['own_strength','price']:
        edges=[0,.25,.5,.75,1.000001] if name=='own_strength' else [0,.000001,.1,.25,1.000001]
        for a,b in zip(edges[:-1],edges[1:]):conditions[f'{name}=[{a},{b})']=(f[name].to_numpy()>=a)&(f[name].to_numpy()<b)
    rows=[]
    for name,take in conditions.items():
        if not take.any():continue
        rows.append({'condition':name,'actions':int(take.sum()),'pools':len(set(g[take])),'baseline_loss':float(np.average(loss0[take],weights=w[take])),'capacity_loss':float(np.average(loss1[take],weights=w[take])),'loss_gain':float(np.average((loss0-loss1)[take],weights=w[take]))})
    out={'training_only':True,'no_new_fit':True,'exploratory_subgroups_do_not_change_any_gate':True,'baseline_oof_sha256':sha(S/'own_style-oof.npz'),'capacity_oof_sha256':sha(S/'depth5_300-oof.npz'),'rows':rows}
    (H/'action-error-analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
