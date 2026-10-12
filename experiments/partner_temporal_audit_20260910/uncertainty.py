"""Pool-resampled uncertainty for the descriptive temporal gap; no new gate."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent
def main():
    plan={'at':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'resamples':2000,'seed':24041,'estimand':'Equal-pair original minus pooled synthetic temporal moments; resample whole pools independently within each source.','limitation':'Conditional on observed simulator policy and source populations. Nine exploratory temporal contrasts are not a multiplicity-controlled discovery test or independent real-world validation.'};(H/'uncertainty-plan.json').write_text(json.dumps(plan,indent=2));real=pl.read_parquet(H/'original-pairs.parquet');syn=pl.concat([pl.read_parquet(H/(n+'-pairs.parquet')) for n in ['b0_sized','b1_sized','fresh_sized']]);columns=[c for c in real.columns if '_lag' in c];rng=np.random.default_rng(24041);rows=[]
    def boot(frame):
        grouped=frame.group_by('table_id').agg(pl.len().alias('n'),*[pl.col(c).sum() for c in columns]).sort('table_id');ids=rng.integers(0,grouped.height,(2000,grouped.height));totals=grouped.select(columns).to_numpy();n=grouped['n'].to_numpy();return totals[ids].sum(1)/n[ids].sum(1)[:,None]
    diff=boot(real)-boot(syn)
    for j,c in enumerate(columns):rows.append({'metric':c,'original_mean':float(real[c].mean()),'synthetic_mean':float(syn[c].mean()),'gap':float(real[c].mean()-syn[c].mean()),'conditional_pool_interval95':np.quantile(diff[:,j],[.025,.975]).tolist()})
    out={'training_only':True,'rows':rows,'not_a_new_gate':True,'not_multiplicity_adjusted':True};(H/'uncertainty.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
