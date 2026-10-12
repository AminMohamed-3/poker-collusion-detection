import sys; sys.dont_write_bytecode=True
import glob
import pyarrow.parquet as pq
R="C:/Data/Projects/Money Making/work/poker/"
files={
 "audit-oof": R+"Claude Version/experiments/label_process_20260911/audit-oof.parquet",
 "audit-struct": R+"Claude Version/experiments/label_process_20260911/audit-structural-features.parquet",
 "symcensor-scores": R+"Codex Resume/experiments/symmetry_censor_20260912/scores.parquet",
 "seq-dev": R+"reports/sequence-evidence-experiment/development.parquet",
 "eq-dev": R+"reports/equity-evidence-experiment/development.parquet",
 "ca-fold0": R+"reports/conditional-action-experiment/policy-screen/hand-features.parquet",
}
for k,f in files.items():
    try:
        m=pq.ParquetFile(f); names=m.schema_arrow.names
        print("==",k,"rows",m.metadata.num_rows,"ncols",len(names))
        print("  cols:",names if len(names)<=45 else names[:25]+["..."]+names[-10:])
    except Exception as e:
        print("==",k,"ERR",e)
raw=sorted(glob.glob(R+"outputs_v2/hands_development/batch_*.parquet"))
tot=sum(pq.ParquetFile(p).metadata.num_rows for p in raw)
n=pq.ParquetFile(raw[0]).schema_arrow.names
print("== raw v2 dev batches",len(raw),"rows",tot,"ncols",len(n))
print("  cols:",n)
import os
print(os.listdir(R+"reports/conditional-action-experiment/policy-screen"))
print(os.listdir(R+"reports/conditional-action-experiment"))
