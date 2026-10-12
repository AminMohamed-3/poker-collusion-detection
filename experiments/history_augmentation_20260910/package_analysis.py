"""Package only training-source diagnostics and frozen feature references."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json,shutil,sys,tarfile
import numpy as np
import polars as pl
H=Path(__file__).resolve().parent; R=H.parents[1]; Q=H/'analysis_package/work/poker'
E=Q/'experiments/history_augmentation_20260910'
OLD=R/'experiments/history_realism_20260910'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def write(dst,s):dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(s)
assert not Q.exists()
for p in (R/'src').glob('*.py'):cp(p,Q/'src'/p.name)
for n in ['conditional-action-experiment/policy-screen/held-models.joblib','conditional-action-experiment/policy-screen/held.json','conditional-bucket-experiment/policy-screen/held.joblib','conditional-bucket-experiment/policy-screen/held.json','conditional-bucket-experiment/screen/pair.joblib','mil-pair-experiment/fold0/features.json']:
    cp(R/'reports'/n,Q/'reports'/n)
m=json.loads((R/'reports/mil-pair-experiment/fold0/metadata.json').read_text())
write(Q/'reports/mil-pair-experiment/fold0/metadata.json',json.dumps({'train_full':m['train_full']}))
a=np.load(R/'experiments/data_expansion/augmentation/synthetic/tree-arrays.npz')
dest=Q/'experiments/data_expansion/augmentation/synthetic/tree-arrays.npz';dest.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(dest,train_pair=a['train_pair'][:4209],train_y=a['train_y'][:4209])
for n in ['original-temporal-features.parquet','attendance-diagnosis.json']:
    cp(OLD/n,Q/'experiments/history_realism_20260910'/n)
for p in OLD.joinpath('vendor').rglob('*.py'):cp(p,Q/'experiments/history_realism_20260910/vendor'/p.relative_to(OLD/'vendor'))
for n in ['training-calibration.json','prepare_varied_features.py','varied_pair_domain_audit.py','audit_varied_pilot.py','replay_varied_pilot.py','temporal_audit.py','audit_seat_schedule.py','auditor-freeze.json']:
    cp(H/n,E/n)
# Compute the same deterministic original hand sample locally, then ship its
# aggregated diagnostics, never original outer-held or evaluation rows.
sys.path.insert(0,str(R/'experiments/data_expansion'))
import audit_pilot
cal=json.loads((H/'training-calibration.json').read_text())
oh=pl.read_parquet(OLD/'training-hands.parquet').with_columns(pl.col('started_at').cast(pl.String))
assert set(oh['table_id'])==set(cal['training_pools'])
oh=oh.with_columns(pl.col('hand_id').hash(seed=908117).alias('_order')).sort('table_id','_order').group_by('table_id',maintain_order=True).head(100).drop('_order')
ids=oh.select('hand_id').lazy()
ss=pl.scan_parquet(OLD/'training-seats.parquet').join(ids,on='hand_id',how='inner').collect()
aa=pl.scan_parquet(R/'data/actions.parquet').join(ids,on='hand_id',how='inner').collect()
original=audit_pilot.make_features(oh,ss,aa)
original.write_parquet(E/'original-hand-diagnostics.parquet')
s=(R/'experiments/data_expansion/audit_pilot.py').read_text()
start=s.index("    calibration=json.loads(");end=s.index("    sampled=synthetic",start)
s=s[:start]+"    original=pl.read_parquet(HERE.parent/'original-hand-diagnostics.parquet')\n"+s[end:]
write(Q/'experiments/data_expansion/audit_pilot.py',s)
receipt={'original_hand_diagnostic_rows':original.height,'training_pools':cal['training_pools'],'original_metadata_only_train_full':True,'original_arrays_only_training':True,'no_original_held_labels_or_evaluation_rows':True,'cached_auditor_change':'Replace original raw-data sample reconstruction with identical cached feature frame; all diagnostics and thresholds unchanged.','builder_sha256':sha(__file__)}
write(E/'reference-package-audit.json',json.dumps(receipt,indent=2))
files=[p for p in Q.rglob('*') if p.is_file()]
manifest={p.relative_to(H/'analysis_package').as_posix():sha(p) for p in files}
write(H/'analysis_package/analysis-manifest.json',json.dumps(manifest,indent=2))
with tarfile.open(H/'analysis-package.tar.gz','w:gz') as t:
    for p in files+[H/'analysis_package/analysis-manifest.json']:t.add(p,arcname=p.relative_to(H/'analysis_package'))
print(json.dumps({'files':len(files),'bytes':(H/'analysis-package.tar.gz').stat().st_size,'sha256':sha(H/'analysis-package.tar.gz')}))
