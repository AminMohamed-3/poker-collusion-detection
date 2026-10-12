"""Audit the fresh sampler repair without changing prior audits or model trials."""
from pathlib import Path
import hashlib,json
import polars as pl
import audit_pilot
H=Path(__file__).resolve().parent;D=H/'history_repair'
audit_pilot.HERE=D
audit_pilot.main()
checks=[]
for folder in sorted((D/'pilot').glob('pool-*')):
    labels=pl.read_csv(folder/'development_labels.csv');e=pl.read_csv(folder/'development_evidence.csv');p=json.loads((folder/'provenance.json').read_text())
    assert all(p['shared_hands'][pid]>=20 for pid in labels['pair_id'])
    assert set(e['pair_id'])<=set(labels.filter(pl.col('label')==1)['pair_id'])
    for pid in labels.filter(pl.col('label')==1)['pair_id']:assert e.filter(pl.col('pair_id')==pid).height>=1
    h=pl.read_parquet(folder/'hands.parquet').select('hand_id');assert not e.join(h,on='hand_id',how='anti').height
    checks.append({'pool':folder.name,'pairs':labels.height,'evidence_rows':e.height,'minimum_labelled_shared_hands':min(p['shared_hands'][pid] for pid in labels['pair_id'])})
(D/'label-membership-audit.json').write_text(json.dumps({'passed':True,'pool_checks':checks,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
