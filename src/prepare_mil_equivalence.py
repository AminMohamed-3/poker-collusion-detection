"""Expected uncompressed archive members, derived only from frozen local arrays."""
import hashlib
import json
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]/'reports/mil-pair-experiment'
for job in ['omit_soft_play','omit_coordinated_isolation','omit_directed_transfer']:
    package=root/job/'package'; archive=package/'arrays.npz'
    with zipfile.ZipFile(archive) as z: members={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    value={'members':members,'original_worker_plan_sha256':hashlib.sha256((package/'worker-plan.json').read_bytes()).hexdigest(),'original_array_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (root/job/'expected-npy-members.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'job':job,'members':len(members)}),flush=True)
