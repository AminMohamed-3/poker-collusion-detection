"""Preserve all local fits, including interrupted work, before remote import."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (HERE/'fold0').resolve()
DEST = (HERE/'local-migration-preserved/fold0').resolve()
assert SOURCE.is_relative_to(HERE) and DEST.is_relative_to(HERE)
DEST.mkdir(parents=True,exist_ok=False)
records=[]
for p in list(SOURCE.iterdir()):
    if p.is_dir() or p.name=='queue.json' or p.suffix=='.log':
        assert p.resolve().is_relative_to(SOURCE)
        if p.is_dir():
            receipts = p/'receipt.json'
            records.append({'fit':p.name,'complete':receipts.exists(),
                            'files':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}})
        p.rename(DEST/p.name)
receipt={'time':datetime.now(timezone.utc).isoformat(),'reason':'User requested new remote GPU. Original local fits were not scored. Full twelve-fit comparison restarts remotely.',
         'partial_fit_preserved':True,'held_predictions_evaluated':False,'runs':records,
         'new_remote_host':'vm-host.invalid:7368','alternate_host':'ssh9.vast.ai:19879'}
(HERE/'migration.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'preserved':len(records),'completed':sum(r['complete'] for r in records),'held_evaluation_performed':False}))
