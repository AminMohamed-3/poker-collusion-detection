"""Bind the unreleased integration probe sources; performs no fit."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2];L=H.parent/'private_card_likelihood'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 src=[P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet',P/'data/development_labels.csv',L/'fit.py',L/'MANIFEST.json',L/'FIT_RECEIPT.json',L/'table_membership.parquet',L/'row_membership.parquet',L/'capped_training_keys.parquet',L/'predictions_outer0.parquet',L/'predictions_outer1.parquet',*sorted((P/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))]
 art=[H/'PLAN.md',H/'prepare.py',H/'run.py',H/'evaluate.py'];m={'status':'prepared_unreleased','quality_read':False,'sources':{str(x.resolve()):sha(x) for x in src},'artifacts':{x.name:sha(x) for x in art},'fit_count':24,'deadline_seconds':900}
 (H/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
