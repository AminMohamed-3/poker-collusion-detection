import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; POKER=HERE.parents[2]; PRIV=HERE.parent/'private_card_likelihood'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    sources=[PRIV/'MANIFEST.json',PRIV/'FIT_RECEIPT.json',PRIV/'fit.py',PRIV/'table_membership.parquet',PRIV/'row_membership.parquet',PRIV/'capped_training_keys.parquet',PRIV/'models_outer0.joblib',PRIV/'models_outer1.joblib',PRIV/'predictions_outer0.parquet',PRIV/'predictions_outer1.parquet',POKER/'data/development_labels.csv',*sorted((POKER/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))]
    artifacts=[HERE/'PROTOCOL.md',HERE/'prepare.py',HERE/'run.py',HERE/'evaluate.py']
    m={'status':'prepared_unreleased','fit_count':8,'deadline_seconds':900,'sources':{str(p.resolve()):sha(p) for p in sources},'artifacts':{p.name:sha(p) for p in artifacts}}
    (HERE/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
