"""Hash-bind the no-fit prefix policy-scoring stage."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2];S=H.parent/'screen';AP=H.parent/'action_policy_fast';PC=H.parent/'private_card_fast';PL=H.parent/'private_card_likelihood'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ex=json.loads((H/'EXTRACT_RECEIPT.json').read_text());assert ex['manifest_sha256']==sha(H/'MANIFEST.json');batches=sorted((H/'prefix_action_cache').glob('*.parquet'));assert len(batches)==len(ex['batches']) and sum(x['rows'] for x in ex['batches'])==103377
 for p,r in zip(batches,ex['batches']):assert p.name==r['file'] and sha(p)==r['sha256']
 sources=[H/'PROTOCOL.md',H/'MANIFEST.json',H/'EXTRACT_RECEIPT.json',H/'score_contexts.py',PL/'MANIFEST.json',PC/'CPU_RECEIPT.json',PC/'cpu_models_outer0.joblib',PC/'cpu_models_outer1.joblib',AP/'CPU_RECEIPT.json',AP/'policy_models_outer0_pipeline2.joblib',AP/'policy_models_outer1_pipeline2.joblib',AP/'fast129_context_fold0.npz',AP/'fast129_context_fold1.npz',S/'context_fold0_prefix60.npz',S/'context_fold1_prefix60.npz',S/'evaluator_fold0_prefix60.npz',S/'evaluator_fold1_prefix60.npz',S/'pf35_predictions.npz',*batches]
 m={'status':'prepared_no_fit_no_score','release':'SCORE_RELEASE.json','action_rows':103377,'policy_fits':0,'nuisance_fits':0,'prefix_primary':'frozen shared_hands>=38 eligible 664','prefix_secondary':813,'sources':{str(x.resolve()):sha(x) for x in sources}}
 (H/'SCORE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
