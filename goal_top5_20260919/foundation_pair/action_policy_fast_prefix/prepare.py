"""Lineage audit only; never creates prefix contexts from the full-history cache."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2];S=H.parent/'screen';AP=H.parent/'action_policy_fast';PC=H.parent/'private_card_fast';PL=H.parent/'private_card_likelihood'
STYLE=['own_style_fold','own_style_check','own_style_call','own_style_raise','partner_style_fold','partner_style_check','partner_style_call','partner_style_raise']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lm=json.loads((PL/'MANIFEST.json').read_text());assert all(x in lm['inputs'] for x in STYLE)
 src=P/'src/build_conditional_action_cache.py';text=src.read_text();assert 'all other development hands' in text and 'totals = actions' in text and '- current_counts[player]' in text
 tiny=json.loads((H/'TINY_PREFLIGHT.json').read_text());assert tiny['passed'] and tiny['nonstyle_exact_columns']==32
 sources=[H/'PROTOCOL.md',H/'generate.py',H/'TINY_PREFLIGHT.json',src,P/'src/conditional_action_features.py',P/'Claude Version/src/pu_pairs/step40_prefix_audit.py',P/'data/hands.parquet',P/'data/actions.parquet',P/'data/seats.parquet',P/'data/development_labels.csv',PL/'MANIFEST.json',PC/'CPU_RECEIPT.json',PC/'cpu_models_outer0.joblib',PC/'cpu_models_outer1.joblib',AP/'CPU_RECEIPT.json',AP/'policy_models_outer0_pipeline2.joblib',AP/'policy_models_outer1_pipeline2.joblib',AP/'fast129_context_fold0.npz',AP/'fast129_context_fold1.npz',S/'context_fold0_prefix60.npz',S/'context_fold1_prefix60.npz',S/'evaluator_fold0_prefix60.npz',S/'evaluator_fold1_prefix60.npz',S/'pf35_predictions.npz']
 result={'status':'ready_for_hash_bound_prefix_extraction','saved_cache_forbidden':True,'extraction':{'hand_universe':'development hands with within-table started_at rank/len < 0.6','style_counts':'retained prefix hands only; subtract current hand; add-one smoothing unchanged','pair_scope':'813 confirmed prefix intersection only','output':'streamed table batches; no raw copies','release':'EXTRACT_RELEASE.json'},'tiny_preflight':tiny,'reusable_after_extraction':['private_card_fast outer pipeline2 full strength model','action_policy_fast outer pipeline2 real target/normal policies','completed nested Fast129 training contexts'],'new_fits_before_gpu':0,'future_gpu_fits':2,'globalpu':'historical held-development table-prefix population. Build unknown co-seated pairs from held development tables and use per-PU-fold nuisance/TN policies trained with those physical tables excluded; never treat it as evaluation-phase validation. The confirmed-813 extraction does not extract GlobalPU.','sources':{str(x.resolve()):sha(x) for x in sources}}
 (H/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
