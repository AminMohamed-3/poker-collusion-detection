"""Label-blind contract check. Does not fit or read held relevance."""
import hashlib,json
from pathlib import Path
import numpy as np,polars as pl
O=Path(__file__).resolve().parent; UP=O.parent/'citation_nested'; PEER=Path(r'C:/Data/Projects/Money Making/work/poker/Codex Resume/experiments/tabm_citation_pilot_20260919')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (O/'PREFLIGHT_OK.json').exists() and not (O/'predictions_frozen.json').exists() and not (O/'results.json').exists()
 m=json.loads((UP/'clean_manifest.json').read_text());p=UP/m['artifact'];assert sha(p)==m['artifact_sha256'];assert m['old_artifact_used'] is False and m['fallback_used'] is False and m['zero_missing_raw'] is True and m['excluded_audit_pools']==104 and m['table_overlap']==m['player_overlap']==m['physical_hand_overlap']==0 and m['nested_reference_fits']==4 and m['bins_training_only'] is True;assert set(m['support_semantics'])=={'pre_count','post_count','pre_support','post_support','pre_available','post_available'} and set(m['residual_semantics'])=={'pre_sum','post_sum','pre_max','post_max'}
 d=pl.read_parquet(p);K=['pair_id','hand_id'];required=K+['split','target']+m['raw_features']+m['support_features']+m['residual_features'];assert d.columns==required and len(m['raw_features'])==662 and len(m['support_features'])==6 and len(m['residual_features'])==4
 tr=d.filter(pl.col('split')=='train').sort(K);te=d.filter(pl.col('split')=='audit').sort(K);assert tr.height==10438 and tr['pair_id'].n_unique()==85 and int(tr['target'].sum())==408 and te.height==940 and te['pair_id'].n_unique()==47 and te['target'].null_count()==te.height
 old=np.load(PEER/'context.npz',allow_pickle=False);tk=pl.read_parquet(PEER/'training-keys.parquet').select(K);ak=pl.read_parquet(PEER/'audit-keys.parquet').select(K);assert tr.select(K).equals(tk) and te.select(K).equals(ak) and np.array_equal(tr['target'].to_numpy().astype(np.float32),old['y_train']);X=d.select(m['raw_features']+m['support_features']+m['residual_features']).to_numpy();assert np.isfinite(X).all()
 out={'ok':True,'quality_read':False,'fit_count':0,'clean_manifest_sha256':sha(UP/'clean_manifest.json'),'artifact_sha256':sha(p),'peer_context_sha256':sha(PEER/'context.npz'),'rows':{'train':tr.height,'audit':te.height},'columns':{'control':668,'candidate':672}};(O/'PREFLIGHT_OK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
