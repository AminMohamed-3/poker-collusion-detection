import site;site.addsitedir(r'C:/Data/Projects/Money Making/work/poker/.venv/Lib/site-packages')
import hashlib,json,io
import numpy as np,polars as pl
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2];S=H.parent/'screen';AP=H.parent/'action_policy_fast';PC=H.parent/'private_card_fast';PL=H.parent/'private_card_likelihood'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 b=io.BytesIO();np.savez_compressed(b,X_train=np.zeros((2,131),np.float32),X_held=np.zeros((1,131),np.float32),feature_name=np.array(['x']*128+['llr_dt','llr_sp','llr_ci']));b.seek(0);assert np.load(b,allow_pickle=False)['X_train'].shape==(2,131);q=io.BytesIO();pl.DataFrame(schema={'pair_id':pl.String,'direction':pl.Int64,'hand_id':pl.String,'action_no':pl.Int64,'table_id':pl.String,'actor_id':pl.String,'weight':pl.Float64}).write_parquet(q);q.seek(0);assert pl.read_parquet(q).width==7
 sources=[H/'PROTOCOL.md',H/'cpu_generate.py',AP/'cpu_generate.py',P/'data/development_labels.csv',P/'Claude Version/experiments/crossphase_20260911/lab_on_dev.parquet',PL/'MANIFEST.json',PL/'row_membership.parquet',PL/'capped_training_keys.parquet',PC/'CPU_RECEIPT.json',PC/'cpu_models_outer0.joblib',PC/'cpu_models_outer1.joblib',AP/'MANIFEST.json',AP/'CPU_RECEIPT.json',*[AP/f'policy_models_outer{o}_pipeline{j}.joblib' for o in [0,1] for j in [0,1,2]],*[AP/f'action_predictions_outer{o}_pipeline{j}.parquet' for o in [0,1] for j in [0,1,2]],*[AP/f'train_membership_outer{o}_pipeline{j}_normal.parquet' for o in [0,1] for j in [0,1,2]],S/'context_fold0.npz',S/'context_fold1.npz',S/'pf35_predictions.npz',*sorted((P/'reports/conditional-action-experiment/action-cache').glob('*.parquet'))]
 m={'status':'prepared_no_fit','release':'CPU_RELEASE.json','new_family_action_fits':18,'reused_normal_policies':6,'reused_nuisance_pipelines':6,'future_fast_fits':2,'primary_comparator':'completed fixed 80/20 Fast129 action_policy_fast on identical keys','sources':{str(x.resolve()):sha(x) for x in sources}};(H/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
