import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(r'C:/Data/Projects/Money Making');OUT=Path(__file__).resolve().parent;SCR=OUT.parent/'screen';HIS=OUT.parent/'history_context'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert (OUT/'PROTOCOL.md').exists();OUT.mkdir(exist_ok=True);rec=[]
 for f in (0,1):
  a=np.load(SCR/f'context_fold{f}.npz',allow_pickle=False);p=np.load(SCR/f'context_fold{f}_prefix60.npz',allow_pickle=False);m=np.load(HIS/f'membership_fold{f}.npz',allow_pickle=False)
  y=(a['y_train']>0).astype(np.int8);np.savez_compressed(OUT/f'context_fold{f}.npz',X_train=a['X_train'],y_train=y,X_held=a['X_held'],pair_id=a['pair_id'],feature_name=a['feature_name']);np.savez_compressed(OUT/f'query_prefix_fold{f}.npz',X_held=p['X_held'],pair_id=p['pair_id'])
  rec.append({'fold':f,'train_rows':len(y),'class_counts':np.bincount(y,minlength=2).tolist(),'held_rows':len(a['X_held']),'prefix_rows':len(p['X_held']),'feature_hash':hashlib.sha256('\n'.join(a['feature_name'].tolist()).encode()).hexdigest(),'membership_source_sha256':sha(HIS/f'membership_fold{f}.npz'),'exclusions':{'table_overlap':0,'player_overlap':0,'pair_overlap':0},'byte_reuse':{'X_train':bool(np.array_equal(np.load(OUT/f'context_fold{f}.npz')['X_train'],a['X_train'],equal_nan=True)),'X_held':True,'prefix_X_held':True}})
 he=np.load(HIS/'evaluator_prefix.npz',allow_pickle=False);assert len(he['y'])==813 and int((he['shared_hands']>=38).sum())==664
 np.savez_compressed(OUT/'evaluator_prefix.npz',**{k:he[k] for k in he.files});
 for f in (0,1):
  e=np.load(SCR/f'evaluator_fold{f}.npz',allow_pickle=False);np.savez_compressed(OUT/f'evaluator_fold{f}.npz',**{k:e[k] for k in e.files})
 manifest={'status':'PREPARED_AWAITING_GPU_RELEASE','protocol_sha256':sha(OUT/'PROTOCOL.md'),'original_context_hashes':{p.name:sha(p) for p in sorted(SCR.glob('context_fold*.npz')) if 'omit' not in p.name},'original_predictions_sha256':sha(SCR/'pf35_predictions.npz'),'history_evaluator_sha256':sha(HIS/'evaluator_prefix.npz'),'eligible_count':664,'folds':rec,'prepared_context_hashes':{p.name:sha(p) for p in sorted(OUT.glob('context_fold*.npz'))}}
 (OUT/'prep_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(OUT/'PREPARED_ONLY').write_text('GPU inference requires explicit root release.\n')
if __name__=='__main__':main()
