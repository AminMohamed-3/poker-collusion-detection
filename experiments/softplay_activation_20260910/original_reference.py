"""Post-screen diagnostic against a verified, unchanged original-only model."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];OLD=H.parent/'softplay_fast_20260910';sys.path.insert(0,str(OLD))
from evaluate_trial import ce,metrics
sys.path.insert(0,str(R/'src'))
from robust_pair_experiment import LABELS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    previous=json.loads((OLD/'trial/original/receipt.json').read_text());assert sha(OLD/'trial/original/predictions.npz')==previous['predictions_sha256'] and sha(OLD/'trial/original/model.joblib')==previous['model_sha256'];a=np.load(OLD/'trial/original.npz');b=np.load(H/'trial/resampled.npz')
    np.testing.assert_array_equal(a['train_x'],b['train_x'][:4209]);np.testing.assert_array_equal(a['train_y'],b['train_y'][:4209])
    for tag in ['full','shortened']:np.testing.assert_array_equal(a[tag+'_x'],b[tag+'_x'])
    assert json.loads((OLD/'study-plan.json').read_text())['tree_parameters']==json.loads((H/'study-plan.json').read_text())['tree_parameters']
    ref=dict(np.load(OLD/'trial/original/predictions.npz'));meta=json.loads((R/'experiments/mhc_local/fold0/held-metadata.json').read_text());data={}
    for tag in ref:
        y=np.array([LABELS[r['behavior_family']] for r in meta[tag]['pairs']]);g=np.array([r['table_id'] for r in meta[tag]['pairs']]);data[tag]=(y,g)
    pools=sorted(set(data['full'][1]));boot=np.random.default_rng(20319).integers(0,len(pools),(1000,len(pools)));out={}
    for mode in ['resampled','gated','calibrated','continuous']:
        pred=dict(np.load(H/'trial'/mode/'predictions.npz'));cases=[];tot=[];count=[]
        for tag,(y,g) in data.items():
            rm=metrics(y,ref[tag]);m=metrics(y,pred[tag]);cases.append({'history':tag,'reference_loss':rm['log_loss'],'new_loss':m['log_loss'],'loss_improvement':rm['log_loss']-m['log_loss'],'pair_ap_delta':m['pair_ap']-rm['pair_ap'],'sparse_ap_delta':m['sparse_ap']-rm['sparse_ap']})
        for pool in pools:
            d=[]
            for tag,(y,g) in data.items():d.extend((ce(y,ref[tag])-ce(y,pred[tag]))[g==pool])
            tot.append(sum(d));count.append(len(d))
        estimates=np.array(tot)[boot].sum(1)/np.array(count)[boot].sum(1)
        out[mode]={'mean_loss_improvement':float(np.mean([r['loss_improvement'] for r in cases])),'pool_bootstrap_95pct_interval':np.quantile(estimates,[.025,.975]).tolist(),'cases':cases}
    result={'post_screen_descriptive_only':True,'no_new_fits_or_changed_gates':True,'exact_original_training_inputs_and_parameters_verified':True,'reference_predictions_sha256':sha(OLD/'trial/original/predictions.npz'),'comparisons_to_original_only':out}
    (H/'original-reference.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='cases'} for k,v in out.items()},indent=2))
if __name__=='__main__':main()
