"""Fit one activation probability from training/generated contexts, never held data."""
import os
for k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib
import numpy as np
import polars as pl
import joblib
H=Path(__file__).resolve().parent;P=H.parent/'softplay_conditional_20260910';OLD=H.parent/'softplay_fast_20260910'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (H/'calibration.json').exists()
    f=pl.read_parquet(P/'learned/soft-actions.parquet');model=joblib.load(P/'controller.joblib');x=f.select(model['inputs']).to_numpy()
    soft=model['soft'].predict(x,num_threads=2);normal=model['normal'].predict(x,num_threads=2);free=f['price'].to_numpy()==0;soft[np.arange(len(f)),np.where(free,0,1)]=0;soft/=soft.sum(1,keepdims=True)
    # Recover ordinary resolved action class at the observed pre-action state.
    # The controller changes only fold/raise to check/call. Changed traces save
    # that ordinary proposal; unchanged rows already carry its resolved class.
    trace={}
    for p in sorted((P/'learned/pilot').glob('pool-*')):
        for hand in json.loads((p/'interventions.json').read_text()):
            for t in hand['interventions']:
                assert t['base_intent'][0] in ['fold','raise']
                trace[(hand['hand_id'],t['action_no'])]=0 if t['base_intent'][0]=='fold' else 3
    base=np.array([trace.get((r['hand_id'],r['action_no']),r['action_class']) for r in f.iter_rows(named=True)])
    pre=f['street_index'].to_numpy()==0;price=f['price'].to_numpy()
    allowed=(~free)&(soft[:,2]>normal[:,2])&((base==3)|((base==0)&(price<=.5)))
    table=f.select('pair_id','street_index').with_columns(pl.Series('ordinary_call',(base==2).astype(float)),pl.Series('available_call_gain',allowed*soft[:,2]))
    bypair=table.filter(pl.col('street_index')==0).group_by('pair_id').agg(pl.col('ordinary_call').mean(),pl.col('available_call_gain').mean())
    target=json.loads((OLD/'action-diagnosis.json').read_text())['results']['soft_play']['preflop']['equal_pair_mean_fold_check_call_raise'][2]
    ordinary=float(bypair['ordinary_call'].mean());gain=float(bypair['available_call_gain'].mean());raw=(target-ordinary)/gain;gamma=float(np.clip(raw,0,1))
    # Training-pair cross-check: infer rate from half the synthetic contexts and
    # estimate expected rate in the other half, without new game generation.
    rows=bypair.to_dicts();halves=np.array([int(hashlib.sha256(r['pair_id'].encode()).hexdigest(),16)%2 for r in rows]);cross=[]
    for h in [0,1]:
        a=bypair.filter(pl.Series(halves!=h));b=bypair.filter(pl.Series(halves==h));g=float(np.clip((target-a['ordinary_call'].mean())/a['available_call_gain'].mean(),0,1));cross.append({'calibration_pairs':a.height,'scored_pairs':b.height,'gamma':g,'expected_call_fraction':float(b['ordinary_call'].mean()+g*b['available_call_gain'].mean())})
    out={'source_sha256':sha(__file__),'input_controller_sha256':sha(P/'controller.joblib'),'input_actions_sha256':sha(P/'learned/soft-actions.parquet'),'original_target_preflop_calls':target,'ordinary_call_probability_on_generated_contexts':ordinary,'available_call_gain_per_unit_activation':gain,'unclipped_gamma':raw,'calibrated_gamma':gamma,'gamma_was_clipped':gamma!=raw,'expected_rate_on_fixed_contexts':ordinary+gamma*gain,'cross_pair_checks':cross,'original_training_only':True,'original_training_pool_count':297,'limitation':'One-moment approximation on previous generated states; changing actions changes later states. Cross-pair check is not held-original or generator validation.'}
    (H/'calibration.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
