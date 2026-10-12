"""One frozen training-only MAP evaluation for fixed-epoch TabPFN3."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,json,sys
from pathlib import Path
import numpy as np
import polars as pl
import tabpfn3_finetune_prepare as prep
import tabpfn3_relative_prepare as parent
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step5_evaluate import half
SHA=prep.sha;READ=prep.read;WRITE=prep.write_once
OUT=prep.OUT

def main(run,package):
    assert not any((OUT/n).exists() for n in ['results.json','evaluation-started.json','predictions-frozen.json','new-scores.parquet'])
    pre=prep.check();ctx=READ(OUT/'context-receipt.json');f=READ(package/'study-freeze.json');done=READ(run/'complete.json')
    assert SHA(Path(__file__))==f['files_sha256'][Path(__file__).name]
    assert done['study_freeze_sha256']==SHA(package/'study-freeze.json')
    assert SHA(OUT/'context-receipt.json')==f['files_sha256']['context-receipt.json']
    assert SHA(run/'baseline.json')==done['baseline_sha256'] and READ(run/'baseline.json')['probabilities_exact']
    assert SHA(run/'synthetic/replay.json')==done['synthetic_replay_sha256']
    assert SHA(run/'contexts'/f['replay_stem']/'replay.json')==done['real_replay_sha256']
    for path in [run/'synthetic/replay.json',run/'contexts'/f['replay_stem']/'replay.json']:
        r=READ(path);assert r['saved_fit_rank_exact'] and r['refitted_checkpoint_rank_exact'] and r['saved_fit_probabilities_exact'] and r['refitted_checkpoint_probabilities_exact']
    source=parent.OUT/'new-scores.parquet';assert SHA(source)==f['local_parent_scores_sha256']
    assert READ(parent.OUT/'predictions-frozen.json')['scores_sha256']==SHA(source)
    d=pl.read_parquet(source).select('pair_id','hand_id','table_id','behavior_family','oof','oof_rank','T','lgbm','salt','fold')
    assert d.height==19325*3
    frames=[];expected={m['stem'] for m in ctx['contexts']};assert expected==set(done['context_receipts']) and len(expected)==45
    for salt in range(3):
        for fold in range(5):
            frame=d.filter((pl.col('salt')==salt)&(pl.col('fold')==fold));n=frame.height
            pred=frame['oof'].to_numpy().copy()-1e6;coverage=np.zeros(n,dtype=int)
            for m in [x for x in ctx['contexts'] if x['salt']==salt and x['fold']==fold]:
                folder=run/'contexts'/m['stem'];r=READ(folder/'receipt.json')
                assert SHA(folder/'receipt.json')==done['context_receipts'][m['stem']] and r['status']=='passed'
                assert r['source_sha256']==f['files_sha256']['tabpfn3_finetune_vm.py']
                assert r['epochs_required']==10 and r['gradient_trace']['optimizer_step_calls']==10 and len(r['losses'])==10
                assert r['saved_state_equals_live'] and r['original_checkpoint_unchanged'] and r['inference_config_matches_original']
                assert r['peak_reserved_bytes']<=13.5*1024**3
                assert SHA(folder/'probabilities.npy')==r['probabilities_sha256']
                assert SHA(OUT/m['context_file'])==m['context_sha256']
                with np.load(OUT/m['context_file'],allow_pickle=False) as c:
                    ix=c['held_indices'];assert np.array_equal(c['held_pair_ids'],frame['pair_id'].to_numpy()[ix])
                    assert np.array_equal(c['groups_held'],frame['table_id'].to_numpy()[ix])
                p=np.load(folder/'probabilities.npy',allow_pickle=False);assert p.shape==(len(ix),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                pred[ix]=p+10;coverage[ix]+=1
            top=(frame['oof_rank']<=20).to_numpy();assert np.all(coverage[top]==1) and np.all(coverage[~top]==0)
            frames.append(frame.with_columns(pl.Series('FT',pred)))
    scored=pl.concat(frames);path=OUT/'new-scores.parquet';scored.write_parquet(path)
    WRITE(OUT/'predictions-frozen.json',{'scores_sha256':SHA(path),'complete_sha256':SHA(run/'complete.json'),'study_freeze_sha256':SHA(package/'study-freeze.json'),'before_metric_read':True})
    WRITE(OUT/'evaluation-started.json',{'invocation':1,'predictions_frozen_sha256':SHA(OUT/'predictions-frozen.json')})
    pf=READ(parent.OUT/'preflight.json');assert not set(pf['audit_pools'])&set(pf['excluded_old_outer_pools'])
    evidence=parent.training_relevance(pf['audit_pools']);truth={k[0]:set(g['hand_id']) for k,g in evidence.group_by('pair_id')}
    fams=dict(scored.select('pair_id','behavior_family').unique().iter_rows());ptab=dict(scored.select('pair_id','table_id').unique().iter_rows());pairs=sorted(fams);pools=sorted(set(ptab.values()))
    assert len(pairs)==162 and len(pools)==104 and set(pairs)==set(truth)
    maps={k:[] for k in ['FT','T','lgbm','oof']};summaries={}
    for salt in range(3):
        fr=scored.filter(pl.col('salt')==salt);summaries[str(salt)]={}
        for name in maps:
            summary,pm=summarize(fr.select('pair_id','hand_id','table_id',pl.col(name).alias('score')),truth,fams)
            maps[name].append(pm);summaries[str(salt)][name]=summary
    avg={k:{p:float(np.mean([m[p] for m in maps[k]])) for p in pairs} for k in maps}
    def compare(base,candidate):
        r=boot(avg[base],avg[candidate],ptab,pools,n=2000,seed=652)
        r['by_family']={f:float(np.mean([avg[candidate][p]-avg[base][p] for p in pairs if fams[p]==f])) for f in parent.FAMILIES}
        r['halves']={str(h):float(np.mean([avg[candidate][p]-avg[base][p] for p in pairs if half(ptab[p])==h])) for h in [0,1]}
        r['salt_means']=[float(np.mean([maps[candidate][s][p]-maps[base][s][p] for p in pairs])) for s in range(3)]
        r['gate_pass']=r['mean_delta']>=.01 and r['ci95'][0]>0 and min(r['by_family'].values())>=-.01 and min(r['halves'].values())>=0 and min(r['salt_means'])>=0
        return r
    primary=compare('T','FT');protect=compare('lgbm','FT');negative=compare('T','oof')
    # Exact old-arm metric parity is an integrity control, not a new candidate selection.
    old=READ(parent.OUT/'per-pair-maps.json')['maps']
    parity=all(np.isclose(avg[k][p],old[k][p],rtol=0,atol=1e-15) for k in ['T','lgbm','oof'] for p in pairs)
    result={'primary_FT_minus_T':primary,'FT_minus_LGBM':protect,'negative_stage1_minus_T':negative,'baseline_metric_parity':parity,'eligible':bool(primary['gate_pass'] and protect['gate_pass'] and not negative['gate_pass'] and parity),'mean_map':{k:float(np.mean(list(v.values()))) for k,v in avg.items()},'summaries':summaries,'evaluation_invocation':1,'selection_surface_reused':True,'old_outer_metrics_read':False,'unique_pairs':162,'unique_pools':104,'appearances':486,'protected_unchanged':prep.protected()==READ(parent.OUT/'preflight.json')['protected']}
    assert result['protected_unchanged'];WRITE(OUT/'results.json',result);WRITE(OUT/'per-pair-maps.json',{'maps':avg,'pool_of_pair':ptab,'family_of_pair':fams});print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--package',type=Path,required=True);v=a.parse_args();main(v.run.resolve(),v.package.resolve())
