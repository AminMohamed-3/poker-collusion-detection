"""Codex no-fit research inventory on the existing training-only audit.
No predictions, public/outer truth, fits, weight tuning, or new held reads.
"""
import os
os.environ['POLARS_MAX_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
HERE=Path(__file__).resolve().parents[1]
SOURCE=HERE/'experiments/tabpfn3_evidence_20260912'
OUT=HERE/'research/analogue_feasibility_20260912'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(frame):
    counts=frame.group_by('table_id','hand_id').agg(pl.col('pair_id').n_unique().alias('multiplicity'))
    joined=frame.join(counts,on=['table_id','hand_id'],validate='m:1')
    qs=joined.group_by('pair_id').agg((pl.col('multiplicity')>1).any().alias('repeated'))
    a=counts['multiplicity'].to_numpy()
    return {'pair_hand_rows':frame.height,'physical_hands':counts.height,'queries':frame['pair_id'].n_unique(),
      'pools':frame['table_id'].n_unique(),'multiplicity_histogram':{str(k):int(v) for k,v in counts.group_by('multiplicity').len().sort('multiplicity').iter_rows()},
      'physical_hand_p95':float(np.quantile(a,.95)),'physical_hand_p99':float(np.quantile(a,.99)),'maximum':int(a.max()),
      'duplicate_excess_rows':frame.height-counts.height,
      'rows_exposed_to_repeated_hand':int(joined.filter(pl.col('multiplicity')>1).height),
      'fraction_rows_exposed':float((joined['multiplicity']>1).mean()),
      'queries_with_any_repeated_hand':int(qs['repeated'].sum())}
def main():
    assert not OUT.exists(),'write-once inventory'
    f=json.loads((SOURCE/'freeze.json').read_text())
    frame=SOURCE/'evaluation-frame.parquet'
    assert sha(frame)==f['sources'][str(frame.resolve())]
    d=pl.read_parquet(frame)
    assert d['pair_id'].n_unique()==162
    assert d.select('pair_id','hand_id').unique().height==d.height
    assert d.group_by('hand_id').agg(pl.col('table_id').n_unique().alias('n'))['n'].max()==1
    assert d.group_by('pair_id').agg(pl.col('table_id').n_unique().alias('n'))['n'].max()==1
    q=d.group_by('pair_id','table_id','behavior_family').agg(pl.col('relevance').sum().alias('listed_count'),pl.len().alias('observed_hands'))
    assert q['listed_count'].min()>=1 and q['listed_count'].max()<=5
    top=d.filter(pl.col('oof_rank')<=20)
    assert top.height==3240
    folds=[]
    for m in f['folds']:
        fam=m['family']; train=set(m['train_pools']); held=set(m['held_pools']);assert not train&held
        tf=top.filter((pl.col('behavior_family')==fam)&pl.col('table_id').is_in(list(train)))
        hf=top.filter((pl.col('behavior_family')==fam)&pl.col('table_id').is_in(list(held)))
        assert tf.height==m['training_rows'] and hf.height==m['query_rows']
        overlap=set(tf['hand_id'])&set(hf['hand_id']);assert not overlap
        tq=q.filter((pl.col('behavior_family')==fam)&pl.col('table_id').is_in(list(train)))
        scarce=tq.filter(pl.col('listed_count')<5)
        scarce_rows=tf.join(scarce.select('pair_id'),on='pair_id',how='semi')
        mult=describe(tf)
        folds.append({'stem':m['stem'],'same_model_train_held_physical_hand_overlap':len(overlap),
            'multiplicity':mult,'teacher_queries':scarce.height,'teacher_pools':scarce['table_id'].n_unique(),
            'teacher_listed_hands_all':int(scarce['listed_count'].sum()),
            'teacher_candidate_rows':scarce_rows.height,'teacher_listed_hands_in_top20':int(scarce_rows['relevance'].sum())})
    families={}
    for fam in sorted(q['behavior_family'].unique()):
        qq=q.filter(pl.col('behavior_family')==fam); scarce=qq.filter(pl.col('listed_count')<5)
        families[fam]={'all_queries':qq.height,'listed_count_histogram':{str(k):int(v) for k,v in qq.group_by('listed_count').len().sort('listed_count').iter_rows()},
          'scarce_queries':scarce.height,'scarce_pools':scarce['table_id'].n_unique(),
          'candidate_multiplicity':describe(top.filter(pl.col('behavior_family')==fam))}
    res={'author':'Codex','no_fits':True,'no_new_prediction':True,'no_outer_truth_or_metric_read':True,
      'scope':'Existing training-only 162-query audit outside outer0/1/2 union. Same frozen family-specific GroupKFold splits.',
      'input_sha256':{str(frame):sha(frame),str(SOURCE/'freeze.json'):sha(SOURCE/'freeze.json')},'source_sha256':sha(Path(__file__)),
      'full_observed_audit_rows':describe(d),'top20_rows':describe(top),'families':families,'folds':folds,
      'feasibility':{'all_within_family_training_multiplicity_exactly_one':all(x['multiplicity']['maximum']==1 for x in folds),
        'minimum_teacher_queries':min(x['teacher_queries'] for x in folds),'minimum_teacher_pools':min(x['teacher_pools'] for x in folds),
        'zero_coverage_teacher_contexts':[x['stem'] for x in folds if x['teacher_queries']==0]},
      'limits':['Listed_count<5 is not proof of exhaustive evidence annotation.','Cross-family duplicates are not cross-fold leakage when their models and grouping differ.','Multiplicity alone does not establish overweighting: physical hands can legitimately support multiple pair labels.','This is feasibility triage, not evidence of predictive utility.']}
    OUT.mkdir(parents=True)
    (OUT/'inventory.json').write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(res,indent=2))
if __name__=='__main__':main()
