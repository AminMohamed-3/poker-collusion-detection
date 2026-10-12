"""Describe frozen held-pool retrieval errors; no fitting or candidate changes."""
import os
os.environ['POLARS_MAX_THREADS']='2'
import json
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha
from evidence_context_experiment import ap5


def main():
    out=ROOT/'reports/conditional-retrieval-diagnostic'; out.mkdir(exist_ok=False)
    source=ROOT/'reports/conditional-behavior-experiment'
    annotations=pl.read_csv(ROOT/'data/development_evidence.csv')
    truth={key[0]:set(group['hand_id']) for key,group in annotations.group_by('pair_id')}
    labels=pl.read_csv(ROOT/'data/development_labels.csv').select('pair_id','behavior_family')
    records=[]; files=[]
    for fold in ['fold0','fold1','fold2']:
        path=source/fold/'conditional_behavior-scores.parquet'
        scored=pl.read_parquet(path)
        assert scored.select('pair_id','hand_id').equals(scored.select('pair_id','hand_id').sort('pair_id','hand_id'))
        scored=scored.with_columns(pl.Series('tie',np.random.default_rng(931207).random(scored.height))).join(labels,on='pair_id',validate='m:1',maintain_order='left')
        expected=pl.read_csv(source/fold/'evidence-pairs.csv').filter(pl.col('candidate')=='conditional_behavior')
        expected=dict(zip(expected['pair_id'],expected['map5']))
        for key,frame in scored.group_by('pair_id'):
            pair=key[0]; ranked=frame.sort(['score','tie'],descending=[True,False])['hand_id'].to_list()
            score=ap5(ranked[:5],truth[pair]); assert abs(score-expected[pair])<1e-12
            positions=[i+1 for i,hand in enumerate(ranked) if hand in truth[pair]]
            assert len(positions)==len(truth[pair])
            row={'fold':fold,'pair_id':pair,'family':frame['behavior_family'][0], 'shared_hands':len(ranked),
                'annotations':len(positions),'map5':score,'first_annotated_rank':min(positions),
                'annotation_median_rank':float(np.median(positions))}
            for cutoff in [1,5,10,20,50]:
                found=sum(rank<=cutoff for rank in positions)
                row[f'recall_at_{cutoff}']=found/len(positions)
                if cutoff>=5:
                    # Optimistic bound if a perfect reranker were restricted to this shortlist.
                    row[f'oracle_map5_at_{cutoff}']=min(found,5)/min(len(positions),5)
            records.append(row)
        files.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    frame=pl.DataFrame(records)
    frame.write_csv(out/'held-pair-diagnostics.csv')
    measures=['map5','shared_hands','first_annotated_rank','annotation_median_rank']+[f'recall_at_{k}' for k in [1,5,10,20,50]]+[f'oracle_map5_at_{k}' for k in [5,10,20,50]]
    unique=frame.group_by('pair_id','family').agg(*[pl.col(c).mean() for c in measures])
    result={'scope':'Diagnostics of frozen models on reused held pools. Overlapping pairs averaged across folds; no model fit, test-population scoring, or submission change.',
        'source_sha256':sha(__file__),'sources':files,'rows_across_folds':frame.height,'unique_pairs':unique.height,
        'pooled_unique_pair_means':unique.select(pl.col(measures).mean()).to_dicts()[0],
        'by_family':unique.group_by('family').agg(pl.len().alias('pairs'),*[pl.col(c).mean() for c in measures]).sort('family').to_dicts(),
        'zero_map_pair_fraction':float((unique['map5']==0).mean()),'all_previous_map5_values_exact':True,
        'caveat':'Oracle shortlist MAP assumes annotation knowledge and is an upper bound, never a candidate or demonstrated attainable gain.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
