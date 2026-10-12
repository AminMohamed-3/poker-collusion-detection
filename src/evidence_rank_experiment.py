"""Private weak-supervision ranking experiment on explicitly labeled target pairs.

Listed evidence is preferred to unlisted hands within that same target pair.
Zero ranking relevance means unannotated, never a claim of non-coordination.
Unlisted development pairs are not included at all. No production outputs change.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from train import FAMILIES, matrix
from evidence_context_experiment import ap5

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT/'data', ROOT/'outputs_v2'
REPORT = ROOT/'reports'/'evidence-rank-experiment'


def add_family(frame, families):
    return frame.with_columns(*[
        pl.Series('family_'+family, (np.asarray(families)==family).astype(np.float32))
        for family in FAMILIES[1:]])


def main():
    REPORT.mkdir(exist_ok=True)
    labels = pl.read_csv(DATA/'development_labels.csv').filter(pl.col('label')==1)
    evidence = pl.read_csv(DATA/'development_evidence.csv').select('pair_id','hand_id').unique()
    pairpred = pl.read_csv(OUT/'validation_pairs.csv').join(
        labels.select('pair_id'), on='pair_id')
    full_pairpred = pl.read_csv(OUT/'validation_pairs.csv')
    raw = pl.read_parquet(sorted((OUT/'hands_development').glob('batch_*.parquet')))
    held_tables = set(raw.filter(pl.col('pair_id').is_in(full_pairpred['pair_id'].to_list()))['table_id'])
    # Match the classifier experiment's independent tie draw for every held hand.
    all_held_keys = raw.filter(pl.col('table_id').is_in(list(held_tables))).select('pair_id','hand_id')
    ties = all_held_keys.with_columns(pl.Series('tie',np.random.default_rng(931207).random(all_held_keys.height)))
    basefeatures = [c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]
    events = [c for c in basefeatures if c.startswith('v2_')]
    raw = raw.join(labels.select('pair_id','behavior_family'), on='pair_id')
    raw = raw.with_columns(*[
        ((pl.col(c)-pl.col(c).mean().over('pair_id')) /
         pl.col(c).std().over('pair_id').fill_null(0).clip(lower_bound=.001))
        .clip(-20,20).fill_null(0).cast(pl.Float32).alias('context_'+c) for c in events])
    raw = raw.join(evidence.with_columns(pl.lit(1).alias('listed_relevance')),
                   on=['pair_id','hand_id'], how='left').with_columns(pl.col('listed_relevance').fill_null(0))
    train = raw.filter(~pl.col('table_id').is_in(list(held_tables))).sort('pair_id','hand_id')
    held = raw.filter(pl.col('table_id').is_in(list(held_tables)))
    assert not set(train['table_id']) & set(held['table_id'])
    assert set(held['pair_id']) == set(pairpred['pair_id'])
    train = add_family(train, train['behavior_family'].to_numpy())
    held = held.join(pairpred.select('pair_id','predicted_behavior'), on='pair_id')
    groups = train.group_by('pair_id',maintain_order=True).len()['len'].to_numpy()
    assert groups.sum() == train.height
    assert train.group_by('pair_id').agg(pl.col('listed_relevance').sum())['listed_relevance'].min()>0
    truth = {key[0]:set(g['hand_id']) for key,g in evidence.group_by('pair_id')}
    configs = [('rank_raw',basefeatures),('rank_raw_plus_pair_zscore',basefeatures+['context_'+c for c in events])]
    summaries, allrecords = [], []
    for name, gameplay in configs:
        started=time.perf_counter()
        features=gameplay+['family_'+f for f in FAMILIES[1:]]
        model=lgb.LGBMRanker(objective='lambdarank',n_estimators=238,learning_rate=.03,
            num_leaves=15,min_child_samples=20,colsample_bytree=.8,reg_lambda=5,
            n_jobs=2,verbosity=-1,random_state=7329,deterministic=True,force_col_wise=True,
            lambdarank_truncation_level=10,label_gain=[0,1])
        model.fit(matrix(train,features),train['listed_relevance'].to_numpy(),group=groups)
        predictions={}
        for mode,familycol in [('oracle','behavior_family'),('predicted','predicted_behavior')]:
            data=add_family(held, held[familycol].to_numpy())
            predictions[mode]=model.predict(matrix(data,features))
        scored=held.select('pair_id','hand_id','behavior_family').with_columns(
            *[pl.Series(k,v) for k,v in predictions.items()]).join(ties,on=['pair_id','hand_id'])
        records=[]
        for key,frame in scored.group_by('pair_id'):
            pair=key[0]
            rec={'model':name,'pair_id':pair,'behavior_family':frame['behavior_family'][0]}
            for mode in ['oracle','predicted']:
                ranking=frame.sort([mode,'tie'],descending=[True,False])['hand_id'].head(5).to_list()
                rec[mode+'_map5']=ap5(ranking,truth[pair])
            records.append(rec)
        metrics={'model':name,'features':len(features),'training_target_pairs':len(groups),
                 'training_hands':train.height,'validation_target_pairs':len(records),
                 **{m:float(np.mean([r[m] for r in records])) for m in ['oracle_map5','predicted_map5']},
                 'by_true_family':{family:{m:float(np.mean([r[m] for r in records if r['behavior_family']==family]))
                    for m in ['oracle_map5','predicted_map5']} for family in FAMILIES[1:]},
                 'seconds':time.perf_counter()-started}
        print(json.dumps(metrics),flush=True)
        summaries.append(metrics); allrecords.extend(records)
    result={'candidates':summaries,'baseline_predicted_family_map5':.3743367346938776,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'notes':['Fixed 238 trees and two declared feature sets; no parameter search.',
                     'Explicit target pairs only. Unlisted pairs are excluded.',
                     'Weak within-pair annotation preference: unlisted target-pair hands are unknown, not confirmed negatives.',
                     'Incomplete evidence creates noisy training preferences; held-out ranking checks utility without asserting hidden labels.',
                     'Family-conditioned inputs use true family for training; predicted-family validation uses saved v2 pair predictions.',
                     'Same exploratory pool holdout reused; score is not an unbiased estimate of leaderboard performance.',
                     'No production models, caches, or submissions modified.']}
    (REPORT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    pl.DataFrame(allrecords).write_csv(REPORT/'per_pair.csv')
    lines=['# Private within-pair evidence ranking experiment','',
           '| Candidate | Oracle-family MAP@5 | Predicted-family MAP@5 |',
           '| --- | ---: | ---: |', '| Existing v2 classifier | 0.380969 | 0.374337 |']
    lines += [f"| {r['model']} | {r['oracle_map5']:.6f} | {r['predicted_map5']:.6f} |" for r in summaries]
    lines += ['',*result['notes']]
    lines += ['', 'Additional fixed-pool checks are recorded in confirmation.json: true-family MAP@5 '
              'classifier/ranker is 0.324694/0.455417 on fold1 and 0.370266/0.459479 on fold2. '
              'These overlapping development splits are not independent test sets.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
