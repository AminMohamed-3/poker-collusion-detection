"""Confirm fixed context ranker against classifier on two additional pool splits."""
from __future__ import annotations
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit
from train import FAMILIES, LABELS, classifier, matrix
from evidence_context_experiment import ap5
from evidence_rank_experiment import add_family

ROOT=Path(__file__).resolve().parents[1]
DATA,OUT=ROOT/'data',ROOT/'outputs_v2'
REPORT=ROOT/'reports'/'evidence-rank-experiment'


def main():
    raw=pl.read_parquet(sorted((OUT/'hands_development').glob('batch_*.parquet')))
    labels=pl.read_csv(DATA/'development_labels.csv')
    pairdata=pl.read_parquet(OUT/'pairs_development.parquet').join(labels,on='pair_id')
    groups=pairdata['table_id'].to_numpy()
    splits=list(GroupShuffleSplit(n_splits=3,test_size=.25,random_state=2901).split(groups,groups=groups))
    evidence=pl.read_csv(DATA/'development_evidence.csv').select('pair_id','hand_id',pl.col('behavior_family').alias('evidence_family'))
    truth={key[0]:set(g['hand_id']) for key,g in evidence.group_by('pair_id')}
    base=[c for c in raw.columns if c not in ['pair_id','hand_id','table_id']]
    events=[c for c in base if c.startswith('v2_')]
    raw=raw.with_columns(*[
        ((pl.col(c)-pl.col(c).mean().over('pair_id')) /
         pl.col(c).std().over('pair_id').fill_null(0).clip(lower_bound=.001))
        .clip(-20,20).fill_null(0).cast(pl.Float32).alias('context_'+c) for c in events])
    rankfeatures=base+['context_'+c for c in events]+['family_'+f for f in FAMILIES[1:]]
    joined=raw.join(labels.select('pair_id','label','behavior_family'),on='pair_id').join(evidence,on=['pair_id','hand_id'],how='left')
    results, allrecords=[],[]
    for fold in [1,2]:
        started=time.perf_counter()
        heldtables=set(groups[splits[fold][1]])
        training=joined.filter(~pl.col('table_id').is_in(list(heldtables)))
        held=joined.filter(pl.col('table_id').is_in(list(heldtables)) & (pl.col('label')==1))
        assert not set(training['table_id']) & set(held['table_id'])
        supervised=training.filter((pl.col('label')==0)|pl.col('evidence_family').is_not_null())
        y=np.array([LABELS[x] if x is not None else 0 for x in supervised['evidence_family']])
        control=classifier(238,hand=True).set_params(n_jobs=2)
        control.fit(matrix(supervised,base),y)
        cp=control.predict_proba(matrix(held,base))
        wanted=np.array([LABELS[f] for f in held['behavior_family']])
        cp=cp[np.arange(len(cp)),wanted]
        targets=training.filter(pl.col('label')==1).sort('pair_id','hand_id')
        targets=add_family(targets,targets['behavior_family'].to_numpy())
        relevance=targets['evidence_family'].is_not_null().cast(pl.Int32).to_numpy()
        group=targets.group_by('pair_id',maintain_order=True).len()['len'].to_numpy()
        assert group.sum()==targets.height
        ranker=lgb.LGBMRanker(objective='lambdarank',n_estimators=238,learning_rate=.03,
            num_leaves=15,min_child_samples=20,colsample_bytree=.8,reg_lambda=5,
            n_jobs=2,verbosity=-1,random_state=7329,deterministic=True,force_col_wise=True,
            lambdarank_truncation_level=10,label_gain=[0,1])
        ranker.fit(matrix(targets,rankfeatures),relevance,group=group)
        rp=ranker.predict(matrix(add_family(held,held['behavior_family'].to_numpy()),rankfeatures))
        scored=held.select('pair_id','hand_id','behavior_family').with_columns(
            pl.Series('classifier',cp),pl.Series('ranker',rp),
            pl.Series('tie',np.random.default_rng(931207).random(held.height)))
        records=[]
        for key,frame in scored.group_by('pair_id'):
            pair=key[0]
            record={'fold':fold,'pair_id':pair,'behavior_family':frame['behavior_family'][0]}
            for name in ['classifier','ranker']:
                ranking=frame.sort([name,'tie'],descending=[True,False])['hand_id'].head(5).to_list()
                record[name]=ap5(ranking,truth[pair])
            records.append(record)
        result={'fold':fold,'target_pairs':len(records),
                **{name:float(np.mean([r[name] for r in records])) for name in ['classifier','ranker']},
                'by_true_family':{family:{name:float(np.mean([r[name] for r in records if r['behavior_family']==family]))
                    for name in ['classifier','ranker']} for family in FAMILIES[1:]},
                'seconds':time.perf_counter()-started}
        print(json.dumps(result),flush=True)
        results.append(result);allrecords.extend(records)
    payload={'folds':results,'metric':'Evidence MAP@5 conditional on true behavior family',
             'notes':['Two additional fixed pool splits; same 238 trees, no early stopping or tuning.',
                      'These are repeated development holdouts and overlap; not independent test sets.',
                      'Control has the existing v2 classifier architecture; candidate has the fixed context ranker architecture.',
                      'Ranking relevance zero denotes missing annotation, not known-negative behavior.',
                      'No production models, caches or submissions are written.']}
    (REPORT/'confirmation.json').write_text(json.dumps(payload,indent=2)+'\n')
    pl.DataFrame(allrecords).write_csv(REPORT/'confirmation_pairs.csv')


if __name__=='__main__':
    main()
