"""Two fixed evidence models: raw gameplay versus pair-relative event features.

Private development experiment. It never writes production caches or submissions.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from train import FAMILIES, LABELS, classifier, matrix

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT/'data', ROOT/'outputs_v2'
REPORT = ROOT/'reports'/'evidence-context-experiment'


def ap5(ranking, truth):
    hits, score = 0, 0.0
    for rank, hand in enumerate(ranking, 1):
        if hand in truth:
            hits += 1
            score += hits/rank
    return score/min(5, len(truth)) if truth else 0.0


def main():
    REPORT.mkdir(exist_ok=True)
    started = time.perf_counter()
    protected = [OUT/'pair_model_robust.joblib', OUT/'hand_model.joblib',
                 OUT/'submission_advanced_robust.csv']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    raw = pl.read_parquet(sorted((OUT/'hands_development').glob('batch_*.parquet')))
    labels = pl.read_csv(DATA/'development_labels.csv')
    evidence = pl.read_csv(DATA/'development_evidence.csv').select(
        'pair_id', 'hand_id', pl.col('behavior_family').alias('evidence_family'))
    pairpred = pl.read_csv(OUT/'validation_pairs.csv')
    held_pairs = set(pairpred['pair_id'])
    held_tables = set(raw.filter(pl.col('pair_id').is_in(list(held_pairs)))['table_id'])
    assert set(raw.filter(pl.col('table_id').is_in(list(held_tables)))['pair_id']) == held_pairs
    features = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in features if c.startswith('v2_')]
    # These statistics use gameplay from each pair's own observable history only.
    # They are label-free and computed independently for every held-out pair.
    raw = raw.with_columns(*[
        ((pl.col(c)-pl.col(c).mean().over('pair_id')) /
         pl.col(c).std().over('pair_id').fill_null(0).clip(lower_bound=.001))
        .clip(-20,20).fill_null(0).cast(pl.Float32).alias('context_'+c)
        for c in events])
    context_features = ['context_'+c for c in events]
    assert raw.select(pl.col(context_features).is_finite().all()).row(0) == (True,)*len(events)
    data = raw.join(labels.select('pair_id','label'), on='pair_id').join(
        evidence, on=['pair_id','hand_id'], how='left')
    data = data.filter((pl.col('label') == 0) | pl.col('evidence_family').is_not_null())
    training = data.filter(~pl.col('table_id').is_in(list(held_tables)))
    validation = data.filter(pl.col('table_id').is_in(list(held_tables)))
    assert not set(training['pair_id']) & held_pairs
    assert not set(training['table_id']) & set(validation['table_id'])
    ytrain = np.array([LABELS[x] if x is not None else 0 for x in training['evidence_family']])
    yval = np.array([LABELS[x] if x is not None else 0 for x in validation['evidence_family']])
    held = raw.filter(pl.col('table_id').is_in(list(held_tables)))
    targets = pairpred.join(labels.select('pair_id','label'), on='pair_id').filter(pl.col('label') == 1)
    truths = {key[0]: set(group['hand_id']) for key, group in evidence.group_by('pair_id')}
    summaries, per_pair = [], []
    for name, cols in [('raw',features), ('raw_plus_pair_zscore',features+context_features)]:
        start = time.perf_counter()
        model = classifier(238, hand=True).set_params(n_jobs=2)
        model.fit(matrix(training,cols), ytrain)
        vp = model.predict_proba(matrix(validation,cols))
        hp = model.predict_proba(matrix(held,cols))
        ranked = held.select('pair_id','hand_id').with_columns(
            pl.Series('tie',np.random.default_rng(931207).random(held.height)),
            *[pl.Series(FAMILIES[k],hp[:,k]) for k in [1,2,3]])
        lookup = {key[0]: group for key,group in ranked.group_by('pair_id')}
        records = []
        for row in targets.iter_rows(named=True):
            pair = row['pair_id']
            scores = {}
            for mode, family in [('oracle',row['behavior_family']), ('predicted',row['predicted_behavior'])]:
                choices = lookup[pair].sort([family,'tie'],descending=[True,False])['hand_id'].head(5).to_list()
                scores[mode+'_map5'] = ap5(choices,truths[pair])
            records.append({'model':name,'pair_id':pair,'behavior_family':row['behavior_family'],**scores})
        metrics = {'model':name,'features':len(cols),'training_hands':training.height,
                   'confirmed_validation_hands':validation.height,'retrieval_target_pairs':len(records),
                   'hand_vs_confirmed_negative_ap':float(average_precision_score(yval>0,1-vp[:,0])),
                   **{m:float(np.mean([r[m] for r in records])) for m in ['oracle_map5','predicted_map5']},
                   'by_true_family': {family:{m:float(np.mean([r[m] for r in records if r['behavior_family']==family]))
                     for m in ['oracle_map5','predicted_map5']} for family in FAMILIES[1:]},
                   'seconds':time.perf_counter()-start}
        print(json.dumps(metrics),flush=True)
        summaries.append(metrics); per_pair.extend(records)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p,value in before.items())
    result = {'candidates':summaries,'protected_files_unchanged':True,'protected_sha256':before,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'heldout_pair_prediction_sha256':hashlib.sha256((OUT/'validation_pairs.csv').read_bytes()).hexdigest(),
              'seconds':time.perf_counter()-started,
              'notes':['Fixed 238 trees; no early stopping or hyperparameter search.',
                       'Same saved pool holdout for both candidates; table and pair disjointness asserted.',
                       'Only listed evidence is positive; confirmed-none pair hands are negative; all other hands excluded from supervised fit.',
                       'Context uses each pair\'s unlabeled gameplay; no train/validation pool sharing.',
                       'Predicted-family retrieval uses saved unaugmented v2 fold0 predictions, not the uploaded robust model.',
                       'This reused development holdout is exploratory and does not establish private leaderboard performance.']}
    (REPORT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    pl.DataFrame(per_pair).write_csv(REPORT/'per_pair.csv')
    lines = ['# Private evidence context experiment','',
             '| Candidate | Oracle-family MAP@5 | Predicted-family MAP@5 | Hand AP vs confirmed negatives |',
             '| --- | ---: | ---: | ---: |']
    lines += [f"| {r['model']} | {r['oracle_map5']:.6f} | {r['predicted_map5']:.6f} | {r['hand_vs_confirmed_negative_ap']:.6f} |" for r in summaries]
    lines += ['',*result['notes'],'','Production models and uploaded submission remained unchanged.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
