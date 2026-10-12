"""Step 27: does seed-bagging the chrono1 LambdaRank ranker add to the frozen chronology gain? (folds 0, 1, 2)

Design frozen in code before any held truth is read: chrono1 recipe (382 features + chrono_frac, PARAMS unchanged)
fitted with five random_state values; seed 7329 must reproduce the frozen fold chrono1 scores bit-for-bit
(control); bag5 = mean of the five raw scores. Gate: pooled pool-bootstrap 95% interval of (bag5 - chrono1)
above zero and no fold below -.002. Truth is read once, after all scores are written.
Outputs: Claude Version/experiments/chronology_bag_20260911/{freeze.json, scores/, results.json}
"""
import os
import sys
import time
import json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, PARAMS, FAMILIES
from evidence_aug.fold import load_fold, fold_roles
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap, half
import numpy as np
import polars as pl

CHRONO_DIR = CV / 'experiments' / 'chronology_20260910'
OUT = CV / 'experiments' / 'chronology_bag_20260911'
SEEDS = [7329, 17329, 27329, 37329, 47329]
EXTRA = ['chrono_frac']


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'recipe': PARAMS, 'seeds': SEEDS, 'variant': 'chrono1 (382 + chrono_frac)',
              'aggregation': 'mean of raw LambdaRank scores over the five seeds', 'control': 'seed 7329 must equal frozen chronology_20260910 chrono1 scores exactly',
              'gate': {'pooled_ci_excludes_zero': True, 'per_fold_min_delta': -.002}, 'folds': [0, 1, 2], 'held_truth_read_before_scores_written': False}
    write_json(OUT / 'freeze.json', freeze)
    timings = {}
    for fold in [0, 1, 2]:
        data, features, pools, identity = load_fold(fold)
        data = add_chronology(data)
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = attach_relevance(data.filter(~pl.col('table_id').is_in(pools)), training_relevance(set(data.filter(~pl.col('table_id').is_in(pools))['table_id'])))
        hf = held.join(fold_roles(fold), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        frozen = pl.read_parquet(CHRONO_DIR / 'scores' / f'fold{fold}-chrono1.parquet')
        assert hf.select('pair_id', 'hand_id').equals(frozen.select('pair_id', 'hand_id'))
        fc = family_columns(hf, 'predicted_behavior')
        scores = []
        for seed in SEEDS:
            t1 = time.perf_counter()
            model = fit_ranker(train, features + EXTRA, dict(PARAMS, random_state=seed))
            s = score(model, fc, features + EXTRA)
            timings[f'fold{fold}/seed{seed}'] = round(time.perf_counter() - t1, 1)
            if seed == 7329:
                assert np.array_equal(s, frozen['score'].to_numpy()), f'fold{fold}: seed 7329 does not reproduce the frozen chrono1 scores'
            scores.append(s)
            print(json.dumps({'fold': fold, 'seed': seed, 'seconds': timings[f'fold{fold}/seed{seed}']}), flush=True)
        bag = np.mean(np.stack(scores), axis=0)
        hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', bag)).write_parquet(OUT / 'scores' / f'fold{fold}-bag5.parquet')
        for seed, s in zip(SEEDS, scores):
            hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(OUT / 'scores' / f'fold{fold}-seed{seed}.parquet')
    write_json(OUT / 'predictions-frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(), 'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').glob('*.parquet'))}, 'fit_seconds': timings})
    # ---- single evaluation (held truth read from here on) ----
    labels = dict(pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    results = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'folds': {}}
    pooled = {}; pooled_table = {}; pooled_single = {}
    for fold in [0, 1, 2]:
        base = pl.read_parquet(CHRONO_DIR / 'scores' / f'fold{fold}-baseline.parquet')
        c1 = pl.read_parquet(CHRONO_DIR / 'scores' / f'fold{fold}-chrono1.parquet')
        bag = pl.read_parquet(OUT / 'scores' / f'fold{fold}-bag5.parquet')
        table = dict(base.select('pair_id', 'table_id').unique().iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        m_base = per_pair_map(base, truths); m_c1 = per_pair_map(c1, truths); m_bag = per_pair_map(bag, truths)
        singles = {seed: per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-seed{seed}.parquet'), truths) for seed in SEEDS}
        d = {p: m_bag[p] - m_c1[p] for p in m_c1}
        rep = {'held_target_pairs': len(table), 'map5': {'baseline': float(np.mean(list(m_base.values()))), 'chrono1': float(np.mean(list(m_c1.values()))), 'bag5': float(np.mean(list(m_bag.values()))),
                                                         'singles': {str(s): float(np.mean(list(m.values()))) for s, m in singles.items()}},
               'delta_bag5_vs_chrono1': float(np.mean(list(d.values()))), 'delta_by_true_family': {f: float(np.mean([d[p] for p in d if labels[p] == f])) for f in FAMILIES},
               'delta_bag5_vs_baseline': float(np.mean([m_bag[p] - m_base[p] for p in m_bag])), 'pool_bootstrap_95': pool_bootstrap(d, table),
               'pairs_improved_worsened': [int(sum(v > 1e-12 for v in d.values())), int(sum(v < -1e-12 for v in d.values()))]}
        results['folds'][str(fold)] = rep
        pooled.update({f'{fold}:{p}': v for p, v in d.items()}); pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in d})
        for seed, m in singles.items():
            pooled_single.setdefault(str(seed), []).extend([m[p] - m_c1[p] for p in m])
        print(json.dumps({'fold': fold, **{k: v for k, v in rep.items() if k != 'delta_by_true_family'}}), flush=True)
    ci = pool_bootstrap(pooled, pooled_table)
    halves = {str(hh): float(np.mean([v for k, v in pooled.items() if half(pooled_table[k].split(':')[1]) == hh])) for hh in [0, 1]}
    gate = {'pooled_delta': float(np.mean(list(pooled.values()))), 'pooled_ci_95': ci, 'pooled_halves': halves,
            'single_seed_pooled_delta_vs_chrono1': {s: float(np.mean(v)) for s, v in pooled_single.items()},
            'per_fold_min_delta_ok': all(results['folds'][str(f)]['delta_bag5_vs_chrono1'] >= -.002 for f in [0, 1, 2]), 'pooled_ci_excludes_zero': ci[0] > 0}
    gate['passed'] = gate['per_fold_min_delta_ok'] and gate['pooled_ci_excludes_zero']
    results['gate'] = gate
    results['seconds'] = round(time.perf_counter() - t0, 1)
    write_json(OUT / 'results.json', results)
    print(json.dumps(gate, indent=1), flush=True)


if __name__ == '__main__':
    main()
