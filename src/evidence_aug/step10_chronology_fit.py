"""Step 10: frozen confirmation of within-pair chronology as an evidence-ranker input (folds 0, 1, 2).

Origin (training pools only, step 9): among each training query's top-20 out-of-fold candidates the listed
hands sit early in the pair's shared history (listed rate by chronological quartile .36/.28/.19/.11), and a
pool-grouped re-ranker with score + chronology gained +.051 MAP@5 out-of-fold versus -.003 for a score-only
control. No held truth was involved. This script freezes the design, fits per fold with the unchanged recipe
(baseline must reproduce the fold's saved scores exactly), scores held pairs under fixed predicted families
and freezes predictions. No held truth is read here.

Variants: baseline (382 features) | chrono3 (382 + chrono_rank, chrono_frac, n_hands; PRIMARY) | chrono1 (382 + chrono_frac).
Outputs: Claude Version/experiments/chronology_20260910/{freeze.json, scores/, predictions-frozen.json}
"""
import os
import sys
import time
import json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, SRC, ORIGINAL, EXP, sha, write_json, read_json, check_protected
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, PARAMS, FAMILIES, cache_identity
from evidence_aug.fold import load_fold, fold_roles, fold_paths
from evidence_aug.step1_reproduce_baseline import fit_ranker
import numpy as np
import polars as pl
import joblib

OUT = CV / 'experiments' / 'chronology_20260910'
CHRONO = {'chrono3': ['chrono_rank', 'chrono_frac', 'n_hands'], 'chrono1': ['chrono_frac']}


def add_chronology(data):
    hands = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').select('hand_id', 'started_at').join(data.select('hand_id').unique().lazy(), on='hand_id').collect()
    out = data.join(hands, on='hand_id', validate='m:1').with_columns(
        pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'), pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'))
    out = out.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac')).drop('started_at').sort('pair_id', 'hand_id')
    assert out.height == data.height and out.select(pl.col(['chrono_rank', 'chrono_frac', 'n_hands']).is_finite().all()).row(0) == (True, True, True)
    return out


def main():
    t0 = time.perf_counter()
    assert not (OUT / 'freeze.json').exists(), 'freeze exists; preserve it'
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    audit = read_json(EXP / 'audit-annotator-selection.json')
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'recipe': PARAMS,
        'code_sha256': {p.name: sha(p) for p in sorted((SRC / 'evidence_aug').glob('*.py'))}, 'original_cache_identity': cache_identity(),
        'hands_parquet_sha256': sha(ORIGINAL / 'data' / 'hands.parquet'),
        'origin': {'source': 'step9 training-only audit (fold-0 training pools, 5-way pool-hashed cross-fit)', 'audit_sha256': sha(EXP / 'audit-annotator-selection.json'),
                   'listed_rate_by_chrono_quartile': audit['listed_rate_by_chrono_quartile'], 'rerank_top20_map5': audit['rerank_top20_map5']},
        'features': {'baseline': 382, 'chrono3': '382 + chrono_rank (0-based order of started_at within the pair frame), chrono_frac (chrono_rank/(n_hands-1)), n_hands', 'chrono1': '382 + chrono_frac'},
        'primary_variant': 'chrono3', 'secondary_variant': 'chrono1',
        'routing': 'none; every held pair scored by the variant under its fixed predicted family one-hots, exactly like the saved baseline',
        'confirmation_folds': [0, 1, 2], 'note': 'hypothesis came from training pools only, so fold 0 counts; all three folds are reused partitions from earlier studies',
        'gate': {'per_fold_overall_gain_min': .01, 'per_fold_true_family_loss_max': .01, 'pooled_pool_bootstrap_95_excludes_zero': True, 'pooled_pool_halves': 'both nonnegative', 'applies_to': 'primary variant chrono3 on folds 0, 1 and 2 separately'},
        'metric': 'MAP@5 per held target pair, tie seed 931207', 'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool'},
        'held_evidence_truth_read_before_freeze': False, 'folds': {},
    }
    write_json(OUT / 'freeze.json', freeze)
    timings = {}
    for fold in [0, 1, 2]:
        data, features, pools, identity = load_fold(fold)
        data = add_chronology(data)
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools))
        evidence = training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence)
        rf = fold_roles(fold)
        hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        saved = pl.read_parquet(fold_paths(fold)['saved_scores'])
        assert hf.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
        info = {'identity': identity, 'held_target_pairs': held['pair_id'].n_unique(), 'held_rows': held.height, 'training_queries': train['pair_id'].n_unique(), 'training_listed_hands': evidence.height, 'variants': {}}
        for name, extra in [('baseline', []), ('chrono3', CHRONO['chrono3']), ('chrono1', CHRONO['chrono1'])]:
            t1 = time.perf_counter()
            model = fit_ranker(train, features + extra)
            timings[f'fold{fold}/{name}'] = time.perf_counter() - t1
            s = score(model, family_columns(hf, 'predicted_behavior'), features + extra)
            assert np.isfinite(s).all()
            if name == 'baseline':
                assert np.array_equal(s, saved['score'].to_numpy()), f'fold{fold}: baseline refit differs from saved'
            hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
            joblib.dump({'model': model, 'features': features + extra}, OUT / 'scores' / f'fold{fold}-{name}.joblib')
            gain = model.booster_.feature_importance('gain'); order = np.argsort(gain)[::-1][:10]
            info['variants'][name] = {'features': len(features + extra), 'fit_seconds': timings[f'fold{fold}/{name}'],
                                      'top_gain_features': [((features + extra)[i], float(gain[i] / gain.sum())) for i in order]}
            print(json.dumps({'fold': fold, 'variant': name, 'seconds': round(timings[f'fold{fold}/{name}'], 1), 'top3': info['variants'][name]['top_gain_features'][:3]}), flush=True)
        rf.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        freeze['folds'][str(fold)] = info
    write_json(OUT / 'freeze.json', freeze)
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(OUT / 'freeze.json'),
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').glob('*.parquet')) + sorted((OUT / 'scores').glob('*.csv'))}, 'fit_seconds': timings, 'held_evidence_truth_read': False, 'seconds': time.perf_counter() - t0}
    write_json(OUT / 'predictions-frozen.json', frozen)
    print(json.dumps({'predictions_frozen': True, 'seconds': round(frozen['seconds'], 1)}), flush=True)


if __name__ == '__main__':
    main()
