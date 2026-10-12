"""Step 6: prespecified real-data confirmation of family query weighting on fold1/fold2 (fold0 reference only).

Hypothesis from the fold-0 screen controls: duplicating real soft-play queries raised routed soft-play
MAP@5 in all three draws. Fold 0 has been observed and cannot count. This script freezes the design,
refits per fold (baseline parity + one duplicated-family variant per family), scores held pairs and
freezes predictions. No held truth is read here.

Variants per fold: baseline | soft_x2 | transfer_x2 | isolation_x2 (every training query of that family duplicated once, '#dup' ids).
Candidates: A 'soft_x2_routed' = soft_x2 for predicted-soft pairs, baseline elsewhere;
            B 'family_specialized' = each pair scored by the ranker whose duplicated family equals its predicted family.
Outputs: Claude Version/experiments/family_weighting_20260910/{freeze.json, scores/, predictions-frozen.json}
"""
import os
import sys
import time
import json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, SRC, sha, write_json, read_json, check_protected
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, PARAMS, FAMILIES, cache_identity
from evidence_aug.fold import load_fold, fold_roles, fold_paths
from evidence_aug.step1_reproduce_baseline import fit_ranker
import numpy as np
import polars as pl
import joblib

OUT = CV / 'experiments' / 'family_weighting_20260910'
SHORT = {'directed_transfer': 'transfer', 'soft_play': 'soft', 'coordinated_isolation': 'isolation'}


def main():
    t0 = time.perf_counter()
    assert not (OUT / 'freeze.json').exists(), 'freeze exists; preserve it'
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'recipe': PARAMS, 'features': 382,
        'code_sha256': {p.name: sha(p) for p in sorted((SRC / 'evidence_aug').glob('*.py'))}, 'original_cache_identity': cache_identity(),
        'origin': 'Fold-0 evidence-augmentation screen controls (Claude Version experiments/evidence_augmentation_20260910): three draws of 78 duplicated real soft-play queries gave routed soft-play MAP@5 gains of +.009/+.019/+.038. Fold 0 is therefore observed and is reported for reference only.',
        'variants': 'baseline (original-only, must reproduce each fold saved conditional-evidence scores exactly); <family>_x2 = every training query of that family duplicated once with pair_id+"#dup" group ids; recipe unchanged.',
        'candidates': {'A_soft_x2_routed': 'predicted soft-play pairs scored by soft_x2, all other pairs keep baseline scores',
                       'B_family_specialized': 'each held pair scored by the variant whose duplicated family equals its fixed predicted family'},
        'routing': 'fixed full-history predicted family of each fold conditional pair candidate (fold0: pair-predictions.csv conditional_class); never true family',
        'confirmation_folds': [1, 2], 'reference_fold': 0,
        'gate': {'per_fold_overall_gain_min': .005, 'per_fold_true_soft_gain_min_for_A': .01, 'per_fold_true_family_loss_max': .01,
                 'pooled_fold1_fold2_pool_halves': 'both nonnegative', 'note': 'must hold on fold1 AND fold2 separately; fold0 excluded'},
        'metric': 'MAP@5 per held target pair, tie seed 931207, ap5 = sum(hits/rank)/min(5,|truth|)', 'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool, per fold'},
        'held_evidence_truth_read_before_freeze': False, 'folds': {},
    }
    write_json(OUT / 'freeze.json', freeze)
    timings = {}
    for fold in [0, 1, 2]:
        data, features, pools, identity = load_fold(fold)
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools))
        evidence = training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence)
        rf = fold_roles(fold)
        hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        # saved-bundle parity under the fold's fixed roles
        bundle = joblib.load(fold_paths(fold)['bundle'])
        assert bundle['features'] == features
        saved = pl.read_parquet(fold_paths(fold)['saved_scores'])
        assert hf.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
        reproduced = score(bundle['model'], family_columns(hf, 'predicted_behavior'), features)
        assert np.array_equal(reproduced, saved['score'].to_numpy()), f'fold{fold}: saved scores not reproduced'
        info = {'identity': identity, 'held_target_pairs': held['pair_id'].n_unique(), 'held_rows': held.height, 'training_queries': train['pair_id'].n_unique(),
                'training_listed_hands': evidence.height, 'training_queries_by_family': {f: int(train.filter(pl.col('behavior_family') == f)['pair_id'].n_unique()) for f in FAMILIES},
                'routed_counts': {f: int(rf.filter(pl.col('predicted_behavior') == f).join(held.select('pair_id').unique(), on='pair_id').height) for f in FAMILIES},
                'saved_scores_reproduced_exactly': True, 'variants': {}}
        variants = {'baseline': train}
        for f in FAMILIES:
            dup = train.filter(pl.col('behavior_family') == f).with_columns((pl.col('pair_id') + '#dup').alias('pair_id'))
            variants[SHORT[f] + '_x2'] = pl.concat([train, dup])
        for name, frame in variants.items():
            t1 = time.perf_counter()
            model = fit_ranker(frame, features)
            timings[f'fold{fold}/{name}'] = time.perf_counter() - t1
            s = score(model, family_columns(hf, 'predicted_behavior'), features)
            assert np.isfinite(s).all()
            if name == 'baseline':
                assert np.array_equal(s, saved['score'].to_numpy()), f'fold{fold}: baseline refit differs from saved'
            hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
            info['variants'][name] = {'queries': frame['pair_id'].n_unique(), 'rows': frame.height, 'positives': int(frame['relevance'].sum()), 'fit_seconds': timings[f'fold{fold}/{name}']}
            print(json.dumps({'fold': fold, 'variant': name, 'seconds': timings[f'fold{fold}/{name}']}), flush=True)
        rf.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        freeze['folds'][str(fold)] = info
    write_json(OUT / 'freeze.json', freeze)
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(OUT / 'freeze.json'),
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').iterdir())}, 'fit_seconds': timings, 'held_evidence_truth_read': False, 'seconds': time.perf_counter() - t0}
    write_json(OUT / 'predictions-frozen.json', frozen)
    print(json.dumps({'predictions_frozen': True, 'seconds': frozen['seconds']}), flush=True)


if __name__ == '__main__':
    main()
