"""Step 65c: frozen confirmation of the per-family second-stage evidence model (structural action features) on the
three held folds. Origin: training-only audit (steps 65, 65b; 162 audit pools outside every held fold): a
coordinated-isolation-only second stage over ALL hands with ~70 structural relational-action features + chronology +
the ranker's out-of-fold score lifts CI MAP@5 .40 -> .56 while a pooled model does not learn it. No held truth is read
here; predictions are frozen with hashes and evaluated once by step 65d.

Per fold: stage 1 = the chrono1 ranker (382 + chrono_frac) fitted on training pools (must reproduce the saved
fold{k}-chrono1.parquet held scores exactly); its 5-way pool-hash out-of-fold scores on the training pools feed
stage 2. Stage 2 = per TRUE family on training pools, LGBMClassifier (step-14 CLF, 3 seeds averaged) over all hands
with structural + chronology + stage-1 score/rank. Held pairs are scored under their FIXED predicted family
(fold_roles), never the true one.
Stage 2 is fitted either on the top-20 stage-1 candidates of each training pair (the audit's best design, CI .40 -> .61,
tail keeps stage-1 order) or on all hands (CI .40 -> .56).
Variants (score files):  chrono1 (baseline) | hybridCI_top20_struct (PRIMARY: predicted-CI pairs get their top-20 re-ranked
by the CI stage-2, others keep chrono1) | hybridCI_all_struct (secondary, all-hands stage 2) | hybridCI_top20_nostruct
(control: stage 2 without structural features) | allfam_top20_struct (every pair re-ranked by its predicted family's stage-2).
Outputs: Claude Version/experiments/family_stage2_20260911/{freeze.json, scores/, predictions-frozen.json,
dev-structural-features.parquet}."""
import os
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import sys
import json
import time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, SRC, ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, FAMILIES, CACHES
from evidence_aug.fold import load_fold, fold_roles, fold_pools
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology, OUT as CHRONO_OUT
from evidence_aug.step14_audit_truncation import split_of, SPLITS, CLF, TOP, OUT as AUDIT_OUT
from evidence_aug.step65_structural_evidence import build, SEEDS
import numpy as np
import polars as pl
import lightgbm as lgb

OUT = CV / 'experiments' / 'family_stage2_20260911'
CHRONO1 = ['chrono_frac']
CHRONO = ['chrono_rank', 'chrono_frac', 'n_hands']
S1 = ['s1', 's1_rank']
CI = 'coordinated_isolation'


def log(*a):
    print(*a, flush=True)


def dev_structural():
    path = OUT / 'dev-structural-features.parquet'
    if path.exists():
        return pl.read_parquet(path)
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    keys = pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet')), columns=['pair_id', 'hand_id']).join(labels.filter(pl.col('label') == 1).select('pair_id'), on='pair_id', how='semi')
    feat = build(keys)
    feat.write_parquet(path)
    return feat


def stage2_fit(train, cols, fam, top_only):
    rows = train.filter(pl.col('behavior_family') == fam)
    if top_only:
        rows = rows.filter(pl.col('s1_rank') <= TOP)
    X = rows.select(cols).to_numpy().astype(np.float64); y = rows['relevance'].to_numpy().astype(float)
    models = []
    for seed in SEEDS:
        clf = lgb.LGBMClassifier(**CLF, random_state=seed); clf.fit(X, y); models.append(clf)
    return models, rows['pair_id'].n_unique(), int(y.sum())


def stage2_predict(models, frame, cols):
    X = frame.select(cols).to_numpy().astype(np.float64)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def with_rank(frame, col, name):
    return frame.with_columns(pl.col(col).rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias(name))


def main():
    t0 = time.perf_counter()
    assert not (OUT / 'freeze.json').exists(), 'freeze exists; preserve it'
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    feat = dev_structural(); struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    audit = read_json(AUDIT_OUT / 'audit-family-stage2.json') if (AUDIT_OUT / 'audit-family-stage2.json').exists() else {}
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected,
        'code_sha256': {p.name: sha(p) for p in sorted((SRC / 'evidence_aug').glob('step65*.py'))},
        'origin': {'source': 'steps 65 / 65b training-only audit (162 audit pools outside every held fold)', 'audit_sha256': {n: sha(AUDIT_OUT / n) for n in ['audit-structural.json', 'audit-family-stage2.json'] if (AUDIT_OUT / n).exists()},
                   'audit_hybrid': {k: {'map5': v['map5'], 'delta_vs_ref': v['delta_vs_ref'], 'ci95': v['bootstrap_all']['ci95']} for k, v in audit.get('hybrid', {}).items()}},
        'stage1': 'chrono1 ranker (382 + chrono_frac, production PARAMS) on training pools; 5-way pool-hash OOF (audit-trunc-v1) on training pools for stage-2 training rows',
        'stage2': {'model': 'LGBMClassifier ' + json.dumps(CLF) + ', seeds ' + str(SEEDS) + ' averaged', 'rows': 'top-20 stage-1 candidates (or all hands for the *_all_* variant) of the training pools, one model per TRUE family',
                   'features': {'struct': struct, 'chrono': CHRONO, 's1': S1}},
        'routing': 'held pairs scored under the FIXED predicted family of the fold (fold_roles); predicted none keeps chrono1',
        'variants': {'chrono1': 'baseline = saved fold{k}-chrono1 scores (must reproduce)', 'hybridCI_top20_struct': 'PRIMARY: predicted-CI pairs get their top-20 stage-1 candidates re-ranked by the CI stage-2 (struct + chrono + s1), tail keeps stage-1 order; others keep chrono1',
                     'hybridCI_all_struct': 'secondary: same routing, stage 2 fitted and applied on all hands', 'hybridCI_top20_nostruct': 'control: same routing as primary, stage 2 = chrono + s1 only',
                     'allfam_top20_struct': 'every pair with a predicted family gets its top-20 re-ranked by that family stage-2'},
        'confirmation_folds': [0, 1, 2],
        'gate': {'applies_to': 'hybridCI_top20_struct vs chrono1 on folds 0, 1, 2', 'per_fold_overall_gain_min': .01, 'per_fold_true_family_loss_max': .01, 'pooled_true_CI_gain_min': .05,
                 'pooled_pool_bootstrap_95_excludes_zero': True, 'pooled_pool_halves': 'both nonnegative'},
        'metric': 'MAP@5 per held target pair, tie seed 931207', 'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool'},
        'held_evidence_truth_read_before_freeze': False, 'folds': {}}
    write_json(OUT / 'freeze.json', freeze)
    for fold in [0, 1, 2]:
        t1 = time.perf_counter()
        data, features, pools, identity = load_fold(fold)
        data = add_chronology(data).join(feat, on=['pair_id', 'hand_id'], how='left', validate='1:1')
        assert data.select(pl.sum_horizontal(pl.col(struct).null_count())).item() == 0
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools))
        evidence = training_relevance(set(train['table_id'])); assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence).sort('pair_id', 'hand_id')
        rf = fold_roles(fold); hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        # stage 1 on held: refit chrono1 and check against the frozen chronology scores
        model = fit_ranker(train, features + CHRONO1)
        s_held = score(model, family_columns(hf, 'predicted_behavior'), features + CHRONO1)
        saved = pl.read_parquet(CHRONO_OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        assert hf.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id')) and np.array_equal(s_held, saved['score'].to_numpy()), f'fold{fold}: chrono1 refit differs from the frozen scores'
        hf = with_rank(hf.with_columns(pl.Series('s1', s_held)), 's1', 's1_rank')
        # stage 1 OOF on the training pools
        train = train.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
        oof = np.full(train.height, np.nan)
        for s in range(SPLITS):
            idx = np.where(train['split'].to_numpy() == s)[0]
            m = fit_ranker(train.filter(pl.col('split') != s), features + CHRONO1)
            oof[idx] = score(m, family_columns(train[idx], 'behavior_family'), features + CHRONO1)
        assert np.isfinite(oof).all()
        train = with_rank(train.with_columns(pl.Series('s1', oof)), 's1', 's1_rank')
        # stage 2 per family
        info = {'identity': identity, 'held_target_pairs': held['pair_id'].n_unique(), 'held_rows': held.height, 'training_queries': train['pair_id'].n_unique(), 'training_listed_hands': evidence.height,
                'predicted_family_counts_held': {k: int(v) for k, v in hf.select('pair_id', 'predicted_behavior').unique()['predicted_behavior'].value_counts().iter_rows()}, 'stage2': {}}
        sets = {'top20_struct': (struct + CHRONO + S1, True), 'all_struct': (struct + CHRONO + S1, False), 'top20_nostruct': (CHRONO + S1, True)}
        base = hf['s1'].to_numpy(); in_top = (hf['s1_rank'] <= TOP).to_numpy()
        preds = {}
        for sname, (cols, top_only) in sets.items():
            preds[sname] = np.full(hf.height, np.nan)
            for fam in FAMILIES:
                models, nq, npos = stage2_fit(train, cols, fam, top_only)
                mask = (hf['predicted_behavior'] == fam).to_numpy()
                if mask.any():
                    preds[sname][mask] = stage2_predict(models, hf.filter(pl.Series(mask)), cols)
                g = np.mean([m.booster_.feature_importance('gain') for m in models], axis=0); order = np.argsort(g)[::-1][:8]
                info['stage2'][f'{sname}/{fam}'] = {'training_queries': nq, 'training_rows': int(train.filter((pl.col('behavior_family') == fam) & ((pl.col('s1_rank') <= TOP) if top_only else True)).height), 'training_listed': npos, 'top_gain': [(cols[i], round(float(g[i] / g.sum()), 3)) for i in order]}
            if top_only:
                preds[sname] = np.where(in_top, preds[sname] + 10.0, base - 1e6)
        is_ci = (hf['predicted_behavior'] == CI).to_numpy(); has_fam = hf['predicted_behavior'].is_in(FAMILIES).to_numpy()
        variants = {'chrono1': base, 'hybridCI_top20_struct': np.where(is_ci, preds['top20_struct'], base), 'hybridCI_all_struct': np.where(is_ci, preds['all_struct'], base),
                    'hybridCI_top20_nostruct': np.where(is_ci, preds['top20_nostruct'], base), 'allfam_top20_struct': np.where(has_fam, preds['top20_struct'], base)}
        for name, s in variants.items():
            assert np.isfinite(s).all(), name
            hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        rf.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        info['seconds'] = round(time.perf_counter() - t1, 1)
        freeze['folds'][str(fold)] = info
        log(json.dumps({'fold': fold, 'held_pairs': info['held_target_pairs'], 'predicted_CI_held': info['predicted_family_counts_held'].get(CI, 0), 'seconds': info['seconds'], 'ci_top_gain': info['stage2'][f'top20_struct/{CI}']['top_gain'][:5]}))
    write_json(OUT / 'freeze.json', freeze)
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(OUT / 'freeze.json'),
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').glob('*.parquet')) + sorted((OUT / 'scores').glob('*.csv'))}, 'held_evidence_truth_read': False, 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'predictions-frozen.json', frozen)
    log(json.dumps({'predictions_frozen': True, 'seconds': frozen['seconds']}))


if __name__ == '__main__':
    main()
