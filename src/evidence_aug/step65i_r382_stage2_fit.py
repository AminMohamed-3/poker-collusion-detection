"""Step 65i: frozen confirmation, on the three held folds, of the per-family stage-2 with the stage-1 ranker's own 382
features added (r382). Origin: training-only audit steps 65h / 65h2 (162 audit pools outside every held fold): on top of
the held-confirmed stage-2 (struct + chrono3 + s1; step 65c/65d allfam_top20_struct, +.041 MAP@5), adding the 382 stage-1
features to each family's second stage gains +.029 [+.007, +.051] (directed transfer +.040, soft play +.034, isolation +.006).
Cross-candidate (type, time) context, top-30 scope and hard counts were null or negative there and are not carried.
No held truth is read here; predictions are frozen with hashes and evaluated once by step 65j.

Per fold (identical to step 65c): stage 1 = chrono1 ranker on the training pools (must reproduce the frozen fold{k}-chrono1
scores exactly), 5-way pool-hash OOF on the training pools for the stage-2 rows; stage 2 = per TRUE family, LGBMClassifier
(step-14 CLF, 3 seeds), top-20 stage-1 candidates, tail keeps stage-1 order; held pairs scored by their FIXED predicted family.
Variants (score files):
  chrono1               baseline (copied from the frozen chronology scores, reproduced)
  allfam_top20_struct   the step-65c design; must reproduce the step-65c frozen scores byte-for-byte (reproduction control)
  allfam_top20_r382     PRIMARY: struct + chrono3 + s1 + 382 ranker features
  allfam_top20_r382ff50 secondary: r382 with column subsampling .5 per tree (audit +.032 [+.012, +.053], = r382 +.003)
  allfam_top20_r382sel  secondary: struct + chrono3 + s1 + the 15 ranker features with the largest |AUC - .5| within the
                        top-20 of the TRAINING rows of that family (selection inside the training pools only)
Note: with the step-14 CLF (no bagging, no column subsampling) LightGBM's random_state has no effect, so the '3 seeds averaged'
of steps 65c/65f/65h are one deterministic model each; only the ff50 variant averages three different models.
Outputs: Claude Version/experiments/family_stage2_r382_20260911/{freeze.json, scores/, predictions-frozen.json}."""
import os
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import sys
import json
import time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, SRC, sha, write_json, read_json, check_protected
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, FAMILIES
from evidence_aug.fold import load_fold, fold_roles
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology, OUT as CHRONO_OUT
from evidence_aug.step14_audit_truncation import split_of, SPLITS, CLF, TOP, OUT as AUDIT_OUT
from evidence_aug.step65_structural_evidence import SEEDS
from evidence_aug.step65c_family_stage2_fit import OUT as PREV, CHRONO1, CHRONO, S1, dev_structural, stage2_predict, with_rank, log
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

OUT = CV / 'experiments' / 'family_stage2_r382_20260911'
SEL = 15
FF50 = dict(CLF, colsample_bytree=.5)


def stage2_fit(train, cols, fam, params):
    rows = train.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP))
    X = rows.select(cols).to_numpy().astype(np.float64); y = rows['relevance'].to_numpy().astype(float)
    models = []
    for seed in SEEDS:
        clf = lgb.LGBMClassifier(**params, random_state=seed); clf.fit(X, y); models.append(clf)
    return models, rows['pair_id'].n_unique(), int(y.sum())


def select_by_auc(rows, cands, k):
    y = rows['relevance'].to_numpy()
    scores = []
    for c in cands:
        x = np.nan_to_num(rows[c].to_numpy().astype(float), nan=-999.0)
        scores.append(abs(roc_auc_score(y, x) - .5) if len(np.unique(x)) > 1 else 0.0)
    order = np.argsort(scores)[::-1][:k]
    return [cands[i] for i in order]


def main():
    t0 = time.perf_counter()
    assert not (OUT / 'freeze.json').exists(), 'freeze exists; preserve it'
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    prev = read_json(PREV / 'results.json')
    assert prev['gates']['allfam_top20_struct']['passed']
    feat = dev_structural(); struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    audit = {n: read_json(AUDIT_OUT / n) for n in ['audit-context-stage2.json', 'audit-r382-variants.json'] if (AUDIT_OUT / n).exists()}
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected,
        'code_sha256': {p.name: sha(p) for p in sorted((SRC / 'evidence_aug').glob('step65*.py'))},
        'origin': {'source': 'steps 65h / 65h2 training-only audit (162 audit pools outside every held fold)', 'audit_sha256': {n: sha(AUDIT_OUT / n) for n in audit},
                   'audit_r382_vs_ref': audit.get('audit-context-stage2.json', {}).get('paired_vs_ref', {}).get('r382'),
                   'audit_r382sel_vs_ref': audit.get('audit-context-stage2.json', {}).get('paired_vs_ref', {}).get('r382_sel'),
                   'audit_variants': {k: {'map5': v['map5'], 'by_family': v['by_family']} for k, v in audit.get('audit-r382-variants.json', {}).get('sets', {}).items()}},
        'previous_confirmation': {'dir': str(PREV), 'results_sha256': sha(PREV / 'results.json'), 'allfam_top20_struct_pooled_delta_vs_chrono1': prev['gates']['allfam_top20_struct']['pooled_delta'],
                                  'held_truth_reads_so_far': 'step 11 (chronology), step 65d (stage-2); this is the third read of the same three folds'},
        'stage1': 'chrono1 ranker (382 + chrono_frac, production PARAMS) on training pools; 5-way pool-hash OOF (audit-trunc-v1) on training pools for stage-2 training rows',
        'stage2': {'model': 'LGBMClassifier ' + json.dumps(CLF) + ', seeds ' + str(SEEDS) + ' averaged', 'rows': 'top-20 stage-1 candidates of the training pools, one model per TRUE family; tail keeps stage-1 order',
                   'features': {'struct': struct, 'chrono': CHRONO, 's1': S1, 'r382': 'the fold ranker feature list minus family_* columns and minus names already in struct/chrono/s1 (pot_bb)', 'r382sel': f'top-{SEL} of r382 by |AUC-.5| within the top-20 training rows of the family'},
                   'ff50': 'LGBMClassifier ' + json.dumps(FF50) + ' (column subsampling; here the three seeds differ)', 'seed_note': 'with the step-14 CLF random_state has no effect (no bagging / column subsampling): the three seeds are one deterministic model'},
        'routing': 'held pairs scored under the FIXED predicted family of the fold (fold_roles); predicted none keeps chrono1',
        'variants': {'chrono1': 'baseline', 'allfam_top20_struct': 'step-65c design, reproduction control (asserted byte-identical to the step-65c frozen scores)',
                     'allfam_top20_r382': 'PRIMARY', 'allfam_top20_r382ff50': 'secondary (column subsampling .5)', 'allfam_top20_r382sel': 'secondary (nested top-15 selection)'},
        'confirmation_folds': [0, 1, 2],
        'gate': {'applies_to': 'allfam_top20_r382 vs allfam_top20_struct on folds 0, 1, 2 (paired per held pair)', 'pooled_delta_min': .01, 'pooled_pool_bootstrap_95_excludes_zero': True,
                 'per_fold_gain_nonnegative_in_at_least': 2, 'per_fold_true_family_loss_max': .015, 'pooled_true_family_loss_max': .0, 'pooled_pool_halves': 'both nonnegative',
                 'decision': 'PASS -> build candidate M3_stage2_r382 on top of the step-65f production recipe with the r382 feature set; FAIL -> keep M3_stage2_chrono3, record the failure. The two secondaries are reported under the same gate; if only a secondary passes, using it is one selection among three and is recorded as such with a widened interval.'},
        'prediction_if_pass': 'held pooled delta vs allfam_top20_struct in [+.01, +.04]; public E about +.01 to +.03 on top of M3_stage2_chrono3 (composite +.002 to +.006)',
        'metric': 'MAP@5 per held target pair, tie seed 931207', 'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool'},
        'held_evidence_truth_read_before_freeze': False, 'folds': {}}
    write_json(OUT / 'freeze.json', freeze)
    for fold in [0, 1, 2]:
        t1 = time.perf_counter()
        data, features, pools, identity = load_fold(fold)
        r382 = [f for f in features if not f.startswith('family_') and f in data.columns and f not in struct + CHRONO + S1]
        data = add_chronology(data).join(feat, on=['pair_id', 'hand_id'], how='left', validate='1:1')
        assert data.select(pl.sum_horizontal(pl.col(struct).null_count())).item() == 0
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools))
        evidence = training_relevance(set(train['table_id'])); assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence).sort('pair_id', 'hand_id')
        rf = fold_roles(fold); hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        model = fit_ranker(train, features + CHRONO1)
        s_held = score(model, family_columns(hf, 'predicted_behavior'), features + CHRONO1)
        saved = pl.read_parquet(CHRONO_OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        assert hf.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id')) and np.array_equal(s_held, saved['score'].to_numpy()), f'fold{fold}: chrono1 refit differs from the frozen scores'
        hf = with_rank(hf.with_columns(pl.Series('s1', s_held)), 's1', 's1_rank')
        train = train.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
        oof = np.full(train.height, np.nan)
        for s in range(SPLITS):
            idx = np.where(train['split'].to_numpy() == s)[0]
            m = fit_ranker(train.filter(pl.col('split') != s), features + CHRONO1)
            oof[idx] = score(m, family_columns(train[idx], 'behavior_family'), features + CHRONO1)
        assert np.isfinite(oof).all()
        train = with_rank(train.with_columns(pl.Series('s1', oof)), 's1', 's1_rank')
        info = {'identity': identity, 'held_target_pairs': held['pair_id'].n_unique(), 'held_rows': held.height, 'training_queries': train['pair_id'].n_unique(), 'training_listed_hands': evidence.height,
                'n_r382': len(r382), 'predicted_family_counts_held': {k: int(v) for k, v in hf.select('pair_id', 'predicted_behavior').unique()['predicted_behavior'].value_counts().iter_rows()}, 'stage2': {}, 'selected': {}}
        base = hf['s1'].to_numpy(); in_top = (hf['s1_rank'] <= TOP).to_numpy()
        preds = {}
        for sname in ['struct', 'r382', 'r382ff50', 'r382sel']:
            preds[sname] = np.full(hf.height, np.nan)
            for fam in FAMILIES:
                params = FF50 if sname == 'r382ff50' else CLF
                if sname == 'struct':
                    cols = struct + CHRONO + S1
                elif sname in ('r382', 'r382ff50'):
                    cols = struct + CHRONO + S1 + r382
                else:
                    rows = train.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP))
                    sel = select_by_auc(rows, r382, SEL); info['selected'][fam] = sel; cols = struct + CHRONO + S1 + sel
                models, nq, npos = stage2_fit(train, cols, fam, params)
                mask = (hf['predicted_behavior'] == fam).to_numpy()
                if mask.any():
                    preds[sname][mask] = stage2_predict(models, hf.filter(pl.Series(mask)), cols)
                g = np.mean([m.booster_.feature_importance('gain') for m in models], axis=0); order = np.argsort(g)[::-1][:10]
                info['stage2'][f'{sname}/{fam}'] = {'training_queries': nq, 'training_rows': int(train.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP)).height), 'training_listed': npos, 'n_features': len(cols),
                                                    'top_gain': [(cols[i], round(float(g[i] / g.sum()), 3)) for i in order]}
            preds[sname] = np.where(in_top, preds[sname] + 10.0, base - 1e6)
        has_fam = hf['predicted_behavior'].is_in(FAMILIES).to_numpy()
        variants = {'chrono1': base, 'allfam_top20_struct': np.where(has_fam, preds['struct'], base), 'allfam_top20_r382': np.where(has_fam, preds['r382'], base),
                    'allfam_top20_r382ff50': np.where(has_fam, preds['r382ff50'], base), 'allfam_top20_r382sel': np.where(has_fam, preds['r382sel'], base)}
        prev_scores = pl.read_parquet(PREV / 'scores' / f'fold{fold}-allfam_top20_struct.parquet')
        assert prev_scores.select('pair_id', 'hand_id').equals(hf.select('pair_id', 'hand_id')) and np.array_equal(prev_scores['score'].to_numpy(), variants['allfam_top20_struct']), f'fold{fold}: step-65c stage-2 not reproduced'
        info['reproduced_step65c_allfam_top20_struct'] = True
        for name, s in variants.items():
            assert np.isfinite(s).all(), name
            hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        rf.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        info['seconds'] = round(time.perf_counter() - t1, 1)
        freeze['folds'][str(fold)] = info
        log(json.dumps({'fold': fold, 'held_pairs': info['held_target_pairs'], 'n_r382': len(r382), 'reproduced_65c': True, 'seconds': info['seconds'], 'dt_top_gain': info['stage2'][f'r382/directed_transfer']['top_gain'][:5]}))
    write_json(OUT / 'freeze.json', freeze)
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(OUT / 'freeze.json'),
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').glob('*.parquet')) + sorted((OUT / 'scores').glob('*.csv'))}, 'held_evidence_truth_read': False, 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'predictions-frozen.json', frozen)
    log(json.dumps({'predictions_frozen': True, 'seconds': frozen['seconds']}))


if __name__ == '__main__':
    main()
