"""Step 68d: frozen held confirmation fit for E1 stage-2 ensemble (+ E3 ranker shared arms).

ONE 65i-pattern fit on held folds 0-2 (this is the 5th held-truth read per tasking:
prior reads steps 11, 65d, 65j; 65j2 derived from pairs.csv without new truth;
reused folds, NOT pristine). No held truth is read here; predictions are frozen
with hashes and evaluated once by step 68e.

Per fold (identical to step 65i): stage 1 = chrono1 ranker on training pools (must
reproduce the frozen fold{k}-chrono1 scores exactly); 5-way pool-hash OOF on the
training pools for stage-2 rows; stage 2 = per TRUE family, LGBMClassifier
(step-14 CLF, seeds 11/12/13 averaged), top-20 stage-1 candidates, tail keeps
stage-1 order; held pairs scored by FIXED predicted family. E3 arms = per TRUE
family LGBMRanker (lambdarank, RNK params, seeds averaged) on the same rows.

Score files per fold:
  chrono1               baseline (reproduced, asserted byte-identical)
  allfam_top20_struct   reproduction control (asserted byte-identical to 65c)
  allfam_top20_r382     reproduction control (asserted byte-identical to 65i)
  hybrid_DTSP_r382      REFERENCE (M3_r382DTSP production pattern: r382 score for
                        predicted transfer/soft-play, struct score for predicted
                        isolation, chrono1 for predicted none; derived, no training)
  ens_rank              PRIMARY: per-pair rank-average of struct & r382 held proba
                        preds within the pair top-20 (68a rank_within_pair pattern)
  rank_ref              SECONDARY E3: lambdarank on ref feats
  rank_r382             SECONDARY E3: lambdarank on r382 feats
Outputs: Claude Version/experiments/stage2_ensemble_held_E1_20260913/{freeze.json,
scores/, predictions-frozen.json}."""
import os
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
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
from evidence_aug.step65c_family_stage2_fit import OUT as PREV_C, CHRONO1, CHRONO, S1, dev_structural, stage2_predict, with_rank, log
from evidence_aug.step65i_r382_stage2_fit import OUT as PREV_I
import lightgbm as lgb
import numpy as np
import polars as pl

OUT = CV / 'experiments' / 'stage2_ensemble_held_E1_20260913'
DT, SP, CI = 'directed_transfer', 'soft_play', 'coordinated_isolation'
RNK = dict(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=40,
           learning_rate=.05, n_jobs=2, verbosity=-1)
THIS_FIT = 'step68d_held_ensemble_fit.py'
THIS_EVAL = 'step68e_held_ensemble_evaluate.py'


def group_sizes(pid):
    _, idx, cnt = np.unique(pid, return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


def stage2_fit(train, cols, fam, params):
    rows = train.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP))
    X = rows.select(cols).to_numpy().astype(np.float64)
    y = rows['relevance'].to_numpy().astype(float)
    models = []
    for seed in SEEDS:
        clf = lgb.LGBMClassifier(**params, random_state=seed)
        clf.fit(X, y)
        models.append(clf)
    return models, rows['pair_id'].n_unique(), int(y.sum())


def ranker_fit(train, cols, fam):
    rows = train.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP)).sort('pair_id', 'hand_id')
    X = rows.select(cols).to_numpy().astype(np.float64)
    y = rows['relevance'].to_numpy().astype(int)
    pid = rows['pair_id'].to_numpy()
    gs = group_sizes(pid)
    models = []
    for seed in SEEDS:
        mdl = lgb.LGBMRanker(objective='lambdarank', random_state=seed, eval_at=[5], **RNK)
        mdl.fit(X, y, group=gs)
        models.append(mdl)
    return models, rows['pair_id'].n_unique(), int(y.sum())


def ranker_predict(models, frame, cols):
    X = frame.select(cols).to_numpy().astype(np.float64)
    return np.mean([m.predict(X) for m in models], axis=0)


def rank_within_pair(pair_ids, pred, cand_mask):
    out = np.zeros(len(pred))
    for p in np.unique(pair_ids[cand_mask]):
        m = cand_mask & (pair_ids == p)
        v = pred[m]
        r = np.empty(len(v))
        r[np.argsort(v)] = np.arange(len(v))
        out[m] = r / max(len(v) - 1, 1)
    return out


def main():
    t0 = time.perf_counter()
    assert not (OUT / 'freeze.json').exists(), 'freeze exists; preserve it'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scores').mkdir(exist_ok=True)
    protected_before = check_protected()
    feat = dev_structural()
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    wave2_freeze_path = CV.parent / '.opencode' / 'continuous' / 'tasks' / 'evidence' / 'freeze.wave2.json'
    m3plan_path = CV.parent / '.opencode' / 'continuous' / 'plan.M3_r382DTSP.json'
    m3pred_path = CV.parent / '.opencode' / 'continuous' / 'prediction.M3_r382DTSP.json'
    audit_out = AUDIT_OUT / 'audit-stage2-ensemble.json'
    prev_i_freeze = read_json(PREV_I / 'freeze.json')
    prev_i_frozen = read_json(PREV_I / 'predictions-frozen.json')
    audit_res = read_json(audit_out)
    freeze = {
        'study': 'E1_stage2_ensemble_held',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'protected_before': protected_before,
        'code_sha256': {
            THIS_FIT: sha(SRC / 'evidence_aug' / THIS_FIT),
            THIS_EVAL: sha(SRC / 'evidence_aug' / THIS_EVAL),
            'step68a_audit_stage2_ensemble.py': sha(SRC / 'evidence_aug' / 'step68a_audit_stage2_ensemble.py'),
            'step65i_r382_stage2_fit.py': sha(SRC / 'evidence_aug' / 'step65i_r382_stage2_fit.py'),
            'step65j_r382_stage2_evaluate.py': sha(SRC / 'evidence_aug' / 'step65j_r382_stage2_evaluate.py'),
            'step65c_family_stage2_fit.py': sha(SRC / 'evidence_aug' / 'step65c_family_stage2_fit.py'),
            'step65b_family_stage2.py': sha(SRC / 'evidence_aug' / 'step65b_family_stage2.py'),
            'step5_evaluate.py': sha(SRC / 'evidence_aug' / 'step5_evaluate.py'),
            'step14_audit_truncation.py': sha(SRC / 'evidence_aug' / 'step14_audit_truncation.py'),
            'frame.py': sha(SRC / 'evidence_aug' / 'frame.py'),
            'fold.py': sha(SRC / 'evidence_aug' / 'fold.py'),
            'paths.py': sha(SRC / 'evidence_aug' / 'paths.py'),
        },
        'origin': {
            'audit': '68a training-only audit (162 pools outside every held fold)',
            'audit_output_sha256': sha(audit_out),
            'audit_ens_rank': audit_res['paired_vs_ref']['ens_rank'],
            'prior_held_65i_freeze_sha256': sha(PREV_I / 'freeze.json'),
            'prior_held_65i_predictions_frozen_sha256': sha(PREV_I / 'predictions-frozen.json'),
            'freeze_wave2_sha256': sha(wave2_freeze_path),
            'plan_M3_r382DTSP_sha256': sha(m3plan_path),
            'prediction_M3_r382DTSP_sha256': sha(m3pred_path),
        },
        'stage1': 'chrono1 ranker (382 + chrono_frac, production PARAMS) on training pools; 5-way pool-hash OOF (audit-trunc-v1) on training pools for stage-2 training rows',
        'stage2': {
            'classifier': 'LGBMClassifier ' + json.dumps(CLF) + ', seeds ' + str(list(SEEDS)) + ' averaged',
            'ranker': 'LGBMRanker lambdarank ' + json.dumps(RNK) + ', seeds ' + str(list(SEEDS)) + ' averaged, group=rows-per-pair (pair-sorted), eval_at=[5]',
            'rows': 'top-20 stage-1 candidates of the training pools, one model per TRUE family; tail keeps stage-1 order',
            'features': {'struct': 'dev-structural (~70) + chrono + s1', 'r382': 'struct set + 378 ranker columns (fold ranker list minus family_* and pot_bb)'},
        },
        'routing': 'held pairs scored under the FIXED predicted family of the fold (fold_roles); predicted none keeps chrono1',
        'reference': 'hybrid_DTSP_r382 (M3_r382DTSP production pattern: r382 for predicted DT/SP, struct for predicted CI, chrono1 for predicted none)',
        'variants': {
            'chrono1': 'baseline = saved fold{k}-chrono1 scores (must reproduce)',
            'allfam_top20_struct': 'reproduction control (asserted byte-identical to step-65c frozen scores)',
            'allfam_top20_r382': 'reproduction control (asserted byte-identical to step-65i frozen scores); base of the ensemble',
            'hybrid_DTSP_r382': 'REFERENCE (derived splice, no training)',
            'ens_rank': 'PRIMARY: per-pair rank-average of struct & r382 held proba preds within top-20',
            'rank_ref': 'SECONDARY E3: lambdarank on ref feats',
            'rank_r382': 'SECONDARY E3: lambdarank on r382 feats',
        },
        'dropped_audit_variants': 'ens_prob/bag_ref/bag_r382/ens_bag audit-portion FAIL (68a); record-only, no held spend under this single-read claim',
        'confirmation_folds': [0, 1, 2],
        'held_truth_read': 'FIFTH read of folds 0-2 per tasking (prior: steps 11, 65d, 65j; 65j2 derived from pairs.csv with no new truth). Reused, NOT pristine; recorded here.',
        'gate': {'applies_to': 'each gated arm (ens_rank, rank_ref, rank_r382) vs hybrid_DTSP_r382 on folds 0, 1, 2 (paired per held pair)',
                 'pooled_delta_ge': .01, 'pooled_pool_bootstrap_95_excludes_zero': True,
                 'per_fold_gain_nonnegative_in_all': 3, 'per_fold_true_family_loss_max': .01,
                 'pooled_true_family_loss_max': .01, 'pooled_pool_halves': 'both nonnegative',
                 'decision': 'PASS -> READY-FOR-PRODUCTION via step65k pattern (prediction JSON first, 112540x562700 validation, user upload). FAIL -> record only, no production, no second read; promote equity4096.'},
        'metric': 'MAP@5 per held target pair, tie seed 931207',
        'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool'},
        'held_evidence_truth_read_before_freeze': False,
        'folds': {},
    }
    write_json(OUT / 'freeze.json', freeze)
    for fold in [0, 1, 2]:
        t1 = time.perf_counter()
        data, features, pools, identity = load_fold(fold)
        r382 = [f for f in features if not f.startswith('family_') and f in data.columns and f not in struct + CHRONO + S1]
        data = add_chronology(data).join(feat, on=['pair_id', 'hand_id'], how='left', validate='1:1')
        assert data.select(pl.sum_horizontal(pl.col(struct).null_count())).item() == 0
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools))
        evidence = training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence).sort('pair_id', 'hand_id')
        rf = fold_roles(fold)
        hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
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
        info = {'identity': identity, 'held_target_pairs': held['pair_id'].n_unique(), 'held_rows': held.height,
                'training_queries': train['pair_id'].n_unique(), 'training_listed_hands': evidence.height,
                'n_r382': len(r382),
                'predicted_family_counts_held': {k: int(v) for k, v in hf.select('pair_id', 'predicted_behavior').unique()['predicted_behavior'].value_counts().iter_rows()},
                'stage2': {}}
        base = hf['s1'].to_numpy()
        in_top = (hf['s1_rank'] <= TOP).to_numpy()
        probs, ranks = {}, {}
        for sname, cols in [('struct', struct + CHRONO + S1), ('r382', struct + CHRONO + S1 + r382)]:
            probs[sname] = np.full(hf.height, np.nan)
            for fam in FAMILIES:
                models, nq, npos = stage2_fit(train, cols, fam, CLF)
                mask = (hf['predicted_behavior'] == fam).to_numpy()
                if mask.any():
                    probs[sname][mask] = stage2_predict(models, hf.filter(pl.Series(mask)), cols)
                g = np.mean([m.booster_.feature_importance('gain') for m in models], axis=0)
                order = np.argsort(g)[::-1][:10]
                info['stage2'][f'{sname}/{fam}'] = {'training_queries': nq, 'n_features': len(cols),
                                                    'top_gain': [(cols[i], round(float(g[i] / g.sum()), 3)) for i in order]}
            probs[sname] = np.where(in_top, probs[sname] + 10.0, base - 1e6)
        for sname, cols in [('rank_ref', struct + CHRONO + S1), ('rank_r382', struct + CHRONO + S1 + r382)]:
            ranks[sname] = np.full(hf.height, np.nan)
            for fam in FAMILIES:
                models, nq, npos = ranker_fit(train, cols, fam)
                mask = (hf['predicted_behavior'] == fam).to_numpy()
                if mask.any():
                    ranks[sname][mask] = ranker_predict(models, hf.filter(pl.Series(mask)), cols)
                info['stage2'][f'{sname}/{fam}'] = {'training_queries': nq, 'n_features': len(cols)}
            ranks[sname] = np.where(in_top, ranks[sname] + 10.0, base - 1e6)
        has_fam = hf['predicted_behavior'].is_in(FAMILIES).to_numpy()
        is_dtsp = hf['predicted_behavior'].is_in([DT, SP]).to_numpy()
        pair_ids = hf['pair_id'].to_numpy()
        cand = in_top & has_fam
        ens01 = 0.5 * (rank_within_pair(pair_ids, probs['struct'], cand) + rank_within_pair(pair_ids, probs['r382'], cand))
        ens_score = np.where(has_fam, np.where(in_top, ens01 + 10.0, base - 1e6), base)
        hybrid = np.where(has_fam, np.where(is_dtsp, probs['r382'], probs['struct']), base)
        variants = {'chrono1': base, 'allfam_top20_struct': np.where(has_fam, probs['struct'], base),
                    'allfam_top20_r382': np.where(has_fam, probs['r382'], base), 'hybrid_DTSP_r382': hybrid,
                    'ens_rank': ens_score, 'rank_ref': np.where(has_fam, ranks['rank_ref'], base),
                    'rank_r382': np.where(has_fam, ranks['rank_r382'], base)}
        prev_c = pl.read_parquet(PREV_C / 'scores' / f'fold{fold}-allfam_top20_struct.parquet')
        assert prev_c.select('pair_id', 'hand_id').equals(hf.select('pair_id', 'hand_id')) and np.array_equal(prev_c['score'].to_numpy(), variants['allfam_top20_struct']), f'fold{fold}: 65c struct not reproduced'
        prev_i = pl.read_parquet(PREV_I / 'scores' / f'fold{fold}-allfam_top20_r382.parquet')
        assert prev_i.select('pair_id', 'hand_id').equals(hf.select('pair_id', 'hand_id')) and np.array_equal(prev_i['score'].to_numpy(), variants['allfam_top20_r382']), f'fold{fold}: 65i r382 not reproduced'
        info['reproduced_65c_struct'] = True
        info['reproduced_65i_r382'] = True
        for name, s in variants.items():
            assert np.isfinite(s).all(), name
            hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        rf.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        info['seconds'] = round(time.perf_counter() - t1, 1)
        freeze['folds'][str(fold)] = info
        log(json.dumps({'fold': fold, 'held_pairs': info['held_target_pairs'], 'n_r382': len(r382), 'reproduced_65c_65i': True, 'seconds': info['seconds']}))
    write_json(OUT / 'freeze.json', freeze)
    protected_after = check_protected()
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha(OUT / 'freeze.json'),
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').glob('*.parquet')) + sorted((OUT / 'scores').glob('*.csv'))},
              'held_evidence_truth_read': False, 'protected_after': protected_after,
              'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'predictions-frozen.json', frozen)
    log(json.dumps({'predictions_frozen': True, 'seconds': frozen['seconds']}))


if __name__ == '__main__':
    main()
