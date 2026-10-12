"""Step 72: frozen confirmation, on the three held folds, of structural + mechanism features INSIDE STAGE 1.

Origin (training-only, 162 audit pools outside every held fold):
  step 71  adding the ~70 step-65 structural columns and the 21 step-70 mechanism columns to the
           stage-1 LambdaRank lifts stage-1 MAP@5 .5574 -> .5813 at the production seed; either block
           alone is null (struct +.0078 crossing zero, f2 -.0001).
  step 71b over three ranker seeds the delta is +.0197 [+.0047, +.0356], positive at every seed
           (+.0239 / +.0058 / +.0295), positive in 4 of the 5 pool-hash splits, per family
           DT +.0129 / SP +.0139 / CI +.0375; the within-pair PERMUTATION of the same 91 columns
           gives -.0099, so the gain is the hand-level signal and not the extra capacity.
Why stage 1 and not stage 2: the public reads of P0 attributed the composite exactly -- the r382
ranker-feature increment transferred to the evaluation set at 88 %, the structural STAGE-2
increment at -1 % (held +.0411, public -.0002).  Stage 1 is the part of the chain that has a
demonstrated transfer rate, and it is also the only part that can change which hands stage 2 sees.

No held truth is read here; predictions are frozen with hashes and evaluated once by step 72b.
This is the FOURTH read of folds 0-2 (steps 11, 65d, 65j, this) and the single read of this study.

Per fold, exactly the step-65c / 65i construction:
  stage 1  fitted on the training pools; 5-way pool-hash OOF (audit-trunc-v1) on the training pools
           supplies the stage-2 training rows; held pairs scored under their FIXED predicted family.
  stage 2  per TRUE family, LGBMClassifier (step-14 CLF, 3 seeds), top-20 stage-1 candidates, tail
           keeps the stage-1 order.  Stage 2 is UNCHANGED from production in every arm, so the only
           thing that moves between an old and a new arm is the stage-1 feature list.
Variants (score files):
  chrono1                 stage 1 = 382 + chrono_frac, no stage 2.  Reproduction control: asserted
                          equal to the frozen step-10 fold{k}-chrono1 scores.
  allfam_top20_struct     chrono1 + the struct stage 2.  Reproduction control vs step 65c.
  allfam_top20_r382       chrono1 + the r382 stage 2.  Reproduction control vs step 65i (what ships).
  s1f2                    PRIMARY: stage 1 = 382 + chrono_frac + struct + f2, no stage 2.
  s1f2_top20_struct       the new stage 1 + the struct stage 2.
  s1f2_top20_r382         the new stage 1 + the r382 stage 2 (the end-to-end production analogue).
Outputs: Claude Version/experiments/stage1_f2_20260912/{freeze.json, scores/, predictions-frozen.json,
dev-f2-features.parquet}."""
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
from evidence_aug.step65c_family_stage2_fit import (OUT as STRUCT_OUT, CHRONO1, CHRONO, S1,
                                                    dev_structural, stage2_predict, with_rank, log)
from evidence_aug.step65i_r382_stage2_fit import OUT as R382_OUT, stage2_fit
from evidence_aug.step70_family_structural import build2
import numpy as np
import polars as pl

OUT = CV / 'experiments' / 'stage1_f2_20260912'


def dev_f2():
    path = OUT / 'dev-f2-features.parquet'
    if path.exists():
        return pl.read_parquet(path)
    struct = dev_structural().select('pair_id', 'hand_id')
    feat = build2(add_chronology(struct))
    OUT.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(path)
    return feat


def oof_stage1(train, cols):
    oof = np.full(train.height, np.nan)
    sp = train['split'].to_numpy()
    for s in range(SPLITS):
        idx = np.where(sp == s)[0]
        m = fit_ranker(train.filter(pl.col('split') != s), cols)
        oof[idx] = score(m, family_columns(train[idx], 'behavior_family'), cols)
    assert np.isfinite(oof).all()
    return oof


def main():
    t0 = time.perf_counter()
    assert not (OUT / 'freeze.json').exists(), 'freeze exists; preserve it'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    sfeat = dev_structural()
    struct = [c for c in sfeat.columns if c not in ('pair_id', 'hand_id')]
    ffeat = dev_f2()
    f2cols = [c for c in ffeat.columns if c.startswith('f2_')]
    audit = {n: read_json(AUDIT_OUT / n) for n in
             ['audit-stage1-structural.json', 'audit-stage1-stability.json', 'audit-f2.json']
             if (AUDIT_OUT / n).exists()}
    prev65c = read_json(STRUCT_OUT / 'results.json')
    prev65i = read_json(R382_OUT / 'results.json')
    freeze = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected,
        'code_sha256': {p.name: sha(p) for p in sorted((SRC / 'evidence_aug').glob('step7*.py'))},
        'origin': {
            'source': 'steps 70 / 71 / 71b training-only audit (162 audit pools outside every held fold)',
            'audit_sha256': {n: sha(AUDIT_OUT / n) for n in audit},
            'audit_stage1_delta_seed_mean': audit.get('audit-stage1-stability.json', {})
            .get('paired', {}).get('chrono_struct_f2', {}).get('seed_mean_delta'),
            'audit_stage1_delta_ci95': audit.get('audit-stage1-stability.json', {})
            .get('paired', {}).get('chrono_struct_f2', {}).get('seed_avg_model', {}).get('ci95'),
            'audit_permutation_control': audit.get('audit-stage1-stability.json', {})
            .get('paired', {}).get('perm_struct_f2', {}).get('seed_mean_delta'),
            'audit_f2_in_stage2_was_null': audit.get('audit-f2.json', {}).get('verdict')},
        'previous_confirmations': {
            'step65c_allfam_top20_struct_vs_chrono1': prev65c['gates']['allfam_top20_struct']['pooled_delta'],
            'step65i_allfam_top20_r382_vs_struct': prev65i['gates']['allfam_top20_r382']['pooled_delta'],
            'held_truth_reads_so_far': 'step 11 (chronology), step 65d (struct stage 2), '
                                       'step 65j (r382 stage 2); this is the fourth read of the same '
                                       'three folds and the single read of the stage-1 study'},
        'stage1': {'old': '382 conditional features + chrono_frac (production PARAMS)',
                   'new': 'the same + ' + str(len(struct)) + ' structural columns (minus pot_bb, '
                          'already in the 382) + ' + str(len(f2cols)) + ' mechanism columns',
                   'oof': '5-way pool-hash OOF (audit-trunc-v1) on the training pools'},
        'stage2': {'model': 'LGBMClassifier ' + json.dumps(CLF) + ', 3 seeds averaged (deterministic '
                            'without bagging: one model)',
                   'rows': 'top-20 stage-1 candidates of the training pools, one model per TRUE family',
                   'struct_features': struct + CHRONO + S1,
                   'r382_features': 'struct + chrono3 + s1 + the fold ranker list minus family_* and '
                                    'minus names already present',
                   'unchanged_between_arms': True},
        'routing': 'held pairs scored under the FIXED predicted family of the fold (fold_roles); '
                   'predicted none keeps the arm stage-1 order',
        'variants': {'chrono1': 'old stage 1 only (reproduction control vs step 10)',
                     'allfam_top20_struct': 'reproduction control vs step 65c',
                     'allfam_top20_r382': 'reproduction control vs step 65i (ships today)',
                     's1f2': 'PRIMARY: new stage 1 only',
                     's1f2_top20_struct': 'new stage 1 + struct stage 2',
                     's1f2_top20_r382': 'new stage 1 + r382 stage 2'},
        'confirmation_folds': [0, 1, 2],
        'gate': {'applies_to': 's1f2 vs chrono1 (paired per held pair, folds 0, 1, 2)',
                 'pooled_delta_min': .010,
                 'pooled_pool_bootstrap_95_excludes_zero': True,
                 'per_fold_gain_nonnegative_in_at_least': 3,
                 'per_fold_true_family_loss_max': .020,
                 'pooled_true_family_loss_max': .010,
                 'pooled_pool_halves': 'both nonnegative',
                 'secondary': 's1f2_top20_r382 vs allfam_top20_r382 under the same numbers',
                 'decision': 'PASS on s1f2 -> refit the production stage-1 ranker with the two blocks, '
                             'rebuild the evaluation structural + f2 caches, and build ONE candidate on '
                             'the S order; the stage-2 partner is whichever of struct / r382 pairs best '
                             'with the new stage 1 on held, recorded as one selection among two with a '
                             'widened interval. FAIL -> keep the current chain and record the failure; '
                             'no further held read of these folds for this study.'},
        'prediction_if_pass': 'held pooled s1f2 - chrono1 in [+.010, +.045]; public E about +.010 to '
                              '+.030 over the stage-1 evidence, composite +.002 to +.006 on top of '
                              'submission_S_stage2_r382DTSP (.89336)',
        'metric': 'MAP@5 per held target pair, tie seed 931207',
        'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool'},
        'held_evidence_truth_read_before_freeze': False, 'folds': {}}
    write_json(OUT / 'freeze.json', freeze)

    for fold in [0, 1, 2]:
        t1 = time.perf_counter()
        data, features, pools, identity = load_fold(fold)
        struct_new = [c for c in struct if c not in set(features)]
        r382 = [f for f in features if not f.startswith('family_') and f in data.columns
                and f not in struct + CHRONO + S1]
        data = (add_chronology(data)
                .join(sfeat, on=['pair_id', 'hand_id'], how='left', validate='1:1')
                .join(ffeat.select(['pair_id', 'hand_id'] + f2cols), on=['pair_id', 'hand_id'],
                      how='left', validate='1:1'))
        assert data.select(pl.sum_horizontal(pl.col(struct + f2cols).null_count())).item() == 0
        data = data.with_columns(pl.col(f2cols).cast(pl.Float64).fill_nan(0.0))
        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools))
        evidence = training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence).sort('pair_id', 'hand_id')
        train = train.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64)
                                   .alias('split'))
        rf = fold_roles(fold)
        hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')

        designs = {'old': features + CHRONO1,
                   'new': features + CHRONO1 + struct_new + f2cols}
        s1_held, s1_train = {}, {}
        for dname, cols in designs.items():
            mdl = fit_ranker(train, cols)
            s1_held[dname] = score(mdl, family_columns(hf, 'predicted_behavior'), cols)
            s1_train[dname] = oof_stage1(train, cols)
        saved = pl.read_parquet(CHRONO_OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        assert (hf.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
                and np.array_equal(s1_held['old'], saved['score'].to_numpy())), \
            f'fold{fold}: chrono1 refit differs from the frozen scores'

        info = {'identity': identity, 'held_target_pairs': held['pair_id'].n_unique(),
                'held_rows': held.height, 'training_queries': train['pair_id'].n_unique(),
                'training_listed_hands': evidence.height, 'n_struct_new': len(struct_new),
                'n_f2': len(f2cols), 'n_r382': len(r382), 'reproduced_step10_chrono1': True,
                'stage2': {}}
        variants = {}
        for dname in designs:
            base = s1_held[dname]
            tr = with_rank(train.with_columns(pl.Series('s1', s1_train[dname])), 's1', 's1_rank')
            hh = with_rank(hf.with_columns(pl.Series('s1', base)), 's1', 's1_rank')
            in_top = (hh['s1_rank'] <= TOP).to_numpy()
            has_fam = hh['predicted_behavior'].is_in(FAMILIES).to_numpy()
            variants['chrono1' if dname == 'old' else 's1f2'] = base
            for sname, cols in (('struct', struct + CHRONO + S1),
                                ('r382', struct + CHRONO + S1 + r382)):
                pred = np.full(hh.height, np.nan)
                for fam in FAMILIES:
                    models, nq, npos = stage2_fit(tr, cols, fam, CLF)
                    mask = (hh['predicted_behavior'] == fam).to_numpy()
                    if mask.any():
                        pred[mask] = stage2_predict(models, hh.filter(pl.Series(mask)), cols)
                    g = np.mean([m.booster_.feature_importance('gain') for m in models], axis=0)
                    order = np.argsort(g)[::-1][:10]
                    info['stage2'][f'{dname}/{sname}/{fam}'] = {
                        'training_queries': nq, 'training_listed': npos, 'n_features': len(cols),
                        'top_gain': [(cols[i], round(float(g[i] / g.sum()), 3)) for i in order]}
                pred = np.where(in_top, pred + 10.0, base - 1e6)
                key = ('allfam_top20_' if dname == 'old' else 's1f2_top20_') + sname
                variants[key] = np.where(has_fam, pred, base)

        for prev_dir, name in ((STRUCT_OUT, 'allfam_top20_struct'), (R382_OUT, 'allfam_top20_r382')):
            p = prev_dir / 'scores' / f'fold{fold}-{name}.parquet'
            prev = pl.read_parquet(p)
            assert prev.select('pair_id', 'hand_id').equals(hf.select('pair_id', 'hand_id')) and \
                np.array_equal(prev['score'].to_numpy(), variants[name]), \
                f'fold{fold}: {name} not reproduced from {prev_dir.name}'
            info[f'reproduced_{name}'] = True

        for name, s in variants.items():
            assert np.isfinite(s).all(), name
            (hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s))
             .write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet'))
        rf.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        info['seconds'] = round(time.perf_counter() - t1, 1)
        freeze['folds'][str(fold)] = info
        log(json.dumps({'fold': fold, 'held_pairs': info['held_target_pairs'],
                        'n_struct_new': len(struct_new), 'n_f2': len(f2cols),
                        'reproduced': [k for k in info if k.startswith('reproduced_')],
                        'seconds': info['seconds']}))
    write_json(OUT / 'freeze.json', freeze)
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(),
              'freeze_sha256': sha(OUT / 'freeze.json'),
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').glob('*.parquet'))
                         + sorted((OUT / 'scores').glob('*.csv'))},
              'held_evidence_truth_read': False, 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'predictions-frozen.json', frozen)
    check_protected()
    log(json.dumps({'predictions_frozen': True, 'seconds': frozen['seconds']}))
    print('DONE')


if __name__ == '__main__':
    main()
