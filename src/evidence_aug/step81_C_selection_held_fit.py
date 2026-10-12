"""Step 81 [C] (author: claude).  Frozen held-fold predictions for the stage-2 SELECTION change.

What is being confirmed.  Stage 2 currently scores each of stage 1's 20 candidates on an ABSOLUTE
scale with a pointwise classifier.  Steps 78-80b add, for every structural column, its rank / z-score /
margin-to-best computed WITHIN the pair's own candidate set, because the task is "choose 5 of THESE
20".  Stage 1 is untouched, so this is a pure stage-2 change and the held stage-1 scores must
reproduce the frozen chrono1 numbers bit-for-bit -- that is the first control.

The arm is NOT chosen here.  It is read from step 80b's `carry` decision, which was pre-registered
before any fit: the highest MINIMUM paired delta across three independent partitions of the audit
surface, among arms positive under all three.  (Step 80 tried to do this with seed triplets and was
aborted -- these LightGBM params are deterministic, so `random_state` is inert and the test measured
nothing.  See step80b's header.)  If step 80b carried no arm, this script refuses and no held read is
taken.

Held variants written per fold (routing ALWAYS by the fold's FIXED PREDICTED family, never true
family -- `fold_roles`):

  chrono1            the frozen stage 1.  Reproduction control against step 10's saved scores.
  inc_struct         chrono1 + struct stage 2, all families.  Reproduction control vs step 65c.
  inc_r382           chrono1 + r382 stage 2, all families.  Reproduction control vs step 65i.
  inc_hybrid         struct on predicted coordinated_isolation, r382 elsewhere.  THE INCUMBENT --
                     this is the stage 2 that actually ships, and it is what the gate protects.
  new_<arm>          the carried arm, un-routed (step 78 showed the CI->struct routing now costs
                     .0585 because r382-with-relative-features beats struct on that family).

The three reproduction controls exist because the whole point of 18.17 was that a gate compared
against a mis-specified reference.  Reproducing inc_struct and inc_r382 from independent frozen files
proves this script rebuilt the same chain the record describes before it is allowed to claim a delta
against it.

PRE-REGISTERED GATE (in freeze.json, written before any fit; folds 0-2 are read exactly once, by
step 81b, and this is read number FIVE of these folds -- steps 11, 65d, 65j, 72b, then 81b):

  PRIMARY    new_<arm> vs inc_hybrid, paired per held pair, pooled over folds 0/1/2
             PASS if pooled delta > +.010 MAP@5 AND the 95 % pool-bootstrap lower bound > 0
             AND no true-family loss worse than -.020 AND per-fold delta non-negative in >= 2 of 3.
  STANDING CHECK (18.15 / 18.17): name the incumbent and verify a known-worse arm fails.
             inc_struct is known worse than inc_hybrid (it is the arm the hybrid routes AWAY from on
             two of three families).  Its delta vs inc_hybrid must be negative; if inc_struct passes
             this gate the gate is measuring a design gap and the result is void.
  SECONDARY  new_<arm> vs inc_r382, to separate the un-routing from the relative features.
  ON PASS    build ONE candidate on the M3 order with the new stage 2, pre-register the composite
             interval, hand it to Amin.  ON FAIL: record the failure; no further held read for this
             study.

Output: experiments/stage2_selection_C_20260912/held/ (freeze.json, predictions-frozen.json, scores/)
"""
import os
import sys

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl

from evidence_aug.fold import fold_roles, load_fold
from evidence_aug.frame import FAMILIES, attach_relevance, family_columns, score, training_relevance
from evidence_aug.paths import CV, SRC, check_protected, read_json, sha, write_json
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import OUT as CHRONO_OUT
from evidence_aug.step10_chronology_fit import add_chronology
from evidence_aug.step14_audit_truncation import CLF, SPLITS, TOP, split_of
from evidence_aug.step65c_family_stage2_fit import CHRONO, CHRONO1, S1
from evidence_aug.step65c_family_stage2_fit import OUT as STRUCT_OUT
from evidence_aug.step65c_family_stage2_fit import dev_structural, log, stage2_predict, with_rank
from evidence_aug.step65i_r382_stage2_fit import OUT as R382_OUT
from evidence_aug.step65i_r382_stage2_fit import stage2_fit
from evidence_aug.step78_C_stage2_selection import RANKER, SEEDS
from evidence_aug.step79_C_selection_v2 import BIG, BIG_RANK, N_WIDE, relativise

AUTHOR = 'claude'
SELDIR = CV / 'experiments' / 'stage2_selection_C_20260912'
OUT = SELDIR / 'held'
CI = 'coordinated_isolation'


def stage2_fit_rank(train, cols, fam, params):
    """lambdarank stage 2: one query per pair, candidates only, groups contiguous."""
    rows = (train.filter((pl.col('behavior_family') == fam) & (pl.col('s1_rank') <= TOP))
            .sort('pair_id', 'hand_id'))
    X = rows.select(cols).to_numpy().astype(np.float64)
    y = rows['relevance'].to_numpy().astype(int)
    _, grp = np.unique(rows['pair_id'].to_numpy(), return_counts=True)
    models = []
    for seed in SEEDS:
        r = lgb.LGBMRanker(**params, random_state=seed)
        r.fit(X, y, group=grp)
        models.append(r)
    return models, rows['pair_id'].n_unique(), int(y.sum())


def rank_predict(models, frame, cols):
    X = frame.select(cols).to_numpy().astype(np.float64)
    return np.mean([m.predict(X) for m in models], axis=0)


def top_cols_from_training(train, cols, n):
    """the n columns carrying most gain, fitted on this fold's TRAINING rows only (never held)."""
    rows = train.filter(pl.col('s1_rank') <= TOP)
    clf = lgb.LGBMClassifier(**CLF, random_state=SEEDS[0])
    clf.fit(rows.select(cols).to_numpy().astype(np.float64),
            rows['relevance'].to_numpy().astype(float))
    g = clf.booster_.feature_importance('gain')
    return [cols[i] for i in np.argsort(-g)[:n]]


def main():
    t0 = time.perf_counter()
    assert not (OUT / 'freeze.json').exists(), 'freeze exists; preserve it'
    stab = read_json(SELDIR / 'audit-C-resplit-stability.json')
    arm = stab['carry']['arm']
    assert arm, 'step 80b carried no arm; nothing to confirm and no held read is taken'
    tf, rc, cap, obj = arm.split('_')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()

    sfeat = dev_structural()
    struct = [c for c in sfeat.columns if c not in ('pair_id', 'hand_id')]

    freeze = {
        'author': AUTHOR, 'step': 81, 'created_at': datetime.now(timezone.utc).isoformat(),
        'protected': protected,
        'code_sha256': {p.name: sha(p) for p in
                        sorted((SRC / 'evidence_aug').glob('step7[6-9]_C_*.py'))
                        + sorted((SRC / 'evidence_aug').glob('step8*_C_*.py'))},
        'arm': {'tag': arm, 'transform': tf, 'relcols': rc, 'capacity': cap, 'objective': obj,
                'chosen_by': 'step 80b pre-registered carry rule (highest minimum paired delta across '
                             'three independent partitions of the audit surface, positive under all '
                             'three).  Step 80 was aborted: seed triplets are a null perturbation '
                             'because these LightGBM params are deterministic.',
                'audit_min_delta_across_partitions': stab['carry']['its_min_delta_across_partitions'],
                'audit_deltas_by_salt': stab['carry']['its_deltas_by_salt'],
                'audit_v1_map5': stab['carry']['its_v1_map5'],
                'audit_v1_ci95': stab['carry']['its_v1_ci95']},
        'origin': {'step76': 'SELECTION carries +.2649 of the .3841 total MAP@5 headroom; ORDER +.0673, '
                             'RETRIEVAL +.0519. stage 1 top-20 holds 4.642 of 5 listed hands, stage 2 '
                             'selects 3.333 of them',
                   'step78': 'within-pair relative features beat plain in all four cells; the listwise '
                             'objective alone is null; the CI->struct routing now costs .0585',
                   'step79': 'two arms cleared the audit gate; the best was an isolated grid peak',
                   'step80b': 'stability under three independent partitions of the audit surface; step 80 '
                              'aborted because random_state is inert for these params'},
        'stage1': 'UNCHANGED. chrono1 refit must reproduce the frozen step-10 fold scores exactly.',
        'stage2_variants': {
            'inc_struct': 'reproduction control vs step 65c',
            'inc_r382': 'reproduction control vs step 65i',
            'inc_hybrid': 'THE INCUMBENT the gate protects: struct on predicted CI, r382 elsewhere',
            f'new_{arm}': 'the carried arm, un-routed'},
        'routing': 'the fold FIXED PREDICTED family (fold_roles); true family is never used to route',
        'gate': {
            'primary': f'new_{arm} vs inc_hybrid, paired per held pair, pooled over folds 0/1/2',
            'pass_if': {'pooled_delta_gt': .010, 'ci95_lower_gt': 0.0,
                        'true_family_loss_max': .020, 'per_fold_nonnegative_in_at_least': 2},
            'standing_check_18_15_18_17': 'inc_struct is known worse than inc_hybrid and MUST fail this '
                                          'gate; if it passes, the gate measures a design gap and the '
                                          'result is void',
            'secondary': f'new_{arm} vs inc_r382, separating the un-routing from the relative features',
            'on_pass': 'build ONE candidate on the M3 order with the new stage 2 and pre-register its '
                       'composite interval',
            'on_fail': 'record the failure; no further held read of these folds for this study'},
        'confirmation_folds': [0, 1, 2],
        'held_read_number': 5, 'previous_reads': ['step 11', 'step 65d', 'step 65j', 'step 72b'],
        'metric': 'MAP@5 per held target pair, tie seed 931207',
        'bootstrap': {'repetitions': 2000, 'seed': 20319, 'unit': 'whole held pool'},
        'held_evidence_truth_read_before_freeze': False, 'folds': {}}

    for fold in [0, 1, 2]:
        t1 = time.perf_counter()
        data, features, pools, identity = load_fold(fold)
        r382 = [f for f in features if not f.startswith('family_') and f in data.columns
                and f not in struct + CHRONO + S1]
        data = add_chronology(data).join(sfeat, on=['pair_id', 'hand_id'], how='left', validate='1:1')
        assert data.select(pl.sum_horizontal(pl.col(struct).null_count())).item() == 0

        held = data.filter(pl.col('table_id').is_in(pools)).sort('pair_id', 'hand_id')
        train = data.filter(~pl.col('table_id').is_in(pools))
        evidence = training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id']) & set(held['pair_id'])
        train = attach_relevance(train, evidence).sort('pair_id', 'hand_id')
        rf = fold_roles(fold)
        hf = held.join(rf, on='pair_id', validate='m:1').sort('pair_id', 'hand_id')

        cols1 = features + CHRONO1
        mdl = fit_ranker(train, cols1)
        s1_held = score(mdl, family_columns(hf, 'predicted_behavior'), cols1)
        saved = pl.read_parquet(CHRONO_OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        assert (hf.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
                and np.array_equal(s1_held, saved['score'].to_numpy())), \
            f'fold{fold}: chrono1 refit differs from the frozen scores'
        # stage-1 OOF on the training pools, exactly as 65c/65i: a single model fitted on all the
        # training rows would be in-sample for them and would not reproduce the frozen stage 2.
        train = train.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64)
                                   .alias('split'))
        oof = np.full(train.height, np.nan)
        for s in range(SPLITS):
            idx = np.where(train['split'].to_numpy() == s)[0]
            m = fit_ranker(train.filter(pl.col('split') != s), cols1)
            oof[idx] = score(m, family_columns(train[idx], 'behavior_family'), cols1)
        assert np.isfinite(oof).all()

        tr = with_rank(train.with_columns(pl.Series('s1', oof)), 's1', 's1_rank')
        hh = with_rank(hf.with_columns(pl.Series('s1', s1_held)), 's1', 's1_rank')
        in_top = (hh['s1_rank'] <= TOP).to_numpy()
        has_fam = hh['predicted_behavior'].is_in(FAMILIES).to_numpy()
        info = {'identity': identity, 'held_target_pairs': held['pair_id'].n_unique(),
                'held_rows': held.height, 'training_queries': train['pair_id'].n_unique(),
                'training_listed_hands': evidence.height, 'n_r382': len(r382),
                'reproduced_step10_chrono1': True, 'stage2': {}}
        variants = {'chrono1': s1_held}

        # ---- the two incumbents, exactly the 65c / 65i construction --------------------------
        base_designs = {'inc_struct': struct + CHRONO + S1, 'inc_r382': struct + CHRONO + S1 + r382}
        for name, cols in base_designs.items():
            pred = np.full(hh.height, np.nan)
            for fam in FAMILIES:
                models, nq, npos = stage2_fit(tr, cols, fam, CLF)
                mask = (hh['predicted_behavior'] == fam).to_numpy()
                if mask.any():
                    pred[mask] = stage2_predict(models, hh.filter(pl.Series(mask)), cols)
                info['stage2'][f'{name}/{fam}'] = {'training_queries': nq, 'training_listed': npos,
                                                   'n_features': len(cols)}
            variants[name] = np.where(has_fam, np.where(in_top, pred + 10.0, s1_held - 1e6), s1_held)

        for prev_dir, name, key in ((STRUCT_OUT, 'allfam_top20_struct', 'inc_struct'),
                                    (R382_OUT, 'allfam_top20_r382', 'inc_r382')):
            prev = pl.read_parquet(prev_dir / 'scores' / f'fold{fold}-{name}.parquet')
            assert prev.select('pair_id', 'hand_id').equals(hf.select('pair_id', 'hand_id')) and \
                np.array_equal(prev['score'].to_numpy(), variants[key]), \
                f'fold{fold}: {key} not reproduced from {prev_dir.name}'
            info[f'reproduced_{key}'] = True

        # the shipping hybrid, routed on the PREDICTED family
        isci = (hh['predicted_behavior'] == CI).to_numpy()
        variants['inc_hybrid'] = np.where(isci, variants['inc_struct'], variants['inc_r382'])

        # ---- the carried arm --------------------------------------------------------------
        relbase = struct + CHRONO
        if rc == 'wide':
            relbase = list(dict.fromkeys(relbase + top_cols_from_training(
                tr, struct + CHRONO + S1 + r382, N_WIDE)))
        trr = relativise(tr, relbase, TOP, margin=(tf == 'rzm'), rankcol='s1_rank')
        hhr = relativise(hh, relbase, TOP, margin=(tf == 'rzm'), rankcol='s1_rank')
        extra = [c for c in trr.columns if c.endswith(('__wr', '__wz', '__wm'))]
        cols = struct + CHRONO + S1 + r382 + extra
        params = ({'clf': CLF, 'rank': RANKER} if cap == 'small' else
                  {'clf': BIG, 'rank': BIG_RANK})[obj]
        fitter = stage2_fit if obj == 'clf' else stage2_fit_rank
        predictor = stage2_predict if obj == 'clf' else rank_predict
        pred = np.full(hhr.height, np.nan)
        for fam in FAMILIES:
            models, nq, npos = fitter(trr, cols, fam, params)
            mask = (hhr['predicted_behavior'] == fam).to_numpy()
            if mask.any():
                pred[mask] = predictor(models, hhr.filter(pl.Series(mask)), cols)
            info['stage2'][f'new/{fam}'] = {'training_queries': nq, 'training_listed': npos,
                                            'n_features': len(cols), 'n_relative': len(extra)}
        variants[f'new_{arm}'] = np.where(has_fam,
                                          np.where(in_top, pred + 10.0, s1_held - 1e6), s1_held)
        info['n_relcols'] = len(relbase)

        for name, s in variants.items():
            assert np.isfinite(s).all(), name
            (hf.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s))
             .write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet'))
        rf.write_csv(OUT / 'scores' / f'fold{fold}-roles.csv')
        info['seconds'] = round(time.perf_counter() - t1)
        freeze['folds'][str(fold)] = info
        log(f'fold{fold}', json.dumps({'pairs': info['held_target_pairs'], 'sec': info['seconds'],
                                       'nrel': len(extra)}))

    write_json(OUT / 'freeze.json', freeze)
    write_json(OUT / 'predictions-frozen.json',
               {'author': AUTHOR, 'freeze_sha256': sha(OUT / 'freeze.json'),
                'frozen_at': datetime.now(timezone.utc).isoformat(),
                'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').glob('*.parquet'))}})
    check_protected()
    print(f'\n  frozen: arm={arm}, {len(freeze["folds"])} folds, '
          f'{round(time.perf_counter() - t0)} s')
    print(f'  next: step81b evaluates ONCE against held truth (read 5 of folds 0-2)')


if __name__ == '__main__':
    main()
