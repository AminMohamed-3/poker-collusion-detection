"""STEP277-A: the frozen 662 mechanical columns for the SOFT-PLAY candidate rows of the evaluation phase,
plus the induced citation rule's score on them.

Why this exists (OBSERVATIONS 19.237). An induced per-hand citation rule, fitted on the SP development pairs that
lie OUTSIDE the 162-query audit and blended into production's stage-2 p for soft play only at w = .30, reads
+.024255 on soft play (+.008148 whole audit, P .9285, 40 lists changed, 17 with a different set); an honest
split-half over 28 (scope, weight) cells pays for the selection and still returns +.003039 with 307/400 positive;
and the controls all fail in the right direction -- within-pair label permutation -.054178, training on the other
families -.023481, five hyperparameter settings +.0185 to +.0319 all positive, ten 70% training subsets 10/10
positive with minimum +.008830. STEP257's own mechplus_p_SP does NOT reproduce this (-.007413 at the same w, with
its permutation control at -.011121 tracking it), so the shipped object has to be this rule, not that score.

The features cannot be taken from STEP254: that build is development-only by construction. They come instead from
step257_mech_eval.build_chunk, which is STEP254's computation imported unchanged with the phase and the key set
parameterised, and from the already-built evaluation chronology. The 662 columns and their order are frozen from
the development file, never re-pruned -- kept_columns() reads them straight off step254_features.parquet.

GATE P (parity, runs first and must pass before any evaluation row is written): build_chunk is run on a sample of
DEVELOPMENT SP candidate rows through exactly this code path, and every one of the 662 columns must reproduce
step254_features.parquet to within float32. If my invocation were wrong in any way -- role orientation, key
columns, chronology join -- this gate fails and nothing downstream is trusted.

Outputs (into experiments/step277_generator_rule_20260919/):
    sp_eval_features.parquet   pair_id, hand_id, the frozen 662, for the SP scoped candidate rows
    sp_rule_eval.parquet       pair_id, hand_id, p_rule
    sp_rule_models/            the fitted boosters, one per seed
"""
from __future__ import annotations

import os

for _k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '4'

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'Claude Version' / 'src' / 'evidence_aug'))
import step257_mech_eval as M                                     # noqa: E402

DATA = ROOT / 'data'
CV = ROOT / 'Claude Version'
OUT = HERE / 'portable_step277_outputs'
S254 = CV / 'experiments/step254_mechanical_tensor_20260917'
S251 = CV / 'outputs_step251_20260917'
PAIRBASE = CV / 'outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv'
SCOPE = ROOT / 'reports/replay-v13/selected-pairs.parquet'
CHRONO_EVAL = HERE / 'recovered_chronology' / 'evaluation'
SP = 'soft_play'
KEYS = ['pair_id', 'hand_id']
SEEDS = (11, 23, 37, 51, 67, 79, 93, 107)
HP = dict(objective='binary', n_estimators=400, num_leaves=15, max_depth=4, min_child_samples=30,
          learning_rate=.05, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_jobs=4, verbosity=-1)
CHUNK = 6000


def log(*a):
    print(*a, flush=True)


def audit_pairs() -> set[str]:
    """the 162 audit pairs, which the rule must never be trained on."""
    sys.path.insert(0, str(CV / 'src')); sys.path.insert(0, str(ROOT / 'Codex Resume/src'))
    from evidence_aug import step165_C_e2e_audit as S165
    return set(S165.load()['d']['pair_id'].unique().to_list())


def gate_parity(PCT169, EQ169, allc, inhand, chrono_cols) -> None:
    """GATE P: this exact code path must reproduce the development tensor on development SP rows."""
    ev = pl.read_csv(DATA / 'development_evidence.csv')
    spp = ev.filter(pl.col('behavior_family') == SP)['pair_id'].unique().to_list()[:12]
    ref = pl.read_parquet(S254 / 'step254_features.parquet').filter(pl.col('pair_id').is_in(spp))
    chd = sorted(glob.glob(str(HERE / 'recovered_chronology' / 'development' / 'chrono_b*.parquet')))
    chrono = pl.read_parquet(chd).filter(pl.col('pair_id').is_in(spp))
    lab = pl.read_csv(DATA / 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
    keys = (ref.select(KEYS).join(lab, on='pair_id')
            .join(chrono.select(KEYS + ['p1_is_A']), on=KEYS))
    assert keys.height == ref.height, (keys.height, ref.height)
    got = M.build_chunk(keys, 'development', PCT169, EQ169)
    got = got.join(chrono.drop('p1_is_A'), on=KEYS, how='left').select(KEYS + allc)
    got = got.with_columns(pl.col(allc).cast(pl.Float32)).sort(KEYS)
    exp = ref.select(KEYS + allc).sort(KEYS)
    assert got.height == exp.height, (got.height, exp.height)
    A = got.select(allc).to_numpy(); B = exp.select(allc).to_numpy()
    bad = ~np.isclose(A, B, rtol=0, atol=1e-6, equal_nan=True)
    if bad.any():
        j = int(np.argmax(bad.sum(0)))
        raise AssertionError('GATE P FAILED: %d of %d cells differ, worst column %s (%d rows)'
                             % (int(bad.sum()), bad.size, allc[j], int(bad[:, j].sum())))
    log('GATE P PASSED: %d development SP rows x %d columns reproduce step254_features.parquet exactly'
        % (exp.height, len(allc)))


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    allc, inhand, chrono_cols = M.kept_columns()
    log('frozen columns: %d total, %d in-hand, %d chronology' % (len(allc), len(inhand), len(chrono_cols)))
    PCT169, EQ169 = M.load_pf169()
    gate_parity(PCT169, EQ169, allc, inhand, chrono_cols)
    # ---- which evaluation rows: the stage-1 candidates of the SP scoped pairs
    scope = set(pl.read_parquet(SCOPE)['pair_id'].unique().to_list())
    sub = pl.read_csv(PAIRBASE, columns=['pair_id', 'predicted_behavior'], infer_schema=False)
    sp_pairs = set(sub.filter((pl.col('predicted_behavior') == SP)
                              & pl.col('pair_id').is_in(list(scope)))['pair_id'].to_list())
    log('SP scoped pairs: %d' % len(sp_pairs))
    cand = []
    for i in range(20):
        p = S251 / 'work' / 'INC' / 'batches' / ('batch_%03d-lp-scores.parquet' % i)
        cand.append(pl.read_parquet(p, columns=KEYS).filter(pl.col('pair_id').is_in(list(sp_pairs))))
    cand = pl.concat(cand).unique(subset=KEYS)
    log('SP candidate rows: %d  (%.1f per pair)' % (cand.height, cand.height / max(len(sp_pairs), 1)))
    chrono = (pl.scan_parquet(sorted(glob.glob(str(Path(CHRONO_EVAL) / 'chrono_b*.parquet'))))
              .filter(pl.col('pair_id').is_in(list(sp_pairs))).collect())
    log('chronology rows for those pairs: %d' % chrono.height)
    ep = pl.read_csv(DATA / 'evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2'])
    keys = cand.join(ep, on='pair_id').join(chrono.select(KEYS + ['p1_is_A']), on=KEYS)
    assert keys.height == cand.height, 'candidate rows lost in the chronology join: %d vs %d' % (keys.height, cand.height)
    # ---- build in chunks over hands
    hl = keys.select('hand_id').unique()['hand_id'].to_list()
    bounds = list(range(0, len(hl), CHUNK)) + [len(hl)]
    parts = []
    for c in range(len(bounds) - 1):
        part = pl.DataFrame({'hand_id': hl[bounds[c]:bounds[c + 1]]})
        kk = keys.join(part, on='hand_id')
        parts.append(M.build_chunk(kk, 'evaluation', PCT169, EQ169))
        log('  chunk %d/%d  hands %d  rows %d  %.0fs'
            % (c + 1, len(bounds) - 1, part.height, parts[-1].height, time.perf_counter() - t0))
    feat = pl.concat(parts).join(chrono.drop('p1_is_A'), on=KEYS, how='left').select(KEYS + allc)
    feat = feat.with_columns(pl.col(allc).cast(pl.Float32)).sort(KEYS)
    assert feat.height == cand.height, (feat.height, cand.height)
    assert feat.select(pl.col(allc).is_null().any()).to_numpy().sum() == 0, 'nulls in the SP evaluation block'
    feat.write_parquet(OUT / 'sp_eval_features.parquet')
    log('wrote sp_eval_features.parquet %s  %.0fs' % (feat.shape, time.perf_counter() - t0))
    # ---- train the VALIDATED rule: SP development pairs OUTSIDE the audit, full candidate universe
    import lightgbm as lgb
    ap = audit_pairs()
    ev = pl.read_csv(DATA / 'development_evidence.csv')
    fam = dict(ev.select('pair_id', 'behavior_family').unique().iter_rows())
    trp = [p for p, f in fam.items() if f == SP and p not in ap]
    log('training pairs: %d SP development pairs outside the audit' % len(trp))
    tr = (pl.scan_parquet(S254 / 'step254_features.parquet')
          .filter(pl.col('pair_id').is_in(trp)).collect()
          .join(ev.select(KEYS + ['evidence_rank']), on=KEYS, how='left'))
    X = np.nan_to_num(tr.select(allc).to_numpy().astype(np.float32), nan=0., posinf=0., neginf=0.)
    y = np.array([v is not None for v in tr['evidence_rank'].to_list()]).astype(int)
    log('training rows %d  positives %d' % (len(y), int(y.sum())))
    Xe = np.nan_to_num(feat.select(allc).to_numpy().astype(np.float32), nan=0., posinf=0., neginf=0.)
    mdir = OUT / 'sp_rule_models'; mdir.mkdir(exist_ok=True)
    acc = np.zeros(len(Xe))
    for sd in SEEDS:
        m = lgb.LGBMClassifier(random_state=sd, **HP).fit(X, y)
        m.booster_.save_model(str(mdir / ('sp_rule_seed%d.txt' % sd)))
        acc += m.predict_proba(Xe)[:, 1]
    p_rule = acc / len(SEEDS)
    out = feat.select(KEYS).with_columns(pl.Series('p_rule', p_rule))
    out.write_parquet(OUT / 'sp_rule_eval.parquet')
    log('wrote sp_rule_eval.parquet  rows %d  p_rule mean %.5f  p10 %.5f  p90 %.5f'
        % (out.height, p_rule.mean(), np.percentile(p_rule, 10), np.percentile(p_rule, 90)))
    json.dump({'step': 'STEP277-A', 'sp_scoped_pairs': len(sp_pairs), 'candidate_rows': cand.height,
               'frozen_columns': len(allc), 'training_pairs': len(trp), 'training_rows': int(len(y)),
               'training_positives': int(y.sum()), 'seeds': list(SEEDS), 'hp': HP,
               'gate_P': 'PASSED', 'seconds': round(time.perf_counter() - t0, 1)},
              open(OUT / 'step277_A_log.json', 'w'), indent=2)
    log('DONE %.0fs' % (time.perf_counter() - t0))


if __name__ == '__main__':
    main()
