"""Step 166 (Claude): re-open R-a of VALIDATION_BOOK_REVIEW_20260915.md under the eased audit rule ER1. The within-top-20 relative
encoding (average rank and z-score over the pair's stage-1 top-20 candidates) of the 70 structural columns and chrono_rank / chrono_frac
/ n_hands (steps 78-81, `rz_base_small_clf`; held +.0082 vs the then-shipping hybrid, FAIL on the old gate) is added to every family's
S* stage 2 on the training-only 162-query audit, with S*'s exact per-family GroupKFold(5) folds and params. Arm S*rz vs S*, both decoded
with the production typed decoder (T1 off for coordinated isolation). No held folds, no evaluation pairs. Never uploads.
Usage: run REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '4'
import sys, json, time, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/rz_on_sstar_C_20260915'
REG = EXP / 'registration166.json'
K = S165.K
FAMS, CI = S165.FAMS, S165.CI
CHRONO = ['chrono_rank', 'chrono_frac', 'n_hands']


def relativise(Q, cols):
    """step78 relativise restricted to the top-20 frame: every row is a candidate."""
    exprs = []
    for c in cols:
        exprs.append(pl.col(c).rank('average').over('pair_id').fill_null(0.0).alias(f'{c}__wr'))
        exprs.append(((pl.col(c) - pl.col(c).mean().over('pair_id')) / (pl.col(c).std().over('pair_id') + 1e-9)).fill_null(0.0).alias(f'{c}__wz'))
    return Q.with_columns(exprs)


def fit(D, Q, extra):
    out = np.full(len(Q), np.nan)
    for f in FAMS:
        ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy()); q = Q[ix]
        y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
        cols = D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] + extra
        z = q.select(cols).to_numpy().astype(float)
        for tr, te in GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g):
            out[ix[te]] = lgb.LGBMClassifier(**D['ac'].CLF, random_state=11).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
    assert np.isfinite(out).all()
    return out


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['runner_sha256']
    assert not (EXP / 'results166.json').exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    D = S165.load()
    Q = D['Q']
    rel = D['struct'] + CHRONO
    assert len(D['struct']) == 70
    Qr = relativise(Q, rel)
    extra = [c for c in Qr.columns if c.endswith(('__wr', '__wz'))]
    assert len(extra) == 146
    base = fit(D, Q, [])
    full = D['SS'].copy(); full[D['QIX']] = base + 10
    rep = float(np.abs(full - D['SS']).max())
    print('S* replay', rep, flush=True)
    assert rep == 0.0
    arm = fit(D, Qr, extra)
    A = D['SS'].copy(); A[D['QIX']] = arm + 10
    ac, dec, d = D['ac'], D['dec'], D['d']
    R = lambda x: round(float(x), 6)

    def cmp(A0, B0):
        c = ac.evaluate(d, {'incumbent': A0, 'arm': B0})['comparisons']['arm']
        return {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                'fam': {kk[:2]: R(x) for kk, x in c['by_family'].items()}, 'halves': {kk: R(x) for kk, x in c['halves'].items()}, 'ac_gate': bool(c['gate_pass'])}
    yv = d['relevance'].to_numpy()[D['QIX']]
    ll = lambda s: R(-np.mean(yv * np.log(np.clip(s[D['QIX']] - 10, 1e-6, 1 - 1e-6)) + (1 - yv) * np.log(1 - np.clip(s[D['QIX']] - 10, 1e-6, 1 - 1e-6))))
    DS, DA = dec(D['SS']), dec(A)
    comp = {'decoded': cmp(DS, DA), 'raw': cmp(D['SS'], A)}
    r = reg['rule']
    c = comp['decoded']
    passes = c['mean'] > 0 and c['ci95'][0] > r['ci_lower_floor'] and min(c['fam'].values()) >= r['family_floor']
    EXP.mkdir(parents=True, exist_ok=True)
    d.select(K).with_columns(pl.Series('S_star', D['SS']), pl.Series('S_star_rz', A), pl.Series('dec_S_star', DS), pl.Series('dec_S_star_rz', DA)).write_parquet(EXP / 'audit_scores166.parquet')
    assert check_protected() == prot
    res = {'registration_sha256': reg_sha, 'S_star_replay_max_abs_diff': rep, 'n_relative_columns': len(extra),
           'decoded_map5': {'S_star': R(ac.evaluate(d, {'incumbent': DS})['sets']['incumbent']['map5']), 'S_star_rz': R(ac.evaluate(d, {'incumbent': DA})['sets']['incumbent']['map5'])},
           'logloss_top20': {'S_star': ll(D['SS']), 'S_star_rz': ll(A)}, 'comparisons': comp, 'ER1_pass': bool(passes),
           'scores_sha256': S165.sha(EXP / 'audit_scores166.parquet'), 'wall_seconds': round(time.perf_counter() - t0, 1)}
    (EXP / 'results166.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
