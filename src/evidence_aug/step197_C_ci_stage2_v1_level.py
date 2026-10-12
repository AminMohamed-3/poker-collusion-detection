"""STEP197: STEP194c taken to the v1 (41-view) level, which is what production actually ships.

194c added the four causal fold columns to coordinated_isolation's stage-2 column list ONLY and read
+0.008621 decoded at u=0, halves +0.010690 / +0.006222, CI +0.032481, DT and SP exactly 0.0. Its own
results file named this run as the precondition for a build: u=0 is a single scoring of the pool,
v1 is the 41-view average, and a gain that only exists at u=0 is a gain production never collects.

Nothing changes here except the scoring level. Same columns, same arms, same untouched network. Each
fold model is averaged over the original pool plus the same 40 label-free views as v1 (20 thinned,
seeds 1241/2241; 20 padded, 1242/2242; step-124 recipe), reusing step165b's own build_views so the
view construction is provably identical -- the S* replay gate below is what proves it.

The four columns are properties of one hand's own action sequence, so thinning or padding the
candidate pool cannot change them. They ride into every view unchanged via the join in finish(),
while chrono_rank / chrono_frac / n_hands / oof_rank / context_* are recomputed per view as always.

Gates: S* v1 replay EXACTLY 0.0 (no network output, no new column -- exactness is legitimate and
proves the views are step124's); DT and SP deltas EXACTLY 0.0 (within-run construction); control
reproduces STEP165b's dec_v1LP 0.692274 to a tolerance of 0.005 registered in advance, because
OBS 19.145 established the stage-1 network does NOT reproduce bit-exactly across days and STEP192
died on that gate twice.

Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step197_C_ci_stage2_v1_level.py REGSHA
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step165b_C_lp_v1_level import build_views

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step197_20260916'
REG = OUT / 'registration197.json'
FEAT = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
NEW = ['pf_others_fold_after_pair_agg', 'pf_others_fold_before_pair_agg',
       'pf_pair_aggresses_preflop', 'pf_others_call_after_pair_agg']
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
STEP165B_DEC_V1LP = 0.692274
TOL = 0.005


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not (OUT / 'results197.json').exists(), 'preserve a completed read'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    # join BEFORE build_views: finish() carries every non-recomputed Q column into each view
    feat = pl.read_parquet(FEAT).select(K + NEW)
    q0 = D['Q'].select(K)
    D['Q'] = D['Q'].join(feat, on=K, how='left', validate='1:1', maintain_order='left')
    assert D['Q'].select(K).equals(q0), 'join reordered Q'
    assert D['Q'].select(NEW).null_count().to_numpy().sum() == 0, 'unmatched audit rows'

    X, C = S165.matrices(D)
    assert X.shape[1] == 453, f'network input changed: {X.shape}'
    d, Q = D['d'], D['Q']
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'],
                 d['relevance'].to_numpy(), pairs, tables)
    jobs = []
    for k, (tr, te) in enumerate(outer):
        for s in S165.SEEDS_INNER_OUTER:
            jobs.append((('outer', k), 'joint', s, rowmask(tr), rowmask(te)))
        for j, (itr, ite) in enumerate(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
            for s in S165.SEEDS_INNER:
                jobs.append((('inner', k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    print(f'X {X.shape}  {len(jobs)} network fits (shared by both arms)', flush=True)
    with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    print(f'network refits done {round(time.perf_counter()-t0)}s', flush=True)

    nq = len(Q)
    AF_te, AF_in = {k: [] for k in range(5)}, {k: {} for k in range(5)}
    for jtag, m, s, rows, tq, p, S, trace, sec in out:
        if jtag[0] == 'outer':
            AF_te[jtag[1]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
        else:
            AF_in[jtag[1]].setdefault(jtag[2], []).append((tq, S165.aligned_features(D, rows, tq, S, p)))
    qpairs = Q['pair_id'].to_numpy()
    folds_q, EA, EB = [], {}, np.full((nq, len(S165.NN_COLS)), np.nan)
    for k, (tr, te) in enumerate(outer):
        trq = np.flatnonzero(np.isin(qpairs, upairs[tr]))
        teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
        ea = np.full((nq, len(S165.NN_COLS)), np.nan)
        for j, lst in AF_in[k].items():
            ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
        eb = np.full((nq, len(S165.NN_COLS)), np.nan)
        eb[AF_te[k][0][0]] = np.mean([a for _, a in AF_te[k]], axis=0)
        assert np.isfinite(ea[trq]).all() and np.isfinite(eb[teq]).all()
        folds_q.append((trq, teq))
        EA[k] = ea[trq]
        EB[teq] = eb[teq]
    assert np.isfinite(EB).all()

    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    ci_cols = dict(base)
    ci_cols[CI] = base[CI] + NEW
    ARMS = {'control': base, 'CI_STAGE2_ONLY': ci_cols}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}

    # stage-2 fold models, one set per arm, both carrying the NN features (this is the LP arm)
    MOD = {}
    for arm, colsf in ARMS.items():
        for f in FAMS:
            z = Q.select(colsf[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds_q):
                a = tr[isf[f][tr]]
                MOD[(arm, f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                    np.column_stack([z[a], EA[k][isf[f][tr]]]), y[a])

    def score_arm(v, arm):
        colsf = ARMS[arm]
        p = D['SS'].copy()
        for f in FAMS:
            z = v.select(colsf[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds_q):
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = MOD[(arm, f, k)].predict_proba(
                        np.column_stack([z[b], EB[b]]))[:, 1] + 10
        return p

    # S* per-family fold models on the step-124 FOLDS, for the view-construction gate only.
    # These never see EA/EB or the new columns, which is exactly why their replay may be exact.
    MS, FOLDS = {}, {}
    for f in FAMS:
        ix = np.flatnonzero(isf[f])
        q = Q[ix]
        yy = q['relevance'].to_numpy().astype(int)
        FOLDS[f] = (ix, list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), yy,
                                                          q['table_id'].to_numpy())))
        z = q.select(base[f]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(FOLDS[f][1]):
            MS[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[tr], yy[tr])

    def score_sstar(v):
        p = D['SS'].copy()
        for f in FAMS:
            ix, folds = FOLDS[f]
            zv = v[ix].select(base[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds):
                p[D['QIX'][ix[te]]] = MS[(f, k)].predict_proba(zv[te])[:, 1] + 10
        return p

    views, pads = build_views(D)
    A0 = {arm: score_arm(Q, arm) for arm in ARMS}
    S0 = score_sstar(Q)
    acc = {n: np.zeros(len(d)) for n in list(ARMS) + ['S']}
    nv = 0
    for maker, seeds in ((views, (1241, 2241)), (pads, (1242, 2242))):
        for s in seeds:
            for v in maker(.5, 10, s):
                for arm in ARMS:
                    acc[arm] += score_arm(v, arm)
                acc['S'] += score_sstar(v)
                nv += 1
            print(f'views {nv}  {round(time.perf_counter()-t0)}s', flush=True)
    assert nv == 40, nv
    V1 = {arm: (acc[arm] + A0[arm]) / (nv + 1) for arm in ARMS}
    V1S = (acc['S'] + S0) / (nv + 1)
    gate_v1 = float(np.abs(V1S - D['V1']).max())
    print(f'S* v1 replay max abs diff {gate_v1}', flush=True)

    dec = D['dec']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    M = {}
    for arm in ARMS:
        _, m = summarize(key.with_columns(pl.Series('score', dec(V1[arm]))), truths, fams)
        M[arm] = np.array([m[q] for q in qids])
    delta = M['CI_STAGE2_ONLY'] - M['control']
    hv = np.array([half(ptabq[q]) for q in qids])
    R = lambda x: round(float(x), 6)

    res = {'registration_sha256': reg_sha, 'level': 'v1, 41 views (original pool + 20 thinned + 20 padded)',
           'is_the_registered_v1_confirmation_of_194c': True, 'n_queries': len(qids), 'n_views': nv + 1,
           'map5_control': R(M['control'].mean()),
           'map5_CI_stage2_only': R(M['CI_STAGE2_ONLY'].mean()),
           'decoded_delta': R(delta.mean()),
           'half0_delta': R(delta[hv == 0].mean()), 'half1_delta': R(delta[hv == 1].mean()),
           'u0_reference_194c': {'decoded_delta': 0.008621, 'halves': [0.01069, 0.006222],
                                 'CI': 0.032481},
           'per_family': {}, 'gates': {}, 'seconds': round(time.perf_counter() - t0, 1)}
    for f in FAMS:
        sel = np.array([fams[q] == f for q in qids])
        fd = delta[sel]
        res['per_family'][f] = {
            'n': int(sel.sum()), 'delta': R(fd.mean()),
            'half0': R(fd[hv[sel] == 0].mean()) if (hv[sel] == 0).any() else None,
            'half1': R(fd[hv[sel] == 1].mean()) if (hv[sel] == 1).any() else None}
    untouched = {f: float(np.abs(delta[[fams[q] == f for q in qids]]).max()) for f in FAMS if f != CI}
    res['gates']['S_star_v1_replay_exact'] = {'max_abs': gate_v1, 'passed': gate_v1 == 0.0}
    res['gates']['DT_SP_exactly_zero'] = {'max_abs': {k: R(v) for k, v in untouched.items()},
                                          'passed': all(v == 0.0 for v in untouched.values())}
    cih = res['per_family'][CI]
    res['gates']['CI_positive_on_both_halves'] = bool(
        cih['half0'] is not None and cih['half1'] is not None and cih['half0'] > 0 and cih['half1'] > 0)
    res['gates']['control_reproduces_165b_v1LP'] = {
        'step165b_dec_v1LP': STEP165B_DEC_V1LP, 'here': R(M['control'].mean()), 'tolerance': TOL,
        'passed': bool(abs(M['control'].mean() - STEP165B_DEC_V1LP) <= TOL),
        'note': 'tolerance registered in advance; bit-identity to a stored network artefact is never gated (OBS 19.145)'}
    res['bar'] = {'corrected_build_bar_decoded': 0.0073, 'clears_bar': bool(delta.mean() >= 0.0073),
                  'both_halves_positive': bool(delta[hv == 0].mean() > 0 and delta[hv == 1].mean() > 0)}
    g = res['gates']
    res['verdict'] = ('VOID' if not (g['S_star_v1_replay_exact']['passed'] and g['DT_SP_exactly_zero']['passed']
                                     and g['control_reproduces_165b_v1LP']['passed'])
                      else 'BUILD' if (res['bar']['clears_bar'] and res['bar']['both_halves_positive']
                                       and g['CI_positive_on_both_halves'])
                      else 'NO_BUILD_CONFIRMED' if delta.mean() > 0 else 'NO_BUILD_REFUTED')
    res['composite_if_built'] = R(delta.mean() * 0.2 * 0.90)
    assert check_protected() == prot
    d.select(K).with_columns(pl.Series('v1_control', V1['control']),
                             pl.Series('v1_ci_stage2', V1['CI_STAGE2_ONLY'])).write_parquet(
        OUT / 'audit_scores197.parquet')
    res['scores_sha256'] = S165.sha(OUT / 'audit_scores197.parquet')
    (OUT / 'results197.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
