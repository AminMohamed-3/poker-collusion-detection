"""STEP203: do STEP188's network ensemble and STEP197's CI causal-fold columns COMPOSE?

Amin asked for the STEP197 CI file if it beats submission_step188_ensemble_on_step169.csv and for
STEP188 otherwise. That contest cannot be settled: +0.006996 against +0.00571 is a gap of 0.0013, and
STEP202 puts the network basin noise that STEP188's arm carries at SD 0.0036. So this measures whether
the two COMPOSE instead, which the audit CAN answer and which if true dissolves the choice.

The two changes touch disjoint parts of the pipeline. STEP188 averages the stage-1 NETWORK across four
architectures at prediction level. STEP194c/197 extends coordinated_isolation's STAGE-2 column list by
four columns and leaves the 453-column network input untouched. Neither can alter the other's inputs.

    control   INC only,  base stage-2 columns          must reproduce step165b dec_v1LP 0.692274
    CI        INC only,  CI columns extended            STEP197 re-run, should land near +0.006996
    ENS4      4 members, base stage-2 columns           STEP188's composition, paired for the first time
    ENS4_CI   4 members, CI columns extended            THE CANDIDATE

The number this exists to produce is ENS4_CI - (ENS4 + CI).

The S* v1 replay gate uses step165b's OWN accumulation grouping, (acc_thin + acc_pad + stored SS) / 41.
STEP197 reimplemented it with one accumulator and was held VOID on a 6.98e-10 addition-order artefact
(gate197_sstar_resolution.json). Calling the original expression is the fix.

Audit only. No evaluation pair, no held fold, never uploads, builds no file.

Usage: python step203_C_ens4_ci_stack.py REGSHA
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
sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug.step165b_C_lp_v1_level import build_views

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step203_20260917'
REG = OUT / 'registration203.json'
FEAT = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
NEW = ['pf_others_fold_after_pair_agg', 'pf_others_fold_before_pair_agg',
       'pf_pair_aggresses_preflop', 'pf_others_call_after_pair_agg']
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
STEP165B_DEC_V1LP = 0.692274
TOL = 0.005
BASE_CONFIG = {'h1': 64, 'medium': 16, 'h2': 32, 'epochs_max': 400, 'patience': 40, 'lr': 1e-3,
               'wd': 1e-2, 'lam_list2': .5, 'w_stage1': 1., 'clip': 6., 'val_share_tables': .2}
MEMBERS = {'INC': {}, 'N06_lam_list2_0': {'lam_list2': 0.0},
           'N07_lam_list2_2': {'lam_list2': 2.0}, 'N08_w_stage1_3': {'w_stage1': 3.0}}


def set_config(cfg):
    """CONFIG must be applied INSIDE the worker: on Windows the children re-import step165_e2e_mlp
    with its DEFAULT CONFIG, so a config set in the parent is silently ignored and every member would
    come out as the incumbent. STEP188's lesson, reused rather than rediscovered."""
    from evidence_aug import step165_e2e_mlp as M
    assert dict(M.CONFIG) == BASE_CONFIG or set_config.applied, \
        'step165_e2e_mlp.CONFIG default has drifted from BASE_CONFIG'
    set_config.applied = True
    M.CONFIG.clear()
    M.CONFIG.update(BASE_CONFIG)
    M.CONFIG.update(cfg)
    return M


set_config.applied = False


def fit_job_cfg(job):
    cfg, inner = job
    set_config(cfg)
    return S165.fit_job(inner)


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not (OUT / 'results203.json').exists(), 'preserve a completed read'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    feat = pl.read_parquet(FEAT).select(K + NEW)
    q0 = D['Q'].select(K)
    D['Q'] = D['Q'].join(feat, on=K, how='left', validate='1:1', maintain_order='left')
    assert D['Q'].select(K).equals(q0), 'join reordered Q'
    assert D['Q'].select(NEW).null_count().to_numpy().sum() == 0, 'unmatched audit rows'

    X, C = S165.matrices(D)
    assert X.shape[1] == 453, 'network input changed: %s' % (X.shape,)
    d, Q = D['d'], D['Q']
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'],
                 d['relevance'].to_numpy(), pairs, tables)

    jobs = []
    for m, cfg in MEMBERS.items():
        for k, (tr, te) in enumerate(outer):
            for s in S165.SEEDS_INNER_OUTER:
                jobs.append((cfg, ((m, 'outer', k), 'joint', s, rowmask(tr), rowmask(te))))
            for j, (itr, ite) in enumerate(
                    GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
                for s in S165.SEEDS_INNER:
                    jobs.append((cfg, ((m, 'inner', k, j), 'joint', s,
                                       rowmask(tr[itr]), rowmask(tr[ite]))))
    print('X %s  %d network fits over %d members' % (X.shape, len(jobs), len(MEMBERS)), flush=True)
    with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(fit_job_cfg, jobs, chunksize=1)
    print('network refits done %ds' % round(time.perf_counter() - t0), flush=True)

    nq = len(Q)
    qpairs = Q['pair_id'].to_numpy()
    folds_q = []
    for k, (tr, te) in enumerate(outer):
        folds_q.append((np.flatnonzero(np.isin(qpairs, upairs[tr])),
                        np.flatnonzero(np.isin(qpairs, upairs[te]))))
    EA, EB = {}, {}
    for m in MEMBERS:
        AF_te, AF_in = {k: [] for k in range(5)}, {k: {} for k in range(5)}
        for jtag, mode, s, rows, tq, p, S, trace, sec in out:
            if jtag[0] != m:
                continue
            if jtag[1] == 'outer':
                AF_te[jtag[2]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
            else:
                AF_in[jtag[2]].setdefault(jtag[3], []).append(
                    (tq, S165.aligned_features(D, rows, tq, S, p)))
        eb = np.full((nq, len(S165.NN_COLS)), np.nan)
        ea = {}
        for k, (trq, teq) in enumerate(folds_q):
            a = np.full((nq, len(S165.NN_COLS)), np.nan)
            for j, lst in AF_in[k].items():
                a[lst[0][0]] = np.mean([x for _, x in lst], axis=0)
            e = np.full((nq, len(S165.NN_COLS)), np.nan)
            e[AF_te[k][0][0]] = np.mean([x for _, x in AF_te[k]], axis=0)
            assert np.isfinite(a[trq]).all() and np.isfinite(e[teq]).all(), m
            ea[k] = a[trq]
            eb[teq] = e[teq]
        assert np.isfinite(eb).all(), m
        EA[m], EB[m] = ea, eb
    del out

    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols']
            for f in FAMS}
    ci_cols = dict(base)
    ci_cols[CI] = base[CI] + NEW
    COLS = {'base': base, 'ci': ci_cols}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}

    MOD = {}
    for m in MEMBERS:
        for ck, colsf in COLS.items():
            for f in FAMS:
                z = Q.select(colsf[f]).to_numpy().astype(float)
                for k, (tr, te) in enumerate(folds_q):
                    a = tr[isf[f][tr]]
                    MOD[(m, ck, f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                        np.column_stack([z[a], EA[m][k][isf[f][tr]]]), y[a])
    print('stage-2 models fitted: %d  %ds' % (len(MOD), round(time.perf_counter() - t0)), flush=True)

    def score_one(v, m, ck):
        colsf = COLS[ck]
        p = D['SS'].copy()
        for f in FAMS:
            z = v.select(colsf[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds_q):
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = MOD[(m, ck, f, k)].predict_proba(
                        np.column_stack([z[b], EB[m][b]]))[:, 1] + 10
        return p

    ARMS = {'control': (['INC'], 'base'), 'CI': (['INC'], 'ci'),
            'ENS4': (list(MEMBERS), 'base'), 'ENS4_CI': (list(MEMBERS), 'ci')}

    def score_all(v):
        one = {(m, ck): score_one(v, m, ck) for m in MEMBERS for ck in COLS}
        return {arm: np.mean([one[(m, ck)] for m in ms], axis=0)
                for arm, (ms, ck) in ARMS.items()}

    # S* per-family fold models on the step-124 FOLDS, for the view-construction gate only.
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

    assert float(np.abs(score_sstar(Q) - D['SS']).max()) == 0.0, 'score_sstar does not reproduce SS'

    views, pads = build_views(D)
    A0 = score_all(Q)
    # TWO accumulator slots, thinned and padded, exactly as step165b grouped them. The stored D['SS']
    # is the 41st term. This grouping is the gate fix -- see gate197_sstar_resolution.json.
    accA = [{a: np.zeros(len(d)) for a in ARMS}, {a: np.zeros(len(d)) for a in ARMS}]
    accS = [np.zeros(len(d)), np.zeros(len(d))]
    nv = 0
    for slot, (maker, seeds) in enumerate(((views, (1241, 2241)), (pads, (1242, 2242)))):
        for s in seeds:
            for v in maker(.5, 10, s):
                sc = score_all(v)
                for a in ARMS:
                    accA[slot][a] += sc[a]
                accS[slot] += score_sstar(v)
                nv += 1
            print('views %d  %ds' % (nv, round(time.perf_counter() - t0)), flush=True)
    assert nv == 40, nv
    V1 = {a: (accA[0][a] + accA[1][a] + A0[a]) / (nv + 1) for a in ARMS}
    V1S = (accS[0] + accS[1] + D['SS']) / (nv + 1)
    gate_v1 = float(np.abs(V1S - D['V1']).max())
    print('S* v1 replay max abs diff %s' % gate_v1, flush=True)

    dec = D['dec']
    truths = {k[0]: set(g['hand_id'])
              for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabq[q]) for q in qids])
    R = lambda x: round(float(x), 6)
    M = {}
    for a in ARMS:
        _, mm = summarize(key.with_columns(pl.Series('score', dec(V1[a]))), truths, fams)
        M[a] = np.array([mm[q] for q in qids])

    def report(a, ref):
        delta = M[a] - M[ref]
        r = {'map5': R(M[a].mean()), 'vs': ref, 'decoded_delta': R(delta.mean()),
             'half0': R(delta[hv == 0].mean()), 'half1': R(delta[hv == 1].mean()),
             'positive_on_both_halves': bool(delta[hv == 0].mean() > 0
                                             and delta[hv == 1].mean() > 0),
             'per_family': {}}
        for f in FAMS:
            sel = np.array([fams[q] == f for q in qids])
            r['per_family'][f] = R(delta[sel].mean())
        return r

    res = {'registration_sha256': reg_sha,
           'level': 'v1, 41 views (original pool + 20 thinned + 20 padded)',
           'n_queries': len(qids), 'n_views': nv + 1, 'n_network_fits': len(jobs),
           'members': list(MEMBERS),
           'map5_control': R(M['control'].mean()),
           'arms': {'CI': report('CI', 'control'), 'ENS4': report('ENS4', 'control'),
                    'ENS4_CI': report('ENS4_CI', 'control'),
                    'ENS4_CI_vs_ENS4': report('ENS4_CI', 'ENS4'),
                    'ENS4_CI_vs_CI': report('ENS4_CI', 'CI')}}
    ci_d = res['arms']['CI']['decoded_delta']
    en_d = res['arms']['ENS4']['decoded_delta']
    st_d = res['arms']['ENS4_CI']['decoded_delta']
    res['THE_NUMBER'] = {
        'CI': ci_d, 'ENS4': en_d, 'ENS4_CI': st_d, 'sum_of_parts': R(ci_d + en_d),
        'additivity': R(st_d - (ci_d + en_d)),
        'stack_beats_both_parts': bool(st_d >= max(ci_d, en_d)),
        'reading': ('additivity near zero means the two changes compose and the contest Amin posed '
                    'dissolves because the stack dominates both of its parts.')}
    res['step197_reproduction_check'] = {
        'step197_CI_delta': 0.006996, 'here': ci_d, 'abs_diff': R(abs(ci_d - 0.006996)),
        'note': ('stage-2-only on shared identically-seeded fits, so this should be a near-exact '
                 'reproduction; a large gap would mean this harness is not STEP197s')}
    untouched = {}
    for a, ref in (('CI', 'control'), ('ENS4_CI', 'ENS4')):
        dl = M[a] - M[ref]
        untouched[a + '_minus_' + ref] = {
            f: float(np.abs(dl[[fams[q] == f for q in qids]]).max()) for f in FAMS if f != CI}
    res['gates'] = {
        'S_star_v1_replay_exact': {'max_abs': gate_v1, 'passed': gate_v1 == 0.0,
                                   'form': "step165b's own grouping (acc_thin + acc_pad + SS)/41"},
        'DT_SP_exactly_zero': {'max_abs': untouched,
                               'passed': all(v == 0.0 for g in untouched.values()
                                             for v in g.values())},
        'control_reproduces_165b_v1LP': {
            'step165b_dec_v1LP': STEP165B_DEC_V1LP, 'here': R(M['control'].mean()),
            'tolerance': TOL,
            'passed': bool(abs(M['control'].mean() - STEP165B_DEC_V1LP) <= TOL)},
        'network_input_453': True}
    g = res['gates']
    gates_ok = (g['S_star_v1_replay_exact']['passed'] and g['DT_SP_exactly_zero']['passed']
                and g['control_reproduces_165b_v1LP']['passed'])
    stack = res['arms']['ENS4_CI']
    res['bar'] = {'corrected_build_bar_decoded': 0.0073, 'clears_bar': bool(st_d >= 0.0073),
                  'note': ('the bar is 2 SD of NETWORK-perturbation noise (STEP202) and ENS4_CI '
                           'contains a network change, so it applies here in full')}
    res['verdict'] = (
        'VOID' if not gates_ok else
        'BUILD_THE_STACK' if (stack['positive_on_both_halves']
                              and res['THE_NUMBER']['stack_beats_both_parts']
                              and res['bar']['clears_bar']) else
        'BUILD_CI_ONLY' if (res['arms']['CI']['positive_on_both_halves'] and ci_d > en_d) else
        'RECOMMEND_STEP188_AS_IS')
    res['composite_if_built'] = {'ENS4_CI': R(st_d * 0.2 * 0.90), 'CI_only': R(ci_d * 0.2 * 0.90),
                                 'ENS4_only': R(en_d * 0.2 * 0.90)}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    OUT.mkdir(parents=True, exist_ok=True)
    d.select(K).with_columns([pl.Series('v1_' + a, V1[a]) for a in ARMS]).write_parquet(
        OUT / 'audit_scores203.parquet')
    res['scores_sha256'] = S165.sha(OUT / 'audit_scores203.parquet')
    (OUT / 'results203.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
