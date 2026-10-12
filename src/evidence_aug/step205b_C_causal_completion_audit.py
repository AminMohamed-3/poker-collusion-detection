"""STEP205 audit: on top of STEP204's composition (ENS4 networks, CI stage-2 + NEW4), does the causal
completion block help?

    control  INC,  base lists                         gate: step165b dec_v1LP 0.692274
    B        ENS4, CI += NEW4                         = STEP204 = STEP203 ENS4_CI (0.705149)
    CI6      ENS4, CI += NEW4 + NEW2
    DT6      ENS4, CI += NEW4;  DT += NEW4 + NEW2
    SP6      ENS4, CI += NEW4;  SP += NEW4 + NEW2
    ALL      ENS4, CI6 + DT6 + SP6

Every arm shares the same 1000 network fits, so arm-vs-B differences are stage-2-only (STEP202: stage-2
noise is exactly zero). Each family's stage-2 is separate, so the three changes are disjoint and ALL must
equal their sum per query -- a gate. Harness is STEP203's, imported where possible (MEMBERS, BASE_CONFIG,
fit_job_cfg, build_views) and otherwise kept line-for-line.

Audit only. No evaluation pair, no held fold, never uploads, builds no file.
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
from evidence_aug import step203_C_ens4_ci_stack as S203
from evidence_aug.step165b_C_lp_v1_level import build_views
from evidence_aug.step205_C_causal_completion import OUT, REG, FEAT4, FEAT2, NEW4, NEW2, DATA

K, FAMS, CI = S165.K, S165.FAMS, S165.CI
DT, SP = 'directed_transfer', 'soft_play'
MEMBERS = S203.MEMBERS
STEP165B_DEC_V1LP = 0.692274
STEP203_ENS4_CI = 0.705149
ARMS = {
    'control': (['INC'], {DT: 'base', SP: 'base', CI: 'base'}),
    'B': (list(MEMBERS), {DT: 'base', SP: 'base', CI: 'c4'}),
    'CI6': (list(MEMBERS), {DT: 'base', SP: 'base', CI: 'c6'}),
    'DT6': (list(MEMBERS), {DT: 'c6', SP: 'base', CI: 'c4'}),
    'SP6': (list(MEMBERS), {DT: 'base', SP: 'c6', CI: 'c4'}),
    'ALL': (list(MEMBERS), {DT: 'c6', SP: 'c6', CI: 'c6'}),
}
CHANGE = {'CI6': CI, 'DT6': DT, 'SP6': SP}


def main(reg_sha):
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    for k, p in (('runner_sha256', Path(__file__)),
                 ('features_sha256', POKER / 'Claude Version/src/evidence_aug/step205_C_causal_completion.py'),
                 ('feature_file_sha256', FEAT2)):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == reg['code'][k], k
    dest = OUT / 'results205.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    feat = (pl.read_parquet(FEAT4).select(K + NEW4)
            .join(pl.read_parquet(FEAT2).select(K + NEW2), on=K, how='inner', validate='1:1'))
    assert feat.height == 45129
    q0 = D['Q'].select(K)
    D['Q'] = D['Q'].join(feat, on=K, how='left', validate='1:1', maintain_order='left')
    assert D['Q'].select(K).equals(q0), 'join reordered Q'
    assert D['Q'].select(NEW4 + NEW2).null_count().to_numpy().sum() == 0, 'unmatched audit rows'
    ph = (pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'phase').collect()
          .join(D['Q'].select('hand_id').unique(), on='hand_id'))
    assert ph['phase'].unique().to_list() == ['development'], 'audit rows must be development-phase'

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
        out = pool.map(S203.fit_job_cfg, jobs, chunksize=1)
    print('network refits done %ds' % round(time.perf_counter() - t0), flush=True)

    nq = len(Q)
    qpairs = Q['pair_id'].to_numpy()
    folds_q = [(np.flatnonzero(np.isin(qpairs, upairs[tr])), np.flatnonzero(np.isin(qpairs, upairs[te])))
               for tr, te in outer]
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
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    LISTS = {(f, 'base'): base[f] for f in FAMS}
    LISTS[(CI, 'c4')] = base[CI] + NEW4
    for f in FAMS:
        LISTS[(f, 'c6')] = base[f] + NEW4 + NEW2
    for (f, nm), cols in LISTS.items():
        assert len(set(cols)) == len(cols), (f, nm)
    need = sorted({(m, f, nm) for arm, (ms, lm) in ARMS.items() for m in ms for f, nm in lm.items()})
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}

    MOD = {}
    for m, f, nm in need:
        z = Q.select(LISTS[(f, nm)]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            MOD[(m, f, nm, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                np.column_stack([z[a], EA[m][k][isf[f][tr]]]), y[a])
    print('stage-2 models fitted: %d  %ds' % (len(MOD), round(time.perf_counter() - t0)), flush=True)

    def score_part(v, m, f, nm):
        """Predictions (+10) for family f's top-20 rows, in D['QIX'] order restricted to f."""
        z = v.select(LISTS[(f, nm)]).to_numpy().astype(float)
        p = np.full(nq, np.nan)
        for k, (tr, te) in enumerate(folds_q):
            b = te[isf[f][te]]
            if len(b):
                p[b] = MOD[(m, f, nm, k)].predict_proba(np.column_stack([z[b], EB[m][b]]))[:, 1] + 10
        return p

    def score_all(v):
        part = {key: score_part(v, *key) for key in need}
        for (m, f, nm), p in part.items():
            assert np.isfinite(p[isf[f]]).all(), (m, f, nm)
        res = {}
        for arm, (ms, lm) in ARMS.items():
            p = D['SS'].copy()
            for f in FAMS:
                rows = np.flatnonzero(isf[f])
                # STEP203 averages whole member vectors; per element this is the same mean over members
                p[D['QIX'][rows]] = np.mean([part[(m, f, lm[f])][rows] for m in ms], axis=0)
            res[arm] = p
        return res

    MS, FOLDS = {}, {}
    for f in FAMS:
        ix = np.flatnonzero(isf[f])
        q = Q[ix]
        yy = q['relevance'].to_numpy().astype(int)
        FOLDS[f] = (ix, list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), yy, q['table_id'].to_numpy())))
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
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabq[q]) for q in qids])
    fq = np.array([fams[q] for q in qids])
    R = lambda x: round(float(x), 6)
    M = {}
    for a in ARMS:
        _, mm = summarize(key.with_columns(pl.Series('score', dec(V1[a]))), truths, fams)
        M[a] = np.array([mm[q] for q in qids])

    def report(a, ref):
        dl = M[a] - M[ref]
        return {'map5': R(M[a].mean()), 'vs': ref, 'decoded_delta': R(dl.mean()),
                'half0': R(dl[hv == 0].mean()), 'half1': R(dl[hv == 1].mean()),
                'positive_on_both_halves': bool(dl[hv == 0].mean() > 0 and dl[hv == 1].mean() > 0),
                'per_family': {f: R(dl[fq == f].mean()) for f in FAMS},
                'per_family_halves': {f: [R(dl[(fq == f) & (hv == h)].mean()) for h in (0, 1)] for f in FAMS},
                'query_se': R(dl.std(ddof=1) / np.sqrt(len(dl)))}

    res = {'registration_sha256': reg_sha, 'level': 'v1, 41 views', 'n_queries': len(qids),
           'n_views': nv + 1, 'n_network_fits': len(jobs), 'members': list(MEMBERS),
           'arms': {'B_vs_control': report('B', 'control')}}
    for a in ('CI6', 'DT6', 'SP6', 'ALL'):
        res['arms'][a] = report(a, 'B')

    # ---- gates ----
    iso = {}
    for a, f in CHANGE.items():
        dl = M[a] - M['B']
        iso[a] = {g: float(np.abs(dl[fq == g]).max()) for g in FAMS if g != f}
    add = M['ALL'] - M['B'] - sum(M[a] - M['B'] for a in CHANGE)
    res['gates'] = {
        'S_star_v1_replay_exact': {'max_abs': gate_v1, 'passed': gate_v1 == 0.0},
        'control_reproduces_165b_v1LP': {'here': R(M['control'].mean()), 'expected': STEP165B_DEC_V1LP,
                                         'tolerance': 0.005,
                                         'passed': bool(abs(M['control'].mean() - STEP165B_DEC_V1LP) <= 0.005)},
        'B_reproduces_STEP203_ENS4_CI': {'here': R(M['B'].mean()), 'expected': STEP203_ENS4_CI,
                                         'tolerance': 0.002, 'exact': bool(R(M['B'].mean()) == STEP203_ENS4_CI),
                                         'passed': bool(abs(M['B'].mean() - STEP203_ENS4_CI) <= 0.002)},
        'untouched_families_exactly_zero': {'max_abs': iso,
                                            'passed': all(v == 0.0 for g in iso.values() for v in g.values())},
        'ALL_equals_sum_of_parts_per_query': {'max_abs': float(np.abs(add).max()),
                                              'passed': bool(np.abs(add).max() < 1e-12)},
        'network_input_453': True, 'audit_rows_development_phase': True}
    gates_ok = all(v['passed'] for v in res['gates'].values() if isinstance(v, dict))

    # ---- registered decision ----
    fam_delta = {a: M[a] - M['B'] for a in CHANGE}

    def include(a, mask):
        dl = fam_delta[a][mask & (fq == CHANGE[a])]
        return bool(len(dl) and dl.mean() > 0)

    incl = [a for a in CHANGE
            if all(include(a, hv == h) for h in (0, 1)) and fam_delta[a].mean() > 0]
    comp = sum((fam_delta[a] for a in incl), np.zeros(len(qids)))
    held = []
    for h in (0, 1):
        sel = [a for a in CHANGE if include(a, hv == h)]
        dh = sum((fam_delta[a] for a in sel), np.zeros(len(qids)))
        held.append((int((hv != h).sum()), float(dh[hv != h].mean()), sel))
    G = sum(n * v for n, v, _ in held) / sum(n for n, _, _ in held)
    both = bool(comp[hv == 0].mean() > 0 and comp[hv == 1].mean() > 0)
    res['decision'] = {
        'included_changes': incl, 'composition_delta_vs_B': R(comp.mean()),
        'composition_halves': [R(comp[hv == 0].mean()), R(comp[hv == 1].mean())],
        'honest_cross_half': [{'selected_on_half': h, 'selected': s, 'held_half_n': n, 'held_half_delta': R(v)}
                              for h, (n, v, s) in enumerate(held)],
        'G_honest': R(G),
        'bars': {'recommended': 0.0073, 'option': 0.0056},
        'composite_expected': R(comp.mean() * 0.2 * 0.886)}
    res['verdict'] = ('VOID' if not gates_ok else
                      'NO_BUILD' if not incl else
                      'BUILD_RECOMMENDED' if (comp.mean() >= 0.0073 and both and G > 0) else
                      'BUILD_OPTION' if (comp.mean() >= 0.0056 and both and G > 0) else
                      'NO_BUILD')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    d.select(K).with_columns([pl.Series('v1_' + a, V1[a]) for a in ARMS]).write_parquet(OUT / 'audit_scores205.parquet')
    res['scores_sha256'] = S165.sha(OUT / 'audit_scores205.parquet')
    pl.DataFrame({'pair_id': qids, 'half': hv, 'family': fq, **{a: M[a] for a in ARMS}}).write_parquet(
        OUT / 'per_query205.parquet')
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)
