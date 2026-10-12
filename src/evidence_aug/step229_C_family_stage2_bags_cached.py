"""STEP229 runner: step221 (per-family stage-2 lists on cached ENS4 networks) plus a STAGE-2 CONFIGURATION axis.
Everything else -- network cache, views, decoder, gates, decision rule -- is step221's, line for line.

Stage-2 configuration axis (new):
    reg['stage2_configs'] = {name: {'extra': {...}, 'seeds': [...]}}
        params = dict(ac.CLF, **extra); a configuration's prediction is the mean over its seeds
        (random_state = seed). A configuration with no stochastic setting must use seeds [11] -- that is
        exactly step221's model, so 'C00' = {'extra': {}, 'seeds': [11]} reproduces its arms bit for bit.
    reg['stage2_sets'] = {set_name: [config names]}
    arm spec: {'members': [...], 'lists': {family: list}, 'stage2': {family: set_name}} (default set 'c00')
    An arm's family score = unweighted mean over the cells (member, configuration) -- STEP186's cell mean.
Nested pruning (topk) is not supported here (asserted absent).
The composite_expected field uses the pooled model-quality transfer 0.74 (OBS 19.178), not 0.886.

--- step221 docstring follows ---
Registered audit for per-family STAGE-2 column lists on the ENS4 networks, with a NETWORK CACHE and
NESTED PRUNING. step206b generalised; identical decision rule.

Why a cache: STEP205/206 showed the 1,000 ENS4 network refits are bit-reproducible (REF and control reproduce
to the sixth decimal), yet each study spends ~800 s recomputing them. The aligned network outputs (EA per
outer fold, EB out of fold) are stored once under CACHE, keyed by a sha256 over everything that determines
them: X, C, QIX, family, relevance, pair and table arrays, every job definition (member config, tag, seed,
train/test masks) and the source of the fitting code. A key mismatch recomputes.

List specs in the registration (per list name, per family):
    [cols]                               append these columns (step206b form)
    {'add': [...], 'drop': [...], 'topk': k}
        add/drop  change the base list;
        topk      NESTED pruning: inside each outer fold, fit on the (added/dropped) list + the member's
                  network columns, rank the list columns by LightGBM gain (ties by position), keep the
                  top k, refit on those + the network columns. The held fold never informs the choice.

Registration fields as step206b (step, code.runner_sha256, code.feature_file_sha256, feature_files, lists,
arms, reference, changes, optional all_arm, control_gate, reference_gate, bars).
Usage: python step209b_C_family_lists_cached.py REGISTRATION_PATH REGSHA
Audit only. Development phase only (asserted). Never uploads, builds no file.
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

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
CACHE = CV / 'cache_ens4_networks'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
MEMBERS = S203.MEMBERS
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cache_key(m, arrays, jobs):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes() if a.dtype != object else '\x1f'.join(map(str, a.tolist())).encode())
    for cfg, (tag, mode, s, tr, te) in jobs:
        h.update(json.dumps([cfg, list(tag), mode, s], sort_keys=True, default=str).encode())
        h.update(np.packbits(tr).tobytes())
        h.update(np.packbits(te).tobytes())
    for src in (S165.__file__, S203.__file__):
        h.update(Path(src).read_bytes())
    h.update(m.encode())
    return h.hexdigest()


def member_jobs(m, outer, rowmask, ptab):
    cfg = MEMBERS[m]
    jobs = []
    for k, (tr, te) in enumerate(outer):
        for s in S165.SEEDS_INNER_OUTER:
            jobs.append((cfg, ((m, 'outer', k), 'joint', s, rowmask(tr), rowmask(te))))
        for j, (itr, ite) in enumerate(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
            for s in S165.SEEDS_INNER:
                jobs.append((cfg, ((m, 'inner', k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite]))))
    return jobs


def spec_of(raw):
    if isinstance(raw, list):
        return {'add': list(raw), 'drop': [], 'topk': None}
    return {'add': list(raw.get('add', [])), 'drop': list(raw.get('drop', [])), 'topk': raw.get('topk')}


def main():
    regp, reg_sha = Path(sys.argv[1]), sys.argv[2]
    assert sha(regp) == reg_sha, 'registration hash mismatch'
    reg = json.loads(regp.read_text(encoding='utf-8'))
    assert sha(__file__) == reg['code']['runner_sha256'], 'runner changed after registration'
    OUT = regp.parent
    tag = reg['step'].lower()
    dest = OUT / f'results_{tag}.json'
    assert not dest.exists(), 'preserve a completed read'
    ARMS = {a: (v['members'], v['lists']) for a, v in reg['arms'].items()}
    S2ARM = {a: {f: v.get('stage2', {}).get(f, 'c00') for f in FAMS} for a, v in reg['arms'].items()}
    S2CFG = reg['stage2_configs']
    S2SET = reg['stage2_sets']
    assert S2CFG['C00'] == {'extra': {}, 'seeds': [11]} and S2SET['c00'] == ['C00']
    for c, v in S2CFG.items():
        stoch = v['extra'].get('colsample_bytree', 1.) < 1. or v['extra'].get('subsample', 1.) < 1.
        assert stoch or v['seeds'] == [11], (c, 'deterministic configuration must use seed 11 only')
        assert len(set(v['seeds'])) == len(v['seeds']), c
    for a, s2 in S2ARM.items():
        for f, sn in s2.items():
            assert sn in S2SET and all(c in S2CFG for c in S2SET[sn]), (a, f, sn)
    REF = reg['reference']
    CHANGE = reg['changes']
    ALL = reg.get('all_arm')
    assert ARMS[REF][0] == list(MEMBERS)
    from evidence_aug.step165b_C_lp_v1_level import build_views
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    q0 = D['Q'].select(K)
    extra_cols = []
    for rel, cols in reg['feature_files'].items():
        p = CV / rel
        assert sha(p) == reg['code']['feature_file_sha256'][rel], rel
        f = pl.read_parquet(p).select(K + cols)
        assert f.height == 45129 or rel in reg.get('audit_row_files', []), rel
        D['Q'] = D['Q'].join(f, on=K, how='left', validate='1:1', maintain_order='left')
        extra_cols += cols
    assert len(set(extra_cols)) == len(extra_cols), 'a column is supplied twice'
    assert D['Q'].select(K).equals(q0), 'join reordered Q'
    if extra_cols:
        assert D['Q'].select(extra_cols).null_count().to_numpy().sum() == 0, 'unmatched audit rows'
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
    rel = d['relevance'].to_numpy()
    arrays = (X.astype(np.float32), C.astype(np.float32), D['QIX'], np.asarray(D['fam']), rel, pairs, tables)
    used = sorted({m for ms, _ in ARMS.values() for m in ms}, key=list(MEMBERS).index)
    nq = len(Q)
    qpairs = Q['pair_id'].to_numpy()
    folds_q = [(np.flatnonzero(np.isin(qpairs, upairs[tr])), np.flatnonzero(np.isin(qpairs, upairs[te])))
               for tr, te in outer]
    CACHE.mkdir(exist_ok=True)
    EA, EB, cache_rec = {}, {}, {}
    todo = {}
    for m in used:
        jobs = member_jobs(m, outer, rowmask, ptab)
        key = cache_key(m, arrays, jobs)
        cp = CACHE / f'{m}_{key[:16]}.npz'
        if cp.exists():
            z = np.load(cp)
            assert str(z['key']) == key
            EA[m] = {k: z[f'ea{k}'] for k in range(5)}
            EB[m] = z['eb']
            cache_rec[m] = {'key': key, 'loaded': cp.name}
        else:
            todo[m] = (jobs, key, cp)
    n_fits = sum(len(v[0]) for v in todo.values())
    print('X %s  %d network fits to run, %d members from cache' % (X.shape, n_fits, len(used) - len(todo)),
          flush=True)
    if todo:
        alljobs = [j for v in todo.values() for j in v[0]]
        init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], rel, pairs, tables)
        with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
            out = pool.map(S203.fit_job_cfg, alljobs, chunksize=1)
        print('network refits done %ds' % round(time.perf_counter() - t0), flush=True)
        for m, (jobs, key, cp) in todo.items():
            AF_te, AF_in = {k: [] for k in range(5)}, {k: {} for k in range(5)}
            for jtag, mode, s, rows, tq, p, S, trace, sec in out:
                if jtag[0] != m:
                    continue
                if jtag[1] == 'outer':
                    AF_te[jtag[2]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
                else:
                    AF_in[jtag[2]].setdefault(jtag[3], []).append((tq, S165.aligned_features(D, rows, tq, S, p)))
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
            np.savez(cp, key=np.array(key), eb=eb, **{f'ea{k}': ea[k] for k in range(5)})
            cache_rec[m] = {'key': key, 'written': cp.name}
        del out

    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    LISTS = {}
    for nm, spec in reg['lists'].items():
        for f in FAMS:
            sp = spec_of(spec.get(f, []))
            assert sp['topk'] is None, (nm, f, 'nested pruning is not supported by this runner')
            assert set(sp['drop']) <= set(base[f]), (nm, f, 'drop outside base')
            cols = [c for c in base[f] if c not in set(sp['drop'])] + sp['add']
            assert len(set(cols)) == len(cols), (f, nm)
            LISTS[(f, nm)] = cols
    need = sorted({(m, f, lm[f], c) for a, (ms, lm) in ARMS.items() for m in ms for f in FAMS
                   for c in S2SET[S2ARM[a][f]]})
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    MOD = {}
    n_fit = 0
    for m, f, nm, c in need:
        z = Q.select(LISTS[(f, nm)]).to_numpy().astype(float)
        params = dict(ac.CLF, **S2CFG[c]['extra'])
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            ztr = np.column_stack([z[a], EA[m][k][isf[f][tr]]])
            MOD[(m, f, nm, c, k)] = [lgb.LGBMClassifier(**params, random_state=s).fit(ztr, y[a])
                                     for s in S2CFG[c]['seeds']]
            n_fit += len(S2CFG[c]['seeds'])
    print('stage-2 model sets: %d, LightGBM fits: %d  %ds' % (len(MOD), n_fit, round(time.perf_counter() - t0)),
          flush=True)

    def score_part(v, m, f, nm, c):
        z = v.select(LISTS[(f, nm)]).to_numpy().astype(float)
        p = np.full(nq, np.nan)
        for k, (tr, te) in enumerate(folds_q):
            b = te[isf[f][te]]
            if len(b):
                zb = np.column_stack([z[b], EB[m][b]])
                preds = [mdl.predict_proba(zb)[:, 1] for mdl in MOD[(m, f, nm, c, k)]]
                p[b] = (preds[0] if len(preds) == 1 else np.mean(preds, axis=0)) + 10
        return p

    def score_all(v):
        part = {key: score_part(v, *key) for key in need}
        for (m, f, nm, c), p in part.items():
            assert np.isfinite(p[isf[f]]).all(), (m, f, nm, c)
        res = {}
        for arm, (ms, lm) in ARMS.items():
            p = D['SS'].copy()
            for f in FAMS:
                rows = np.flatnonzero(isf[f])
                p[D['QIX'][rows]] = np.mean([part[(m, f, lm[f], c)][rows] for m in ms
                                             for c in S2SET[S2ARM[arm][f]]], axis=0)
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
                'per_family_up_down': {f: [int((dl[fq == f] > 0).sum()), int((dl[fq == f] < 0).sum())] for f in FAMS},
                'query_se': R(dl.std(ddof=1) / np.sqrt(len(dl)))}

    res = {'registration_sha256': reg_sha, 'step': reg['step'], 'level': 'v1, 41 views',
           'n_queries': len(qids), 'n_views': nv + 1, 'network_fits_run': n_fits, 'network_cache': cache_rec,
           'arms': {}}
    cg = reg['control_gate']
    if cg['arm'] in ARMS:
        res['arms'][f"{REF}_vs_{cg['arm']}"] = report(REF, cg['arm'])
    for a in ARMS:
        if a not in (REF, cg['arm']):
            res['arms'][a] = report(a, REF)
    res['stage2_model_sets'] = len(MOD)
    res['stage2_lightgbm_fits'] = n_fit

    iso = {}
    for a, f in CHANGE.items():
        dl = M[a] - M[REF]
        iso[a] = {g: float(np.abs(dl[fq == g]).max()) for g in FAMS if g != f}
    gates = {
        'S_star_v1_replay_exact': {'max_abs': gate_v1, 'passed': gate_v1 == 0.0},
        'reference_reproduces': {'here': R(M[REF].mean()), **reg['reference_gate'],
                                 'exact': bool(R(M[REF].mean()) == reg['reference_gate']['expected']),
                                 'passed': bool(abs(M[REF].mean() - reg['reference_gate']['expected'])
                                                <= reg['reference_gate']['tolerance'])},
        'untouched_families_exactly_zero': {'max_abs': iso,
                                            'passed': all(v == 0.0 for g in iso.values() for v in g.values())}}
    if cg['arm'] in ARMS:
        gates['control_reproduces'] = {'here': R(M[cg['arm']].mean()), **cg,
                                       'passed': bool(abs(M[cg['arm']].mean() - cg['expected']) <= cg['tolerance'])}
    if ALL:
        add = M[ALL] - M[REF] - sum(M[a] - M[REF] for a in CHANGE)
        gates['all_equals_sum_of_parts'] = {'max_abs': float(np.abs(add).max()),
                                            'passed': bool(np.abs(add).max() < 1e-12)}
    res['gates'] = gates
    gates_ok = all(v['passed'] for v in gates.values())

    fam_delta = {a: M[a] - M[REF] for a in CHANGE}

    def include(a, mask):
        dl = fam_delta[a][mask & (fq == CHANGE[a])]
        return bool(len(dl) and dl.mean() > 0)

    incl = [a for a in CHANGE if all(include(a, hv == h) for h in (0, 1)) and fam_delta[a].mean() > 0]
    byfam = {}
    for a in incl:
        f = CHANGE[a]
        if f not in byfam or fam_delta[a].mean() > fam_delta[byfam[f]].mean():
            byfam[f] = a
    incl = sorted(byfam.values())
    comp = sum((fam_delta[a] for a in incl), np.zeros(len(qids)))
    held = []
    for h in (0, 1):
        pick = {}
        for a in CHANGE:
            if include(a, hv == h):
                f = CHANGE[a]
                m_h = fam_delta[a][(hv == h) & (fq == f)].mean()
                if f not in pick or m_h > pick[f][1]:
                    pick[f] = (a, m_h)
        sel = sorted(v[0] for v in pick.values())
        dh = sum((fam_delta[a] for a in sel), np.zeros(len(qids)))
        held.append((int((hv != h).sum()), float(dh[hv != h].mean()), sel))
    G = sum(n * v for n, v, _ in held) / sum(n for n, _, _ in held)
    both = bool(comp[hv == 0].mean() > 0 and comp[hv == 1].mean() > 0)
    bars = reg['bars']
    res['decision'] = {
        'included_changes': incl, 'composition_delta_vs_reference': R(comp.mean()),
        'composition_halves': [R(comp[hv == 0].mean()), R(comp[hv == 1].mean())],
        'honest_cross_half': [{'selected_on_half': h, 'selected': s, 'held_half_n': n, 'held_half_delta': R(v)}
                              for h, (n, v, s) in enumerate(held)],
        'G_honest': R(G), 'bars': bars, 'composite_expected': R(comp.mean() * 0.2 * 0.74)}
    res['verdict'] = ('VOID' if not gates_ok else
                      'NO_BUILD' if not incl else
                      'BUILD_RECOMMENDED' if (comp.mean() >= bars['recommended'] and both and G > 0) else
                      'BUILD_OPTION' if (comp.mean() >= bars['option'] and both and G > 0) else
                      'NO_BUILD')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    d.select(K).with_columns([pl.Series('v1_' + a, V1[a]) for a in ARMS]).write_parquet(OUT / f'audit_scores_{tag}.parquet')
    res['scores_sha256'] = sha(OUT / f'audit_scores_{tag}.parquet')
    pl.DataFrame({'pair_id': qids, 'half': hv, 'family': fq, **{a: M[a] for a in ARMS}}).write_parquet(
        OUT / f'per_query_{tag}.parquet')
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('gates', 'decision', 'verdict', 'seconds')}, indent=1), flush=True)
    print('"verdict": "%s"' % res['verdict'], flush=True)


if __name__ == '__main__':
    main()
