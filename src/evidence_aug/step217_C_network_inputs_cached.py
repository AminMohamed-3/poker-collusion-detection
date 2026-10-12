"""Registered audit for NETWORK INPUT variants on the ENS4 evidence networks (step209b generalised).

The ENS4 networks read X (every hand row of a pair: production rcols + struct + chronology + family) and C (the
top-20 context: cs_ + rs_ + chronology). A variant appends registered columns to X and/or C; with
`permute: true` the appended values are shuffled across rows by a fixed generator (a PLACEBO: same marginal
distribution, no information), which measures what a mere change of network input costs or gains.

Network outputs are cached per (member, variant) under CACHE, keyed by sha256 over the arrays, the job
definitions and the fitting code, exactly as step209b. Stage 2 is production's per-family LightGBM with the
registered column lists, on each arm's own network outputs.

A network change moves all three families, so the decision is on the OVERALL audit:
  a network arm is a candidate iff (arm - REF) > 0 on both halves AND (arm - placebo) > 0 on both halves;
  at most one network arm is included (the larger overall delta);
  G_honest: on each half pick the candidate-on-that-half with the larger half delta, score it on the other half;
  BUILD_RECOMMENDED iff included delta >= bars.recommended and G > 0; BUILD_OPTION iff >= bars.option and G > 0.

Registration fields: step, code.runner_sha256, code.feature_file_sha256, feature_files, lists, network_variants
{name: {X_add, C_add, permute}}, arms {name: {members, lists, net}}, reference, network_arms, placebo,
control_gate, reference_gate, bars.
Usage: python step217_C_network_inputs_cached.py REGISTRATION_PATH REGSHA [POOL_SIZE]
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
PERM_SEED = 217
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


def main():
    regp, reg_sha = Path(sys.argv[1]), sys.argv[2]
    pool_size = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    assert sha(regp) == reg_sha, 'registration hash mismatch'
    reg = json.loads(regp.read_text(encoding='utf-8'))
    assert sha(__file__) == reg['code']['runner_sha256'], 'runner changed after registration'
    OUT = regp.parent
    tag = reg['step'].lower()
    dest = OUT / f'results_{tag}.json'
    assert not dest.exists(), 'preserve a completed read'
    ARMS = {a: (v['members'], v['lists'], v.get('net', 'base')) for a, v in reg['arms'].items()}
    REF = reg['reference']
    NETARMS = list(reg['network_arms'])
    PLACEBO = reg['placebo']
    VARS = reg['network_variants']
    assert ARMS[REF][0] == list(MEMBERS) and ARMS[REF][2] == 'base'
    for a in NETARMS + [PLACEBO]:
        assert ARMS[a][1] == ARMS[REF][1] and ARMS[a][0] == ARMS[REF][0], 'network arms change the network only'
    from evidence_aug.step165b_C_lp_v1_level import build_views
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    q0 = D['Q'].select(K)
    d0 = D['d'].select(K)
    extra_cols = []
    dx = d0
    for rel, cols in reg['feature_files'].items():
        p = CV / rel
        assert sha(p) == reg['code']['feature_file_sha256'][rel], rel
        f = pl.read_parquet(p).select(K + cols)
        assert f.height == 45129, rel
        D['Q'] = D['Q'].join(f, on=K, how='left', validate='1:1', maintain_order='left')
        dx = dx.join(f, on=K, how='left', validate='1:1', maintain_order='left')
        extra_cols += cols
    assert len(set(extra_cols)) == len(extra_cols), 'a column is supplied twice'
    assert D['Q'].select(K).equals(q0), 'join reordered Q'
    assert dx.select(K).equals(d0), 'join reordered d'
    if extra_cols:
        assert D['Q'].select(extra_cols).null_count().to_numpy().sum() == 0, 'unmatched audit rows'
        assert dx.select(extra_cols).null_count().to_numpy().sum() == 0, 'unmatched network rows'
    ph = (pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'phase').collect()
          .join(dx.select('hand_id').unique(), on='hand_id'))
    assert ph['phase'].unique().to_list() == ['development'], 'network rows must be development-phase'

    X0, C0 = S165.matrices(D)
    assert X0.shape[1] == 453 and C0.shape[1] == 62, 'base network input changed: %s %s' % (X0.shape, C0.shape)
    d, Q = D['d'], D['Q']
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    rel = d['relevance'].to_numpy()

    XC = {}
    var_rec = {}
    for vn, spec in VARS.items():
        xa, ca = list(spec.get('X_add', [])), list(spec.get('C_add', []))
        X, C = X0, C0
        rec = {'X_add': len(xa), 'C_add': len(ca), 'permute': bool(spec.get('permute', False))}
        if xa:
            v = dx.select(xa).to_numpy().astype(np.float64)
            if spec.get('permute'):
                v = v[np.random.default_rng([PERM_SEED, 1]).permutation(len(v))]
            X = np.column_stack([X0, v])
        if ca:
            v = Q.select(ca).to_numpy().astype(np.float64)
            if spec.get('permute'):
                v = v[np.random.default_rng([PERM_SEED, 2]).permutation(len(v))]
            C = np.column_stack([C0, v])
        assert np.isfinite(X).all() and np.isfinite(C).all()
        rec['shapes'] = [list(X.shape), list(C.shape)]
        XC[vn] = (X, C)
        var_rec[vn] = rec
    assert VARS['base'] == {} or (not VARS['base'].get('X_add') and not VARS['base'].get('C_add'))

    used = sorted({(m, v) for ms, _, v in ARMS.values() for m in ms},
                  key=lambda t: (list(VARS).index(t[1]), list(MEMBERS).index(t[0])))
    nq = len(Q)
    qpairs = Q['pair_id'].to_numpy()
    folds_q = [(np.flatnonzero(np.isin(qpairs, upairs[tr])), np.flatnonzero(np.isin(qpairs, upairs[te])))
               for tr, te in outer]
    CACHE.mkdir(exist_ok=True)
    EA, EB, cache_rec = {}, {}, {}
    todo = {}
    for m, vn in used:
        X, C = XC[vn]
        arrays = (X.astype(np.float32), C.astype(np.float32), D['QIX'], np.asarray(D['fam']), rel, pairs, tables)
        jobs = member_jobs(m, outer, rowmask, ptab)
        key = cache_key(m, arrays, jobs)
        cp = CACHE / f'{m}_{key[:16]}.npz'
        if cp.exists():
            z = np.load(cp)
            assert str(z['key']) == key
            EA[(m, vn)] = {k: z[f'ea{k}'] for k in range(5)}
            EB[(m, vn)] = z['eb']
            cache_rec[f'{m}|{vn}'] = {'key': key, 'loaded': cp.name}
        else:
            todo.setdefault(vn, {})[m] = (jobs, key, cp)
    n_fits = sum(len(j[0]) for v in todo.values() for j in v.values())
    print('X %s C %s  %d network fits to run, %d (member, variant) from cache' % (
        X0.shape, C0.shape, n_fits, len(used) - sum(len(v) for v in todo.values())), flush=True)
    for vn, tmems in todo.items():
        X, C = XC[vn]
        alljobs = [j for v in tmems.values() for j in v[0]]
        init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], rel, pairs, tables)
        with Pool(pool_size, initializer=S165.pool_init, initargs=init_args) as pool:
            out = pool.map(S203.fit_job_cfg, alljobs, chunksize=1)
        print('network refits done for %s  %ds' % (vn, round(time.perf_counter() - t0)), flush=True)
        for m, (jobs, key, cp) in tmems.items():
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
            EA[(m, vn)], EB[(m, vn)] = ea, eb
            np.savez(cp, key=np.array(key), eb=eb, **{f'ea{k}': ea[k] for k in range(5)})
            cache_rec[f'{m}|{vn}'] = {'key': key, 'written': cp.name}
        del out

    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    LISTS = {}
    for nm, spec in reg['lists'].items():
        for f in FAMS:
            add = spec.get(f, [])
            assert isinstance(add, list)
            cols = base[f] + list(add)
            assert len(set(cols)) == len(cols), (f, nm)
            LISTS[(f, nm)] = cols
    need = sorted({(m, v, f, lm[f]) for ms, lm, v in ARMS.values() for m in ms for f in FAMS})
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    MOD = {}
    for m, vn, f, nm in need:
        z = Q.select(LISTS[(f, nm)]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            ztr = np.column_stack([z[a], EA[(m, vn)][k][isf[f][tr]]])
            MOD[(m, vn, f, nm, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(ztr, y[a])
    print('stage-2 models fitted: %d  %ds' % (len(MOD), round(time.perf_counter() - t0)), flush=True)

    def score_part(v, m, vn, f, nm):
        z = v.select(LISTS[(f, nm)]).to_numpy().astype(float)
        p = np.full(nq, np.nan)
        for k, (tr, te) in enumerate(folds_q):
            b = te[isf[f][te]]
            if len(b):
                p[b] = MOD[(m, vn, f, nm, k)].predict_proba(np.column_stack([z[b], EB[(m, vn)][b]]))[:, 1] + 10
        return p

    def score_all(v):
        part = {key: score_part(v, *key) for key in need}
        for (m, vn, f, nm), p in part.items():
            assert np.isfinite(p[isf[f]]).all(), (m, vn, f, nm)
        res = {}
        for arm, (ms, lm, vn) in ARMS.items():
            p = D['SS'].copy()
            for f in FAMS:
                rows = np.flatnonzero(isf[f])
                p[D['QIX'][rows]] = np.mean([part[(m, vn, f, lm[f])][rows] for m in ms], axis=0)
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
                'up_down': [int((dl > 0).sum()), int((dl < 0).sum())],
                'query_se': R(dl.std(ddof=1) / np.sqrt(len(dl)))}

    res = {'registration_sha256': reg_sha, 'step': reg['step'], 'level': 'v1, 41 views',
           'n_queries': len(qids), 'n_views': nv + 1, 'network_fits_run': n_fits, 'network_cache': cache_rec,
           'network_variants': var_rec, 'pool_size': pool_size, 'arms': {}}
    cg = reg['control_gate']
    if cg['arm'] in ARMS:
        res['arms'][f"{REF}_vs_{cg['arm']}"] = report(REF, cg['arm'])
    for a in ARMS:
        if a not in (REF, cg['arm']):
            res['arms'][a] = report(a, REF)
    for a in NETARMS:
        res['arms'][a + '_vs_' + PLACEBO] = report(a, PLACEBO)

    gates = {
        'S_star_v1_replay_exact': {'max_abs': gate_v1, 'passed': gate_v1 == 0.0},
        'reference_reproduces': {'here': R(M[REF].mean()), **reg['reference_gate'],
                                 'exact': bool(R(M[REF].mean()) == reg['reference_gate']['expected']),
                                 'passed': bool(abs(M[REF].mean() - reg['reference_gate']['expected'])
                                                <= reg['reference_gate']['tolerance'])}}
    if cg['arm'] in ARMS:
        gates['control_reproduces'] = {'here': R(M[cg['arm']].mean()), **cg,
                                       'passed': bool(abs(M[cg['arm']].mean() - cg['expected']) <= cg['tolerance'])}
    res['gates'] = gates
    gates_ok = all(v['passed'] for v in gates.values())

    dR = {a: M[a] - M[REF] for a in NETARMS}
    dP = {a: M[a] - M[PLACEBO] for a in NETARMS}
    pos = lambda x, mask: bool(mask.sum() and x[mask].mean() > 0)
    cand = lambda a, mask: pos(dR[a], mask) and pos(dP[a], mask)
    incl = [a for a in NETARMS if all(cand(a, hv == h) for h in (0, 1)) and dR[a].mean() > 0]
    incl = sorted(incl, key=lambda a: -dR[a].mean())[:1]
    comp = dR[incl[0]] if incl else np.zeros(len(qids))
    held = []
    for h in (0, 1):
        c = [a for a in NETARMS if cand(a, hv == h)]
        sel = max(c, key=lambda a: dR[a][hv == h].mean()) if c else None
        dh = dR[sel] if sel else np.zeros(len(qids))
        held.append((int((hv != h).sum()), float(dh[hv != h].mean()), sel))
    G = sum(n * v for n, v, _ in held) / sum(n for n, _, _ in held)
    bars = reg['bars']
    res['decision'] = {
        'included_network_arm': incl, 'delta_vs_reference': R(comp.mean()),
        'halves': [R(comp[hv == 0].mean()), R(comp[hv == 1].mean())],
        'honest_cross_half': [{'selected_on_half': h, 'selected': s, 'held_half_n': n, 'held_half_delta': R(v)}
                              for h, (n, v, s) in enumerate(held)],
        'G_honest': R(G), 'bars': bars, 'composite_expected': R(comp.mean() * 0.2 * 0.886)}
    res['verdict'] = ('VOID' if not gates_ok else
                      'NO_BUILD' if not incl else
                      'BUILD_RECOMMENDED' if (comp.mean() >= bars['recommended'] and G > 0) else
                      'BUILD_OPTION' if (comp.mean() >= bars['option'] and G > 0) else
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
