"""STEP235 (Claude): LATE FUSION of an absolute "event-ness" probability into the production stage-2 listing
probability, measured on the cached 162-query ENS4 evidence audit.

Idea (lead session, 2026-09-17). Positive pairs carry about 13 injected hands but only the first five (type,
then time) are listed, so production stage 2 -- trained within positive pairs on listed (1) vs unlisted (0)
top-20 rows -- labels censored injected hands 0, and the typed decoder then multiplies that listing
probability by P(fewer than five listed before), double counting time. A cleaner quantity is EVENT-NESS:
P(hand is an injected event), learned against CLEAN negatives (all hands of the 1,488 confirmed non-target
pairs). STEP232 built that classifier out of fold by table on the 163 raw per-hand features
(experiments/step232_event_detector_20260917/hand_oof.parquet).

This runner does NOT retrain anything of the production chain. It reproduces the current audit reference
(.727099 = STEP204 + CI6 + DTs16 + DTv56 + CIv8 + DTod, control .692274) from the ENS4 network cache exactly
as step221/step229/step230 do, and then applies, per registered arm, a single geometric late fusion to the
41-view-averaged stage-2 probability of ONE family, before the typed decoder:

    odds(p) = p / (1 - p);   p_evt = odds(p_family) * F / (1 + odds(p_family) * F)
    p'      = p_stage2 ** (1 - w) * p_evt ** w                (rows of the arm's family only)

Every arm shares the same network fits and the same stage-2 fits, so the arm-vs-reference deltas carry no
measurement noise at all (OBS 19.154); the fusion is a pure post-processing of the reference scores.

Usage: python step235_C_event_fusion_audit.py REGISTRATION_PATH REGSHA
Audit only. Development phase only (asserted). Reads no evaluation row. Builds no file. Never uploads.
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
from sklearn.model_selection import GroupKFold

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug import step203_C_ens4_ci_stack as S203
from evidence_aug import step221_C_family_lists_cached_auditrows as S221

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
CACHE = CV / 'cache_ens4_networks'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
MEMBERS = S203.MEMBERS
FAMCOL = {f: 'p_' + f for f in FAMS}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_shift(p, F):
    """odds x F, back to a probability. p is clipped away from 0 and 1 first."""
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * float(F)
    return o / (1 + o)


def main():
    regp, reg_sha = Path(sys.argv[1]), sys.argv[2]
    assert sha(regp) == reg_sha, 'registration hash mismatch'
    reg = json.loads(regp.read_text(encoding='utf-8'))
    assert sha(__file__) == reg['code']['runner_sha256'], 'runner changed after registration'
    OUT = regp.parent
    tag = reg['step'].lower()
    dest = OUT / f'results_{tag}.json'
    assert not dest.exists(), 'preserve a completed read'
    REF, CTRL = reg['reference'], reg['control_gate']['arm']
    STAGE2 = {a: (v['members'], v['lists']) for a, v in reg['stage2_arms'].items()}
    assert STAGE2[REF][0] == list(MEMBERS)
    from evidence_aug.step165b_C_lp_v1_level import build_views
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    # ---------------------------------------------------------------- stage 1: data, features, phase
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
    used = sorted({m for ms, _ in STAGE2.values() for m in ms}, key=list(MEMBERS).index)
    nq = len(Q)
    qpairs = Q['pair_id'].to_numpy()
    folds_q = [(np.flatnonzero(np.isin(qpairs, upairs[tr])), np.flatnonzero(np.isin(qpairs, upairs[te])))
               for tr, te in outer]

    # ---------------------------------------------------------------- network outputs: cache only
    EA, EB, cache_rec = {}, {}, {}
    for m in used:
        jobs = S221.member_jobs(m, outer, rowmask, ptab)
        key = S221.cache_key(m, arrays, jobs)
        cp = CACHE / f'{m}_{key[:16]}.npz'
        assert cp.exists(), 'network cache miss for %s (%s); this runner never refits networks' % (m, key[:16])
        z = np.load(cp)
        assert str(z['key']) == key
        EA[m] = {k: z[f'ea{k}'] for k in range(5)}
        EB[m] = z['eb']
        cache_rec[m] = {'key': key, 'loaded': cp.name}
    print('X %s  %d members from cache' % (X.shape, len(used)), flush=True)

    # ---------------------------------------------------------------- stage 2 (identical to step221)
    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    LISTS = {}
    for nm, spec in reg['lists'].items():
        for f in FAMS:
            add = list(spec.get(f, []))
            cols = list(base[f]) + add
            assert len(set(cols)) == len(cols), (f, nm)
            LISTS[(f, nm)] = cols
    need = sorted({(m, f, lm[f]) for ms, lm in STAGE2.values() for m in ms for f in FAMS})
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    MOD = {}
    for m, f, nm in need:
        z = Q.select(LISTS[(f, nm)]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            ztr = np.column_stack([z[a], EA[m][k][isf[f][tr]]])
            MOD[(m, f, nm, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(ztr, y[a])
    print('stage-2 models fitted: %d  %ds' % (len(MOD), round(time.perf_counter() - t0)), flush=True)

    def score_part(v, m, f, nm):
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
        for arm, (ms, lm) in STAGE2.items():
            p = D['SS'].copy()
            for f in FAMS:
                rows = np.flatnonzero(isf[f])
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
    accA = [{a: np.zeros(len(d)) for a in STAGE2}, {a: np.zeros(len(d)) for a in STAGE2}]
    accS = [np.zeros(len(d)), np.zeros(len(d))]
    nv = 0
    for slot, (maker, seeds) in enumerate(((views, (1241, 2241)), (pads, (1242, 2242)))):
        for s in seeds:
            for v in maker(.5, 10, s):
                sc = score_all(v)
                for a in STAGE2:
                    accA[slot][a] += sc[a]
                accS[slot] += score_sstar(v)
                nv += 1
            print('views %d  %ds' % (nv, round(time.perf_counter() - t0)), flush=True)
    assert nv == 40, nv
    V1 = {a: (accA[0][a] + accA[1][a] + A0[a]) / (nv + 1) for a in STAGE2}
    V1S = (accS[0] + accS[1] + D['SS']) / (nv + 1)
    gate_v1 = float(np.abs(V1S - D['V1']).max())
    print('S* v1 replay max abs diff %s' % gate_v1, flush=True)

    # ---------------------------------------------------------------- event probabilities (STEP232 OOF)
    ev_rel = reg['event_file']['path']
    evp = CV / ev_rel
    assert sha(evp) == reg['event_file']['sha256'], 'event file changed after registration'
    oof = pl.read_parquet(evp)
    need_cols = K + ['table_id', 'label', 'behavior_family', 'fold'] + [FAMCOL[f] for f in FAMS] + ['p_any']
    ev = oof.select(need_cols)
    j = Q.select(K + ['table_id', 'behavior_family']).join(
        ev, on=K, how='left', validate='1:1', maintain_order='left', suffix='_ev')
    assert j.height == nq and j.select(K).equals(Q.select(K)), 'event join reordered or dropped audit rows'
    cover = {c: int(j[c].null_count()) for c in [FAMCOL[f] for f in FAMS] + ['p_any', 'fold', 'label']}
    assert sum(cover.values()) == 0, ('event coverage is not 100 %%', cover)
    assert (j['label'] == 1).all(), 'an audit row is not a labelled positive pair in the event file'
    assert (j['table_id'] == j['table_id_ev']).all(), 'table_id disagrees between the audit and the event file'
    assert (j['behavior_family'] == j['behavior_family_ev']).all(), 'family disagrees'
    # every audit pair sits on exactly one table and therefore in exactly one STEP232 fold
    chk = j.group_by('pair_id').agg(pl.col('table_id').n_unique().alias('t'), pl.col('fold').n_unique().alias('f'))
    assert chk['t'].max() == 1 and chk['f'].max() == 1, 'an audit pair spans tables or folds'
    fold_of_pair = dict(j.select('pair_id', 'fold').unique().iter_rows())
    PEVT = {f: j[FAMCOL[f]].to_numpy().astype(np.float64) for f in FAMS}

    # ---------------------------------------------------------------- arms
    QIX = D['QIX']
    p_ref_top = V1[REF][QIX] - 10.0
    ARMS = {}          # arm -> full-length score vector
    ARMFAM = {}
    ARMS[REF] = V1[REF]
    ARMS[CTRL] = V1[CTRL]
    diag = {}
    for a, spec in reg['fusion_arms'].items():
        f, w, F = spec['family'], float(spec['w']), float(spec['F'])
        pe = prior_shift(PEVT[f], F)
        m = isf[f]
        newp = p_ref_top.copy()
        newp[m] = (np.clip(p_ref_top[m], 1e-9, 1.0) ** (1 - w)) * (np.clip(pe[m], 1e-9, 1 - 1e-9) ** w)
        s = V1[REF].copy()
        s[QIX] = 10.0 + newp
        ARMS[a] = s
        ARMFAM[a] = f
        diag[a] = {'family': f, 'w': w, 'F': F, 'rows_changed': int(m.sum()),
                   'mean_p_stage2_on_family': round(float(p_ref_top[m].mean()), 6),
                   'mean_p_evt_on_family': round(float(pe[m].mean()), 6),
                   'mean_p_fused_on_family': round(float(newp[m].mean()), 6),
                   'spearman_p_stage2_vs_p_fused_within_family': round(float(np.corrcoef(
                       np.argsort(np.argsort(p_ref_top[m])), np.argsort(np.argsort(newp[m])))[0, 1]), 6)}

    # ---------------------------------------------------------------- decode and score
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
    for a, s in ARMS.items():
        _, mm = summarize(key.with_columns(pl.Series('score', dec(s))), truths, fams)
        M[a] = np.array([mm[q] for q in qids])

    def report(a, ref):
        dl = M[a] - M[ref]
        return {'map5': R(M[a].mean()), 'vs': ref, 'decoded_delta': R(dl.mean()),
                'half0': R(dl[hv == 0].mean()), 'half1': R(dl[hv == 1].mean()),
                'positive_on_both_halves': bool(dl[hv == 0].mean() > 0 and dl[hv == 1].mean() > 0),
                'per_family': {f: R(dl[fq == f].mean()) for f in FAMS},
                'per_family_halves': {f: [R(dl[(fq == f) & (hv == h)].mean()) for h in (0, 1)] for f in FAMS},
                'per_family_up_down': {f: [int((dl[fq == f] > 0).sum()), int((dl[fq == f] < 0).sum())] for f in FAMS},
                'queries_up_down': [int((dl > 0).sum()), int((dl < 0).sum())],
                'query_se': R(dl.std(ddof=1) / np.sqrt(len(dl)))}

    res = {'registration_sha256': reg_sha, 'step': reg['step'], 'level': 'v1, 41 views',
           'n_queries': len(qids), 'n_views': nv + 1, 'network_fits_run': 0, 'network_cache': cache_rec,
           'event_coverage': {'audit_rows': nq, 'matched': nq, 'nulls': cover,
                              'pairs': int(len(fold_of_pair)),
                              'folds_used': sorted(set(fold_of_pair.values())),
                              'one_fold_per_pair': True},
           'fusion_diagnostics': diag, 'arms': {}}
    res['arms'][f'{REF}_vs_{CTRL}'] = report(REF, CTRL)
    for a in reg['fusion_arms']:
        res['arms'][a] = report(a, REF)

    # ---------------------------------------------------------------- gates
    iso = {}
    for a, f in ARMFAM.items():
        dl = M[a] - M[REF]
        iso[a] = {g: float(np.abs(dl[fq == g]).max()) for g in FAMS if g != f}
    gates = {
        'S_star_v1_replay_exact': {'max_abs': gate_v1, 'passed': gate_v1 == 0.0},
        'reference_reproduces': {'here': R(M[REF].mean()), **reg['reference_gate'],
                                 'exact': bool(R(M[REF].mean()) == reg['reference_gate']['expected']),
                                 'passed': bool(abs(M[REF].mean() - reg['reference_gate']['expected'])
                                                <= reg['reference_gate']['tolerance'])},
        'control_reproduces': {'here': R(M[CTRL].mean()), **reg['control_gate'],
                               'exact': bool(R(M[CTRL].mean()) == reg['control_gate']['expected']),
                               'passed': bool(abs(M[CTRL].mean() - reg['control_gate']['expected'])
                                              <= reg['control_gate']['tolerance'])},
        'untouched_families_exactly_zero': {'max_abs': iso,
                                            'passed': all(v == 0.0 for g in iso.values() for v in g.values())},
        'event_coverage_100pct': {'nulls': cover, 'passed': sum(cover.values()) == 0},
        'event_folds_are_by_table': {'pairs': int(len(fold_of_pair)), 'passed': True}}
    res['gates'] = gates
    gates_ok = all(v['passed'] for v in gates.values())

    # ---------------------------------------------------------------- decision (identical rule to step221)
    CHANGE = ARMFAM
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
    prim = reg['primary_arm']
    pd_ = fam_delta[prim]
    res['decision'] = {
        'primary_arm': prim,
        'primary_delta': R(pd_.mean()),
        'primary_halves': [R(pd_[hv == 0].mean()), R(pd_[hv == 1].mean())],
        'primary_family_halves': [R(pd_[(hv == h) & (fq == CHANGE[prim])].mean()) for h in (0, 1)],
        'primary_passes_eased_rule': bool(include(prim, hv == 0) and include(prim, hv == 1)
                                          and pd_.mean() > 0),
        'included_changes': incl, 'composition_delta_vs_reference': R(comp.mean()),
        'composition_halves': [R(comp[hv == 0].mean()), R(comp[hv == 1].mean())],
        'honest_cross_half': [{'selected_on_half': h, 'selected': s, 'held_half_n': n, 'held_half_delta': R(v)}
                              for h, (n, v, s) in enumerate(held)],
        'G_honest': R(G), 'bars': bars,
        'composite_expected': R(comp.mean() * 0.2 * reg['transfer_ratio'])}
    res['verdict'] = ('VOID' if not gates_ok else
                      'NO_BUILD' if not incl else
                      'BUILD_RECOMMENDED' if (comp.mean() >= bars['recommended'] and both and G > 0) else
                      'BUILD_OPTION' if (comp.mean() >= bars['option'] and both and G > 0) else
                      'NO_BUILD')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    d.select(K).with_columns([pl.Series('v1_' + a, ARMS[a]) for a in ARMS]).write_parquet(
        OUT / f'audit_scores_{tag}.parquet')
    res['scores_sha256'] = sha(OUT / f'audit_scores_{tag}.parquet')
    pl.DataFrame({'pair_id': qids, 'half': hv, 'family': fq, **{a: M[a] for a in ARMS}}).write_parquet(
        OUT / f'per_query_{tag}.parquet')
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('gates', 'decision', 'verdict', 'seconds')}, indent=1), flush=True)
    print('"verdict": "%s"' % res['verdict'], flush=True)


if __name__ == '__main__':
    main()
