"""STEP246 (Claude): the STEP237 invisible-event block as extra STAGE-2 columns per family, audited on the
cached 162-query ENS4 evidence audit exactly the way STEP221 audited Codex's od_ block.

STEP237 read 360 rendered hands and built 25 per-hand pair features aimed at the injected events the 163
production raw features cannot see (experiments/step237_invisible_events_20260917/REPORT.md section 4).
Against a clean-negative event classifier the block moved coordinated_isolation AP .5225 -> .5887 and
TPR@1e-4 .2326 -> .3326, and cut the directed_transfer invisible share .1448 -> .1159.  None of that was
measured inside production stage 2, which already carries rs_, eq_, cs_, seq_, od_, DTs16, CI causal-fold,
CIv8, DTv56 and DTod; the risk is dilution (OBS 19.172, 19.175).

What this runner does, and what it deliberately does not do:
  * networks: CACHE ONLY.  It asserts a cache hit for every member and never fits a network, so every
    arm-vs-reference delta is stage-2-only and carries exactly zero network noise (OBS 19.154).
  * stage 2: identical to step221 -- per family, per outer fold, LightGBM on (list columns | member network
    columns), scored out of fold, averaged over the four ENS4 members, 41 v1 views, decoded once.
  * view averaging is accumulated PER (member, family, list) rather than per arm.  Averaging over members and
    averaging over views commute, so any arm whose per-family lists are already in the measured set can be
    assembled after the fact at no extra cost.  That is what makes the ALL composition -- whose membership is
    only known after the per-family arms are read -- a genuinely measured arm rather than a sum of parts.
  * ALL is chosen by the fixed rule in `included()` below, never by eye, and is gated against the sum of its
    parts.

Reported per arm: decoded MAP@5 delta vs REF overall / per family / per half / per family-half, queries up and
down, a paired bootstrap 90 % interval over the 162 queries, whether the point landed inside the registered
80 % band, and both G_honest figures (the shared STEP221 rule over every changed arm, and G restricted to the
arms whose registered point prediction is >= 0).  If a selection half admits an arm registered as an expected
negative, the runner says so explicitly -- that is what dragged G below zero in STEP235.

Usage: python step246_C_step237_block_audit.py REGISTRATION_PATH REGSHA
Audit only.  Development phase only (asserted).  Reads no evaluation row.  Builds no submission.  Never uploads.
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
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = lambda x: round(float(x), 6)


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
    REF = reg['reference']
    CTRL = reg['control_gate']['arm']
    CHANGE = dict(reg['changes'])
    ROLES = dict(reg['arm_roles'])
    PRIM = reg['primary_arm']
    PRED = reg['prediction']
    NEGATIVE = set(reg['prediction']['expected_negative_arms'])
    CAND = [a for a in CHANGE if a not in NEGATIVE]
    assert ARMS[REF][0] == list(MEMBERS)
    assert PRIM in CHANGE and PRIM not in NEGATIVE
    from evidence_aug.step165b_C_lp_v1_level import build_views
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ data, extra columns, phase assert
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
    used = sorted({m for ms, _ in ARMS.values() for m in ms}, key=list(MEMBERS).index)
    nq = len(Q)
    qpairs = Q['pair_id'].to_numpy()
    folds_q = [(np.flatnonzero(np.isin(qpairs, upairs[tr])), np.flatnonzero(np.isin(qpairs, upairs[te])))
               for tr, te in outer]

    # ------------------------------------------------------------------ network outputs: cache only
    EA, EB, cache_rec = {}, {}, {}
    for m in used:
        jobs = S221.member_jobs(m, outer, rowmask, ptab)
        key = S221.cache_key(m, arrays, jobs)
        cp = CACHE / f'{m}_{key[:16]}.npz'
        assert cp.exists(), 'network cache miss for %s (%s); this runner never fits networks' % (m, key[:16])
        z = np.load(cp)
        assert str(z['key']) == key
        EA[m] = {k: z[f'ea{k}'] for k in range(5)}
        EB[m] = z['eb']
        cache_rec[m] = {'key': key, 'loaded': cp.name}
    print('X %s  %d members from cache, 0 network fits' % (X.shape, len(used)), flush=True)

    # ------------------------------------------------------------------ stage 2 (identical to step221)
    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    LISTS = {}
    for nm, spec in reg['lists'].items():
        for f in FAMS:
            cols = list(base[f]) + list(spec.get(f, []))
            assert len(set(cols)) == len(cols), (f, nm)
            LISTS[(f, nm)] = cols
    need = sorted({(m, f, lm[f]) for ms, lm in ARMS.values() for m in ms for f in FAMS})
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
        z = v.select(LISTS[(f, nm)]).to_numpy().astype(float)
        p = np.full(nq, np.nan)
        for k, (tr, te) in enumerate(folds_q):
            b = te[isf[f][te]]
            if len(b):
                p[b] = MOD[(m, f, nm, k)].predict_proba(np.column_stack([z[b], EB[m][b]]))[:, 1] + 10
        return p

    def parts_of(v):
        out = {}
        for keyc in need:
            p = score_part(v, *keyc)
            assert np.isfinite(p[isf[keyc[1]]]).all(), keyc
            out[keyc] = p
        return out

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

    # ------------------------------------------------------------------ one 41-view pass, accumulated per part
    views, pads = build_views(D)
    PACC = {keyc: np.zeros(nq) for keyc in need}
    for keyc, p in parts_of(Q).items():
        PACC[keyc] += np.nan_to_num(p)
    accS = np.zeros(len(d))
    nv = 0
    for maker, seeds in ((views, (1241, 2241)), (pads, (1242, 2242))):
        for s in seeds:
            for v in maker(.5, 10, s):
                for keyc, p in parts_of(v).items():
                    PACC[keyc] += np.nan_to_num(p)
                accS += score_sstar(v)
                nv += 1
            print('views %d  %ds' % (nv, round(time.perf_counter() - t0)), flush=True)
    assert nv == 40, nv
    PACC = {keyc: p / (nv + 1) for keyc, p in PACC.items()}
    V1S = (accS + D['SS']) / (nv + 1)
    gate_v1 = float(np.abs(V1S - D['V1']).max())
    print('S* v1 replay max abs diff %s' % gate_v1, flush=True)

    def assemble(ms, lm):
        p = D['SS'].copy()
        for f in FAMS:
            rows = np.flatnonzero(isf[f])
            p[D['QIX'][rows]] = np.mean([PACC[(m, f, lm[f])][rows] for m in ms], axis=0)
        return p

    V1 = {a: assemble(*ARMS[a]) for a in ARMS}

    # ------------------------------------------------------------------ decode and score
    dec = D['dec']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    keyd = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabq[q]) for q in qids])
    fq = np.array([fams[q] for q in qids])
    nqy = len(qids)

    def mapvec(score):
        _, mm = summarize(keyd.with_columns(pl.Series('score', dec(score))), truths, fams)
        return np.array([mm[q] for q in qids])

    M = {a: mapvec(V1[a]) for a in ARMS}

    # ------------------------------------------------------------------ the ALL composition, by the fixed rule
    fam_delta = {a: M[a] - M[REF] for a in CHANGE}

    def included(a, mask):
        dl = fam_delta[a][mask & (fq == CHANGE[a])]
        return bool(len(dl) and dl.mean() > 0)

    def compose(pool):
        """the fixed rule: own family positive on BOTH halves and overall positive; best arm per family."""
        ok = [a for a in pool if all(included(a, hv == h) for h in (0, 1)) and fam_delta[a].mean() > 0]
        byfam = {}
        for a in ok:
            f = CHANGE[a]
            if f not in byfam or fam_delta[a].mean() > fam_delta[byfam[f]].mean():
                byfam[f] = a
        return sorted(byfam.values())

    incl = compose(list(CHANGE))
    all_lists = {f: 'ref' for f in FAMS}
    for a in incl:
        all_lists[CHANGE[a]] = ARMS[a][1][CHANGE[a]]
    if incl:
        ARMS['ALL'] = (list(MEMBERS), all_lists)
        V1['ALL'] = assemble(*ARMS['ALL'])
        M['ALL'] = mapvec(V1['ALL'])
    comp_parts = sum((fam_delta[a] for a in incl), np.zeros(nqy))

    # ------------------------------------------------------------------ bootstrap over the 162 queries
    bs = reg['bootstrap']
    rng = np.random.default_rng(bs['seed'])
    BIDX = rng.integers(0, nqy, size=(int(bs['n_boot']), nqy))
    lo_q, hi_q = (1 - bs['level']) / 2 * 100, (1 + bs['level']) / 2 * 100

    def boot(dl):
        m = dl[BIDX].mean(axis=1)
        return [R(np.percentile(m, lo_q)), R(np.percentile(m, hi_q)), R(float((m > 0).mean()))]

    def report(a, ref):
        dl = M[a] - M[ref]
        b = boot(dl)
        out = {'map5': R(M[a].mean()), 'vs': ref, 'decoded_delta': R(dl.mean()),
               'half0': R(dl[hv == 0].mean()), 'half1': R(dl[hv == 1].mean()),
               'positive_on_both_halves': bool(dl[hv == 0].mean() > 0 and dl[hv == 1].mean() > 0),
               'per_family': {f: R(dl[fq == f].mean()) for f in FAMS},
               'per_family_halves': {f: [R(dl[(fq == f) & (hv == h)].mean()) for h in (0, 1)] for f in FAMS},
               'per_family_up_down': {f: [int((dl[fq == f] > 0).sum()), int((dl[fq == f] < 0).sum())] for f in FAMS},
               'queries_up_down': [int((dl > 0).sum()), int((dl < 0).sum())],
               'query_se': R(dl.std(ddof=1) / np.sqrt(len(dl))),
               'boot90': b[:2], 'boot_P_positive': b[2]}
        if a in PRED and ref == REF:
            band = PRED[a]['band80']
            out['registered_point'] = PRED[a]['overall']
            out['registered_band80'] = band
            out['inside_band'] = bool(band[0] <= dl.mean() <= band[1])
            out['role'] = ROLES.get(a, 'composition')
        return out

    res = {'registration_sha256': reg_sha, 'step': reg['step'], 'level': 'v1, 41 views',
           'n_queries': nqy, 'n_views': nv + 1, 'network_fits_run': 0, 'network_cache': cache_rec,
           'family_query_counts': {f: int((fq == f).sum()) for f in FAMS},
           'half_query_counts': [int((hv == 0).sum()), int((hv == 1).sum())],
           'stage2_models_fitted': len(MOD),
           'list_sizes': {f'{f}|{nm}': len(LISTS[(f, nm)]) for (f, nm) in sorted(LISTS)},
           'arms': {}}
    res['arms'][f'{REF}_vs_{CTRL}'] = report(REF, CTRL)
    for a in list(CHANGE) + (['ALL'] if incl else []):
        res['arms'][a] = report(a, REF)

    # ------------------------------------------------------------------ gates
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
        'control_reproduces': {'here': R(M[CTRL].mean()), **reg['control_gate'],
                               'exact': bool(R(M[CTRL].mean()) == reg['control_gate']['expected']),
                               'passed': bool(abs(M[CTRL].mean() - reg['control_gate']['expected'])
                                              <= reg['control_gate']['tolerance'])},
        'untouched_families_exactly_zero': {'max_abs': iso,
                                            'passed': all(v == 0.0 for g in iso.values() for v in g.values())}}
    if incl:
        add = M['ALL'] - M[REF] - comp_parts
        gates['all_equals_sum_of_parts'] = {'max_abs': float(np.abs(add).max()),
                                            'members': incl, 'lists': all_lists,
                                            'passed': bool(np.abs(add).max() < 1e-12)}
    res['gates'] = gates
    gates_ok = all(v['passed'] for v in gates.values())

    # ------------------------------------------------------------------ G_honest, shared rule and candidates only
    def cross_half(pool):
        held = []
        for h in (0, 1):
            pick = {}
            for a in pool:
                if included(a, hv == h):
                    f = CHANGE[a]
                    m_h = fam_delta[a][(hv == h) & (fq == f)].mean()
                    if f not in pick or m_h > pick[f][1]:
                        pick[f] = (a, m_h)
            sel = sorted(v[0] for v in pick.values())
            dh = sum((fam_delta[a] for a in sel), np.zeros(nqy))
            held.append({'selected_on_half': h, 'selected': sel, 'held_half_n': int((hv != h).sum()),
                         'held_half_delta': R(dh[hv != h].mean()),
                         'admits_expected_negative': sorted(set(sel) & NEGATIVE)})
        G = sum(x['held_half_n'] * x['held_half_delta'] for x in held) / sum(x['held_half_n'] for x in held)
        return held, R(G)

    held_all, G_all = cross_half(list(CHANGE))
    held_cand, G_cand = cross_half(CAND)
    comp = (M['ALL'] - M[REF]) if incl else np.zeros(nqy)
    both = bool(comp[hv == 0].mean() > 0 and comp[hv == 1].mean() > 0)
    bars = reg['bars']
    pd_ = fam_delta[PRIM]
    res['decision'] = {
        'primary_arm': PRIM,
        'primary_delta': R(pd_.mean()),
        'primary_halves': [R(pd_[hv == 0].mean()), R(pd_[hv == 1].mean())],
        'primary_family_halves': [R(pd_[(hv == h) & (fq == CHANGE[PRIM])].mean()) for h in (0, 1)],
        'primary_passes_eased_rule': bool(PRIM in incl),
        'included_changes': incl, 'all_lists': all_lists,
        'composition_delta_vs_reference': R(comp.mean()),
        'composition_halves': [R(comp[hv == 0].mean()), R(comp[hv == 1].mean())],
        'composition_boot90': boot(comp)[:2] if incl else None,
        'honest_cross_half': held_all, 'G_honest': G_all,
        'honest_cross_half_candidates_only': held_cand, 'G_honest_candidates_only': G_cand,
        'expected_negative_admitted_by_a_selection_half':
            sorted({a for x in held_all for a in x['admits_expected_negative']}),
        'bars': bars, 'carried_gain_bar': reg['carried_gain_bar'],
        'composite_expected': R(comp.mean() * 0.2 * reg['transfer_ratio'])}
    res['verdict'] = ('VOID' if not gates_ok else
                      'NO_BUILD' if not incl else
                      'BUILD_RECOMMENDED' if (comp.mean() >= bars['recommended'] and both and G_all > 0) else
                      'BUILD_OPTION' if (comp.mean() >= bars['option'] and both and G_all > 0) else
                      'CARRY' if (comp.mean() >= reg['carried_gain_bar'] and both) else
                      'NO_BUILD')
    res['stage3_allowed'] = bool(gates_ok and incl and both
                                 and (PRIM in incl or len(incl) > 0)
                                 and comp.mean() >= reg['carried_gain_bar'])
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    d.select(K).with_columns([pl.Series('v1_' + a, V1[a]) for a in ARMS]).write_parquet(
        OUT / f'audit_scores_{tag}.parquet')
    res['scores_sha256'] = sha(OUT / f'audit_scores_{tag}.parquet')
    pl.DataFrame({'pair_id': qids, 'half': hv, 'family': fq, **{a: M[a] for a in ARMS}}).write_parquet(
        OUT / f'per_query_{tag}.parquet')
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('gates', 'decision', 'verdict', 'stage3_allowed', 'seconds')},
                     indent=1), flush=True)
    print('"verdict": "%s"' % res['verdict'], flush=True)


if __name__ == '__main__':
    main()
