"""STEP249 (Claude): four levers on the cached 162-query ENS4 evidence audit, in one registration.

  1. DTdir   -- a better, label-free donor DIRECTION for every role-dependent directed_transfer column.
                STEP244 found the donor is fixed per pair (133/133) and that production's direction (the sign
                of the pool's summed chip flow, step206c line 71) is right for .940 of DT pairs on full
                histories and only .847 at 40 % of the hands, while an event rule -- the donor is the member
                who more often FOLDS FACING THE PARTNER'S AGGRESSION WITH EQUITY > .5 -- is right for
                .985 / .909.  Evaluation histories are two thirds as long, so production is wrong for about
                one DT target in ten there.
                Mechanics: step206c, step208, step211 and step215 all assign the role by
                `donor, recip = (p1, p2) if sign > 0 else (p2, p1)`, so a sign flip is EXACTLY a swap of the
                dt_donor_* / dt_recip_* column pair.  The arm recomputes the sign PER POOL VIEW -- on the
                training frame (the full pool) and on each of the 40 views alike, so the DT stage-2 model is
                trained and scored under the same estimator -- and swaps the 40 role pairs on the rows of
                every pair whose sign changes.
                DTdirchip is the registered control: production's own chip-flow rule, also recomputed per
                view.  It separates "recompute the sign on the pool you actually see" from "use a better
                estimator".
  2. A03      -- stage-2 capacity (300 trees, learning rate .025, everything else C00), per family and as the
                two-cell mean C00+A03 per family.  STEP229 closed column SAMPLING on the enlarged lists;
                capacity was never re-tested on them.
  3. DTe8     -- STEP215's four partner-call equity columns x role, re-measured on the current reference,
                and DTe8dir, the same columns under the DTdir direction.
  4. DTfp3 / DTfp1 -- the fold-shape columns of the STEP237 block that directed_transfer's trees ranked
                highest in STEP246.

Cache only: every ENS4 network output is loaded from cache_ens4_networks and asserted present; zero network
fits, so every arm-vs-REF delta is stage-2-only and carries no network measurement noise (OBS 19.154).
The S* view scores are accumulated in TWO slot accumulators exactly as step221 does, so the replay gate reads
exactly 0.0 (STEP246 used one accumulator and tripped the gate at 7e-10).

Development phase only (asserted).  Reads no evaluation row.  Builds no production file.  Never uploads.
Usage: python step249_C_direction_capacity_audit.py REGISTRATION_PATH REGSHA
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
DT = 'directed_transfer'
MEMBERS = S203.MEMBERS
POOLF4 = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof_rank']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = lambda x: round(float(x), 6)


# --------------------------------------------------------------------------- pool views, reconstructed
def view_pools(d, u, m, seed, pad):
    """The pool membership of build_views' views(u, m, seed) / pads(u, m, seed), as a per-view row COUNT
    multiplicity over d's rows.  build_views draws exactly one random vector per copy with the same
    generator, keeps ~((r < u) & ~top0) for a view and appends the picked rows again for a pad."""
    top0 = (d['oof_rank'].to_numpy() <= 20)
    n = d.height
    out = []
    for c in range(m):
        r = np.random.default_rng([seed, c]).random(n)
        pick = (r < u) & ~top0
        out.append((np.ones(n) + pick.astype(float)) if pad else (~pick).astype(float))
    return out


def signs_for(mult, ev1, ev2, flow, pair_ix, npair):
    """Per pair: event counts and chip flow summed over the view's pool (multiplicity-weighted)."""
    e1 = np.bincount(pair_ix, weights=mult * ev1, minlength=npair)
    e2 = np.bincount(pair_ix, weights=mult * ev2, minlength=npair)
    fl = np.bincount(pair_ix, weights=mult * flow, minlength=npair)
    chip = np.where(fl >= 0, 1.0, -1.0)
    evt = np.where(e1 > e2, 1.0, np.where(e2 > e1, -1.0, chip))
    return {'chip': chip, 'evt': evt}, e1, e2, fl


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
    PRIM = reg['primary_arm']
    PRED = reg['prediction']
    NEGATIVE = set(PRED['expected_negative_arms'])
    CHANGE = dict(reg['changes'])
    ROLES = dict(reg['arm_roles'])
    SWAP = [tuple(p) for p in reg['role_swap_pairs']]
    from evidence_aug.step165b_C_lp_v1_level import build_views
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    # ---------------------------------------------------------------- data, extra columns, phase
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
    for a, b in SWAP:
        assert a in D['Q'].columns and b in D['Q'].columns, (a, b)

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
    used = sorted({m for v in reg['arms'].values() for m in v['members']}, key=list(MEMBERS).index)
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
        assert cp.exists(), 'network cache miss for %s (%s); this runner never fits networks' % (m, key[:16])
        z = np.load(cp)
        assert str(z['key']) == key
        EA[m] = {k: z[f'ea{k}'] for k in range(5)}
        EB[m] = z['eb']
        cache_rec[m] = {'key': key, 'loaded': cp.name}
    print('X %s  %d members from cache, 0 network fits' % (X.shape, len(used)), flush=True)

    # ---------------------------------------------------------------- the donor-direction estimators
    fp = reg['direction']['fold_facts']
    ffp = CV / fp['path']
    assert sha(ffp) == fp['sha256'], 'fold-fact cache changed after registration'
    M = pl.read_parquet(ffp).select(K + ['n1_bb', 'n2_bb', 'ftp_eq_1', 'ftp_eq_2'])
    dk = d.select(K)
    j = dk.join(M, on=K, how='left', validate='1:1', maintain_order='left')
    assert j.select(K).equals(dk), 'fold-fact join reordered the pool'
    dtpairs = Q.filter(pl.col('behavior_family') == DT)['pair_id'].unique().to_numpy()
    isdt_pool = np.isin(pairs, dtpairs)
    cov = int(j.filter(pl.Series(isdt_pool)).select('n1_bb').null_count().item())
    assert cov == 0, 'the fold-fact cache does not cover every directed_transfer pool row'
    thr = float(reg['direction']['equity_threshold'])
    ev1 = (j['ftp_eq_1'].fill_null(-1).to_numpy() > thr).astype(float) * isdt_pool
    ev2 = (j['ftp_eq_2'].fill_null(-1).to_numpy() > thr).astype(float) * isdt_pool
    flow = ((j['n2_bb'].fill_null(0).to_numpy() - j['n1_bb'].fill_null(0).to_numpy()) / 2.0) * isdt_pool
    pcodes, pair_ix = np.unique(pairs, return_inverse=True)
    npair = len(pcodes)
    rp = CV / reg['direction']['production_sign']['path']
    assert sha(rp) == reg['direction']['production_sign']['sha256'], 'production role file changed'
    ps = (pl.read_parquet(rp).select('pair_id', 'dt_sign').unique()
          .join(pl.DataFrame({'pair_id': pcodes}), on='pair_id', how='semi'))
    assert ps.height == npair, 'production sign is not one value per audit pair'
    prod_sign = ps.sort('pair_id')['dt_sign'].to_numpy().astype(float)
    assert (np.sort(ps['pair_id'].to_numpy()) == pcodes).all()

    # the unviewed frame first: the chip rule must reproduce production's own sign exactly
    full = np.ones(d.height)
    S0, _, _, _ = signs_for(full, ev1, ev2, flow, pair_ix, npair)
    dtmask = np.isin(pcodes, dtpairs)
    chip_gate = int((S0['chip'][dtmask] != prod_sign[dtmask]).sum())

    # every view's pool multiplicity, in the exact order the scoring loop will walk them
    VIEW_MULT = [full]
    for pad, seeds in ((False, (1241, 2241)), (True, (1242, 2242))):
        for s in seeds:
            VIEW_MULT += view_pools(d, .5, 10, s, pad)
    assert len(VIEW_MULT) == 41
    SIGNS, EVSTAT = [], []
    for mult in VIEW_MULT:
        s, e1, e2, fl = signs_for(mult, ev1, ev2, flow, pair_ix, npair)
        SIGNS.append(s)
        EVSTAT.append((e1, e2, fl))
    FLIP = {}
    for i, s in enumerate(SIGNS):
        for fr in ('chip', 'evt'):
            fp_ = (s[fr] != prod_sign)
            FLIP[(fr, i)] = fp_[np.searchsorted(pcodes, qpairs)]
    pool_sizes = [np.bincount(pair_ix, weights=mult, minlength=npair) for mult in VIEW_MULT]

    def apply_frame(v, frame, i):
        if frame == 'prod':
            return v
        f = FLIP[(frame, i)]
        if not f.any():
            return v
        sel = pl.Series('__flip', f)
        return v.with_columns([e for a, b in SWAP for e in (
            pl.when(sel).then(pl.col(b)).otherwise(pl.col(a)).alias(a),
            pl.when(sel).then(pl.col(a)).otherwise(pl.col(b)).alias(b))])

    # ---------------------------------------------------------------- stage 2
    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    LISTS = {}
    for nm, spec in reg['lists'].items():
        for f in FAMS:
            cols = list(base[f]) + list(spec.get(f, []))
            assert len(set(cols)) == len(cols), (f, nm)
            LISTS[(f, nm)] = cols
    CFGS = {'C00': dict(ac.CLF)}
    for nm, over in reg['configs'].items():
        CFGS[nm] = dict(ac.CLF, **over)
    assert CFGS['C00'] == dict(ac.CLF), 'C00 must be the incumbent stage-2 configuration'

    def spec_of(arm):
        v = reg['arms'][arm]
        return (v['members'],
                {f: v.get('lists', {}).get(f, 'ref') for f in FAMS},
                {f: list(v.get('cfgs', {}).get(f, ['C00'])) for f in FAMS},
                {f: v.get('frames', {}).get(f, 'prod') for f in FAMS})

    def parts_needed(arms):
        need = set()
        for a in arms:
            ms, lm, cm, fm = spec_of(a)
            for m in ms:
                for f in FAMS:
                    for c in cm[f]:
                        need.add((m, f, lm[f], c, fm[f]))
        return sorted(need)

    MOD = {}

    def fit_parts(need):
        Qf = {fr: apply_frame(Q, fr, 0) for fr in {k[4] for k in need}}
        for m, f, nm, c, fr in need:
            if (m, f, nm, c, fr, 0) in MOD:
                continue
            z = Qf[fr].select(LISTS[(f, nm)]).to_numpy().astype(float)
            isf_f = (Q['behavior_family'] == f).to_numpy()
            for k, (tr, te) in enumerate(folds_q):
                a = tr[isf_f[tr]]
                MOD[(m, f, nm, c, fr, k)] = lgb.LGBMClassifier(**CFGS[c], random_state=11).fit(
                    np.column_stack([z[a], EA[m][k][isf_f[tr]]]), y[a])

    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}

    def score_parts(v, i, need):
        vf = {fr: apply_frame(v, fr, i) for fr in {k[4] for k in need}}
        out = {}
        for keyc in need:
            m, f, nm, c, fr = keyc
            z = vf[fr].select(LISTS[(f, nm)]).to_numpy().astype(float)
            p = np.full(nq, np.nan)
            for k, (tr, te) in enumerate(folds_q):
                b = te[isf[f][te]]
                if len(b):
                    p[b] = MOD[(m, f, nm, c, fr, k)].predict_proba(
                        np.column_stack([z[b], EB[m][b]]))[:, 1] + 10
            assert np.isfinite(p[isf[f]]).all(), keyc
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

    # ---------------------------------------------------------------- one 41-view pass
    PACC = {}
    accS = [np.zeros(len(d)), np.zeros(len(d))]          # two slots, exactly as step221
    nhgate = []

    def view_pass(need, first):
        nonlocal nv
        for keyc in need:
            PACC.setdefault(keyc, [np.zeros(nq), np.zeros(nq), None])
        for keyc, p in score_parts(Q, 0, need).items():
            PACC[keyc][2] = np.nan_to_num(p)
        i = 0
        nv = 0
        views, pads = build_views(D)
        for slot, (maker, seeds) in enumerate(((views, (1241, 2241)), (pads, (1242, 2242)))):
            for s in seeds:
                for v in maker(.5, 10, s):
                    i += 1
                    if first:
                        got = dict(v.group_by('pair_id').agg(pl.col('n_hands').first()).iter_rows())
                        mine = {pcodes[k]: pool_sizes[i][k] for k in range(npair)}
                        nhgate.append(max(abs(got[p] - mine[p]) for p in got))
                        accS[slot] += score_sstar(v)
                    for keyc, p in score_parts(v, i, need).items():
                        PACC[keyc][slot] += np.nan_to_num(p)
                    nv += 1
                print('views %d  %ds' % (nv, round(time.perf_counter() - t0)), flush=True)
        assert nv == 40 and i == 40, (nv, i)

    nv = 0
    need1 = parts_needed([a for a in reg['arms']])
    fit_parts(need1)
    print('stage-2 models fitted: %d  %ds' % (len(MOD), round(time.perf_counter() - t0)), flush=True)
    view_pass(need1, True)
    gate_nh = float(max(nhgate))
    V1S = (accS[0] + accS[1] + D['SS']) / (nv + 1)
    gate_v1 = float(np.abs(V1S - D['V1']).max())
    print('S* v1 replay max abs diff %s   pool reconstruction max abs %s' % (gate_v1, gate_nh), flush=True)

    def assemble(ms, lm, cm, fm):
        p = D['SS'].copy()
        for f in FAMS:
            rows = np.flatnonzero(isf[f])
            cells = [PACC[(m, f, lm[f], c, fm[f])] for m in ms for c in cm[f]]
            p[D['QIX'][rows]] = np.mean([(a[0] + a[1] + a[2])[rows] / (nv + 1) for a in cells], axis=0)
        return p

    # ---------------------------------------------------------------- decode
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

    V1 = {a: assemble(*spec_of(a)) for a in reg['arms']}
    M_ = {a: mapvec(V1[a]) for a in reg['arms']}

    # ---------------------------------------------------------------- inclusion and compositions
    fam_delta = {a: M_[a] - M_[REF] for a in CHANGE}

    def included(a):
        dl = fam_delta[a]
        f = CHANGE[a]
        return bool(all(dl[(hv == h) & (fq == f)].mean() > 0 for h in (0, 1))
                    and all(dl[hv == h].mean() > 0 for h in (0, 1)) and dl.mean() > 0)

    incl = sorted(a for a in CHANGE if included(a))
    AXIS = {a: reg['arm_axis'][a] for a in CHANGE}
    comp_spec = {'lists': {f: 'ref' for f in FAMS}, 'cfgs': {f: ['C00'] for f in FAMS},
                 'frames': {f: 'prod' for f in FAMS}}
    picked = {}
    for f in FAMS:
        for ax in ('list', 'cfg', 'frame'):
            cand = [a for a in incl if CHANGE[a] == f and AXIS[a] == ax]
            if not cand:
                continue
            best = max(cand, key=lambda a: fam_delta[a].mean())
            picked[(f, ax)] = best
            _, lm, cm, fm = spec_of(best)
            comp_spec[{'list': 'lists', 'cfg': 'cfgs', 'frame': 'frames'}[ax]][f] = (
                lm[f] if ax == 'list' else cm[f] if ax == 'cfg' else fm[f])
    strict = {}
    for a in incl:
        f = CHANGE[a]
        if f not in strict or fam_delta[a].mean() > fam_delta[strict[f]].mean():
            strict[f] = a
    strict_arms = sorted(strict.values())

    EXTRA = {}
    if incl:
        EXTRA['COMPOSITION'] = (list(MEMBERS), comp_spec['lists'], comp_spec['cfgs'], comp_spec['frames'])
        s2 = {'lists': {f: 'ref' for f in FAMS}, 'cfgs': {f: ['C00'] for f in FAMS},
              'frames': {f: 'prod' for f in FAMS}}
        for a in strict_arms:
            _, lm, cm, fm = spec_of(a)
            f = CHANGE[a]
            s2['lists'][f], s2['cfgs'][f], s2['frames'][f] = lm[f], cm[f], fm[f]
        EXTRA['COMPOSITION_STRICT'] = (list(MEMBERS), s2['lists'], s2['cfgs'], s2['frames'])
    need2 = sorted({(m, f, lm[f], c, fm[f]) for ms, lm, cm, fm in EXTRA.values()
                    for m in ms for f in FAMS for c in cm[f]} - set(need1))
    if need2:
        print('composition needs %d new cells; second view pass' % len(need2), flush=True)
        fit_parts(need2)
        view_pass(need2, False)
    for a, sp in EXTRA.items():
        V1[a] = assemble(*sp)
        M_[a] = mapvec(V1[a])

    # ---------------------------------------------------------------- report
    bs = reg['bootstrap']
    rng = np.random.default_rng(bs['seed'])
    BIDX = rng.integers(0, nqy, size=(int(bs['n_boot']), nqy))
    lo_q, hi_q = (1 - bs['level']) / 2 * 100, (1 + bs['level']) / 2 * 100

    def boot(dl):
        mm = dl[BIDX].mean(axis=1)
        return [R(np.percentile(mm, lo_q)), R(np.percentile(mm, hi_q)), R(float((mm > 0).mean()))]

    def report(a, ref):
        dl = M_[a] - M_[ref]
        b = boot(dl)
        out = {'map5': R(M_[a].mean()), 'vs': ref, 'decoded_delta': R(dl.mean()),
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
            out['axis'] = AXIS.get(a)
            out['included_by_rule'] = a in incl
        return out

    res = {'registration_sha256': reg_sha, 'step': reg['step'], 'level': 'v1, 41 views',
           'n_queries': nqy, 'n_views': nv + 1, 'network_fits_run': 0, 'network_cache': cache_rec,
           'family_query_counts': {f: int((fq == f).sum()) for f in FAMS},
           'half_query_counts': [int((hv == 0).sum()), int((hv == 1).sum())],
           'stage2_models_fitted': len(MOD), 'cells_scored': len(PACC), 'arms': {}}
    res['arms'][f'{REF}_vs_{CTRL}'] = report(REF, CTRL)
    for a in list(CHANGE) + list(EXTRA):
        res['arms'][a] = report(a, REF)

    # ---------------------------------------------------------------- direction diagnostics (label-free)
    dis = np.array([(SIGNS[i]['evt'][dtmask] != SIGNS[i]['chip'][dtmask]).mean() for i in range(41)])
    fl_evt = np.array([(SIGNS[i]['evt'][dtmask] != prod_sign[dtmask]).mean() for i in range(41)])
    fl_chip = np.array([(SIGNS[i]['chip'][dtmask] != prod_sign[dtmask]).mean() for i in range(41)])
    und = np.array([float(((EVSTAT[i][0][dtmask] == EVSTAT[i][1][dtmask])).mean()) for i in range(41)])
    dirdiag = {
        'dt_pairs': int(dtmask.sum()),
        'chip_rule_reproduces_production_sign_on_the_full_pool': {'disagreements': chip_gate,
                                                                  'passed': chip_gate == 0},
        'share_of_pair_x_view_cells_where_evt_differs_from_chip': R(dis.mean()),
        'per_view_evt_vs_chip_min_max': [R(dis.min()), R(dis.max())],
        'share_of_cells_flipped_vs_production_evt': R(fl_evt.mean()),
        'share_of_cells_flipped_vs_production_chip': R(fl_chip.mean()),
        'share_of_cells_with_no_event_majority': R(und.mean()),
        'pool_fraction_vs_full': {
            'thinned_views_1_20': R(float(np.mean([pool_sizes[i][dtmask].sum() for i in range(1, 21)])
                                          / pool_sizes[0][dtmask].sum())),
            'padded_views_21_40': R(float(np.mean([pool_sizes[i][dtmask].sum() for i in range(21, 41)])
                                          / pool_sizes[0][dtmask].sum()))},
        'flipped_share_vs_production_by_view_type': {
            'full_pool': [R(float(fl_evt[0])), R(float(fl_chip[0]))],
            'thinned_mean': [R(float(fl_evt[1:21].mean())), R(float(fl_chip[1:21].mean()))],
            'padded_mean': [R(float(fl_evt[21:].mean())), R(float(fl_chip[21:].mean()))],
            'order': 'evt, chip'}}
    tr = CV / reg['direction']['known_donor']['path']
    assert sha(tr) == reg['direction']['known_donor']['sha256'], 'STEP244 pair_roles changed'
    PR = pl.read_parquet(tr).select('pair_id', 'truth')
    tmap = dict(PR.filter(pl.col('truth') > 0).iter_rows())
    kn = np.array([tmap.get(p, 0) for p in pcodes])
    km = dtmask & (kn > 0)
    tsign = np.where(kn == 1, 1.0, -1.0)     # truth 1 = donor is player_1 = sign +1
    dirdiag['known_donor_pairs_in_the_audit'] = int(km.sum())
    dirdiag['accuracy_against_the_known_donor'] = {
        'production_constant_sign': R(float((prod_sign[km] == tsign[km]).mean())),
        'chip_per_view_mean_over_41_views': R(float(np.mean([(SIGNS[i]['chip'][km] == tsign[km]).mean()
                                                             for i in range(41)]))),
        'evt_per_view_mean_over_41_views': R(float(np.mean([(SIGNS[i]['evt'][km] == tsign[km]).mean()
                                                            for i in range(41)]))),
        'chip_on_the_full_pool': R(float((SIGNS[0]['chip'][km] == tsign[km]).mean())),
        'evt_on_the_full_pool': R(float((SIGNS[0]['evt'][km] == tsign[km]).mean())),
        'chip_thinned_views_mean': R(float(np.mean([(SIGNS[i]['chip'][km] == tsign[km]).mean()
                                                    for i in range(1, 21)]))),
        'evt_thinned_views_mean': R(float(np.mean([(SIGNS[i]['evt'][km] == tsign[km]).mean()
                                                   for i in range(1, 21)]))),
        'note': 'label-derived diagnostic (truth comes from listed hands); it selects nothing'}
    res['direction_diagnostics'] = dirdiag

    # ---------------------------------------------------------------- gates
    iso = {}
    for a, f in CHANGE.items():
        dl = M_[a] - M_[REF]
        iso[a] = {g: float(np.abs(dl[fq == g]).max()) for g in FAMS if g != f}
    gates = {
        'S_star_v1_replay_exact': {'max_abs': gate_v1, 'passed': gate_v1 == 0.0},
        'view_pool_reconstruction_exact': {'max_abs': gate_nh, 'passed': gate_nh == 0.0},
        'reference_reproduces': {'here': R(M_[REF].mean()), **reg['reference_gate'],
                                 'exact': bool(R(M_[REF].mean()) == reg['reference_gate']['expected']),
                                 'passed': bool(abs(M_[REF].mean() - reg['reference_gate']['expected'])
                                                <= reg['reference_gate']['tolerance'])},
        'control_reproduces': {'here': R(M_[CTRL].mean()), **reg['control_gate'],
                               'exact': bool(R(M_[CTRL].mean()) == reg['control_gate']['expected']),
                               'passed': bool(abs(M_[CTRL].mean() - reg['control_gate']['expected'])
                                              <= reg['control_gate']['tolerance'])},
        'untouched_families_exactly_zero': {'max_abs': iso,
                                            'passed': all(v == 0.0 for g in iso.values() for v in g.values())},
        'chip_rule_reproduces_production_sign': {'disagreements': chip_gate, 'passed': chip_gate == 0}}
    res['gates'] = gates
    gates_ok = all(v['passed'] for v in gates.values())

    # ---------------------------------------------------------------- G_honest
    def cross_half(pool):
        held = []
        for h in (0, 1):
            pick = {}
            for a in pool:
                f = CHANGE[a]
                dl = fam_delta[a]
                if dl[(hv == h) & (fq == f)].mean() > 0 and dl[hv == h].mean() > 0:
                    mh = dl[(hv == h) & (fq == f)].mean()
                    if f not in pick or mh > pick[f][1]:
                        pick[f] = (a, mh)
            sel = sorted(v[0] for v in pick.values())
            dh = sum((fam_delta[a] for a in sel), np.zeros(nqy))
            held.append({'selected_on_half': h, 'selected': sel, 'held_half_n': int((hv != h).sum()),
                         'held_half_delta': R(dh[hv != h].mean()),
                         'admits_expected_negative': sorted(set(sel) & NEGATIVE)})
        G = sum(x['held_half_n'] * x['held_half_delta'] for x in held) / sum(x['held_half_n'] for x in held)
        return held, R(G)

    held_all, G_all = cross_half(list(CHANGE))
    held_cand, G_cand = cross_half([a for a in CHANGE if a not in NEGATIVE])
    comp = (M_['COMPOSITION'] - M_[REF]) if incl else np.zeros(nqy)
    cs = (M_['COMPOSITION_STRICT'] - M_[REF]) if incl else np.zeros(nqy)
    both = bool(comp[hv == 0].mean() > 0 and comp[hv == 1].mean() > 0)
    bars = reg['bars']
    pd_ = fam_delta[PRIM]
    res['decision'] = {
        'primary_arm': PRIM, 'primary_delta': R(pd_.mean()),
        'primary_halves': [R(pd_[hv == 0].mean()), R(pd_[hv == 1].mean())],
        'primary_family_halves': [R(pd_[(hv == h) & (fq == CHANGE[PRIM])].mean()) for h in (0, 1)],
        'primary_passes_rule': bool(PRIM in incl),
        'included_changes': incl,
        'composition_axiswise_picks': {f'{f}|{ax}': a for (f, ax), a in picked.items()},
        'composition_spec': comp_spec,
        'composition_delta_vs_reference': R(comp.mean()),
        'composition_halves': [R(comp[hv == 0].mean()), R(comp[hv == 1].mean())],
        'composition_boot90': boot(comp)[:2] if incl else None,
        'composition_minus_sum_of_parts': R(comp.mean() - sum(fam_delta[a].mean() for a in incl)) if incl else 0.0,
        'composition_strict_arms': strict_arms,
        'composition_strict_delta': R(cs.mean()),
        'composition_strict_halves': [R(cs[hv == 0].mean()), R(cs[hv == 1].mean())],
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
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    d.select(K).with_columns([pl.Series('v1_' + a, V1[a]) for a in V1]).write_parquet(
        OUT / f'audit_scores_{tag}.parquet')
    res['scores_sha256'] = sha(OUT / f'audit_scores_{tag}.parquet')
    pl.DataFrame({'pair_id': qids, 'half': hv, 'family': fq, **{a: M_[a] for a in M_}}).write_parquet(
        OUT / f'per_query_{tag}.parquet')
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('gates', 'direction_diagnostics', 'decision', 'verdict', 'seconds')},
                     indent=1), flush=True)
    print('"verdict": "%s"' % res['verdict'], flush=True)


if __name__ == '__main__':
    main()
