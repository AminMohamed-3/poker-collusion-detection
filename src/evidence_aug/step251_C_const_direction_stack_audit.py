"""STEP251 (Claude) part 1: the CONSTANT-direction arm, the STEP235 event fusion re-measured here, and the
joint compositions with DTfp3, on the cached 162-query ENS4 evidence audit.

STEP249 measured a per-view donor direction and it LOST (-.005715 overall, DT -.012859) even though the
estimator is measurably better than production's (accuracy against STEP244's known donor .984 vs .969 on the
full pool, .995 vs .948 on the thinned views).  The diagnosis was view instability of the SIGN, not estimator
quality: production bakes ONE sign per pair into the role columns, so any per-view re-estimation makes
dt_donor_* mean one thing in some views and the opposite in others and the 41-view average mixes two
orientations.  This step measures the arm production would actually ship:

  DTdirconst  -- the event rule ("the donor is the member who more often FOLDS FACING THE PARTNER'S
                 AGGRESSION WITH EQUITY > .5; ties and zero-event pools fall back to the chip-flow sign")
                 evaluated ONCE on the FULL pool and held CONSTANT across all 41 views, exactly where
                 production holds its own dt_sign constant.  Six of the 72 audit DT pairs change sign.
  DTfuse25    -- STEP235's carried arm, p' = p**0.75 * prior_shift(p_evt, F=10)**0.25 on directed_transfer
                 rows only, a pure post-processing of the view-averaged stage-2 probability before the typed
                 decoder.  step235's own prior_shift is imported and called read-only.  STEP235 read +.003400
                 on this same reference, so reproducing that number to the digit is a gate here.
  DTfp3       -- STEP249's included arm (fp_gap, call_partner_dominated_bb, fp_best_pot added to the
                 directed_transfer list).  STEP249 read +.006641; reproducing it is a gate.
  J_*         -- the joint refits: fp3+dirconst, fp3+fuse, fp3+dirconst+fuse.

The production stack is chosen by the registered sequential rule: start from DTfp3, then add an arm only if it
is positive on BOTH table halves alone AND the joint refit is not below the current stack on EITHER half.

Cache only: every ENS4 network output is loaded from cache_ens4_networks and asserted present; zero network
fits, so every arm-vs-REF delta is stage-2-only and carries no network measurement noise (OBS 19.154).
The S* view scores use TWO slot accumulators exactly as step221 does, so the replay gate reads exactly 0.0.

Development phase only (asserted).  Reads no evaluation row.  Builds no production file.  Never uploads.
Usage: python step251_C_const_direction_stack_audit.py REGISTRATION_PATH REGSHA
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
from evidence_aug import step235_C_event_fusion_audit as S235

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
CACHE = CV / 'cache_ens4_networks'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
DT = 'directed_transfer'
MEMBERS = S203.MEMBERS
FAMCOL = {f: 'p_' + f for f in FAMS}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = lambda x: round(float(x), 6)


def view_pools(d, u, m, seed, pad):
    """The pool membership of build_views' views(u, m, seed) / pads(u, m, seed), as a per-view row COUNT
    multiplicity over d's rows.  Used only for the reconstruction gate here: STEP251's direction is
    constant, so no view needs its own sign."""
    top0 = (d['oof_rank'].to_numpy() <= 20)
    n = d.height
    out = []
    for c in range(m):
        r = np.random.default_rng([seed, c]).random(n)
        pick = (r < u) & ~top0
        out.append((np.ones(n) + pick.astype(float)) if pad else (~pick).astype(float))
    return out


def main():
    regp, reg_sha = Path(sys.argv[1]), sys.argv[2]
    assert sha(regp) == reg_sha, 'registration hash mismatch'
    reg = json.loads(regp.read_text(encoding='utf-8'))
    assert sha(__file__) == reg['code']['runner_sha256'], 'runner changed after registration'
    assert sha(CV / 'src/evidence_aug/step235_C_event_fusion_audit.py') == reg['code']['step235_sha256']
    OUT = regp.parent
    tag = reg['step'].lower()
    dest = OUT / f'results_{tag}.json'
    assert not dest.exists(), 'preserve a completed read'
    REF, CTRL = reg['reference'], reg['control_gate']['arm']
    PRIM = reg['primary_arm']
    PRED = reg['prediction']
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

    # ---------------------------------------------------------------- the CONSTANT donor direction
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

    # ONE evaluation, on the unviewed full pool, held constant everywhere
    e1 = np.bincount(pair_ix, weights=ev1, minlength=npair)
    e2 = np.bincount(pair_ix, weights=ev2, minlength=npair)
    fl = np.bincount(pair_ix, weights=flow, minlength=npair)
    chip_sign = np.where(fl >= 0, 1.0, -1.0)
    evt_sign = np.where(e1 > e2, 1.0, np.where(e2 > e1, -1.0, chip_sign))
    dtmask = np.isin(pcodes, dtpairs)
    chip_gate = int((chip_sign[dtmask] != prod_sign[dtmask]).sum())
    FRAME_SIGN = {'prod': prod_sign, 'evtconst': evt_sign}
    flip_pair = {fr: (FRAME_SIGN[fr] != prod_sign) & dtmask for fr in FRAME_SIGN}
    FLIP = {fr: flip_pair[fr][np.searchsorted(pcodes, qpairs)] for fr in FRAME_SIGN}
    print('constant evt rule flips %d of %d DT pairs' % (int(flip_pair['evtconst'].sum()), int(dtmask.sum())),
          flush=True)

    def apply_frame(v, frame):
        """The sign is CONSTANT, so the same swap is applied to the training frame and to every view."""
        f = FLIP[frame]
        if not f.any():
            return v
        sel = pl.Series('__flip', f)
        return v.with_columns([e for a, b in SWAP for e in (
            pl.when(sel).then(pl.col(b)).otherwise(pl.col(a)).alias(a),
            pl.when(sel).then(pl.col(a)).otherwise(pl.col(b)).alias(b))])

    # ---------------------------------------------------------------- event probabilities (STEP232 OOF)
    evp = CV / reg['event_file']['path']
    assert sha(evp) == reg['event_file']['sha256'], 'event file changed after registration'
    oof = pl.read_parquet(evp)
    ev = oof.select(K + ['table_id', 'label', 'behavior_family', 'fold'] + [FAMCOL[f] for f in FAMS] + ['p_any'])
    je = Q.select(K + ['table_id', 'behavior_family']).join(
        ev, on=K, how='left', validate='1:1', maintain_order='left', suffix='_ev')
    assert je.height == nq and je.select(K).equals(Q.select(K)), 'event join reordered or dropped audit rows'
    ev_cover = {c: int(je[c].null_count()) for c in [FAMCOL[f] for f in FAMS] + ['p_any', 'fold', 'label']}
    assert sum(ev_cover.values()) == 0, ('event coverage is not 100 %', ev_cover)
    assert (je['label'] == 1).all(), 'an audit row is not a labelled positive pair in the event file'
    assert (je['table_id'] == je['table_id_ev']).all(), 'table_id disagrees between the audit and the event file'
    assert (je['behavior_family'] == je['behavior_family_ev']).all(), 'family disagrees'
    chk = je.group_by('pair_id').agg(pl.col('table_id').n_unique().alias('t'), pl.col('fold').n_unique().alias('f'))
    assert chk['t'].max() == 1 and chk['f'].max() == 1, 'an audit pair spans tables or folds'
    PEVT = {f: je[FAMCOL[f]].to_numpy().astype(np.float64) for f in FAMS}
    ev_folds = sorted(set(je.select('pair_id', 'fold').unique()['fold'].to_list()))

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

    def spec_of(arm):
        v = reg['arms'][arm]
        return (v['members'],
                {f: v.get('lists', {}).get(f, 'ref') for f in FAMS},
                {f: v.get('frames', {}).get(f, 'prod') for f in FAMS},
                v.get('fuse'))

    need = sorted({(m, f, lm[f], fm[f]) for a in reg['arms']
                   for ms, lm, fm, _ in [spec_of(a)] for m in ms for f in FAMS})
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    MOD = {}
    Qf = {fr: apply_frame(Q, fr) for fr in {k[3] for k in need}}
    for m, f, nm, fr in need:
        z = Qf[fr].select(LISTS[(f, nm)]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            MOD[(m, f, nm, fr, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                np.column_stack([z[a], EA[m][k][isf[f][tr]]]), y[a])
    print('stage-2 models fitted: %d (%d cells)  %ds' % (len(MOD), len(need), round(time.perf_counter() - t0)),
          flush=True)

    def score_parts(v):
        vf = {fr: apply_frame(v, fr) for fr in {k[3] for k in need}}
        out = {}
        for keyc in need:
            m, f, nm, fr = keyc
            z = vf[fr].select(LISTS[(f, nm)]).to_numpy().astype(float)
            p = np.full(nq, np.nan)
            for k, (tr, te) in enumerate(folds_q):
                b = te[isf[f][te]]
                if len(b):
                    p[b] = MOD[(m, f, nm, fr, k)].predict_proba(
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
    PACC = {keyc: [np.zeros(nq), np.zeros(nq), None] for keyc in need}
    for keyc, p in score_parts(Q).items():
        PACC[keyc][2] = np.nan_to_num(p)
    accS = [np.zeros(len(d)), np.zeros(len(d))]          # two slots, exactly as step221
    pool_sizes = [np.bincount(pair_ix, weights=mult, minlength=npair)
                  for mult in [np.ones(d.height)]
                  + [x for pad, seeds in ((False, (1241, 2241)), (True, (1242, 2242)))
                     for s in seeds for x in view_pools(d, .5, 10, s, pad)]]
    assert len(pool_sizes) == 41
    nhgate, i, nv = [], 0, 0
    views, pads = build_views(D)
    for slot, (maker, seeds) in enumerate(((views, (1241, 2241)), (pads, (1242, 2242)))):
        for s in seeds:
            for v in maker(.5, 10, s):
                i += 1
                got = dict(v.group_by('pair_id').agg(pl.col('n_hands').first()).iter_rows())
                mine = {pcodes[k]: pool_sizes[i][k] for k in range(npair)}
                nhgate.append(max(abs(got[p] - mine[p]) for p in got))
                accS[slot] += score_sstar(v)
                for keyc, p in score_parts(v).items():
                    PACC[keyc][slot] += np.nan_to_num(p)
                nv += 1
            print('views %d  %ds' % (nv, round(time.perf_counter() - t0)), flush=True)
    assert nv == 40 and i == 40, (nv, i)
    gate_nh = float(max(nhgate))
    V1S = (accS[0] + accS[1] + D['SS']) / (nv + 1)
    gate_v1 = float(np.abs(V1S - D['V1']).max())
    print('S* v1 replay max abs diff %s   pool reconstruction max abs %s' % (gate_v1, gate_nh), flush=True)

    QIX = D['QIX']

    def assemble(ms, lm, fm, fuse):
        p = D['SS'].copy()
        for f in FAMS:
            rows = np.flatnonzero(isf[f])
            cells = [PACC[(m, f, lm[f], fm[f])] for m in ms]
            p[D['QIX'][rows]] = np.mean([(a[0] + a[1] + a[2])[rows] / (nv + 1) for a in cells], axis=0)
        if fuse:
            f, w, F = fuse['family'], float(fuse['w']), float(fuse['F'])
            pe = S235.prior_shift(PEVT[f], F)          # step235's own function, read-only
            m = isf[f]
            pt = p[QIX] - 10.0
            newp = pt.copy()
            newp[m] = (np.clip(pt[m], 1e-9, 1.0) ** (1 - w)) * (np.clip(pe[m], 1e-9, 1 - 1e-9) ** w)
            p = p.copy()
            p[QIX] = 10.0 + newp
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
    fam_delta = {a: M_[a] - M_[REF] for a in reg['arms'] if a != REF}

    # ---------------------------------------------------------------- the registered sequential stack rule
    ORDER = reg['stack']['order']
    JOINT = reg['stack']['joint_arms']
    joint_of = lambda st: JOINT['|'.join(st)]

    def run_stack(mask, label):
        st, steps = [ORDER[0]], []
        for a in ORDER[1:]:
            al = fam_delta[a]
            ok_alone = all(al[mask & (hv == h)].mean() > 0 for h in (0, 1))
            cur, new = M_[joint_of(st)], M_[joint_of(st + [a])]
            dd = new - cur
            ok_joint = all(dd[mask & (hv == h)].mean() >= 0 for h in (0, 1))
            steps.append({'arm': a, 'on': label,
                          'alone_halves': [R(al[mask & (hv == h)].mean()) for h in (0, 1)],
                          'positive_on_both_halves_alone': bool(ok_alone),
                          'joint_vs_current_stack': joint_of(st + [a]) + ' - ' + joint_of(st),
                          'joint_minus_stack_halves': [R(dd[mask & (hv == h)].mean()) for h in (0, 1)],
                          'joint_minus_stack_overall': R(dd[mask].mean()),
                          'not_below_stack_on_either_half': bool(ok_joint),
                          'added': bool(ok_alone and ok_joint),
                          'own_family_halves_alone': [R(al[mask & (hv == h) & (fq == CHANGE[a])].mean())
                                                      for h in (0, 1)]})
            if ok_alone and ok_joint:
                st.append(a)
        return st, steps

    allm = np.ones(nqy, bool)
    STACK, STEPS = run_stack(allm, 'both halves')
    STACK_ARM = joint_of(STACK)

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
            out['role'] = ROLES.get(a, 'joint')
        return out

    res = {'registration_sha256': reg_sha, 'step': reg['step'], 'level': 'v1, 41 views',
           'n_queries': nqy, 'n_views': nv + 1, 'network_fits_run': 0, 'network_cache': cache_rec,
           'family_query_counts': {f: int((fq == f).sum()) for f in FAMS},
           'half_query_counts': [int((hv == 0).sum()), int((hv == 1).sum())],
           'stage2_models_fitted': len(MOD), 'cells_scored': len(need),
           'event_coverage': {'audit_rows': nq, 'nulls': ev_cover, 'folds_used': ev_folds}, 'arms': {}}
    res['arms'][f'{REF}_vs_{CTRL}'] = report(REF, CTRL)
    for a in reg['arms']:
        if a not in (REF, CTRL):
            res['arms'][a] = report(a, REF)

    # ---------------------------------------------------------------- direction diagnostics
    tr = CV / reg['direction']['known_donor']['path']
    assert sha(tr) == reg['direction']['known_donor']['sha256'], 'STEP244 pair_roles changed'
    PR = pl.read_parquet(tr).select('pair_id', 'truth')
    tmap = dict(PR.filter(pl.col('truth') > 0).iter_rows())
    kn = np.array([tmap.get(p, 0) for p in pcodes])
    km = dtmask & (kn > 0)
    tsign = np.where(kn == 1, 1.0, -1.0)      # truth 1 = donor is player_1 = sign +1
    fl_pairs = pcodes[flip_pair['evtconst']]
    qid_ix = {q: n for n, q in enumerate(qids)}
    dda = fam_delta[PRIM]
    ddj = M_[JOINT['|'.join([ORDER[0], PRIM])]] - M_[ORDER[0]] if PRIM in ORDER else None
    know = {p: ('production_wrong' if (kn[np.searchsorted(pcodes, p)] > 0
                                       and prod_sign[np.searchsorted(pcodes, p)]
                                       != tsign[np.searchsorted(pcodes, p)])
                else 'production_right' if kn[np.searchsorted(pcodes, p)] > 0 else 'no_known_donor')
            for p in fl_pairs}
    res['direction_diagnostics'] = {
        'dt_pairs': int(dtmask.sum()),
        'pairs_whose_sign_changes': int(flip_pair['evtconst'].sum()),
        'changed_pairs': [{'pair_id': str(p), 'known_donor_status': know[p],
                           'production_sign': int(prod_sign[np.searchsorted(pcodes, p)]),
                           'constant_event_sign': int(evt_sign[np.searchsorted(pcodes, p)]),
                           'pool_hands': int(pool_sizes[0][np.searchsorted(pcodes, p)]),
                           'delta_%s' % PRIM: R(dda[qid_ix[p]]),
                           'delta_in_joint_with_%s' % ORDER[0]:
                               (R(ddj[qid_ix[p]]) if ddj is not None else None)}
                          for p in fl_pairs],
        'delta_on_unchanged_dt_queries': R(dda[[qid_ix[q] for q in qids
                                                if fq[qid_ix[q]] == DT and q not in set(fl_pairs)]].mean()),
        'unchanged_dt_queries_nonzero': int(sum(1 for q in qids if fq[qid_ix[q]] == DT
                                                and q not in set(fl_pairs) and dda[qid_ix[q]] != 0)),
        'no_event_majority_pairs': int(((e1 == e2) & dtmask).sum()),
        'known_donor_pairs_in_the_audit': int(km.sum()),
        'accuracy_against_the_known_donor': {
            'production_constant_sign': R(float((prod_sign[km] == tsign[km]).mean())),
            'constant_event_sign': R(float((evt_sign[km] == tsign[km]).mean())),
            'note': 'label-derived diagnostic (truth comes from listed hands); it selects nothing'},
        'chip_rule_reproduces_production_sign_on_the_full_pool': {'disagreements': chip_gate,
                                                                  'passed': chip_gate == 0}}

    # ---------------------------------------------------------------- gates
    iso = {}
    for a, f in CHANGE.items():
        dl = M_[a] - M_[REF]
        iso[a] = {g: float(np.abs(dl[fq == g]).max()) for g in FAMS if g != f}
    rp_ = reg['replay_gates']
    replay = {a: {'here': res['arms'][a]['decoded_delta'], 'expected': v['expected'], 'from': v['from'],
                  'passed': bool(res['arms'][a]['decoded_delta'] == v['expected'])}
              for a, v in rp_.items()}
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
        'chip_rule_reproduces_production_sign': {'disagreements': chip_gate, 'passed': chip_gate == 0},
        'event_coverage_100pct': {'nulls': ev_cover, 'passed': sum(ev_cover.values()) == 0},
        'prior_step_replays_exact': {'arms': replay, 'passed': all(v['passed'] for v in replay.values())}}
    res['gates'] = gates
    gates_ok = all(v['passed'] for v in gates.values())

    # ---------------------------------------------------------------- G_honest (cross-half, same rule)
    held = []
    for h in (0, 1):
        sel, _ = run_stack(hv == h, 'half %d' % h)
        dh = M_[joint_of(sel)] - M_[REF]
        held.append({'selected_on_half': h, 'selected_stack': sel, 'selected_arm': joint_of(sel),
                     'held_half_n': int((hv != h).sum()), 'held_half_delta': R(dh[hv != h].mean())})
    G = sum(x['held_half_n'] * x['held_half_delta'] for x in held) / sum(x['held_half_n'] for x in held)
    comp = fam_delta[STACK_ARM] if STACK_ARM != REF else np.zeros(nqy)
    both = bool(comp[hv == 0].mean() > 0 and comp[hv == 1].mean() > 0)
    bars = reg['bars']
    add = R(comp.mean() - sum(fam_delta[a].mean() for a in STACK if a in fam_delta))
    res['decision'] = {
        'primary_arm': PRIM, 'primary_delta': R(fam_delta[PRIM].mean()),
        'primary_halves': [R(fam_delta[PRIM][hv == h].mean()) for h in (0, 1)],
        'primary_family_halves': [R(fam_delta[PRIM][(hv == h) & (fq == CHANGE[PRIM])].mean()) for h in (0, 1)],
        'stack_rule': reg['stack']['rule'],
        'stack_steps': STEPS, 'stack': STACK, 'stack_arm': STACK_ARM,
        'stack_delta_vs_reference': R(comp.mean()),
        'stack_halves': [R(comp[hv == 0].mean()), R(comp[hv == 1].mean())],
        'stack_per_family': {f: R(comp[fq == f].mean()) for f in FAMS},
        'stack_queries_up_down': [int((comp > 0).sum()), int((comp < 0).sum())],
        'stack_boot90': boot(comp)[:2], 'stack_boot_P_positive': boot(comp)[2],
        'stack_minus_sum_of_parts': add,
        'honest_cross_half': held, 'G_honest': R(G),
        'bars': bars, 'carried_gain_bar': reg['carried_gain_bar'],
        'composite_expected': R(comp.mean() * 0.2 * reg['transfer_ratio']),
        'composite_expected_shrunk': R(max(G, 0.0) * 0.2 * reg['transfer_ratio'])}
    res['verdict'] = ('VOID' if not gates_ok else
                      'NO_BUILD' if comp.mean() <= 0 else
                      'BUILD_RECOMMENDED' if (comp.mean() >= bars['recommended'] and both and G > 0) else
                      'BUILD_OPTION' if (comp.mean() >= bars['option'] and both and G > 0) else
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
    print('"verdict": "%s"  stack=%s  delta=%s' % (res['verdict'], STACK, R(comp.mean())), flush=True)


if __name__ == '__main__':
    main()
