"""STEP256 B -- Minimum-Bayes-Risk decoding of the five-hand evidence list, measured on the cached 162-query audit.

PURE POST-PROCESSING of saved development audit scores: no model is refitted, no network is touched, no
evaluation row is read, nothing is uploaded.  Arms, gates and predicted bands are fixed in registration256.json,
which was written before any MBR arm was scored.

Usage: python step256_B_mbr_audit.py REGISTRATION_PATH REGSHA
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '4'
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
CV = POKER / 'Claude Version'
DATA = POKER / 'data'
sys.path.insert(0, str(CV / 'src'))
sys.dont_write_bytecode = True

OUT = CV / 'experiments' / 'step256_mbr_decoder_20260917'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
S_SAMPLES = 4000
SEED_SAMPLE, SEED_SHUF, SEED_BOOT = 2560001, 2560002, 2560003
TEMPS = [0.6, 0.8, 1.25, 1.67, 2.5]
WIDTH = 8
PERMS = np.array(list(itertools.permutations(range(WIDTH), 5)), dtype=np.int64)   # first row is (0,1,2,3,4) = REF
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R6 = lambda x: round(float(x), 6)


# ------------------------------------------------------------------ production decoder (step104)
def less_than_five(ps):
    dist, out = np.zeros(6), np.empty(len(ps))
    dist[0] = 1.
    for i, p in enumerate(ps):
        out[i] = dist[:5].sum()
        new = dist * (1 - p)
        new[1:] += dist[:-1] * p
        new[5] += dist[5] * p
        dist = new
    return out


def marginal_q(p, order):
    q = np.zeros(len(p))
    q[order] = less_than_five(p[order])
    return q


def ref_rank_of(p, order, emit):
    """Production decoded score -> the ranking production would emit (indices into the full row array)."""
    sc = np.full(len(p), -np.inf)
    q = marginal_q(p, order)
    sc[emit] = (p * q + 1e-9 * p)[emit]
    return np.argsort(-sc, kind='stable')


# ------------------------------------------------------------------ MBR machinery
def draw_listed(p, order, S, rng):
    """R[s, i] = 1 iff row i is LISTED in draw s under the exact (priority, cap-5) rule; nz[s] = list length."""
    zo = rng.random((S, len(order))) < p[order][None, :]
    listed_o = zo & (np.cumsum(zo, axis=1) <= 5)
    R = np.zeros((S, len(p)), dtype=bool)
    R[:, order] = listed_o
    return R, listed_o.sum(1)


def greedy(Rc, W):
    """Rc: S x m draws restricted to the emittable candidates, ALREADY column-sorted by REF preference.
    W: S weights (1/n_z). Returns the 5 chosen column positions and their marginal expected-AP gains."""
    S, m = Rc.shape
    H = np.zeros(S)
    live = np.ones(m, bool)
    picks, gains = [], []
    for slot in range(1, 6):
        w = W * (H + 1.0) / slot
        g = Rc.T @ w / S
        g[~live] = -np.inf
        j = int(np.argmax(g))
        picks.append(j)
        gains.append(float(g[j]))
        live[j] = False
        H = H + Rc[:, j]
    return picks, gains


def expected_ap(Rc, W, cols):
    r = Rc[:, list(cols)]
    cum = np.cumsum(r, axis=1)
    return float(((r * cum / np.arange(1, 6)).sum(1) * W).mean())


def exhaustive(Rc8, nz):
    """Exhaustive expected AP@5 over every ordered 5-subset of 8 candidates, via (pattern, n_z) compression."""
    S = Rc8.shape[0]
    B = (Rc8 > 0).astype(np.int32)
    nzi = np.asarray(nz, dtype=np.int32)
    code = B @ (1 << np.arange(WIDTH)) * 8 + nzi
    uc, cnt = np.unique(code, return_counts=True)
    bits = ((uc // 8)[:, None] >> np.arange(WIDTH)) & 1
    nzp = uc % 8
    wp = np.where(nzp > 0, 1.0 / np.maximum(nzp, 1), 0.0) * cnt
    E = np.zeros(len(PERMS))
    kk = np.arange(1, 6, dtype=np.float32)
    for s in range(0, len(uc), 64):
        RT = bits[s:s + 64].astype(np.float32)[:, PERMS]
        ap = (RT * np.cumsum(RT, axis=2) / kk).sum(2)
        E += ap.T @ wp[s:s + 64].astype(np.float32)
    return E / S


def temper(p, T):
    a = np.clip(p, 1e-9, 1 - 1e-9) ** (1.0 / T)
    b = (1 - np.clip(p, 1e-9, 1 - 1e-9)) ** (1.0 / T)
    return a / (a + b)


def ap5_of(hand_ix, rel, n_listed):
    r = rel[np.asarray(hand_ix)]
    hits = np.cumsum(r)
    return float((hits / np.arange(1, len(r) + 1) * r).sum() / min(5, n_listed))


# ------------------------------------------------------------------ per-pair evaluation of every arm
def run_pair(G, qi, n_listed, arms_wanted):
    """G: the pair's rows already sorted by (t, hand_id). Returns {arm: (list of row indices, ap5)} plus diagnostics."""
    t1 = G['T1eff'].to_numpy()
    top = G['top20'].to_numpy()
    rel = G['rel'].to_numpy()
    n = G.height
    torder = np.arange(n, dtype=float)
    emit = np.flatnonzero(top)
    p_full = np.zeros(n)
    p_full[emit] = np.clip(G['v1_REF'].to_numpy()[emit] - 10.0, 0, 1)
    p_evt = np.clip(G['p_fam'].to_numpy(), 0, 1)
    p_mass = p_full.copy()
    p_mass[~top] = p_evt[~top]

    order_all = np.lexsort((torder, ~t1))
    # production restricts the Poisson-binomial to the top-20 rows only
    ot = np.lexsort((torder[emit], ~t1[emit]))
    order_top = emit[ot]

    out, diag = {}, {}

    def decode_marginal(p_vec, ordr):
        sc = np.full(n, -np.inf)
        q = np.zeros(n)
        q[ordr] = less_than_five(p_vec[ordr])
        sc[emit] = (p_vec * q + 1e-9 * p_vec)[emit]
        return list(np.argsort(-sc, kind='stable')[:5])

    ref_list = decode_marginal(p_full, order_top)
    out['REF'] = ref_list
    # REF preference ranking over the emittable candidates (drives every MBR tie-break)
    scr = np.full(n, -np.inf)
    qr = np.zeros(n)
    qr[order_top] = less_than_five(p_full[order_top])
    scr[emit] = (p_full * qr + 1e-9 * p_full)[emit]
    cand = np.argsort(-scr, kind='stable')[:len(emit)]        # all 20 candidates, best REF first

    out['MASS'] = decode_marginal(p_mass, order_all)

    rng = np.random.default_rng([SEED_SAMPLE, qi])
    R, nz = draw_listed(p_full, order_top, S_SAMPLES, rng)
    W = np.where(nz > 0, 1.0 / np.maximum(nz, 1), 0.0)
    Rc = R[:, cand].astype(np.float64)
    picks, gains = greedy(Rc, W)
    out['MBR-G'] = [int(cand[j]) for j in picks]
    diag['E_ap_MBR_G'] = expected_ap(Rc, W, picks)
    diag['E_ap_REF'] = expected_ap(Rc, W, [int(np.flatnonzero(cand == r)[0]) for r in ref_list])
    diag['mean_nz'] = float(nz.mean())
    diag['p_ge5_active'] = float((nz >= 5).mean())

    # exhaustive over the top-8 by REF marginal
    Eplain = exhaustive(Rc[:, :WIDTH], nz)
    jx = int(np.argmax(Eplain))
    out['MBR-X'] = [int(cand[c]) for c in PERMS[jx]]
    diag['E_ap_MBR_X'] = float(Eplain[jx])
    # section 9 span prior (arbitrary Gamma(1,1); labelled diagnostic only)
    t5 = torder[cand[:WIDTH]][PERMS].max(1) / max(n - 1, 1)
    js = int(np.argmax(Eplain * (1.0 + t5) ** -6.0))
    out['DIAG-SPAN'] = [int(cand[c]) for c in PERMS[js]]

    # whole-history sampler
    Rm, nzm = draw_listed(p_mass, order_all, S_SAMPLES, np.random.default_rng([SEED_SAMPLE + 7, qi]))
    Wm = np.where(nzm > 0, 1.0 / np.maximum(nzm, 1), 0.0)
    pm, _ = greedy(Rm[:, cand].astype(np.float64), Wm)
    out['MBR-G+MASS'] = [int(cand[j]) for j in pm]

    # control 1: p permuted within pair
    ps = p_full.copy()
    ps[emit] = np.random.default_rng([SEED_SHUF, qi]).permutation(p_full[emit])
    cs = np.argsort(-np.where(top, ps * marginal_q(ps, order_top) + 1e-9 * ps, -np.inf), kind='stable')[:len(emit)]
    Rs, nzs = draw_listed(ps, order_top, S_SAMPLES, np.random.default_rng([SEED_SAMPLE + 13, qi]))
    Ws = np.where(nzs > 0, 1.0 / np.maximum(nzs, 1), 0.0)
    pk, _ = greedy(Rs[:, cs].astype(np.float64), Ws)
    out['CTRL-SHUF'] = [int(cs[j]) for j in pk]

    # control 2: independent relevance at the REF marginal, constant denominator 5 -> must be the marginal sort
    qmar = np.zeros(n)
    qmar[emit] = (p_full * qr)[emit]
    rng2 = np.random.default_rng([SEED_SAMPLE + 29, qi])
    Ri = (rng2.random((S_SAMPLES, len(cand))) < qmar[cand][None, :]).astype(np.float64)
    pi_, _ = greedy(Ri, np.full(S_SAMPLES, 0.2))
    out['CTRL-IND'] = [int(cand[j]) for j in pi_]

    # gate G2: degenerate posterior must give back the REF list
    pd = (p_full >= 0.5).astype(float)
    dref = decode_marginal(pd, order_top)
    dcand = np.argsort(-np.where(top, pd * marginal_q(pd, order_top) + 1e-9 * pd, -np.inf), kind='stable')[:len(emit)]
    Rd, nzd = draw_listed(pd, order_top, 64, np.random.default_rng([SEED_SAMPLE + 31, qi]))
    Wd = np.where(nzd > 0, 1.0 / np.maximum(nzd, 1), 0.0)
    pdk, _ = greedy(Rd[:, dcand].astype(np.float64), Wd)
    diag['identity_ok'] = bool([int(dcand[j]) for j in pdk] == dref)

    # temperature diagnostic and seed stability
    for T in TEMPS:
        pt = np.zeros(n)
        pt[emit] = temper(p_full[emit], T)
        ct = np.argsort(-np.where(top, pt * marginal_q(pt, order_top) + 1e-9 * pt, -np.inf), kind='stable')[:len(emit)]
        Rt, nzt = draw_listed(pt, order_top, S_SAMPLES, np.random.default_rng([SEED_SAMPLE + 101, qi]))
        Wt = np.where(nzt > 0, 1.0 / np.maximum(nzt, 1), 0.0)
        kt, _ = greedy(Rt[:, ct].astype(np.float64), Wt)
        out['DIAG-TEMP_T%s' % T] = [int(ct[j]) for j in kt]
    for e, s in enumerate((211, 307)):
        Rb, nzb = draw_listed(p_full, order_top, S_SAMPLES, np.random.default_rng([SEED_SAMPLE + s, qi]))
        Wb = np.where(nzb > 0, 1.0 / np.maximum(nzb, 1), 0.0)
        kb, _ = greedy(Rb[:, cand].astype(np.float64), Wb)
        out['MBR-G_seed%d' % (e + 2)] = [int(cand[j]) for j in kb]

    hid = G['hand_id'].to_list()
    res = {a: {'ap5': ap5_of(v, rel, n_listed), 'hands': [hid[i] for i in v]} for a, v in out.items()}
    return res, diag


def main():
    regp, reg_sha = Path(sys.argv[1]), sys.argv[2]
    assert sha(regp) == reg_sha, 'registration hash mismatch'
    reg = json.loads(regp.read_text(encoding='utf-8'))
    dest = OUT / 'results256.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    d = pl.read_parquet(OUT / 'inputs256.parquet')
    assert sha(OUT / 'inputs256.parquet') == reg['inputs']['inputs256.parquet']
    ev = pl.read_csv(DATA / 'development_evidence.csv').select('pair_id', 'hand_id')
    n_listed = dict(ev.group_by('pair_id').len().iter_rows())
    ph = (pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'phase')
          .filter(pl.col('hand_id').is_in(d['hand_id'].unique().implode())).collect())
    assert ph['phase'].unique().to_list() == ['development'], 'STEP256 touches only development hands'
    from evidence_aug.step5_evaluate import half as half_of

    d = d.sort('pair_id', 't', 'hand_id')
    qids = sorted(d['pair_id'].unique().to_list())
    qindex = {q: i for i, q in enumerate(qids)}
    fam = dict(d.select('pair_id', 'family').unique().iter_rows())
    hlf = dict(d.select('pair_id', 'half').unique().iter_rows())
    tab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    assert all(hlf[q] == half_of(tab[q]) for q in qids), 'table halves differ from STEP235'
    ref_rec = dict(pl.read_parquet(OUT / 'ref_per_query256.parquet').select('pair_id', 'REF').iter_rows())

    per, diags = {}, {}
    for (pid,), G in d.group_by(['pair_id'], maintain_order=True):
        r, dg = run_pair(G, qindex[pid], n_listed[pid], None)
        per[pid] = r
        diags[pid] = dg
    print('decoded %d queries in %ds' % (len(per), round(time.perf_counter() - t0)), flush=True)

    ARMS = list(next(iter(per.values())).keys())
    M = {a: np.array([per[q][a]['ap5'] for q in qids]) for a in ARMS}
    LI = {a: [tuple(per[q][a]['hands']) for q in qids] for a in ARMS}
    fq = np.array([fam[q] for q in qids])
    hv = np.array([hlf[q] for q in qids])
    assert float(np.abs(M['REF'] - np.array([ref_rec[q] for q in qids])).max()) == 0.0, 'REF drifted'

    boot_rng = np.random.default_rng(SEED_BOOT)
    BIX = boot_rng.integers(0, len(qids), size=(10000, len(qids)))

    def report(a):
        dl = M[a] - M['REF']
        chg = np.array([LI[a][i] != LI['REF'][i] for i in range(len(qids))])
        setchg = np.array([set(LI[a][i]) != set(LI['REF'][i]) for i in range(len(qids))])
        slots = np.array([sum(1 for k in range(5) if LI[a][i][k] != LI['REF'][i][k]) for i in range(len(qids))])
        bs = dl[BIX].mean(1)
        return {'map5': R6(M[a].mean()), 'delta': R6(dl.mean()),
                'halves': [R6(dl[hv == 0].mean()), R6(dl[hv == 1].mean())],
                'map5_halves': [R6(M[a][hv == 0].mean()), R6(M[a][hv == 1].mean())],
                'per_family': {f: R6(dl[fq == f].mean()) for f in FAMS},
                'map5_per_family': {f: R6(M[a][fq == f].mean()) for f in FAMS},
                'queries_changed_order': int(chg.sum()), 'queries_changed_set': int(setchg.sum()),
                'mean_slots_changed': R6(slots.mean()), 'mean_slots_changed_when_changed':
                    R6(slots[chg].mean()) if chg.any() else 0.0,
                'queries_up_down': [int((dl > 0).sum()), int((dl < 0).sum())],
                'boot90': [R6(np.quantile(bs, .05)), R6(np.quantile(bs, .95))],
                'query_se': R6(dl.std(ddof=1) / np.sqrt(len(dl)))}

    res = {'step': 'STEP256', 'registration_sha256': reg_sha, 'runner_sha256': sha(__file__),
           'n_queries': len(qids), 'samples': S_SAMPLES, 'refits': 0,
           'phase_asserted': 'development', 'arms': {a: report(a) for a in ARMS}}
    res['arms']['REF']['map5_per_family'] = {f: R6(M['REF'][fq == f].mean()) for f in FAMS}

    # ---- gates -------------------------------------------------------------------------------
    PRIM = ['MBR-G', 'MBR-X', 'MASS', 'MBR-G+MASS']
    ident = all(diags[q]['identity_ok'] for q in qids)
    ind_agree = float(np.mean([LI['CTRL-IND'][i] == LI['REF'][i] for i in range(len(qids))]))
    gx_agree = float(np.mean([LI['MBR-X'][i] == LI['MBR-G'][i] for i in range(len(qids))]))
    seeds = [M['MBR-G'].mean(), M['MBR-G_seed2'].mean(), M['MBR-G_seed3'].mean()]
    gates = {
        'G1_reference_reproduces': {'map5': R6(M['REF'].mean()), 'expected': 0.727099,
                                    'per_family': {f: R6(M['REF'][fq == f].mean()) for f in FAMS},
                                    'passed': R6(M['REF'].mean()) == 0.727099},
        'G2_identity_degenerate_posterior': {'queries_ok': int(sum(diags[q]['identity_ok'] for q in qids)),
                                             'of': len(qids), 'passed': ident},
        'G3_shuffle_control_loses': {'delta': R6((M['CTRL-SHUF'] - M['REF']).mean()),
                                     'passed': bool((M['CTRL-SHUF'] - M['REF']).mean() <= -0.02)},
        'G4_independent_control_equals_marginal_sort': {
            'delta': R6((M['CTRL-IND'] - M['REF']).mean()), 'ordered_list_agreement': R6(ind_agree),
            'passed': bool(abs((M['CTRL-IND'] - M['REF']).mean()) <= 0.004 and ind_agree >= 0.90)},
        'G7_monte_carlo_stability': {'map5_three_seeds': [R6(x) for x in seeds],
                                     'spread': R6(max(seeds) - min(seeds)),
                                     'passed': bool(max(seeds) - min(seeds) <= 0.004)}}
    incl = {}
    for a in PRIM:
        r = res['arms'][a]
        incl[a] = {'delta': r['delta'], 'halves': r['halves'], 'min_family': R6(min(r['per_family'].values())),
                   'gain_bar': bool(r['delta'] >= 0.0034),
                   'both_halves_positive': bool(r['halves'][0] > 0 and r['halves'][1] > 0),
                   'no_family_below': bool(min(r['per_family'].values()) >= -0.005)}
        incl[a]['INCLUDED'] = bool(incl[a]['gain_bar'] and incl[a]['both_halves_positive'] and incl[a]['no_family_below'])
    gates['G5_inclusion'] = incl
    gates['G6_no_calibration_in_primaries'] = {'fitted_parameters_in_primaries': 0, 'passed': True}
    res['gates'] = gates
    res['agreement_MBR_X_vs_MBR_G'] = R6(gx_agree)
    res['expected_ap_model_side'] = {
        'mean_E_ap_REF': R6(np.mean([diags[q]['E_ap_REF'] for q in qids])),
        'mean_E_ap_MBR_G': R6(np.mean([diags[q]['E_ap_MBR_G'] for q in qids])),
        'mean_E_ap_MBR_X': R6(np.mean([diags[q]['E_ap_MBR_X'] for q in qids])),
        'queries_where_MBR_G_beats_REF_in_expectation':
            int(sum(diags[q]['E_ap_MBR_G'] > diags[q]['E_ap_REF'] + 1e-12 for q in qids)),
        'mean_sampled_list_length': R6(np.mean([diags[q]['mean_nz'] for q in qids])),
        'mean_P_five_or_more_active': R6(np.mean([diags[q]['p_ge5_active'] for q in qids]))}

    # ---- temperature diagnostic, cross-half selection only ------------------------------------
    tarms = ['MBR-G'] + ['DIAG-TEMP_T%s' % T for T in TEMPS]
    held = []
    for h in (0, 1):
        pick = max(tarms, key=lambda a: M[a][hv == h].mean())
        held.append({'selected_on_half': h, 'selected': pick,
                     'held_half_n': int((hv != h).sum()),
                     'held_half_delta': R6((M[pick] - M['REF'])[hv != h].mean())})
    res['diagnostic_temperature'] = {
        'note': 'LABELLED DIAGNOSTIC, never a candidate (registration G6). STEP175 scored +.0085 on this same '
                'audit and -.00120 on the public board, so a positive read here is evidence of nothing.',
        'per_T': {a: res['arms'][a]['delta'] for a in tarms},
        'cross_half': held,
        'G_honest': R6(sum(x['held_half_n'] * x['held_half_delta'] for x in held)
                       / sum(x['held_half_n'] for x in held))}
    res['diagnostic_span'] = {'note': 'section 9 span prior with an arbitrary Gamma(1,1); not gated.',
                              'delta_vs_REF': res['arms']['DIAG-SPAN']['delta'],
                              'delta_vs_MBR_X': R6((M['DIAG-SPAN'] - M['MBR-X']).mean())}

    included = [a for a in PRIM if incl[a]['INCLUDED']]
    res['included_primary_arms'] = included
    res['verdict'] = ('VOID' if not all(gates[g]['passed'] for g in
                                        ('G1_reference_reproduces', 'G2_identity_degenerate_posterior',
                                         'G3_shuffle_control_loses',
                                         'G4_independent_control_equals_marginal_sort'))
                      else ('BUILD_RECOMMENDED' if included else 'NO_BUILD'))
    res['seconds'] = round(time.perf_counter() - t0, 1)

    pl.DataFrame({'pair_id': qids, 'half': hv, 'family': fq,
                  **{'ap5_' + a: M[a] for a in ARMS},
                  **{'list_' + a: ['|'.join(LI[a][i]) for i in range(len(qids))] for a in ARMS}
                  }).write_parquet(OUT / 'per_query_step256.parquet')
    assert check_protected() == prot
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({'arms': {a: res['arms'][a] for a in ['REF'] + PRIM + ['CTRL-SHUF', 'CTRL-IND']},
                      'gates': res['gates'], 'verdict': res['verdict'],
                      'expected_ap': res['expected_ap_model_side'],
                      'temp': res['diagnostic_temperature'], 'span': res['diagnostic_span'],
                      'agreement_X_vs_G': res['agreement_MBR_X_vs_MBR_G'],
                      'seconds': res['seconds']}, indent=1), flush=True)


if __name__ == '__main__':
    main()
