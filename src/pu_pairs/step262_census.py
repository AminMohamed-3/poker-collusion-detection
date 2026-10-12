"""STEP262 addendum arm CENSUS (registration262_addendum_census.json).

A PURE PROMOTION, no fit: on each development surface the TAIL pairs (prefix risk < .5) that clear
the census rule are ordered by cnt_q999 descending, ties by top3_logit descending, and inserted as a
block immediately below the .5 seam; every other pair keeps its place in the CURRENT stack's order.
Implemented by lifting those pairs' key strictly above max(current key) and running the unchanged
step159c.permute, so the metric code path is identical to every other arm.

  variant A (PRIMARY, can authorise a file)  cnt_q999 >= 3
  variant B (reported only)                  binomial p < 1e-4 for cnt_q999 given n hands, rate .001

Baseline = the stack step262_arms.py's greedy ended on.  Development labels only.

Usage:  python step262_census.py
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
import sys, json, time, hashlib
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl                                        # noqa: E402
from scipy.stats import binom                                           # noqa: E402
from pu_pairs import step262_repro_tail as H                            # noqa: E402
from pu_pairs import step262_arms as A                                  # noqa: E402
from pu_pairs import step159c_C_combine_dev as S159                     # noqa: E402

OUT = H.OUT
SURF = H.SURF
NBOOT = A.NBOOT
RATE_B = .001
ALPHA_B = 1e-4
MIN_A = 3


def log(*a):
    print(*a, flush=True)


def prep(d, base_key):
    """the arrays every readout needs, for one surface."""
    s = d['risk'].to_numpy().astype(float)
    lab = d['label'].to_numpy()
    rf = d['risk_full'].fill_null(0).to_numpy().astype(float)
    r1 = d['R1_full'].fill_null(False).to_numpy()
    yr = (lab == 1).astype(int)
    unl = lab == -1
    yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int)
    yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    return {'s': s, 'lab': lab, 'yr': yr, 'yp': yp, 'yc': yc, 'r1': r1,
            'fam': d['behavior_family'].to_numpy(), 'pred': d['predicted_class'].to_numpy(),
            'tail': s < .5, 'key': base_key, 'table': d['table_id'].to_numpy(), 'fold': d['fold'].to_numpy(),
            'pair': np.array(d['pair_id'].to_list()),
            'cnt': d['mechplus_cnt_q999'].to_numpy().astype(float),
            'top3': d['mechplus_top3_logit'].to_numpy().astype(float),
            'n': d['n_hands'].to_numpy().astype(float)}


def measure_arr(P, key):
    sc = S159.permute(P['s'], key, P['tail'])
    return {'PP90': S159.ap(P['yp'], sc), 'PPcls': S159.ap(P['yc'], sc), 'RAW': S159.ap(P['yr'], sc),
            'B': S159.bmap(sc, P['lab'], P['fam'], P['pred'])}


def promo_key(P, sel):
    """lift the selected pairs above every other key, ordered by cnt desc then top3 desc.
    cnt*100 + top3 is exactly lexicographic because |top3_logit| <= 13.82 < 50."""
    z = P['key'].copy()
    if sel.any():
        v = P['cnt'] * 100.0 + P['top3']
        z[sel] = P['key'].max() + 1.0 + (v[sel] - v[sel].min())
    return z


def select(P, variant):
    if variant == 'A':
        sel = P['cnt'] >= MIN_A
    else:
        p = binom.sf(P['cnt'] - 1, P['n'].astype(int), RATE_B)
        sel = p < ALPHA_B
    return sel & P['tail']


def boot_keys(PP, kA, kB, n=NBOOT, seed=262):
    pre = {}
    for p in SURF:
        P = PP[p]
        tab = P['table']
        order = np.argsort(tab, kind='stable')
        starts = np.flatnonzero(np.r_[True, tab[order][1:] != tab[order][:-1]])
        rows = [order[a:b] for a, b in zip(starts, np.r_[starts[1:], len(order)])]
        tfold = np.array([P['fold'][r[0]] for r in rows])
        pre[p] = (P['s'], P['yp'], kA[p], kB[p], rows, tfold)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for b in range(n):
        acc = 0.0
        for p in SURF:
            s, yp, a, bb, rows, tfold = pre[p]
            pick = []
            for f in (1, 2):
                idx = np.flatnonzero(tfold == f)
                if len(idx):
                    pick.append(rng.choice(idx, size=len(idx), replace=True))
            ix = np.concatenate([rows[i] for i in np.concatenate(pick)])
            ss, yy = s[ix], yp[ix]
            blk = ss < .5
            acc += S159.ap(yy, S159.permute(ss, a[ix], blk)) - S159.ap(yy, S159.permute(ss, bb[ix], blk))
        out[b] = acc / len(SURF)
    return {'P_gt_0': float((out > 0).mean()), 'p05': float(np.percentile(out, 5)), 'p50': float(np.percentile(out, 50)),
            'p95': float(np.percentile(out, 95)), 'mean': float(out.mean()), 'n': int(n)}


def moves(P, kA, kB, lowdose):
    idx = np.flatnonzero(P['tail'])
    rA = np.empty(len(idx), np.int64); rA[np.argsort(-kA[idx], kind='mergesort')] = np.arange(len(idx))
    rB = np.empty(len(idx), np.int64); rB[np.argsort(-kB[idx], kind='mergesort')] = np.arange(len(idx))
    mv = (rA - rB).astype(float)
    r1, lab, pid = P['r1'][idx], P['lab'][idx], P['pair'][idx]
    ld = np.isin(pid, list(lowdose))
    return {'n_tail': int(len(idx)), 'R1_pairs': int(r1.sum()),
            'mean_move_R1': float(mv[r1].mean()) if r1.any() else None,
            'mean_move_nonR1': float(mv[~r1].mean()),
            'n_lowdose_pos': int(ld.sum()), 'mean_move_lowdose_pos': float(mv[ld].mean()) if ld.any() else None,
            'n_labelled_pos': int((lab == 1).sum()),
            'mean_move_labelled_pos': float(mv[lab == 1].mean()) if (lab == 1).any() else None}


def main():
    t0 = time.time()
    dest = OUT + 'results262_census.json'
    assert not os.path.exists(dest), dest
    reg = OUT + 'registration262_addendum_census.json'
    res = json.load(open(OUT + 'results262.json', encoding='utf-8'))
    rec = {'addendum_sha256': hashlib.sha256(open(reg, 'rb').read()).hexdigest(),
           'results262_sha256': hashlib.sha256(open(OUT + 'results262.json', 'rb').read()).hexdigest(),
           'baseline_stack': res['stack_after_greedy'], 'kept_types_from_main_run': res['kept_types']}

    dev = pl.read_parquet(OUT + 'aggregates262.parquet')
    tr = pl.read_parquet(OUT + 'train_aggs262.parquet')
    V = H.stack_V('arch3')
    D0 = H.build_D(V)
    M = A.scale_map(D0)
    extra = []
    for t in res['kept_types']:
        if t == 'MECH_T2':
            raw = dev['mechplus_top3_logit'].to_numpy().astype(float)
        elif t == 'MECH_T3':
            raw, _ = A.fit_scores_t3(dev, tr)
        else:
            raw, _ = A.fit_scores(dev, tr, 'plus' if t.startswith('PLUS') else 'mechplus')
        extra.append(A.typecol(dev, raw, t, M))
    D = H.build_D(V, extra=extra) if extra else D0
    stack = res['stack_after_greedy']
    base_abs = H.measure(D, stack)
    passed = [g for g in res['greedy'] if g['passes']]
    want = passed[-1]['abs'] if passed else res['FINAL2_16_abs']
    rec['baseline_reproduction_max_abs_diff'] = max(abs(base_abs[p][m] - want[p][m]) for p in want for m in want[p])
    assert rec['baseline_reproduction_max_abs_diff'] < 1e-12, rec['baseline_reproduction_max_abs_diff']
    rec['baseline_abs'] = base_abs

    ag = dev.select('pair_id', 'surface', 'n_hands', 'mechplus_cnt_q999', 'mechplus_top3_logit')
    PP, KB = {}, {}
    for p in SURF:
        d = D[p].join(ag.filter(pl.col('surface') == p).drop('surface'), on='pair_id', how='left',
                      validate='1:1', maintain_order='left')
        assert d['mechplus_cnt_q999'].null_count() == 0 and d.height == D[p].height
        bk = d.select(pl.mean_horizontal(stack)).to_series().to_numpy()
        PP[p] = prep(d, bk)
        KB[p] = bk
        assert abs(measure_arr(PP[p], bk)['PP90'] - base_abs[p]['PP90']) < 1e-12

    ld = A.lowdose_pairs()
    rec['variants'] = {}
    for variant in ('A', 'B'):
        sel = {p: select(PP[p], variant) for p in SURF}
        K = {p: promo_key(PP[p], sel[p]) for p in SURF}
        cand = {p: measure_arr(PP[p], K[p]) for p in SURF}
        r = S159.rule({p: {'a': cand[p], 'b': base_abs[p]} for p in cand}, 'a', 'b')
        r['variant'] = variant
        r['abs'] = cand
        r['per_surface'] = {p: {m: cand[p][m] - base_abs[p][m] for m in cand[p]} for p in cand}
        r['promoted'] = {p: {'n': int(sel[p].sum()),
                             'n_PP90_positive': int(PP[p]['yp'][sel[p]].sum()),
                             'n_PPcls_positive': int(PP[p]['yc'][sel[p]].sum()),
                             'n_labelled_positive': int((PP[p]['lab'][sel[p]] == 1).sum()),
                             'n_R1': int(PP[p]['r1'][sel[p]].sum()),
                             'cnt_q999_range': [float(PP[p]['cnt'][sel[p]].min()), float(PP[p]['cnt'][sel[p]].max())] if sel[p].any() else None,
                             'base_risk_median': float(np.median(PP[p]['s'][sel[p]])) if sel[p].any() else None,
                             'base_rank_median_in_tail': float(np.median(
                                 np.argsort(np.argsort(-KB[p][PP[p]['tail']]))[sel[p][PP[p]['tail']]])) if sel[p].any() else None}
                         for p in SURF}
        r['bootstrap'] = boot_keys(PP, K, KB)
        r['soft_rule_S'] = bool(r['bootstrap']['P_gt_0'] >= .75)
        r['rank_moves'] = {p: moves(PP[p], K[p], KB[p], ld) for p in SURF}
        rec['variants'][variant] = r
        log('CENSUS', variant, json.dumps({k: v for k, v in r.items() if k not in ('abs', 'per_surface', 'rank_moves')}))

    a = rec['variants']['A']
    rec['CENSUS_joins'] = bool(a['passes'] and a['soft_rule_S'])
    rec['noise_control_passes'] = bool(res['controls']['NOISE_T1']['passes'])
    rec['build_authorised'] = bool(rec['CENSUS_joins'] and not rec['noise_control_passes'])
    if rec['CENSUS_joins']:
        rec['prediction_vs_step251_file'] = {
            'point': .65 * .37 * a['mean_comp'], 'band80_halfwidth': .0025,
            'honest_note': 'the three tail reads were +.0015 / +.0015 / +.0009 and the read SE is about .00065'}
    rec['seconds'] = round(time.time() - t0, 1)
    json.dump(rec, open(dest, 'w', encoding='utf-8'), indent=1)
    log('CENSUS joins', rec['CENSUS_joins'], 'build_authorised', rec['build_authorised'], rec['seconds'], 's')


if __name__ == '__main__':
    main()
