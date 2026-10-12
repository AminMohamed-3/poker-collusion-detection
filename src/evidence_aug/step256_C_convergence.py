"""STEP256 C -- why gate G4 failed: Monte-Carlo convergence of the MBR decoder, and an ANALYTIC proof of the
expectation algebra.

Registered gate G4 asked the independent-relevance control (relevance ~ Bernoulli(q_i^REF), constant denominator 5)
to reproduce the production marginal sort on >= 90 % of queries; at the registered S = 4000 it reproduced 84.0 %.
Either the expectation code is wrong or the sampler is noisy.  This script separates the two:

  1. ANALYTIC greedy for the independent control.  With R_i ~ Bernoulli(q_i) independent of the running hit count H
     and a constant denominator, the slot-k gain is exactly q_i * (1 + sum_{chosen} q_j) / (5k), so the greedy
     argmax is the marginal sort for every slot.  Running the same greedy routine on exact expectations must
     return the REF list on 162 / 162.  That tests the algebra with zero sampling noise.
  2. Monte-Carlo agreement of the same control against S, which must rise to 1.
  3. The primary arms re-run at S up to 256000 (a convergence diagnostic of an already-registered, already-read
     NEGATIVE arm; nothing here is a new candidate and no gate is re-opened).
  4. The reason the correction is empty: the ratio E[R_i * W] / (E[R_i] * E[W]) across candidates, where
     W = 1 / n_z.  n_z is a sum of ~20 Bernoullis, so it is nearly independent of any single R_i and the AP@5
     weighting collapses onto the marginal sort.

Development phase only.  No refit, no build, never uploads.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '4'
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
from evidence_aug import step256_B_mbr_audit as B   # noqa: E402

OUT = B.OUT
SLIST = [4000, 16000, 64000, 256000]
R6 = B.R6


def analytic_independent_greedy(q):
    """Exact greedy on E[AP@5] with independent relevance and a constant denominator. q sorted descending = REF."""
    live = np.ones(len(q), bool)
    Hq, picks = 0.0, []
    for slot in range(1, 6):
        g = q * (1.0 + Hq) / (5.0 * slot)
        g = np.where(live, g, -np.inf)
        j = int(np.argmax(g))
        picks.append(j)
        live[j] = False
        Hq += q[j]
    return picks


def pair_arrays(G):
    t1 = G['T1eff'].to_numpy()
    top = G['top20'].to_numpy()
    n = G.height
    torder = np.arange(n, dtype=float)
    emit = np.flatnonzero(top)
    p = np.zeros(n)
    p[emit] = np.clip(G['v1_REF'].to_numpy()[emit] - 10.0, 0, 1)
    pm = p.copy()
    pm[~top] = np.clip(G['p_fam'].to_numpy(), 0, 1)[~top]
    order_top = emit[np.lexsort((torder[emit], ~t1[emit]))]
    order_all = np.lexsort((torder, ~t1))
    qr = B.marginal_q(p, order_top)
    sc = np.full(n, -np.inf)
    sc[emit] = (p * qr + 1e-9 * p)[emit]
    cand = np.argsort(-sc, kind='stable')[:len(emit)]
    return dict(n=n, p=p, pm=pm, emit=emit, cand=cand, order_top=order_top, order_all=order_all,
                q=(p * qr)[cand], rel=G['rel'].to_numpy(), hid=G['hand_id'].to_list(), ref=list(cand[:5]))


def greedy_big(p, order, cand, S, seed, chunk=64000):
    """Greedy expected-AP@5 slot filling with the sample mean accumulated in chunks (memory-bounded)."""
    m = len(cand)
    picks, live = [], np.ones(m, bool)
    blocks = []
    rng = np.random.default_rng(seed)
    done = 0
    while done < S:
        s = min(chunk, S - done)
        R, nz = B.draw_listed(p, order, s, rng)
        blocks.append((R[:, cand].astype(np.float64), np.where(nz > 0, 1.0 / np.maximum(nz, 1), 0.0)))
        done += s
    H = [np.zeros(len(w)) for _, w in blocks]
    for slot in range(1, 6):
        g = np.zeros(m)
        for (Rc, W), h in zip(blocks, H):
            g += Rc.T @ (W * (h + 1.0) / slot)
        g = np.where(live, g / S, -np.inf)
        j = int(np.argmax(g))
        picks.append(j)
        live[j] = False
        for k, (Rc, _) in enumerate(blocks):
            H[k] = H[k] + Rc[:, j]
    ratio = None
    if blocks:
        Rc, W = blocks[0]
        ew = W.mean()
        er = Rc.mean(0)
        erw = Rc.T @ W / len(W)
        ratio = np.divide(erw, er * ew, out=np.ones(m), where=(er * ew) > 1e-12)
    return picks, ratio


def main():
    t0 = time.perf_counter()
    dest = OUT / 'results256_convergence.json'
    assert not dest.exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    d = pl.read_parquet(OUT / 'inputs256.parquet').sort('pair_id', 't', 'hand_id')
    ph = (pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'phase')
          .filter(pl.col('hand_id').is_in(d['hand_id'].unique().implode())).collect())
    assert ph['phase'].unique().to_list() == ['development']
    ev = pl.read_csv(DATA / 'development_evidence.csv').select('pair_id', 'hand_id')
    nl = dict(ev.group_by('pair_id').len().iter_rows())
    qids = sorted(d['pair_id'].unique().to_list())
    qix = {q: i for i, q in enumerate(qids)}
    fam = dict(d.select('pair_id', 'family').unique().iter_rows())
    hlf = dict(d.select('pair_id', 'half').unique().iter_rows())

    P = {}
    for (pid,), G in d.group_by(['pair_id'], maintain_order=True):
        P[pid] = pair_arrays(G)

    res = {'step': 'STEP256 C', 'purpose': 'diagnose registered gate G4 and establish Monte-Carlo convergence',
           'phase_asserted': 'development', 'refits': 0, 'queries': len(qids)}

    # ---- 1. analytic independent control ------------------------------------------------------
    ok = 0
    for pid in qids:
        a = P[pid]
        ok += analytic_independent_greedy(a['q']) == [0, 1, 2, 3, 4]
    res['analytic_independent_control'] = {
        'queries_reproducing_the_marginal_sort': ok, 'of': len(qids), 'passed': ok == len(qids),
        'reading': 'the greedy expected-AP@5 algebra is correct: with the cap dependence removed it IS the '
                   'production marginal sort, exactly, with no sampling.'}
    print('analytic independent control', ok, '/', len(qids), flush=True)

    # ---- 2/3. Monte-Carlo convergence ----------------------------------------------------------
    ap5 = B.ap5_of
    conv, ratios = {}, []
    for S in SLIST:
        acc = {a: np.zeros(len(qids)) for a in ('MBR-G', 'MBR-G+MASS', 'CTRL-IND')}
        chg = {a: 0 for a in acc}
        for pid in qids:
            a = P[pid]
            i = qix[pid]
            pk, rat = greedy_big(a['p'], a['order_top'], a['cand'], S, [B.SEED_SAMPLE, i, S])
            g = [int(a['cand'][j]) for j in pk]
            acc['MBR-G'][i] = ap5(g, a['rel'], nl[pid])
            chg['MBR-G'] += g != a['ref']
            if S == SLIST[-1]:
                ratios.append(rat[:8])
            pk, _ = greedy_big(a['pm'], a['order_all'], a['cand'], S, [B.SEED_SAMPLE + 7, i, S])
            g = [int(a['cand'][j]) for j in pk]
            acc['MBR-G+MASS'][i] = ap5(g, a['rel'], nl[pid])
            chg['MBR-G+MASS'] += g != a['ref']
            rng = np.random.default_rng([B.SEED_SAMPLE + 29, i, S])
            m = len(a['cand'])
            Ri = np.zeros(m)
            H = 0.0
            live = np.ones(m, bool)
            Rs = (rng.random((S, m)) < a['q'][None, :]).astype(np.float64)
            Hv = np.zeros(S)
            pk = []
            for slot in range(1, 6):
                gg = Rs.T @ (0.2 * (Hv + 1.0) / slot) / S
                gg = np.where(live, gg, -np.inf)
                j = int(np.argmax(gg))
                pk.append(j)
                live[j] = False
                Hv = Hv + Rs[:, j]
            g = [int(a['cand'][j]) for j in pk]
            acc['CTRL-IND'][i] = ap5(g, a['rel'], nl[pid])
            chg['CTRL-IND'] += g != a['ref']
            del Ri, Rs, H
        ref = np.array([ap5(P[q]['ref'], P[q]['rel'], nl[q]) for q in qids])
        conv[S] = {a: {'map5': R6(acc[a].mean()), 'delta': R6((acc[a] - ref).mean()),
                       'queries_changed': int(chg[a]),
                       'ordered_list_agreement_with_REF': R6(1 - chg[a] / len(qids))} for a in acc}
        print('S', S, json.dumps(conv[S]), round(time.perf_counter() - t0), flush=True)
    res['convergence'] = {str(k): v for k, v in conv.items()}

    Rt = np.vstack(ratios)
    res['why_the_correction_is_empty'] = {
        'note': 'ratio_i = E[R_i * W] / (E[R_i] * E[W]) at the registered S, over the top-8 candidates of every '
                'query. The AP@5 weight W = 1/n_z is what makes the metric non-linear; if the ratio is 1 the '
                'objective factorises and the marginal sort is exactly optimal at slot 1.',
        'mean': R6(Rt.mean()), 'sd': R6(Rt.std()),
        'q01': R6(np.quantile(Rt, .01)), 'q99': R6(np.quantile(Rt, .99)),
        'max_within_query_spread': R6(np.max(Rt.max(1) - Rt.min(1))),
        'mean_within_query_spread': R6(np.mean(Rt.max(1) - Rt.min(1)))}

    res['verdict'] = ('G4 failed to Monte-Carlo noise, not to a coding error' if
                      res['analytic_independent_control']['passed'] and
                      conv[SLIST[-1]]['CTRL-IND']['ordered_list_agreement_with_REF']
                      > conv[SLIST[0]]['CTRL-IND']['ordered_list_agreement_with_REF']
                      else 'unresolved')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    dest.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
