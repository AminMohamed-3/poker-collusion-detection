"""STEP248 Part B runner: count the events per table, score the partner-specific excess, and
build the permutation null.  One phase at a time (the chunk files are already phase-pure).

RELATIVE SEAT OFFSET.  A table seats 30 players six at a time; within a session the six seats
are fixed, so the offset d = (seat_b - seat_a) mod 6 is CONSTANT for the pair inside a session,
and d alone determines how often b acts before a (over a full button orbit b precedes a in
exactly d of 6 hands).  Measured on evaluation table 0: the median pair spends 51 % of its
shared hands at a single offset against 20 % under uniform.  A naive within-hand relabelling
null therefore manufactures "partner specificity" for every order-dependent event.  Both the
baseline and the null here are OFFSET-STRATIFIED:

  expectation  m(a,b,e) = sum_d D[a,b,den(e),d] * p_other(a,e,d),
               p_other(a,e,d) = (sum_{c!=b} C[a,c,e,d] + .5) / (sum_{c!=b} D[a,c,den(e),d] + 1)
               and the cell's evidence is the Poisson upper tail P(Pois(m) >= c), which is
               conservative against the true Poisson-binomial.

  null         for each actor A and each offset d, the map from A's hands to the opponent
               sitting at offset d is randomly permuted ACROSS A's hands.  This preserves A's
               identity, A's own per-hand event pattern, the relative-position structure and
               the exact number of hands A shares with each partner at each offset; it destroys
               only the association between a partner's identity and A's behaviour.

Because every hand is 6-handed with seats 0..5 occupied, local seat index == seat number and
the partner of A at offset d is the player in seat (seat_A + d) mod 6.
"""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_b_events import (EDEN, EVI, NAMES, NCODE, ND, NN, PLACEBO, SCRATCH, DENS, replay)

N_PL = 30
N_OFF = 5
R_PERM = 12
PSIG = 1e-3
IU, JU = np.triu_indices(N_PL, 1)


def load_chunk(tag, ch):
    d = SCRATCH / tag
    return (pl.read_parquet(d / f'h_{ch:03d}.parquet'),
            pl.read_parquet(d / f's_{ch:03d}.parquet'),
            pl.read_parquet(d / f'a_{ch:03d}.parquet'))


def emit_table(hmeta, sfr, afr, record=None):
    """Replay every hand of one table.

    Returns packed emissions, P6 (hand x seat -> table player index), HRANK (hand x seat ->
    the rank of that hand inside that player's own hand list), PART (player x offset x rank ->
    the table player index of the partner at that offset) and nH (hands per player).
    `record` optionally collects (hloc, a, b, code) rows for a set of numerator codes.
    """
    nh = hmeta.height
    hloc = hmeta['hloc'].to_numpy()
    assert (hloc == np.arange(nh)).all(), 'hloc must be the dense within-table index'
    button = hmeta['button_seat'].to_list()
    bb = hmeta['big_blind'].to_list()
    board = [(b or '').split() for b in hmeta['board_cards'].to_list()]
    fpot = hmeta['final_pot'].to_list()
    hidx_h = hmeta['hidx'].to_list()

    s_h = sfr['hidx'].to_numpy()
    s_pl = sfr['ploc'].to_numpy()
    s_seat = sfr['seat_no'].to_numpy()
    s_c1 = sfr['hole_card_1'].to_list()
    s_c2 = sfr['hole_card_2'].to_list()
    s_net = sfr['net_chips'].to_numpy()
    s_fold = sfr['folded'].to_numpy()
    s_sd = sfr['went_to_showdown'].to_numpy()
    s_con = sfr['total_contribution'].to_numpy()
    a_h = afr['hidx'].to_numpy()
    a_pl = afr['ploc'].to_numpy()
    a_st = afr['street'].to_numpy()
    a_kd = afr['action'].to_numpy()
    a_am = afr['amount'].to_numpy()
    a_at = afr['amount_to'].to_numpy()
    a_pb = afr['pot_before'].to_numpy()
    a_tc = afr['to_call'].to_numpy()

    hh = np.array(hidx_h)
    sstart = np.searchsorted(s_h, hh, 'left')
    send = np.searchsorted(s_h, hh, 'right')
    astart = np.searchsorted(a_h, hh, 'left')
    aend = np.searchsorted(a_h, hh, 'right')

    P6 = np.full((nh, 6), -1, np.int32)
    out = []
    for i in range(nh):
        s0, s1 = sstart[i], send[i]
        idx = np.arange(s0, s1)
        seat = s_seat[idx]
        assert (seat == np.arange(6)).all(), 'hands must be 6-handed with seats 0..5'
        ploc = s_pl[idx]
        P6[i] = ploc
        loc = {int(p): k for k, p in enumerate(ploc)}
        acts = []
        for j in range(astart[i], aend[i]):
            acts.append((loc[int(a_pl[j])], int(a_st[j]), int(a_kd[j]), int(a_am[j]),
                         int(a_at[j]), int(a_pb[j]), int(a_tc[j])))
        replay(int(hloc[i]), int(button[i]), int(bb[i]), board[i], int(fpot[i]),
               seat.tolist(), ploc, [s_c1[k] for k in idx], [s_c2[k] for k in idx],
               s_net[idx].tolist(), s_fold[idx].tolist(), s_sd[idx].tolist(),
               s_con[idx].tolist(), acts, out)
    v = np.asarray(out, np.int64)

    HRANK = np.zeros((nh, 6), np.int32)
    nH = np.zeros(N_PL, np.int32)
    maxh = 0
    locs = []
    for A in range(N_PL):
        hi, li = np.where(P6 == A)
        nH[A] = len(hi)
        maxh = max(maxh, len(hi))
        HRANK[hi, li] = np.arange(len(hi))
        locs.append((hi, li))
    PART = np.full((N_PL, 6, maxh), -1, np.int32)
    for A in range(N_PL):
        hi, li = locs[A]
        for d in range(1, 6):
            PART[A, d, :len(hi)] = P6[hi, (li + d) % 6]
    return v, P6, HRANK, PART, nH


def accumulate(v, P6, HRANK, PART, perm=None):
    """Scatter packed emissions into counts[30, 30, NCODE, 5] (last axis = offset d-1)."""
    hl = (v >> 17).astype(np.int64)
    al = ((v >> 14) & 7).astype(np.int64)
    mk = ((v >> 8) & 63).astype(np.int64)
    cd = (v & 255).astype(np.int64)
    size = N_PL * N_PL * NCODE * N_OFF
    tot = np.zeros(size, np.int64)
    A = P6[hl, al].astype(np.int64)
    K = HRANK[hl, al].astype(np.int64)
    for j in range(6):
        sel = (mk >> j) & 1 == 1
        if not sel.any():
            continue
        a = al[sel]
        d = (j - a) % 6
        assert (d != 0).all()
        Asel = A[sel]
        k = K[sel] if perm is None else perm[Asel, d, K[sel]].astype(np.int64)
        bg = PART[Asel, d, k].astype(np.int64)
        ok = bg != Asel
        tot += np.bincount((((Asel[ok] * N_PL + bg[ok]) * NCODE + cd[sel][ok]) * N_OFF
                            + (d[ok] - 1)), minlength=size)
    return tot.reshape(N_PL, N_PL, NCODE, N_OFF)


def score(counts, want_detail=False):
    """Offset-stratified leave-partner-out Poisson tails; pair-level summaries."""
    C4 = counts[:, :, :NN, :].astype(np.float64)
    D4 = counts[:, :, NN:, :][:, :, EDEN, :].astype(np.float64)
    rowC = C4.sum(1)[:, None, :, :]
    rowD = D4.sum(1)[:, None, :, :]
    p = (rowC - C4 + 0.5) / np.maximum(rowD - D4 + 1.0, 1e-9)
    np.clip(p, 1e-9, 1 - 1e-9, out=p)
    m = (D4 * p).sum(3)
    c = C4.sum(3)
    n = D4.sum(3)
    valid = (c >= 2) & (n >= 3) & (c <= n)
    score.violations = int((c > n).sum())
    valid &= ~np.eye(N_PL, dtype=bool)[:, :, None]
    pv = np.ones_like(c)
    if valid.any():
        pv[valid] = poisson.sf(c[valid] - 1, m[valid])
        np.clip(pv, 1e-300, 1.0, out=pv)
    minp = pv.min(2)
    arg = pv.argmin(2)
    nsig = (pv < PSIG).sum(2)
    pl_min = pv[:, :, PLACEBO].min(2)
    pl_n = (pv[:, :, PLACEBO] < PSIG).sum(2)
    detail = None
    if want_detail:
        a, b, e = np.where(pv < PSIG)
        detail = (e, a, b, c[a, b, e], n[a, b, e], m[a, b, e], pv[a, b, e])
    return minp, arg, nsig, pl_min, pl_n, detail


def pair_summary(minp, arg, nsig, pl_min, pl_n):
    mo = np.minimum(minp[IU, JU], minp[JU, IU])
    which = np.where(minp[IU, JU] <= minp[JU, IU], arg[IU, JU], arg[JU, IU])
    direction = (minp[IU, JU] > minp[JU, IU]).astype(np.int8)
    ns = nsig[IU, JU] + nsig[JU, IU]
    pm = np.minimum(pl_min[IU, JU], pl_min[JU, IU])
    pn = pl_n[IU, JU] + pl_n[JU, IU]
    return (-np.log10(np.maximum(mo, 1e-300)), ns, which, direction,
            -np.log10(np.maximum(pm, 1e-300)), pn)


def make_perm(nH, maxh, rng):
    perm = np.zeros((N_PL, 6, maxh), np.int32)
    for A in range(N_PL):
        nh = int(nH[A])
        if nh == 0:
            continue
        for d in range(1, 6):
            perm[A, d, :nh] = rng.permutation(nh)
    return perm


def run_chunk(args):
    tag, ch, seed = args
    t0 = time.time()
    h, s, a = load_chunk(tag, ch)
    rows, nulls, details = [], [], []
    ev_real, ev_null = [], []
    rng = np.random.default_rng(seed)
    viol = 0
    for tid, hm in h.partition_by('tidx', as_dict=True).items():
        t = tid[0] if isinstance(tid, tuple) else tid
        sfr = s.filter(pl.col('tidx') == t).sort(['hidx', 'seat_no'])
        afr = a.filter(pl.col('tidx') == t).sort(['hidx', 'action_no'])
        hm = hm.sort('hloc')
        v, P6, HRANK, PART, nH = emit_table(hm, sfr, afr)
        maxh = PART.shape[2]
        counts = accumulate(v, P6, HRANK, PART)
        minp, arg, nsig, pm, pn, det = score(counts, want_detail=True)
        viol += score.violations
        ev_real.append(np.bincount(det[0], minlength=NN) if len(det[0])
                       else np.zeros(NN, np.int64))
        S1, S2, which, direction, S1p, S2p = pair_summary(minp, arg, nsig, pm, pn)
        nsh = counts[:, :, NN + 0, :].sum(2)[IU, JU]
        rows.append(pl.DataFrame({
            'tidx': np.full(len(IU), t, np.int32), 'a': IU.astype(np.int32),
            'b': JU.astype(np.int32), 'n_shared': nsh.astype(np.int32),
            'S1': S1.astype(np.float32), 'S2': S2.astype(np.int32),
            'best_event': which.astype(np.int32), 'best_dir': direction,
            'S1_plc': S1p.astype(np.float32), 'S2_plc': S2p.astype(np.int32)}))
        if det is not None and len(det[0]):
            e, aa, bb_, cc, dd, mm, vv = det
            details.append(pl.DataFrame({
                'tidx': np.full(len(e), t, np.int32), 'a': aa.astype(np.int32),
                'b': bb_.astype(np.int32), 'event': e.astype(np.int32),
                'c': cc.astype(np.int32), 'n': dd.astype(np.int32),
                'expected': mm.astype(np.float32), 'pval': vv.astype(np.float64)}))
        for r in range(R_PERM):
            perm = make_perm(nH, maxh, rng)
            cnull = accumulate(v, P6, HRANK, PART, perm)
            mp, ar, ns2, pm2, pn2, dn = score(cnull, want_detail=True)
            ev_null.append(np.bincount(dn[0], minlength=NN) if len(dn[0])
                           else np.zeros(NN, np.int64))
            s1n, s2n, _, _, s1pn, s2pn = pair_summary(mp, ar, ns2, pm2, pn2)
            nulls.append(np.stack([s1n, s2n.astype(np.float32),
                                   s1pn, s2pn.astype(np.float32)], 1).astype(np.float32))
    out = SCRATCH / tag / 'out'
    out.mkdir(parents=True, exist_ok=True)
    pl.concat(rows).write_parquet(out / f'pairs_{ch:03d}.parquet')
    np.save(out / f'null_{ch:03d}.npy', np.concatenate(nulls))
    np.save(out / f'evreal_{ch:03d}.npy', np.stack(ev_real))
    np.save(out / f'evnull_{ch:03d}.npy', np.stack(ev_null))
    if details:
        pl.concat(details).write_parquet(out / f'detail_{ch:03d}.parquet')
    else:
        pl.DataFrame(schema={'tidx': pl.Int32, 'a': pl.Int32, 'b': pl.Int32, 'event': pl.Int32,
                             'c': pl.Int32, 'n': pl.Int32, 'expected': pl.Float32,
                             'pval': pl.Float64}).write_parquet(out / f'detail_{ch:03d}.parquet')
    return ch, time.time() - t0, viol


def main(tag, nproc=6, chunks=None):
    t0 = time.time()
    d = SCRATCH / tag
    allch = sorted(int(p.name[2:5]) for p in d.glob('h_*.parquet'))
    if chunks is not None:
        allch = allch[:chunks]
    jobs = [(tag, c, 248_000 + c) for c in allch]
    print(f'{tag}: {len(jobs)} chunks, {NN} events, {ND} denominators, R={R_PERM}', flush=True)
    tv = 0
    with Pool(nproc) as pool:
        for ch, dt, viol in pool.imap_unordered(run_chunk, jobs):
            tv += viol
            print(f'  chunk {ch} done in {dt:.0f}s viol={viol}  [{time.time()-t0:.0f}s]',
                  flush=True)
    print(f'ALL DONE {tag}  total numerator>opportunity violations {tv}  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6,
         int(sys.argv[3]) if len(sys.argv) > 3 else None)
