"""STEP281 stage A2: the CROSS-TABLE null.

The block-permutation null of stage A failed its own blind control (a statistic with no
mechanism content flagged 126 pairs per 112,540 against a nominal 30).  The reason is
structural: permuting player columns inside a table averages five different people into every
column, so the null loses the person-to-person spread that the real population has, and every
real pair then looks extreme.  OBS 16.5's rule applies -- a negative control that fires
falsifies the test, not the world.

This null keeps each player exactly as they are.  Player a keeps his own table, his own hands,
his own opponents and his own chips; only his PARTNER is replaced by a player from a different
table, aligned by hand index.  Nothing about either person changes; the only thing destroyed is
the alignment a coordinating pair would need.  Victim quantities are read off a's table, which
is the table the hand actually happened at.

The one statistic this null cannot carry is `v_conflict` (aggression between the two members):
players at different tables never act on each other, so it is identically zero here and is
excluded from the cross-table cells.
"""
import argparse
import os

os.environ['POLARS_MAX_THREADS'] = '6'
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step281_a_victim_stats import DATA, EPS, OUT, action_summary, hand_frame  # noqa: E402


def cross_stats(A, B, tidA, tidB):
    """Victim statistics for the 900 pairs (a in table A, b in table B), aligned by hand index.

    Every victim quantity is a row total over A's hand minus a's own entry; b was not at that
    table, so nothing of b's is removed.  The B-minus-S contrast is taken with the identical
    victim definition on both sides, so the one-extra-victim offset cancels.
    """
    n = min(A['V'].shape[0], B['V'].shape[0])
    Va, Ua, Wa = A['V'][:n], A['U'][:n], A['W'][:n]
    Vb, Ub, Wb = B['V'][:n], B['U'][:n], B['W'][:n]
    neta, netb = A['net'][:n], B['net'][:n]
    ffa, n3a, n3b = A['ff'][:n], A['n3'][:n], B['n3'][:n]
    cfa = A['cfold'][:n]

    n_B = Va.T @ Vb
    n_S = Va.T @ Ub + Ua.T @ Vb
    n_sh = Wa.T @ Wb
    nb, ns = np.maximum(n_B, EPS), np.maximum(n_S, EPS)

    def vic(X):
        T = X.sum(axis=1)
        b = (Va * T[:, None]).T @ Vb - (Va * X).T @ Vb
        s = ((Va * T[:, None]).T @ Ub - (Va * X).T @ Ub
             + (Ua * T[:, None]).T @ Vb - (Ua * X).T @ Vb)
        return b, s

    lv_B, lv_S = vic(Va)
    vf_B, vf_S = vic(Va * ffa)
    vb_B, vb_S = vic(cfa)
    vt_B, vt_S = vic(-neta)
    dv_B, dv_S = vic(Wa)
    fa_B, fa_S = vic(ffa)
    lvb, lvs = np.maximum(lv_B, EPS), np.maximum(lv_S, EPS)
    dvb, dvs = np.maximum(dv_B, EPS), np.maximum(dv_S, EPS)

    v_fold_excess = vf_B / lvb - vf_S / lvs
    v_foldany_excess = fa_B / dvb - fa_S / dvs
    v_bleed_excess = vb_B / lvb - vb_S / lvs
    v_take_excess = vt_B / nb - vt_S / ns
    prB = (Va * n3a).T @ Vb + Va.T @ (Vb * n3b)
    prS = (Va * n3a).T @ Ub + Ua.T @ (Vb * n3b)
    v_pressure_excess = prB / nb - prS / ns

    Aa = (n3a > 0).astype(float) * Va
    Ab = (n3b > 0).astype(float) * Vb
    Tff = (Va * ffa).sum(axis=1)
    v_squeeze = ((Aa * Tff[:, None]).T @ Ab - (Aa * Va * ffa).T @ Ab) / nb
    v_bothaggr = (Aa.T @ Ab) / nb

    VnA, VnB = Va * neta, Vb * netb
    sxy = VnA.T @ VnB
    sxa = VnA.T @ Vb
    sxb = Va.T @ VnB
    sxxa = (Va * neta * neta).T @ Vb
    sxxb = Va.T @ (Vb * netb * netb)
    cov = sxy / nb - (sxa / nb) * (sxb / nb)
    va_ = np.maximum(sxxa / nb - (sxa / nb) ** 2, EPS)
    vb_ = np.maximum(sxxb / nb - (sxb / nb) ** 2, EPS)
    v_jointnet_z = np.arctanh(np.clip(cov / np.sqrt(va_ * vb_), -0.999, 0.999))

    loss = Va * np.maximum(-neta, 0.0)
    P = Va.shape[1]
    G = np.empty((P, P, P))
    for c in range(P):
        G[:, :, c] = (Va * loss[:, c][:, None]).T @ Vb
    G[np.arange(P), :, np.arange(P)] = 0.0
    gs = G.sum(axis=2)
    v_conc = G.max(axis=2) / np.maximum(gs, EPS)

    names = ['n_shared', 'n_B', 'n_S', 'v_fold_excess', 'v_foldany_excess', 'v_bleed_excess',
             'v_take_excess', 'v_pressure_excess', 'v_squeeze', 'v_bothaggr', 'v_jointnet_z',
             'v_conc', 'v_victim_take']
    mats = [n_sh, n_B, n_S, v_fold_excess, v_foldany_excess, v_bleed_excess, v_take_excess,
            v_pressure_excess, v_squeeze, v_bothaggr, v_jointnet_z, v_conc, gs / nb]
    d = {'table_id': [f'{tidA}|{tidB}'] * (P * P),
         'p1': [a for a in A['players'] for _ in range(P)],
         'p2': list(B['players']) * P}
    for nm, m in zip(names, mats):
        d[nm] = np.where(np.isfinite(m), m, 0.0).ravel()
    return pl.DataFrame(d)


def build_matrices(seats, tid):
    st = seats.filter(pl.col('table_id') == tid)
    uh = st['hand_id'].unique(maintain_order=True).to_list()
    hidx = {h: i for i, h in enumerate(uh)}
    up = sorted(st['player_id'].unique().to_list())
    pidx = {p: i for i, p in enumerate(up)}
    H, P = len(uh), len(up)
    hi = np.fromiter((hidx[h] for h in st['hand_id'].to_list()), int, st.height)
    pi = np.fromiter((pidx[p] for p in st['player_id'].to_list()), int, st.height)
    bb = np.maximum(st['big_blind'].to_numpy().astype(float), 1.0)
    W = np.zeros((H, P)); W[hi, pi] = 1.0
    V = np.zeros((H, P)); V[hi, pi] = st['vpip'].to_numpy().astype(float)
    net = np.zeros((H, P)); net[hi, pi] = st['net_chips'].to_numpy() / bb
    fold = st['folded'].to_numpy().astype(float)
    cfold = np.zeros((H, P)); cfold[hi, pi] = st['total_contribution'].to_numpy() / bb * fold
    ff = np.zeros((H, P)); ff[hi, pi] = st['fold_faced'].to_numpy().astype(float)
    n3 = np.zeros((H, P)); n3[hi, pi] = st['n_aggr3'].to_numpy().astype(float)
    return {'W': W, 'V': V, 'U': W * (1 - V), 'net': net, 'cfold': cfold, 'ff': ff, 'n3': n3,
            'players': up}


def run(phase, truncate, offset, out_name, batch=40):
    t0 = time.time()
    hands = hand_frame(phase, truncate)
    tables = sorted(hands['table_id'].unique().to_list())
    chunks = []
    for s in range(0, len(tables), batch):
        tb = tables[s:s + batch]
        hb = hands.filter(pl.col('table_id').is_in(tb))
        summ, _ = action_summary(hb.select('hand_id'))
        seats = (pl.scan_parquet(DATA / 'seats.parquet')
                 .select('hand_id', 'player_id', 'net_chips', 'total_contribution', 'folded')
                 .join(hb.select('hand_id', 'table_id', 'big_blind').lazy(),
                       on='hand_id', how='inner')
                 .join(summ.lazy(), on=['hand_id', 'player_id'], how='left')
                 .collect(engine='streaming')
                 .with_columns(pl.col('vpip').fill_null(0), pl.col('n_aggr3').fill_null(0),
                               pl.col('fold_faced').fill_null(0)))
        mats = [build_matrices(seats, t) for t in tb]
        for i, t in enumerate(tb):
            j = (i + offset) % len(tb)
            if j == i:
                continue
            chunks.append(cross_stats(mats[i], mats[j], t, tb[j]))
        del mats
        print(f'  tables {s + len(tb)}/{len(tables)}  {time.time() - t0:.0f}s', flush=True)
    res = pl.concat(chunks)
    res.write_parquet(OUT / out_name)
    print(f'wrote {OUT / out_name}  rows={res.height}  {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--truncate', type=int, default=None)
    ap.add_argument('--offset', type=int, default=7)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run(a.phase, a.truncate, a.offset, a.out)
