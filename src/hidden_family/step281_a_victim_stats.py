"""STEP281 stage A: victim-centred pair statistics, straight from the raw hand data.

OBS 19.198 closed the fourth family inside the evaluation block with three instruments and
named the one remaining direction verbatim:

    "a VICTIM-centred instrument (what happens to the third player when both members are
     in the pot)"

Every instrument built so far is PAIR-INTERNAL.  `ca_*` are the members' own policy residuals
conditional on the partner being in the hand; hf10's H1-H5 are avoidance, unconditional
profit-source overlap, chain, churn and selection.  None of them asks what the VICTIM does.
OBS 16.2 is the reason this matters: coordinated_isolation "barely interacts and the victim
carries the whole signal", which is why CI sits at .43-.64 on every pair-internal statistic
and at .024 power on the STEP248 event instrument.

Every statistic here conditions on JOINT participation and is contrasted against the SAME
table's hands where only one member is in, which removes "a is simply a strong player" and
"c is simply a fish" without any model in the loop.

    B(a,b) = hands where both a and b voluntarily put money in
    S(a,b) = hands where exactly one of them does while the other is DEALT

Chips are exactly conserved per hand (verified: 0 of 200,000 hands have a non-zero net sum),
so victim chip loss equals pair joint profit by identity; the new content is in the victims'
BEHAVIOUR (what they contribute and then fold), in the CONTRAST, and in the concentration.

Every quantity is a per-hand row sum minus the two members' own entries, so all 435 pairs of
a table are one 30x30 matrix product.  Read-only; writes only into experiments/.
"""
import argparse
import os

os.environ['POLARS_MAX_THREADS'] = '6'
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data'
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
EPS = 1e-9
AGGR = ['bet', 'raise']


def hand_frame(phase: str, truncate: int | None, skip: int = 0):
    """Hands of one phase, with a per-table sequence number so development can be cut to the
    evaluation's length (STEP248 convention)."""
    h = (pl.scan_parquet(DATA / 'hands.parquet')
         .filter(pl.col('phase') == phase)
         .select('hand_id', 'table_id', 'started_at', 'big_blind')
         .with_columns(pl.col('started_at').cast(pl.Int64).alias('ts'))
         .drop('started_at')
         .collect())
    h = h.with_columns(pl.col('ts').rank('ordinal').over('table_id').alias('seq'))
    if skip:
        h = h.filter(pl.col('seq') > skip)
    if truncate is not None:
        h = h.filter(pl.col('seq') <= truncate + skip)
    return h


def action_summary(hand_ids: pl.DataFrame):
    """Per (hand_id, player_id): vpip, aggression, aggression with a third player still live,
    and whether the player folded while facing a bet.  Plus the last-aggressor edges."""
    a = (pl.scan_parquet(DATA / 'actions.parquet')
         .select('hand_id', 'action_no', 'player_id', 'action', 'amount', 'to_call',
                 'players_active')
         .join(hand_ids.lazy(), on='hand_id', how='inner'))
    is_aggr = (pl.col('action').is_in(AGGR)
               | ((pl.col('action') == 'all_in') & (pl.col('amount') > pl.col('to_call'))))
    a = a.with_columns(is_aggr.alias('aggr'),
                       pl.col('action').is_in(AGGR + ['call', 'all_in']).alias('vp'))
    a = a.sort('hand_id', 'action_no')
    # the aggressor a player raised INTO: the previous aggressive actor in the same hand
    a = a.with_columns(pl.when(pl.col('aggr')).then(pl.col('player_id'))
                       .otherwise(None).alias('_ag'))
    a = a.with_columns(pl.col('_ag').shift(1).forward_fill().over('hand_id').alias('prev_ag'))
    frame = a.collect(engine='streaming')

    summ = (frame.group_by('hand_id', 'player_id')
            .agg(pl.col('vp').max().cast(pl.Int8).alias('vpip'),
                 pl.col('aggr').sum().cast(pl.Int16).alias('n_aggr'),
                 (pl.col('aggr') & (pl.col('players_active') >= 3)).sum()
                 .cast(pl.Int16).alias('n_aggr3'),
                 ((pl.col('action') == 'fold') & (pl.col('to_call') > 0)).max()
                 .cast(pl.Int8).alias('fold_faced')))
    edges = (frame.filter(pl.col('aggr') & pl.col('prev_ag').is_not_null()
                          & (pl.col('prev_ag') != pl.col('player_id')))
             .group_by('hand_id', 'player_id', 'prev_ag').len().rename({'len': 'n'}))
    return summ, edges


def table_stats(V, W, netbb, cfold, ff, n3, C, tid, players):
    """All victim statistics for one table's 435 pairs.

    V present-and-vpip (H,P), W dealt (H,P), netbb net chips in bb, cfold contribution in bb
    if the player folded, ff folded while facing a bet, n3 aggressive actions with >= 3 active,
    C the 30x30 last-aggressor edge count.
    """
    U = W * (1.0 - V)                                     # dealt but not in the pot
    A = (n3 > 0).astype(np.float64) * V                   # in the pot AND aggressive
    n_B = V.T @ V
    n_S = V.T @ U + U.T @ V
    n_sh = W.T @ W

    def victim_sums(X, Vw, Uw=None):
        """sum over the weighted hand set of (row total of X, minus the two members' own X)."""
        T = X.sum(axis=1)
        if Uw is None:                                    # B: both weighted by Vw
            tot = (Vw * T[:, None]).T @ Vw
            own = (Vw * X).T @ Vw
            return tot - own - own.T, Vw.T @ Vw
        tot = (Vw * T[:, None]).T @ Uw + (Uw * T[:, None]).T @ Vw
        own = ((Vw * X).T @ Uw + (Uw * X).T @ Vw)
        own = own + ((Vw.T @ (Uw * X)) + (Uw.T @ (Vw * X)))
        return tot - own, Vw.T @ Uw + Uw.T @ Vw

    out = {}
    for tag, args in (('B', (V, None)), ('S', (V, U))):
        lv, _ = victim_sums(V, *args)                     # live victims
        vf, _ = victim_sums(V * ff, *args)                # live victims who folded facing a bet
        vb, _ = victim_sums(cfold, *args)                 # chips the folding victims put in
        vt, _ = victim_sums(-netbb, *args)                # chips the victims lost
        dv, _ = victim_sums(W, *args)                     # dealt victims
        fa, _ = victim_sums(ff, *args)                    # ANY victim folding to a bet,
        out[f'lv_{tag}'], out[f'vf_{tag}'] = lv, vf       # including a blind squeezed preflop
        out[f'vb_{tag}'], out[f'vt_{tag}'] = vb, vt
        out[f'dv_{tag}'], out[f'fa_{tag}'] = dv, fa

    nb = np.maximum(n_B, EPS)
    ns = np.maximum(n_S, EPS)
    lvb = np.maximum(out['lv_B'], EPS)
    lvs = np.maximum(out['lv_S'], EPS)

    # --- the four contrasts ------------------------------------------------------
    dvb = np.maximum(out['dv_B'], EPS)
    dvs = np.maximum(out['dv_S'], EPS)
    v_fold_excess = out['vf_B'] / lvb - out['vf_S'] / lvs
    v_foldany_excess = out['fa_B'] / dvb - out['fa_S'] / dvs
    v_bleed_excess = out['vb_B'] / lvb - out['vb_S'] / lvs
    v_take_excess = out['vt_B'] / nb - out['vt_S'] / ns
    p3 = (V * n3).T @ V
    v_pressure_excess = (p3 + p3.T) / nb - (((V * n3).T @ U) + (U.T @ (V * n3))) / ns

    # --- squeeze: both aggressive, a victim folds facing a bet --------------------
    Tff = (V * ff).sum(axis=1)
    sq_tot = (A * Tff[:, None]).T @ A
    sq_own = (A * V * ff).T @ A
    v_squeeze = (sq_tot - sq_own - sq_own.T) / nb
    v_bothaggr = (A.T @ A) / nb

    # --- do they win together off the victim, or off each other? ------------------
    Vn = V * netbb
    sxy = Vn.T @ Vn
    sx = Vn.T @ V
    sxx = (V * netbb * netbb).T @ V
    cov = sxy / nb - (sx / nb) * (sx.T / nb)
    vx = np.maximum(sxx / nb - (sx / nb) ** 2, EPS)
    r = np.clip(cov / np.sqrt(vx * vx.T), -0.999, 0.999)
    v_jointnet_z = np.arctanh(r)

    # --- concentration of the joint take on ONE victim ----------------------------
    P = V.shape[1]
    loss = V * np.maximum(-netbb, 0.0)
    G = np.empty((P, P, P))
    for c in range(P):
        G[:, :, c] = (V * loss[:, c][:, None]).T @ V
    idx = np.arange(P)
    G[idx, :, idx] = 0.0
    G[:, idx, idx] = 0.0
    gs = G.sum(axis=2)
    v_conc = G.max(axis=2) / np.maximum(gs, EPS)
    v_victim_take = gs / nb

    v_conflict = (C + C.T) / nb

    names = ['n_shared', 'n_B', 'n_S', 'v_fold_excess', 'v_foldany_excess', 'v_bleed_excess',
             'v_take_excess', 'v_pressure_excess', 'v_squeeze', 'v_bothaggr', 'v_jointnet_z',
             'v_conc', 'v_victim_take', 'v_conflict']
    mats = [n_sh, n_B, n_S, v_fold_excess, v_foldany_excess, v_bleed_excess,
            v_take_excess, v_pressure_excess, v_squeeze, v_bothaggr, v_jointnet_z,
            v_conc, v_victim_take, v_conflict]
    iu = np.triu_indices(P, 1)
    data = {'table_id': [tid] * len(iu[0]),
            'p1': [players[i] for i in iu[0]], 'p2': [players[j] for j in iu[1]]}
    for n, m in zip(names, mats):
        m = np.where(np.isfinite(m), m, 0.0)
        data[n] = ((m + m.T) / 2.0)[iu] if n in ('v_jointnet_z',) else m[iu]
    return pl.DataFrame(data)


def block_permute(mats, eh, ei, ej, en, H, P, rng, n_blocks=5):
    """Null that destroys pair coordination while keeping every hand intact.

    The hands are cut into contiguous blocks and the player COLUMN assignment is permuted
    independently inside each block.  Each hand keeps exactly its six dealt players, its
    action counts and its (exactly conserved) chip totals; what is destroyed is the identity
    continuity that a coordinating pair would need.  Any two columns therefore co-participate
    at chance, which is what a null for a pair statistic has to mean.
    """
    edges = np.zeros((P, P))
    bounds = np.linspace(0, H, n_blocks + 1).astype(int)
    out = [np.empty_like(m) for m in mats]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e <= s:
            continue
        perm = rng.permutation(P)
        for m, o in zip(mats, out):
            o[s:e] = m[s:e][:, perm]
        blk = np.zeros((P, P))
        sel = (eh >= s) & (eh < e)
        if sel.any():
            np.add.at(blk, (ei[sel], ej[sel]), en[sel])
        edges += blk[np.ix_(perm, perm)]     # new column i carries old player perm[i]
    return out, edges


def run(phase: str, truncate: int | None, out_name: str, batch: int = 40,
        nullseed: int | None = None, skip: int = 0):
    t0 = time.time()
    hands = hand_frame(phase, truncate, skip)
    tables = sorted(hands['table_id'].unique().to_list())
    print(f'{phase}: {hands.height} hands, {len(tables)} tables', flush=True)
    chunks = []
    for s in range(0, len(tables), batch):
        tb = tables[s:s + batch]
        hb = hands.filter(pl.col('table_id').is_in(tb))
        summ, edges = action_summary(hb.select('hand_id'))
        seats = (pl.scan_parquet(DATA / 'seats.parquet')
                 .select('hand_id', 'player_id', 'net_chips', 'total_contribution', 'folded')
                 .join(hb.select('hand_id', 'table_id', 'big_blind').lazy(),
                       on='hand_id', how='inner')
                 .join(summ.lazy(), on=['hand_id', 'player_id'], how='left')
                 .collect(engine='streaming')
                 .with_columns(pl.col('vpip').fill_null(0), pl.col('n_aggr').fill_null(0),
                               pl.col('n_aggr3').fill_null(0),
                               pl.col('fold_faced').fill_null(0)))
        edges = edges.join(hb.select('hand_id', 'table_id'), on='hand_id', how='inner')
        for tid in tb:
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
            netbb = np.zeros((H, P)); netbb[hi, pi] = st['net_chips'].to_numpy() / bb
            fold = st['folded'].to_numpy().astype(float)
            cfold = np.zeros((H, P))
            cfold[hi, pi] = st['total_contribution'].to_numpy() / bb * fold
            ff = np.zeros((H, P)); ff[hi, pi] = st['fold_faced'].to_numpy().astype(float)
            n3 = np.zeros((H, P)); n3[hi, pi] = st['n_aggr3'].to_numpy().astype(float)
            et = edges.filter(pl.col('table_id') == tid)
            C = np.zeros((P, P))
            eh = ei = ej = en = np.zeros(0, int)
            if et.height:
                ei = np.fromiter((pidx[p] for p in et['player_id'].to_list()), int, et.height)
                ej = np.fromiter((pidx[p] for p in et['prev_ag'].to_list()), int, et.height)
                eh = np.fromiter((hidx[h] for h in et['hand_id'].to_list()), int, et.height)
                en = et['n'].to_numpy().astype(float)
                np.add.at(C, (ei, ej), en)
            if nullseed is None:
                chunks.append(table_stats(V, W, netbb, cfold, ff, n3, C, tid, up))
            else:
                seed = zlib.crc32(f'{tid}|{nullseed}'.encode())   # reproducible across runs
                rng = np.random.default_rng(seed)
                (Wp, Vp, np_, cp, fp, n3p), Cp = block_permute(
                    [W, V, netbb, cfold, ff, n3], eh, ei, ej, en, H, P, rng)
                chunks.append(table_stats(Vp, Wp, np_, cp, fp, n3p, Cp, tid, up))
        print(f'  tables {s + len(tb)}/{len(tables)}  {time.time() - t0:.0f}s', flush=True)
    res = pl.concat(chunks)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet(OUT / out_name)
    print(f'wrote {OUT / out_name}  rows={res.height}  {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['development', 'evaluation'])
    ap.add_argument('--truncate', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nullseed', type=int, default=None)
    ap.add_argument('--skip', type=int, default=0)
    a = ap.parse_args()
    run(a.phase, a.truncate, a.out, nullseed=a.nullseed, skip=a.skip)
