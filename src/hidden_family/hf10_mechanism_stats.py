"""HF10 stage 1: mechanism-specific pair statistics computed from the RAW hand data.

The three disclosed families are all "in-hand action against a partner who is in the pot",
which is exactly the space the 33 ca_* features model -- and why they reach AUC .99 there.
A fourth mechanism the generator kept out of the public labels is most likely to live
somewhere those features cannot represent at all.  One statistic family per hypothesis,
straight from seats/hands, no model in the loop:

  H1 multi-accounting          mutual pot AVOIDANCE plus session synchrony.  Two seats driven
                               by one human never contest the same pot; our conditional
                               features need the partner IN the hand, so they lose support.
  H2 shared-victim harvesting  a and b take their money from the SAME third players while
                               transferring nothing to each other -- zero pair interaction.
  H3 laundered transfer        a -> c -> b in the directed chip-flow graph; each leg ordinary
                               and the pair itself flat.  We only ever look at (a,b).
  H4 alternating churn         a dumps to b, later b dumps back; net ~ 0, per-session |flow|
                               large.  Directional features cancel exactly.
  H5 selection                 co-presence, seat geometry, joint join/leave beyond chance.
                               Selection is not action (OBS 15.4's one untested class).

Every pair is within one table and every table is 30 players x 3,000 hands x 6 seats, so all
of this is dense 30x30 linear algebra.  Read-only; writes only into experiments/.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import ORIGINAL, CV

OUT = CV / 'experiments' / 'hidden_family_20260912'
SESSION_GAP_US = 600 * 1_000_000        # 10 min of silence at a table ends a session
EPS = 1e-9
N_SEATS = 6


def table_stats(seat_no, pid, net, contrib, folded, showdown, hand_idx, sess_hand,
                n_hands, n_sess, n_pl):
    """Every pair statistic for one table, as 30x30 matrices (symmetric unless noted).

    `sess_hand` is indexed by HAND (length n_hands); the seat-row version is derived here.
    """
    H, S, P = n_hands, n_sess, n_pl
    sess_row = sess_hand[hand_idx]

    present = np.zeros((H, P)); present[hand_idx, pid] = 1.0
    contest = np.zeros((H, P)); contest[hand_idx, pid] = (~folded).astype(float)
    sdm = np.zeros((H, P)); sdm[hand_idx, pid] = showdown.astype(float)
    netm = np.zeros((H, P)); netm[hand_idx, pid] = net
    conm = np.zeros((H, P)); conm[hand_idx, pid] = contrib

    n_both = present.T @ present
    n_solo = present.sum(axis=0)
    c_both = contest.T @ contest
    c_given = contest.T @ present            # [i,j] = i reached the end while j was dealt
    sd_both = sdm.T @ sdm
    sd_given = sdm.T @ present

    # ---- H1: do they duck each other? -------------------------------------------
    # joint rate against the product of the two marginals measured on the SAME hands
    denom = np.maximum(n_both, 1.0)
    exp_c = (c_given * c_given.T) / denom
    avoid_lr = np.log((c_both + 0.5) / (exp_c + 0.5))
    avoid_z = (c_both - exp_c) / np.sqrt(exp_c + 1.0)
    exp_sd = (sd_given * sd_given.T) / denom
    avoid_sd = np.log((sd_both + 0.5) / (exp_sd + 0.5))

    # ---- directed chip flow (H2, H3, H4) ----------------------------------------
    # attribute each hand's losses to its winners proportionally: T[i,k] = chips i took from k
    gain = np.maximum(netm, 0.0)
    loss = np.maximum(-netm, 0.0)
    lw = loss / np.maximum(loss.sum(axis=1, keepdims=True), EPS)
    T = gain.T @ lw
    np.fill_diagonal(T, 0.0)
    flow = T - T.T
    scale = np.maximum(conm.sum(axis=0), 1.0)           # money each player put in overall
    norm = np.sqrt(np.outer(scale, scale))
    direct = np.abs(flow) / norm

    # ---- H2: same third-party sources of profit, excluding each other ------------
    dot = T @ T.T
    sq = (T * T).sum(axis=1)
    # dropping components k=a and k=b leaves the dot product untouched (diagonal is zero)
    # but shrinks each norm by exactly the direct-transfer term.
    na = np.maximum(sq[:, None] - T ** 2, EPS)
    nb = np.maximum(sq[None, :] - (T.T) ** 2, EPS)
    shared_victim = dot / np.sqrt(na * nb)
    np.fill_diagonal(shared_victim, 0.0)
    # cosine over raw chips turned out to be nearly blind under injection (hf12): a planted
    # common victim moves it from .49 to .59.  The overlap coefficient between the two
    # profit-SOURCE distributions is far sharper, because it is scale free and reacts to a
    # single shared component rather than to the whole 28-dimensional vector.
    Sh = T / np.maximum(T.sum(axis=1, keepdims=True), EPS)
    mn = np.minimum(Sh[:, None, :], Sh[None, :, :])
    sv_overlap = mn.sum(axis=2)                         # 1 - total variation distance
    sv_conc = mn.max(axis=2)                            # the single biggest shared source
    np.fill_diagonal(sv_overlap, 0.0)
    np.fill_diagonal(sv_conc, 0.0)

    # ---- H3: best intermediary, a -> c -> b -------------------------------------
    fpos = np.maximum(flow, 0.0) / norm                 # fpos[i,k] = normalised gain of i off k
    chain = np.zeros((P, P))
    for c in range(P):
        leg1 = fpos[c, :][:, None]                      # c gains off a, indexed by a
        leg2 = fpos[:, c][None, :]                      # b gains off c, indexed by b
        np.maximum(chain, np.minimum(leg1, leg2), out=chain)
    np.fill_diagonal(chain, 0.0)
    chain = np.minimum(chain, chain.T)                  # a->c->b or b->c->a, whichever is weaker
    # the max over 28 candidate intermediaries is dominated by sampling noise unless each leg
    # is first standardised by the noise it would have anyway.
    pot = max(float(np.median(np.abs(net[net != 0]))), 1.0) if np.any(net != 0) else 1.0
    zf = flow / (pot * np.sqrt(n_both + 1.0))
    flow_z = np.abs(zf)
    chain_z = np.zeros((P, P))
    for c in range(P):
        np.maximum(chain_z, np.minimum(zf[c, :][:, None], zf[:, c][None, :]), out=chain_z)
    np.fill_diagonal(chain_z, 0.0)
    chain_z = np.minimum(chain_z, chain_z.T)

    # ---- H4: per-session back-and-forth -----------------------------------------
    A = netm * contest
    churn = np.zeros((P, P)); signed = np.zeros((P, P))
    for s in range(S):
        m = sess_hand == s
        if not m.any():
            continue
        X = A[m].T @ contest[m]
        D = X - X.T                                     # chips a took off b within this session
        churn += np.abs(D)
        signed += D
    churn_n = churn / norm
    net_n = np.abs(signed) / norm
    churn_ratio = np.log((churn_n + 1e-4) / (net_n + 1e-4))
    churn_cancel = churn_n - net_n      # the part of the session flow that cancels out

    # ---- H5: selection -----------------------------------------------------------
    exp_both = np.outer(n_solo, n_solo) / max(H, 1)
    copresence = np.log((n_both + 0.5) / (exp_both + 0.5))

    seat_of = np.full((H, N_SEATS), -1, np.int64)
    seat_of[hand_idx, seat_no] = pid
    gapc = np.zeros((N_SEATS - 1, P, P))
    for gv in range(1, N_SEATS):
        i = seat_of.ravel()
        j = np.roll(seat_of, -gv, axis=1).ravel()
        ok = (i >= 0) & (j >= 0)
        cnt = np.bincount(i[ok] * P + j[ok], minlength=P * P).reshape(P, P).astype(float)
        gapc[gv - 1] = cnt + cnt.T                      # gap g one way is 6-g the other
    gtot = np.maximum(gapc.sum(axis=0), 1.0)
    gshare = gapc / gtot
    seat_conc = gshare.max(axis=0) - 1.0 / (N_SEATS - 1)
    adjacency = gshare[0] + gshare[N_SEATS - 2] - 2.0 / (N_SEATS - 1)

    qp = np.zeros((S, P)); qp[sess_row, pid] = 1.0
    q_both = qp.T @ qp
    q_solo = qp.sum(axis=0)
    exp_q = np.outer(q_solo, q_solo) / max(S, 1)
    synchrony = np.log((q_both + 0.5) / (exp_q + 0.5))

    both_win = np.outer(netm.sum(axis=0) > 0, netm.sum(axis=0) > 0).astype(float)

    return {'n_both': n_both, 'contest_both': c_both,
            'avoid_lr': avoid_lr, 'avoid_z': avoid_z, 'avoid_sd': avoid_sd,
            'shared_victim': shared_victim, 'sv_overlap': sv_overlap, 'sv_conc': sv_conc,
            'chain': chain, 'chain_z': chain_z, 'flow_z': flow_z, 'direct': direct,
            'churn': churn_n, 'net_flow': net_n, 'churn_ratio': churn_ratio,
            'churn_cancel': churn_cancel,
            'copresence': copresence, 'seat_conc': seat_conc, 'adjacency': adjacency,
            'synchrony': synchrony, 'both_win': both_win}


def table_arrays(frame):
    """Flat per-seat-row arrays for one table, in a form hf12 can mutate and replay."""
    frame = frame.sort('ts', 'seat_no')
    uh = frame.select('hand_id', 'ts').unique(subset='hand_id').sort('ts')
    hidx = {h: i for i, h in enumerate(uh['hand_id'].to_list())}
    tss = uh['ts'].to_numpy()
    sess = np.r_[0, np.cumsum(np.diff(tss) > SESSION_GAP_US)].astype(int)
    up = frame['player_id'].unique(maintain_order=True).to_list()
    pidx = {p: i for i, p in enumerate(up)}
    return {'seat_no': frame['seat_no'].to_numpy().copy(),
            'pid': np.fromiter((pidx[p] for p in frame['player_id'].to_list()), int,
                               frame.height),
            'net': frame['net_chips'].to_numpy().astype(float),
            'contrib': frame['total_contribution'].to_numpy().astype(float),
            'folded': frame['folded'].to_numpy().copy(),
            'showdown': frame['went_to_showdown'].to_numpy().copy(),
            'hand_idx': np.fromiter((hidx[h] for h in frame['hand_id'].to_list()), int,
                                    frame.height),
            'sess': sess, 'n_hands': len(hidx), 'n_sess': int(sess.max()) + 1,
            'players': up}


def stats_from_arrays(A):
    return table_stats(A['seat_no'], A['pid'], A['net'], A['contrib'], A['folded'],
                       A['showdown'], A['hand_idx'], A['sess'], A['n_hands'], A['n_sess'],
                       len(A['players']))


def build_table(frame):
    A = table_arrays(frame)
    return stats_from_arrays(A), A['players']


def main(phase='development'):
    t0 = time.time()
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet',
                            columns=['hand_id', 'table_id', 'started_at', 'phase'])
    hands = hands.filter(pl.col('phase') == phase).with_columns(
        pl.col('started_at').dt.replace_time_zone(None).cast(pl.Int64).alias('ts')
    ).drop('started_at', 'phase')
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet',
                            columns=['hand_id', 'player_id', 'seat_no', 'net_chips',
                                     'total_contribution', 'folded', 'went_to_showdown'])
    seats = seats.join(hands, on='hand_id', how='inner')
    print(f'{phase}: {hands.height} hands, {seats.height} seat rows, '
          f'{hands["table_id"].n_unique()} tables  [{time.time() - t0:.0f}s]', flush=True)

    parts = seats.partition_by('table_id', as_dict=True)
    recs = []
    for k, (key, frame) in enumerate(parts.items()):
        tid = key[0] if isinstance(key, tuple) else key
        st, up = build_table(frame)
        iu, ju = np.triu_indices(len(up), 1)
        rec = {'table_id': [tid] * len(iu),
               'player_a': [up[i] for i in iu], 'player_b': [up[j] for j in ju]}
        for name, M in st.items():
            rec[name] = M[iu, ju]
        recs.append(pl.DataFrame(rec))
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(parts)} tables  [{time.time() - t0:.0f}s]', flush=True)

    out = pl.concat(recs)
    dest = OUT / f'hf10-mechstats-{phase}.parquet'
    out.write_parquet(dest)
    print(f'{out.height} pairs -> {dest}  [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'development')
