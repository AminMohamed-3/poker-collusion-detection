"""STEP248 Part A (HF15): the hf10 mechanism battery on the EVALUATION phase, plus a
length-matched development null (first 2,000 hands per table).

Differences from src/hidden_family/hf10_mechanism_stats.py, all additive:

  * `chain_sym`  = the ORIGINAL hf10 statistic, np.minimum(chain, chain.T).  It is identically
                   zero for a one-way laundering chain a -> c -> b, which is why OBS 16.4 found
                   it blind at 4x planted strength.  Kept as the declared BLIND NEGATIVE CONTROL.
  * `chain_max`  = the CORRECTED directed 2-path statistic, np.maximum(chain, chain.T): the
                   stronger of a -> c -> b and b -> c -> a.  Tested here for the first time.
  * `chain_z_sym` / `chain_z_max` the same for the noise-standardised legs.
  * a `limit` argument truncating each table to its first N hands in started_at order, so the
    development population can be length-matched to evaluation's 2,000 hands per table.

Phase is asserted on every hand row.  Read-only on the original tree; writes only into
Claude Version/experiments/step248_hidden_family_eval_20260917/.
"""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ORIGINAL = Path('C:/Data/Projects/Money Making/work/poker')
CV = ORIGINAL / 'Claude Version'
OUT = CV / 'experiments' / 'step248_hidden_family_eval_20260917'
SESSION_GAP_US = 600 * 1_000_000
EPS = 1e-9
N_SEATS = 6


def table_stats(seat_no, pid, net, contrib, folded, showdown, hand_idx, sess_hand,
                n_hands, n_sess, n_pl):
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
    c_given = contest.T @ present
    sd_both = sdm.T @ sdm
    sd_given = sdm.T @ present

    denom = np.maximum(n_both, 1.0)
    exp_c = (c_given * c_given.T) / denom
    avoid_lr = np.log((c_both + 0.5) / (exp_c + 0.5))
    avoid_z = (c_both - exp_c) / np.sqrt(exp_c + 1.0)
    exp_sd = (sd_given * sd_given.T) / denom
    avoid_sd = np.log((sd_both + 0.5) / (exp_sd + 0.5))

    gain = np.maximum(netm, 0.0)
    loss = np.maximum(-netm, 0.0)
    lw = loss / np.maximum(loss.sum(axis=1, keepdims=True), EPS)
    T = gain.T @ lw
    np.fill_diagonal(T, 0.0)
    flow = T - T.T
    scale = np.maximum(conm.sum(axis=0), 1.0)
    norm = np.sqrt(np.outer(scale, scale))
    direct = np.abs(flow) / norm

    dot = T @ T.T
    sq = (T * T).sum(axis=1)
    na = np.maximum(sq[:, None] - T ** 2, EPS)
    nb = np.maximum(sq[None, :] - (T.T) ** 2, EPS)
    shared_victim = dot / np.sqrt(na * nb)
    np.fill_diagonal(shared_victim, 0.0)
    Sh = T / np.maximum(T.sum(axis=1, keepdims=True), EPS)
    mn = np.minimum(Sh[:, None, :], Sh[None, :, :])
    sv_overlap = mn.sum(axis=2)
    sv_conc = mn.max(axis=2)
    np.fill_diagonal(sv_overlap, 0.0)
    np.fill_diagonal(sv_conc, 0.0)

    # ---- H3 laundering: best intermediary, a -> c -> b --------------------------------
    fpos = np.maximum(flow, 0.0) / norm
    chain = np.zeros((P, P))
    for c in range(P):
        leg1 = fpos[c, :][:, None]      # c gains off a, indexed by a
        leg2 = fpos[:, c][None, :]      # b gains off c, indexed by b
        np.maximum(chain, np.minimum(leg1, leg2), out=chain)
    np.fill_diagonal(chain, 0.0)
    chain_sym = np.minimum(chain, chain.T)   # ORIGINAL hf10 (blind control)
    chain_max = np.maximum(chain, chain.T)   # CORRECTED directed 2-path

    pot = max(float(np.median(np.abs(net[net != 0]))), 1.0) if np.any(net != 0) else 1.0
    zf = flow / (pot * np.sqrt(n_both + 1.0))
    flow_z = np.abs(zf)
    chain_z = np.zeros((P, P))
    for c in range(P):
        np.maximum(chain_z, np.minimum(zf[c, :][:, None], zf[:, c][None, :]), out=chain_z)
    np.fill_diagonal(chain_z, 0.0)
    chain_z_sym = np.minimum(chain_z, chain_z.T)
    chain_z_max = np.maximum(chain_z, chain_z.T)

    A = netm * contest
    churn = np.zeros((P, P)); signed = np.zeros((P, P))
    for s in range(S):
        m = sess_hand == s
        if not m.any():
            continue
        X = A[m].T @ contest[m]
        D = X - X.T
        churn += np.abs(D)
        signed += D
    churn_n = churn / norm
    net_n = np.abs(signed) / norm
    churn_ratio = np.log((churn_n + 1e-4) / (net_n + 1e-4))
    churn_cancel = churn_n - net_n

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
        gapc[gv - 1] = cnt + cnt.T
    gtot = np.maximum(gapc.sum(axis=0), 1.0)
    gshare = gapc / gtot
    seat_conc = gshare.max(axis=0) - 1.0 / (N_SEATS - 1)
    adjacency = gshare[0] + gshare[N_SEATS - 2] - 2.0 / (N_SEATS - 1)

    qp = np.zeros((S, P)); qp[sess_row, pid] = 1.0
    q_both = qp.T @ qp
    q_solo = qp.sum(axis=0)
    exp_q = np.outer(q_solo, q_solo) / max(S, 1)
    synchrony = np.log((q_both + 0.5) / (exp_q + 0.5))

    return {'n_both': n_both, 'contest_both': c_both,
            'avoid_lr': avoid_lr, 'avoid_z': avoid_z, 'avoid_sd': avoid_sd,
            'shared_victim': shared_victim, 'sv_overlap': sv_overlap, 'sv_conc': sv_conc,
            'chain_sym': chain_sym, 'chain_max': chain_max,
            'chain_z_sym': chain_z_sym, 'chain_z_max': chain_z_max,
            'flow_z': flow_z, 'direct': direct,
            'churn': churn_n, 'net_flow': net_n, 'churn_ratio': churn_ratio,
            'churn_cancel': churn_cancel,
            'copresence': copresence, 'seat_conc': seat_conc, 'adjacency': adjacency,
            'synchrony': synchrony}


def build_table(frame):
    frame = frame.sort('ts', 'seat_no')
    uh = frame.select('hand_id', 'ts').unique(subset='hand_id').sort('ts')
    hidx = {h: i for i, h in enumerate(uh['hand_id'].to_list())}
    tss = uh['ts'].to_numpy()
    sess = np.r_[0, np.cumsum(np.diff(tss) > SESSION_GAP_US)].astype(int)
    up = frame['player_id'].unique(maintain_order=True).to_list()
    pidx = {p: i for i, p in enumerate(up)}
    st = table_stats(
        frame['seat_no'].to_numpy().copy(),
        np.fromiter((pidx[p] for p in frame['player_id'].to_list()), int, frame.height),
        frame['net_chips'].to_numpy().astype(float),
        frame['total_contribution'].to_numpy().astype(float),
        frame['folded'].to_numpy().copy(),
        frame['went_to_showdown'].to_numpy().copy(),
        np.fromiter((hidx[h] for h in frame['hand_id'].to_list()), int, frame.height),
        sess, len(hidx), int(sess.max()) + 1, len(up))
    return st, up


def load_hands(phase, limit=None):
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet',
                            columns=['hand_id', 'table_id', 'started_at', 'phase'])
    hands = hands.filter(pl.col('phase') == phase)
    assert hands.height > 0
    assert (hands['phase'] == phase).all(), 'phase assertion failed'
    hands = hands.with_columns(pl.col('started_at').dt.epoch('us').alias('ts')).drop(
        'started_at', 'phase')
    if limit is not None:
        hands = hands.sort('ts').group_by('table_id', maintain_order=True).head(limit)
    return hands


def main(phase='evaluation', limit=None, tag=None):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    hands = load_hands(phase, limit)
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet',
                            columns=['hand_id', 'player_id', 'seat_no', 'net_chips',
                                     'total_contribution', 'folded', 'went_to_showdown'])
    seats = seats.join(hands, on='hand_id', how='inner')
    per_tab = hands.group_by('table_id').len()
    print(f'{phase} limit={limit}: {hands.height} hands, {seats.height} seat rows, '
          f'{hands["table_id"].n_unique()} tables, hands/table '
          f'{per_tab["len"].min()}-{per_tab["len"].max()}  [{time.time() - t0:.0f}s]', flush=True)

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
    tag = tag or (phase if limit is None else f'{phase}{limit}')
    dest = OUT / f'step248a-mechstats-{tag}.parquet'
    out.write_parquet(dest)
    print(f'{out.height} pairs -> {dest}  [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    ph = sys.argv[1] if len(sys.argv) > 1 else 'evaluation'
    lim = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'none' else None
    main(ph, lim)
