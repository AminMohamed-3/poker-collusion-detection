"""STEP281 stage H: the CANCELLING-TRANSFER features, and whether they add to the pair model.

19.234 addendum 2 found that the positives production misses at every history prefix are
directed transfers whose money largely goes back: net/gross .42 against the family's .753.  A
column search of the 899-column production pair frame returned zero matches for session, churn,
cancel, sign, gross, flow, transfer or direction.  hf10 already computes the statistic and OBS
16.3 measured it at DT AUC .956 standalone, higher than net flow's .925, but it has never been
offered to the pair model.

Features, per pair, on the same prefix the fold harness scores (first 1,800 of 3,000 hands):
  churn        per-session absolute chips taken off each other, normalised by money staked
  net_flow     the same summed WITH sign, then absolute -- what a directional feature sees
  churn_ratio  log(churn / net_flow): how much of the movement cancels
  churn_cancel churn - net_flow
  gross_bb     pairwise chips moved between the two, summed absolute over hands
  net_bb       the same summed with sign
  cancel_frac  net_bb / gross_bb -- the statistic the four missed pairs are extreme on

Evaluated with the identical stage-G protocol: table-grouped out-of-fold logistic, AP computed
INSIDE each held-out group, and 20 row-permuted null banks so that "adding columns helps a
logistic by construction" is priced in.  Kill criterion, fixed here before the read: delta below
the shuffled-column q95 on def C, exactly as the victim features were killed.
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
from step281_a_victim_stats import DATA, OUT, hand_frame  # noqa: E402

SESSION_GAP_US = 600 * 1_000_000
EPS = 1e-9


def table_churn(netm, contest, contrib, sess_hand, tid, players):
    H, P = netm.shape
    A = netm * contest
    churn = np.zeros((P, P))
    signed = np.zeros((P, P))
    for s in range(int(sess_hand.max()) + 1):
        m = sess_hand == s
        if not m.any():
            continue
        X = A[m].T @ contest[m]
        D = X - X.T
        churn += np.abs(D)
        signed += D
    scale = np.maximum(contrib.sum(axis=0), 1.0)
    norm = np.sqrt(np.outer(scale, scale))
    churn_n = churn / norm
    net_n = np.abs(signed) / norm
    churn_ratio = np.log((churn_n + 1e-4) / (net_n + 1e-4))
    churn_cancel = churn_n - net_n

    # pairwise chips actually moved between the two, hand by hand
    gain = np.maximum(netm, 0.0)
    loss = np.maximum(-netm, 0.0)
    x = (np.minimum(gain[:, :, None], loss[:, None, :])
         - np.minimum(gain[:, None, :], loss[:, :, None]))
    gross = np.abs(x).sum(axis=0)
    net = np.abs(x.sum(axis=0))
    cancel_frac = net / np.maximum(gross, EPS)

    names = ['churn', 'net_flow', 'churn_ratio', 'churn_cancel', 'gross_bb', 'net_bb',
             'cancel_frac']
    mats = [churn_n, net_n, churn_ratio, churn_cancel, gross, net, cancel_frac]
    iu = np.triu_indices(P, 1)
    d = {'table_id': [tid] * len(iu[0]),
         'p1': [players[i] for i in iu[0]], 'p2': [players[j] for j in iu[1]]}
    for n, m in zip(names, mats):
        m = np.where(np.isfinite(m), m, 0.0)
        d[n] = ((m + m.T) / 2.0)[iu]
    return pl.DataFrame(d)


def build(phase, truncate, out_name, batch=25):
    t0 = time.time()
    hands = hand_frame(phase, truncate)
    tables = sorted(hands['table_id'].unique().to_list())
    chunks = []
    for s in range(0, len(tables), batch):
        tb = tables[s:s + batch]
        hb = hands.filter(pl.col('table_id').is_in(tb))
        seats = (pl.scan_parquet(DATA / 'seats.parquet')
                 .select('hand_id', 'player_id', 'net_chips', 'total_contribution', 'folded')
                 .join(hb.select('hand_id', 'table_id', 'big_blind', 'ts').lazy(),
                       on='hand_id', how='inner')
                 .collect(engine='streaming'))
        for tid in tb:
            st = seats.filter(pl.col('table_id') == tid).sort('ts')
            uh = st.unique(subset='hand_id', maintain_order=True).select('hand_id', 'ts')
            ids = uh['hand_id'].to_list()
            hidx = {h: i for i, h in enumerate(ids)}
            ts = uh['ts'].to_numpy()
            sess = np.r_[0, np.cumsum(np.diff(ts) > SESSION_GAP_US)].astype(int)
            up = sorted(st['player_id'].unique().to_list())
            pidx = {p: i for i, p in enumerate(up)}
            H, P = len(ids), len(up)
            hi = np.fromiter((hidx[h] for h in st['hand_id'].to_list()), int, st.height)
            pi = np.fromiter((pidx[p] for p in st['player_id'].to_list()), int, st.height)
            bb = np.maximum(st['big_blind'].to_numpy().astype(float), 1.0)
            netm = np.zeros((H, P)); netm[hi, pi] = st['net_chips'].to_numpy() / bb
            contest = np.zeros((H, P))
            contest[hi, pi] = (~st['folded'].to_numpy()).astype(float)
            contrib = np.zeros((H, P))
            contrib[hi, pi] = st['total_contribution'].to_numpy() / bb
            chunks.append(table_churn(netm, contest, contrib, sess, tid, up))
        print(f'  {s + len(tb)}/{len(tables)}  {time.time() - t0:.0f}s', flush=True)
    res = pl.concat(chunks)
    res.write_parquet(OUT / out_name)
    print(f'wrote {OUT / out_name} rows={res.height} {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='development')
    ap.add_argument('--truncate', type=int, default=1800)
    ap.add_argument('--out', default='churn_dev_prefix60.parquet')
    a = ap.parse_args()
    build(a.phase, a.truncate, a.out)
