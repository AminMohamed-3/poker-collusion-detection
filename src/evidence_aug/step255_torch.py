"""STEP255 ARM T -- a small recurrent encoder that READS the token sequence of a (pair, hand).

Two modes, because the torch interpreter (ROOT\\.venv_tabicl, torch 2.14.0+cpu) has no polars:
  python step255_torch.py export     # main venv: writes meta255.npz (folds / labels / family)
  python step255_torch.py train      # torch venv: writes preds255_T_gru.npz

Multi-task: one encoder per outer fold with four sigmoid heads (any, directed_transfer, soft_play,
coordinated_isolation). The clean contrast is enforced per head by a loss mask: a listed hand of
family f is a positive for the 'any' head and for head f, and is MASKED OUT of the other two heads;
every hand of a confirmed-negative pair is a negative for all four heads; unlisted hands of positive
pairs are excluded from training entirely. Leave-one-out cross-fitting on the step232 table folds.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = r'C:\Data\Projects\Money Making\work\poker'
OUT = ROOT + r'\Claude Version\experiments\step255_sequence_view_20260917'
S232 = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
MODELS = ('any',) + FAMS

EMB = 64
HID = 96
DROP = .2
LR = 2e-3
WD = 1e-4
BATCH = 256
EPOCHS = 15
NEG_PER_EPOCH = 40000
POS_WEIGHT_CAP = 20.0
SEED = 255
THREADS = 5


def export() -> None:
    os.environ.setdefault('POLARS_MAX_THREADS', '4')
    import numpy as np
    import polars as pl
    order = pl.read_parquet(OUT + r'\row_order255.parquet').with_row_index('rid')
    oof = pl.read_parquet(S232)
    j = order.join(oof, on=['pair_id', 'hand_id'], how='left').sort('rid')
    assert j.height == order.height and j['fold'].null_count() == 0
    famc = np.zeros(j.height, dtype=np.int8)
    for i, f in enumerate(FAMS):
        famc[(j['behavior_family'].to_numpy() == f)] = i + 1
    np.savez(OUT + r'\meta255.npz', fold=j['fold'].to_numpy().astype(np.int8),
             lbl=j['label'].to_numpy().astype(np.int8),
             listed=j['listed'].to_numpy().astype(np.int8), fam=famc)
    print('wrote meta255.npz', j.height)


def train() -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    z = np.load(OUT + r'\seq255.npz')
    seq, length = z['seq'].astype(np.int64), z['length'].astype(np.int64)
    m = np.load(OUT + r'\meta255.npz')
    fold, lbl, listed, fam = m['fold'], m['lbl'], m['listed'], m['fam']
    N = len(fold)
    V = int(seq.max()) + 1
    L = int(length.max())
    seq = seq[:, :L]
    print('rows', N, 'vocab', V, 'maxlen', L, 'mean len', round(float(length.mean()), 1), flush=True)

    Y = np.zeros((N, 4), dtype=np.float32)
    M = np.zeros((N, 4), dtype=np.float32)
    honest = lbl == 0
    Y[listed == 1, 0] = 1.
    M[honest | (listed == 1), 0] = 1.
    for i, f in enumerate(FAMS):
        sel = (listed == 1) & (fam == i + 1)
        Y[sel, i + 1] = 1.
        M[honest | sel, i + 1] = 1.
    elig = (M.sum(1) > 0)
    print('training-eligible rows', int(elig.sum()), 'positives per head', Y.sum(0).tolist(),
          flush=True)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, EMB, padding_idx=0)
            self.gru = nn.GRU(EMB, HID, num_layers=1, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(DROP)
            self.head = nn.Linear(4 * HID, 4)

        def forward(self, x):
            msk = (x != 0)
            o, _ = self.gru(self.emb(x))
            om = o.masked_fill(~msk.unsqueeze(-1), 0.)
            mean = om.sum(1) / msk.sum(1, keepdim=True).clamp(min=1)
            mx = o.masked_fill(~msk.unsqueeze(-1), -1e4).max(1).values
            return self.head(self.drop(torch.cat([mean, mx], -1)))

    def batches(idx, bs, shuffle):
        ln = length[idx]
        o = np.argsort(ln, kind='stable')
        idx = idx[o]
        starts = np.arange(0, len(idx), bs)
        if shuffle:
            rng.shuffle(starts)
        for s in starts:
            b = idx[s:s + bs]
            w = int(length[b].max())
            yield b, torch.from_numpy(seq[b, :max(w, 1)])

    preds = {k: np.full(N, np.nan, dtype=np.float32) for k in MODELS}
    for k in range(5):
        tr_all = np.flatnonzero(elig & (fold != k))
        te = np.flatnonzero(fold == k)
        pos_idx = tr_all[listed[tr_all] == 1]
        neg_idx = tr_all[lbl[tr_all] == 0]
        net = Enc()
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        npos = np.maximum(Y[tr_all].sum(0), 1.)
        nneg = min(NEG_PER_EPOCH, len(neg_idx))
        pw = np.minimum(nneg / npos, POS_WEIGHT_CAP).astype(np.float32)
        lossf = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.from_numpy(pw))
        Yt, Mt = torch.from_numpy(Y), torch.from_numpy(M)
        nstep = 0
        for ep in range(EPOCHS):
            sub = np.concatenate([pos_idx, rng.choice(neg_idx, size=min(NEG_PER_EPOCH,
                                                                        len(neg_idx)),
                                                      replace=False)])
            net.train()
            tot, nb = 0., 0
            for b, xb in batches(sub, BATCH, True):
                opt.zero_grad()
                out = net(xb)
                mm = Mt[b]
                ls = (lossf(out, Yt[b]) * mm).sum() / mm.sum().clamp(min=1)
                ls.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                tot += float(ls)
                nb += 1
                nstep += 1
            print(f'fold {k} epoch {ep} loss {tot / max(nb, 1):.4f} steps {nstep} '
                  f'{time.time() - t0:.0f}s', flush=True)
        net.eval()
        with torch.no_grad():
            for b, xb in batches(te, 1024, False):
                p = torch.sigmoid(net(xb)).numpy()
                for i, nm in enumerate(MODELS):
                    preds[nm][b] = p[:, i]
        print(f'fold {k} done {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(OUT + r'\preds255_T_gru.npz',
                        **{f'loo_T_gru_{k}': v for k, v in preds.items()})
    print('DONE', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    (export if sys.argv[1:2] == ['export'] else train)()
