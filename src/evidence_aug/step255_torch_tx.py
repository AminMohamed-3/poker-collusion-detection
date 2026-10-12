"""STEP255 ARM T_tx -- a 4-layer transformer encoder over the same (pair, hand) token sequence,
on the RTX 3050 Ti via ROOT\\.venv_mhc (torch 2.11.0+cu128).

Secondary arm (registration255.json): it may not enter STACK_lr and cannot change the G1 verdict.
Same clean contrast, same per-head loss mask and the same leave-one-out cross-fitting on the
step232 table folds as ARM T_gru. Reads seq255.npz and meta255.npz, writes preds255_T_tx.npz.
No downloads, no installs: weights are initialised from scratch.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = r'C:\Data\Projects\Money Making\work\poker'
OUT = ROOT + r'\Claude Version\experiments\step255_sequence_view_20260917'
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
MODELS = ('any',) + FAMS

DM = 128
NHEAD = 4
FF = 256
NLAYER = 4
DROP = .1
LR = 3e-4
WD = 1e-2
BATCH = 256
EPOCHS = 15
NEG_PER_EPOCH = 40000
POS_WEIGHT_CAP = 20.0
SEED = 255


class TX(nn.Module):
    def __init__(self, V, L):
        super().__init__()
        self.emb = nn.Embedding(V, DM, padding_idx=0)
        self.pos = nn.Embedding(L + 1, DM)
        lay = nn.TransformerEncoderLayer(DM, NHEAD, FF, dropout=DROP, activation='gelu',
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, NLAYER)
        self.drop = nn.Dropout(DROP)
        self.head = nn.Linear(2 * DM, 4)

    def forward(self, x):
        pad = x == 0
        p = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.emb(x) * math.sqrt(DM) + self.pos(p)
        h = self.enc(h, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1)
        hm = h.masked_fill(~keep, 0.)
        mean = hm.sum(1) / keep.sum(1).clamp(min=1)
        mx = h.masked_fill(~keep, -1e4).max(1).values
        return self.head(self.drop(torch.cat([mean, mx], -1)))


def main() -> None:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '', flush=True)
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
    honest = lbl == 0
    Y = np.zeros((N, 4), dtype=np.float32)
    M = np.zeros((N, 4), dtype=np.float32)
    Y[listed == 1, 0] = 1.
    M[honest | (listed == 1), 0] = 1.
    for i, f in enumerate(FAMS):
        sel = (listed == 1) & (fam == i + 1)
        Y[sel, i + 1] = 1.
        M[honest | sel, i + 1] = 1.
    elig = M.sum(1) > 0
    Yt, Mt = torch.from_numpy(Y), torch.from_numpy(M)
    print('rows', N, 'vocab', V, 'maxlen', L, 'eligible', int(elig.sum()), flush=True)

    def batches(idx, bs, shuffle):
        o = np.argsort(length[idx], kind='stable')
        idx = idx[o]
        starts = np.arange(0, len(idx), bs)
        if shuffle:
            rng.shuffle(starts)
        for s in starts:
            b = idx[s:s + bs]
            w = max(int(length[b].max()), 1)
            yield b, torch.from_numpy(seq[b, :w]).to(dev)

    preds = {k: np.full(N, np.nan, dtype=np.float32) for k in MODELS}
    for k in range(5):
        tr_all = np.flatnonzero(elig & (fold != k))
        te = np.flatnonzero(fold == k)
        pos_idx = tr_all[listed[tr_all] == 1]
        neg_idx = tr_all[lbl[tr_all] == 0]
        net = TX(V, L).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        nneg = min(NEG_PER_EPOCH, len(neg_idx))
        pw = np.minimum(nneg / np.maximum(Y[tr_all].sum(0), 1.), POS_WEIGHT_CAP).astype(np.float32)
        lossf = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.from_numpy(pw).to(dev))
        for ep in range(EPOCHS):
            sub = np.concatenate([pos_idx, rng.choice(neg_idx, size=nneg, replace=False)])
            net.train()
            tot, nb = 0., 0
            for b, xb in batches(sub, BATCH, True):
                opt.zero_grad(set_to_none=True)
                out = net(xb)
                mm = Mt[b].to(dev)
                ls = (lossf(out, Yt[b].to(dev)) * mm).sum() / mm.sum().clamp(min=1)
                ls.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                tot += float(ls.detach())
                nb += 1
            if ep % 5 == 4 or ep == 0:
                print(f'fold {k} epoch {ep} loss {tot / max(nb, 1):.4f} {time.time() - t0:.0f}s',
                      flush=True)
        net.eval()
        with torch.no_grad():
            for b, xb in batches(te, 1024, False):
                p = torch.sigmoid(net(xb)).float().cpu().numpy()
                for i, nm in enumerate(MODELS):
                    preds[nm][b] = p[:, i]
        print(f'fold {k} done {time.time() - t0:.0f}s', flush=True)
        del net, opt
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    np.savez_compressed(OUT + r'\preds255_T_tx.npz',
                        **{f'loo_T_tx_{k}': v for k, v in preds.items()})
    print('DONE', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
