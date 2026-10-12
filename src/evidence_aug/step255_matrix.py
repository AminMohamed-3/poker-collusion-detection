"""STEP255 step 2 -- document frequency, vocabulary and the sparse n-gram design matrix.

Reads the per-chunk int64 n-gram key files written by step255_tokenize.py, computes the document
frequency of every key (label-free), keeps the keys with df >= MIN_DF capped at CAP by df, and
writes a scipy CSR matrix of presence counts in the row order of row_order255.parquet.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import json
import time

import numpy as np
import polars as pl
from scipy import sparse

ROOT = r'C:\Data\Projects\Money Making\work\poker'
OUT = ROOT + r'\Claude Version\experiments\step255_sequence_view_20260917'
TMP = (r'C:\Users\USER\AppData\Local\Temp\claude'
       r'\C--Data-Projects-Money-Making-work-poker'
       r'\f2bb4f17-323f-4806-98b3-984023c70212\scratchpad\step255')
NCHUNK = 24
MIN_DF = 25
CAP = 150_000

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step255_tokenize import decode  # noqa: E402


def main() -> None:
    t0 = time.time()
    indptr = np.load(TMP + r'\indptr.npy')
    nrow = len(indptr) - 1
    chunk_rows = []
    lo = 0
    # chunk boundaries in row space: recover from the per-chunk key file lengths
    lens = []
    for c in range(NCHUNK):
        lens.append(len(np.load(TMP + f'\\keys_{c:02d}.npy', mmap_mode='r')))
    cum = np.cumsum([0] + lens)
    bounds = np.searchsorted(indptr, cum)
    assert bounds[-1] == nrow and bounds[0] == 0, (bounds[:3], bounds[-3:], nrow)
    for c in range(NCHUNK):
        chunk_rows.append((int(bounds[c]), int(bounds[c + 1])))

    # ---- document frequency
    parts = []
    for c in range(NCHUNK):
        k = np.load(TMP + f'\\keys_{c:02d}.npy')
        r0, r1 = chunk_rows[c]
        cnt = np.diff(indptr[r0:r1 + 1])
        rows = np.repeat(np.arange(r0, r1, dtype=np.int64), cnt)
        assert len(rows) == len(k)
        order = np.lexsort((rows, k))
        ks, rs = k[order], rows[order]
        new = np.r_[True, (ks[1:] != ks[:-1]) | (rs[1:] != rs[:-1])]
        parts.append(ks[new])
    allk = np.concatenate(parts)
    del parts
    uniq, df = np.unique(allk, return_counts=True)
    del allk
    print(f'distinct keys {len(uniq):,} df>=1; df>={MIN_DF}: {(df >= MIN_DF).sum():,} '
          f'({time.time() - t0:.0f}s)', flush=True)
    keep = np.flatnonzero(df >= MIN_DF)
    if len(keep) > CAP:
        keep = keep[np.argsort(df[keep])[::-1][:CAP]]
    vocab = np.sort(uniq[keep])
    vdf = df[np.searchsorted(uniq, vocab)]
    print(f'vocabulary {len(vocab):,} keys, df range {vdf.min()}..{vdf.max()}', flush=True)

    # ---- CSR
    data_l, ind_l, ptr = [], [], np.zeros(nrow + 1, dtype=np.int64)
    for c in range(NCHUNK):
        k = np.load(TMP + f'\\keys_{c:02d}.npy')
        r0, r1 = chunk_rows[c]
        cnt = np.diff(indptr[r0:r1 + 1])
        rows = np.repeat(np.arange(r0, r1, dtype=np.int64), cnt)
        pos = np.searchsorted(vocab, k)
        pos[pos >= len(vocab)] = 0
        ok = vocab[pos] == k
        k2, rows2, pos2 = k[ok], rows[ok], pos[ok]
        order = np.lexsort((pos2, rows2))
        rs, cs = rows2[order], pos2[order]
        new = np.r_[True, (rs[1:] != rs[:-1]) | (cs[1:] != cs[:-1])]
        starts = np.flatnonzero(new)
        counts = np.diff(np.r_[starts, len(rs)])
        ind_l.append(cs[starts].astype(np.int32))
        data_l.append(counts.astype(np.float32))
        per_row = np.bincount(rs[starts] - r0, minlength=r1 - r0)
        ptr[r0 + 1:r1 + 1] = per_row
        print(f'chunk {c + 1}/{NCHUNK} nnz={len(starts)} ({time.time() - t0:.0f}s)', flush=True)
    ptr = np.cumsum(ptr)
    X = sparse.csr_matrix((np.concatenate(data_l), np.concatenate(ind_l), ptr),
                          shape=(nrow, len(vocab)))
    X.sort_indices()
    print('X', X.shape, 'nnz', X.nnz, 'mean nnz/row', round(X.nnz / nrow, 1), flush=True)
    sparse.save_npz(OUT + r'\ngrams255.npz', X)
    tok = json.load(open(OUT + r'\tokeniser255.json', encoding='utf-8'))
    inv = tok['symbols']
    txt = [decode(int(k), inv) for k in vocab]
    pl.DataFrame({'col': np.arange(len(vocab), dtype=np.int32), 'key': vocab.astype(np.int64),
                  'df': vdf.astype(np.int64), 'text': txt}).write_parquet(OUT + r'\vocab255.parquet')
    print('DONE', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
