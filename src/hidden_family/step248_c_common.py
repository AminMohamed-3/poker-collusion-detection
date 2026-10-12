"""STEP248 Part B/C shared loading: pair scores, the permutation null aligned to them, and
the phase-pure identity maps."""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')
from pathlib import Path

import numpy as np
import polars as pl

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_b_events import NAMES, SCRATCH
from step248_b_run import R_PERM

ORIGINAL = Path('C:/Data/Projects/Money Making/work/poker')
CV = ORIGINAL / 'Claude Version'
OUT = CV / 'experiments' / 'step248_hidden_family_eval_20260917'
N_PAIRS_TAB = 435
EVAL_POP = 112540


def load_phase(tag):
    """Pair-level scores for one phase plus the permutation null, aligned row for row.

    Returns (pairs, null) where null has columns [S1, S2, S1_plc, S2_plc] and the parallel
    arrays null_nshared / null_tidx so it can be stratified exactly like the real pairs.
    """
    d = SCRATCH / tag / 'out'
    chunks = sorted(int(p.name[6:9]) for p in d.glob('pairs_*.parquet'))
    P, Nl, Nsh, Ntid = [], [], [], []
    for ch in chunks:
        pr = pl.read_parquet(d / f'pairs_{ch:03d}.parquet')
        nl = np.load(d / f'null_{ch:03d}.npy')
        ntab = pr.height // N_PAIRS_TAB
        assert pr.height == ntab * N_PAIRS_TAB
        assert nl.shape[0] == ntab * R_PERM * N_PAIRS_TAB, (nl.shape, ntab)
        ns = pr['n_shared'].to_numpy().reshape(ntab, N_PAIRS_TAB)
        td = pr['tidx'].to_numpy().reshape(ntab, N_PAIRS_TAB)
        Nsh.append(np.repeat(ns[:, None, :], R_PERM, axis=1).ravel())
        Ntid.append(np.repeat(td[:, None, :], R_PERM, axis=1).ravel())
        P.append(pr)
        Nl.append(nl)
    pairs = pl.concat(P)
    null = np.concatenate(Nl)
    return pairs, null, np.concatenate(Nsh), np.concatenate(Ntid)


def attach_players(pairs, tag):
    pmap = pl.read_parquet(SCRATCH / tag / 'players.parquet')
    tmap = pl.read_parquet(SCRATCH / tag / 'tables.parquet')
    pa = pmap.rename({'ploc': 'a', 'player_id': 'player_a'})
    pb = pmap.rename({'ploc': 'b', 'player_id': 'player_b'})
    out = pairs.join(pa, on=['tidx', 'a'], how='left').join(pb, on=['tidx', 'b'], how='left')
    out = out.join(tmap, on='tidx', how='left')
    assert out['player_a'].null_count() == 0 and out['player_b'].null_count() == 0
    return out.with_columns(
        pl.min_horizontal('player_a', 'player_b').alias('p_lo'),
        pl.max_horizontal('player_a', 'player_b').alias('p_hi'))


def load_detail(tag):
    d = SCRATCH / tag / 'out'
    fr = [pl.read_parquet(p) for p in sorted(d.glob('detail_*.parquet'))]
    fr = [f for f in fr if f.height]
    return pl.concat(fr) if fr else None


def strata(n_shared, edges):
    return np.clip(np.digitize(n_shared, edges), 0, len(edges))


def null_thresholds(null_s1, null_ns, edges, expected_flags, pop=EVAL_POP):
    """Per-stratum S1 threshold whose permutation-null exceedance rate gives `expected_flags`
    false flags in a population of `pop` pairs."""
    q = 1.0 - expected_flags / pop
    st = strata(null_ns, edges)
    return np.array([np.quantile(null_s1[st == s], q) for s in range(len(edges) + 1)])
