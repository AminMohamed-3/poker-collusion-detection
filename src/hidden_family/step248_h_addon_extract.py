"""STEP248 ADD-ON, stage 1: extract the FULL 126-event partner-excess vector for every pair in
groups Z, A and H.

The instrument is the frozen STEP248 Part B scorer, unchanged (offset-stratified leave-partner-out
Poisson tail).  Only the extraction is new: the main run stored just the cells with p < 1e-3,
while a three-group comparison needs every cell for the selected pairs.

Groups (defined in registration248_addon.json, written first):
  Z  block pairs (STEP207 risk >= .5) with the coordinator's cnt999 == 0
  A  block pairs with risk >= .99 and cnt999 >= 3      (known-family positive control)
  H  4,000 random tail pairs with risk < .01, seed 248250  (honest negative control)

Evaluation phase only; the chunk files carry the phase assertion from step248_b_prep.
"""
import os

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_b_events import NAMES, NN, SCRATCH
from step248_b_run import (N_PL, accumulate, emit_table, load_chunk, score)
from step248_c_common import ORIGINAL, OUT, attach_players, load_phase

TABLES_PER_CHUNK = 10
STEP250 = (ORIGINAL / 'Claude Version' / 'experiments' /
           'step250_event_count_tail_20260917' / 'eval_pair_event_counts.parquet')
SUB207 = (ORIGINAL / 'Claude Version' / 'outputs_step207_20260917' /
          'submission_step207_ens4_ci_dt_rolestrength_on_step169.csv')
SEED_H = 248250
N_H = 4000


def build_groups():
    pairs, _, _, _ = load_phase('evaluation')
    pairs = attach_players(pairs, 'evaluation')
    ep = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p_lo'),
        pl.max_horizontal('player_1', 'player_2').alias('p_hi'))
    sub = pl.read_csv(SUB207)
    cnt = pl.read_parquet(STEP250).select('pair_id', 'cnt999', 'cnt999_directed_transfer',
                                          'cnt999_soft_play',
                                          'cnt999_coordinated_isolation')
    ep = ep.join(sub, on='pair_id', how='inner').join(cnt, on='pair_id', how='inner')
    cand = pairs.join(ep.select('p_lo', 'p_hi', 'pair_id', 'shared_hands', 'risk_score',
                                'predicted_behavior', 'cnt999', 'evidence_hand_1',
                                'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4',
                                'evidence_hand_5'),
                      on=['p_lo', 'p_hi'], how='inner')
    assert cand.height == 112540
    hi99 = cand.filter(pl.col('risk_score') >= 0.99)
    hip = set(hi99['player_a'].to_list()) | set(hi99['player_b'].to_list())
    cand = cand.with_columns(
        (pl.col('player_a').is_in(list(hip)) | pl.col('player_b').is_in(list(hip)))
        .alias('shares_hi99_player'))
    Z = cand.filter((pl.col('risk_score') >= 0.5) & (pl.col('cnt999') == 0))
    A = cand.filter((pl.col('risk_score') >= 0.99) & (pl.col('cnt999') >= 3))
    rng = np.random.default_rng(SEED_H)
    tail = cand.filter(pl.col('risk_score') < 0.01)
    idx = rng.choice(tail.height, size=N_H, replace=False)
    H = tail[np.sort(idx)]
    Z = Z.with_columns(pl.lit('Z').alias('grp'))
    A = A.with_columns(pl.lit('A').alias('grp'))
    H = H.with_columns(pl.lit('H').alias('grp'))
    return cand, pl.concat([Z, A, H])


def run_chunk(args):
    ch, sel = args
    h, s, a = load_chunk('evaluation', ch)
    rows = []
    for tid, hm in h.partition_by('tidx', as_dict=True).items():
        t = tid[0] if isinstance(tid, tuple) else tid
        g = sel.filter(pl.col('tidx') == t)
        if not g.height:
            continue
        sfr = s.filter(pl.col('tidx') == t).sort(['hidx', 'seat_no'])
        afr = a.filter(pl.col('tidx') == t).sort(['hidx', 'action_no'])
        v, P6, HR, PART, nH = emit_table(hm.sort('hloc'), sfr, afr)
        counts = accumulate(v, P6, HR, PART)
        C4 = counts[:, :, :NN, :].astype(np.float64)
        Dall = counts[:, :, NN:, :]
        from step248_b_events import EDEN
        D4 = Dall[:, :, EDEN, :].astype(np.float64)
        rowC = C4.sum(1)[:, None, :, :]
        rowD = D4.sum(1)[:, None, :, :]
        p = (rowC - C4 + 0.5) / np.maximum(rowD - D4 + 1.0, 1e-9)
        np.clip(p, 1e-9, 1 - 1e-9, out=p)
        m = (D4 * p).sum(3)
        c = C4.sum(3)
        n = D4.sum(3)
        from scipy.stats import poisson
        valid = (c >= 2) & (n >= 3) & (c <= n)
        pv = np.ones_like(c)
        pv[valid] = poisson.sf(c[valid] - 1, m[valid])
        np.clip(pv, 1e-300, 1.0, out=pv)
        for r in g.iter_rows(named=True):
            for (x, y, d) in ((r['a'], r['b'], 0), (r['b'], r['a'], 1)):
                rows.append(pl.DataFrame({
                    'pair_id': [r['pair_id']] * NN, 'grp': [r['grp']] * NN,
                    'direction': np.full(NN, d, np.int8),
                    'event': np.arange(NN, dtype=np.int32),
                    'c': c[x, y].astype(np.int32), 'n': n[x, y].astype(np.int32),
                    'expected': m[x, y].astype(np.float32), 'pval': pv[x, y]}))
    return pl.concat(rows) if rows else None


def main():
    t0 = time.time()
    cand, grp = build_groups()
    print('groups:', grp.group_by('grp').len().sort('grp').to_dicts())
    grp.write_parquet(OUT / 'step248addon-groups.parquet')
    cand.select('pair_id', 'tidx', 'a', 'b', 'player_a', 'player_b', 'table_id', 'n_shared',
                'risk_score', 'predicted_behavior', 'cnt999', 'shares_hi99_player',
                'evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4',
                'evidence_hand_5').write_parquet(OUT / 'step248addon-eval-pairs.parquet')
    sel = grp.select('tidx', 'a', 'b', 'pair_id', 'grp')
    chunks = sorted(set((sel['tidx'] // TABLES_PER_CHUNK).to_list()))
    jobs = [(ch, sel.filter((pl.col('tidx') // TABLES_PER_CHUNK) == ch)) for ch in chunks]
    print(f'{len(jobs)} chunks to rescore  [{time.time()-t0:.0f}s]', flush=True)
    out = []
    with Pool(6) as pool:
        for k, fr in enumerate(pool.imap_unordered(run_chunk, jobs)):
            if fr is not None:
                out.append(fr)
            if (k + 1) % 10 == 0:
                print(f'  {k+1}/{len(jobs)}  [{time.time()-t0:.0f}s]', flush=True)
    res = pl.concat(out)
    res = res.with_columns(pl.Series('event_name', [NAMES[i] for i in res['event'].to_list()]))
    res.write_parquet(OUT / 'step248addon-cells.parquet')
    print(f'{res.height} cells -> step248addon-cells.parquet  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
