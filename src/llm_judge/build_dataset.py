"""STEP264: build train.jsonl / heldout.jsonl and the two key parquets.

Development phase only (asserted inside render_hand.HandContext).  Split by the step232 table
folds: TRAIN = folds 0,1,2, HELD-OUT = folds 3,4.  A fold is a set of whole tables and a hand
belongs to exactly one table, so no table, pair or hand can straddle the split; that is asserted
rather than assumed.

Classes - see registration264.json for the full statement.
  TRAIN     POS (all listed hands), NEG_RANDOM 1200, NEG_HARD 1200, NEG_WITHIN <=1400
  HELD-OUT  POS (all listed hands), NEG_RANDOM 3000, NEG_HARD 1500, NEG_WITHIN all, AUDIT

NEG_WITHIN uses the fact established 2026-09-18: the 32 positive pairs with only 3 or 4 listed
evidence hands have COMPLETE lists, so their unlisted hands are certified ordinary hands of a
colluding pair.  Unlisted hands of 5-listed pairs are AMBIGUOUS and are used neither way.

keys_train.parquet / keys_heldout.parquet map uid -> pair_id, hand_id.  They NEVER go to the GPU
box; the JSONL carries only the opaque uid.

Usage:  python build_dataset.py            (full build)
        python build_dataset.py --probe N  (render N rows only, print the length distribution)
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import hashlib
import json
import sys
import time

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import render_hand as R                                          # noqa: E402

CV = R.CV
OUT = CV + r'\experiments\step264_llm_hand_judge_20260918'
OOF254 = CV + r'\experiments\step254_mechanical_tensor_20260917\hand_oof_step254.parquet'
K = ['pair_id', 'hand_id']
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
SALT = 'step264'
TRAIN_FOLDS, HELD_FOLDS = (0, 1, 2), (3, 4)
N_TR_RANDOM, N_TR_HARD, CAP_TR_WITHIN = 1200, 1200, 1400
N_HO_RANDOM, N_HO_HARD = 3000, 1500
HARD_Q = .98                       # top 2 % of mechplus_p_fmax among clean-pair rows in the side
SEEDS = {'a_is_player_1': 264, 'neg_random': 2641, 'neg_hard': 2642, 'neg_within': 2643,
         'shuffle_train': 264, 'shuffle_heldout': 2644}


def uid_of(pair_id: str, hand_id: str) -> str:
    return hashlib.sha1(f'{pair_id}|{hand_id}|{SALT}'.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------- key assembly


def base_keys() -> pl.DataFrame:
    sys.path.insert(0, CV + r'\src')
    from evidence_aug import step254_eval as E
    h, _ = E.load()
    h = h.select('pair_id', 'hand_id', 'table_id', 'label', 'behavior_family', 'listed', 'fold')
    oof = pl.read_parquet(OOF254, columns=K + ['mechplus_p_fmax'])
    h = h.join(oof, on=K, how='left')
    assert h['mechplus_p_fmax'].null_count() == 0, 'mechplus join left nulls'
    assert h.height == 223749, h.height
    return h


def audit_rows() -> pl.DataFrame:
    sys.path.insert(0, CV + r'\src')
    sys.path.insert(0, R.ROOT + r'\Codex Resume\src')
    from evidence_aug import step165_C_e2e_audit as S165
    Q = S165.load()['Q']
    return Q.select('pair_id', 'hand_id', 'relevance').unique(subset=K)


def short_list_pairs(h: pl.DataFrame) -> pl.DataFrame:
    """The positive pairs with 3 or 4 listed hands: COMPLETE evidence lists."""
    pp = (h.filter(pl.col('label') == 1).group_by('pair_id')
          .agg(pl.col('listed').sum().alias('n_listed'), pl.col('fold').first()))
    return pp.filter(pl.col('n_listed') < 5)


def stratified_by_pair(df: pl.DataFrame, cap: int, seed: int) -> pl.DataFrame:
    """Equal quota per pair, remainder filled round-robin, deterministic order."""
    if df.height <= cap:
        return df
    rng = np.random.default_rng(seed)
    df = df.sort(K)
    buckets: dict[str, list[int]] = {}
    for i, p in enumerate(df['pair_id'].to_list()):
        buckets.setdefault(p, []).append(i)
    for p in buckets:
        idx = np.array(buckets[p])
        rng.shuffle(idx)
        buckets[p] = idx.tolist()
    keep: list[int] = []
    pos = 0
    names = sorted(buckets)
    while len(keep) < cap:
        added = False
        for p in names:
            if pos < len(buckets[p]):
                keep.append(buckets[p][pos])
                added = True
                if len(keep) == cap:
                    break
        if not added:
            break
        pos += 1
    return df[sorted(keep)]


def build_side(h: pl.DataFrame, folds: tuple, n_random: int, n_hard: int, cap_within: int | None,
               audit: pl.DataFrame | None, seed_off: int) -> pl.DataFrame:
    side = h.filter(pl.col('fold').is_in(list(folds)))
    clean = side.filter(pl.col('label') == 0)

    pos = (side.filter(pl.col('listed') == 1)
           .select(*K, 'table_id', 'fold', 'behavior_family')
           .with_columns(pl.lit(1).alias('label'), pl.lit('POS').alias('source_class'))
           .rename({'behavior_family': 'family'}))

    rng = np.random.default_rng(SEEDS['neg_random'] + seed_off)
    cl = clean.sort(K)
    pick = rng.choice(cl.height, size=min(n_random, cl.height), replace=False)
    neg_r = (cl[np.sort(pick)].select(*K, 'table_id', 'fold')
             .with_columns(pl.lit(0).alias('label'), pl.lit('NONE').alias('family'),
                           pl.lit('NEG_RANDOM').alias('source_class')))

    thr = float(np.quantile(cl['mechplus_p_fmax'].to_numpy(), HARD_Q))
    hard_pool = cl.filter(pl.col('mechplus_p_fmax') >= thr).sort(K)
    rng = np.random.default_rng(SEEDS['neg_hard'] + seed_off)
    pick = rng.choice(hard_pool.height, size=min(n_hard, hard_pool.height), replace=False)
    neg_h = (hard_pool[np.sort(pick)].select(*K, 'table_id', 'fold')
             .with_columns(pl.lit(0).alias('label'), pl.lit('NONE').alias('family'),
                           pl.lit('NEG_HARD').alias('source_class')))
    # a hand can be drawn by both rules; NEG_HARD wins, the row is not duplicated
    neg_r = neg_r.join(neg_h.select(K), on=K, how='anti')

    sl = short_list_pairs(h).filter(pl.col('fold').is_in(list(folds))).select('pair_id')
    within = (side.join(sl, on='pair_id').filter(pl.col('listed') == 0)
              .select(*K, 'table_id', 'fold'))
    if cap_within is not None:
        within = stratified_by_pair(within, cap_within, SEEDS['neg_within'] + seed_off)
    neg_w = within.with_columns(pl.lit(0).alias('label'), pl.lit('NONE').alias('family'),
                                pl.lit('NEG_WITHIN').alias('source_class'))

    out = pl.concat([pos.select(*K, 'table_id', 'fold', 'label', 'family', 'source_class'),
                     neg_r.select(*K, 'table_id', 'fold', 'label', 'family', 'source_class'),
                     neg_h.select(*K, 'table_id', 'fold', 'label', 'family', 'source_class'),
                     neg_w.select(*K, 'table_id', 'fold', 'label', 'family', 'source_class')])
    assert out.select(K).n_unique() == out.height, 'duplicate key inside a side'

    if audit is not None:
        aud = (audit.join(h.select(*K, 'table_id', 'fold'), on=K)
               .filter(pl.col('fold').is_in(list(folds)))
               .join(out.select(K), on=K, how='anti')
               .with_columns(pl.col('relevance').cast(pl.Int32).alias('label'),
                             pl.lit('NONE').alias('family'),
                             pl.lit('AUDIT').alias('source_class'))
               .select(*K, 'table_id', 'fold', 'label', 'family', 'source_class'))
        out = pl.concat([out, aud])
    assert out.select(K).n_unique() == out.height, 'duplicate key after the audit rows'
    return out.sort(K)


def attach_uid_and_role(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    df = df.sort(K)
    rng = np.random.default_rng(seed)
    a1 = rng.integers(0, 2, df.height).astype(bool)
    uids = [uid_of(p, hh) for p, hh in zip(df['pair_id'].to_list(), df['hand_id'].to_list())]
    return df.with_columns(pl.Series('uid', uids), pl.Series('a_is_player_1', a1))


# ---------------------------------------------------------------------------- main


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    probe = int(sys.argv[sys.argv.index('--probe') + 1]) if '--probe' in sys.argv else 0
    h = base_keys()
    Q = audit_rows()
    tr = build_side(h, TRAIN_FOLDS, N_TR_RANDOM, N_TR_HARD, CAP_TR_WITHIN, None, 0)
    ho = build_side(h, HELD_FOLDS, N_HO_RANDOM, N_HO_HARD, None, Q, 10)
    tr = attach_uid_and_role(tr, SEEDS['a_is_player_1'])
    ho = attach_uid_and_role(ho, SEEDS['a_is_player_1'] + 1)

    # ---- disjointness assertions
    for col in ('table_id', 'pair_id', 'hand_id'):
        a, b = set(tr[col].to_list()), set(ho[col].to_list())
        assert not (a & b), f'{col} appears on both sides: {sorted(a & b)[:3]}'
    assert set(tr['fold'].to_list()) <= set(TRAIN_FOLDS)
    assert set(ho['fold'].to_list()) <= set(HELD_FOLDS)
    assert len(set(tr['uid'].to_list())) == tr.height
    assert len(set(ho['uid'].to_list())) == ho.height
    assert not (set(tr['uid'].to_list()) & set(ho['uid'].to_list()))

    if probe:
        tr = tr.sample(min(probe, tr.height), seed=1, shuffle=True).sort(K)
        ho = ho.sample(min(probe, ho.height), seed=1, shuffle=True).sort(K)

    ctx, render = R.make_renderer(pl.concat([tr.select(K), ho.select(K)]))
    print(f'context loaded in {time.time() - t0:.0f}s: '
          f'{tr.height} train rows, {ho.height} held-out rows', flush=True)

    stats = {}
    for name, df, shuffle_seed in (('train', tr, SEEDS['shuffle_train']),
                                   ('heldout', ho, SEEDS['shuffle_heldout'])):
        rows, lens = [], []
        for i, r in enumerate(df.iter_rows(named=True)):
            txt = render(r['pair_id'], r['hand_id'], bool(r['a_is_player_1']))
            lens.append(len(txt))
            rows.append({'uid': r['uid'], 'prompt': txt, 'label': int(r['label']),
                         'family': r['family'] if int(r['label']) == 1 else 'NONE',
                         'source_class': r['source_class'], 'fold': int(r['fold']),
                         'a_is_player_1': bool(r['a_is_player_1'])})
            if (i + 1) % 2000 == 0:
                print(f'  {name}: {i + 1}/{df.height}  {time.time() - t0:.0f}s', flush=True)
        order = np.random.default_rng(shuffle_seed).permutation(len(rows))
        rows = [rows[j] for j in order]
        if not probe:
            with open(f'{OUT}\\{name}.jsonl', 'w', encoding='utf-8', newline='\n') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=True) + '\n')
            df.select('uid', 'pair_id', 'hand_id', 'table_id', 'fold', 'label', 'family',
                      'source_class', 'a_is_player_1').write_parquet(f'{OUT}\\keys_{name}.parquet')
        toks = np.array(lens) / 3.6
        stats[name] = {
            'n': len(rows),
            'by_class': {k: int(v) for k, v in
                         zip(*np.unique(df['source_class'].to_numpy(), return_counts=True))},
            'by_class_family': {f'{c}|{fm}': int(n) for c, fm, n in
                                df.group_by('source_class', 'family').len().sort(
                                    'source_class', 'family').iter_rows()},
            'label_1': int(df['label'].sum()),
            'est_tokens': {q: round(float(np.quantile(toks, q / 100)), 1)
                           for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
            'est_tokens_mean': round(float(toks.mean()), 1),
            'share_over_800': round(float((toks > 800).mean()), 4),
            'share_under_350': round(float((toks < 350).mean()), 4),
        }
        print(name, json.dumps(stats[name], indent=1), flush=True)
    if not probe:
        json.dump(stats, open(f'{OUT}\\build_stats264.json', 'w', encoding='utf-8'), indent=1)
    print('done in', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
