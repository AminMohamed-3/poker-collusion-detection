"""STEP268 A -- build the ENCODER universe: one mechanical document per (pair, hand).

Registration: experiments/step268_qwen06b_encoder_heads_20260918/registration268.json (frozen
before this file was written; nothing here changes it).

THE UNIVERSE (development phase only, asserted inside render_hand.HandContext)

  1. POOL          for EVERY labelled development pair (1,860; 372 positive) the TOP-40 hands by
                   the OOF MECHPLUS event probability `mechplus_p_fmax` of
                   experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet,
                   ties broken by hand_id ascending; a pair with fewer than 40 hands takes all
                   of them.  in_pool = True.  THE SAME RULE FOR POSITIVE AND CLEAN PAIRS, so a
                   pooled feature cannot encode the pair label through the hand count - that is
                   asserted, not assumed.
  2. EXTRA_LISTED  every listed evidence hand that the top-40 of its own pair missed.
                   in_pool = False, extra_listed = True (excluded from pooling so the pooled
                   features stay a fixed-size read of the same rule on every pair).
  3. KEYS          every uid of STEP264's keys_train.parquet / keys_heldout.parquet that rules 1
                   and 2 did not already bring in, so the evidence heads can be read on exactly
                   the same held-out rows as STEP264 / STEP267.  in_pool = False.

uid: where (pair_id, hand_id) is a STEP264 key row the STEP264 uid and its a_is_player_1 are
REUSED verbatim (asserted by recomputing uid_of on those rows); every other row gets
uid_of(pair_id, hand_id) (sha1, SALT 'step264' - the same function, so a uid means the same hand
everywhere in the project) and an a_is_player_1 drawn from np.random.default_rng(268).

WHAT LEAVES THIS MACHINE (data/, minus keys_universe.parquet)
  universe_docs.jsonl.gz   {"uid", "doc"}            - doc = render_hand_v2.body, nothing else
  universe_meta.jsonl      {"uid", "pair_uid", "fold", "label_pair", "listed", "family",
                            "in_pool", "extra_listed", "source_class"}
  heldout_scope57_uids.txt / heldout_subset_uids.txt  - uid order of the STEP264 JSONL files
  instruction.txt          - ONE fixed template sentence, written in this file, no LLM text
`pair_uid` = sha1(f'{pair_id}|step268')[:12]: a random 48-bit token that exists only so the box
can GROUP a pair's rows for pooling.  It is not the pair_id and nothing about the pair_id can be
read out of it; the bijection and the absence of rank correlation are asserted below.

NEVER LEAVES: data/keys_universe.parquet (uid -> pair_id / hand_id / table_id / pair_uid).

Usage
  python build_encoder_universe.py [--procs 6] [--chunk 4000] [--limit N] [--no-write]
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
if CV + r'\src' not in sys.path:
    sys.path.insert(0, CV + r'\src')

import render_hand_v2 as V                                                     # noqa: E402
from build_dataset import SALT, uid_of                                         # noqa: E402

EXP = CV + r'\experiments\step268_qwen06b_encoder_heads_20260918'
DOUT = EXP + r'\data'
S264 = CV + r'\experiments\step264_llm_hand_judge_20260918'
OOF254 = CV + r'\experiments\step254_mechanical_tensor_20260917\hand_oof_step254.parquet'
BATCHES = ROOT + r'\outputs_v2\hands_development\batch_*.parquet'

K = ['pair_id', 'hand_id']
TOP_K = 40
N_PAIRS = 1860
N_HANDS = 223749
SEED_A1 = 268
SEED_SHUFFLE = 2680
PAIR_SALT = 'step268'
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')

# ONE fixed template sentence.  Written by hand in code, never by a model (registration268:
# "instruction = one fixed template sentence").
INSTRUCTION = ('Retrieve hands that an investigator would cite as evidence of coordination '
               'between player A and player B.')

# the STEP264 prompt-hygiene token pattern (src/llm_judge/checks264.py ID_RE), reused verbatim
ID_RE = re.compile(r'[PHUT][0-9A-F]{8,}')


def pair_uid_of(pair_id: str) -> str:
    return hashlib.sha1(f'{pair_id}|{PAIR_SALT}'.encode()).hexdigest()[:12]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- the key assembly
def universe_keys() -> pl.DataFrame:
    """Rules 1-3 above, as one frame sorted by (pair_id, hand_id)."""
    oof = pl.read_parquet(OOF254, columns=K + ['fold', 'listed', 'label', 'family',
                                               'mechplus_p_fmax'])
    assert oof.height == N_HANDS, oof.height
    assert oof['pair_id'].n_unique() == N_PAIRS, oof['pair_id'].n_unique()
    tab = pl.read_parquet(BATCHES, columns=K + ['table_id'])
    oof = oof.join(tab, on=K, how='left', validate='1:1')
    assert oof['table_id'].null_count() == 0, 'table_id join left nulls'
    assert oof.select(K).n_unique() == oof.height, 'duplicate (pair_id, hand_id)'

    # --- rule 1: top-40 by mechplus_p_fmax, ties by hand_id ascending
    ranked = (oof.sort(['pair_id', 'mechplus_p_fmax', 'hand_id'],
                       descending=[False, True, False])
              .with_columns(pl.int_range(pl.len()).over('pair_id').alias('rank_in_pair')))
    u = ranked.with_columns((pl.col('rank_in_pair') < TOP_K).alias('in_pool'))
    # --- rule 2: listed hands the top-40 missed
    u = u.with_columns(((pl.col('listed') == 1) & ~pl.col('in_pool')).alias('extra_listed'))

    keys = pl.concat([pl.read_parquet(S264 + r'\keys_train.parquet')
                      .with_columns(pl.lit('train').alias('keys_side')),
                      pl.read_parquet(S264 + r'\keys_heldout.parquet')
                      .with_columns(pl.lit('heldout').alias('keys_side'))])
    assert keys['uid'].n_unique() == keys.height, 'duplicate uid across the STEP264 key files'
    assert keys.select(K).n_unique() == keys.height, 'duplicate (pair_id, hand_id) in the keys'
    bad = [u_ for u_, p, h in zip(keys['uid'], keys['pair_id'], keys['hand_id'])
           if u_ != uid_of(p, h)]
    assert not bad, f'uid_of does not reproduce {len(bad)} STEP264 uids (salt {SALT})'

    u = u.join(keys.select(K + ['uid', 'source_class', 'a_is_player_1', 'keys_side']),
               on=K, how='left', validate='1:1')
    # --- rule 3: keep the pool, the extra listed rows and every key row
    u = u.filter(pl.col('in_pool') | pl.col('extra_listed') | pl.col('uid').is_not_null())

    # uid / a_is_player_1 for the rows STEP264 never saw
    u = u.sort(K)
    fresh = u['uid'].is_null().to_numpy()
    new_uid = [uid_of(p, h) for p, h in zip(u['pair_id'].to_list(), u['hand_id'].to_list())]
    a1_fresh = np.random.default_rng(SEED_A1).integers(0, 2, int(fresh.sum())).astype(bool)
    a1 = u['a_is_player_1'].to_numpy(allow_copy=True).astype(object).copy()
    a1[fresh] = a1_fresh
    src = np.array(u['source_class'].fill_null('').to_list(), dtype=object)
    src[fresh & u['in_pool'].to_numpy()] = 'POOL'
    src[fresh & u['extra_listed'].to_numpy()] = 'EXTRA_LISTED'
    u = u.with_columns(pl.Series('uid', new_uid),
                       pl.Series('a_is_player_1', a1.astype(bool)),
                       pl.Series('source_class', src.astype(str)),
                       pl.Series('pair_uid', [pair_uid_of(p) for p in u['pair_id'].to_list()]),
                       pl.col('keys_side').fill_null('none'),
                       pl.col('label').alias('label_pair'))
    assert u['uid'].n_unique() == u.height, 'uid collision in the universe'
    return u


def assert_design(u: pl.DataFrame) -> dict:
    """The claims the registration makes about this universe, checked mechanically."""
    per = (u.filter(pl.col('in_pool')).group_by('pair_id').len()
           .join(u.select('pair_id', 'label_pair').unique(), on='pair_id'))
    assert per.height == N_PAIRS, f'{per.height} pairs in the pool, expected {N_PAIRS}'
    cnt_pos = sorted(set(per.filter(pl.col('label_pair') == 1)['len'].to_list()))
    cnt_neg = sorted(set(per.filter(pl.col('label_pair') == 0)['len'].to_list()))
    assert cnt_pos == cnt_neg, ('the top-40 rule is NOT identical for positive and clean pairs: '
                                f'positive counts {cnt_pos}, clean counts {cnt_neg}')
    assert cnt_pos == [TOP_K], f'unexpected pool sizes {cnt_pos}'

    # pair_uid: a bijection, 12 hex characters, and not derivable from the pair_id
    m = u.select('pair_id', 'pair_uid').unique()
    assert m.height == N_PAIRS and m['pair_uid'].n_unique() == N_PAIRS, 'pair_uid is not 1:1'
    assert all(len(x) == 12 and all(c in '0123456789abcdef' for c in x)
               for x in m['pair_uid'].to_list()), 'pair_uid is not 12 hex characters'
    for p, q in m.iter_rows():
        assert q.lower() not in p.lower() and p.lower() not in q.lower(), \
            f'pair_uid shares a substring with the pair_id ({p} / {q})'
    rp = np.argsort(np.argsort(np.array(m['pair_id'].to_list())))
    rq = np.argsort(np.argsort(np.array(m['pair_uid'].to_list())))
    rho = float(np.corrcoef(rp, rq)[0, 1])
    assert abs(rho) < .05, f'pair_uid order tracks pair_id order (spearman {rho})'

    listed = u.filter(pl.col('listed') == 1)
    rec = {}
    for f in FAMS:
        g = listed.filter(pl.col('family') == f)
        rec[f] = {'n_listed': g.height,
                  'n_in_pool': int(g['in_pool'].sum()),
                  'recall_in_pool': round(float(g['in_pool'].mean()), 5)}
    rec['ALL'] = {'n_listed': listed.height, 'n_in_pool': int(listed['in_pool'].sum()),
                  'recall_in_pool': round(float(listed['in_pool'].mean()), 5)}
    return {'pool_per_pair': {'positive': cnt_pos, 'clean': cnt_neg},
            'pair_uid_rank_corr_vs_pair_id': round(rho, 5),
            'listed_recall_in_pool_by_family': rec}


# --------------------------------------------------------------------------- rendering
def _render_chunk(job):
    """One worker: its own HandContextV2 over its own hand ids, then body() per row."""
    idx, rows, out_path = job
    keys = pl.DataFrame({'pair_id': [r[1] for r in rows], 'hand_id': [r[2] for r in rows]})
    ctx = V.HandContextV2(keys)
    n_id, n_lit, chars = 0, 0, np.empty(len(rows), np.int64)
    t0 = time.time()
    with open(out_path, 'w', encoding='utf-8', newline='\n') as fh:
        for i, (uid, pid, hid, a1) in enumerate(rows):
            doc = V.body(ctx, pid, hid, bool(a1))
            chars[i] = len(doc)
            if ID_RE.search(doc):
                n_id += 1
            players = [s['player_id'] for s in ctx.s[hid]]
            if any(lit in doc for lit in (pid, hid, *players)):
                n_lit += 1
            fh.write(json.dumps({'uid': uid, 'doc': doc}) + '\n')
    return {'chunk': idx, 'n': len(rows), 'id_hits': n_id, 'literal_hits': n_lit,
            'chars': chars, 'seconds': round(time.time() - t0, 1)}


def render_all(u: pl.DataFrame, procs: int, chunk: int, tmpdir: str) -> tuple[list[str], dict]:
    rows = list(zip(u['uid'].to_list(), u['pair_id'].to_list(), u['hand_id'].to_list(),
                    u['a_is_player_1'].to_list()))
    jobs = []
    for i in range(0, len(rows), chunk):
        jobs.append((len(jobs), rows[i:i + chunk], os.path.join(tmpdir, f'part{len(jobs):04d}.jsonl')))
    print(f'rendering {len(rows)} documents in {len(jobs)} chunks on {procs} processes',
          flush=True)
    t0 = time.time()
    stats, done = [], 0
    with mp.get_context('spawn').Pool(processes=procs) as pool:
        for s in pool.imap_unordered(_render_chunk, jobs):
            stats.append(s)
            done += s['n']
            print(f'  chunk {s["chunk"]:>4}  {done:>6}/{len(rows)}  '
                  f'{s["seconds"]:>5.1f}s  elapsed {time.time() - t0:>6.1f}s', flush=True)
    id_hits = sum(s['id_hits'] for s in stats)
    lit_hits = sum(s['literal_hits'] for s in stats)
    assert id_hits == 0, f'{id_hits} documents contain an id-like token {ID_RE.pattern}'
    assert lit_hits == 0, f'{lit_hits} documents contain a literal pair/hand/player id'
    chars = np.concatenate([s['chars'] for s in sorted(stats, key=lambda x: x['chunk'])])
    return [j[2] for j in jobs], {'n_chunks': len(jobs), 'procs': procs,
                                  'render_seconds': round(time.time() - t0, 1),
                                  'id_like_token_hits': id_hits, 'literal_id_hits': lit_hits,
                                  'chars': chars}


def quantiles(a: np.ndarray) -> dict:
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {'n': int(len(a)), 'mean': round(float(a.mean()), 1),
            **{f'p{x}': round(float(np.percentile(a, x)), 1) for x in q}}


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--chunk', type=int, default=4000)
    ap.add_argument('--limit', type=int, default=0, help='render only the first N rows (probe)')
    ap.add_argument('--no-write', action='store_true', help='keys and checks only, no rendering')
    a = ap.parse_args(argv)
    assert 1 <= a.procs <= 6, 'at most 6 worker processes on this machine'
    assert os.path.exists(EXP + r'\registration268.json'), 'register first'
    os.makedirs(DOUT, exist_ok=True)
    t0 = time.time()

    u = universe_keys()
    design = assert_design(u)
    print('universe', u.height, 'rows,', u['pair_id'].n_unique(), 'pairs;',
          int(u['in_pool'].sum()), 'in pool,', int(u['extra_listed'].sum()), 'extra listed',
          flush=True)
    print('listed recall in the top-40 pool:',
          json.dumps(design['listed_recall_in_pool_by_family']), flush=True)

    # file order carries no id ordering
    rng = np.random.default_rng(SEED_SHUFFLE)
    u = u.with_columns(pl.Series('_o', rng.permutation(u.height))).sort('_o').drop('_o')
    if a.limit:
        u = u.head(a.limit)
        print(f'--limit {a.limit}: PROBE RUN, the outputs are partial', flush=True)

    checks = {'step': 'STEP268 A build_encoder_universe',
              'registration': 'experiments/step268_qwen06b_encoder_heads_20260918/'
                              'registration268.json',
              'probe_limit': a.limit or None,
              'universe': {
                  'n_rows': u.height, 'n_pairs': int(u['pair_id'].n_unique()),
                  'n_hands': int(u['hand_id'].n_unique()),
                  'n_in_pool': int(u['in_pool'].sum()),
                  'n_extra_listed': int(u['extra_listed'].sum()),
                  'n_listed': int((u['listed'] == 1).sum()),
                  'n_from_step264_keys': int((u['keys_side'] != 'none').sum()),
                  'n_step264_keys_added_outside_pool': int(
                      ((u['keys_side'] != 'none') & ~u['in_pool'] & ~u['extra_listed']).sum()),
                  'top_k_per_pair': TOP_K,
                  'uid_salt': SALT, 'pair_uid_salt': PAIR_SALT,
                  'a_is_player_1_seed_for_new_rows': SEED_A1,
                  'row_order_shuffle_seed': SEED_SHUFFLE},
              'design_asserts': design,
              'counts': {
                  'by_fold': {str(k): int(v) for k, v in
                              sorted(u.group_by('fold').len().iter_rows())},
                  'by_in_pool': {str(k): int(v) for k, v in
                                 sorted(u.group_by('in_pool').len().iter_rows())},
                  'by_listed': {str(k): int(v) for k, v in
                                sorted(u.group_by('listed').len().iter_rows())},
                  'by_source_class': {str(k): int(v) for k, v in
                                      sorted(u.group_by('source_class').len().iter_rows())},
                  'by_label_pair': {str(k): int(v) for k, v in
                                    sorted(u.group_by('label_pair').len().iter_rows())},
                  'by_family': {str(k): int(v) for k, v in
                                sorted(u.group_by('family').len().iter_rows())},
                  'by_keys_side': {str(k): int(v) for k, v in
                                   sorted(u.group_by('keys_side').len().iter_rows())}},
              'pairs_x_hands_in_pool': {}}
    hp = u.filter(pl.col('in_pool')).group_by('pair_uid').len()['len'].to_numpy()
    checks['pairs_x_hands_in_pool'] = {'n_pairs': int(len(hp)),
                                       'distinct_counts': {str(int(k)): int(v) for k, v in
                                                           zip(*np.unique(hp, return_counts=True))}}
    hu = u.group_by('pair_uid').len()['len'].to_numpy()
    checks['pairs_x_rows_total'] = {'min': int(hu.min()), 'p50': float(np.median(hu)),
                                    'max': int(hu.max())}

    if a.no_write:
        print(json.dumps(checks, indent=1))
        return

    tmpdir = tempfile.mkdtemp(prefix='step268_render_')
    try:
        parts, rstat = render_all(u, a.procs, a.chunk, tmpdir)
        docs_path = DOUT + r'\universe_docs.jsonl.gz'
        with gzip.open(docs_path, 'wt', encoding='utf-8', newline='\n', compresslevel=6) as out:
            for p in parts:
                with open(p, encoding='utf-8') as fh:
                    shutil.copyfileobj(fh, out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    chars = rstat.pop('chars')
    checks['render'] = rstat
    checks['doc_length'] = {'chars': quantiles(chars),
                            'est_tokens': quantiles(chars / 3.6),
                            'est_tokens_definition': 'render_hand_v2.est_tokens = chars / 3.6',
                            'share_over_2048_est_tokens': round(
                                float(((chars / 3.6) > 2048).mean()), 6)}

    meta_path = DOUT + r'\universe_meta.jsonl'
    with open(meta_path, 'w', encoding='utf-8', newline='\n') as fh:
        for r in u.select('uid', 'pair_uid', 'fold', 'label_pair', 'listed', 'family',
                          'in_pool', 'extra_listed', 'source_class').iter_rows(named=True):
            fh.write(json.dumps({'uid': r['uid'], 'pair_uid': r['pair_uid'],
                                 'fold': int(r['fold']), 'label_pair': int(r['label_pair']),
                                 'listed': int(r['listed']), 'family': r['family'],
                                 'in_pool': bool(r['in_pool']),
                                 'extra_listed': bool(r['extra_listed']),
                                 'source_class': r['source_class']}) + '\n')

    keys_path = DOUT + r'\keys_universe.parquet'
    u.select('uid', 'pair_uid', 'pair_id', 'hand_id', 'table_id', 'fold', 'label_pair', 'listed',
             'family', 'in_pool', 'extra_listed', 'source_class', 'a_is_player_1', 'keys_side',
             'rank_in_pair', 'mechplus_p_fmax').write_parquet(keys_path)

    # the two held-out uid lists, in the order of the STEP264 JSONL files
    have = set(u['uid'].to_list())
    lists = {}
    for name, src in (('heldout_scope57_uids.txt', S264 + r'\heldout_scope57.jsonl'),
                      ('heldout_subset_uids.txt', S264 + r'\heldout_subset.jsonl')):
        uids = []
        with open(src, encoding='utf-8') as fh:
            for ln in fh:
                if ln.strip():
                    uids.append(json.loads(ln)['uid'])
        assert len(set(uids)) == len(uids), 'duplicate uid in ' + src
        missing = [x for x in uids if x not in have]
        if not a.limit:
            assert not missing, f'{len(missing)} uids of {name} are not in the universe'
        with open(DOUT + '\\' + name, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(uids) + '\n')
        lists[name] = {'n': len(uids), 'n_missing_from_universe': len(missing)}
    checks['heldout_uid_lists'] = lists

    instr_path = DOUT + r'\instruction.txt'
    with open(instr_path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(INSTRUCTION + '\n')
    checks['instruction'] = INSTRUCTION

    checks['files'] = {}
    for p in (docs_path, meta_path, keys_path, instr_path,
              DOUT + r'\heldout_scope57_uids.txt', DOUT + r'\heldout_subset_uids.txt'):
        checks['files'][os.path.basename(p)] = {'bytes': os.path.getsize(p),
                                                'sha256': sha256_of(p)}
    checks['files'][os.path.basename(keys_path)]['NOTE'] = \
        'LOCAL ONLY - carries pair_id / hand_id / table_id, must never reach a GPU box'
    checks['wall_seconds'] = round(time.time() - t0, 1)
    with open(EXP + r'\checks268.json', 'w', encoding='utf-8') as fh:
        json.dump(checks, fh, indent=1)
    print(json.dumps({k: v for k, v in checks.items() if k != 'counts'}, indent=1))
    print('wrote', EXP + r'\checks268.json', 'in', checks['wall_seconds'], 's')


if __name__ == '__main__':
    main()
