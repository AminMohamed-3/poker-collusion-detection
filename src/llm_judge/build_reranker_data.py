"""STEP267: build the TEXT-RERANKER data package (documents + queries) from the v2 renderer.

A reranker is scored on (query, document) pairs, so this step needs exactly two artefacts that
STEP264 does not have:

  * DOCUMENTS - ``render_hand_v2.body(...)`` alone: the fully mechanical hand block (Table,
    Action, Result, Street summary, Counts, Derived facts).  No PREAMBLE, no worked examples, no
    Task block, no answer format: a reranker never generates, it only scores.
  * QUERIES   - four fixed template strings written in code (one family-agnostic, three per
    family) built MECHANICALLY from ``render_hand_v2.PREAMBLE``'s cue lists and
    ``render_hand.INSTRUCTION``'s family glosses.  Never per sample, never LLM-written.

TRAINING POOL (registration267.json).  Folds 0-2 of the step232 TABLE folds only:
  (i)  EVERY candidate hand of EVERY positive pair in folds 0-2 (hand_oof_step254, label == 1),
       source_class POS_LISTED when listed else POS_UNLISTED, row label = listed.  This is the
       pool the reranker actually has to separate in production: the stage-1 candidate list of a
       colluding pair.
  (ii) the v1 keys_train rows of classes NEG_HARD / NEG_RANDOM / NEG_WITHIN, with their existing
       uid and a_is_player_1.

  (i) and (ii) OVERLAP: every v1 NEG_WITHIN row is an unlisted hand of a positive pair in folds
  0-2, so it is already in (i).  (i) WINS the collision (it carries the true family and the
  listed-derived label); the number of absorbed rows is recorded in checks267.json.  No row is
  written twice and uid uniqueness is asserted.

HELD-OUT.  Exactly the uids of heldout_scope57.jsonl (1,146) and heldout_subset.jsonl (2,595), in
the same order, with the same label / family / source_class / fold copied verbatim from those
files and the document rendered with keys_heldout's a_is_player_1.  Nothing is re-derived, so the
rr files and the STEP264 files address the same rows.

Development phase only (asserted inside render_hand.HandContext).  keys_train_rr.parquet is LOCAL
ONLY - it maps uid back to (pair_id, hand_id) and must never be copied to a GPU box.

Usage:  python build_reranker_data.py                 full build
        python build_reranker_data.py --probe 300     render 300 train rows, write nothing
        python build_reranker_data.py --workers 6 --chunk 2000
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import hashlib
import json
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import render_hand as R                                                        # noqa: E402
import render_hand_v2 as V                                                     # noqa: E402

CV = R.CV
DATA = R.DATA
S264 = CV + r'\experiments\step264_llm_hand_judge_20260918'
OUT = CV + r'\experiments\step267_text_rerankers_20260918'
DOUT = OUT + r'\data'
OOF254 = CV + r'\experiments\step254_mechanical_tensor_20260917\hand_oof_step254.parquet'
K = ['pair_id', 'hand_id']
SALT = 'step264'                       # the STEP264 uid salt, reused so uids are comparable
SEED = 267
TRAIN_FOLDS, HELD_FOLDS = (0, 1, 2), (3, 4)
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
QKEYS = ('agnostic', 'directed_transfer', 'soft_play', 'coordinated_isolation')
# the six STEP264 PROMPT v2 worked-example uids (experiments/.../v2/checks_v2.json)
V2_EXAMPLE_UIDS = {'3c9af0c683f45832': 'directed_transfer', '3f9fa77961f3bf3a': 'soft_play',
                   '634046c6ddaa6925': 'coordinated_isolation',
                   'a0922e0be43ee1eb': 'NEG_RANDOM', '4ffb12123a65b9e5': 'NEG_HARD',
                   'b0db3e9840c3c262': 'NEG_WITHIN'}

INSTRUCTION_TXT = ('Judge whether the hand document is a cited piece of evidence of coordination '
                   'between players A and B matching the query.')


def uid_of(pair_id: str, hand_id: str) -> str:
    return hashlib.sha1(f'{pair_id}|{hand_id}|{SALT}'.encode()).hexdigest()[:16]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- the queries


def _cue_lists() -> dict[str, list[str]]:
    """The three MECHANICAL CUES bullet lists of render_hand_v2.PREAMBLE, parsed mechanically."""
    txt = V.PREAMBLE.split('MECHANICAL CUES', 1)[1]
    cues: dict[str, list[str]] = {}
    cur = None
    for raw in txt.splitlines():
        ln = raw.rstrip()
        if not ln.strip():
            continue
        m = re.fullmatch(r'([A-Z][A-Z_]+)', ln.strip())
        if m and not ln.startswith(' '):
            cur = m.group(1).lower()
            cues[cur] = []
            continue
        if cur is None:
            continue
        s = ln.strip()
        if s.startswith('- '):
            cues[cur].append(s[2:].strip())
        elif cues[cur] and not s.startswith('Cues are'):
            cues[cur][-1] = cues[cur][-1] + ' ' + s          # a wrapped continuation line
        else:
            break
    for f in FAMS:
        assert f in cues and len(cues[f]) == 3, (f, cues.get(f))
    return cues


def _family_glosses() -> dict[str, str]:
    """The one-line family definitions of render_hand.INSTRUCTION, parsed mechanically."""
    out = {}
    for ln in R.INSTRUCTION.splitlines():
        m = re.fullmatch(r'\s{2}([A-Z][A-Z_]+)\s{2,}(.+)', ln)
        if m and m.group(1) != 'NONE':
            out[m.group(1).lower()] = m.group(2).strip().rstrip('.')
    for f in FAMS:
        assert f in out, f
    return out


def build_queries() -> dict[str, str]:
    cues, gl = _cue_lists(), _family_glosses()
    q = {}
    for f in FAMS:
        body = '; '.join(c.rstrip('.;').strip() for c in cues[f])
        q[f] = f'Evidence of {f.upper()} between A and B: {body}.'
    q['agnostic'] = ('Cited evidence of coordination between A and B: '
                     + ', '.join(f'{f.upper()} ({gl[f]})' for f in FAMS[:-1])
                     + f', or {FAMS[-1].upper()} ({gl[FAMS[-1]]}).')
    return {k: q[k] for k in QKEYS}


# ---------------------------------------------------------------------------- key assembly


def dev_table_map(hand_ids: list[str]) -> dict[str, str]:
    ids = pl.DataFrame({'hand_id': sorted(set(hand_ids))})
    h = (pl.scan_parquet(DATA + r'\hands.parquet').join(ids.lazy(), on='hand_id')
         .select('hand_id', 'table_id', 'phase').collect())
    assert h.height == ids.height, 'hands missing'
    assert (h['phase'] == 'development').all(), 'a non-development hand reached the key builder'
    return dict(zip(h['hand_id'].to_list(), h['table_id'].to_list()))


def build_train_keys() -> tuple[pl.DataFrame, dict]:
    oof = pl.read_parquet(OOF254, columns=K + ['fold', 'listed', 'label', 'family'])
    assert oof.height == 223749, oof.height
    pos = (oof.filter((pl.col('label') == 1) & pl.col('fold').is_in(list(TRAIN_FOLDS)))
           .select(*K, 'fold',
                   pl.col('listed').cast(pl.Int32).alias('label'),
                   pl.col('family'),
                   pl.when(pl.col('listed') == 1).then(pl.lit('POS_LISTED'))
                   .otherwise(pl.lit('POS_UNLISTED')).alias('source_class'))
           .sort(K))
    assert set(pos['family'].to_list()) <= set(FAMS), 'unexpected family on a positive pair'

    kt = pl.read_parquet(S264 + r'\keys_train.parquet')
    assert set(kt['fold'].to_list()) <= set(TRAIN_FOLDS)
    negs = (kt.filter(pl.col('source_class').is_in(['NEG_HARD', 'NEG_RANDOM', 'NEG_WITHIN']))
            .select('uid', *K, 'fold', 'label', 'source_class', 'a_is_player_1')
            .with_columns(pl.lit('none').alias('family')))
    absorbed = negs.join(pos.select(K), on=K, how='semi')
    negs_keep = negs.join(pos.select(K), on=K, how='anti').sort(K)
    assert int(negs_keep['label'].sum()) == 0, 'a clean-pair negative carries label 1'

    # reuse the v1 uid / a_is_player_1 wherever the (pair, hand) already has one
    v1 = kt.select(*K, pl.col('uid').alias('uid_v1'),
                   pl.col('a_is_player_1').alias('a1_v1'))
    pos = pos.join(v1, on=K, how='left')
    rng = np.random.default_rng(SEED)
    a_new = rng.integers(0, 2, pos.height).astype(bool)
    uids = [u if u is not None else uid_of(p, h) for u, p, h in
            zip(pos['uid_v1'].to_list(), pos['pair_id'].to_list(), pos['hand_id'].to_list())]
    a1 = [bool(a) if a is not None else bool(b) for a, b in zip(pos['a1_v1'].to_list(), a_new)]
    reused = int(sum(u is not None for u in pos['uid_v1'].to_list()))
    pos = (pos.with_columns(pl.Series('uid', uids), pl.Series('a_is_player_1', a1))
           .drop('uid_v1', 'a1_v1'))
    # the shared salt makes uid_of reproduce the v1 uid exactly; assert rather than assume
    assert all(u == uid_of(p, h) for u, p, h in
               zip(pos['uid'].to_list(), pos['pair_id'].to_list(), pos['hand_id'].to_list()))

    cols = ['uid', 'pair_id', 'hand_id', 'fold', 'label', 'family', 'source_class',
            'a_is_player_1']
    tr = pl.concat([pos.select(cols), negs_keep.select(cols)])
    tmap = dev_table_map(tr['hand_id'].to_list())
    tr = tr.with_columns(pl.Series('table_id', [tmap[h] for h in tr['hand_id'].to_list()]))
    order = np.random.default_rng(SEED).permutation(tr.height)
    tr = tr[order]
    assert tr['uid'].n_unique() == tr.height, 'duplicate uid in train_rr'
    assert tr.select(K).n_unique() == tr.height, 'duplicate (pair, hand) in train_rr'
    meta = {'pos_rows': pos.height, 'neg_rows_kept': negs_keep.height,
            'neg_within_absorbed_into_POS_UNLISTED': absorbed.height,
            'v1_uids_reused_on_positive_pair_rows': reused,
            'positive_pairs': int(pos['pair_id'].n_unique())}
    return tr, meta


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def build_held_keys(name: str) -> tuple[pl.DataFrame, list[dict]]:
    rows = read_jsonl(f'{S264}\\{name}.jsonl')
    kh = pl.read_parquet(S264 + r'\keys_heldout.parquet')
    m = {r['uid']: r for r in kh.iter_rows(named=True)}
    miss = [r['uid'] for r in rows if r['uid'] not in m]
    assert not miss, f'{len(miss)} uids of {name}.jsonl are not held-out keys'
    for r in rows:
        k = m[r['uid']]
        assert bool(k['a_is_player_1']) == bool(r['a_is_player_1']), 'a_is_player_1 disagrees'
        assert int(k['fold']) == int(r['fold']) and int(k['label']) == int(r['label'])
        assert k['source_class'] == r['source_class'] and k['family'] == r['family']
    df = pl.DataFrame({'uid': [r['uid'] for r in rows],
                       'pair_id': [m[r['uid']]['pair_id'] for r in rows],
                       'hand_id': [m[r['uid']]['hand_id'] for r in rows],
                       'table_id': [m[r['uid']]['table_id'] for r in rows],
                       'fold': [int(r['fold']) for r in rows],
                       'label': [int(r['label']) for r in rows],
                       'family': [r['family'] for r in rows],
                       'source_class': [r['source_class'] for r in rows],
                       'a_is_player_1': [bool(r['a_is_player_1']) for r in rows]})
    assert set(df['fold'].to_list()) <= set(HELD_FOLDS)
    return df, rows


# ---------------------------------------------------------------------------- rendering


def _render_chunk(task):
    """One worker: build a HandContextV2 for this chunk's keys and render its documents."""
    idx, keys = task                            # keys: list of (uid, pair_id, hand_id, a1)
    os.environ['POLARS_MAX_THREADS'] = '1'
    kf = pl.DataFrame({'pair_id': [k[1] for k in keys], 'hand_id': [k[2] for k in keys]}).unique()
    ctx = V.HandContextV2(kf)
    out = []
    for uid, pid, hid, a1 in keys:
        out.append((uid, V.body(ctx, pid, hid, bool(a1))))
    return idx, out


def render_all(keys: pl.DataFrame, workers: int, chunk: int, tag: str,
               t0: float) -> dict[str, str]:
    """uid -> document, rendered in a pool.  keys must be unique on (pair_id, hand_id)."""
    rows = list(zip(keys['uid'].to_list(), keys['pair_id'].to_list(), keys['hand_id'].to_list(),
                    keys['a_is_player_1'].to_list()))
    tasks = [(i, rows[i:i + chunk]) for i in range(0, len(rows), chunk)]
    docs: dict[str, str] = {}
    done = 0
    with Pool(workers) as pool:
        for _idx, part in pool.imap_unordered(_render_chunk, tasks):
            for uid, txt in part:
                docs[uid] = txt
            done += len(part)
            print(f'  {tag}: {done}/{len(rows)} docs  {time.time() - t0:.0f}s', flush=True)
    assert len(docs) == len(rows), 'a document went missing'
    return docs


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=True) + '\n')


def quantiles(a: np.ndarray) -> dict:
    return {str(q): round(float(np.quantile(a, q / 100)), 1)
            for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def counts(df: pl.DataFrame) -> dict:
    return {
        'n': df.height,
        'by_source_class': {c: int(n) for c, n in
                            df.group_by('source_class').len().sort('source_class').iter_rows()},
        'by_fold': {str(c): int(n) for c, n in
                    df.group_by('fold').len().sort('fold').iter_rows()},
        'by_label': {str(c): int(n) for c, n in
                     df.group_by('label').len().sort('label').iter_rows()},
        'by_family': {c: int(n) for c, n in
                      df.group_by('family').len().sort('family').iter_rows()},
        'by_source_class_label': {f'{c}|{l}': int(n) for c, l, n in
                                  df.group_by('source_class', 'label').len()
                                  .sort('source_class', 'label').iter_rows()},
        'n_pairs': int(df['pair_id'].n_unique()),
        'n_tables': int(df['table_id'].n_unique()),
    }


# ---------------------------------------------------------------------------- main


def main() -> None:
    t0 = time.time()
    av = sys.argv
    probe = int(av[av.index('--probe') + 1]) if '--probe' in av else 0
    workers = int(av[av.index('--workers') + 1]) if '--workers' in av else 6
    chunk = int(av[av.index('--chunk') + 1]) if '--chunk' in av else 1500
    os.makedirs(DOUT, exist_ok=True)

    tr, meta = build_train_keys()
    scope, scope_rows = build_held_keys('heldout_scope57')
    subset, subset_rows = build_held_keys('heldout_subset')
    print(f'keys: train {tr.height} ({json.dumps(meta)}), scope57 {scope.height}, '
          f'subset {subset.height}  {time.time() - t0:.0f}s', flush=True)

    # ---- disjointness of the two sides
    for col in ('table_id', 'pair_id', 'hand_id'):
        a = set(tr[col].to_list())
        b = set(subset[col].to_list()) | set(scope[col].to_list())
        assert not (a & b), f'{col} appears on both sides: {sorted(a & b)[:3]}'
    assert not (set(tr['uid'].to_list()) & set(subset['uid'].to_list()))

    queries = build_queries()
    for k, v in queries.items():
        print(f'QUERY[{k}] ({len(v)} chars): {v}', flush=True)

    if probe:
        tr = tr.head(probe)
        scope = scope.head(min(probe, scope.height))
        subset = subset.head(min(probe, subset.height))

    # scope57's uids are a subset of subset's, so the held-out sides share one render pass
    assert set(scope['uid'].to_list()) <= set(subset['uid'].to_list()) or probe, \
        'heldout_scope57 is not contained in heldout_subset'
    held_keys = (pl.concat([subset, scope]).unique(subset=['uid'], keep='first')
                 .sort('uid'))

    docs_tr = render_all(tr, workers, chunk, 'train', t0)
    docs_ho = render_all(held_keys, workers, chunk, 'heldout', t0)
    print(f'rendered {len(docs_tr)} train + {len(docs_ho)} held-out documents '
          f'{time.time() - t0:.0f}s', flush=True)

    def rows_of(df: pl.DataFrame, docs: dict) -> list[dict]:
        return [{'uid': r['uid'], 'doc': docs[r['uid']], 'label': int(r['label']),
                 'family': r['family'], 'source_class': r['source_class'], 'fold': int(r['fold'])}
                for r in df.iter_rows(named=True)]

    out_rows = {'train_rr': rows_of(tr, docs_tr),
                'heldout_scope57_rr': rows_of(scope, docs_ho),
                'heldout_subset_rr': rows_of(subset, docs_ho)}
    if probe:
        for nm, rs in out_rows.items():
            ln = np.array([len(r['doc']) for r in rs], dtype=float)
            print(nm, len(rs), 'chars', quantiles(ln), 'est_tokens',
                  quantiles(ln / 3.6), flush=True)
        print('PROBE: nothing written.  total', round(time.time() - t0, 1), 's')
        return

    files = {}
    for nm, rs in out_rows.items():
        p = f'{DOUT}\\{nm}.jsonl'
        write_jsonl(p, rs)
        files[nm + '.jsonl'] = p
    tr.select('uid', 'pair_id', 'hand_id', 'table_id', 'fold', 'label', 'family', 'source_class',
              'a_is_player_1').write_parquet(f'{DOUT}\\keys_train_rr.parquet')
    files['keys_train_rr.parquet'] = f'{DOUT}\\keys_train_rr.parquet'

    # ---- OPAQUE group index for the group_ce training form (1 listed + n unlisted OF THE SAME
    # PAIR).  It carries NO pair_id / hand_id / table_id / player_id: the pairs are relabelled
    # 0..n-1 by a seed-267 permutation, so the file says only "these uids belong together".
    ppairs = sorted(set(tr.filter(pl.col('source_class').str.starts_with('POS'))
                        ['pair_id'].to_list()))
    perm = np.random.default_rng(SEED).permutation(len(ppairs))
    gid = {p: int(perm[i]) for i, p in enumerate(ppairs)}
    gmap = {r['uid']: gid[r['pair_id']] for r in tr.iter_rows(named=True)
            if r['pair_id'] in gid}
    json.dump(gmap, open(f'{DOUT}\\train_rr.groups.json', 'w', encoding='utf-8'))
    files['train_rr.groups.json'] = f'{DOUT}\\train_rr.groups.json'
    json.dump(queries, open(f'{OUT}\\queries.json', 'w', encoding='utf-8'), indent=1)
    files['queries.json'] = f'{OUT}\\queries.json'
    with open(f'{OUT}\\instruction.txt', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(INSTRUCTION_TXT + '\n')
    files['instruction.txt'] = f'{OUT}\\instruction.txt'

    # ---------------------------------------------------------------- checks267.json
    chk: dict = {'step': 'STEP267 reranker data build',
                 'builder': os.path.abspath(__file__),
                 'document': 'render_hand_v2.body(ctx, pair_id, hand_id, a_is_player_1) - the '
                             'mechanical hand block only; no preamble, no worked examples, no '
                             'Task block',
                 'uid_salt': SALT, 'seed': SEED, 'workers': workers, 'chunk': chunk,
                 'train_pool': meta,
                 'counts': {'train_rr': counts(tr), 'heldout_scope57_rr': counts(scope),
                            'heldout_subset_rr': counts(subset)}}
    chk['uid_order_equality_with_step264'] = {}
    for nm, src, rs in (('heldout_scope57_rr', scope_rows, out_rows['heldout_scope57_rr']),
                        ('heldout_subset_rr', subset_rows, out_rows['heldout_subset_rr'])):
        same_order = [r['uid'] for r in src] == [r['uid'] for r in rs]
        same_meta = all(int(a['label']) == int(b['label']) and a['family'] == b['family']
                        and a['source_class'] == b['source_class']
                        and int(a['fold']) == int(b['fold'])
                        for a, b in zip(src, rs))
        chk['uid_order_equality_with_step264'][nm] = {
            'step264_file': nm.replace('_rr', '') + '.jsonl',
            'n': len(rs), 'uid_order_identical': same_order,
            'label_family_class_fold_identical': same_meta}
        assert same_order and same_meta, nm

    a, b = set(tr['uid'].to_list()), set(subset['uid'].to_list())
    chk['disjointness'] = {
        'train_vs_heldout_shared_tables': len(set(tr['table_id'].to_list())
                                              & set(subset['table_id'].to_list())),
        'train_vs_heldout_shared_pairs': len(set(tr['pair_id'].to_list())
                                             & set(subset['pair_id'].to_list())),
        'train_vs_heldout_shared_hands': len(set(tr['hand_id'].to_list())
                                             & set(subset['hand_id'].to_list())),
        'train_vs_heldout_shared_uids': len(a & b),
        'train_folds': sorted({int(x) for x in tr['fold'].to_list()}),
        'heldout_folds': sorted({int(x) for x in subset['fold'].to_list()})}
    assert max(chk['disjointness'][k] for k in list(chk['disjointness'])[:4]) == 0

    chk['doc_length'] = {}
    for nm, rs in out_rows.items():
        ln = np.array([len(r['doc']) for r in rs], dtype=float)
        tk = np.array([V.est_tokens(r['doc']) for r in rs], dtype=float)
        chk['doc_length'][nm] = {'chars': quantiles(ln), 'chars_mean': round(float(ln.mean()), 1),
                                 'est_tokens': quantiles(tk),
                                 'est_tokens_mean': round(float(tk.mean()), 1),
                                 'share_est_tokens_over_2048': round(float((tk > 2048).mean()), 5),
                                 'share_est_tokens_over_1500': round(float((tk > 1500).mean()), 5)}

    trmap = set(tr['uid'].to_list())
    chk['step264_v2_worked_example_uids'] = [
        {'uid': u, 'class': c, 'in_train_rr': u in trmap,
         'note': 'these six hands are worked examples INSIDE the STEP264 v2 prompt; a reranker '
                 'never sees that prompt, so their presence in train_rr is not leakage'}
        for u, c in V2_EXAMPLE_UIDS.items()]
    chk['step264_v2_worked_examples_in_train'] = sum(
        1 for u in V2_EXAMPLE_UIDS if u in trmap)

    chk['group_index'] = {
        'file': 'data/train_rr.groups.json',
        'uids': len(gmap), 'groups': len(ppairs),
        'what': 'uid -> OPAQUE group index (a seed-267 permutation of the positive pairs); it '
                'carries no pair_id / hand_id / table_id / player_id and exists so that '
                'finetune_reranker.py --loss group_ce can draw 1 listed + n unlisted hands OF '
                'THE SAME PAIR.  Without it that arm groups by family instead.'}
    chk['queries'] = queries
    chk['instruction'] = INSTRUCTION_TXT
    chk['query_chars'] = {k: len(v) for k, v in queries.items()}
    chk['sha256'] = {os.path.basename(p): sha256(p) for p in files.values()}
    chk['bytes'] = {os.path.basename(p): os.path.getsize(p) for p in files.values()}
    chk['wall_seconds'] = round(time.time() - t0, 1)
    json.dump(chk, open(f'{OUT}\\checks267.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: chk[k] for k in ('counts', 'train_pool', 'doc_length',
                                          'step264_v2_worked_examples_in_train',
                                          'wall_seconds')}, indent=1), flush=True)
    print('wrote', OUT + r'\checks267.json')


if __name__ == '__main__':
    main()
