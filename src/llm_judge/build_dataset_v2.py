"""STEP264 PROMPT v2: re-render train / heldout / scope57 / subset with the v2 prompt.

Nothing about the DATASET changes: the uid, label, family, source_class, fold and a_is_player_1
of every row are copied from the v1 JSONL byte-for-byte and the row ORDER is preserved.  Only
``prompt`` is rewritten.  The only row-level difference is that the six hands used as worked
examples inside the prompt are DELETED from train_v2.jsonl, so that no training row is also a
solved example in its own prompt.

Everything is mechanical: the examples are picked by a deterministic rule (the median
example-rendering length inside the class, ties broken by uid), their derived-fact blocks are
trimmed by a fixed per-class priority list of fact keys, and their rationales are TEMPLATES -
an ordered list of gated sentences per class, of which the first three that fire are used, plus
one fixed closing sentence per class.  No sentence is written per hand and no LLM is called.

Usage:  python build_dataset_v2.py             full build
        python build_dataset_v2.py --probe 400 render 400 rows a side, write nothing
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import hashlib
import json
import re
import sys
import time

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import render_hand_v2 as V                                                     # noqa: E402
from render_hand_v2 import nf, pc                                              # noqa: E402

CV = V.CV
SRC = CV + r'\experiments\step264_llm_hand_judge_20260918'
OUT = SRC + r'\v2'
FILES = (('train', 'train.jsonl', 'train_v2.jsonl'),
         ('heldout', 'heldout.jsonl', 'heldout_v2.jsonl'),
         ('heldout_scope57', 'heldout_scope57.jsonl', 'heldout_scope57_v2.jsonl'),
         ('heldout_subset', 'heldout_subset.jsonl', 'heldout_subset_v2.jsonl'))

# ---------------------------------------------------------------------------- the constants

CONF = {'directed_transfer': 85, 'soft_play': 85, 'coordinated_isolation': 85,
        'NEG_RANDOM': 10, 'NEG_HARD': 25, 'NEG_WITHIN': 15}
VERDICT = {'directed_transfer': 'DIRECTED_TRANSFER', 'soft_play': 'SOFT_PLAY',
           'coordinated_isolation': 'COORDINATED_ISOLATION',
           'NEG_RANDOM': 'NONE', 'NEG_HARD': 'NONE', 'NEG_WITHIN': 'NONE'}
EX_CLASSES = ('directed_transfer', 'soft_play', 'coordinated_isolation',
              'NEG_RANDOM', 'NEG_HARD', 'NEG_WITHIN')
EX_WHY = {
    'directed_transfer': 'median-length training hand among the directed_transfer hands whose '
                         'family cue actually fires',
    'soft_play': 'median-length training hand among the soft_play hands whose family cue '
                 'actually fires',
    'coordinated_isolation': 'median-length training hand among the coordinated_isolation hands '
                             'whose family cue actually fires',
    'NEG_RANDOM': 'median-length training hand of NEG_RANDOM (plain NONE: a random hand of a '
                  'confirmed clean pair), no gate',
    'NEG_HARD': 'median-length training hand of NEG_HARD (hard NONE: a clean-pair hand in the '
                'top 2 % of the tabular event score), no gate',
    'NEG_WITHIN': 'median-length training hand of NEG_WITHIN (within-pair NONE: a certified '
                  'ordinary hand of a colluding pair), no gate',
}

# which six derived-fact lines survive in the worked examples, first six present, in order
FACT_PRIORITY = {
    'directed_transfer': ['flow', 'transfer', 'fold_to_partner', 'fp_price', 'fold_ahead',
                          'loser_share', 'eq_ab', 'pot_share', 'winner', 'ranks'],
    'soft_play': ['hu_streets', 'checkdown_best', 'passive_winner', 'eq_ab', 'showdown',
                  'transfer', 'pot_share', 'winner', 'ranks', 'fold_to_partner'],
    'coordinated_isolation': ['out_folds', 'out_price', 'iso_clean', 'weak_agg_behind',
                              'both_agg', 'pre_agg', 'pot_share', 'winner', 'eq_ab', 'ranks'],
    'NEG_RANDOM': ['fold_to_partner', 'out_folds', 'hu_streets', 'transfer', 'flow', 'pot_share',
                   'eq_ab', 'winner', 'ranks'],
    'NEG_HARD': ['transfer', 'flow', 'fold_to_partner', 'out_folds', 'hu_streets', 'pot_share',
                 'eq_ab', 'winner', 'ranks'],
    'NEG_WITHIN': ['hu_streets', 'fold_to_partner', 'transfer', 'out_folds', 'pot_share',
                   'eq_ab', 'flow', 'winner', 'ranks'],
}

CLOSER = {
    'directed_transfer': 'Value moving one way between the two members instead of being '
                         'contested is what DIRECTED_TRANSFER looks like.',
    'soft_play': 'Declining to bet or raise against each other in a pot the two of them control '
                 'is SOFT_PLAY.',
    'coordinated_isolation': 'Two members pushing the outsiders out of one pot together is '
                             'COORDINATED_ISOLATION.',
    'NEG_RANDOM': 'None of the three cues is present, so this is an ordinary hand.',
    'NEG_HARD': 'Some of these numbers are extreme, but no member acted for the other, so this '
                'is an ordinary hand.',
    'NEG_WITHIN': 'Nothing here is strong enough for an investigator to cite, so this is an '
                  'ordinary hand.',
}


# ---------------------------------------------------------------------------- the slot values


def slots(ctx: V.HandContextV2, pair_id: str, hand_id: str, a_is_player_1: bool) -> dict:
    """Every number a rationale template can use, computed from the hand record."""
    p1, p2 = ctx.members[pair_id]
    ma, mb = (p1, p2) if a_is_player_1 else (p2, p1)
    f = ctx.f237[(pair_id, hand_id)]
    g = ctx.f254[(pair_id, hand_id)]
    st = ctx.state(hand_id)
    _rows, order, alias, _pos = V.R._aliases(ctx, hand_id, ma, mb)
    flip = ((ctx.p1_is_A[(pair_id, hand_id)] == 1) != (ma == p1))
    eq = .5
    for col in ('t2_eq_turn_A', 't2_eq_flop_A', 't2_eq_pre_A'):
        if float(g[col]) >= 0:
            eq = float(g[col])
            break
    eq = (1.0 - eq) if flip else eq
    tot = st['tot_loss']
    fab = (st['loss'][ma] * st['gain'][mb] / tot) if tot > 0 else 0.0
    fba = (st['loss'][mb] * st['gain'][ma] / tot) if tot > 0 else 0.0
    n_agg = sum(st['counts'][m][0] + st['counts'][m][1] for m in (ma, mb))
    # the largest price any outsider faced from a member
    members = {ma, mb}
    best = (0.0, None, False)
    cur, street = None, None
    for a in st['acts']:
        if a['street'] != street:
            street, cur = a['street'], None
        if a['player_id'] not in members and cur in members and a['to_call'] > 0:
            price = float(a['to_call']) / st['bb']
            if price > best[0]:
                best = (price, alias[a['player_id']], a['action'] == 'fold')
        if V._is_agg(a):
            cur = a['player_id']
    fa_line = V._fold_ahead_line(st, ma, mb, alias, st['nb'])
    return {
        'n_fp': int(f['fp_count']),
        'price_min': float(f['fp_price_min']),
        'has_price': int(f['fp_count']) > 0 and float(f['fp_price_min']) < 1.0,
        'eq': pc(eq),
        'transfer': float(f['tr_internal_bb']) or float(g['t5_transfer_bb']),
        'pot_share': pc(float(f['pair_pot_share'])),
        'hu': int(f['hu_pair_streets']),
        'nout': int(round(float(f['iso_out_fold_share']) * 4)),
        'flow_bb': max(fab, fba),
        'flow_src': 'A' if fab > 0 else ('B' if fba > 0 else None),
        'flow_dst': 'B' if fab > 0 else ('A' if fba > 0 else None),
        'winner': ', '.join(alias[p] for p in st['winners']),
        'n_agg': n_agg,
        'both_agg': float(f['both_agg_live']) > 0,
        'iso_clean': float(f['iso_clean']) > 0,
        'checkdown_best': float(f['checkdown_best']) > 0,
        'passive_winner': float(f['passive_winner']) > 0,
        'weak_agg_behind': float(f['iso_weak_agg_behind']) > .6,
        'fold_ahead': 'would have beaten' in fa_line,
        'out_price': best[0],
        'out_price_who': best[1],
        'out_price_folded': best[2],
        'alias_order': [alias[p] for p in order],
    }


# ---------------------------------------------------------------------------- the templates
# Each entry is (gate, template).  The first three whose gate is true are emitted, then the
# fixed per-class closing sentence.  Nothing is written per hand.

def _sentences(cls: str, s: dict) -> list[str]:
    price = f', the cheapest at {pc(s["price_min"])} pot odds' if s['has_price'] else ''
    T: list[tuple[bool, str]] = []
    if cls == 'directed_transfer':
        T = [(s['flow_bb'] > 0,
              f'The chips end up on one side: {nf(s["flow_bb"])} bb of net flow from '
              f'{s["flow_src"]} to {s["flow_dst"]}, and the two of them supplied '
              f'{s["pot_share"]} of the pot.'),
             (s['n_fp'] > 0,
              f'A member folded to the other\'s bet or raise {s["n_fp"]}x{price}.'),
             (s['fold_ahead'],
              f'The folded hand would have beaten {s["winner"]}, who collected the pot.'),
             (s['transfer'] > 0,
              f'{nf(s["transfer"])} bb moved directly from one member to the other.'),
             (True, f'A\'s cards beat B\'s {s["eq"]} of the time.')]
    elif cls == 'soft_play':
        T = [(s['hu'] > 0,
              f'A and B were the only players left for {s["hu"]} postflop '
              f'street{"s" if s["hu"] > 1 else ""}.'),
             (s['checkdown_best'],
              'On the last street a member checked holding the best hand still in, with the '
              'other member still in the pot.'),
             (s['passive_winner'],
              'A member took the pot at showdown without betting or raising after the flop.'),
             (True, f'Between them the two members bet or raised {s["n_agg"]} '
                    f'time{"s" if s["n_agg"] != 1 else ""} in the whole hand.'),
             (True, f'A\'s cards beat B\'s {s["eq"]} of the time.')]
    elif cls == 'coordinated_isolation':
        nout_txt = {0: 'No non-member folded facing a bet or raise from A or B.',
                    4: 'All four non-members folded facing a bet or raise from A or B.'}.get(
                        s['nout'], f'{s["nout"]} of the four non-members folded facing a bet or '
                                   f'raise from A or B.')
        T = [(True, nout_txt),
             (s['iso_clean'],
              'A member made the last preflop raise, the other folded to a member\'s bet, and '
              'at most one non-member saw the flop.'),
             (s['out_price'] > 0,
              f'The largest price an outsider faced from a member was {nf(s["out_price"])} bb, '
              f'and {s["out_price_who"]} '
              f'{"folded to it" if s["out_price_folded"] else "paid it"}.'),
             (s['both_agg'], 'Both members bet or raised at some point.'),
             (True, f'Together they supplied {s["pot_share"]} of the final pot.')]
    else:
        marks = []
        if s['n_fp'] > 0:
            marks.append(f'a member folded to the other {s["n_fp"]}x')
        if s['transfer'] > 0:
            marks.append(f'{nf(s["transfer"])} bb moved between them')
        if s['nout'] >= 2:
            marks.append(f'{s["nout"]} non-members folded to a member\'s aggression')
        if s['hu'] > 0:
            marks.append(f'they were heads-up for {s["hu"]} postflop '
                         f'street{"s" if s["hu"] > 1 else ""}')
        T = [(cls == 'NEG_HARD' and bool(marks),
              'The hand does carry surface marks: ' + ', '.join(marks) + '.'),
             (s['n_fp'] == 0, 'Neither member folded facing a bet or raise from the other.'),
             (s['transfer'] == 0, 'No chips moved directly between A and B.'),
             (s['hu'] == 0, 'A and B were never the only two players left after the flop.'),
             (s['nout'] == 0, 'No non-member folded facing a bet or raise from A or B.'),
             (True, f'Together they supplied {s["pot_share"]} of the final pot and A\'s cards '
                    f'beat B\'s {s["eq"]} of the time.')]
    return [t for gate, t in T if gate][:3] + [CLOSER[cls]]


def rationale(cls: str, s: dict) -> str:
    return 'Why: ' + ' '.join(_sentences(cls, s))


# ---------------------------------------------------------------------------- example picking


def example_body(ctx: V.HandContextV2, cls: str, pair_id: str, hand_id: str,
                 a1: bool) -> str:
    return V.body(ctx, pair_id, hand_id, a1, fact_keys=FACT_PRIORITY[cls],
                  with_counts=False, compact=True)


def one_example(ctx: V.HandContextV2, i: int, cls: str, pair_id: str, hand_id: str,
                a1: bool) -> str:
    b = example_body(ctx, cls, pair_id, hand_id, a1)
    s = slots(ctx, pair_id, hand_id, a1)
    return (f'--- EXAMPLE {i}/6 ---\n{b}\n{rationale(cls, s)}\n'
            f'VERDICT: {VERDICT[cls]}\nCONFIDENCE: {CONF[cls]}')


def ex_class(row: dict) -> str:
    return row['family'] if row['source_class'] == 'POS' else row['source_class']


def cue_gate(cls: str, s: dict) -> bool:
    """Does this hand actually DISPLAY its family's cue?  A positive example that does not show
    the cue teaches nothing, so each POS family is first restricted to the hands where the cue
    fires and the median is taken inside that pool.  The three negative classes have no gate."""
    if cls == 'directed_transfer':
        return s['flow_bb'] > 0 and s['n_fp'] > 0
    if cls == 'soft_play':
        return s['hu'] > 0 and (s['checkdown_best'] or s['passive_winner'])
    if cls == 'coordinated_isolation':
        return s['both_agg'] and s['nout'] >= 3
    return True


GATE_DOC = {
    'directed_transfer': 'net chip flow between the members > 0 AND a member folded facing the '
                         'other at least once',
    'soft_play': 'at least one postflop street heads-up between the members AND '
                 '(checkdown_best OR passive_winner)',
    'coordinated_isolation': 'BOTH members bet or raised at some point AND at least 3 of the 4 '
                             'non-members folded facing a member\'s bet or raise',
    'NEG_RANDOM': 'no gate', 'NEG_HARD': 'no gate', 'NEG_WITHIN': 'no gate',
}


# ---------------------------------------------------------------------------- io helpers


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stats_of(lens: list[int]) -> dict:
    a = np.array(lens, dtype=float)
    t = a / 3.6
    return {'n': len(lens),
            'chars': {k: int(round(float(v))) for k, v in
                      (('min', a.min()), ('median', np.median(a)), ('p90', np.quantile(a, .9)),
                       ('p95', np.quantile(a, .95)), ('p99', np.quantile(a, .99)),
                       ('max', a.max()), ('mean', a.mean()))},
            'est_tokens': {k: int(round(float(v))) for k, v in
                           (('min', t.min()), ('median', np.median(t)),
                            ('p90', np.quantile(t, .9)), ('p95', np.quantile(t, .95)),
                            ('p99', np.quantile(t, .99)), ('max', t.max()), ('mean', t.mean()))}}


ID_RE = re.compile(r'[PHUT][0-9A-Fa-f]{8,}')


# ---------------------------------------------------------------------------- main


def main() -> None:
    t0 = time.time()
    probe = int(sys.argv[sys.argv.index('--probe') + 1]) if '--probe' in sys.argv else 0
    os.makedirs(OUT, exist_ok=True)

    v1 = {name: read_jsonl(f'{SRC}\\{src}') for name, src, _dst in FILES}
    keys = pl.concat([pl.read_parquet(f'{SRC}\\keys_train.parquet'),
                      pl.read_parquet(f'{SRC}\\keys_heldout.parquet')])
    kmap = {r['uid']: r for r in keys.iter_rows(named=True)}
    assert len(kmap) == keys.height, 'duplicate uid in the key parquets'

    # every v1 row must be described by the key parquets, with identical metadata
    for name in v1:
        for r in v1[name]:
            k = kmap[r['uid']]
            assert (int(k['label']), k['source_class'], int(k['fold']),
                    bool(k['a_is_player_1'])) == (r['label'], r['source_class'], r['fold'],
                                                  r['a_is_player_1']), f'metadata drift {name}'
    print(f'v1 loaded: ' + ', '.join(f'{n} {len(v1[n])}' for n in v1), flush=True)

    need = sorted({r['uid'] for rows in v1.values() for r in rows})
    kk = pl.DataFrame({'pair_id': [kmap[u]['pair_id'] for u in need],
                       'hand_id': [kmap[u]['hand_id'] for u in need]})
    ctx = V.HandContextV2(kk)
    print(f'context: {len(need)} rows in {time.time() - t0:.0f}s', flush=True)

    # ---- pick the six examples: median EXAMPLE-RENDERING length inside the class, tie -> uid
    by_cls: dict[str, list[tuple[int, str]]] = {c: [] for c in EX_CLASSES}
    gated: dict[str, list[tuple[int, str]]] = {c: [] for c in EX_CLASSES}
    for r in v1['train']:
        c = ex_class(r)
        k = kmap[r['uid']]
        n = len(example_body(ctx, c, k['pair_id'], k['hand_id'], bool(k['a_is_player_1'])))
        by_cls[c].append((n, r['uid']))
        if cue_gate(c, slots(ctx, k['pair_id'], k['hand_id'], bool(k['a_is_player_1']))):
            gated[c].append((n, r['uid']))
    chosen: dict[str, str] = {}
    pool_sizes: dict[str, dict] = {}
    for c in EX_CLASSES:
        lst = sorted(gated[c]) or sorted(by_cls[c])
        chosen[c] = lst[len(lst) // 2][1]
        pool_sizes[c] = {'in_class': len(by_cls[c]), 'passing_cue_gate': len(gated[c]),
                         'pool_used': len(lst), 'gate': GATE_DOC[c]}
    print('examples picked in', round(time.time() - t0), 's',
          {c: pool_sizes[c]['pool_used'] for c in EX_CLASSES}, flush=True)

    ex_uids = [chosen[c] for c in EX_CLASSES]
    assert len(set(ex_uids)) == 6, 'an example hand was picked twice'
    for name in ('heldout', 'heldout_scope57', 'heldout_subset'):
        clash = {r['uid'] for r in v1[name]} & set(ex_uids)
        assert not clash, f'example uid leaked into {name}: {clash}'

    blocks, ex_txt = [], []
    for i, c in enumerate(EX_CLASSES, 1):
        k = kmap[chosen[c]]
        blk = one_example(ctx, i, c, k['pair_id'], k['hand_id'], bool(k['a_is_player_1']))
        blocks.append(blk)
        ex_txt.append(f'### EXAMPLE {i}/6  class={c}  uid={chosen[c]}  '
                      f'verdict={VERDICT[c]}  confidence={CONF[c]}\n'
                      f'# chosen because: {EX_WHY[c]}\n{blk}')
    examples_block = '\n\n'.join([V.EX_HEAD] + blocks + [V.EX_FOOT])
    print(f'examples block: {len(examples_block)} chars '
          f'({V.est_tokens(examples_block):.0f} est tokens)', flush=True)

    # ---- render every file
    out_rows, out_lens, sample = {}, {}, None
    for name, _src, dst in FILES:
        rows, lens = [], []
        drop = set(ex_uids) if name == 'train' else set()
        for j, r in enumerate(v1[name]):
            if r['uid'] in drop:
                continue
            if probe and len(rows) >= probe:
                break
            k = kmap[r['uid']]
            txt = V.assemble(examples_block,
                             V.body(ctx, k['pair_id'], k['hand_id'], bool(k['a_is_player_1'])))
            assert not ID_RE.search(txt), f'id-like token in the prompt of {r["uid"]}'
            for bad in (k['pair_id'], k['hand_id'], k['table_id']):
                assert bad not in txt, f'raw id in the prompt of {r["uid"]}'
            nr = dict(r)
            nr['prompt'] = txt
            rows.append(nr)
            lens.append(len(txt))
            if name == 'heldout' and sample is None and r['source_class'] == 'POS':
                sample = txt
            if (j + 1) % 2000 == 0:
                print(f'  {name}: {j + 1}/{len(v1[name])}  {time.time() - t0:.0f}s', flush=True)
        out_rows[name], out_lens[name] = rows, lens
        print(f'{name}: {len(rows)} rows, '
              f'median {np.median(lens):.0f} chars / {np.median(lens) / 3.6:.0f} tokens',
              flush=True)
        if not probe:
            with open(f'{OUT}\\{dst}', 'w', encoding='utf-8', newline='\n') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=True) + '\n')
    if probe:
        print('probe only, nothing written; total', round(time.time() - t0, 1), 's')
        return

    # ---- the side files
    with open(f'{OUT}\\examples_v2.txt', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('STEP264 PROMPT v2 - the six worked examples exactly as they appear inside '
                 'every prompt.\n'
                 'Each hand is a TRAINING hand (folds 0-2) and has been removed from '
                 'train_v2.jsonl.\n'
                 'The "Why:" line is a template filled from the derived facts; no prose was '
                 'written per hand.\n\n' + '\n\n'.join(ex_txt) + '\n')
    with open(f'{OUT}\\sample_prompt_v2.txt', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(sample if sample else out_rows['heldout'][0]['prompt'])
    write_prompt_md(chosen, out_lens, pool_sizes)

    # ---- checks
    checks = {
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'examples': [{'rank': i + 1, 'class': c, 'uid': chosen[c], 'verdict': VERDICT[c],
                      'confidence': CONF[c], 'chosen_because': EX_WHY[c],
                      'fact_priority': FACT_PRIORITY[c], 'pool': pool_sizes[c]}
                     for i, c in enumerate(EX_CLASSES)],
        'confidence_constants': CONF,
        'examples_block_chars': len(examples_block),
        'examples_block_est_tokens': round(V.est_tokens(examples_block), 1),
        'preamble_chars': len(V.PREAMBLE),
        'instruction_chars': len(V.INSTRUCTION),
        'files': {}, 'assertions': {},
    }
    ok = checks['assertions']
    ok['example_uids_are_train_fold_0_2'] = all(kmap[u]['fold'] in (0, 1, 2) for u in ex_uids)
    ok['example_uids_not_in_any_heldout_file'] = all(
        not (set(ex_uids) & {r['uid'] for r in out_rows[n]})
        for n in ('heldout', 'heldout_scope57', 'heldout_subset'))
    ok['example_uids_removed_from_train'] = not (set(ex_uids)
                                                 & {r['uid'] for r in out_rows['train']})
    for name, src, dst in FILES:
        path = f'{OUT}\\{dst}'
        rows = out_rows[name]
        u1 = [r['uid'] for r in v1[name]]
        u2 = [r['uid'] for r in rows]
        expect = [u for u in u1 if u not in set(ex_uids)] if name == 'train' else u1
        same_fields = all(
            {k: v for k, v in a.items() if k != 'prompt'}
            == {k: v for k, v in b.items() if k != 'prompt'}
            for a, b in zip([r for r in v1[name] if r['uid'] not in set(ex_uids)], rows))
        cls = {}
        fld = {}
        for r in rows:
            cls[r['source_class']] = cls.get(r['source_class'], 0) + 1
            fld[str(r['fold'])] = fld.get(str(r['fold']), 0) + 1
        checks['files'][dst] = {
            'source': src, 'n_v1': len(v1[name]), 'n_v2': len(rows),
            'removed_example_rows': len(v1[name]) - len(rows),
            'uid_order_identical_to_v1': u2 == expect,
            'uid_set_identical_to_v1_minus_examples': set(u2) == set(expect),
            'non_prompt_fields_identical': same_fields,
            'by_source_class': dict(sorted(cls.items())),
            'by_fold': dict(sorted(fld.items())),
            'by_label': {'0': sum(1 for r in rows if r['label'] == 0),
                         '1': sum(1 for r in rows if r['label'] == 1)},
            'prompt_length': stats_of(out_lens[name]),
            'sha256': sha256(path),
        }
    for extra in ('PROMPT_V2.md', 'examples_v2.txt', 'sample_prompt_v2.txt'):
        checks['files'][extra] = {'sha256': sha256(f'{OUT}\\{extra}')}
    json.dump(checks, open(f'{OUT}\\checks_v2.json', 'w', encoding='utf-8'), indent=1)
    bad = {k: v for k, v in ok.items() if not v}
    for f, d in checks['files'].items():
        for key in ('uid_order_identical_to_v1', 'uid_set_identical_to_v1_minus_examples',
                    'non_prompt_fields_identical'):
            if key in d and not d[key]:
                bad[f + ':' + key] = False
    assert not bad, f'FAILED CHECKS: {bad}'
    print('all checks pass; done in', round(time.time() - t0, 1), 's')


# ---------------------------------------------------------------------------- PROMPT_V2.md


def write_prompt_md(chosen: dict, out_lens: dict, pool_sizes: dict) -> None:
    ex_rows = '\n'.join(
        f'| {i} | {c} | `{chosen[c]}` | {VERDICT[c]} | {CONF[c]} | {EX_WHY[c]} |'
        for i, c in enumerate(EX_CLASSES, 1))
    fact_rows = '\n'.join(f'| {c} | {", ".join(FACT_PRIORITY[c])} |' for c in EX_CLASSES)
    gate_rows = '\n'.join(
        f'| {c} | {pool_sizes[c]["gate"]} | {pool_sizes[c]["in_class"]} | '
        f'{pool_sizes[c]["pool_used"]} |' for c in EX_CLASSES)
    tmpl = '\n\n'.join(
        f'**{c}** (closing sentence is fixed)\n\n```\n'
        + '\n'.join(f'  {i + 1}. {t}' for i, t in enumerate(_TEMPLATE_DOC[c]))
        + f'\n  closer. {CLOSER[c]}\n```' for c in EX_CLASSES)
    lens = '\n'.join(
        f'| {dst} | {len(out_lens[n])} | {int(np.median(out_lens[n]))} | '
        f'{int(np.quantile(out_lens[n], .9))} | {int(max(out_lens[n]))} | '
        f'{int(np.median(out_lens[n]) / 3.6)} | {int(np.quantile(out_lens[n], .9) / 3.6)} | '
        f'{int(max(out_lens[n]) / 3.6)} |'
        for n, _s, dst in FILES)
    md = f"""# STEP264 PROMPT v2

Built by `src/llm_judge/build_dataset_v2.py` on top of `src/llm_judge/render_hand_v2.py`.
Every number in every prompt is computed from the hand record (hands / seats / actions parquet
plus the STEP237 and STEP254 feature tables).  No LLM was called and no rationale was written by
hand: the six worked-example rationales are templates, filled mechanically.

## 1. Skeleton

```
PREAMBLE                 fixed, {len(V.PREAMBLE)} chars
  - table conventions (A, B, O1-O4, bb, odds, SPR)
  - WHAT COUNTS AS A POSITIVE
  - CONFIDENCE (definition + the NONE/<50 rule)
  - MECHANICAL CUES (2-3 per family)

=== SIX WORKED EXAMPLES ... ===          fixed, identical in every prompt
  --- EXAMPLE i/6 ---
  Table / Action / Result / Street summary      (v2 rendering in COMPACT form: the Counts
                                                 block, the "Collected:" line and the SPR
                                                 segment are dropped to save tokens)
  Derived facts (the six most relevant lines)   (per-class priority list, see section 4)
  Why: <templated rationale, 2-4 sentences>
  VERDICT: <fixed per class>
  CONFIDENCE: <fixed per class>
=== END OF WORKED EXAMPLES ===

=== THE HAND TO JUDGE ===
  Table / Action / Result                       (identical to v1)
  Street summary                                (NEW)
  Counts per player                             (NEW)
  Derived facts (computed from the hand record) (v1's ten lines + six NEW lines)

Task ...                                        v1's INSTRUCTION verbatim, {len(V.INSTRUCTION)} chars
VERDICT: <NONE|DIRECTED_TRANSFER|SOFT_PLAY|COORDINATED_ISOLATION>
CONFIDENCE: <integer 0-100 ...>
```

The answer contract is byte-identical to v1, so `reward.py` keeps parsing completions unchanged
(it takes the LAST well-formed VERDICT / CONFIDENCE pair outside `<think>`; the VERDICT lines
inside the worked examples live in the PROMPT, which the reward function never parses).

## 2. What v2 adds to the rendered hand

`Street summary`, one line per street that was actually played (a street with at least one action
row; an all-in run-out does not create a street):

  * pot at the START of the street, in bb;
  * `A+B share`: the fraction of that pot the two members had put in, blinds included;
  * `outsiders in`: how many of O1-O4 had not folded yet;
  * `SPR A / B`: each member's remaining stack divided by that pot (`out` if already folded);
  * `A beats B`: A's equity against B's real two cards on that street (STEP254 `t2_eq_*_A`,
    flipped when the prompt's A is STEP254's B), omitted when the feature is not applicable;
  * `vs field`: A's and B's EXACT equity against the players still live at the start of that
    street, enumerating every remaining board card with all twelve hole cards removed from the
    deck, ties split.  Flop, turn and river only;
  * preflop only: `start-hand pct`, the project's starting-hand strength percentile for A and B.
    Preflop equity against the live field is NOT computed - it is C(40,5) = 658,008 board
    completions per hand, which is not cheap, and saying so is better than faking it.

`Counts per player, bets/raises/calls/checks/folds`: one compact cell per seat, in preflop
action order, for A, B and every outsider.  An `all_in` counts as a raise when
`amount > to_call` and as a call otherwise (the STEP264 raw-data convention).

Six new derived-fact lines, after v1's ten:

| key | line |
|---|---|
| `pre_agg` | the last preflop aggressor, or "Nobody raised before the flop" |
| `winner` | which seat collected the most chips, and how much |
| `showdown` | who showed and the CATEGORY of each hand shown, ordered best first |
| `fold_ahead` | whether a member folded a hand that would have beaten the seat that collected |
| `out_price` | the largest price each outsider faced from A or B, and whether they folded to it |
| `flow` | net chip flow A->B and B->A separately |

Definitions used, so that they can be checked:

  * **aggressor** - the last player on that street whose action was `bet`, `raise`, or an
    `all_in` with `amount > to_call`.  Posting a blind is NOT aggression, so before the first
    voluntary raise preflop there is no aggressor and no outsider is recorded as "facing" A or B.
  * **pot winner** - the seat with the largest `collected = net_chips + total_contribution`.
    That includes an uncalled bet handed back, which is why the line says "most chips collected"
    rather than "won the pot"; ties are listed.
  * **net chip flow** - proportional attribution.  With `loss_i = max(0, -net_i)`,
    `gain_j = max(0, net_j)` and `T` the sum of the losses, `flow(i->j) = loss_i * gain_j / T`.
    At most one of `flow(A->B)`, `flow(B->A)` can be non-zero.  This is a pot-level attribution,
    not a claim that those exact chips changed hands.
  * **hand categories** - phevaluator rank classes (1-10 straight flush, 11-166 quads, 167-322
    full house, 323-1599 flush, 1600-1609 straight, 1610-2467 trips, 2468-3325 two pair,
    3326-6185 one pair, 6186-7462 high card), verified against nine constructed hands.

## 3. The definition block (verbatim)

```
{V.PREAMBLE}
```

## 4. The six worked examples

| # | class | uid | verdict | confidence | chosen because |
|---|---|---|---|---|---|
{ex_rows}

Selection rule, deterministic and seedless: inside each of the six classes, every TRAINING row
(folds 0-2 only) is rendered in its example form; for the three POS families the pool is first
restricted to the hands whose FAMILY CUE actually fires (a positive example that does not show
the cue teaches nothing); the surviving rows are sorted by `(length_in_chars, uid)` and the row
at index `n // 2` is taken.  If a gate emptied a pool the whole class would be used instead;
none of them did.

| class | cue gate | rows in class | rows in the pool used |
|---|---|---|---|
{gate_rows}

All six hands are DELETED from `train_v2.jsonl`, and `checks_v2.json` asserts that none of the
six uids appears in `heldout_v2.jsonl`, `heldout_scope57_v2.jsonl` or `heldout_subset_v2.jsonl`
(they cannot: the fold split is by whole tables).

Derived-fact lines kept per example (first six of the priority list that the hand actually has):

| class | fact keys |
|---|---|
{fact_rows}

### Confidence constants

```
{json.dumps(CONF, indent=2)}
```

Positives 85, plain NONE (NEG_RANDOM) 10, hard NONE (NEG_HARD) 25, within-pair NONE (NEG_WITHIN)
15.  They are constants per class, never per hand, and they obey the stated rule (NONE < 50,
anything else >= 50).

## 5. The rationale templates

An ordered list of gated sentences per class; the first THREE whose gate is true are emitted,
then one fixed closing sentence.  So a rationale is 2-4 sentences and every slot is a number
that is also printed in that example's own fact block.

{tmpl}

## 6. Prompt length

| file | rows | median chars | p90 chars | max chars | median tok | p90 tok | max tok |
|---|---|---|---|---|---|---|---|
{lens}

Estimated tokens are chars / 3.6, the project-wide estimate.  Budget for a rollout:
prompt + 768 thinking + ~20 answer.

## 7. What is NOT in v2, and why

  * **Per-player preflop equity against the whole field.**  Not cheap: 658,008 board completions
    per hand.  The preflop line prints the starting-hand strength percentile of A and B instead
    and says that is what it is.  Postflop equity against the live field IS exact and is printed.
  * **Which member is A.**  Still randomised per row from the v1 `a_is_player_1`, unchanged, so
    v1 and v2 scores are comparable row by row.
  * **Anything chronological or any identifier.**  Every prompt is re-scanned for
    `[PHUT][0-9A-Fa-f]{{8,}}` and for the literal pair / hand / table id of its own row.
"""
    with open(f'{OUT}\\PROMPT_V2.md', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(md)


_TEMPLATE_DOC = {
    'directed_transfer': [
        'if net flow > 0: "The chips end up on one side: {flow} bb of net flow from {src} to '
        '{dst}, and the two of them supplied {pot_share} of the pot."',
        'if a member folded to the other: "A member folded to the other\'s bet or raise {n}x'
        '[, the cheapest at {price} pot odds]."',
        'if the folded hand beat the collector: "The folded hand would have beaten {winner}, '
        'who collected the pot."',
        'if direct transfer > 0: "{transfer} bb moved directly from one member to the other."',
        'always: "A\'s cards beat B\'s {eq} of the time."'],
    'soft_play': [
        'if heads-up streets > 0: "A and B were the only players left for {hu} postflop '
        'street(s)."',
        'if checkdown_best: "On the last street a member checked holding the best hand still '
        'in, with the other member still in the pot."',
        'if passive_winner: "A member took the pot at showdown without betting or raising '
        'after the flop."',
        'always: "Between them the two members bet or raised {n_agg} time(s) in the whole '
        'hand."',
        'always: "A\'s cards beat B\'s {eq} of the time."'],
    'coordinated_isolation': [
        'always: "{nout} of the four non-members folded facing a bet or raise from A or B." '
        '(0 -> "No non-member folded...", 4 -> "All four non-members folded...")',
        'if iso_clean: "A member made the last preflop raise, the other folded to a member\'s '
        'bet, and at most one non-member saw the flop."',
        'if an outsider faced a member: "The largest price an outsider faced from a member was '
        '{price} bb, and {who} folded to it / paid it."',
        'if both members were aggressive: "Both members bet or raised at some point."',
        'always: "Together they supplied {pot_share} of the final pot."'],
    'NEG_RANDOM': [
        '(NEG_HARD only) if any surface mark: "The hand does carry surface marks: {marks}."',
        'if no member folded to the other: "Neither member folded facing a bet or raise from '
        'the other."',
        'if no direct transfer: "No chips moved directly between A and B."',
        'if never heads-up: "A and B were never the only two players left after the flop."',
        'if no outsider folded to them: "No non-member folded facing a bet or raise from A '
        'or B."',
        'always: "Together they supplied {pot_share} of the final pot and A\'s cards beat B\'s '
        '{eq} of the time."'],
}
_TEMPLATE_DOC['NEG_HARD'] = _TEMPLATE_DOC['NEG_RANDOM']
_TEMPLATE_DOC['NEG_WITHIN'] = _TEMPLATE_DOC['NEG_RANDOM']


if __name__ == '__main__':
    main()
