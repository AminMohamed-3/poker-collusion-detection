"""STEP264: write examples.txt - 24 rendered prompts for human review.

Two per (source_class x family) cell, six cells from train.jsonl and six from heldout.jsonl.
Held-out NEG_RANDOM is not sampled separately because it is drawn by exactly the same rule as the
train NEG_RANDOM shown above it; held-out AUDIT has no train counterpart and is shown instead.

The label of each prompt is printed AFTER a separator line so the prompt can be read blind first.
Usage:  python make_examples.py
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import build_dataset as B                                                    # noqa: E402

OUT = B.OUT
CELLS = [
    ('train', 'POS', 'directed_transfer'), ('train', 'POS', 'soft_play'),
    ('train', 'POS', 'coordinated_isolation'), ('train', 'NEG_RANDOM', None),
    ('train', 'NEG_HARD', None), ('train', 'NEG_WITHIN', None),
    ('heldout', 'POS', 'directed_transfer'), ('heldout', 'POS', 'soft_play'),
    ('heldout', 'POS', 'coordinated_isolation'), ('heldout', 'AUDIT', None),
    ('heldout', 'NEG_HARD', None), ('heldout', 'NEG_WITHIN', None),
]
PER_CELL = 2


def load(name):
    with open(f'{OUT}\\{name}.jsonl', encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main() -> None:
    rows = {'train': load('train'), 'heldout': load('heldout')}
    rng = np.random.default_rng(264)
    picked = []
    for split, cls, fam in CELLS:
        pool = [r for r in rows[split]
                if r['source_class'] == cls and (fam is None or r['family'] == fam)]
        pool.sort(key=lambda r: r['uid'])
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(PER_CELL, len(pool)), replace=False)
        for j in sorted(idx):
            picked.append((split, pool[j]))
    with open(OUT + r'\examples.txt', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('STEP264 - 24 example prompts for human review.\n'
                 'Each prompt is exactly what the model sees. The label follows the ANSWER KEY '
                 'separator, so read the prompt first.\n'
                 'Two per (source_class x family) cell: six cells from train.jsonl and six from '
                 'heldout.jsonl. Held-out NEG_RANDOM is omitted because it is drawn by the same '
                 'rule as the train NEG_RANDOM; held-out AUDIT is shown instead.\n'
                 'No pair_id / hand_id / player_id / table_id, no timestamp and no hand-order '
                 'index appears in any prompt (checks264.json, prompt_hygiene).\n')
        for i, (split, r) in enumerate(picked, 1):
            fh.write('\n' + '=' * 100 + '\n')
            fh.write(f'EXAMPLE {i:02d} of {len(picked)}\n')
            fh.write('-' * 100 + '\n')
            fh.write(r['prompt'] + '\n')
            fh.write('-' * 38 + ' ANSWER KEY ' + '-' * 38 + '\n')
            fh.write(f'  split={split}  source_class={r["source_class"]}  label={r["label"]}  '
                     f'family={r["family"]}  fold={r["fold"]}\n')
            fh.write(f'  a_is_player_1={r["a_is_player_1"]}  uid={r["uid"]}  '
                     f'chars={len(r["prompt"])}  est_tokens={len(r["prompt"]) / 3.6:.0f}\n')
        fh.write('\n' + '=' * 100 + '\n')
    print('wrote', OUT + r'\examples.txt', 'with', len(picked), 'examples')


if __name__ == '__main__':
    main()
