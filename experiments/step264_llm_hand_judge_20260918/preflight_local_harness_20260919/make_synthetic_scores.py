"""Preflight harness (2026-09-19), READ-ONLY except for this folder.

Builds a synthetic heldout_scores.jsonl that byte-for-byte matches the row schema written by
experiments/step264_llm_hand_judge_20260918/vm2/score_vllm_qwen35.py (lines ~206-212 of that
file), for all 1,146 prompts of v2/heldout_scope57_v2.jsonl, plus a partial variant covering only
the first 600 prompts in file order (what --limit 600 would have scored).

Deterministic: single numpy Generator seeded with 264, consumed in file order. A small fraction of
rows are marked malformed (k=0, confidences=[], verdict_counts={}, mean_conf=50.0, malformed=True)
to exercise that path, exactly as the real scorer would for a prompt whose k=1 sample failed to
parse (reward.parse() format_ok=False).

Does not touch any existing file. Writes only inside this new preflight_local_harness_20260919/
folder.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)  # experiments/step264_llm_hand_judge_20260918
SRC = os.path.join(EXP, 'v2', 'heldout_scope57_v2.jsonl')

VERDICTS = ('NONE', 'DIRECTED_TRANSFER', 'SOFT_PLAY', 'COORDINATED_ISOLATION')
MALFORMED_P = 0.004  # small fraction, matches "a few" out of 1,146


def load_uids(path):
    uids = []
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip():
                uids.append(json.loads(ln)['uid'])
    return uids


def make_row(rng, uid):
    """Exactly the score_vllm_qwen35.py row-construction logic, for k=1 sample."""
    u1 = rng.random()
    if u1 < MALFORMED_P:
        confs, verdicts = [], {}
    else:
        conf = int(rng.integers(0, 101))          # 0..100 inclusive, as reward.parse allows
        verdict = VERDICTS[int(rng.integers(0, len(VERDICTS)))]
        confs, verdicts = [conf], {verdict: 1}
    row = {'uid': uid, 'k': len(confs),
           'mean_conf': round(sum(confs) / len(confs), 3) if confs else 50.0,
           'confidences': confs, 'verdict_counts': verdicts}
    if not confs:
        row['malformed'] = True
    return row


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')


def main():
    uids = load_uids(SRC)
    assert len(uids) == 1146, f'expected 1146 prompts in {SRC}, got {len(uids)}'
    assert len(set(uids)) == 1146, 'duplicate uid in source file'

    rng = np.random.default_rng(264)
    rows_full = [make_row(rng, u) for u in uids]  # single RNG stream, file order, seed 264

    n_malformed = sum(1 for r in rows_full if r.get('malformed'))
    print(f'[make_synthetic_scores] {len(rows_full)} rows, {n_malformed} malformed '
          f'({n_malformed / len(rows_full):.4%})')

    full_path = os.path.join(HERE, 'synthetic_scores_full1146.jsonl')
    write_jsonl(full_path, rows_full)
    print('wrote', full_path)

    partial_path = os.path.join(HERE, 'synthetic_scores_first600.jsonl')
    write_jsonl(partial_path, rows_full[:600])
    print('wrote', partial_path)

    # sanity echo of the first malformed row and first well-formed row, for the record
    first_mal = next((r for r in rows_full if r.get('malformed')), None)
    first_ok = next(r for r in rows_full if not r.get('malformed'))
    print('example malformed row:', json.dumps(first_mal))
    print('example well-formed row:', json.dumps(first_ok))


if __name__ == '__main__':
    main()
