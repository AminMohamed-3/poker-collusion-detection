#!/usr/bin/env python
"""STEP264 - rejection sampling on the TRAIN prompts with the final adapter.

*** UNTESTED LOCALLY.  Written on a machine with no GPU, no torch and no network. ***
*** Run `--smoke` first (8 prompts, a few minutes). ***

Draws k = 8 samples at temperature 0.8 for every TRAIN prompt and logs every rollout through
trace_logger.py with checkpoint tag 'final_rs'.  That trace file is the input of
select_best_traces.py, which distils the good ones into sft_traces.jsonl for a small student.

TRAIN prompts only - never heldout.jsonl.  The script refuses a data file whose rows carry a
fold in {3, 4}, and the selector refuses a held-out uid a second time.

  python rejection_sample.py --model Qwen/Qwen3-8B --adapter runs/r1/adapter_final --smoke
  python rejection_sample.py --model Qwen/Qwen3-8B --adapter runs/r1/adapter_final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import reward as RW                                                          # noqa: E402
from score_heldout import generate, load_jsonl, load_policy, render_chat     # noqa: E402
from trace_logger import TraceLogger, sha1                                   # noqa: E402

BANNER = '*** STEP264 vm/rejection_sample.py - UNTESTED ON A GPU. Run --smoke first. ***'
TRAIN_FOLDS = {0, 1, 2}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='STEP264 rejection sampling (UNTESTED ON GPU)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--data', default='train.jsonl')
    ap.add_argument('--traces', default='traces/final_rs.jsonl.gz')
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--checkpoint-tag', default='final_rs')
    ap.add_argument('-k', '--samples', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=.8)
    ap.add_argument('--max-new', type=int, default=900)
    ap.add_argument('--batch', type=int, default=2, help='PROMPTS per generate call (x k seqs)')
    ap.add_argument('--no-4bit', action='store_true')
    ap.add_argument('--no-chat-template', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    return ap


def main(argv=None) -> None:
    print(BANNER)
    a = build_argparser().parse_args(argv)
    if a.smoke:
        a.limit, a.samples, a.max_new, a.batch = 8, 2, 256, 2
        a.traces = a.traces.replace('.jsonl.gz', '_smoke.jsonl.gz')
        print('[smoke] 8 prompts, k=2, max_new 256')
    rows = load_jsonl(a.data)
    bad = sorted({int(r['fold']) for r in rows} - TRAIN_FOLDS)
    assert not bad, (f'REFUSING: {a.data} contains folds {bad}; rejection sampling runs on TRAIN '
                     f'prompts only (folds 0,1,2). Held-out rollouts must never reach the '
                     f'student SFT set.')
    if a.limit:
        rows = rows[:a.limit]
    print(f'{len(rows)} TRAIN prompts, k={a.samples} at T={a.temperature} '
          f'-> {len(rows) * a.samples} rollouts')

    model, tok = load_policy(a.model, a.adapter, not a.no_4bit)
    run_id = a.run_id or f'step264_rs_{int(time.time())}'
    t0 = time.time()
    n_done = n_ok = 0
    with TraceLogger(a.traces, run_id=run_id) as logger:
        for b in range(0, len(rows), a.batch):
            chunk = rows[b:b + a.batch]
            texts = [render_chat(tok, r['prompt'], not a.no_chat_template) for r in chunk]
            t1 = time.time()
            gen = generate(model, tok, texts, a.samples, a.temperature, a.max_new)
            dt = (time.time() - t1) / max(len(chunk) * a.samples, 1)
            for r, samples in zip(chunk, gen):
                for j, (txt, ntok, finish) in enumerate(samples):
                    think, answer = RW.split_think(txt)
                    try:
                        n_think = len(tok(think, add_special_tokens=False).input_ids)
                    except Exception:
                        n_think = int(len(think) / 3.6)
                    rw = RW.reward(txt, r['label'], r['family'], n_think)
                    logger.log(step=None, checkpoint=a.checkpoint_tag, uid=r['uid'],
                               fold=r['fold'], source_class=r['source_class'], label=r['label'],
                               family=r['family'], sample_idx=j, temperature=a.temperature,
                               prompt_sha1=sha1(r['prompt']), completion=txt, reward=rw,
                               n_think_tokens=n_think, n_answer_tokens=max(ntok - n_think, 0),
                               finish_reason=finish, wall_s=round(dt, 3))
                    n_ok += bool(rw['format_ok'])
                n_done += 1
            if b % (a.batch * 10) == 0:
                el = time.time() - t0
                print(f'[step264] {n_done}/{len(rows)} prompts  {el:.0f}s  '
                      f'eta {el / max(n_done, 1) * (len(rows) - n_done) / 60:.1f} min', flush=True)
    tot = n_done * a.samples
    print(f'[step264] {tot} rollouts logged to {a.traces}; format_ok {n_ok / max(tot, 1):.3f}')
    print('[step264] next: select_best_traces.py --traces <dir> --train-jsonl train.jsonl '
          '--heldout-jsonl heldout.jsonl --out sft_traces.jsonl')
    if a.smoke:
        assert os.path.exists(a.traces) and tot > 0, 'SMOKE FAILED: nothing logged'
        print('[smoke] OK - generation and trace logging ran on TRAIN prompts.')


if __name__ == '__main__':
    main()
