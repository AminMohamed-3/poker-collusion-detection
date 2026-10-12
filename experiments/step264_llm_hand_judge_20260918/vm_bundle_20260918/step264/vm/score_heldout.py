#!/usr/bin/env python
"""STEP264 - score heldout.jsonl with the FINAL adapter and write heldout_scores.jsonl.

*** UNTESTED LOCALLY.  Written on a machine with no GPU, no torch and no network. ***
*** Run `--smoke` first (8 prompts, a few minutes). ***

Loads the base model (4-bit nf4 by default) plus the final LoRA adapter, draws k samples per
held-out prompt at temperature 0.7, and writes exactly the format score_heldout_eval.py reads:

    {"uid": "...", "mean_conf": 47.5, "k": 4,
     "confidences": [40, 55, 45, 50],
     "verdict_counts": {"NONE": 3, "SOFT_PLAY": 1}}

mean_conf is the mean over the samples that parsed; k is how many of them survived.  A prompt
where NO sample parsed is written with mean_conf 50.0, k 0 and a "malformed": true flag - the
local evaluator still joins it, and a large number of those is itself the result.

Every rollout is also logged through trace_logger.py with checkpoint tag 'final_heldout'.
select_best_traces.py refuses any rollout whose checkpoint tag contains "heldout", so these can
never leak into the student's SFT set.

  python score_heldout.py --model Qwen/Qwen3-8B --adapter runs/r1/adapter_final --smoke
  python score_heldout.py --model Qwen/Qwen3-8B --adapter runs/r1/adapter_final
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
from trace_logger import TraceLogger, sha1                                   # noqa: E402

BANNER = '*** STEP264 vm/score_heldout.py - UNTESTED ON A GPU. Run --smoke first. ***'
KEEP_ALWAYS = ('POS', 'NEG_HARD', 'NEG_WITHIN', 'AUDIT')


def load_jsonl(path: str) -> list:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_policy(model_id: str, adapter: str | None, four_bit: bool = True):
    """Base model (+ optional LoRA adapter) in eval mode, and its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16) if four_bit else None)
    mk = dict(quantization_config=q, trust_remote_code=True, device_map={'': 0},
              attn_implementation='sdpa')
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **mk)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **mk)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f'[step264] adapter loaded: {adapter}')
    else:
        print('[step264] NO adapter: this is the BASE model (use --adapter for the trained one)')
    model.eval()
    model.config.use_cache = True
    return model, tok


def render_chat(tok, prompt: str, chat: bool) -> str:
    if not chat:
        return prompt
    try:
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                       add_generation_prompt=True)
    except Exception as e:
        print(f'[warn] chat template failed ({e}); falling back to the raw prompt')
        return prompt


def generate(model, tok, texts, k, temperature, max_new_tokens, top_p=.95):
    """Return, per input text, a list of k (completion_text, n_new_tokens, finish_reason)."""
    import torch
    enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
              max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                             max_new_tokens=max_new_tokens, num_return_sequences=k,
                             pad_token_id=tok.pad_token_id)
    plen = enc['input_ids'].shape[1]
    eos = {tok.eos_token_id, tok.pad_token_id}
    res = []
    for i in range(len(texts)):
        got = []
        for j in range(k):
            seq = out[i * k + j][plen:]
            ids = [int(t) for t in seq.tolist()]
            while ids and ids[-1] == tok.pad_token_id:
                ids.pop()
            finish = 'stop' if (ids and ids[-1] in eos) else 'length'
            got.append((tok.decode(ids, skip_special_tokens=True), len(ids), finish))
        res.append(got)
    return res


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='STEP264 held-out scoring (UNTESTED ON GPU)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', default=None, help='runs/<r>/adapter_final')
    ap.add_argument('--data', default='heldout.jsonl')
    ap.add_argument('--out', default='heldout_scores.jsonl')
    ap.add_argument('--traces', default='traces/heldout.jsonl.gz')
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--checkpoint-tag', default='final_heldout')
    ap.add_argument('-k', '--samples', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=.7)
    ap.add_argument('--max-new', type=int, default=900)
    ap.add_argument('--batch', type=int, default=4, help='PROMPTS per generate call (x k seqs)')
    ap.add_argument('--no-4bit', action='store_true')
    ap.add_argument('--no-chat-template', action='store_true')
    ap.add_argument('--limit', type=int, default=0,
                    help='score at most N prompts; POS/NEG_HARD/NEG_WITHIN/AUDIT are kept first '
                         'and NEG_RANDOM is subsampled (seeded)')
    ap.add_argument('--seed', type=int, default=264)
    ap.add_argument('--smoke', action='store_true')
    return ap


def choose(rows, limit, seed):
    if not limit or limit >= len(rows):
        return rows
    import random as _r
    rng = _r.Random(seed)
    keep = [r for r in rows if r['source_class'] in KEEP_ALWAYS]
    rest = [r for r in rows if r['source_class'] not in KEEP_ALWAYS]
    rng.shuffle(keep)
    rng.shuffle(rest)
    out = keep[:limit] + rest[:max(0, limit - len(keep))]
    print(f'[step264] --limit {limit}: scoring {len(out)} of {len(rows)} prompts '
          f'(NEG_RANDOM subsampled; the local evaluator reports the surviving class counts)')
    return out


def main(argv=None) -> None:
    print(BANNER)
    a = build_argparser().parse_args(argv)
    if a.smoke:
        a.limit, a.samples, a.max_new, a.batch = 8, 2, 256, 2
        a.out = a.out.replace('.jsonl', '_smoke.jsonl')
        a.traces = a.traces.replace('.jsonl.gz', '_smoke.jsonl.gz')
        print('[smoke] 8 prompts, k=2, max_new 256')
    rows = choose(load_jsonl(a.data), a.limit, a.seed)
    model, tok = load_policy(a.model, a.adapter, not a.no_4bit)
    run_id = a.run_id or f'step264_score_{int(time.time())}'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)

    t0 = time.time()
    n_done = n_malformed = 0
    with TraceLogger(a.traces, run_id=run_id) as logger, \
            open(a.out, 'w', encoding='utf-8', newline='\n') as fh:
        for b in range(0, len(rows), a.batch):
            chunk = rows[b:b + a.batch]
            texts = [render_chat(tok, r['prompt'], not a.no_chat_template) for r in chunk]
            t1 = time.time()
            gen = generate(model, tok, texts, a.samples, a.temperature, a.max_new)
            dt = (time.time() - t1) / max(len(chunk) * a.samples, 1)
            for r, samples in zip(chunk, gen):
                confs, verdicts = [], {}
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
                    if rw['format_ok']:
                        confs.append(int(rw['confidence']))
                        verdicts[rw['verdict']] = verdicts.get(rw['verdict'], 0) + 1
                row = {'uid': r['uid'], 'k': len(confs),
                       'mean_conf': round(sum(confs) / len(confs), 3) if confs else 50.0,
                       'confidences': confs, 'verdict_counts': verdicts}
                if not confs:
                    row['malformed'] = True
                    n_malformed += 1
                fh.write(json.dumps(row) + '\n')
                n_done += 1
            if b % (a.batch * 10) == 0:
                el = time.time() - t0
                print(f'[step264] {n_done}/{len(rows)} prompts  {el:.0f}s  '
                      f'eta {el / max(n_done, 1) * (len(rows) - n_done) / 60:.1f} min', flush=True)
    print(f'[step264] wrote {a.out}: {n_done} rows, {n_malformed} with no parsable sample '
          f'({n_malformed / max(n_done, 1):.1%})')
    print(f'[step264] traces: {a.traces}')
    print('[step264] copy heldout_scores.jsonl back and run score_heldout_eval.py LOCALLY; '
          'the uid -> pair/hand map never comes here.')
    if a.smoke:
        assert n_done == len(rows) and os.path.getsize(a.out) > 0, 'SMOKE FAILED: no output'
        assert os.path.exists(a.traces), 'SMOKE FAILED: no trace file'
        print('[smoke] OK - generation, parsing, scoring and trace logging all ran.')


if __name__ == '__main__':
    main()
