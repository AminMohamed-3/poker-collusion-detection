"""STEP264 vm/score_vllm.py -- vLLM offline scorer for the hand judge (base model or LoRA adapter).

Same inputs/outputs as score_heldout.py (rows {uid, k, mean_conf, confidences, verdict_counts[, malformed]}
and a trace file), but generation goes through vLLM so a 2,600-prompt x k pass takes minutes, not hours.
The adapter is served through vLLM's LoRA support (no merge needed).

  python score_vllm.py --model $MODEL --data ../heldout_subset.jsonl --out base_scores.jsonl \
      --traces base_traces.jsonl.gz -k 2 --max-new 2048
  python score_vllm.py --model $MODEL --adapter /workspace/runs/r1/adapter_final ...

--no-think renders the chat template with enable_thinking=False (Qwen3 answers without a <think> block).
Prints a summary: format_ok share, completion-length quantiles, clipped share, mean reward by source_class.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward as RW  # noqa: E402
from trace_logger import TraceLogger, sha1  # noqa: E402


def load_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--data', default='../heldout_subset.jsonl')
    ap.add_argument('--out', default='scores.jsonl')
    ap.add_argument('--traces', default='traces.jsonl.gz')
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--checkpoint-tag', default='vllm')
    ap.add_argument('-k', '--samples', type=int, default=2)
    ap.add_argument('--temperature', type=float, default=.7)
    ap.add_argument('--top-p', type=float, default=.95)
    ap.add_argument('--max-new', type=int, default=2048)
    ap.add_argument('--max-model-len', type=int, default=6144)
    ap.add_argument('--gpu-frac', type=float, default=.85)
    ap.add_argument('--limit', type=int, default=0, help='first N rows of --data')
    ap.add_argument('--seed', type=int, default=264)
    ap.add_argument('--no-think', action='store_true')
    ap.add_argument('--no-chat-template', action='store_true')
    a = ap.parse_args(argv)

    rows = load_jsonl(a.data)
    if a.limit:
        rows = rows[:a.limit]
    print(f'[vllm] {len(rows)} prompts from {a.data}  k={a.samples}  max_new={a.max_new}  '
          f'think={not a.no_think}', flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    def render(prompt):
        if a.no_chat_template:
            return prompt
        kw = {} if not a.no_think else {'enable_thinking': False}
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                       add_generation_prompt=True, **kw)

    texts = [render(r['prompt']) for r in rows]
    llm_kw = dict(model=a.model, dtype='bfloat16', gpu_memory_utilization=a.gpu_frac,
                  max_model_len=a.max_model_len, enable_prefix_caching=True, seed=a.seed,
                  trust_remote_code=True)
    lora_req = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        llm_kw.update(enable_lora=True, max_lora_rank=64, max_loras=1)
        lora_req = LoRARequest('step264', 1, a.adapter)
    llm = LLM(**llm_kw)
    sp = SamplingParams(n=a.samples, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_new, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(texts, sp, lora_request=lora_req, use_tqdm=True)
    gen_s = time.time() - t0
    run_id = a.run_id or f'step264_vllm_{int(time.time())}'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    n_malformed = 0
    lens, fmt, clipped, think_lens = [], [], [], []
    tot_by_cls, n_by_cls = {}, {}
    with TraceLogger(a.traces, run_id=run_id) as logger, \
            open(a.out, 'w', encoding='utf-8', newline='\n') as fh:
        for r, o in zip(rows, outs):
            confs, verdicts = [], {}
            for j, c in enumerate(o.outputs):
                txt = c.text
                ntok = len(c.token_ids)
                finish = 'length' if c.finish_reason == 'length' else 'stop'
                think, answer = RW.split_think(txt)
                n_think = len(tok(think, add_special_tokens=False).input_ids) if think else 0
                rw = RW.reward(txt, r['label'], r['family'], n_think)
                logger.log(step=None, checkpoint=a.checkpoint_tag, uid=r['uid'], fold=r['fold'],
                           source_class=r['source_class'], label=r['label'], family=r['family'],
                           sample_idx=j, temperature=a.temperature, prompt_sha1=sha1(r['prompt']),
                           completion=txt, reward=rw, n_think_tokens=n_think,
                           n_answer_tokens=max(ntok - n_think, 0), finish_reason=finish,
                           wall_s=round(gen_s / max(len(rows) * a.samples, 1), 4))
                lens.append(ntok)
                think_lens.append(n_think)
                fmt.append(bool(rw['format_ok']))
                clipped.append(finish == 'length')
                cls = r['source_class']
                tot_by_cls[cls] = tot_by_cls.get(cls, 0.0) + rw['total']
                n_by_cls[cls] = n_by_cls.get(cls, 0) + 1
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
    lens_s = sorted(lens)

    def q(p):
        return lens_s[min(int(p * len(lens_s)), len(lens_s) - 1)]

    summ = {'prompts': len(rows), 'samples': len(lens), 'gen_s': round(gen_s, 1),
            'tok_per_s': round(sum(lens) / max(gen_s, 1e-9)),
            'format_ok': round(sum(fmt) / max(len(fmt), 1), 4),
            'clipped': round(sum(clipped) / max(len(clipped), 1), 4),
            'malformed_prompts': n_malformed,
            'len_q10_50_90_max': [q(.1), q(.5), q(.9), lens_s[-1]],
            'think_mean': round(sum(think_lens) / max(len(think_lens), 1)),
            'mean_reward_by_class': {c: round(tot_by_cls[c] / n_by_cls[c], 4)
                                     for c in sorted(tot_by_cls)}}
    print('[vllm] SUMMARY ' + json.dumps(summ), flush=True)
    with open(a.out.replace('.jsonl', '_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summ, fh, indent=1)
    print(f'[vllm] wrote {a.out} and {a.traces}')


if __name__ == '__main__':
    main()
