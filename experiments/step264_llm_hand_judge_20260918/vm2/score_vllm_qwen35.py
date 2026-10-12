"""STEP264 r2  vm2/score_vllm_qwen35.py -- vLLM offline scorer for Qwen3.5-9B (base or LoRA).

Byte-for-byte the SAME outputs as vm/score_vllm.py: rows {uid, k, mean_conf, confidences,
verdict_counts[, malformed]}, a <out>_summary.json with the same keys, and a trace file written
through trace_logger.py.  Only the engine handling differs.

Differences from the v1 scorer, all Qwen3.5-specific:
  * THINKING IS ON BY DEFAULT and is always passed EXPLICITLY through the chat template, because
    the documented default disagrees between sources (HF card: enable_thinking defaults True;
    Unsloth's Qwen3.5 page: the 9B "disables by default").  --no-think flips it to False.
  * --max-model-len 18432 (prompt ~4-5k + a 12k completion cap + headroom).
  * Qwen3.5 is a hybrid gated-DeltaNet + full-attention MULTIMODAL model.  Engine kwargs that
    exist only on that path (mamba cache mode, separate object groups, text-only loading) are
    passed as EXTRAS and automatically retried without them if the installed vLLM rejects them,
    so an unknown kwarg costs one failed constructor call instead of the run.
  * Thinking-mode sampling defaults follow the model card: temperature 1.0 is the model's own
    recommendation, but we keep --temperature .7 as the DEFAULT here so the scorer stays
    comparable with the v1 numbers already on file.  Pass --temperature 1.0 --top-k 20 to score
    at the vendor setting.

  python score_vllm_qwen35.py --model Qwen/Qwen3.5-9B --data ../heldout_subset.jsonl \
      --out base_scores.jsonl --traces base_traces.jsonl.gz -k 2 --max-new 12288
  python score_vllm_qwen35.py --model Qwen/Qwen3.5-9B --adapter /workspace/runs/r2/adapter_final ...
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


def build_llm(llm_kw, extras):
    """Construct vllm.LLM, dropping Qwen3.5-only kwargs one round at a time if rejected."""
    from vllm import LLM
    extras = dict(extras)
    while True:
        try:
            return LLM(**llm_kw, **extras)
        except TypeError as e:
            if not extras:
                raise
            print(f'[vllm] constructor rejected {sorted(extras)} ({e}); retrying without them.',
                  flush=True)
            extras = {}
        except ValueError as e:
            if not extras:
                raise
            print(f'[vllm] engine rejected {sorted(extras)} ({e}); retrying without them.',
                  flush=True)
            extras = {}


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
    ap.add_argument('--top-k', type=int, default=0, help='0 = off; 20 is the Qwen3.5 thinking rec')
    ap.add_argument('--max-new', type=int, default=12288)
    ap.add_argument('--max-model-len', type=int, default=18432)
    ap.add_argument('--gpu-frac', type=float, default=.85)
    ap.add_argument('--max-lora-rank', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0, help='first N rows of --data')
    ap.add_argument('--seed', type=int, default=264)
    ap.add_argument('--no-think', action='store_true')
    ap.add_argument('--no-chat-template', action='store_true')
    ap.add_argument('--no-prefix-caching', action='store_true')
    ap.add_argument('--enforce-eager', action='store_true',
                    help="enforce_eager=True: skip vLLM's torch.compile/CUDA graphs.  Needed on "
                         'vLLM 0.20.2 + Qwen3.5, whose graph splitter dies with '
                         '"Tried to erase Node size but it still had 2 users" (STEP264 r3).')
    ap.add_argument('--language-model-only', action='store_true', default=True,
                    help='skip the vision encoder to free KV cache (default on)')
    ap.add_argument('--with-vision', dest='language_model_only', action='store_false')
    ap.add_argument('--mamba-cache-mode', default='align',
                    help="hybrid-model KV/SSM cache mode; 'align' is the documented setting for "
                         "gated-DeltaNet models.  Pass '' to omit the kwarg.")
    ap.add_argument('--dry-run', action='store_true',
                    help='load data, print the resolved engine config and exit without vLLM')
    a = ap.parse_args(argv)

    # r1: FlashInfer's sampler failed on sm_120 (RTX PRO 6000 Blackwell).
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')

    rows = load_jsonl(a.data)
    if a.limit:
        rows = rows[:a.limit]
    think = not a.no_think
    print(f'[vllm] {len(rows)} prompts from {a.data}  k={a.samples}  max_new={a.max_new}  '
          f'think={think}', flush=True)
    print(f'[vllm] reward length constants: LEN_FREE={RW.LEN_FREE} LEN_STEP={RW.LEN_STEP} '
          f'LEN_RATE={RW.LEN_RATE}', flush=True)

    llm_kw = dict(model=a.model, dtype='bfloat16', gpu_memory_utilization=a.gpu_frac,
                  max_model_len=a.max_model_len,
                  enable_prefix_caching=not a.no_prefix_caching, seed=a.seed,
                  trust_remote_code=True)
    if a.enforce_eager:
        llm_kw['enforce_eager'] = True
    extras = {}
    if a.language_model_only:
        extras['language_model_only'] = True
    if a.mamba_cache_mode:
        extras['mamba_cache_mode'] = a.mamba_cache_mode
    if a.adapter:
        llm_kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=1)

    if a.dry_run:
        print('--- RESOLVED ENGINE CONFIG (dry run) ---')
        print(json.dumps({'llm_kw': llm_kw, 'extras': extras, 'adapter': a.adapter,
                          'enable_thinking': None if a.no_chat_template else think,
                          'sampling': {'n': a.samples, 'temperature': a.temperature,
                                       'top_p': a.top_p, 'top_k': a.top_k or None,
                                       'max_tokens': a.max_new},
                          'n_prompts': len(rows)}, indent=1))
        print('[dry-run] OK: data loaded, engine config built, vLLM not started.')
        return

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    def render(prompt):
        if a.no_chat_template:
            return prompt
        # ALWAYS explicit: this model's documented default is contradictory across sources.
        try:
            return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=think)
        except TypeError:
            # template does not take the kwarg -> fall back to chat_template_kwargs, then bare
            try:
                return tok.apply_chat_template(
                    [{'role': 'user', 'content': prompt}], tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={'enable_thinking': think})
            except TypeError:
                print('[warn] the chat template accepts no thinking kwarg; using its default.')
                return tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                               tokenize=False, add_generation_prompt=True)

    texts = [render(r['prompt']) for r in rows]
    lora_req = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest('step264', 1, a.adapter)
    llm = build_llm(llm_kw, extras)
    sp_kw = dict(n=a.samples, temperature=a.temperature, top_p=a.top_p,
                 max_tokens=a.max_new, seed=a.seed)
    if a.top_k:
        sp_kw['top_k'] = a.top_k
    sp = SamplingParams(**sp_kw)
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
                thk, answer = RW.split_think(txt)
                n_think = len(tok(thk, add_special_tokens=False).input_ids) if thk else 0
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
