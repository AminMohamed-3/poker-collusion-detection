#!/usr/bin/env python
"""STEP264 - GRPO + QLoRA training of the hand-level collusion judge.

*** UNTESTED LOCALLY.  Written on a machine with no GPU, no torch and no network. ***
*** ALWAYS run `--smoke` first: it exercises this entire path on 8 prompts in a few minutes, ***
*** so a mistake costs minutes of GPU time instead of hours. ***

What it does
  4-bit nf4 QLoRA policy (peft LoRA r 32 on all linear layers) trained with TRL's GRPOTrainer on
  train.jsonl.  The reward wraps reward.py, and - important - it logs EVERY completion through
  trace_logger.py, because TRL's reward hook is the one place that sees all rollouts.

Model id is a REQUIRED argument; nothing is hard-coded.

Version drift: TRL, peft and transformers rename arguments often.  Every config object here is
built by filtering a dict of wishes against the real signature (see `filtered`), so an unknown
argument is dropped with a printed warning instead of crashing after the model has loaded.

  python train_grpo_qlora.py --model Qwen/Qwen3-8B --train train.jsonl --out runs/r1 --smoke
  python train_grpo_qlora.py --model Qwen/Qwen3-8B --train train.jsonl --out runs/r1 --vllm
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import reward as RW                                                          # noqa: E402
from trace_logger import TraceLogger, sha1                                   # noqa: E402

BANNER = '*** STEP264 vm/train_grpo_qlora_v4.py (v3 + --vllm-sleep) - UNTESTED ON A GPU. Run --smoke first. ***'


# ---------------------------------------------------------------------------- helpers


def filtered(cls, wishes: dict, name: str) -> dict:
    """Keep only the kwargs this installed version actually accepts."""
    try:
        sig = set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        return dict(wishes)
    keep, drop = {}, []
    for k, v in wishes.items():
        if k in sig:
            keep[k] = v
        else:
            drop.append(k)
    if drop:
        print(f'[warn] {name}: this version does not accept {sorted(drop)}; dropped.')
    return keep


def text_of(x) -> str:
    """A prompt/completion may be a plain string or a list of chat message dicts."""
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get('content', ''))
    if isinstance(x, (list, tuple)):
        return '\n'.join(text_of(e) for e in x)
    return str(x)


def load_jsonl(path: str) -> list:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


# ---------------------------------------------------------------------------- reward hook


def make_reward_fn(tok, logger, state, args):
    """TRL calls this once per generation batch with the dataset columns as keyword lists."""

    def step264_reward(completions=None, **kw):
        completions = completions if completions is not None else kw.get('completions') or []
        n = len(completions)

        def col(name, default=None):
            v = kw.get(name)
            if v is None:
                return [default] * n
            return list(v) if isinstance(v, (list, tuple)) else [v] * n

        prompts = col('prompts')
        uid, label = col('uid'), col('label', 0)
        family, source_class, fold = col('family', 'NONE'), col('source_class'), col('fold')
        totals = []
        for i, c in enumerate(completions):
            t = text_of(c)
            think, answer = RW.split_think(t)
            try:
                n_think = len(tok(think, add_special_tokens=False).input_ids) if think else 0
                n_ans = len(tok(answer, add_special_tokens=False).input_ids)
            except Exception:                       # never let tokenisation kill a training step
                n_think, n_ans = int(len(think) / 3.6), int(len(answer) / 3.6)
            r = RW.reward(t, label[i], family[i], n_think)
            finish = 'length' if (n_think + n_ans) >= args.max_completion - 2 else 'stop'
            try:
                logger.log(step=state.get('step'), checkpoint=state.get('ckpt', 'unknown'),
                           uid=uid[i], fold=fold[i], source_class=source_class[i],
                           label=label[i], family=family[i],
                           sample_idx=i % max(args.group_size, 1),
                           temperature=args.temperature,
                           prompt_sha1=sha1(text_of(prompts[i])) if prompts[i] is not None
                           else None,
                           completion=t, reward=r, n_think_tokens=n_think, n_answer_tokens=n_ans,
                           finish_reason=finish, wall_s=None)
            except Exception as e:                  # logging must never break training
                print(f'[warn] trace logging failed: {e}')
            totals.append(float(r['total']))
        state['seen'] = state.get('seen', 0) + n
        fo = sum(1 for c in completions if RW.parse(text_of(c))['format_ok'])
        state['format_ok'] = state.get('format_ok', 0) + fo
        return totals

    return step264_reward


# ---------------------------------------------------------------------------- main


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='STEP264 GRPO + QLoRA (UNTESTED ON GPU)')
    ap.add_argument('--model', required=True, help='HF model id or local path (NOT hard-coded)')
    ap.add_argument('--train', default='train.jsonl')
    ap.add_argument('--out', default='runs/step264')
    ap.add_argument('--traces', default=None, help='default <out>/traces/train.jsonl.gz')
    ap.add_argument('--run-id', default=None)
    # GRPO
    ap.add_argument('--group-size', type=int, default=8, help='completions per prompt')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-completion', type=int, default=900, help='>= thinking budget + answer')
    ap.add_argument('--max-prompt', type=int, default=1024)
    ap.add_argument('--beta', type=float, default=.005, help='KL to the frozen base')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--per-device-batch', type=int, default=8, help='COMPLETIONS per device step')
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--save-steps', type=int, default=50)
    ap.add_argument('--logging-steps', type=int, default=1)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=264)
    # LoRA / quantisation
    ap.add_argument('--lora-r', type=int, default=32)
    ap.add_argument('--lora-alpha', type=int, default=64)
    ap.add_argument('--lora-dropout', type=float, default=.05)
    ap.add_argument('--no-4bit', action='store_true', help='full bf16 policy (needs a big card)')
    # rollouts
    ap.add_argument('--vllm', action='store_true', help='use vLLM for rollouts if installed')
    ap.add_argument('--vllm-gpu-frac', type=float, default=.3)
    ap.add_argument('--no-chat-template', action='store_true',
                    help='feed the raw prompt string instead of a one-turn chat message')
    ap.add_argument('--limit', type=int, default=0, help='train on the first N prompts only')
    ap.add_argument('--vllm-max-model-len', type=int, default=None)
    ap.add_argument('--vllm-sleep', action='store_true', help='vllm_enable_sleep_mode=True: vLLM releases GPU memory during the training pass')
    ap.add_argument('--no-vllm-is', action='store_true', help='vllm_importance_sampling_correction=False (sequence_mask ratio underflows on long completions)')
    ap.add_argument('--chat-think', choices=['on', 'off', 'default'], default='default')
    ap.add_argument('--smoke', action='store_true',
                    help='8 prompts, 4 completions, 2 steps, short completions, no vLLM')
    return ap


def main(argv=None) -> None:
    print(BANNER)
    args = build_argparser().parse_args(argv)
    if args.smoke:
        args.limit = 8
        args.group_size = 4
        args.per_device_batch = 4
        args.grad_accum = 1
        args.max_steps = 2
        args.max_completion = 256
        args.save_steps = 1000000
        args.vllm = False
        args.out = args.out.rstrip('/') + '_smoke'
        print('[smoke] 8 prompts, group 4, 2 steps, max_completion 256, vLLM off')

    glob_batch = args.per_device_batch * args.grad_accum
    assert glob_batch % args.group_size == 0, (
        f'per-device-batch * grad-accum ({glob_batch}) must be a multiple of group-size '
        f'({args.group_size}); GRPO needs whole groups in a step')

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    run_id = args.run_id or f'step264_{int(time.time())}'
    traces = args.traces or os.path.join(args.out, 'traces', 'train.jsonl.gz')

    rows = load_jsonl(args.train)
    if args.limit:
        rows = rows[:args.limit]
    print(f'{len(rows)} training prompts from {args.train}')

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    def to_example(r):
        p = r['prompt'] if args.no_chat_template else [{'role': 'user', 'content': r['prompt']}]
        return {'prompt': p, 'uid': r['uid'], 'label': int(r['label']),
                'family': r['family'], 'source_class': r['source_class'], 'fold': int(r['fold'])}

    ds = Dataset.from_list([to_example(r) for r in rows])

    qcfg = None
    if not args.no_4bit:
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    mk = dict(quantization_config=qcfg, trust_remote_code=True, device_map={'': 0},
              attn_implementation='sdpa')
    try:                                   # transformers >= 4.56 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **mk)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **mk)
    model.config.use_cache = False

    lora = LoraConfig(**filtered(LoraConfig, dict(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias='none', task_type='CAUSAL_LM', target_modules='all-linear'), 'LoraConfig'))

    wishes = dict(
        output_dir=args.out, seed=args.seed, bf16=True, report_to=[],
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_strategy='steps', save_total_limit=20,
        num_generations=args.group_size, temperature=args.temperature, top_p=.95,
        max_completion_length=args.max_completion, max_prompt_length=args.max_prompt,
        beta=args.beta, scale_rewards=True, log_completions=False,
        remove_unused_columns=False,
    )
    if args.vllm:
        wishes.update(use_vllm=True, vllm_mode='colocate',
                      vllm_gpu_memory_utilization=args.vllm_gpu_frac)
        if args.vllm_max_model_len:
            wishes['vllm_max_model_length'] = args.vllm_max_model_len
        if args.no_vllm_is:
            wishes['vllm_importance_sampling_correction'] = False
        if args.vllm_sleep:
            wishes['vllm_enable_sleep_mode'] = True
    if args.chat_think != 'default':
        wishes['chat_template_kwargs'] = {'enable_thinking': args.chat_think == 'on'}
    cfg = GRPOConfig(**filtered(GRPOConfig, wishes, 'GRPOConfig'))

    state = {'step': 0, 'ckpt': 'step_00000'}

    class StepCB(TrainerCallback):
        def on_step_begin(self, a, s, c, **kw):
            state['step'] = int(s.global_step)
            state['ckpt'] = f'step_{int(s.global_step):05d}'

        def on_log(self, a, s, c, logs=None, **kw):
            if state.get('seen'):
                print(f'[step264] rollouts {state["seen"]}  '
                      f'format_ok {state["format_ok"] / state["seen"]:.3f}')

    with TraceLogger(traces, run_id=run_id) as logger:
        fn = make_reward_fn(tok, logger, state, args)
        tkw = filtered(GRPOTrainer, dict(
            model=model, reward_funcs=[fn], args=cfg, train_dataset=ds, peft_config=lora,
            processing_class=tok), 'GRPOTrainer')
        if 'processing_class' not in tkw:          # older TRL called it tokenizer
            tkw['tokenizer'] = tok
        trainer = GRPOTrainer(**tkw)
        trainer.add_callback(StepCB())
        json.dump({'run_id': run_id, 'model': args.model, 'args': vars(args),
                   'n_prompts': len(rows), 'traces': traces, 'banner': BANNER},
                  open(os.path.join(args.out, 'run_config.json'), 'w', encoding='utf-8'), indent=1)
        trainer.train()
        trainer.save_model(os.path.join(args.out, 'adapter_final'))

    seen = state.get('seen', 0)
    print(f'[step264] done. rollouts logged: {seen}  '
          f'format_ok share: {state.get("format_ok", 0) / max(seen, 1):.3f}')
    print(f'[step264] traces: {traces}   adapter: {os.path.join(args.out, "adapter_final")}')
    if args.smoke:
        assert seen > 0, 'SMOKE FAILED: the reward hook never saw a completion'
        assert os.path.exists(traces), 'SMOKE FAILED: no trace file was written'
        print('[smoke] OK - reward hook fired, traces written, adapter saved.')


if __name__ == '__main__':
    main()
