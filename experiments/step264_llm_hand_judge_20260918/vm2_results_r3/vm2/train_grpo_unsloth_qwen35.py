#!/usr/bin/env python
"""STEP264 run r2 - GRPO fine-tuning of Qwen3.5-9B (thinking) with the UNSLOTH pipeline.

*** UNTESTED ON A GPU.  Written on a Windows box with no torch and no CUDA. ***
*** ALWAYS run --dry-run, then --smoke, before the real run.  See RUNBOOK_v2.txt. ***

This follows UNSLOTH'S PUBLISHED Qwen3.5 GRPO RECIPE, i.e. the notebook
nb/Qwen3_5_(4B)_Vision_GRPO.ipynb in github.com/unslothai/notebooks, adapted to a 9B text-only
task.  The three things that recipe does differently from a generic TRL GRPO script, and that
this file therefore does too:

  * FastVisionModel, not FastLanguageModel.  Qwen3.5 is Qwen3_5ForConditionalGeneration - a
    multimodal hybrid-attention model - and Unsloth loads the whole family through its vision
    path even when you never pass an image.  LoRA is confined to the language side by
    finetune_vision_layers=False, which is also what vLLM requires ("vLLM also does not yet
    support LoRA on the vision layers, so we can only add them on the language layers").
  * fast_inference=False.  Unsloth's docs are explicit: "You can run Qwen3.5 RL with Unsloth even
    though it is not supported by vLLM, by setting fast_inference=False".  Generation therefore
    goes through HF/Unsloth generate.  --fast-inference opts into the vLLM path if a later
    Unsloth/vLLM pair supports it; only then are the use_vllm / vllm_* keys sent at all.
  * Conversational prompts whose content is a LIST OF PARTS
    ([{"type": "text", "text": ...}]), which is what the vision processor expects.

Run r1's three failures still shape the defaults (registration264_r1.json, amendments r1b/r1e/r1f):
  r1b  a SEQUENCE-level importance-sampling ratio underflowed the loss to zero on 4k+ token
       completions (grad_norm 1.6e-20 at reward_std .49).  --importance-sampling-level defaults
       to 'token' here and the notebook's 'sequence' is NOT adopted.
  r1f  fp32 logits (batch x seq x vocab) OOMed the backward at micro-batch 4, and Qwen3.5's vocab
       is ~248,320 against Qwen3's ~152k.  --micro-batch defaults to 2.
  r1e  the delivered length penalty dominated the reward at 4.5k thinking tokens.  reward.py
       reads STEP264_LEN_FREE / LEN_STEP / LEN_RATE from the environment and this script PRINTS
       the values in force on its first line.

Version drift: every config object is built by filtering a dict of wishes against the real
installed signature (see `filtered`), so an unknown argument is dropped with a printed warning
instead of crashing after an 18 GB model has loaded.  Keys in CRITICAL_KEYS additionally raise a
loud banner when dropped.  The box runs TRL 0.24.0 (not the notebook's 0.22.2), so the filter is
doing real work - read its warnings.

  python train_grpo_unsloth_qwen35.py --train ../v2/train_v2.jsonl --dry-run --limit 16
  python train_grpo_unsloth_qwen35.py --train ../v2/train_v2.jsonl --out runs/r2 --smoke
  python train_grpo_unsloth_qwen35.py --train ../v2/train_v2.jsonl --out runs/r2 --max-steps 100
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
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reward as RW                                                          # noqa: E402
from trace_logger import TraceLogger, sha1                                   # noqa: E402

BANNER = ('*** STEP264 r2  vm2/train_grpo_unsloth_qwen35.py  (Unsloth published Qwen3.5 GRPO '
          'recipe) - UNTESTED ON A GPU.  Run --dry-run then --smoke first. ***')

# Dropping any of these silently would reproduce a known r1 failure or void the experiment.
CRITICAL_KEYS = {
    'num_generations',                       # without it GRPO has no groups
    'max_completion_length',                 # without it completions hit a short default
    'importance_sampling_level',             # r1b: 'sequence' underflowed the loss to zero
}

# Language-loader only: the standard attention + MLP projections.  The vision loader instead uses
# Unsloth's finetune_* switches, which is what the published notebook does.
LM_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj']
VISION_PATH_HINTS = ('visual', 'vision_tower', 'vision_model', 'image_encoder', 'merger',
                     'mm_projector', 'multi_modal_projector', 'audio_tower')


# ---------------------------------------------------------------------------- helpers


def _has_var_kw(obj) -> bool:
    try:
        return any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return False


def filtered(obj, wishes: dict, name: str, report: dict | None = None) -> dict:
    """Keep only the kwargs this installed version actually accepts."""
    try:
        sig = set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        print(f'[warn] {name}: signature not introspectable; passing every wish through.')
        return dict(wishes)
    if _has_var_kw(obj):
        print(f'[warn] {name}: accepts **kwargs; passing every wish through unfiltered.')
        return dict(wishes)
    keep, drop = {}, []
    for k, v in wishes.items():
        if k in sig:
            keep[k] = v
        else:
            drop.append(k)
    if drop:
        print(f'[warn] {name}: this version does not accept {sorted(drop)}; dropped.')
        if report is not None:
            report.setdefault(name, []).extend(sorted(drop))
        crit = sorted(set(drop) & CRITICAL_KEYS)
        if crit:
            print('!' * 100)
            print(f'[CRITICAL] {name} dropped {crit}.  These guard a KNOWN r1 failure mode.')
            print('[CRITICAL] Find the new spelling in this TRL version BEFORE the real run.')
            print('!' * 100)
    return keep


def text_of(x) -> str:
    """Prompt/completion may be a string, a chat message, or content as a list of typed parts."""
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if 'content' in x:
            return text_of(x['content'])
        if 'text' in x:                     # {'type': 'text', 'text': ...}
            return str(x['text'])
        return ''                           # an image/video part contributes no text
    if isinstance(x, (list, tuple)):
        return '\n'.join(t for t in (text_of(e) for e in x) if t)
    return str(x)


def load_jsonl(path: str) -> list:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def env_lengths() -> dict:
    """The reward length constants ACTUALLY in force, read back from the imported module."""
    return {'STEP264_LEN_FREE': RW.LEN_FREE, 'STEP264_LEN_STEP': RW.LEN_STEP,
            'STEP264_LEN_RATE': RW.LEN_RATE,
            'raw_env': {k: os.environ.get(k) for k in
                        ('STEP264_LEN_FREE', 'STEP264_LEN_STEP', 'STEP264_LEN_RATE')}}


def inner_tokenizer(tok):
    """FastVisionModel returns a PROCESSOR; the plain tokenizer under it is what counts tokens."""
    return getattr(tok, 'tokenizer', tok)


def count_tokens(tk, text: str) -> int:
    """Token count with a hard fallback; never raise inside a training step."""
    if not text:
        return 0
    try:
        return len(tk(text, add_special_tokens=False).input_ids)
    except Exception:
        try:
            return len(tk.encode(text, add_special_tokens=False))
        except Exception:
            return int(len(text) / 3.6)


def measure_prompts(rows, model_id, use_tokenizer=True):
    """Longest prompt in TOKENS.  Real tokenizer when importable, else chars/3.6."""
    texts = [r['prompt'] for r in rows]
    if use_tokenizer:
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tk = inner_tokenizer(tk)
            lens = sorted(count_tokens(tk, t) for t in texts)
            return {'method': f'tokenizer({model_id})', 'n': len(lens), 'min': lens[0],
                    'median': lens[len(lens) // 2], 'p95': lens[int(.95 * (len(lens) - 1))],
                    'max': lens[-1]}
        except Exception as e:
            print(f'[warn] tokenizer unavailable for the length check ({type(e).__name__}: {e}); '
                  f'falling back to chars/3.6.')
    lens = sorted(int(len(t) / 3.6) for t in texts)
    return {'method': 'chars/3.6', 'n': len(lens), 'min': lens[0],
            'median': lens[len(lens) // 2], 'p95': lens[int(.95 * (len(lens) - 1))],
            'max': lens[-1]}


# ---------------------------------------------------------------------------- reward hook


def make_reward_fn(tok, logger, state, args):
    """TRL calls this once per generation batch with the dataset columns as keyword lists.

    Signature matches the published notebook's (prompts, completions, **kwargs).  Completions
    arrive as plain strings in standard mode and as [{'role': 'assistant', 'content': ...}]
    lists in conversational mode; text_of handles both, including content-as-parts.
    """
    tk = inner_tokenizer(tok)

    def step264_reward(prompts=None, completions=None, **kw):
        completions = completions if completions is not None else kw.get('completions') or []
        prompts = prompts if prompts is not None else kw.get('prompts')
        n = len(completions)

        now = time.time()
        prev = state.get('last_call')
        state['last_call'] = now
        per = round((now - prev) / n, 4) if (prev and n) else None   # wall seconds per rollout

        def col(name, default=None, src=None):
            v = src if src is not None else kw.get(name)
            if v is None:
                return [default] * n
            return list(v) if isinstance(v, (list, tuple)) else [v] * n

        prompts = col('prompts', None, prompts)
        uid, label = col('uid'), col('label', 0)
        family, source_class, fold = col('family', 'NONE'), col('source_class'), col('fold')
        totals = []
        for i, c in enumerate(completions):
            t = text_of(c)
            think, answer = RW.split_think(t)
            n_think = count_tokens(tk, think)
            n_ans = count_tokens(tk, answer)
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
                           finish_reason=finish, wall_s=per)
            except Exception as e:                  # logging must never break training
                print(f'[warn] trace logging failed: {e}')
            totals.append(float(r['total']))
            state['think_tok'] = state.get('think_tok', 0) + n_think
            if finish == 'length':
                state['clipped'] = state.get('clipped', 0) + 1
        state['seen'] = state.get('seen', 0) + n
        state['format_ok'] = state.get('format_ok', 0) + sum(
            1 for c in completions if RW.parse(text_of(c))['format_ok'])
        return totals

    return step264_reward


# ---------------------------------------------------------------------------- config


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='STEP264 r2 GRPO via Unsloth (UNTESTED ON GPU)')
    ap.add_argument('--model', default='unsloth/Qwen3.5-9B',
                    help='HF id or local path; Qwen/Qwen3.5-9B for the upstream weights')
    ap.add_argument('--train', default='../v2/train_v2.jsonl')
    ap.add_argument('--out', default='runs/r2')
    ap.add_argument('--traces', default=None, help='default <out>/traces/train.jsonl.gz')
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--limit', type=int, default=0, help='train on the first N prompts only')
    # ---- loader: the published recipe is the vision path
    ap.add_argument('--loader', choices=['vision', 'language'], default='vision',
                    help="'vision' = FastVisionModel, the PUBLISHED Qwen3.5 recipe (LoRA on the "
                         "language layers only).  'language' = FastLanguageModel fallback.")
    # ---- GRPO shape
    ap.add_argument('--group-size', type=int, default=8, help='completions per prompt')
    ap.add_argument('--prompts-per-step', type=int, default=8, help='prompts per optimiser step')
    ap.add_argument('--micro-batch', type=int, default=2,
                    help='per_device_train_batch_size in COMPLETIONS; r1 OOMed at 4 and Qwen3.5 '
                         'has a ~248k vocab, so keep this at 1-2')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=.95)
    ap.add_argument('--top-k', type=int, default=20, help='Qwen3.5 thinking-mode recommendation')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--beta', type=float, default=0.0,
                    help='KL to the frozen base; 0 skips the reference forward entirely')
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--save-steps', type=int, default=10)
    ap.add_argument('--logging-steps', type=int, default=1)
    ap.add_argument('--seed', type=int, default=264)
    # ---- optimiser / loss: values from the published notebook unless noted
    ap.add_argument('--optim', default='adamw_8bit', help='notebook value; needs bitsandbytes')
    ap.add_argument('--adam-beta1', type=float, default=.9)
    ap.add_argument('--adam-beta2', type=float, default=.99)
    ap.add_argument('--weight-decay', type=float, default=.1)
    ap.add_argument('--warmup-ratio', type=float, default=.1)
    ap.add_argument('--lr-scheduler', default='cosine')
    ap.add_argument('--max-grad-norm', type=float, default=.1)
    ap.add_argument('--loss-type', default='dr_grpo',
                    help='notebook value; TRL also has grpo / bnpo / dapo / gspo')
    ap.add_argument('--importance-sampling-level', choices=['token', 'sequence'], default='token',
                    help="NOT the notebook's 'sequence': that is the r1b underflow")
    ap.add_argument('--mask-truncated', action='store_true',
                    help='mask_truncated_completions=True (notebook default is False)')
    ap.add_argument('--scale-rewards', default='group',
                    help="TRL scale_rewards; 'group' / 'batch' / 'none' or a bool-ish string")
    # ---- lengths
    ap.add_argument('--max-completion', type=int, default=12288)
    ap.add_argument('--max-prompt', type=int, default=0,
                    help='max_prompt_length; 0 -> None (do NOT truncate: prompt v2 is ~5k)')
    ap.add_argument('--max-seq-len', type=int, default=16384,
                    help='Unsloth max_seq_length: prompt + completion')
    ap.add_argument('--allow-truncation', action='store_true',
                    help='proceed even when longest_prompt + max_completion > max_seq_len')
    ap.add_argument('--no-tokenize', action='store_true',
                    help='length check uses chars/3.6 instead of loading the real tokenizer')
    # ---- LoRA
    ap.add_argument('--lora-r', type=int, default=32)
    ap.add_argument('--lora-alpha', type=int, default=32, help='notebook uses alpha == r')
    ap.add_argument('--lora-dropout', type=float, default=0.0,
                    help='Unsloth fast paths require 0.0')
    ap.add_argument('--grad-checkpointing', default='unsloth',
                    help="use_gradient_checkpointing: 'unsloth' | 'True' | 'False'")
    # ---- engine
    ap.add_argument('--trl-vllm', dest='trl_vllm', action='store_true', default=False,
                    help="TRL's own vLLM colocate engine (use_vllm=True, sleep mode, weights re-synced each step; "
                         'the r1 path).  The Unsloth loader keeps fast_inference=False.  Added 2026-09-18 after '
                         "Unsloth's fast_inference path failed inside its empty-model builder for Qwen3.5.")
    ap.add_argument('--fast-inference', dest='fast_inference', action='store_true', default=False,
                    help='vLLM-backed rollouts.  OFF by default: Unsloth says Qwen3.5 RL needs '
                         'fast_inference=False because vLLM does not support it.')
    ap.add_argument('--gpu-frac', type=float, default=.85,
                    help='gpu_memory_utilization; only used when --fast-inference is on')
    ap.add_argument('--no-standby', dest='standby', action='store_false', default=True,
                    help='do not set UNSLOTH_VLLM_STANDBY=1 (only matters with --fast-inference)')
    ap.add_argument('--vllm-max-model-len', type=int, default=None,
                    help='default = --max-seq-len + 2048; only with --fast-inference')
    ap.add_argument('--load-in-4bit', action='store_true',
                    help='NOT recommended: Unsloth advises against QLoRA on Qwen3.5')
    ap.add_argument('--unsloth-mini-batch', type=int, default=0,
                    help='unsloth_grpo_mini_batch (hidden-state chunking); 0 = auto-tune')
    ap.add_argument('--unsloth-logit-chunk', type=int, default=0,
                    help='unsloth_logit_chunk_multiplier (logit chunking); 0 = auto-tune')
    # ---- prompt / thinking
    ap.add_argument('--think', choices=['on', 'off'], default='on',
                    help='enable_thinking; ALWAYS sent explicitly, never left to the default')
    ap.add_argument('--prerender-chat', action='store_true',
                    help='render the chat template HERE (enable_thinking baked in) and feed a '
                         'plain-text prompt.  Auto-enabled if GRPOConfig has no '
                         'chat_template_kwargs.')
    # ---- modes
    ap.add_argument('--no-rope-fix', action='store_true',
                    help='do NOT install the Qwen3.5 rope_deltas fix (see install_qwen35_rope_deltas_fix)')
    ap.add_argument('--smoke', action='store_true',
                    help='8 prompts, group 4, 2 steps, max_completion 512')
    ap.add_argument('--dry-run', action='store_true',
                    help='load data, run the length check, build the resolved config, print it '
                         'and exit 0 WITHOUT importing torch / unsloth / trl')
    return ap


def apply_smoke(args) -> None:
    args.limit = 8
    args.group_size = 4
    args.prompts_per_step = 2
    args.micro_batch = 1
    args.max_steps = 2
    args.max_completion = 512
    args.save_steps = 1000000
    args.out = args.out.rstrip('/') + '_smoke'
    print('[smoke] 8 prompts, group 4, 2 prompts/step, micro-batch 1, 2 steps, '
          'max_completion 512')


def resolve(args) -> dict:
    """Everything the run will do, as plain data.  No torch, no trl, no model."""
    completions_per_step = args.prompts_per_step * args.group_size
    if completions_per_step % args.micro_batch:
        raise SystemExit(
            f'micro-batch ({args.micro_batch}) must divide prompts-per-step x group-size '
            f'({completions_per_step}); GRPO needs whole groups inside a step')
    grad_accum = completions_per_step // args.micro_batch
    gc = args.grad_checkpointing
    gc_val = {'true': True, 'false': False}.get(str(gc).lower(), gc)
    sr = args.scale_rewards
    sr_val = {'true': True, 'false': False}.get(str(sr).lower(), sr)

    trainer_wishes = dict(
        output_dir=args.out, seed=args.seed, bf16=True, fp16=False, report_to='none',
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps, log_completions=False,
        save_steps=args.save_steps, save_strategy='steps', save_total_limit=20,
        num_generations=args.group_size,
        # ---- generation (must reach GenerationConfig even without vLLM)
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_completion_length=args.max_completion,
        max_prompt_length=(args.max_prompt or None),
        # ---- optimiser, from the published notebook
        optim=args.optim, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler, max_grad_norm=args.max_grad_norm,
        # ---- loss
        beta=args.beta, loss_type=args.loss_type,
        importance_sampling_level=args.importance_sampling_level,
        mask_truncated_completions=bool(args.mask_truncated),
        scale_rewards=sr_val,
        remove_unused_columns=False,
    )
    if args.fast_inference:
        trainer_wishes.update(
            use_vllm=True, vllm_mode='colocate',
            vllm_gpu_memory_utilization=args.gpu_frac,
            vllm_max_model_len=args.vllm_max_model_len or (args.max_seq_len + 2048),
            # r1b: never let a sequence-level vLLM/HF ratio multiply the loss
            vllm_importance_sampling_correction=False,
        )
    elif args.trl_vllm:
        trainer_wishes.update(
            use_vllm=True, vllm_mode='colocate',
            vllm_gpu_memory_utilization=args.gpu_frac,
            vllm_enable_sleep_mode=True,                 # KV cache released during the training pass
            vllm_importance_sampling_correction=False,   # r1b
            # TRL 0.24 derives vLLM max_model_len = max_prompt_length + max_completion_length, so the prompt
            # cap must be set (prompt v2 max is 7,428 tokens -> 7,500 truncates nothing).
            max_prompt_length=(args.max_prompt or 7500),
        )
    if args.unsloth_mini_batch:
        trainer_wishes['unsloth_grpo_mini_batch'] = args.unsloth_mini_batch
    if args.unsloth_logit_chunk:
        trainer_wishes['unsloth_logit_chunk_multiplier'] = args.unsloth_logit_chunk
    if not args.prerender_chat:
        trainer_wishes['chat_template_kwargs'] = {'enable_thinking': args.think == 'on'}

    model_wishes = dict(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=bool(args.load_in_4bit),
        fast_inference=bool(args.fast_inference),
    )
    if args.fast_inference:
        model_wishes['gpu_memory_utilization'] = args.gpu_frac
        model_wishes['max_lora_rank'] = args.lora_r

    if args.loader == 'vision':
        lora_wishes = dict(                     # verbatim from the published notebook
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias='none', random_state=args.seed, use_rslora=False, loftq_config=None,
            use_gradient_checkpointing=gc_val,
        )
    else:
        lora_wishes = dict(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=list(LM_TARGET_MODULES), bias='none',
            use_gradient_checkpointing=gc_val, random_state=args.seed, use_rslora=False,
            task_type='CAUSAL_LM',
        )
    return {
        'banner': BANNER,
        'model': args.model,
        'loader': args.loader,
        'loader_class': 'FastVisionModel' if args.loader == 'vision' else 'FastLanguageModel',
        'fast_inference': bool(args.fast_inference),
        'generation_backend': 'vLLM (Unsloth colocate)' if args.fast_inference else 'vLLM (TRL colocate)' if args.trl_vllm
                              else 'HF/Unsloth generate (published Qwen3.5 RL path)',
        'reward_length_constants': env_lengths(),
        'batch_math': {
            'prompts_per_step': args.prompts_per_step,
            'group_size': args.group_size,
            'completions_per_step': completions_per_step,
            'per_device_train_batch_size': args.micro_batch,
            'gradient_accumulation_steps': grad_accum,
            'check': f'{args.micro_batch} x {grad_accum} = {completions_per_step} '
                     f'= {args.prompts_per_step} prompts x {args.group_size} completions',
        },
        'lengths': {'max_seq_len': args.max_seq_len, 'max_completion': args.max_completion,
                    'max_prompt': args.max_prompt or None},
        'thinking': {'think': args.think, 'enable_thinking': args.think == 'on',
                     'route': 'prerender' if args.prerender_chat else 'chat_template_kwargs'},
        'env': {'UNSLOTH_VLLM_STANDBY': '1' if (args.standby and args.fast_inference) else None,
                'VLLM_USE_FLASHINFER_SAMPLER': '0'},
        'model_wishes': model_wishes,
        'lora_wishes': lora_wishes,
        'trainer_wishes': trainer_wishes,
    }


def length_check(cfg, args, rows) -> None:
    """Print the longest prompt in tokens and FAIL LOUDLY if the budget does not fit."""
    stats = measure_prompts(rows, args.model, use_tokenizer=not args.no_tokenize)
    budget = stats['max'] + args.max_completion
    cfg['prompt_tokens'] = stats
    cfg['seq_budget'] = {'longest_prompt_tokens': stats['max'],
                         'max_completion': args.max_completion,
                         'worst_case_total': budget, 'max_seq_len': args.max_seq_len,
                         'headroom': args.max_seq_len - budget,
                         'measured_by': stats['method']}
    print(f'[step264] prompt tokens ({stats["method"]}): n={stats["n"]} '
          f'min={stats["min"]} median={stats["median"]} p95={stats["p95"]} max={stats["max"]}')
    print(f'[step264] seq budget: longest prompt {stats["max"]} + max_completion '
          f'{args.max_completion} = {budget} vs max_seq_len {args.max_seq_len} '
          f'(headroom {args.max_seq_len - budget})')
    if budget > args.max_seq_len:
        need = budget + 256
        msg = (f'SEQUENCE BUDGET TOO SMALL: longest prompt {stats["max"]} tokens + '
               f'max_completion {args.max_completion} = {budget} > max_seq_len '
               f'{args.max_seq_len}.\n'
               f'  The longest prompts would be TRUNCATED, which silently destroys the task '
               f'(the hand data is at the end of the prompt).\n'
               f'  Fix ONE of:\n'
               f'    --max-seq-len {need}            (recommended; costs KV cache)\n'
               f'    --max-completion {args.max_seq_len - stats["max"]}   (shorter thinking '
               f'budget)\n'
               f'    --allow-truncation             (only if you have decided truncation is ok)')
        if args.allow_truncation:
            print('!' * 100)
            print('[CRITICAL] ' + msg.replace('\n', '\n[CRITICAL] '))
            print('[CRITICAL] --allow-truncation was passed, so the run CONTINUES anyway.')
            print('!' * 100)
        else:
            print('!' * 100)
            for ln in msg.split('\n'):
                print('[FATAL] ' + ln)
            print('!' * 100)
            raise SystemExit(2)


# ---------------------------------------------------------------------------- main



def install_qwen35_rope_deltas_fix(model) -> int:
    """Work around a transformers-5.2 Qwen3.5 (and Qwen3-VL style) interaction with GRPO loops.

    Qwen3_5Model.forward caches ``self.rope_deltas`` (shape (generation_batch, 1)) during generate().
    A later cache-less forward with a DIFFERENT batch size (the GRPO logprob / training pass, micro-batch
    1 or 2 after a 64-rollout generation) takes the decode branch and does
    ``delta.repeat_interleave(batch_size // delta.shape[0])`` -> 0 rows -> position_ids (3, 0, L) ->
    ``apply_rotary_pos_emb``: "Sizes of tensors must match ... Expected size 0 but got size 1".
    Observed on the r2 smoke (2026-09-18, generation batch 8, micro-batch 1), both loaders, compile on or off.
    Fix: on any forward WITHOUT a KV cache and WITHOUT explicit position_ids (never a generate() step),
    drop the stale deltas so the prefill branch recomputes position ids from input_ids + attention_mask
    via get_rope_index.  Returns the number of classes patched (0 = model has no rope_deltas, nothing done)."""
    classes = []
    for mod in model.modules():
        cls = type(mod)
        if hasattr(mod, 'get_rope_index') and hasattr(mod, 'rope_deltas') and cls not in classes:
            classes.append(cls)
    n = 0
    for cls in classes:
        if getattr(cls, '_step264_rope_fix', False):
            continue
        _orig = cls.forward

        def forward(self, *args, _orig=_orig, **kw):
            if kw.get('past_key_values') is None and kw.get('position_ids') is None                     and getattr(self, 'rope_deltas', None) is not None:
                self.rope_deltas = None
            return _orig(self, *args, **kw)
        cls.forward = forward
        cls._step264_rope_fix = True
        n += 1
    print(f'[step264] Qwen3.5 rope_deltas fix installed on {n} class(es): '
          f'{[c.__name__ for c in classes]}', flush=True)
    return n

def main(argv=None) -> None:
    print(BANNER)
    args = build_argparser().parse_args(argv)
    if args.smoke:
        apply_smoke(args)

    el = env_lengths()
    print(f'[step264] reward length constants IN FORCE: LEN_FREE={el["STEP264_LEN_FREE"]} '
          f'LEN_STEP={el["STEP264_LEN_STEP"]} LEN_RATE={el["STEP264_LEN_RATE"]} '
          f'(env: {el["raw_env"]})')
    print(f'[step264] loader={args.loader} '
          f'({"FastVisionModel" if args.loader == "vision" else "FastLanguageModel"})  '
          f'fast_inference={args.fast_inference}  '
          f'generation={"vLLM(unsloth)" if args.fast_inference else ("vLLM(TRL colocate)" if args.trl_vllm else "HF/Unsloth generate")}')

    cfg = resolve(args)
    rows = load_jsonl(args.train)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f'{args.train}: no rows')
    needed = {'prompt', 'uid', 'label', 'family', 'source_class', 'fold'}
    missing = needed - set(rows[0])
    if missing:
        raise SystemExit(f'{args.train}: rows are missing {sorted(missing)}')
    cfg['n_prompts'] = len(rows)
    cfg['train'] = os.path.abspath(args.train)
    length_check(cfg, args, rows)

    if args.dry_run:
        print('--- RESOLVED CONFIG (dry run; no model, no torch) ---')
        print(json.dumps(cfg, indent=1, default=str))
        print('--- END RESOLVED CONFIG ---')
        print('[dry-run] OK: data loaded, length check passed, config built.')
        return

    # ------------------------------------------------------------------ heavy imports
    if args.fast_inference and args.standby:
        # must be set BEFORE `import unsloth`
        os.environ['UNSLOTH_VLLM_STANDBY'] = '1'
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')   # r1: FlashInfer sampler on sm_120
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    import unsloth                                                            # noqa: F401
    loader = None
    want = 'FastVisionModel' if args.loader == 'vision' else 'FastLanguageModel'
    for name in (want, 'FastModel', 'FastLanguageModel', 'FastVisionModel'):
        loader = getattr(unsloth, name, None)
        if loader is not None:
            if name != want:
                print(f'[warn] unsloth has no {want}; using {name} instead.')
            print(f'[step264] loader class: unsloth.{name}')
            cfg['loader_class'] = name
            break
    if loader is None:
        raise SystemExit('unsloth exposes none of FastVisionModel/FastLanguageModel/FastModel')

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    run_id = args.run_id or f'step264_r2_{int(time.time())}'
    traces = args.traces or os.path.join(args.out, 'traces', 'train.jsonl.gz')
    dropped: dict = {}

    print(f'[step264] {len(rows)} training prompts from {args.train}')
    print(f'[step264] batch math: {cfg["batch_math"]["check"]}')

    mkw = filtered(loader.from_pretrained, cfg['model_wishes'], 'from_pretrained', dropped)
    print(f'[step264] loading {args.model} ...')
    OPTIONAL = ('max_lora_rank', 'gpu_memory_utilization', 'load_in_4bit')
    model = tok = None
    while True:
        try:
            model, tok = loader.from_pretrained(**mkw)
            break
        except TypeError as e:
            droppable = [k for k in OPTIONAL if k in mkw]
            if not droppable:
                raise
            print(f'[warn] from_pretrained rejected something ({e}); '
                  f'retrying without {droppable}.')
            for k in droppable:
                mkw.pop(k, None)

    tk = inner_tokenizer(tok)
    try:
        if tk.pad_token is None:
            tk.pad_token = tk.eos_token
        tk.padding_side = 'left'
    except Exception as e:
        print(f'[warn] could not set pad_token/padding_side ({e}); continuing.')

    get_peft = getattr(loader, 'get_peft_model', None)
    if get_peft is None:
        raise SystemExit('this Unsloth version has no get_peft_model on the loader class')
    pkw = filtered(get_peft, {'model': model, **cfg['lora_wishes']}, 'get_peft_model', dropped)
    pkw.pop('model', None)
    model = get_peft(model, **pkw)
    try:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f'[step264] trainable params {tr:,} / {tot:,} ({100 * tr / max(tot, 1):.4f}%)')
        cfg['trainable_params'] = {'trainable': tr, 'total': tot}
    except Exception as e:
        print(f'[warn] could not count trainable params ({e}).')
    if not args.no_rope_fix:
        cfg['rope_deltas_fix_classes'] = install_qwen35_rope_deltas_fix(model)

    # ---- thinking route -------------------------------------------------------------
    gcfg_wishes = dict(cfg['trainer_wishes'])
    accepts = set()
    try:
        accepts = set(inspect.signature(GRPOConfig).parameters)
    except (TypeError, ValueError):
        pass
    prerender = args.prerender_chat
    if not prerender and accepts and 'chat_template_kwargs' not in accepts:
        print('!' * 100)
        print('[CRITICAL] this TRL has no GRPOConfig.chat_template_kwargs, so enable_thinking '
              'cannot reach generation that way.')
        print('[CRITICAL] falling back to --prerender-chat: the chat template is rendered here '
              'with enable_thinking=%s and fed as plain text.' % (args.think == 'on'))
        print('!' * 100)
        prerender = True
        gcfg_wishes.pop('chat_template_kwargs', None)
    cfg['thinking']['route'] = 'prerender' if prerender else 'chat_template_kwargs'
    print(f'[step264] thinking route ACTIVE: {cfg["thinking"]["route"]} '
          f'(enable_thinking={args.think == "on"})')

    # generation params must reach GenerationConfig even without vLLM
    gen_fallback = {k: gcfg_wishes[k] for k in ('temperature', 'top_p', 'top_k')
                    if k in gcfg_wishes}
    gcfg_kw = filtered(GRPOConfig, gcfg_wishes, 'GRPOConfig', dropped)
    lost_gen = {k: v for k, v in gen_fallback.items() if k not in gcfg_kw}
    if lost_gen and 'generation_kwargs' in accepts:
        print(f'[step264] {sorted(lost_gen)} are not GRPOConfig fields in this TRL; '
              f'passing them through generation_kwargs instead.')
        merged = dict(gcfg_kw.get('generation_kwargs') or {})
        merged.update(lost_gen)
        gcfg_kw['generation_kwargs'] = merged
    elif lost_gen:
        print(f'[warn] {sorted(lost_gen)} could not be passed to generation at all.')
    gcfg = GRPOConfig(**gcfg_kw)

    # ---- dataset --------------------------------------------------------------------
    enable_thinking = args.think == 'on'

    def render(p: str):
        try:
            return tok.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': p}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        except Exception:
            return tok.apply_chat_template(
                [{'role': 'user', 'content': p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

    def to_example(r):
        if prerender:
            p = render(r['prompt'])                       # plain text, standard (non-chat) mode
        elif args.loader == 'vision':
            # published notebook form: content is a LIST OF TYPED PARTS (no image part here)
            p = [{'role': 'user', 'content': [{'type': 'text', 'text': r['prompt']}]}]
        else:
            p = [{'role': 'user', 'content': r['prompt']}]
        return {'prompt': p, 'uid': r['uid'], 'label': int(r['label']),
                'family': r['family'], 'source_class': r['source_class'], 'fold': int(r['fold'])}

    ds = Dataset.from_list([to_example(r) for r in rows])
    if prerender:
        ex = ds[0]['prompt']
        print(f'[step264] prerendered prompt[0]: {len(ex)} chars, '
              f'{count_tokens(tk, ex)} tokens; <think> present in template: '
              f'{"<think>" in ex}')

    state = {'step': 0, 'ckpt': 'step_00000'}

    class StepCB(TrainerCallback):
        def on_step_begin(self, a, s, c, **kw):
            state['step'] = int(s.global_step)
            state['ckpt'] = f'step_{int(s.global_step):05d}'

        def on_log(self, a, s, c, logs=None, **kw):
            seen = state.get('seen', 0)
            if seen:
                print(f'[step264] rollouts {seen}  '
                      f'format_ok {state.get("format_ok", 0) / seen:.3f}  '
                      f'clipped {state.get("clipped", 0) / seen:.3f}  '
                      f'mean_think_tok {state.get("think_tok", 0) / seen:.0f}')

    with TraceLogger(traces, run_id=run_id) as logger:
        fn = make_reward_fn(tok, logger, state, args)
        tkw = filtered(GRPOTrainer, dict(
            model=model, reward_funcs=[fn], args=gcfg, train_dataset=ds,
            processing_class=tok), 'GRPOTrainer', dropped)
        if 'processing_class' not in tkw:          # older TRL called it tokenizer
            tkw['tokenizer'] = tok
        trainer = GRPOTrainer(**tkw)
        trainer.add_callback(StepCB())
        cfg.update(run_id=run_id, traces=traces, args=vars(args), dropped_keys=dropped,
                   resolved_grpo_config=sorted(gcfg_kw))
        with open(os.path.join(args.out, 'run_config.json'), 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, indent=1, default=str)
        trainer.train()
        out_adapter = os.path.join(args.out, 'adapter_final')
        try:
            model.save_lora(out_adapter)           # Unsloth's own LoRA saver
        except Exception as e:
            print(f'[warn] model.save_lora failed ({e}); falling back to trainer.save_model')
            trainer.save_model(out_adapter)

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
