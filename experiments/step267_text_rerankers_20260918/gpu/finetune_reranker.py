"""STEP267: fine-tune a text reranker on the STEP267 documents (FT_A / FT_B / FT_C arms).

NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU.  --dry-run exercises argument parsing, data
loading, group construction and template construction WITHOUT importing torch, transformers or
peft, and is the only path that has been executed.

qwen3  LoRA (peft) on q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj of the
       causal LM; the trainable score is (logit_yes - logit_no) at the LAST position; BCE with
       logits against the row label.
bge    full fine-tune of the AutoModelForSequenceClassification head+body (or LoRA when
       --lora-r > 0); BCE with logits on the single output logit.

--loss group_ce  each step samples GROUPS of 1 listed hand + --neg-per-pos unlisted hands OF THE
       SAME PAIR (FlagEmbedding-style listwise training); a pair with too few unlisted hands is
       topped up from the clean-pair negatives.  Loss = softmax cross-entropy over the group with
       the listed hand as the target.
--query-mode family  a positive-pair row uses its own family query; a clean-pair row uses a
       uniformly drawn family query (seeded, so the draw is reproducible).

After every epoch the --eval file is scored with the SAME code path as score_reranker.py
(imported, not copied) and DIR/eval_epochN_scores.jsonl + an AUC/AP summary are written.  The
adapter (qwen3) or the model (bge) is saved under DIR/final at the end.

    python finetune_reranker.py --model Qwen/Qwen3-Reranker-0.6B --arch qwen3
        --train data/train_rr.jsonl --eval data/heldout_scope57_rr.jsonl
        --queries queries.json --instruction instruction.txt --query-mode agnostic
        --out runs/FT_A_s267 --epochs 2 --lr 1e-4 --lora-r 16 --lora-alpha 32
        --batch 16 --grad-accum 1 --max-len 2048 --seed 267 --loss bce
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rr_common as C                                                          # noqa: E402
import score_reranker as S                                                     # noqa: E402

QWEN_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj']


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='fine-tune a text reranker on STEP267 documents')
    ap.add_argument('--model', required=True)
    ap.add_argument('--arch', default=None, choices=['qwen3', 'bge'])
    ap.add_argument('--train', required=True)
    ap.add_argument('--eval', required=True, dest='eval_path')
    ap.add_argument('--queries', required=True)
    ap.add_argument('--instruction', required=True)
    ap.add_argument('--query-mode', default='agnostic', choices=['agnostic', 'family'],
                    dest='query_mode')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora-r', type=int, default=16, dest='lora_r')
    ap.add_argument('--lora-alpha', type=int, default=32, dest='lora_alpha')
    ap.add_argument('--lora-dropout', type=float, default=0.05, dest='lora_dropout')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--grad-accum', type=int, default=1, dest='grad_accum')
    ap.add_argument('--max-len', type=int, default=2048, dest='max_len')
    ap.add_argument('--seed', type=int, default=267)
    ap.add_argument('--loss', default='bce', choices=['bce', 'group_ce'])
    ap.add_argument('--neg-per-pos', type=int, default=7, dest='neg_per_pos')
    ap.add_argument('--grad-checkpointing', action='store_true', dest='grad_ckpt')
    ap.add_argument('--warmup-frac', type=float, default=0.05, dest='warmup_frac')
    ap.add_argument('--eval-batch', type=int, default=32, dest='eval_batch')
    ap.add_argument('--eval-primary', default='agnostic', dest='eval_primary')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', dest='dry_run')
    a = ap.parse_args(argv)
    a.arch = C.detect_arch(a.model, a.arch)
    assert a.neg_per_pos >= 1
    assert a.batch >= 1 and a.epochs >= 1
    if a.loss == 'group_ce':
        assert a.batch % (a.neg_per_pos + 1) == 0, \
            ('--batch %d must be a multiple of the group size %d for --loss group_ce'
             % (a.batch, a.neg_per_pos + 1))
    return a


# ------------------------------------------------------------------------------ data helpers
def query_for(row, queries, mode, rng):
    """The query key this row trains against."""
    if mode == 'agnostic':
        return 'agnostic'
    f = C.family_of(row)
    return f if f else rng.choice(list(C.FAMILY_KEYS))


def build_groups(rows, keys, neg_per_pos, rng):
    """[(pos_index, [neg indices])] - 1 listed hand + neg_per_pos unlisted hands OF THE SAME PAIR.

    The pair is not in the JSONL (the GPU box never sees pair_id), so the pair is reconstructed
    from source_class + family: POS_LISTED / POS_UNLISTED rows of the same family form the
    positive-pair pool.  keys, when given, is a uid -> group-id map for an exact grouping.
    Clean-pair rows (source_class NEG_*) top a short group up.
    """
    pos = [i for i, r in enumerate(rows) if int(r['label']) == 1]
    clean = [i for i, r in enumerate(rows) if str(r.get('source_class', '')).startswith('NEG')]
    by_group = {}
    for i, r in enumerate(rows):
        if int(r['label']) == 0 and not str(r.get('source_class', '')).startswith('NEG'):
            g = keys.get(r['uid']) if keys else C.family_of(r)
            by_group.setdefault(g, []).append(i)
    groups = []
    for p in pos:
        g = keys.get(rows[p]['uid']) if keys else C.family_of(rows[p])
        pool = list(by_group.get(g, []))
        rng.shuffle(pool)
        negs = pool[:neg_per_pos]
        while len(negs) < neg_per_pos and clean:
            negs.append(clean[rng.randrange(len(clean))])
        if len(negs) == neg_per_pos:
            groups.append((p, negs))
    rng.shuffle(groups)
    return groups


def group_key_map(train_path):
    """Optional exact grouping: <train>.groups.json = {uid: group_id}.  The local builder does
    NOT write one (the pair id must not leave the laptop); without it the family is the group."""
    p = os.path.splitext(train_path)[0] + '.groups.json'
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    return None


# ------------------------------------------------------------------------------- the dry run
def dry_run(a):
    rows = C.load_data(a.train, a.limit)
    ev = C.load_data(a.eval_path)
    queries = C.load_queries(a.queries, C.QUERY_KEYS)
    instruction = C.load_instruction(a.instruction)
    rng = random.Random(a.seed)
    print('DRY RUN - no torch, no transformers, no peft, no training')
    print('  model %s  arch %s  out %s' % (a.model, a.arch, a.out))
    print('  train %s: %d rows, %d labelled 1' % (a.train, len(rows),
                                                  sum(int(r['label']) for r in rows)))
    print('  eval  %s: %d rows, %d labelled 1' % (a.eval_path, len(ev),
                                                  sum(int(r['label']) for r in ev)))
    print('  epochs %d  lr %g  batch %d x accum %d  max_len %d  seed %d  loss %s'
          % (a.epochs, a.lr, a.batch, a.grad_accum, a.max_len, a.seed, a.loss))
    if a.arch == 'qwen3':
        print('  LoRA r%d alpha%d dropout%g on %s'
              % (a.lora_r, a.lora_alpha, a.lora_dropout, QWEN_TARGET_MODULES))
    else:
        print('  bge: %s' % ('LoRA r%d' % a.lora_r if a.lora_r > 0 else 'full fine-tune'))
    steps = math.ceil(len(rows) / (a.batch * a.grad_accum)) * a.epochs
    print('  optimiser steps %d (warmup %d, cosine decay)'
          % (steps, max(1, int(a.warmup_frac * steps))))
    by = {}
    for r in rows:
        by[r.get('source_class', 'NA')] = by.get(r.get('source_class', 'NA'), 0) + 1
    print('  train by source_class: %s' % json.dumps(by, sort_keys=True))
    if a.loss == 'group_ce':
        g = build_groups(rows, group_key_map(a.train), a.neg_per_pos, rng)
        print('  group_ce: %d groups of 1+%d (groups per batch %d)'
              % (len(g), a.neg_per_pos, a.batch // (a.neg_per_pos + 1)))
        if g:
            p, ns = g[0]
            print('    example group: pos uid %s (family %s) + negs %s'
                  % (rows[p]['uid'], rows[p].get('family'),
                     [rows[i]['uid'] for i in ns]))
    qk = [query_for(r, queries, a.query_mode, rng) for r in rows[:2000]]
    cnt = {k: qk.count(k) for k in sorted(set(qk))}
    print('  --query-mode %s query usage on the first %d rows: %s'
          % (a.query_mode, len(qk), json.dumps(cnt)))
    r0 = rows[0]
    k0 = query_for(r0, queries, a.query_mode, random.Random(a.seed))
    txt = (C.qwen_full_text(instruction, queries[k0], r0['doc']) if a.arch == 'qwen3'
           else C.bge_query_text(instruction, queries[k0]) + '\n[SEP]\n' + r0['doc'])
    print('  training template[%s] %d chars, first 300:' % (k0, len(txt)))
    print(txt[:300].replace('\n', '\\n'))
    print('  would write %s/train_log.jsonl, %s/eval_epoch{1..%d}_scores.jsonl, %s/final'
          % (a.out, a.out, a.epochs, a.out))
    return 0


# -------------------------------------------------------------------------------- the model
def build_model(a):
    import torch
    from transformers import AutoTokenizer

    side = 'left' if a.arch == 'qwen3' else 'right'
    tok = AutoTokenizer.from_pretrained(a.model, padding_side=side, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.arch == 'qwen3':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model, num_labels=1, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if a.arch == 'qwen3' or a.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                         bias='none',
                         task_type='CAUSAL_LM' if a.arch == 'qwen3' else 'SEQ_CLS',
                         target_modules=(QWEN_TARGET_MODULES if a.arch == 'qwen3'
                                         else ['query', 'key', 'value', 'dense']))
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return tok, model


def batch_logits(a, tok, model, fmt, rows, qkeys, queries, instruction, torch):
    """The trainable score of each row: (logit_yes - logit_no) for qwen3, the single logit for
    bge.  Keeps the graph, so this is the training forward."""
    dev = next(model.parameters()).device
    if a.arch == 'qwen3':
        built = []
        for r, qk in zip(rows, qkeys):
            head = fmt.head_ids(instruction, queries[qk])
            ids, _cut = fmt.build(head, r['doc'])
            built.append(ids)
        width = max(len(x) for x in built)
        pad_id = tok.pad_token_id
        inp = torch.full((len(built), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(built), width), dtype=torch.long)
        for j, ids in enumerate(built):                     # LEFT padding
            inp[j, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            att[j, width - len(ids):] = 1
        try:      # last position only: the full 152k-vocab logits tensor is the OOM at batch 8
            out = model(input_ids=inp.to(dev), attention_mask=att.to(dev), logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            out = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits[:, -1, :]
        return (out[:, fmt.yes_id] - out[:, fmt.no_id]).float()
    qtexts, docs = [], []
    for r, qk in zip(rows, qkeys):
        qt = C.bge_query_text(instruction, queries[qk])
        d, _cut = fmt.pretruncate(qt, r['doc'])
        qtexts.append(qt)
        docs.append(d)
    enc = tok(qtexts, docs, padding=True, truncation=True, max_length=a.max_len,
              return_tensors='pt')
    return model(input_ids=enc['input_ids'].to(dev),
                 attention_mask=enc['attention_mask'].to(dev)).logits.view(-1).float()


def run_eval(a, epoch, tok, model):
    """Score --eval with score_reranker.py's own code path and write the epoch files."""
    rows = C.load_data(a.eval_path)
    queries = C.load_queries(a.queries, list(C.QUERY_KEYS))
    instruction = C.load_instruction(a.instruction)
    args = S.parse_args(['--model', a.model, '--arch', a.arch, '--data', a.eval_path,
                         '--queries', a.queries, '--instruction', a.instruction,
                         '--primary', a.eval_primary,
                         '--out', os.path.join(a.out, 'eval_epoch%d_scores.jsonl' % epoch),
                         '--batch', str(a.eval_batch), '--max-len', str(a.max_len)])
    model.eval()
    per_p, per_l, trunc, ntok = S.score_rows(rows, queries, instruction, args, a.arch, tok, model)
    model.train()
    out = []
    for i, r in enumerate(rows):
        pk = {k: round(per_p[k][i], 6) for k in queries}
        out.append({'uid': r['uid'], 'k': 1,
                    'mean_conf': round(100.0 * S.primary_prob(pk, args.primary), 4),
                    'p': pk, 'logit': {k: round(per_l[k][i], 5) for k in queries}})
    C.write_jsonl(args.out, out)
    s = S.summarize(rows, per_p, args.primary, {'epoch': epoch, 'model': a.model,
                                                'adapter': 'in-memory', 'arch': a.arch,
                                                'max_length_truncation_share': round(trunc, 6),
                                                'tokens_processed': int(ntok)})
    json.dump(s, open(os.path.splitext(args.out)[0] + '_summary.json', 'w', encoding='utf-8'),
              indent=1)
    print('epoch %d eval: %s' % (epoch, json.dumps(s.get('metrics_primary'))), flush=True)
    return s


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        return dry_run(a)

    import torch
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    C.seed_everything(a.seed)
    rng = random.Random(a.seed)

    rows = C.load_data(a.train, a.limit)
    queries = C.load_queries(a.queries, list(C.QUERY_KEYS))
    instruction = C.load_instruction(a.instruction)
    tok, model = build_model(a)
    fmt = C.QwenFormatter(tok, a.max_len) if a.arch == 'qwen3' else C.BgeFormatter(tok, a.max_len)

    if a.loss == 'group_ce':
        groups = build_groups(rows, group_key_map(a.train), a.neg_per_pos, rng)
        gsize = a.neg_per_pos + 1
        per_batch = a.batch // gsize
        units = [groups[i:i + per_batch] for i in range(0, len(groups), per_batch)]
    else:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        units = [idx[i:i + a.batch] for i in range(0, len(idx), a.batch)]

    total_steps = math.ceil(len(units) / a.grad_accum) * a.epochs
    warm = max(1, int(a.warmup_frac * total_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        t = (step - warm) / max(1, total_steps - warm)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    log = os.path.join(a.out, 'train_log.jsonl')
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w', encoding='utf-8'), indent=1)
    model.train()
    step = 0
    for ep in range(1, a.epochs + 1):
        if a.loss == 'group_ce':
            rng.shuffle(groups)
            units = [groups[i:i + per_batch] for i in range(0, len(groups), per_batch)]
        else:
            rng.shuffle(idx)
            units = [idx[i:i + a.batch] for i in range(0, len(idx), a.batch)]
        opt.zero_grad(set_to_none=True)
        for ui, unit in enumerate(units):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if a.loss == 'group_ce':
                    flat = []
                    for p, ns in unit:
                        flat.append(p)
                        flat.extend(ns)
                    br = [rows[i] for i in flat]
                    qk = [query_for(rows[p], queries, a.query_mode, rng)
                          for p, ns in unit for _ in range(gsize)]
                    sc = batch_logits(a, tok, model, fmt, br, qk, queries, instruction, torch)
                    sc = sc.view(len(unit), gsize)
                    tgt = torch.zeros(len(unit), dtype=torch.long, device=sc.device)
                    loss = torch.nn.functional.cross_entropy(sc, tgt)
                else:
                    br = [rows[i] for i in unit]
                    qk = [query_for(r, queries, a.query_mode, rng) for r in br]
                    sc = batch_logits(a, tok, model, fmt, br, qk, queries, instruction, torch)
                    y = torch.tensor([float(r['label']) for r in br], device=sc.device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(sc, y)
            (loss / a.grad_accum).backward()
            if (ui + 1) % a.grad_accum == 0 or ui == len(units) - 1:
                for g in opt.param_groups:
                    g['lr'] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    C.append_jsonl(log, {'epoch': ep, 'step': step,
                                         'loss': round(float(loss.item()), 6),
                                         'lr': lr_at(step),
                                         'wall': round(time.time() - t0, 1)})
                    print('ep %d step %d/%d loss %.4f lr %.2e %.0fs'
                          % (ep, step, total_steps, float(loss.item()), lr_at(step),
                             time.time() - t0), flush=True)
        run_eval(a, ep, tok, model)

    fin = os.path.join(a.out, 'final')
    os.makedirs(fin, exist_ok=True)
    model.save_pretrained(fin)          # peft writes the ADAPTER only; bge full writes the model
    tok.save_pretrained(fin)
    print('done in %.0fs; saved %s' % (time.time() - t0, fin))
    return 0


if __name__ == '__main__':
    sys.exit(main())
