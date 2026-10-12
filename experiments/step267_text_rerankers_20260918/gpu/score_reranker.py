"""STEP267: score a text reranker over a document file, in the STEP264 harness row format.

NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU.  --dry-run exercises argument parsing, data
loading, query/instruction loading and template construction WITHOUT importing torch or
transformers, and is the only path that has been executed (on a laptop with neither installed).

Output rows (exactly what src/llm_judge/step264_fusion_check.py reads):

    {"uid": ..., "k": 1, "mean_conf": 100 * p_primary,
     "p": {query_key: p, ...}, "logit": {query_key: raw logit or logit difference, ...}}

mean_conf is the PRIMARY query's probability times 100.  --primary agnostic uses the agnostic
query; --primary max3 uses the maximum over the three family queries (the "max-over-family" Z arm
of registration267.json).  The other queries stay in the row so a later read can use them without
re-running the model.

    python score_reranker.py --model Qwen/Qwen3-Reranker-0.6B --data data/heldout_scope57_rr.jsonl
        --queries queries.json --instruction instruction.txt --primary agnostic
        --out runs/Z_qwen06_agnostic/scores.jsonl --batch 32 --max-len 2048
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rr_common as C                                                          # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='score a text reranker over STEP267 documents')
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', default=None, help='a LoRA adapter directory (qwen3 only)')
    ap.add_argument('--arch', default=None, choices=['qwen3', 'bge'],
                    help='override the model-name architecture detection')
    ap.add_argument('--data', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--instruction', required=True)
    ap.add_argument('--query-keys', default=','.join(C.QUERY_KEYS), dest='query_keys')
    ap.add_argument('--primary', default='agnostic',
                    help='agnostic | max3 | any single query key')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=2048, dest='max_len')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', dest='dry_run')
    a = ap.parse_args(argv)
    a.query_keys = [k.strip() for k in a.query_keys.split(',') if k.strip()]
    assert a.query_keys, '--query-keys is empty'
    if a.primary != 'max3':
        assert a.primary in a.query_keys, \
            '--primary %s is not among --query-keys %s' % (a.primary, a.query_keys)
    else:
        missing = [k for k in C.FAMILY_KEYS if k not in a.query_keys]
        assert not missing, '--primary max3 needs the three family queries; missing %s' % missing
    return a


def primary_prob(p_by_key, primary):
    if primary == 'max3':
        return max(p_by_key[k] for k in C.FAMILY_KEYS)
    return p_by_key[primary]


def summarize(rows, per_key, primary, meta):
    """per_key: {query_key: [p per row]} aligned with rows."""
    out = dict(meta)
    out['n'] = len(rows)
    out['query_keys'] = sorted(per_key)
    prim = [primary_prob({k: per_key[k][i] for k in per_key}, primary) for i in range(len(rows))]
    out['primary'] = primary
    out['primary_p'] = {'mean': round(sum(prim) / max(len(prim), 1), 5),
                        'min': round(min(prim), 5) if prim else None,
                        'max': round(max(prim), 5) if prim else None}
    out['mean_p_by_source_class'] = {k: C.group_stats(rows, per_key[k], 'source_class')
                                     for k in per_key}
    if C.has_labels(rows):
        y = [int(r['label']) for r in rows]
        out['mean_p_by_label'] = {k: C.group_stats(rows, per_key[k], 'label') for k in per_key}
        out['metrics_by_query'] = {k: C.metrics(y, per_key[k]) for k in per_key}
        out['metrics_primary'] = C.metrics(y, prim)
        out['metrics_primary_by_family'] = {}
        for f in C.FAMILY_KEYS:
            sel = [i for i, r in enumerate(rows)
                   if int(r['label']) == 0 or C.family_of(r) == f]
            if sel and any(int(rows[i]['label']) for i in sel):
                out['metrics_primary_by_family'][f] = C.metrics(
                    [int(rows[i]['label']) for i in sel], [prim[i] for i in sel])
    return out


# ------------------------------------------------------------------------------ the dry run
def dry_run(a):
    rows = C.load_data(a.data, a.limit)
    queries = C.load_queries(a.queries, a.query_keys)
    instruction = C.load_instruction(a.instruction)
    arch = C.detect_arch(a.model, a.arch)
    print('DRY RUN - no torch, no transformers, no model call')
    print('  model %s  arch %s  adapter %s' % (a.model, arch, a.adapter))
    print('  data %s: %d rows (labels present: %s)' % (a.data, len(rows), C.has_labels(rows)))
    print('  queries %s  primary %s  max_len %d  batch %d'
          % (a.query_keys, a.primary, a.max_len, a.batch))
    print('  instruction (%d chars): %s' % (len(instruction), instruction))
    doc = rows[0]['doc']
    lens = [len(r['doc']) for r in rows]
    print('  doc chars: min %d median %d max %d  est_tokens(max) %.0f'
          % (min(lens), sorted(lens)[len(lens) // 2], max(lens), max(lens) / 3.6))
    for k in a.query_keys:
        if arch == 'qwen3':
            txt = C.qwen_full_text(instruction, queries[k], doc)
        else:
            txt = C.bge_query_text(instruction, queries[k]) + '\n[SEP]\n' + doc
        print('  --- template[%s] %d chars, first 400 ---' % (k, len(txt)))
        print(txt[:400].replace('\n', '\\n'))
    print('  --- template tail (the suffix must survive truncation) ---')
    if arch == 'qwen3':
        print(repr(C.QWEN_SUFFIX))
    print('  would write %s and %s' % (a.out, os.path.splitext(a.out)[0] + '_summary.json'))
    fake = {k: [0.5] * len(rows) for k in a.query_keys}
    s = summarize(rows, fake, a.primary, {'dry_run': True, 'model': a.model, 'arch': arch})
    print('  summary keys: %s' % sorted(s))
    return 0


# --------------------------------------------------------------------------------- the model
def load_model(a, arch):
    import torch
    from transformers import AutoTokenizer

    side = 'left' if arch == 'qwen3' else 'right'
    tok = AutoTokenizer.from_pretrained(a.model, padding_side=side, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if arch == 'qwen3':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
            model = model.merge_and_unload()
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
            model = model.merge_and_unload()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return tok, model


def score_rows(rows, queries, instruction, a, arch, tok, model):
    """{query_key: [p]}, {query_key: [logit]}, truncated share, tokens processed."""
    import torch

    per_p = {k: [0.0] * len(rows) for k in queries}
    per_l = {k: [0.0] * len(rows) for k in queries}
    ntrunc = 0
    ntok = 0
    if arch == 'qwen3':
        fmt = C.QwenFormatter(tok, a.max_len)
    else:
        fmt = C.BgeFormatter(tok, a.max_len)
    pad_id = tok.pad_token_id
    dev = next(model.parameters()).device

    for qk, qtext in queries.items():
        if arch == 'qwen3':
            head = fmt.head_ids(instruction, qtext)
            built = []
            for i, r in enumerate(rows):
                ids, cut = fmt.build(head, r['doc'])
                ntrunc += int(cut)
                built.append((i, ids))
        else:
            qtxt = C.bge_query_text(instruction, qtext)
            built = []
            for i, r in enumerate(rows):
                doc, cut = fmt.pretruncate(qtxt, r['doc'])
                ntrunc += int(cut)
                built.append((i, doc))
        # sorted-by-length batching: fewest pad tokens per batch
        built.sort(key=lambda t: len(t[1]))
        for b0 in range(0, len(built), a.batch):
            chunk = built[b0:b0 + a.batch]
            with torch.no_grad():
                if arch == 'qwen3':
                    width = max(len(ids) for _, ids in chunk)
                    inp = torch.full((len(chunk), width), pad_id, dtype=torch.long)
                    att = torch.zeros((len(chunk), width), dtype=torch.long)
                    for j, (_i, ids) in enumerate(chunk):          # LEFT padding
                        inp[j, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
                        att[j, width - len(ids):] = 1
                    ntok += int(att.sum())
                    p, lg = C.qwen_probs(model, torch, inp.to(dev), att.to(dev),
                                         fmt.yes_id, fmt.no_id)
                else:
                    enc = tok([qtxt] * len(chunk), [d for _i, d in chunk], padding=True,
                              truncation=True, max_length=a.max_len, return_tensors='pt')
                    ntok += int(enc['attention_mask'].sum())
                    p, lg = C.bge_probs(model, torch, enc['input_ids'].to(dev),
                                        enc['attention_mask'].to(dev))
            p = p.detach().float().cpu().tolist()
            lg = lg.detach().float().cpu().tolist()
            for (i, _), pp, ll in zip(chunk, p, lg):
                per_p[qk][i] = float(pp)
                per_l[qk][i] = float(ll)
        print('  scored query %s' % qk, flush=True)
    return per_p, per_l, ntrunc / max(len(rows) * len(queries), 1), ntok


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        return dry_run(a)

    t0 = time.time()
    rows = C.load_data(a.data, a.limit)
    queries = C.load_queries(a.queries, a.query_keys)
    instruction = C.load_instruction(a.instruction)
    arch = C.detect_arch(a.model, a.arch)
    C.seed_everything(267)
    tok, model = load_model(a, arch)
    print('loaded %s (%s) in %.0fs; %d rows x %d queries'
          % (a.model, arch, time.time() - t0, len(rows), len(queries)), flush=True)

    t1 = time.time()
    per_p, per_l, trunc, ntok = score_rows(rows, queries, instruction, a, arch, tok, model)
    wall = time.time() - t1

    out = []
    for i, r in enumerate(rows):
        pk = {k: round(per_p[k][i], 6) for k in queries}
        out.append({'uid': r['uid'], 'k': 1,
                    'mean_conf': round(100.0 * primary_prob(pk, a.primary), 4),
                    'p': pk, 'logit': {k: round(per_l[k][i], 5) for k in queries}})
    C.write_jsonl(a.out, out)
    meta = {'model': a.model, 'adapter': a.adapter, 'arch': arch, 'data': a.data,
            'max_len': a.max_len, 'batch': a.batch, 'limit': a.limit,
            'wall_seconds': round(wall, 1),
            'tokens_processed': int(ntok),
            'tokens_per_second': round(ntok / max(wall, 1e-9), 1),
            'max_length_truncation_share': round(trunc, 6),
            'instruction': instruction, 'queries': queries}
    s = summarize(rows, per_p, a.primary, meta)
    sp = os.path.splitext(a.out)[0] + '_summary.json'
    json.dump(s, open(sp, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: s[k] for k in s if k not in ('queries', 'mean_p_by_source_class')},
                     indent=1), flush=True)
    print('wrote %s and %s in %.0fs' % (a.out, sp, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
