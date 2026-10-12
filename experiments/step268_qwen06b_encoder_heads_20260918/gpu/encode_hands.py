"""STEP268 B1 -- encode every universe document ONCE with a frozen Qwen3 0.6B encoder.

NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU.  It was written on a laptop that has no torch
and no transformers; only `--dry-run` (which never imports either) and `py_compile` have been
executed.  Every transformers / Qwen API detail below is written from documentation and is listed
again in RUNBOOK_268.txt under "VERIFY ON THE GPU BOX".

WHAT IT DOES
  reads  universe_docs.jsonl.gz   {"uid", "doc"}          (~78.6k rows, ~2.4k chars each)
  writes emb_universe.npy         float16, (N, 1024), L2-normalised, row i <-> uid i
         emb_uids.txt             one uid per line, the row order of the .npy
         <out>_summary.json       n, tokens/s, truncation share, wall, every setting

THE RECIPE (Qwen/Qwen3-Embedding-0.6B)
  * AutoModel, NOT AutoModelForCausalLM; transformers >= 4.51.
  * tokenizer is built with padding_side='left' so the LAST position of every row is a real
    token; pooling then takes last_hidden_state[:, -1].
  * the model is instruction-aware: the QUERY side is formatted
        f'Instruct: {task}\\nQuery: {text}'
    and the DOCUMENT side is the plain text with NO prefix.  In STEP268 every hand is a
    DOCUMENT, so --side document (the default) sends the plain hand block and the instruction
    file is only recorded in the summary.  --side query exists so the same script can embed the
    instruction as a query later without a second implementation.
  * embeddings are L2-normalised, so a dot product is a cosine.
  * --pooling mean is the second arm (Qwen/Qwen3-0.6B, the plain causal LM used as an encoder):
    mask-weighted mean of the last hidden state.  Qwen3-Embedding must use --pooling last.

  Batching is by SORTED LENGTH (fewest pad tokens); the output is written back in the INPUT order
  of the JSONL, and emb_uids.txt is that same order, so downstream code can join on uid alone.

USAGE (on the box)
  python encode_hands.py --model Qwen/Qwen3-Embedding-0.6B --docs universe_docs.jsonl.gz \\
      --instruction instruction.txt --out emb_universe.npy --uids emb_uids.txt \\
      --batch 64 --max-len 2048 --pooling last --bf16
  python encode_hands.py ... --dry-run            (IO + shapes only, no torch, no model)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

HID = 1024                      # Qwen3-Embedding-0.6B / Qwen3-0.6B hidden size - ASSERTED below
QUERY_TEMPLATE = 'Instruct: {task}\nQuery: {text}'


def read_docs(path: str, limit: int = 0):
    """uid list and doc list in file order.  Accepts .jsonl and .jsonl.gz."""
    op = gzip.open if path.endswith('.gz') else open
    uids, docs = [], []
    with op(path, 'rt', encoding='utf-8') as fh:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            uids.append(r['uid'])
            docs.append(r['doc'])
            if limit and len(uids) >= limit:
                break
    assert len(set(uids)) == len(uids), 'duplicate uid in the document file'
    return uids, docs


def sorted_batches(docs, batch: int):
    """Indices grouped into batches of similar length (fewest pad tokens)."""
    order = np.argsort([len(d) for d in docs], kind='stable')
    return [order[i:i + batch].tolist() for i in range(0, len(order), batch)]


def format_texts(docs, side: str, instruction: str):
    if side == 'query':
        return [QUERY_TEMPLATE.format(task=instruction, text=d) for d in docs]
    return list(docs)


# --------------------------------------------------------------------------- dry run
def dry_run(a, uids, docs):
    """Everything except the model: IO, formatting, batching, shapes, output writing."""
    texts = format_texts(docs, a.side, a.instruction_text)
    batches = sorted_batches(texts, a.batch)
    approx_tok = np.array([len(t) / 3.6 for t in texts])        # the project's cheap estimate
    n = len(texts)
    emb = np.zeros((n, HID), dtype=np.float16)                  # the shape the real run writes
    rng = np.random.default_rng(268)
    v = rng.standard_normal((n, HID)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    emb[:] = v.astype(np.float16)
    np.save(a.out, emb)
    with open(a.uids, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(uids) + '\n')
    summ = {'DRY_RUN': True,
            'WARNING': 'emb_universe.npy holds RANDOM UNIT VECTORS, not embeddings',
            'model': a.model, 'side': a.side, 'pooling': a.pooling, 'bf16': a.bf16,
            'batch': a.batch, 'max_len': a.max_len, 'n': n, 'hidden': HID,
            'n_batches': len(batches),
            'batch_size_distinct': sorted({len(b) for b in batches}),
            'est_tokens': {'mean': round(float(approx_tok.mean()), 1),
                           'p50': round(float(np.percentile(approx_tok, 50)), 1),
                           'p99': round(float(np.percentile(approx_tok, 99)), 1),
                           'max': round(float(approx_tok.max()), 1)},
            'est_share_over_max_len': round(float((approx_tok > a.max_len).mean()), 6),
            'instruction': a.instruction_text,
            'example_formatted_head': texts[0][:200] if texts else None}
    with open(a.summary, 'w', encoding='utf-8') as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))
    print('DRY RUN: wrote', a.out, a.uids, a.summary)


# --------------------------------------------------------------------------- real run
def real_run(a, uids, docs):
    import torch                                                          # noqa: PLC0415
    import torch.nn.functional as F                                       # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer                     # noqa: PLC0415

    texts = format_texts(docs, a.side, a.instruction_text)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side='left')
    dtype = torch.bfloat16 if a.bf16 else torch.float32
    model = AutoModel.from_pretrained(a.model, torch_dtype=dtype)
    model.eval().cuda()
    hid = int(model.config.hidden_size)
    assert hid == HID, f'hidden size {hid}, expected {HID} for a 0.6B Qwen3 encoder'

    n = len(texts)
    emb = np.zeros((n, hid), dtype=np.float16)
    batches = sorted_batches(texts, a.batch)
    n_tok = 0
    n_trunc = 0
    t0 = time.time()
    with torch.inference_mode():
        for bi, idx in enumerate(batches):
            bd = tok([texts[i] for i in idx], padding=True, truncation=True,
                     max_length=a.max_len, return_tensors='pt')
            lens = bd['attention_mask'].sum(1)
            n_trunc += int((lens >= a.max_len).sum())
            n_tok += int(lens.sum())
            bd = {k: v.cuda() for k, v in bd.items()}
            out = model(**bd).last_hidden_state
            if a.pooling == 'last':
                # padding_side='left' => the last position of every row is a real token
                assert int(bd['attention_mask'][:, -1].sum()) == bd['attention_mask'].shape[0], \
                    'the tokenizer did not left-pad; last-token pooling would read a pad token'
                v = out[:, -1]
            else:
                m = bd['attention_mask'].unsqueeze(-1).to(out.dtype)
                v = (out * m).sum(1) / m.sum(1).clamp(min=1)
            v = F.normalize(v.float(), p=2, dim=1)
            emb[idx] = v.cpu().numpy().astype(np.float16)
            if bi % 50 == 0 or bi == len(batches) - 1:
                el = time.time() - t0
                print(f'  batch {bi + 1}/{len(batches)}  rows {min((bi + 1) * a.batch, n)}/{n}  '
                      f'{n_tok / max(el, 1e-6):.0f} tok/s  {el:.0f}s', flush=True)

    assert np.isfinite(emb.astype(np.float32)).all(), 'non-finite embedding'
    nrm = np.linalg.norm(emb.astype(np.float32), axis=1)
    assert np.allclose(nrm, 1.0, atol=2e-2), f'rows are not unit norm (min {nrm.min()})'
    np.save(a.out, emb)
    with open(a.uids, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(uids) + '\n')
    wall = time.time() - t0
    summ = {'DRY_RUN': False, 'model': a.model, 'side': a.side, 'pooling': a.pooling,
            'bf16': a.bf16, 'batch': a.batch, 'max_len': a.max_len,
            'n': n, 'hidden': hid, 'n_batches': len(batches),
            'tokens': n_tok, 'tokens_per_second': round(n_tok / max(wall, 1e-6), 1),
            'truncated_rows': n_trunc, 'truncation_share': round(n_trunc / max(n, 1), 6),
            'wall_seconds': round(wall, 1), 'instruction': a.instruction_text,
            'norm_min': round(float(nrm.min()), 5), 'norm_max': round(float(nrm.max()), 5)}
    with open(a.summary, 'w', encoding='utf-8') as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--docs', default='universe_docs.jsonl.gz')
    ap.add_argument('--instruction', default='instruction.txt')
    ap.add_argument('--out', default='emb_universe.npy')
    ap.add_argument('--uids', default='emb_uids.txt')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max-len', type=int, default=2048, dest='max_len')
    ap.add_argument('--pooling', choices=['last', 'mean'], default='last')
    ap.add_argument('--side', choices=['document', 'query'], default='document')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', dest='dry')
    a = ap.parse_args(argv)
    a.summary = os.path.splitext(a.out)[0] + '_summary.json'
    a.instruction_text = open(a.instruction, encoding='utf-8').read().strip()
    assert a.instruction_text, 'instruction.txt is empty'
    if 'Embedding' in a.model and a.pooling != 'last':
        print('WARNING: Qwen3-Embedding is trained for LAST-token pooling; --pooling '
              f'{a.pooling} is the base-model arm', flush=True)

    t0 = time.time()
    uids, docs = read_docs(a.docs, a.limit)
    print(f'read {len(docs)} documents from {a.docs} in {time.time() - t0:.1f}s', flush=True)
    (dry_run if a.dry else real_run)(a, uids, docs)


if __name__ == '__main__':
    main()
