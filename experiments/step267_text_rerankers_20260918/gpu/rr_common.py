"""STEP267 GPU side: shared IO, prompt formatting and scoring helpers for text rerankers.

NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU.  Every API detail below was written from the
model cards / library documentation, not from a run; RUNBOOK_267.txt lists each one under
"VERIFY ON THE GPU BOX".

Two architectures are supported.

Qwen3-Reranker (any size) - a CAUSAL LM used as a yes/no classifier, official template:

    prefix = '<|im_start|>system\\nJudge whether the Document meets the requirements based on '
             'the Query and the Instruct provided. Note that the answer can only be "yes" or '
             '"no".<|im_end|>\\n<|im_start|>user\\n'
    body   = '<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}'
    suffix = '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'

    score  = softmax over the logits of the tokens "yes" and "no" at the LAST position -> P(yes).
    The tokenizer must pad on the LEFT, because the last position of a right-padded row is a pad
    token and the score would be read off padding.

BAAI/bge-reranker-v2-m3 - an XLM-R cross-encoder: the tokenizer takes the PAIR (query, document),
the model is an AutoModelForSequenceClassification with one output logit, score = sigmoid(logit).
The query text used there is the instruction and the query joined, so both architectures see the
same information.

TRUNCATION.  Only the DOCUMENT is ever cut, and it is cut on the LEFT (its first tokens are
dropped, its tail is kept), so the fixed prefix, the instruction, the query and the assistant
suffix always survive intact.  With --max-len 2048 and ~960 estimated tokens at the 100th
percentile of the built documents this should never fire; the share of truncated rows is reported
by score_reranker.py so that a silent change is visible.

Nothing here imports torch or transformers at module level: --dry-run of the scripts that use it
must run on a machine with neither.
"""
from __future__ import annotations

import json
import os
import random
import sys

QWEN_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
               'the Query and the Instruct provided. Note that the answer can only be "yes" or '
               '"no".<|im_end|>\n<|im_start|>user\n')
QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

QUERY_KEYS = ('agnostic', 'directed_transfer', 'soft_play', 'coordinated_isolation')
FAMILY_KEYS = ('directed_transfer', 'soft_play', 'coordinated_isolation')


# --------------------------------------------------------------------------------- plain IO
def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def write_jsonl(path, rows):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=True) + '\n')


def append_jsonl(path, row):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'a', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + '\n')


def load_queries(path, keys=None):
    q = json.load(open(path, encoding='utf-8'))
    keys = list(keys) if keys else list(q)
    missing = [k for k in keys if k not in q]
    assert not missing, 'queries.json has no key(s) %s (has %s)' % (missing, sorted(q))
    return {k: q[k] for k in keys}


def load_instruction(path):
    s = open(path, encoding='utf-8').read().strip()
    assert s, 'instruction.txt is empty'
    return s


def load_data(path, limit=0):
    rows = read_jsonl(path)
    assert rows, 'no rows in ' + path
    for r in rows:
        assert 'uid' in r and 'doc' in r, 'a data row lacks uid/doc: ' + str(sorted(r))
    uids = [r['uid'] for r in rows]
    assert len(set(uids)) == len(uids), 'duplicate uid in ' + path
    if limit:
        rows = rows[:limit]
    return rows


def has_labels(rows):
    return all('label' in r for r in rows)


def family_of(row):
    """the row's family query key, or None for a clean-pair / unlabelled-family row."""
    f = str(row.get('family', 'none')).strip().lower()
    return f if f in FAMILY_KEYS else None


# ------------------------------------------------------------------- architecture detection
def detect_arch(model_name, override=None):
    if override:
        assert override in ('qwen3', 'bge'), override
        return override
    m = model_name.lower()
    if 'qwen3-reranker' in m or ('qwen' in m and 'rerank' in m):
        return 'qwen3'
    if 'bge' in m:
        return 'bge'
    raise SystemExit('cannot infer the architecture from %r - pass --arch qwen3|bge' % model_name)


# ------------------------------------------------------------------------- template building
def qwen_body(instruction, query, doc):
    return '<Instruct>: %s\n<Query>: %s\n<Document>: %s' % (instruction, query, doc)


def bge_query_text(instruction, query):
    """bge takes one query string; the instruction is prepended so both architectures see it."""
    return '%s %s' % (instruction, query)


def qwen_full_text(instruction, query, doc):
    """The exact string the model sees (used by --dry-run; the real path tokenises the parts)."""
    return QWEN_PREFIX + qwen_body(instruction, query, doc) + QWEN_SUFFIX


def left_truncate_ids(ids, budget):
    """Drop tokens from the LEFT of the document, keep its tail.  Returns (ids, was_truncated)."""
    if budget <= 0:
        return [], True
    if len(ids) <= budget:
        return ids, False
    return ids[len(ids) - budget:], True


# ------------------------------------------------------------------------------ tokenisation
class QwenFormatter:
    """Builds Qwen3-Reranker input ids.  Needs a transformers tokenizer, so it is constructed
    only on the GPU path; --dry-run never instantiates it."""

    def __init__(self, tokenizer, max_len=2048):
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.prefix_ids = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
        self.yes_id = tokenizer.convert_tokens_to_ids('yes')
        self.no_id = tokenizer.convert_tokens_to_ids('no')
        assert isinstance(self.yes_id, int) and isinstance(self.no_id, int) \
            and self.yes_id >= 0 and self.no_id >= 0, \
            'the tokenizer does not map "yes"/"no" to single ids - VERIFY ON THE GPU BOX'
        assert self.yes_id != self.no_id

    def head_ids(self, instruction, query):
        """everything before the document, tokenised once per (instruction, query)."""
        head = '<Instruct>: %s\n<Query>: %s\n<Document>: ' % (instruction, query)
        return self.tok.encode(head, add_special_tokens=False)

    def build(self, head_ids, doc):
        doc_ids = self.tok.encode(doc, add_special_tokens=False)
        budget = self.max_len - len(self.prefix_ids) - len(self.suffix_ids) - len(head_ids)
        doc_ids, cut = left_truncate_ids(doc_ids, budget)
        return self.prefix_ids + head_ids + doc_ids + self.suffix_ids, cut


class BgeFormatter:
    """bge-reranker-v2-m3: tokenizer pair input, document truncated on the left."""

    def __init__(self, tokenizer, max_len=2048):
        self.tok = tokenizer
        self.max_len = int(max_len)

    def pretruncate(self, query_text, doc):
        """Cut the DOCUMENT on the left so that the pair fits; the tokenizer then never has to
        touch the query.  Returns (doc, was_truncated)."""
        q_ids = self.tok.encode(query_text, add_special_tokens=False)
        d_ids = self.tok.encode(doc, add_special_tokens=False)
        budget = self.max_len - len(q_ids) - 4          # <s> q </s></s> d </s>
        d_ids, cut = left_truncate_ids(d_ids, budget)
        if not cut:
            return doc, False
        return self.tok.decode(d_ids, skip_special_tokens=True), True


# ----------------------------------------------------------------------------------- scoring
def qwen_probs(model, torch, input_ids, attention_mask, yes_id, no_id):
    """P(yes) from the last-position logits.  Left padding is assumed (see the module docstring)."""
    try:                      # only the last position's logits: the full 152k-vocab tensor is 16 GB at batch 32
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1).logits[:, -1, :]
    except TypeError:         # older transformers without logits_to_keep
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
    two = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
    lp = torch.nn.functional.log_softmax(two, dim=1)
    return lp[:, 1].exp(), (logits[:, yes_id] - logits[:, no_id]).float()


def bge_probs(model, torch, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1).float()
    return torch.sigmoid(logits), logits


# -------------------------------------------------------------------------------- misc utils
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def metrics(y, s):
    """AP and AUC without sklearn (the GPU box may not have it); ties get the average rank."""
    n = len(y)
    if n == 0 or sum(y) == 0 or sum(y) == n:
        return {'n': n, 'n_pos': int(sum(y)), 'ap': None, 'auc': None}
    order = sorted(range(n), key=lambda i: -s[i])
    tp = 0
    ap = 0.0
    npos = int(sum(y))
    for rank, i in enumerate(order, 1):
        if y[i]:
            tp += 1
            ap += tp / rank
    ap /= npos
    # AUC via the rank-sum identity with average ranks for ties
    idx = sorted(range(n), key=lambda i: s[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[idx[j + 1]] == s[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[idx[t]] = avg
        i = j + 1
    rsum = sum(ranks[i] for i in range(n) if y[i])
    nneg = n - npos
    auc = (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)
    return {'n': n, 'n_pos': npos, 'ap': round(ap, 5), 'auc': round(auc, 5)}


def group_stats(rows, probs, key):
    """mean probability by row[key]."""
    acc = {}
    for r, p in zip(rows, probs):
        acc.setdefault(str(r.get(key, 'NA')), []).append(float(p))
    return {k: {'n': len(v), 'mean_p': round(sum(v) / len(v), 5)} for k, v in sorted(acc.items())}


def print_err(*a):
    print(*a, file=sys.stderr, flush=True)
