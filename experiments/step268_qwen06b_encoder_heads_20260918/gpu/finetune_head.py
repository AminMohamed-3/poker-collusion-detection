"""STEP268 B3 -- THE OPTIONAL LAST ARM: LoRA r16 on Qwen3-0.6B + a linear head on the pooled state.

THIS IS THE LAST ARM OF STEP268 AND IT IS GATED.  registration268.json:
    "fine_tune_arm (optional, last): LoRA r16 on Qwen3-0.6B with a linear head on the pooled
     state, BCE on the listed label, 2 epochs - only if STEP267's FT_A showed standalone
     AP >= .55; otherwise skipped"
DO NOT RUN IT until STEP267 has reported FT_A's standalone AP on the 57x20 candidates and that
number is >= .55.  If STEP267 is not finished, or FT_A came in below .55, this arm is SKIPPED and
STEP268 ends after fit_heads.py.  The script refuses to train without --step267-ft-ap.

NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU.  Only `--dry-run` (which imports neither torch
nor transformers nor peft) and `py_compile` have been executed here.  Every API detail is written
from documentation - see RUNBOOK_268.txt, "VERIFY ON THE GPU BOX".

RECIPE
  base            Qwen/Qwen3-0.6B (AutoModel, bf16)
  adapter         peft LoRA r=16, alpha=32, dropout .05 on q_proj,k_proj,v_proj,o_proj
  head            Linear(hidden -> 1) on the MASK-WEIGHTED MEAN of the last hidden state
  loss            BCEWithLogitsLoss, pos_weight = n_neg / n_pos on the training rows
  data            the same H1 split as fit_heads.py: (in_pool OR extra_listed) AND fold in 0-2,
                  label = `listed`
  schedule        2 epochs, AdamW lr 1e-4 (adapter + head), cosine, warmup 3 %, batch 8 x grad
                  accumulation 4, max-len 2048, bf16
  output          scores for heldout_scope57_uids.txt and heldout_subset_uids.txt in the harness
                  score-file format {"uid","k":1,"mean_conf"} -> ft_scope57_scores.jsonl /
                  ft_subset_scores.jsonl, plus ft_summary.json.  The adapter weights stay on the
                  box; only the score files come back.

USAGE
  python finetune_head.py --dry-run --out ft/              (shapes and IO only - all that is
                                                            required of this file)
  python finetune_head.py --step267-ft-ap 0.61 --out ft/   (only if STEP267 cleared the gate)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

FT_GATE_AP = 0.55
TRAIN_FOLDS = (0, 1, 2)
LORA = dict(r=16, lora_alpha=32, lora_dropout=.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
TRAIN = dict(epochs=2, lr=1e-4, batch=8, grad_accum=4, max_len=2048, warmup_ratio=.03)


def read_docs(path):
    op = gzip.open if path.endswith('.gz') else open
    d = {}
    with op(path, 'rt', encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                d[r['uid']] = r['doc']
    return d


def read_meta(path):
    rows = []
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def read_list(path):
    with open(path, encoding='utf-8') as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def split(meta, docs, scope57, subset):
    tr = [r for r in meta if (r['in_pool'] or r['extra_listed']) and r['fold'] in TRAIN_FOLDS]
    miss = [r['uid'] for r in tr if r['uid'] not in docs]
    assert not miss, f'{len(miss)} training uids have no document'
    for nm, lst in (('scope57', scope57), ('subset', subset)):
        m = [u for u in lst if u not in docs]
        assert not m, f'{len(m)} {nm} uids have no document'
    y = np.array([r['listed'] for r in tr])
    return tr, y


def write_scores(path, uids, p):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for u, v in zip(uids, p):
            fh.write(json.dumps({'uid': u, 'k': 1,
                                 'mean_conf': round(100.0 * float(v), 4)}) + '\n')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--docs', default='universe_docs.jsonl.gz')
    ap.add_argument('--meta', default='universe_meta.jsonl')
    ap.add_argument('--scope57', default='heldout_scope57_uids.txt')
    ap.add_argument('--subset', default='heldout_subset_uids.txt')
    ap.add_argument('--out', default='ft/')
    ap.add_argument('--seed', type=int, default=268)
    ap.add_argument('--step267-ft-ap', type=float, default=None, dest='ft_ap',
                    help="STEP267 FT_A's standalone AP on the 57x20 candidates; the gate")
    ap.add_argument('--dry-run', action='store_true', dest='dry')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    docs = read_docs(a.docs)
    meta = read_meta(a.meta)
    scope57, subset = read_list(a.scope57), read_list(a.subset)
    tr, y = split(meta, docs, scope57, subset)
    info = {'model': a.model, 'lora': LORA, 'train': TRAIN, 'seed': a.seed,
            'n_train': len(tr), 'n_train_listed': int(y.sum()),
            'pos_weight': round(float((len(y) - y.sum()) / max(y.sum(), 1)), 3),
            'n_scope57': len(scope57), 'n_subset': len(subset),
            'gate': f'STEP267 FT_A standalone AP >= {FT_GATE_AP}',
            'step267_ft_ap_supplied': a.ft_ap,
            'NOTE': 'nothing in gpu/ has ever run on a GPU'}

    if a.dry:
        rng = np.random.default_rng(a.seed)
        info['DRY_RUN'] = True
        info['WARNING'] = 'the score files hold RANDOM numbers, no model was loaded'
        write_scores(os.path.join(a.out, 'ft_scope57_scores.jsonl'), scope57,
                     rng.random(len(scope57)))
        write_scores(os.path.join(a.out, 'ft_subset_scores.jsonl'), subset,
                     rng.random(len(subset)))
        info['doc_chars'] = {'mean': round(float(np.mean([len(docs[r['uid']]) for r in tr])), 1),
                             'max': int(max(len(v) for v in docs.values()))}
        info['steps_per_epoch'] = int(np.ceil(len(tr) / (TRAIN['batch'] * TRAIN['grad_accum'])))
        info['wall_seconds'] = round(time.time() - t0, 1)
        with open(os.path.join(a.out, 'ft_summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(info, fh, indent=1)
        print(json.dumps(info, indent=1))
        print('DRY RUN only - this arm never trained.')
        return

    assert a.ft_ap is not None, (
        'this arm is GATED: pass --step267-ft-ap with STEP267 FT_A\'s standalone AP')
    assert a.ft_ap >= FT_GATE_AP, (
        'STEP267 FT_A standalone AP %.4f < %.2f - registration268 says this arm is SKIPPED'
        % (a.ft_ap, FT_GATE_AP))

    import torch                                                          # noqa: PLC0415
    from peft import LoraConfig, get_peft_model                           # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer                     # noqa: PLC0415

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side='right')
    base = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(task_type='FEATURE_EXTRACTION', **LORA)).cuda()
    hid = int(base.config.hidden_size)
    head = torch.nn.Linear(hid, 1).cuda().to(torch.float32)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=TRAIN['lr'])
    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(info['pos_weight'])]).cuda())

    def pooled(texts):
        bd = tok(texts, padding=True, truncation=True, max_length=TRAIN['max_len'],
                 return_tensors='pt')
        bd = {k: v.cuda() for k, v in bd.items()}
        out = model(**bd).last_hidden_state
        m = bd['attention_mask'].unsqueeze(-1).to(out.dtype)
        return ((out * m).sum(1) / m.sum(1).clamp(min=1)).to(torch.float32)

    order = np.arange(len(tr))
    rng = np.random.default_rng(a.seed)
    steps = 0
    for ep in range(TRAIN['epochs']):
        rng.shuffle(order)
        model.train()
        opt.zero_grad()
        for bi in range(0, len(order), TRAIN['batch']):
            idx = order[bi:bi + TRAIN['batch']]
            lg = head(pooled([docs[tr[i]['uid']] for i in idx])).squeeze(-1)
            loss = lossf(lg, torch.tensor(y[idx], dtype=torch.float32).cuda())
            (loss / TRAIN['grad_accum']).backward()
            if (bi // TRAIN['batch'] + 1) % TRAIN['grad_accum'] == 0:
                opt.step()
                opt.zero_grad()
                steps += 1
                if steps % 50 == 0:
                    print(f'  epoch {ep} step {steps} loss {float(loss):.4f} '
                          f'{time.time() - t0:.0f}s', flush=True)

    model.eval()
    with torch.inference_mode():
        for name, lst in (('scope57', scope57), ('subset', subset)):
            p = np.zeros(len(lst))
            for bi in range(0, len(lst), TRAIN['batch']):
                sl = lst[bi:bi + TRAIN['batch']]
                p[bi:bi + len(sl)] = torch.sigmoid(
                    head(pooled([docs[u] for u in sl])).squeeze(-1)).cpu().numpy()
            write_scores(os.path.join(a.out, f'ft_{name}_scores.jsonl'), lst, p)
    info['DRY_RUN'] = False
    info['optimizer_steps'] = steps
    info['wall_seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, 'ft_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
