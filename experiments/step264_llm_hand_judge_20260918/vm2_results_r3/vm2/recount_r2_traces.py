import gzip, json, os, sys, statistics as st
os.environ.setdefault('HF_HOME', '/workspace/.hf_home')
os.environ['STEP264_LEN_FREE'] = '10000'
os.environ['STEP264_LEN_STEP'] = '1000'
os.environ['STEP264_LEN_RATE'] = '-0.1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, '/workspace/step264/vm2')
import reward as RW
print('LEN_FREE=%s LEN_STEP=%s LEN_RATE=%s' % (RW.LEN_FREE, RW.LEN_STEP, RW.LEN_RATE))

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')

path = '/workspace/step264/runs/r2/traces/train_step1_copy.jsonl.gz'
rows = []
truncated = False
try:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    truncated = True          # last line cut mid-write
except EOFError:
    truncated = True                          # snapshot taken while the file was being written
print('rows=%d truncated_tail=%s' % (len(rows), truncated))
if rows:
    print('keys=%s' % sorted(rows[0].keys()))

old_think, new_think, new_ans, pens, fmt_old, fmt_new = [], [], [], [], 0, 0
open_tag = 0
for r in rows:
    c = r.get('completion') or ''
    if '<think>' in c.lower():
        open_tag += 1
    think, answer = RW.split_think(c)
    nt = len(tk(think, add_special_tokens=False)['input_ids']) if think else 0
    na = len(tk(answer, add_special_tokens=False)['input_ids']) if answer else 0
    rw = RW.reward(c, r.get('label', 0), r.get('family'), nt)
    old_think.append(r.get('n_think_tokens') or 0)
    new_think.append(nt); new_ans.append(na); pens.append(rw['length_penalty'])
    fmt_old += 1 if (r.get('reward') or {}).get('format_ok') else 0
    fmt_new += 1 if rw['format_ok'] else 0


def d(v, name):
    v = sorted(v)
    print('%-14s n=%d min=%d p50=%d p95=%d max=%d' %
          (name, len(v), v[0], v[len(v)//2], v[int(len(v)*.95)-1], v[-1]))


d(old_think, 'OLD n_think')
d(new_think, 'NEW n_think')
d(new_ans, 'NEW n_answer')
over = [t for t in new_think if t > 10000]
fired = [p for p in pens if p != 0.0]
print('rollouts with completions containing an OPENING <think>: %d' % open_tag)
print('rollouts with NEW n_think > 10000: %d' % len(over), over)
print('rollouts with non-zero length_penalty: %d' % len(fired),
      sorted(set(round(p, 4) for p in fired)))
print('penalty fires exactly on >10000: %s' % (len(over) == len(fired)))
print('format_ok old=%d new=%d of %d' % (fmt_old, fmt_new, len(rows)))
mism = [(t, p) for t, p in zip(new_think, pens) if (t > 10000) != (p != 0.0)]
print('mismatches: %d' % len(mism))
for t, p in [(t, p) for t, p in zip(new_think, pens) if p != 0.0][:5]:
    print('  think=%d penalty=%.4f (expected %.4f)' % (t, p, -0.1 * ((t - 10000) / 1000.0)))
