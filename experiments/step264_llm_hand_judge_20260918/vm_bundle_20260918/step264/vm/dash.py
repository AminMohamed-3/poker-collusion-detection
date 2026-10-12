"""STEP264 vm/dash.py -- tiny live dashboard for the GRPO run (standard library only).

Every --every seconds it parses /workspace/train_r1.log (TRL step dicts) and the trace file, then writes
/workspace/dash/index.html (+ state.json).  Serve it with:
    python -m http.server 8080 --bind 127.0.0.1 --directory /workspace/dash
and open http://localhost:8080 through the SSH tunnel (-L 8080:localhost:8080).
Evaluation JSONs dropped into /workspace/dash/eval/ are listed at the bottom.
"""
from __future__ import annotations

import collections
import glob
import gzip
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time

LOG = os.environ.get('DASH_LOG', '/workspace/train_r1.log')
RUN = os.environ.get('DASH_RUN', '/workspace/runs/r1')
OUT = os.environ.get('DASH_OUT', '/workspace/dash')
TARGETS = (60, 100)
_PAIR = re.compile(r"'([^']+)': '([^']*)'")


def parse_log():
    steps = []
    try:
        txt = open(LOG, encoding='utf-8', errors='replace').read()
    except FileNotFoundError:
        return steps, None
    for m in re.finditer(r"\{'loss'.*?\}", txt):
        d = dict(_PAIR.findall(m.group(0)))
        if 'reward' not in d:
            continue
        row = {}
        for k in ('loss', 'grad_norm', 'reward', 'reward_std', 'step_time', 'epoch', 'entropy',
                  'completions/mean_length', 'completions/max_length', 'completions/clipped_ratio',
                  'frac_reward_zero_std', 'num_tokens', 'learning_rate'):
            try:
                row[k] = float(d.get(k, 'nan'))
            except ValueError:
                row[k] = float('nan')
        steps.append(row)
    err = None
    for line in txt.splitlines()[-400:]:
        if (re.match(r'^\S*(Traceback|OutOfMemoryError|.*Error:)', line)
                and not any(t in line for t in ('TRL currently', 'WARNING', 'UserWarning', 'import_utils'))):
            err = line[:300]
    return steps, err


def parse_traces():
    path = os.path.join(RUN, 'traces', 'train.jsonl.gz')
    by = collections.defaultdict(list)
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except Exception:
                    break
                by[int(r.get('step') or 0)].append(r)
    except (EOFError, OSError):
        pass
    out = []
    for s in sorted(by):
        g = by[s]
        n = len(g)
        okg = [r for r in g if r.get('format_ok')]
        cons = sum(((r['verdict'] == 'NONE') == (r['confidence'] < 50)) for r in okg) / max(len(okg), 1)
        pos = [r['confidence'] for r in okg if r['label'] == 1]
        neg = [r['confidence'] for r in okg if r['label'] == 0]
        out.append({'step': s + 1, 'n': n, 'format_ok': len(okg) / n,
                    'think': sum(r['n_think_tokens'] for r in g) / n,
                    'consistency': cons,
                    'brier': sum(r['brier'] for r in g) / n,
                    'family_bonus': sum(r['family_bonus'] for r in g) / n,
                    'length_penalty': sum(r['length_penalty'] for r in g) / n,
                    'total': sum(r['total'] for r in g) / n,
                    'conf_pos': (sum(pos) / len(pos)) if pos else None, 'n_pos': len(pos),
                    'conf_neg': (sum(neg) / len(neg)) if neg else None, 'n_neg': len(neg),
                    'clipped': sum(r['finish_reason'] == 'length' for r in g) / n,
                    'verdicts': dict(collections.Counter(r['verdict'] for r in okg))})
    return out


def gpu():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        a = [x.strip() for x in o.strip().split(',')]
        return {'mem_used_gb': round(int(a[0]) / 1024, 1), 'mem_total_gb': round(int(a[1]) / 1024, 1), 'util': int(a[2])}
    except Exception:
        return {}


def svg_line(series, w=560, h=150, ylab='', colors=('#2a7fff', '#e8590c', '#2f9e44')):
    """series: list of (name, [(x, y), ...])"""
    pts_all = [p for _, s in series for p in s if p[1] is not None and p[1] == p[1]]
    if len(pts_all) < 2:
        return '<div class="empty">no data yet</div>'
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    pad = 34

    def X(x):
        return pad + (x - x0) / max(x1 - x0, 1) * (w - pad - 10)

    def Y(y):
        return 10 + (y1 - y) / (y1 - y0) * (h - 30)

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for yy in (y0, (y0 + y1) / 2, y1):
        parts.append(f'<line x1="{pad}" y1="{Y(yy):.1f}" x2="{w - 10}" y2="{Y(yy):.1f}" class="grid"/>'
                     f'<text x="{pad - 4}" y="{Y(yy) + 4:.1f}" class="lab" text-anchor="end">{yy:.3g}</text>')
    parts.append(f'<text x="{pad}" y="{h - 4}" class="lab">step {x0:g}</text>'
                 f'<text x="{w - 10}" y="{h - 4}" class="lab" text-anchor="end">step {x1:g}</text>')
    leg = []
    for (name, s), c in zip(series, colors):
        s = [p for p in s if p[1] is not None and p[1] == p[1]]
        if not s:
            continue
        d = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in s)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        leg.append(f'<span style="color:{c}">&#9632; {html.escape(name)}</span>')
    parts.append('</svg><div class="legend">' + ' &nbsp; '.join(leg) + '</div>')
    return ''.join(parts)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return '-'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def build():
    steps, err = parse_log()
    tr = parse_traces()
    cfg = {}
    try:
        cfg = json.load(open(os.path.join(RUN, 'run_config.json'), encoding='utf-8'))
    except Exception:
        pass
    ck = sorted(glob.glob(os.path.join(RUN, 'checkpoint-*')), key=lambda p: int(p.rsplit('-', 1)[-1]))
    keep = sorted(glob.glob(os.path.join(RUN, 'keep', 'checkpoint-*')), key=lambda p: int(p.rsplit('-', 1)[-1]))
    du = shutil.disk_usage('/workspace')
    g = gpu()
    alive = subprocess.run(['pgrep', '-fc', 'train_grpo'], capture_output=True, text=True).stdout.strip() not in ('', '0')
    n = len(steps)
    last = steps[-1] if steps else {}
    recent = [s['step_time'] for s in steps[-5:] if s.get('step_time') == s.get('step_time')]
    st = sorted(recent)[len(recent) // 2] if recent else None
    eta = {}
    if st:
        for t in TARGETS:
            if t > n:
                eta[t] = time.strftime('%H:%M UTC', time.gmtime(time.time() + (t - n) * st))
    max_steps = (cfg.get('args') or {}).get('max_steps')
    now = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())

    def series(key, src, scale=1.0):
        return [(r['step'] if 'step' in r else i + 1, (r[key] * scale if r.get(key) is not None else None))
                for i, r in enumerate(src)]

    rows_log = ''.join(
        f"<tr><td>{i + 1}</td><td>{fmt(s['reward'])}</td><td>{fmt(s['reward_std'])}</td>"
        f"<td>{fmt(s['completions/mean_length'], 0)}</td><td>{fmt(s['completions/clipped_ratio'], 2)}</td>"
        f"<td>{fmt(s['grad_norm'], 4)}</td><td>{fmt(s['step_time'], 0)}</td></tr>"
        for i, s in list(enumerate(steps))[-15:][::-1])
    rows_tr = ''.join(
        f"<tr><td>{t['step']}</td><td>{t['n']}</td><td>{fmt(t['format_ok'], 2)}</td><td>{fmt(t['think'], 0)}</td>"
        f"<td>{fmt(t['consistency'], 2)}</td><td>{fmt(t['brier'])}</td><td>{fmt(t['family_bonus'])}</td>"
        f"<td>{fmt(t['length_penalty'], 2)}</td><td>{fmt(t['total'])}</td>"
        f"<td>{fmt(t['conf_pos'], 0)} ({t['n_pos']})</td><td>{fmt(t['conf_neg'], 0)} ({t['n_neg']})</td>"
        f"<td>{html.escape(json.dumps(t['verdicts']))}</td></tr>"
        for t in tr[-15:][::-1])
    evals = ''
    for p in sorted(glob.glob(os.path.join(OUT, 'eval', '*.json'))):
        try:
            j = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        evals += f'<details><summary>{html.escape(os.path.basename(p))}</summary><pre>{html.escape(json.dumps(j, indent=1)[:6000])}</pre></details>'
    status = ('<span class="ok">RUNNING</span>' if alive else '<span class="bad">NOT RUNNING</span>')
    if err:
        status += f' <span class="bad">last error: {html.escape(err)}</span>'
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>STEP264 GRPO</title>
<style>
:root{{--bg:#fff;--fg:#1b1b1b;--mut:#666;--line:#e3e3e3;--card:#f7f7f8}}
@media(prefers-color-scheme:dark){{:root{{--bg:#121316;--fg:#e8e8e8;--mut:#9a9a9a;--line:#2a2b2f;--card:#1b1c20}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1200px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:18px 0 6px}} .mut{{color:var(--mut)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.big{{font-size:26px;font-weight:600}} .ok{{color:#2f9e44;font-weight:600}} .bad{{color:#e03131;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} .chart{{width:100%;height:auto}} .grid line.grid{{stroke:var(--line)}}
.lab{{font-size:10px;fill:var(--mut)}} .legend{{font-size:12px}} .empty{{color:var(--mut);padding:20px}}
pre{{white-space:pre-wrap;font-size:12px}} .wrap{{overflow-x:auto}}
</style></head><body>
<h1>STEP264 GRPO hand judge &mdash; run {html.escape(str(cfg.get('run_id', '?')))}</h1>
<div class="mut">updated {now} &middot; refreshes every 60 s &middot; {status}</div>
<div class="grid" style="margin-top:12px">
<div class="card"><div class="mut">optimizer step</div><div class="big">{n}{(' / ' + str(max_steps)) if max_steps else ''}</div>
<div class="mut">median step time {fmt(st, 0)} s &middot; ETA {' &middot; '.join(f'step {k}: {v}' for k, v in eta.items()) or '-'}</div></div>
<div class="card"><div class="mut">reward (last step)</div><div class="big">{fmt(last.get('reward'))}</div>
<div class="mut">std {fmt(last.get('reward_std'))} &middot; grad norm {fmt(last.get('grad_norm'), 4)}</div></div>
<div class="card"><div class="mut">mean completion tokens (last step)</div><div class="big">{fmt(last.get('completions/mean_length'), 0)}</div>
<div class="mut">clipped share {fmt(last.get('completions/clipped_ratio'), 2)} &middot; max {fmt(last.get('completions/max_length'), 0)}</div></div>
<div class="card"><div class="mut">machine</div><div class="big">{g.get('mem_used_gb', '-')} / {g.get('mem_total_gb', '-')} GB</div>
<div class="mut">GPU util {g.get('util', '-')}% &middot; disk free {du.free / 2**30:.1f} GB &middot; checkpoints {', '.join(os.path.basename(c).split('-')[1] for c in ck) or 'none'} &middot; kept {len(keep)}</div></div>
</div>
<h2>Reward per step (mean, from the trainer)</h2>{svg_line([('reward', series('reward', steps)), ('reward std', series('reward_std', steps))])}
<div class="grid">
<div><h2>Completion length (tokens)</h2>{svg_line([('mean', series('completions/mean_length', steps)), ('max', series('completions/max_length', steps))])}</div>
<div><h2>Reward components (from traces)</h2>{svg_line([('Brier', [(t['step'], t['brier']) for t in tr]), ('consistency', [(t['step'], t['consistency']) for t in tr]), ('length penalty', [(t['step'], t['length_penalty']) for t in tr])])}</div>
<div><h2>Mean confidence by label (traces)</h2>{svg_line([('positives', [(t['step'], t['conf_pos']) for t in tr]), ('negatives', [(t['step'], t['conf_neg']) for t in tr])])}</div>
<div><h2>Grad norm</h2>{svg_line([('grad norm', series('grad_norm', steps))])}</div>
</div>
<h2>Last steps (trainer log)</h2><div class="wrap"><table><tr><th>step</th><th>reward</th><th>std</th><th>mean len</th><th>clipped</th><th>grad norm</th><th>step s</th></tr>{rows_log}</table></div>
<h2>Last steps (reward decomposition from traces)</h2><div class="wrap"><table><tr><th>step</th><th>n</th><th>format</th><th>think</th><th>consist.</th><th>Brier</th><th>family</th><th>len pen</th><th>total</th><th>conf pos (n)</th><th>conf neg (n)</th><th>verdicts</th></tr>{rows_tr}</table></div>
<h2>Evaluations</h2>{evals or '<div class="mut">none yet (gate reads appear here)</div>'}
<h2>Run config</h2><details><summary>args</summary><pre>{html.escape(json.dumps(cfg.get('args', {}), indent=1))}</pre></details>
</body></html>"""
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, 'index.html.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(page)
    os.replace(tmp, os.path.join(OUT, 'index.html'))
    with open(os.path.join(OUT, 'state.json'), 'w', encoding='utf-8') as f:
        json.dump({'updated': now, 'steps': steps, 'traces': tr, 'eta': eta, 'gpu': g, 'alive': alive,
                   'disk_free_gb': round(du.free / 2**30, 1), 'error': err}, f)
    return n


if __name__ == '__main__':
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    while True:
        try:
            n = build()
        except Exception as e:  # keep the loop alive
            print('dash error', repr(e), flush=True)
        if every <= 0:
            break
        time.sleep(every)
