"""STEP264 dash_artifact.py -- render a snapshot page (Artifact HTML fragment) from the VM dashboard's state.json.

  python dash_artifact.py <state.json> <eval_dir> <out.html> [note]

The page is a plain HTML fragment (title + style + body content) following the Artifact page contract:
tokens on :root, dark-mode blocks, phone-width layout, inline SVG charts, no external scripts.
"""
from __future__ import annotations

import glob
import html
import json
import os
import sys
import time

C = {'a': 'var(--s1)', 'b': 'var(--s2)', 'c': 'var(--s3)'}


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return '–'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def chart(series, w=600, h=170, cols=('var(--s1)', 'var(--s2)', 'var(--s3)')):
    pts_all = [p for _, s in series for p in s if p[1] is not None and p[1] == p[1]]
    if len(pts_all) < 2:
        return '<div class="empty">Not enough steps yet</div>'
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 24

    def X(x):
        return pad_l + (x - x0) / max(x1 - x0, 1) * (w - pad_l - pad_r)

    def Y(y):
        return pad_t + (y1 - y) / (y1 - y0) * (h - pad_t - pad_b)

    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for yy in (y0, (y0 + y1) / 2, y1):
        out.append(f'<line x1="{pad_l}" y1="{Y(yy):.1f}" x2="{w - pad_r}" y2="{Y(yy):.1f}" class="gl"/>'
                   f'<text x="{pad_l - 6}" y="{Y(yy) + 4:.1f}" class="tl" text-anchor="end">{yy:.3g}</text>')
    out.append(f'<text x="{pad_l}" y="{h - 6}" class="tl">step {x0:g}</text>'
               f'<text x="{w - pad_r}" y="{h - 6}" class="tl" text-anchor="end">step {x1:g}</text>')
    leg = []
    for (name, s), c in zip(series, cols):
        s = [p for p in s if p[1] is not None and p[1] == p[1]]
        if not s:
            continue
        d = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in s)
        out.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        lx, ly = s[-1]
        out.append(f'<circle cx="{X(lx):.1f}" cy="{Y(ly):.1f}" r="3.5" fill="{c}"/>')
        leg.append(f'<span class="lg"><i style="background:{c}"></i>{html.escape(name)}</span>')
    out.append('</svg><div class="legend">' + ''.join(leg) + '</div>')
    return ''.join(out)


def main():
    state_p, eval_dir, out_p = sys.argv[1:4]
    note = sys.argv[4] if len(sys.argv) > 4 else ''
    st = json.load(open(state_p, encoding='utf-8'))
    steps, tr = st['steps'], st['traces']
    n = len(steps)
    last = steps[-1] if steps else {}
    eta = st.get('eta') or {}
    g = st.get('gpu') or {}
    alive = st.get('alive')
    status = '<span class="pill ok">Running</span>' if alive else '<span class="pill bad">Not running</span>'
    err = st.get('error')

    def ser(key, scale=1.0):
        return [(i + 1, (s.get(key) * scale if s.get(key) is not None else None)) for i, s in enumerate(steps)]

    rows_log = ''.join(
        f"<tr><td>{i + 1}</td><td>{fmt(s.get('reward'))}</td><td>{fmt(s.get('reward_std'))}</td>"
        f"<td>{fmt(s.get('completions/mean_length'), 0)}</td><td>{fmt(s.get('completions/clipped_ratio'), 2)}</td>"
        f"<td>{fmt(s.get('grad_norm'), 4)}</td><td>{fmt(s.get('step_time'), 0)}</td></tr>"
        for i, s in list(enumerate(steps))[-12:][::-1])
    rows_tr = ''.join(
        f"<tr><td>{t['step']}</td><td>{fmt(t['think'], 0)}</td><td>{fmt(t['consistency'], 2)}</td>"
        f"<td>{fmt(t['brier'])}</td><td>{fmt(t['family_bonus'])}</td><td>{fmt(t['length_penalty'], 2)}</td>"
        f"<td>{fmt(t['total'])}</td><td>{fmt(t['conf_pos'], 0)} <small>({t['n_pos']})</small></td>"
        f"<td>{fmt(t['conf_neg'], 0)} <small>({t['n_neg']})</small></td></tr>"
        for t in tr[-12:][::-1])
    evals = ''
    for p in sorted(glob.glob(os.path.join(eval_dir, '*.json'))):
        try:
            j = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        evals += (f'<details><summary>{html.escape(os.path.basename(p))}</summary>'
                  f'<pre>{html.escape(json.dumps(j, indent=1)[:5000])}</pre></details>')
    snap = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
    page = f"""<title>Hand Judge Training</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f5f6f8;--card:#ffffff;--ink:#1c2330;--mut:#66707f;--line:#dfe3e9;--acc:#1f7a4d;--s1:#1f7a4d;--s2:#c7791b;--s3:#4a6fa5;--ok:#1f7a4d;--bad:#b83a2c;--okbg:#e4f3ea;--badbg:#f8e4e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#111519;--card:#181d24;--ink:#e6e9ee;--mut:#98a2b3;--line:#2a313b;--acc:#43b57c;--s1:#43b57c;--s2:#e0a04a;--s3:#7f9fd0;--ok:#43b57c;--bad:#e2695a;--okbg:#173225;--badbg:#3a1f1b}}}}
:root[data-theme="dark"]{{--bg:#111519;--card:#181d24;--ink:#e6e9ee;--mut:#98a2b3;--line:#2a313b;--acc:#43b57c;--s1:#43b57c;--s2:#e0a04a;--s3:#7f9fd0;--ok:#43b57c;--bad:#e2695a;--okbg:#173225;--badbg:#3a1f1b}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif;margin:0}}
.wrap{{max-width:1080px;margin:0 auto;padding-block:20px 40px;padding-inline:16px}}
h1{{font-size:22px;font-weight:600;margin:0;text-wrap:balance}} h2{{font-size:13px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);margin:26px 0 8px}}
.sub{{color:var(--mut);font-size:13px;margin-top:4px;display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center}}
.pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}} .ok{{background:var(--okbg);color:var(--ok)}} .bad{{background:var(--badbg);color:var(--bad)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.tile .k{{font-size:12px;color:var(--mut)}} .tile .v{{font:500 28px/1.2 "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;margin:2px 0}} .tile .d{{font-size:12px;color:var(--mut)}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px 8px}} .panel h3{{font-size:13px;font-weight:600;margin:0 0 6px}}
.chart{{width:100%;height:auto;display:block}} .gl{{stroke:var(--line)}} .tl{{font:10px "IBM Plex Mono",monospace;fill:var(--mut)}}
.legend{{font-size:12px;color:var(--mut);display:flex;gap:12px;flex-wrap:wrap;margin-top:4px}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.empty{{color:var(--mut);padding:24px 0;text-align:center;font-size:13px}}
.tbl{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}} table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
td,th{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--mut);font-weight:500;font-size:12px}} tr:last-child td{{border-bottom:0}} td:first-child,th:first-child{{text-align:left}} small{{color:var(--mut)}}
details{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin-bottom:8px}} summary{{cursor:pointer;font-size:13px;font-weight:500}} pre{{white-space:pre-wrap;font:12px/1.4 "IBM Plex Mono",monospace;margin:8px 0 0;overflow-x:auto}}
.note{{background:var(--card);border-left:3px solid var(--acc);border-radius:6px;padding:8px 12px;font-size:13px;margin-top:14px}}
.err{{color:var(--bad);font-size:12px;word-break:break-word}}
</style>
<div class="wrap">
<h1>Hand judge GRPO — Qwen3-8B on the poker collusion hands</h1>
<div class="sub">{status}<span>snapshot {snap}</span><span>VM data {html.escape(st.get('updated', ''))}</span><span>run {html.escape(str((st.get('run_id') or 'r1f')))}</span></div>
{('<div class="err">Last error: ' + html.escape(err[:220]) + '</div>') if err and not alive else ''}
{('<div class="note">' + html.escape(note) + '</div>') if note else ''}
<div class="tiles">
<div class="tile"><div class="k">Optimizer step</div><div class="v">{n} / 100</div><div class="d">{' · '.join(f'step {k} ≈ {v}' for k, v in eta.items()) or 'ETA once steps accumulate'}</div></div>
<div class="tile"><div class="k">Reward, last step</div><div class="v">{fmt(last.get('reward'), 2)}</div><div class="d">spread {fmt(last.get('reward_std'), 2)} · grad norm {fmt(last.get('grad_norm'), 3)}</div></div>
<div class="tile"><div class="k">Mean completion, tokens</div><div class="v">{fmt(last.get('completions/mean_length'), 0)}</div><div class="d">clipped {fmt(last.get('completions/clipped_ratio'), 2)} · longest {fmt(last.get('completions/max_length'), 0)}</div></div>
<div class="tile"><div class="k">GPU memory</div><div class="v">{g.get('mem_used_gb', '–')} <small style="font-size:14px">/ {g.get('mem_total_gb', '–')} GB</small></div><div class="d">util {g.get('util', '–')}% · disk free {st.get('disk_free_gb', '–')} GB</div></div>
</div>
<h2>Trajectory</h2>
<div class="charts">
<div class="panel"><h3>Reward per step</h3>{chart([('mean reward', ser('reward')), ('spread (std)', ser('reward_std'))])}</div>
<div class="panel"><h3>Completion length, tokens</h3>{chart([('mean', ser('completions/mean_length')), ('longest', ser('completions/max_length'))])}</div>
<div class="panel"><h3>Reward components (from traces)</h3>{chart([('Brier', [(t['step'], t['brier']) for t in tr]), ('consistency', [(t['step'], t['consistency']) for t in tr]), ('length penalty', [(t['step'], t['length_penalty']) for t in tr])])}</div>
<div class="panel"><h3>Mean confidence by true label</h3>{chart([('collusion hands', [(t['step'], t['conf_pos']) for t in tr]), ('ordinary hands', [(t['step'], t['conf_neg']) for t in tr])])}</div>
</div>
<h2>Last steps, trainer</h2>
<div class="tbl"><table><thead><tr><th>step</th><th>reward</th><th>spread</th><th>mean len</th><th>clipped</th><th>grad norm</th><th>step s</th></tr></thead><tbody>{rows_log or '<tr><td colspan="7" class="empty">no steps yet</td></tr>'}</tbody></table></div>
<h2>Last steps, reward decomposition</h2>
<div class="tbl"><table><thead><tr><th>step</th><th>think tok</th><th>consistency</th><th>Brier</th><th>family</th><th>len pen</th><th>total</th><th>conf on collusion (n)</th><th>conf on ordinary (n)</th></tr></thead><tbody>{rows_tr or '<tr><td colspan="9" class="empty">no traces yet</td></tr>'}</tbody></table></div>
<h2>Evaluations</h2>
{evals or '<div class="empty">Gate reads appear here</div>'}
<h2>What the numbers mean</h2>
<div class="panel" style="font-size:13px;line-height:1.55">
<p style="margin:0 0 6px"><b>Reward</b> per rollout = Brier score (1 − (confidence/100 − label)²) + .15 if the verdict names the right family − .2 if the verdict and confidence disagree (NONE should come with confidence &lt; 50) − .1 per 1,000 thinking tokens beyond 6,000; 0 if the answer lines are missing.</p>
<p style="margin:0 0 6px"><b>Consistency</b> is the share of rollouts whose verdict and confidence agree. <b>Mean confidence by label</b> is the discrimination watch: the gap between collusion and ordinary hands should open if the judge learns anything.</p>
<p style="margin:0"><b>Gate 2</b> (read on the 57 audit queries after a checkpoint): the judge fused as a third factor must add ≥ +.005 MAP@5 over the production recipe with bootstrap P ≥ .75, against a noise floor of −.017. Registered prediction: fail.</p>
</div>
</div>
"""
    with open(out_p, 'w', encoding='utf-8') as f:
        f.write(page)
    print('wrote', out_p, len(page), 'bytes; steps', n, 'traces', len(tr))


if __name__ == '__main__':
    main()
