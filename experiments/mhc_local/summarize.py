"""Produce a complete local report from the frozen evaluation, without selection."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'plan.json').read_text())
result = json.loads((HERE / 'fold0/evaluation.json').read_text())
assert result['plan_sha256'] == hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest()
names = {'residual': 'Residual', 'residual_budget': 'Residual, matched budget',
         'mhc': 'mHC', 'single_pass': 'Single-Pass mHC'}
summary = []
for variant in names:
    rows = [r for r in result['seed_metrics'] if r['variant'] == variant]
    receipts = [r for r in result['receipts'] if r['variant'] == variant]
    assert len(rows) == 6 and len(receipts) == 3
    summary.append({'variant': variant, 'parameters': plan['parameters'][variant],
                    'mean_log_loss': statistics.mean(r['log_loss'] for r in rows),
                    'mean_pair_ap': statistics.mean(r['pair_ap'] for r in rows),
                    'mean_accuracy': statistics.mean(r['accuracy'] for r in rows),
                    'mean_training_seconds': statistics.mean(r['training_seconds'] for r in receipts),
                    'max_gpu_allocated_bytes': max(r['peak_gpu_allocated_bytes'] for r in receipts)})
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2))
primary = result['comparisons']['single_pass_vs_residual_budget']
lines = ['# Local mHC results', '',
         '**Twelve fits completed: four variants, three fixed seeds, 20 epochs each.**', '',
         'All values below average the same full and shortened histories of 457 held pairs. '
         'Those histories are correlated, and the player pools have been reused in earlier development. '
         'This is an exploratory comparison, not an untouched benchmark or a Kaggle-score prediction.', '',
         '| Variant | Parameters | Held log loss ↓ | Pair AP ↑ | Accuracy | Mean fit time |',
         '|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f'| {names[r["variant"]]} | {r["parameters"]:,} | {r["mean_log_loss"]:.6f} | {r["mean_pair_ap"]:.6f} | {r["mean_accuracy"]:.4%} | {r["mean_training_seconds"]/60:.2f} min |')
lines += ['', '## Predeclared decisions', '',
          f'Single-Pass versus the parameter-matched residual baseline: mean log-loss improvement '
          f'**{primary["mean_logloss_improvement"]:+.6f}**, mean Pair AP delta '
          f'**{primary["mean_pair_ap_delta"]:+.6f}**. '
          f'The frozen architecture criterion **{"passed" if result["primary_architecture_test_passed"] else "did not pass"}** '
          '(required loss improvement ≥ .005 and Pair AP delta ≥ −.002).', '',
          '| Fixed 50/50 blend with incumbent | Mean weighted pair/behavior change | Worst change | Mean sparse AP change | Screen passed |',
          '|---|---:|---:|---:|---|']
for variant, d in result['kaggle_screen'].items():
    lines.append(f'| {names[variant]} | {d["mean_pair_behavior_delta"]:+.6f} | {d["worst_pair_behavior_delta"]:+.6f} | {d["mean_sparse_delta"]:+.6f} | {d["screen_passed"]} |')
mhc_pass = any(result['kaggle_screen'][v]['screen_passed'] for v in ['mhc', 'single_pass'])
lines += ['', ('A passing mHC screen requires the additional omitted-family checks in the frozen plan before any further admission decision.'
               if mhc_pass else 'Neither mHC blend passed the incumbent screen. The frozen stopping rule therefore ends this experiment without omitted-family fits or a new submission.'), '',
          '## Checks and limits', '',
          '- Training never loaded held labels. All twelve checkpoints reload exactly and have hash-verified predictions.',
          '- All original production submission hashes remain unchanged.',
          '- GPU and CPU predictions were compared on synthetic data before fitting; architecture tests cover causality, mixer orientation, conservation, delayed pre-mixing, decoder memory reuse, gradients, and toy learning.',
          '- There are only four layers and about 277,000 parameters. Results do not establish behavior at language-model scale, and the feature-token adapter may limit performance.',
          '- No best seed, epoch, blending weight, depth, or learning rate was chosen from these results. Ordinary residual and original-mHC outcomes are retained even when they outperform the primary candidate.',
          '- Runtime is for this unfused local PyTorch implementation. It cannot validate the published production-kernel speedup.', '',
          'See [architecture and reproduction](README.md), [frozen plan](plan.json), and [complete metrics and per-seed comparisons](fold0/evaluation.json).', '']
(HERE / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

# Standard plotting is optional for environment portability; numerical report is complete above.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('Numerical report complete; matplotlib unavailable in this interpreter.')
else:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ['#667085', '#334155', '#0891b2', '#7c3aed']
    for ax, metric, title in zip(axes, ['log_loss', 'pair_ap'], ['Held log loss · lower is better', 'Pair average precision · higher is better']):
        values = []
        for i, (variant, color) in enumerate(zip(names, colors)):
            vals = [statistics.mean(r[metric] for r in result['seed_metrics'] if r['variant'] == variant and r['seed'] == seed) for seed in plan['seeds']]
            values.extend(vals)
            ax.scatter([i-.09, i, i+.09], vals, color=color, s=38, zorder=3)
            ax.plot([i-.2, i+.2], [statistics.mean(vals)]*2, color=color, linewidth=3)
        if metric == 'pair_ap':
            base = statistics.mean(r['pair_ap'] for r in result['metrics'] if r['candidate'] == 'baseline')
            values.append(base)
            ax.axhline(base, color='#b45309', linestyle='--', linewidth=1.3, label='Incumbent')
            ax.legend(frameon=False, loc='lower left')
        span = max(values)-min(values)
        margin=max(.0003 if metric=='pair_ap' else .005, span*.15)
        ax.set_ylim(min(values)-margin, min(1.0001,max(values)+margin) if metric=='pair_ap' else max(values)+margin)
        ax.set_xticks(range(4), ['Residual', 'Matched\nbudget', 'mHC', 'Single-Pass\nmHC'])
        ax.set_title(title, loc='left', pad=12, fontweight='bold')
        ax.grid(axis='y', color='#e2e8f0')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Tiny poker model · three fixed seeds per architecture', fontsize=15, fontweight='bold')
    fig.supxlabel('Dots: seeds, each averaged over two history views. Bars: means. Reused development pools; no confidence intervals.', fontsize=9)
    fig.savefig(HERE / 'results.png', dpi=180)
    plt.close(fig)
print(json.dumps({'architecture_passed': result['primary_architecture_test_passed'], 'mhc_incumbent_screen_passed': mhc_pass, 'summary': summary}, indent=2))
