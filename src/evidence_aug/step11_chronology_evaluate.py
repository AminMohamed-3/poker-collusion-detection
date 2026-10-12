"""Step 11: single evaluation of the frozen chronology predictions (reads held truth for folds 0-2).

Outputs: results.json, pairs.csv, RESULTS.md under Claude Version/experiments/chronology_20260910/
"""
import os
import sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap, half
import numpy as np
import polars as pl

OUT = CV / 'experiments' / 'chronology_20260910'
VARIANTS = ['chrono3', 'chrono1']


def main():
    frozen = read_json(OUT / 'predictions-frozen.json')
    assert frozen['freeze_sha256'] == sha(OUT / 'freeze.json')
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest, name
    assert not (OUT / 'results.json').exists(), 'evaluation already recorded; preserve it'
    freeze = read_json(OUT / 'freeze.json')
    protected = check_protected()
    labels = dict(pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    results = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'folds': {}}
    rows = []
    pooled = {v: {} for v in VARIANTS}; pooled_table = {}
    for fold in freeze['confirmation_folds']:
        base = pl.read_parquet(OUT / 'scores' / f'fold{fold}-baseline.parquet')
        table = dict(base.select('pair_id', 'table_id').unique().iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        assert set(truths) == set(table)
        predicted = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {'baseline': per_pair_map(base, truths)}
        for v in VARIANTS:
            maps[v] = per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-{v}.parquet'), truths)
        report = {'held_target_pairs': len(table), 'candidates': {}}
        for name, m in maps.items():
            delta = {p: m[p] - maps['baseline'][p] for p in m}
            rows += [{'fold': fold, 'candidate': name, 'pair_id': p, 'table_id': table[p], 'family': labels[p], 'predicted': predicted[p], 'map5': m[p]} for p in m]
            report['candidates'][name] = {'map5': float(np.mean(list(m.values()))), 'by_true_family': {f: float(np.mean([m[p] for p in m if labels[p] == f])) for f in FAMILIES},
                                          'delta': float(np.mean(list(delta.values()))), 'delta_by_true_family': {f: float(np.mean([delta[p] for p in delta if labels[p] == f])) for f in FAMILIES},
                                          'delta_by_half': {str(hh): float(np.mean([delta[p] for p in delta if half(table[p]) == hh])) for hh in [0, 1]},
                                          'pairs_improved_worsened': [int(sum(d > 1e-12 for d in delta.values())), int(sum(d < -1e-12 for d in delta.values()))],
                                          'pool_bootstrap_95': pool_bootstrap(delta, table) if name != 'baseline' else None}
            if name in pooled:
                pooled[name].update({f'{fold}:{p}': d for p, d in delta.items()}); pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in delta})
        results['folds'][str(fold)] = report
    g = freeze['gate']; gates = {}
    for v in VARIANTS:
        per_fold = {str(f): results['folds'][str(f)]['candidates'][v] for f in freeze['confirmation_folds']}
        halves = {str(hh): float(np.mean([d for k, d in pooled[v].items() if half(pooled_table[k].split(':')[1]) == hh])) for hh in [0, 1]}
        ci = pool_bootstrap(pooled[v], pooled_table)
        checks = {'overall_gain_each_fold': all(c['delta'] >= g['per_fold_overall_gain_min'] for c in per_fold.values()),
                  'no_true_family_loss_each_fold': all(min(c['delta_by_true_family'].values()) >= -g['per_fold_true_family_loss_max'] for c in per_fold.values()),
                  'pooled_ci_excludes_zero': ci[0] > 0, 'pooled_halves_nonnegative': min(halves.values()) >= 0,
                  'pooled_delta': float(np.mean(list(pooled[v].values()))), 'pooled_pool_bootstrap_95': ci, 'pooled_halves': halves}
        checks['passed'] = all(x for x in checks.values() if isinstance(x, bool))
        gates[v] = checks
    results['gates'] = gates
    results['limits'] = ['All three folds are reused partitions from earlier studies (pool overlaps 27/23/29), not untouched data; public generalisation is unknown.',
                         'Fixed predicted families are near-perfect on these folds.', 'Full histories only; the shortened-history submission path was not scored.']
    pl.DataFrame(rows).write_csv(OUT / 'pairs.csv')
    write_json(OUT / 'results.json', results)
    lines = ['# Chronology as an evidence-ranker input: confirmation results (Claude Version)', '', f"Evaluated {results['evaluated_at']}. Primary variant: {freeze['primary_variant']}. Folds 0, 1, 2.", '',
             '| Fold | Variant | MAP@5 | Δ vs baseline | 95% pool CI | Δ transfer | Δ soft | Δ isolation | Δ half0 | Δ half1 | pairs +/− |', '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|']
    for fold in freeze['confirmation_folds']:
        for name, c in results['folds'][str(fold)]['candidates'].items():
            ci = c['pool_bootstrap_95']; d = c['delta_by_true_family']; pw = c['pairs_improved_worsened']
            lines.append(f"| {fold} | {name} | {c['map5']:.5f} | {c['delta']:+.5f} | {'' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {d['directed_transfer']:+.4f} | {d['soft_play']:+.4f} | {d['coordinated_isolation']:+.4f} | {c['delta_by_half']['0']:+.4f} | {c['delta_by_half']['1']:+.4f} | {pw[0]}/{pw[1]} |")
    lines += ['', '| Variant | pooled Δ (3 folds) | 95% pool CI | halves | gate |', '|---|---:|---|---|---|']
    for v, c in gates.items():
        lines.append(f"| {v} | {c['pooled_delta']:+.5f} | [{c['pooled_pool_bootstrap_95'][0]:+.4f}, {c['pooled_pool_bootstrap_95'][1]:+.4f}] | {c['pooled_halves']['0']:+.4f} / {c['pooled_halves']['1']:+.4f} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines += ['', 'Limits: ' + ' '.join(results['limits'])]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
