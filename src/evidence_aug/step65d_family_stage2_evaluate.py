"""Step 65d: single evaluation of the frozen step-65c predictions against held evidence truth (folds 0-2).
Refuses to run unless predictions-frozen.json exists and every score file still matches its hash; refuses to overwrite
results. Outputs: results.json, pairs.csv, RESULTS.md under Claude Version/experiments/family_stage2_20260911/."""
import os
import sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap, half
from evidence_aug.step65c_family_stage2_fit import OUT, CI
import numpy as np
import polars as pl

VARIANTS = ['hybridCI_top20_struct', 'hybridCI_all_struct', 'hybridCI_top20_nostruct', 'allfam_top20_struct']


def main():
    frozen = read_json(OUT / 'predictions-frozen.json')
    assert frozen['freeze_sha256'] == sha(OUT / 'freeze.json')
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest, name
    assert not (OUT / 'results.json').exists(), 'evaluation already recorded; preserve it'
    freeze = read_json(OUT / 'freeze.json'); protected = check_protected()
    labels = dict(pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    results = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'folds': {}}
    rows = []; pooled = {v: {} for v in VARIANTS}; pooled_table = {}; pooled_pred = {}
    for fold in freeze['confirmation_folds']:
        base = pl.read_parquet(OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        table = dict(base.select('pair_id', 'table_id').unique().iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        assert set(truths) == set(table)
        predicted = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {'chrono1': per_pair_map(base, truths)}
        for v in VARIANTS:
            maps[v] = per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-{v}.parquet'), truths)
        report = {'held_target_pairs': len(table), 'predicted_CI_pairs': sum(1 for p in table if predicted[p] == CI), 'true_CI_pairs': sum(1 for p in table if labels[p] == CI),
                  'predicted_CI_true_families': {f: sum(1 for p in table if predicted[p] == CI and labels[p] == f) for f in FAMILIES}, 'candidates': {}}
        for name, m in maps.items():
            delta = {p: m[p] - maps['chrono1'][p] for p in m}
            rows += [{'fold': fold, 'candidate': name, 'pair_id': p, 'table_id': table[p], 'family': labels[p], 'predicted': predicted[p], 'map5': m[p]} for p in m]
            report['candidates'][name] = {'map5': float(np.mean(list(m.values()))), 'by_true_family': {f: float(np.mean([m[p] for p in m if labels[p] == f])) for f in FAMILIES},
                                          'by_predicted_family': {f: float(np.mean([m[p] for p in m if predicted[p] == f])) for f in FAMILIES if any(predicted[p] == f for p in m)},
                                          'delta': float(np.mean(list(delta.values()))), 'delta_by_true_family': {f: float(np.mean([delta[p] for p in delta if labels[p] == f])) for f in FAMILIES},
                                          'delta_predicted_CI_pairs': float(np.mean([delta[p] for p in delta if predicted[p] == CI])) if any(predicted[p] == CI for p in delta) else None,
                                          'delta_by_half': {str(hh): float(np.mean([delta[p] for p in delta if half(table[p]) == hh])) for hh in [0, 1]},
                                          'pairs_improved_worsened': [int(sum(d > 1e-12 for d in delta.values())), int(sum(d < -1e-12 for d in delta.values()))],
                                          'pool_bootstrap_95': pool_bootstrap(delta, table) if name != 'chrono1' else None}
            if name in pooled:
                pooled[name].update({f'{fold}:{p}': d for p, d in delta.items()}); pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in delta}); pooled_pred.update({f'{fold}:{p}': (predicted[p], labels[p]) for p in delta})
        results['folds'][str(fold)] = report
    g = freeze['gate']; gates = {}
    for v in VARIANTS:
        per_fold = {str(f): results['folds'][str(f)]['candidates'][v] for f in freeze['confirmation_folds']}
        halves = {str(hh): float(np.mean([d for k, d in pooled[v].items() if half(pooled_table[k].split(':')[1]) == hh])) for hh in [0, 1]}
        ci = pool_bootstrap(pooled[v], pooled_table)
        true_ci = float(np.mean([d for k, d in pooled[v].items() if pooled_pred[k][1] == CI]))
        checks = {'overall_gain_each_fold': all(c['delta'] >= g['per_fold_overall_gain_min'] for c in per_fold.values()),
                  'no_true_family_loss_each_fold': all(min(c['delta_by_true_family'].values()) >= -g['per_fold_true_family_loss_max'] for c in per_fold.values()),
                  'pooled_true_CI_gain': true_ci >= g['pooled_true_CI_gain_min'],
                  'pooled_ci_excludes_zero': ci[0] > 0, 'pooled_halves_nonnegative': min(halves.values()) >= 0,
                  'pooled_delta': float(np.mean(list(pooled[v].values()))), 'pooled_delta_true_CI': true_ci, 'pooled_delta_by_true_family': {f: float(np.mean([d for k, d in pooled[v].items() if pooled_pred[k][1] == f])) for f in FAMILIES},
                  'pooled_pool_bootstrap_95': ci, 'pooled_halves': halves}
        checks['passed'] = all(x for x in checks.values() if isinstance(x, bool))
        gates[v] = checks
    results['gates'] = gates
    results['limits'] = ['All three folds are reused partitions from earlier studies; the hypothesis came from the audit pools (outside every held fold), so all three count.',
                         'Fixed predicted families of the folds (near-perfect); on the evaluation set the predicted family is the submission behaviour column.', 'Full histories only.']
    pl.DataFrame(rows).write_csv(OUT / 'pairs.csv')
    write_json(OUT / 'results.json', results)
    lines = ['# Per-family second-stage evidence model (structural features): confirmation results (Claude Version)', '', f"Evaluated {results['evaluated_at']}. Primary: hybridCI_top20_struct. Folds 0, 1, 2.", '',
             '| Fold | Variant | MAP@5 | Δ vs chrono1 | 95% pool CI | Δ transfer | Δ soft | Δ isolation | Δ predicted-CI pairs | Δ half0 | Δ half1 | pairs +/− |', '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for fold in freeze['confirmation_folds']:
        r = results['folds'][str(fold)]
        lines.append(f"| {fold} | (held {r['held_target_pairs']} pairs, predicted CI {r['predicted_CI_pairs']}, true CI {r['true_CI_pairs']}) | | | | | | | | | | |")
        for name, c in r['candidates'].items():
            ci = c['pool_bootstrap_95']; d = c['delta_by_true_family']; pw = c['pairs_improved_worsened']; dc = c['delta_predicted_CI_pairs']
            lines.append(f"| {fold} | {name} | {c['map5']:.4f} | {c['delta']:+.4f} | {'' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {d['directed_transfer']:+.4f} | {d['soft_play']:+.4f} | {d['coordinated_isolation']:+.4f} | {'' if dc is None else f'{dc:+.4f}'} | {c['delta_by_half']['0']:+.4f} | {c['delta_by_half']['1']:+.4f} | {pw[0]}/{pw[1]} |")
    lines += ['', '| Variant | pooled Δ (3 folds) | pooled Δ true CI | 95% pool CI | halves | gate |', '|---|---:|---:|---|---|---|']
    for v, c in gates.items():
        lines.append(f"| {v} | {c['pooled_delta']:+.4f} | {c['pooled_delta_true_CI']:+.4f} | [{c['pooled_pool_bootstrap_95'][0]:+.4f}, {c['pooled_pool_bootstrap_95'][1]:+.4f}] | {c['pooled_halves']['0']:+.4f} / {c['pooled_halves']['1']:+.4f} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines += ['', 'Limits: ' + ' '.join(results['limits'])]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
