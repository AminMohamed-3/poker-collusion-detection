"""Step 7: single evaluation of the frozen family-weighting predictions (reads held truth for folds 0-2).

Outputs: results.json, pairs.csv, RESULTS.md under Claude Version/experiments/family_weighting_20260910/
"""
import os
import sys
import json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap, half
import numpy as np
import polars as pl

OUT = CV / 'experiments' / 'family_weighting_20260910'
SHORT = {'directed_transfer': 'transfer', 'soft_play': 'soft', 'coordinated_isolation': 'isolation'}


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
    pooled = {'A_soft_x2_routed': {}, 'B_family_specialized': {}}
    pooled_table = {}
    for fold in [0, 1, 2]:
        base = pl.read_parquet(OUT / 'scores' / f'fold{fold}-baseline.parquet')
        held_pairs = base.select('pair_id', 'table_id').unique()
        table = dict(held_pairs.iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in evidence_all.join(held_pairs.select('pair_id'), on='pair_id', how='semi').group_by('pair_id')}
        assert set(truths) == set(table)
        predicted = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        variant_scores = {v: pl.read_parquet(OUT / 'scores' / f'fold{fold}-{v}.parquet') for v in ['baseline', 'soft_x2', 'transfer_x2', 'isolation_x2']}
        routed_soft = sorted(p for p in table if predicted[p] == 'soft_play')
        candidates = {'baseline': base}
        candidates['A_soft_x2_routed'] = pl.concat([variant_scores['soft_x2'].filter(pl.col('pair_id').is_in(routed_soft)), base.filter(~pl.col('pair_id').is_in(routed_soft))])
        pieces = []
        for f in FAMILIES:
            ids = sorted(p for p in table if predicted[p] == f)
            pieces.append(variant_scores[SHORT[f] + '_x2'].filter(pl.col('pair_id').is_in(ids)))
        other = sorted(p for p in table if predicted[p] not in FAMILIES)
        candidates['B_family_specialized'] = pl.concat(pieces + [base.filter(pl.col('pair_id').is_in(other))])
        for v in ['soft_x2', 'transfer_x2', 'isolation_x2']:
            candidates[v + '_unrouted'] = variant_scores[v]
        maps = {}
        for name, frame in candidates.items():
            assert frame.height == base.height, name
            maps[name] = per_pair_map(frame, truths)
            rows += [{'fold': fold, 'candidate': name, 'pair_id': p, 'table_id': table[p], 'family': labels[p], 'predicted': predicted[p], 'map5': m} for p, m in maps[name].items()]
        report = {'held_target_pairs': len(table), 'routed_soft_pairs': len(routed_soft), 'routed_true_family_counts': {f: {g: sum(1 for p in table if predicted[p] == f and labels[p] == g) for g in FAMILIES} for f in FAMILIES}, 'candidates': {}}
        for name, m in maps.items():
            delta = {p: m[p] - maps['baseline'][p] for p in m}
            report['candidates'][name] = {'map5': float(np.mean(list(m.values()))), 'by_true_family': {f: float(np.mean([m[p] for p in m if labels[p] == f])) for f in FAMILIES},
                                          'delta': float(np.mean(list(delta.values()))), 'delta_by_true_family': {f: float(np.mean([delta[p] for p in delta if labels[p] == f])) for f in FAMILIES},
                                          'delta_by_half': {str(hh): float(np.mean([delta[p] for p in delta if half(table[p]) == hh])) for hh in [0, 1]},
                                          'pool_bootstrap_95': pool_bootstrap(delta, table) if name != 'baseline' else None}
            if name in pooled and fold in freeze['confirmation_folds']:
                pooled[name].update({f'{fold}:{p}': d for p, d in delta.items()}); pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in delta})
        results['folds'][str(fold)] = report
    g = freeze['gate']
    gates = {}
    for name in pooled:
        per_fold = {str(f): results['folds'][str(f)]['candidates'][name] for f in freeze['confirmation_folds']}
        halves = {str(hh): float(np.mean([d for k, d in pooled[name].items() if half(pooled_table[k].split(':')[1]) == hh])) for hh in [0, 1]}
        checks = {'overall_gain_each_fold': all(v['delta'] >= g['per_fold_overall_gain_min'] for v in per_fold.values()),
                  'no_true_family_loss_each_fold': all(min(v['delta_by_true_family'].values()) >= -g['per_fold_true_family_loss_max'] for v in per_fold.values()),
                  'pooled_halves_nonnegative': min(halves.values()) >= 0, 'pooled_halves': halves,
                  'pooled_fold1_fold2_delta': float(np.mean(list(pooled[name].values()))), 'pooled_pool_bootstrap_95': pool_bootstrap(pooled[name], pooled_table)}
        if name == 'A_soft_x2_routed':
            checks['true_soft_gain_each_fold'] = all(v['delta_by_true_family']['soft_play'] >= g['per_fold_true_soft_gain_min_for_A'] for v in per_fold.values())
        checks['passed'] = all(v for k, v in checks.items() if isinstance(v, bool))
        gates[name] = checks
    results['gates'] = gates
    results['limits'] = ['fold0 is the observed screen fold and is excluded from the gate.', 'fold1/fold2 are reused overlapping pool splits (27/23 pools shared with fold0), not untouched data.',
                         'Routing uses fixed predicted families; on these folds routing is near-perfect, which flatters routed candidates relative to public data.',
                         'Full histories only.']
    pl.DataFrame(rows).write_csv(OUT / 'pairs.csv')
    write_json(OUT / 'results.json', results)
    lines = ['# Family query weighting: confirmation results (Claude Version)', '', f"Evaluated {results['evaluated_at']}. Gate folds: 1 and 2 (fold 0 = observed reference).", '',
             '| Fold | Candidate | MAP@5 | Δ vs baseline | 95% pool CI | Δ transfer | Δ soft | Δ isolation | Δ half0 | Δ half1 |', '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for fold in [0, 1, 2]:
        for name, c in results['folds'][str(fold)]['candidates'].items():
            ci = c['pool_bootstrap_95']; d = c['delta_by_true_family']
            lines.append(f"| {fold} | {name} | {c['map5']:.5f} | {c['delta']:+.5f} | {'' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {d['directed_transfer']:+.5f} | {d['soft_play']:+.5f} | {d['coordinated_isolation']:+.5f} | {c['delta_by_half']['0']:+.5f} | {c['delta_by_half']['1']:+.5f} |")
    lines += ['', '| Candidate | pooled fold1+2 Δ | 95% pool CI | halves | gate |', '|---|---:|---|---|---|']
    for name, c in gates.items():
        lines.append(f"| {name} | {c['pooled_fold1_fold2_delta']:+.5f} | [{c['pooled_pool_bootstrap_95'][0]:+.4f}, {c['pooled_pool_bootstrap_95'][1]:+.4f}] | {c['pooled_halves']['0']:+.4f} / {c['pooled_halves']['1']:+.4f} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines += ['', 'Limits: ' + ' '.join(results['limits'])]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
