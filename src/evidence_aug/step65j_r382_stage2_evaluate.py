"""Step 65j: single evaluation of the frozen step-65i predictions against held evidence truth (folds 0-2).
Refuses to run unless predictions-frozen.json exists and every score file still matches its hash; refuses to overwrite
results. Deltas are paired per held pair against allfam_top20_struct (the step-65c design, reproduced in 65i), with the
chrono1 baseline reported for reference. Outputs: results.json, pairs.csv, RESULTS.md under
Claude Version/experiments/family_stage2_r382_20260911/."""
import os
import sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap, half
from evidence_aug.step65i_r382_stage2_fit import OUT
import numpy as np
import polars as pl

BASE = 'allfam_top20_struct'
VARIANTS = ['allfam_top20_r382', 'allfam_top20_r382ff50', 'allfam_top20_r382sel']


def main():
    frozen = read_json(OUT / 'predictions-frozen.json')
    assert frozen['freeze_sha256'] == sha(OUT / 'freeze.json')
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest, name
    assert not (OUT / 'results.json').exists(), 'evaluation already recorded; preserve it'
    freeze = read_json(OUT / 'freeze.json'); protected = check_protected()
    variants = [v for v in VARIANTS if (OUT / 'scores' / f'fold0-{v}.parquet').exists()]
    labels = dict(pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    results = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'base': BASE, 'variants': variants, 'folds': {}}
    rows = []; pooled = {v: {} for v in variants + [BASE]}; pooled_table = {}; pooled_fam = {}
    for fold in freeze['confirmation_folds']:
        chrono = pl.read_parquet(OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        table = dict(chrono.select('pair_id', 'table_id').unique().iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        assert set(truths) == set(table)
        predicted = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {'chrono1': per_pair_map(chrono, truths)}
        for v in [BASE] + variants:
            maps[v] = per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-{v}.parquet'), truths)
        report = {'held_target_pairs': len(table), 'true_family_counts': {f: sum(1 for p in table if labels[p] == f) for f in FAMILIES}, 'candidates': {}}
        for name, m in maps.items():
            ref = maps['chrono1'] if name == BASE else maps[BASE]
            delta = {p: m[p] - ref[p] for p in m}
            rows += [{'fold': fold, 'candidate': name, 'pair_id': p, 'table_id': table[p], 'family': labels[p], 'predicted': predicted[p], 'map5': m[p]} for p in m]
            report['candidates'][name] = {'map5': float(np.mean(list(m.values()))), 'by_true_family': {f: float(np.mean([m[p] for p in m if labels[p] == f])) for f in FAMILIES},
                                          'delta_vs': 'chrono1' if name == BASE else BASE, 'delta': float(np.mean(list(delta.values()))),
                                          'delta_by_true_family': {f: float(np.mean([delta[p] for p in delta if labels[p] == f])) for f in FAMILIES},
                                          'delta_by_half': {str(hh): float(np.mean([delta[p] for p in delta if half(table[p]) == hh])) for hh in [0, 1]},
                                          'pairs_improved_worsened': [int(sum(d > 1e-12 for d in delta.values())), int(sum(d < -1e-12 for d in delta.values()))],
                                          'pool_bootstrap_95': pool_bootstrap(delta, table) if name != 'chrono1' else None}
            if name in pooled:
                pooled[name].update({f'{fold}:{p}': d for p, d in delta.items()}); pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in delta}); pooled_fam.update({f'{fold}:{p}': labels[p] for p in delta})
        results['folds'][str(fold)] = report
    g = freeze['gate']; gates = {}
    for v in variants + [BASE]:
        per_fold = {str(f): results['folds'][str(f)]['candidates'][v] for f in freeze['confirmation_folds']}
        halves = {str(hh): float(np.mean([d for k, d in pooled[v].items() if half(pooled_table[k].split(':')[1]) == hh])) for hh in [0, 1]}
        ci = pool_bootstrap(pooled[v], pooled_table)
        by_fam = {f: float(np.mean([d for k, d in pooled[v].items() if pooled_fam[k] == f])) for f in FAMILIES}
        checks = {'pooled_delta_min': float(np.mean(list(pooled[v].values()))) >= g['pooled_delta_min'],
                  'per_fold_gain_nonnegative_in_at_least': sum(c['delta'] >= 0 for c in per_fold.values()) >= g['per_fold_gain_nonnegative_in_at_least'],
                  'per_fold_true_family_loss': all(min(c['delta_by_true_family'].values()) >= -g['per_fold_true_family_loss_max'] for c in per_fold.values()),
                  'pooled_true_family_loss': min(by_fam.values()) >= -g['pooled_true_family_loss_max'],
                  'pooled_ci_excludes_zero': ci[0] > 0, 'pooled_halves_nonnegative': min(halves.values()) >= 0,
                  'pooled_delta': float(np.mean(list(pooled[v].values()))), 'pooled_delta_by_true_family': by_fam, 'pooled_pool_bootstrap_95': ci, 'pooled_halves': halves,
                  'per_fold_delta': {f: c['delta'] for f, c in per_fold.items()}, 'delta_vs': 'chrono1' if v == BASE else BASE}
        checks['passed'] = all(x for x in checks.values() if isinstance(x, bool))
        gates[v] = checks
    results['gates'] = gates
    results['limits'] = ['Third read of the same three held folds for an evidence study (step 11, step 65d, this); the r382 hypothesis and its variants were selected on the audit pools, outside every held fold.',
                         'Fixed predicted families of the folds (near-perfect); on the evaluation set the predicted family is the submission behaviour column.', 'Full histories only.']
    pl.DataFrame(rows).write_csv(OUT / 'pairs.csv')
    write_json(OUT / 'results.json', results)
    lines = ['# Per-family second stage with the 382 ranker features (r382): confirmation results (Claude Version)', '', f"Evaluated {results['evaluated_at']}. Primary: allfam_top20_r382 vs {BASE}. Folds 0, 1, 2.", '',
             '| Fold | Variant | MAP@5 | Δ | vs | 95% pool CI | Δ transfer | Δ soft | Δ isolation | Δ half0 | Δ half1 | pairs +/− |', '|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|']
    for fold in freeze['confirmation_folds']:
        r = results['folds'][str(fold)]
        lines.append(f"| {fold} | (held {r['held_target_pairs']} pairs; true DT/SP/CI {r['true_family_counts']['directed_transfer']}/{r['true_family_counts']['soft_play']}/{r['true_family_counts']['coordinated_isolation']}) | | | | | | | | | | |")
        for name, c in r['candidates'].items():
            ci = c['pool_bootstrap_95']; d = c['delta_by_true_family']; pw = c['pairs_improved_worsened']
            lines.append(f"| {fold} | {name} | {c['map5']:.4f} | {c['delta']:+.4f} | {c['delta_vs']} | {'' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {d['directed_transfer']:+.4f} | {d['soft_play']:+.4f} | {d['coordinated_isolation']:+.4f} | {c['delta_by_half']['0']:+.4f} | {c['delta_by_half']['1']:+.4f} | {pw[0]}/{pw[1]} |")
    lines += ['', '| Variant | vs | pooled Δ (3 folds) | per fold | 95% pool CI | by true family DT / SP / CI | halves | gate |', '|---|---|---:|---|---|---|---|---|']
    for v, c in gates.items():
        f = c['pooled_delta_by_true_family']; pf = c['per_fold_delta']
        lines.append(f"| {v} | {c['delta_vs']} | {c['pooled_delta']:+.4f} | {' / '.join(f'{pf[k]:+.4f}' for k in sorted(pf))} | [{c['pooled_pool_bootstrap_95'][0]:+.4f}, {c['pooled_pool_bootstrap_95'][1]:+.4f}] | {f['directed_transfer']:+.4f} / {f['soft_play']:+.4f} / {f['coordinated_isolation']:+.4f} | {c['pooled_halves']['0']:+.4f} / {c['pooled_halves']['1']:+.4f} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines += ['', 'Limits: ' + ' '.join(results['limits'])]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
