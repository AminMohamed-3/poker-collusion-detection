"""Step 68e: single evaluation of the frozen step-68d predictions against held evidence
truth (folds 0-2). FIFTH held read per tasking (reused, not pristine).

Refuses to run unless predictions-frozen.json exists and every score file still
matches its hash; refuses to overwrite results. Deltas are paired per held pair
against hybrid_DTSP_r382 (the M3_r382DTSP production pattern, reproduced in 68d),
with chrono1 / struct context reported. Frozen gate (ALL must hold per gated arm
vs hybrid_DTSP_r382): pooled>=.01 AND CI lower>0 AND 3/3 folds>=0 AND no true
family<-.01 AND both halves>=0.
Outputs: results.json, pairs.csv, RESULTS.md under
Claude Version/experiments/stage2_ensemble_held_E1_20260913/."""
import os
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
import sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap, half
from evidence_aug.step68d_held_ensemble_fit import OUT
import numpy as np
import polars as pl

BASE = 'hybrid_DTSP_r382'
GATED = ['ens_rank', 'rank_ref', 'rank_r382']
CONTEXT = ['allfam_top20_struct', 'allfam_top20_r382']


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
    results = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'base': BASE,
               'gated': GATED, 'context': CONTEXT, 'held_read': freeze['held_truth_read'], 'folds': {}}
    rows = []
    pooled, pooled_table, pooled_fam = {v: {} for v in GATED + CONTEXT + [BASE]}, {}, {}
    for fold in freeze['confirmation_folds']:
        chrono = pl.read_parquet(OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        table = dict(chrono.select('pair_id', 'table_id').unique().iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        assert set(truths) == set(table)
        predicted = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {}
        for v in ['chrono1', BASE] + CONTEXT + GATED:
            maps[v] = per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-{v}.parquet'), truths)
        report = {'held_target_pairs': len(table),
                  'true_family_counts': {f: sum(1 for p in table if labels[p] == f) for f in FAMILIES}, 'candidates': {}}
        for name, m in maps.items():
            if name == 'chrono1':
                ref = maps['chrono1']
            elif name == BASE:
                ref = maps['chrono1']
            else:
                ref = maps[BASE]
            delta = {p: m[p] - ref[p] for p in m}
            rows += [{'fold': fold, 'candidate': name, 'pair_id': p, 'table_id': table[p], 'family': labels[p],
                      'predicted': predicted[p], 'map5': m[p]} for p in m]
            report['candidates'][name] = {
                'map5': float(np.mean(list(m.values()))),
                'by_true_family': {f: float(np.mean([m[p] for p in m if labels[p] == f])) for f in FAMILIES},
                'delta_vs': 'self' if name == 'chrono1' else ('chrono1' if name == BASE else BASE),
                'delta': 0.0 if name == 'chrono1' else float(np.mean(list(delta.values()))),
                'delta_by_true_family': {f: 0.0 for f in FAMILIES} if name == 'chrono1' else {f: float(np.mean([delta[p] for p in delta if labels[p] == f])) for f in FAMILIES},
                'delta_by_half': {'0': 0.0, '1': 0.0} if name == 'chrono1' else {str(hh): float(np.mean([delta[p] for p in delta if half(table[p]) == hh])) for hh in [0, 1]},
                'pairs_improved_worsened': [0, 0] if name == 'chrono1' else [int(sum(d > 1e-12 for d in delta.values())), int(sum(d < -1e-12 for d in delta.values()))],
                'pool_bootstrap_95': None if name == 'chrono1' else pool_bootstrap(delta, table)}
            if name in pooled:
                pooled[name].update({f'{fold}:{p}': d for p, d in delta.items()})
                pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in delta})
                pooled_fam.update({f'{fold}:{p}': labels[p] for p in delta})
        results['folds'][str(fold)] = report
    g = freeze['gate']
    gates = {}
    for v in GATED + CONTEXT + [BASE]:
        if v == 'chrono1':
            continue
        per_fold = {str(f): results['folds'][str(f)]['candidates'][v] for f in freeze['confirmation_folds']}
        halves = {str(hh): float(np.mean([d for k, d in pooled[v].items() if half(pooled_table[k].split(':')[1]) == hh])) for hh in [0, 1]}
        ci = pool_bootstrap(pooled[v], pooled_table)
        by_fam = {f: float(np.mean([d for k, d in pooled[v].items() if pooled_fam[k] == f])) for f in FAMILIES}
        pd = float(np.mean(list(pooled[v].values())))
        if v in GATED:
            checks = {'pooled_delta_ge_01': pd >= g['pooled_delta_ge'],
                      'ci_lower_gt_0': ci[0] > 0,
                      'all_3_folds_nonneg': sum(c['delta'] >= 0 for c in per_fold.values()) == 3,
                      'no_true_family_below_m01': min(by_fam.values()) >= -g['pooled_true_family_loss_max'],
                      'both_halves_nonneg': min(halves.values()) >= 0}
            checks.update({'pooled_delta': pd, 'pooled_delta_by_true_family': by_fam, 'pooled_pool_bootstrap_95': ci,
                           'pooled_halves': halves, 'per_fold_delta': {f: c['delta'] for f, c in per_fold.items()},
                           'per_fold_delta_by_true_family': {f: c['delta_by_true_family'] for f, c in per_fold.items()},
                           'pairs_improved_worsened_pooled': [int(sum(d > 1e-12 for d in pooled[v].values())), int(sum(d < -1e-12 for d in pooled[v].values()))],
                           'composite_dComp_point': 0.2 * pd, 'composite_dComp_interval': [0.2 * ci[0], 0.2 * ci[1]],
                           'delta_vs': BASE})
            checks['passed'] = all(x for x in checks.values() if isinstance(x, bool))
        else:
            checks = {'pooled_delta': pd, 'pooled_delta_by_true_family': by_fam, 'pooled_pool_bootstrap_95': ci,
                      'pooled_halves': halves, 'per_fold_delta': {f: c['delta'] for f, c in per_fold.items()},
                      'composite_dComp_point': 0.2 * pd, 'composite_dComp_interval': [0.2 * ci[0], 0.2 * ci[1]],
                      'delta_vs': 'chrono1' if v == BASE else BASE, 'gated': False}
        gates[v] = checks
    results['gates'] = gates
    results['limits'] = ['Fifth read of the same three held folds for an evidence study (steps 11, 65d, 65j, this; 65j2 derived with no new truth); reused, not pristine; selection was on the 162-pool audit only.',
                         'Fixed predicted families of the folds (near-perfect); on the evaluation set the predicted family is the submission behaviour column.', 'Full histories only.']
    pl.DataFrame(rows).write_csv(OUT / 'pairs.csv')
    protected_after = check_protected()
    results['protected_after'] = protected_after
    write_json(OUT / 'results.json', results)
    lines = ['# E1 stage-2 ensemble (+E3 rankers): held confirmation (folds 0-2, 5th read, reused-not-pristine)', '',
             f"Evaluated {results['evaluated_at']}. Reference: {BASE} (M3_r382DTSP pattern). Gated arms: {', '.join(GATED)}.", '',
             '| Fold | Variant | MAP@5 | d vs ref | 95% pool CI | d DT | d SP | d CI | d half0 | d half1 | pairs +/- |',
             '|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|']
    for fold in freeze['confirmation_folds']:
        r = results['folds'][str(fold)]
        lines.append(f"| {fold} | (held {r['held_target_pairs']} pairs; true DT/SP/CI {r['true_family_counts']['directed_transfer']}/{r['true_family_counts']['soft_play']}/{r['true_family_counts']['coordinated_isolation']}) | | | | | | | | | | |")
        for name, c in r['candidates'].items():
            ci = c['pool_bootstrap_95']
            d = c['delta_by_true_family']
            pw = c['pairs_improved_worsened']
            lines.append(f"| {fold} | {name} | {c['map5']:.4f} | {c['delta']:+.4f} | {'' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {d['directed_transfer']:+.4f} | {d['soft_play']:+.4f} | {d['coordinated_isolation']:+.4f} | {c['delta_by_half']['0']:+.4f} | {c['delta_by_half']['1']:+.4f} | {pw[0]}/{pw[1]} |")
    lines += ['', '| Gated arm | vs | pooled d | per fold | 95% pool CI | by true family DT / SP / CI | halves | dComp .2x (point [lo,hi]) | gate |',
              '|---|---|---:|---|---|---|---|---|---|']
    for v in GATED:
        c = gates[v]
        f = c['pooled_delta_by_true_family']
        pf = c['per_fold_delta']
        lines.append(f"| {v} | {c['delta_vs']} | {c['pooled_delta']:+.4f} | {' / '.join(f'{pf[k]:+.4f}' for k in sorted(pf))} | [{c['pooled_pool_bootstrap_95'][0]:+.4f}, {c['pooled_pool_bootstrap_95'][1]:+.4f}] | {f['directed_transfer']:+.4f} / {f['soft_play']:+.4f} / {f['coordinated_isolation']:+.4f} | {c['pooled_halves']['0']:+.4f} / {c['pooled_halves']['1']:+.4f} | {c['composite_dComp_point']:+.5f} [{c['composite_dComp_interval'][0]:+.5f},{c['composite_dComp_interval'][1]:+.5f}] | {'PASS' if c['passed'] else 'FAIL'} |")
    lines += ['', 'Limits: ' + ' '.join(results['limits'])]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
