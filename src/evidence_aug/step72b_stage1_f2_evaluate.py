"""Step 72b: THE single evaluation of the frozen step-72 predictions against held evidence truth (folds 0-2).

Refuses to run unless predictions-frozen.json exists and every score file still matches its hash, and
refuses to overwrite an existing results.json.  Fourth read of these three folds (steps 11, 65d, 65j,
this) and the only read of the stage-1 study.

Pairings (each delta is paired per held pair):
  allfam_top20_struct vs chrono1              reproduction control, reported against step 65c
  allfam_top20_r382   vs allfam_top20_struct  reproduction control, reported against step 65i
  s1f2                vs chrono1              PRIMARY: the stage-1 change on its own
  s1f2_top20_struct   vs allfam_top20_struct  the stage-1 change under the struct stage 2
  s1f2_top20_r382     vs allfam_top20_r382    the stage-1 change under what ships today
  s1f2_top20_r382     vs s1f2                 diagnostic: does stage 2 still pay on the new stage 1?
Outputs: results.json, pairs.csv, RESULTS.md under Claude Version/experiments/stage1_f2_20260912/."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, sha, write_json, read_json, check_protected
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap, half
from evidence_aug.step72_stage1_f2_fit import OUT
import numpy as np
import polars as pl

PRIMARY = 's1f2'
VS = {'allfam_top20_struct': 'chrono1',
      'allfam_top20_r382': 'allfam_top20_struct',
      's1f2': 'chrono1',
      's1f2_top20_struct': 'allfam_top20_struct',
      's1f2_top20_r382': 'allfam_top20_r382'}
EXTRA = {'s1f2_top20_r382 vs s1f2': ('s1f2_top20_r382', 's1f2'),
         's1f2_top20_struct vs s1f2': ('s1f2_top20_struct', 's1f2')}
ALL = ['chrono1'] + list(VS)


def main():
    frozen = read_json(OUT / 'predictions-frozen.json')
    assert frozen['freeze_sha256'] == sha(OUT / 'freeze.json')
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest, name
    assert not (OUT / 'results.json').exists(), 'evaluation already recorded; preserve it'
    freeze = read_json(OUT / 'freeze.json')
    protected = check_protected()
    labels = dict(pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
                  .select('pair_id', 'behavior_family').iter_rows())
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    results = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protected': protected,
               'primary': PRIMARY, 'pairings': VS, 'folds': {}}
    rows = []
    pooled = {v: {} for v in list(VS) + list(EXTRA)}
    pooled_table, pooled_fam = {}, {}
    for fold in freeze['confirmation_folds']:
        chrono = pl.read_parquet(OUT / 'scores' / f'fold{fold}-chrono1.parquet')
        table = dict(chrono.select('pair_id', 'table_id').unique().iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in
                  evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        assert set(truths) == set(table)
        predicted = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {v: per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-{v}.parquet'), truths)
                for v in ALL}
        report = {'held_target_pairs': len(table),
                  'true_family_counts': {f: sum(1 for p in table if labels[p] == f) for f in FAMILIES},
                  'candidates': {}, 'deltas': {}}
        for name, m in maps.items():
            rows += [{'fold': fold, 'candidate': name, 'pair_id': p, 'table_id': table[p],
                      'family': labels[p], 'predicted': predicted[p], 'map5': m[p]} for p in m]
            report['candidates'][name] = {
                'map5': float(np.mean(list(m.values()))),
                'by_true_family': {f: float(np.mean([m[p] for p in m if labels[p] == f]))
                                   for f in FAMILIES}}
        for key, (a, b) in list((k, (k, v)) for k, v in VS.items()) + list(EXTRA.items()):
            delta = {p: maps[a][p] - maps[b][p] for p in maps[a]}
            report['deltas'][key] = {
                'a': a, 'b': b, 'delta': float(np.mean(list(delta.values()))),
                'delta_by_true_family': {f: float(np.mean([delta[p] for p in delta if labels[p] == f]))
                                         for f in FAMILIES},
                'delta_by_half': {str(hh): float(np.mean([delta[p] for p in delta
                                                          if half(table[p]) == hh])) for hh in [0, 1]},
                'pairs_improved_worsened': [int(sum(d > 1e-12 for d in delta.values())),
                                            int(sum(d < -1e-12 for d in delta.values()))],
                'pool_bootstrap_95': pool_bootstrap(delta, table)}
            pooled[key].update({f'{fold}:{p}': d for p, d in delta.items()})
        pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in maps['chrono1']})
        pooled_fam.update({f'{fold}:{p}': labels[p] for p in maps['chrono1']})
        results['folds'][str(fold)] = report

    g = freeze['gate']
    gates = {}
    for key in pooled:
        d = pooled[key]
        per_fold = {str(f): results['folds'][str(f)]['deltas'][key]['delta']
                    for f in freeze['confirmation_folds']}
        halves = {str(hh): float(np.mean([v for k, v in d.items()
                                          if half(pooled_table[k].split(':')[1]) == hh]))
                  for hh in [0, 1]}
        ci = pool_bootstrap(d, pooled_table)
        by_fam = {f: float(np.mean([v for k, v in d.items() if pooled_fam[k] == f])) for f in FAMILIES}
        per_fold_fam_ok = all(
            min(results['folds'][str(f)]['deltas'][key]['delta_by_true_family'].values())
            >= -g['per_fold_true_family_loss_max'] for f in freeze['confirmation_folds'])
        checks = {'pooled_delta_min': float(np.mean(list(d.values()))) >= g['pooled_delta_min'],
                  'per_fold_gain_nonnegative_in_at_least':
                      sum(v >= 0 for v in per_fold.values()) >= g['per_fold_gain_nonnegative_in_at_least'],
                  'per_fold_true_family_loss': per_fold_fam_ok,
                  'pooled_true_family_loss': min(by_fam.values()) >= -g['pooled_true_family_loss_max'],
                  'pooled_ci_excludes_zero': ci[0] > 0,
                  'pooled_halves_nonnegative': min(halves.values()) >= 0,
                  'pooled_delta': float(np.mean(list(d.values()))), 'pooled_delta_by_true_family': by_fam,
                  'pooled_pool_bootstrap_95': ci, 'pooled_halves': halves, 'per_fold_delta': per_fold}
        checks['passed'] = all(x for x in checks.values() if isinstance(x, bool))
        gates[key] = checks
    results['gates'] = gates
    results['primary_passed'] = gates[PRIMARY]['passed']
    results['limits'] = [
        'Fourth read of the same three held folds for an evidence study (steps 11, 65d, 65j, this); '
        'the stage-1 hypothesis and its arms were selected on the 162 audit pools, outside every held fold.',
        'Fixed predicted families of the folds (near-perfect); on the evaluation set the predicted '
        'family is the submission behaviour column.',
        'Full histories only; evaluation histories are shorter (selected pairs median ~78 hands vs ~121).',
        'The audit measured the stage-1 delta at +.0197 [+.0047, +.0356] over three ranker seeds; the '
        'held arms use the single production seed 7329.']
    pl.DataFrame(rows).write_csv(OUT / 'pairs.csv')
    write_json(OUT / 'results.json', results)

    lines = ['# Structural + mechanism features inside stage 1: confirmation results (Claude Version)', '',
             f"Evaluated {results['evaluated_at']}. Primary: {PRIMARY} vs chrono1. Folds 0, 1, 2.", '',
             '| Fold | Candidate | MAP@5 | DT | SP | CI |', '|---|---|---:|---:|---:|---:|']
    for fold in freeze['confirmation_folds']:
        r = results['folds'][str(fold)]
        for name, c in r['candidates'].items():
            f = c['by_true_family']
            lines.append(f"| {fold} | {name} | {c['map5']:.4f} | {f['directed_transfer']:.4f} | "
                         f"{f['soft_play']:.4f} | {f['coordinated_isolation']:.4f} |")
    lines += ['', '| Comparison | pooled Δ (3 folds) | per fold | 95% pool CI | DT / SP / CI | halves | gate |',
              '|---|---:|---|---|---|---|---|']
    for key, c in gates.items():
        f = c['pooled_delta_by_true_family']
        pf = c['per_fold_delta']
        lines.append(
            f"| {key} | {c['pooled_delta']:+.4f} | "
            f"{' / '.join(f'{pf[k]:+.4f}' for k in sorted(pf))} | "
            f"[{c['pooled_pool_bootstrap_95'][0]:+.4f}, {c['pooled_pool_bootstrap_95'][1]:+.4f}] | "
            f"{f['directed_transfer']:+.4f} / {f['soft_play']:+.4f} / {f['coordinated_isolation']:+.4f} | "
            f"{c['pooled_halves']['0']:+.4f} / {c['pooled_halves']['1']:+.4f} | "
            f"{'PASS' if c['passed'] else 'FAIL'} |")
    lines += ['', f"PRIMARY ({PRIMARY} vs chrono1): {'PASS' if results['primary_passed'] else 'FAIL'}",
              '', 'Limits: ' + ' '.join(results['limits'])]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
