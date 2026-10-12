"""Step 81b [C] (author: claude).  THE single evaluation of step 81's frozen predictions against held
evidence truth (folds 0-2).  Read number FIVE of these three folds -- steps 11, 65d, 65j, 72b, this --
and the only read of the stage-2 selection study.

Refuses to run unless `predictions-frozen.json` exists and every score file still matches its hash,
and refuses to overwrite an existing results.json.  Nothing here chooses anything: the arm, the
pairings and every gate threshold were written into freeze.json before the fits, which were themselves
run before this file ever touched held truth.

Pairings (each delta is paired per held pair):
  inc_struct  vs chrono1      reproduction control, reported against step 65c
  inc_r382    vs inc_struct   reproduction control, reported against step 65i
  inc_hybrid  vs inc_r382     what the CI routing is currently worth on held data
  new_<arm>   vs inc_hybrid   PRIMARY: the selection change against the stage 2 that ships
  new_<arm>   vs inc_r382     SECONDARY: separates the relative features from the un-routing
  inc_struct  vs inc_hybrid   STANDING CHECK (18.15 / 18.17): a known-worse arm that must FAIL the
                              primary gate.  If it passes, the gate is measuring a design gap rather
                              than the change, and the primary result is void regardless of outcome.

Outputs: results.json, pairs.csv, RESULTS.md under
Claude Version/experiments/stage2_selection_C_20260912/held/.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import ORIGINAL, check_protected, read_json, sha, write_json
from evidence_aug.step5_evaluate import half, per_pair_map, pool_bootstrap
from evidence_aug.step81_C_selection_held_fit import OUT

AUTHOR = 'claude'
STANDING = 'inc_struct vs inc_hybrid'


def main():
    frozen = read_json(OUT / 'predictions-frozen.json')
    assert frozen['freeze_sha256'] == sha(OUT / 'freeze.json'), 'freeze.json changed after freezing'
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest, name
    assert not (OUT / 'results.json').exists(), 'evaluation already recorded; preserve it'
    freeze = read_json(OUT / 'freeze.json')
    protected = check_protected()
    arm = freeze['arm']['tag']
    new = f'new_{arm}'
    folds = freeze['confirmation_folds']

    ALL = ['chrono1', 'inc_struct', 'inc_r382', 'inc_hybrid', new]
    VS = {'inc_struct vs chrono1': ('inc_struct', 'chrono1'),
          'inc_r382 vs inc_struct': ('inc_r382', 'inc_struct'),
          'inc_hybrid vs inc_r382': ('inc_hybrid', 'inc_r382'),
          f'{new} vs inc_hybrid': (new, 'inc_hybrid'),
          f'{new} vs inc_r382': (new, 'inc_r382'),
          STANDING: ('inc_struct', 'inc_hybrid')}
    PRIMARY = f'{new} vs inc_hybrid'

    labels = dict(pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
                  .select('pair_id', 'behavior_family').iter_rows())
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    results = {'author': AUTHOR, 'step': '81b',
               'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protected': protected,
               'arm': freeze['arm'], 'primary': PRIMARY, 'standing_check': STANDING,
               'held_read_number': freeze['held_read_number'], 'folds': {}}
    rows = []
    pooled = {k: {} for k in VS}
    pooled_table, pooled_fam = {}, {}

    for fold in folds:
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
        for key, (a, b) in VS.items():
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

    g = freeze['gate']['pass_if']
    gates = {}
    for key, d in pooled.items():
        per_fold = {str(f): results['folds'][str(f)]['deltas'][key]['delta'] for f in folds}
        ci = pool_bootstrap(d, pooled_table)
        by_fam = {f: float(np.mean([v for k, v in d.items() if pooled_fam[k] == f])) for f in FAMILIES}
        halves = {str(hh): float(np.mean([v for k, v in d.items()
                                          if half(pooled_table[k].split(':')[1]) == hh])) for hh in [0, 1]}
        pd_ = float(np.mean(list(d.values())))
        checks = {'pooled_delta_gt': pd_ > g['pooled_delta_gt'],
                  'ci95_lower_gt': ci[0] > g['ci95_lower_gt'],
                  'true_family_loss_max': min(by_fam.values()) >= -g['true_family_loss_max'],
                  'per_fold_nonnegative_in_at_least':
                      sum(v >= 0 for v in per_fold.values()) >= g['per_fold_nonnegative_in_at_least'],
                  'pooled_delta': pd_, 'pooled_pool_bootstrap_95': ci,
                  'pooled_delta_by_true_family': by_fam, 'pooled_halves': halves,
                  'per_fold_delta': per_fold}
        checks['passed'] = all(v for v in checks.values() if isinstance(v, bool))
        gates[key] = checks

    results['gates'] = gates
    standing_ok = not gates[STANDING]['passed']
    results['standing_check_result'] = {
        'comparison': STANDING,
        'known_worse_arm_delta': gates[STANDING]['pooled_delta'],
        'it_must_fail_the_gate': True, 'it_failed': standing_ok,
        'verdict': 'gate discriminates' if standing_ok else
                   'VOID: a known-worse arm passes this gate, so the gate measures a design gap'}
    results['primary_passed'] = bool(gates[PRIMARY]['passed'] and standing_ok)
    results['limits'] = [
        f"Fifth read of the same three held folds for an evidence study (steps 11, 65d, 65j, 72b, "
        f"this).  The arm and every threshold were selected on the 162 audit pools, which lie outside "
        f"every held fold, and were frozen in freeze.json before any held score was written.",
        f"The arm was chosen from 22 arms scored on that audit surface; step 80b re-tested it under "
        f"three independent partitions of it (deltas {freeze['arm']['audit_deltas_by_salt']}) because "
        f"step 80's seed-triplet test turned out to be a null perturbation -- these LightGBM params "
        f"are deterministic, so random_state changes nothing.",
        'Fixed predicted families of the folds (near-perfect); on the evaluation set the predicted '
        'family is the submission behaviour column, so held performance is an upper estimate of what '
        'the routing contributes.',
        'Full histories only; evaluation histories are shorter (selected pairs median ~78 hands vs '
        '~121), so the chronology-dependent part of any gain may not transfer at full size.',
        'MAP@5 on this held set converts to composite at 0.2x; a +.010 held gain is +.002 composite.']

    pl.DataFrame(rows).write_csv(OUT / 'pairs.csv')
    write_json(OUT / 'results.json', results)

    lines = ['# Stage-2 selection formulation: confirmation results (Claude Version) [C]', '',
             f"Evaluated {results['evaluated_at']}.  Arm `{arm}`.  Primary: {PRIMARY}.  Folds 0, 1, 2.",
             f"Held read number {results['held_read_number']}.", '',
             '| Fold | Candidate | MAP@5 | DT | SP | CI |', '|---|---|---:|---:|---:|---:|']
    for fold in folds:
        for name, c in results['folds'][str(fold)]['candidates'].items():
            f = c['by_true_family']
            lines.append(f"| {fold} | {name} | {c['map5']:.4f} | {f['directed_transfer']:.4f} | "
                         f"{f['soft_play']:.4f} | {f['coordinated_isolation']:.4f} |")
    lines += ['', '| Comparison | pooled Δ | per fold | 95% pool CI | DT / SP / CI | halves | gate |',
              '|---|---:|---|---|---|---|---|']
    for key, c in gates.items():
        f, pf = c['pooled_delta_by_true_family'], c['per_fold_delta']
        lines.append(
            f"| {key} | {c['pooled_delta']:+.4f} | "
            f"{' / '.join(f'{pf[k]:+.4f}' for k in sorted(pf))} | "
            f"[{c['pooled_pool_bootstrap_95'][0]:+.4f}, {c['pooled_pool_bootstrap_95'][1]:+.4f}] | "
            f"{f['directed_transfer']:+.4f} / {f['soft_play']:+.4f} / {f['coordinated_isolation']:+.4f} | "
            f"{c['pooled_halves']['0']:+.4f} / {c['pooled_halves']['1']:+.4f} | "
            f"{'PASS' if c['passed'] else 'FAIL'} |")
    sc = results['standing_check_result']
    lines += ['', f"STANDING CHECK ({STANDING}): known-worse arm scores {sc['known_worse_arm_delta']:+.4f} "
                  f"and {'FAILS the gate as required' if sc['it_failed'] else 'PASSES -- RESULT VOID'}.",
              '', f"PRIMARY ({PRIMARY}): {'PASS' if results['primary_passed'] else 'FAIL'}",
              '', 'Limits:'] + [f'- {x}' for x in results['limits']]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
