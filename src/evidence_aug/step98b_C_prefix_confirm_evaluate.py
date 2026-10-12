"""Step 98b (Claude): the one held evaluation of the frozen step-98 confirmation, in the form of Codex
contestable_state_confirm_evaluate.py (duplicate outer pools clustered). Held read count (mine) becomes 6."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
from datetime import datetime, timezone
from evidence_aug.paths import ORIGINAL
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap

OUT = CV / 'experiments/prefix_strength_confirmation_C_20260913'
PREREG = OUT / 'prediction-prefix-confirmation.json'
PREREG_SHA = 'd28237612dcd656509df2fc33dd9d74b99d8a9b77d8b1e7dfc68abcc87ea2e85'
CODEX_CONF = HERE / 'experiments/contestable_state_confirmation_20260912'
ARMS = ('incumbent', 'B0', 'S_star', 'stage1_only')
COMPARISONS = {'S_star_vs_B0': ('S_star', 'B0'), 'B0_vs_incumbent': ('B0', 'incumbent'),
               'stage1_only_vs_B0': ('stage1_only', 'B0'), 'S_star_vs_incumbent': ('S_star', 'incumbent')}


def summary(q):
    delta = dict(q.select('pair_id', 'delta').iter_rows())
    pool = dict(q.select('pair_id', 'table_id').iter_rows())
    return {'n_pairs': q.height, 'mean_delta': float(q['delta'].mean()),
            'ci95': pool_bootstrap(delta, pool),
            'by_family': {f: float(q.filter(pl.col('family') == f)['delta'].mean()) for f in FAMILIES},
            'halves': {str(h): float(q.filter(pl.col('table_id').map_elements(half, return_dtype=pl.Int64) == h)['delta'].mean()) for h in (0, 1)},
            'pairs_better': int((q['delta'] > 1e-12).sum()), 'pairs_worse': int((q['delta'] < -1e-12).sum())}


def main():
    assert not (OUT / 'results.json').exists(), 'Preserve completed evaluation'
    assert sha(PREREG) == PREREG_SHA
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    frozen = json.loads((OUT / 'predictions-frozen.json').read_text(encoding='utf-8'))
    assert frozen['freeze_sha256'] == sha(OUT / 'freeze.json') and frozen['protected_unchanged'] and frozen['codex_resume_unchanged']
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest, name
    design = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    for name, digest in design['source_hashes'].items():
        assert sha(Path(name)) == digest, name
    gate = reg['gate']
    assert design['gate'] == gate
    started = datetime.now(timezone.utc).isoformat()
    # every candidate score is hashed above; held truth is read only from here on
    evidence = pl.read_csv(ORIGINAL / 'data/development_evidence.csv').select('pair_id', 'hand_id')
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    families = dict(pl.read_csv(ORIGINAL / 'data/development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    rows = []
    for fold in (0, 1, 2):
        ref = pl.read_parquet(OUT / 'scores' / f'fold{fold}-incumbent.parquet')
        table = dict(ref.select('pair_id', 'table_id').unique().iter_rows())
        role = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {a: per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-{a}.parquet'), truths) for a in ARMS}
        for comp, (cand, base) in COMPARISONS.items():
            for p in maps[base]:
                rows.append({'fold': fold, 'comparison': comp, 'pair_id': p, 'table_id': table[p], 'family': families[p], 'predicted': role[p],
                             'base_map5': maps[base][p], 'map5': maps[cand][p], 'delta': maps[cand][p] - maps[base][p]})
    frame = pl.DataFrame(rows)
    frame.write_parquet(OUT / 'pair_metrics.parquet')
    result = {'step': 98, 'author': 'Claude', 'evaluated_at': started, 'reused_outer': True, 'held_read_count_mine_after': 6,
              'comparisons': {}, 'protected_unchanged': check_protected() == design['protected']}
    for comp in COMPARISONS:
        q = frame.filter(pl.col('comparison') == comp)
        byfold = {str(f): summary(q.filter(pl.col('fold') == f)) for f in (0, 1, 2)}
        unique = q.group_by('pair_id', 'table_id', 'family').agg(pl.col('delta').mean(), pl.col('base_map5').mean(), pl.col('map5').mean())
        report = summary(unique)
        report.update({'unique_pool_count': unique['table_id'].n_unique(), 'fold_appearance_count': q.height,
                       'appearance_weighted_delta': float(q['delta'].mean()), 'by_fold': byfold,
                       'unique_pair_base_map5': float(unique['base_map5'].mean()), 'unique_pair_map5': float(unique['map5'].mean())})
        checks = {'mean_delta_min': report['mean_delta'] >= gate['primary_mean_unique_pair_delta_min'],
                  'ci_lower_positive': report['ci95'][0] > gate['raw_pool_bootstrap_lower_gt'],
                  'all_fold_deltas_nonnegative': min(r['mean_delta'] for r in byfold.values()) >= 0,
                  'pooled_family_floor': min(report['by_family'].values()) >= gate['pooled_true_family_floor'],
                  'every_fold_family_floor': min(v for r in byfold.values() for v in r['by_family'].values()) >= gate['each_fold_true_family_floor'],
                  'both_halves_nonnegative': min(report['halves'].values()) >= 0}
        report['checks'] = checks
        report['gate_pass'] = all(checks.values())
        result['comparisons'][comp] = report
    codex = json.loads((CODEX_CONF / 'results.json').read_text(encoding='utf-8'))['comparisons']['contestable_state']
    rep = result['comparisons']['B0_vs_incumbent']
    result['controls'] = {'B0_reproduces_codex_confirmation': bool(abs(rep['mean_delta'] - codex['mean_delta']) < 1e-12 and all(abs(x - y) < 1e-12 for x, y in zip(rep['ci95'], codex['ci95']))),
                          'negative_control_rejected': not result['comparisons']['stage1_only_vs_B0']['gate_pass']}
    prim = result['comparisons']['S_star_vs_B0']
    result['passed'] = bool(prim['gate_pass'] and all(result['controls'].values()))
    u = reg['prediction']['updated_before_any_step98_output']['S_star_minus_B0_point_low80_high80']
    result['prediction_check'] = {'observed': prim['mean_delta'], 'point_low80_high80': u, 'inside80': u[1] <= prim['mean_delta'] <= u[2], 'ci_lower_above_zero': prim['ci95'][0] > 0}
    write(OUT / 'results.json', result)
    for comp, r in result['comparisons'].items():
        print(comp, round(r['mean_delta'], 6), [round(x, 6) for x in r['ci95']], 'folds', {k: round(v['mean_delta'], 6) for k, v in r['by_fold'].items()},
              'fam', {k[:5]: round(v, 6) for k, v in r['by_family'].items()}, 'halves', {k: round(v, 6) for k, v in r['halves'].items()}, 'gate', r['gate_pass'], flush=True)
    print('controls', result['controls'], '| PASSED', result['passed'], '| prediction', result['prediction_check'], flush=True)


if __name__ == '__main__':
    main()
