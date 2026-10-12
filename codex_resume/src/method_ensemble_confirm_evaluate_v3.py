"""One evaluation of the frozen primary ensemble; duplicate outer pools clustered."""
from audit_common import *
from evidence_aug.paths import ORIGINAL
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap
from datetime import datetime, timezone

OUT = HERE / 'experiments/method_ensemble_confirmation_v3_20260912'


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
    frozen = json.loads((OUT / 'predictions-frozen.json').read_text(encoding='utf-8'))
    assert frozen['freeze_sha256'] == sha(OUT / 'freeze.json')
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest
    design = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    for name, digest in design['source_hashes'].items():
        assert sha(Path(name)) == digest
    # All candidate scores now exist and are hashed before held metrics.
    evidence = pl.read_csv(ORIGINAL / 'data/development_evidence.csv').select('pair_id', 'hand_id')
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    families = dict(pl.read_csv(ORIGINAL / 'data/development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    rows = []
    for fold in design['confirmation_folds']:
        ref = pl.read_parquet(OUT / 'scores' / f'fold{fold}-incumbent.parquet')
        table = dict(ref.select('pair_id', 'table_id').unique().iter_rows())
        role = dict(pl.read_csv(OUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        baseline = per_pair_map(ref, truths)
        for arm in ['equal_methods', 'stage1_only']:
            m = per_pair_map(pl.read_parquet(OUT / 'scores' / f'fold{fold}-{arm}.parquet'), truths)
            for p in baseline:
                rows.append({'fold': fold, 'arm': arm, 'pair_id': p, 'table_id': table[p], 'family': families[p],
                             'predicted': role[p], 'base_map5': baseline[p], 'map5': m[p], 'delta': m[p] - baseline[p]})
    frame = pl.DataFrame(rows)
    frame.write_parquet(OUT / 'pair_metrics.parquet')
    result = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'reused_outer': True, 'comparisons': {}, 'protected_unchanged': check_protected() == design['protected']}
    for arm in ['equal_methods', 'stage1_only']:
        q = frame.filter(pl.col('arm') == arm)
        byfold = {str(f): summary(q.filter(pl.col('fold') == f)) for f in design['confirmation_folds']}
        unique = q.group_by('pair_id', 'table_id', 'family').agg(pl.col('delta').mean(), pl.col('base_map5').mean(), pl.col('map5').mean())
        report = summary(unique)
        report.update({'unique_pool_count': unique['table_id'].n_unique(), 'fold_appearance_count': q.height,
                       'appearance_weighted_delta': float(q['delta'].mean()), 'by_fold': byfold,
                       'unique_pair_base_map5': float(unique['base_map5'].mean()), 'unique_pair_map5': float(unique['map5'].mean())})
        checks = {'mean_delta_min': report['mean_delta'] >= .01, 'ci_lower_positive': report['ci95'][0] > 0,
                  'all_fold_deltas_nonnegative': min(r['mean_delta'] for r in byfold.values()) >= 0,
                  'pooled_family_floor': min(report['by_family'].values()) >= -.01,
                  'every_fold_family_floor': min(v for r in byfold.values() for v in r['by_family'].values()) >= -.01,
                  'both_halves_nonnegative': min(report['halves'].values()) >= 0}
        report['checks'] = checks
        report['gate_pass'] = all(checks.values())
        result['comparisons'][arm] = report
        print(arm, json.dumps(report), flush=True)
    result['passed'] = result['comparisons']['equal_methods']['gate_pass']
    result['negative_control_rejected'] = not result['comparisons']['stage1_only']['gate_pass']
    write(OUT / 'results.json', result)
    print('DONE', json.dumps({'passed': result['passed']}), flush=True)


if __name__ == '__main__':
    main()
