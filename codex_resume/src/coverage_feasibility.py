"""Read-only accounting of current evidence coverage and fallback implementation."""
from audit_common import *
from evidence_aug.fold import fold_paths

OUT = HERE / 'experiments/coverage_feasibility_20260912'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'receipt.json').exists()
    best_path = CV / 'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
    best = pl.read_csv(best_path, infer_schema=False)
    old = pl.read_csv(ROOT / 'outputs_v6/submission_sequence_evidence.csv', infer_schema=False)
    selection = pl.read_parquet(ROOT / 'reports/replay-v13/selected-pairs.parquet').select('pair_id')
    outside = best.join(selection, on='pair_id', how='anti').with_columns(pl.col('risk_score').cast(pl.Float64).alias('_risk'))
    ev = [c for c in best.columns if c not in ('pair_id', 'risk_score', 'predicted_behavior')]
    compare = outside.join(old.select('pair_id', *ev), on='pair_id', suffix='_v6', validate='1:1')
    exact = compare.select(pl.all_horizontal([pl.col(c) == pl.col(c + '_v6') for c in ev]).all()).item()
    counts = []
    for path in sorted((ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet')):
        key = pl.scan_parquet(path).select('pair_id', 'hand_id').collect()
        counts.append({'batch': path.stem, 'all_pair_hands': key.height,
                       'outside_pair_hands': key.join(outside.select('pair_id'), on='pair_id', how='semi').height,
                       'cache_bytes': path.stat().st_size})
    metadata = {}
    for fold in (0, 1, 2):
        path = fold_paths(fold).get('roles')
        if path:
            f = pl.read_csv(path)
            metadata[str(fold)] = {'path': str(path), 'columns': f.columns,
                                   'rows': f.height, 'categorical_values': {k:f[k].unique().to_list() for k in ['history','candidate'] if k in f.columns}}
    res = {'source_sha256':sha(best_path), 'selected_pairs':selection.height, 'fallback_pairs':outside.height,
           'fallback_exactly_v6':exact, 'fallback_risk_max':float(outside['_risk'].max()),
           'fallback_risk_sum_not_positive_count':float(outside['_risk'].sum()),
           'fallback_by_predicted_family':outside.group_by('predicted_behavior').len().to_dicts(),
           'cache_counts':counts, 'fallback_pair_hands':sum(c['outside_pair_hands'] for c in counts),
           'fold_risk_metadata':metadata,
           'interpretation':'A coverage extension must be compared with the ACTUAL v6 fallback only on the population it would change. It must retain the current hybrid inside the selection. Risk sum is not a calibrated estimate of hidden positives; previous P/B-based hidden-family counts are not established.',
           'next':'Check whether existing honest grouped risk predictions provide enough confirmed positives below the actual coverage boundary to test this scope. If not, document validation gap before investing in all-population feature generation.',
           'new_fit':False, 'evaluation_labels_read':False}
    write(OUT/'receipt.json',res)
    print(json.dumps({k:v for k,v in res.items() if k!='cache_counts'},indent=2),flush=True)


if __name__ == '__main__':
    main()
