"""Adaptive but fixed-weight ensemble screen across all three completed methods.

No fitted mixture weights and no favorable-family routing. This is selection on
the reused audit; a pass earns confirmation, never immediate production.
"""
from audit_common import *
from datetime import datetime, timezone

OUT = HERE / 'experiments/method_ensemble_20260912'
GROUPS = {
    'symmetry_censor': ('symmetry_censor_20260912', ['swap_average', 'symmetric_training', 'censor_half', 'censor_quarter', 'symmetric_censor_half']),
    'catboost': ('catboost_evidence_20260912', ['cat_logloss', 'cat_map']),
    'comparison': ('pair_comparator_20260912', ['pair_delta', 'pair_context']),
}
WEIGHTS = {
    'equal_methods': {'incumbent': .25, 'symmetry_censor': .25, 'catboost': .25, 'comparison': .25},
    'half_incumbent': {'incumbent': .5, 'symmetry_censor': 1/6, 'catboost': 1/6, 'comparison': 1/6},
    'proposal_only': {'incumbent': 0., 'symmetry_censor': 1/3, 'catboost': 1/3, 'comparison': 1/3},
}


def ranks(pid, values):
    return pl.DataFrame({'pair_id': pid, 'v': values}).select((-pl.col('v').rank('average', descending=True).over('pair_id')).alias('r'))['r'].to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'freeze.json').exists()
    protected = check_protected()
    src = [Path(__file__), HERE / 'src/audit_common.py'] + [HERE / 'experiments' / folder / 'scores.parquet' for folder, _ in GROUPS.values()]
    write(OUT / 'freeze.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'primary': 'equal_methods',
        'source_hashes': {str(p): sha(p) for p in src}, 'groups': GROUPS, 'weights': WEIGHTS,
        'rationale': 'All three method studies have small average gains but fail uncertainty gates. Test complementary errors with equal mass per methodology, keeping every standalone trained proposal and the intended symmetry averaging proposal; exclude diagnostic-only swap and previously derived blends to avoid counting incumbent multiple times.',
        'aggregation': 'Average negative within-query ranks of standalone models inside each method group; then weighted arithmetic sum with incumbent negative rank. Same weights for every family and pair. Top20 stays fixed.',
        'disclosure': 'Adaptively proposed after three failed screens; aggregate outcomes not inspected before this freeze. This is not independent validation or a new dataset.',
        'gate': {'mean_map5_gain_min': .01, 'pool_bootstrap_lower_gt': 0, 'family_floor': -.01, 'both_pool_halves_nonnegative': True},
        'progression': 'If a gate passes, primary first; otherwise half_incumbent then proposal_only. Freeze one strict nested grouped confirmation against complete shipping hybrid, predicted-family routing. Report all failures.',
        'protected': protected, 'outer_truth_read': False})
    d, _, _, saved = load_current()
    top = (d['oof_rank'] <= 20).to_numpy()
    pid = d['pair_id'].to_numpy()[top]
    group_scores = {'incumbent': ranks(pid, saved['incumbent'].to_numpy()[top])}
    for name, (folder, arms) in GROUPS.items():
        scores = pl.read_parquet(HERE / 'experiments' / folder / 'scores.parquet').sort('pair_id', 'hand_id')
        assert scores.select('pair_id', 'hand_id').equals(saved.select('pair_id', 'hand_id'))
        assert np.array_equal(scores['incumbent'].to_numpy(), saved['incumbent'].to_numpy())
        group_scores[name] = np.mean([ranks(pid, scores[arm].to_numpy()[top]) for arm in arms], axis=0)
    pred = {'incumbent': saved['incumbent'].to_numpy()}
    for name, weights in WEIGHTS.items():
        assert np.isclose(sum(weights.values()), 1.)
        p = d['oof'].to_numpy().copy() - 1e6
        p[top] = sum(weights[k] * group_scores[k] for k in weights) + 1000
        pred[name] = p
    sf = OUT / 'scores.parquet'
    d.select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(k, v) for k, v in pred.items()]).write_parquet(sf)
    write(OUT / 'predictions-frozen.json', {'freeze_sha256': sha(OUT / 'freeze.json'), 'scores_sha256': sha(sf), 'outer_truth_read': False})
    results = evaluate(d, dict(pred, stage1_only=d['oof'].to_numpy()))
    results['protected_unchanged'] = check_protected() == protected
    write(OUT / 'results.json', results)
    for arm, metrics in results['comparisons'].items():
        print(arm, json.dumps(metrics), flush=True)
    print('DONE', json.dumps({'passed': results['passed']}), flush=True)


if __name__ == '__main__':
    main()
