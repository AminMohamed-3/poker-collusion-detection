"""Step 106b (Claude): the single held evaluation of the frozen step-106 residual reranker arms. Held truth is read only after every score hash checks."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
from datetime import datetime, timezone
from evidence_aug.paths import ORIGINAL
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step98b_C_prefix_confirm_evaluate import summary

OUT = CV / 'experiments/residual_reranker_C_20260913'
PREREG = OUT / 'prediction-residual-reranker.json'
PREREG_SHA = '2ee464189ba41bb258629272422b98321094cb881fa1b792db0c6dad034fbde5'
S98 = CV / 'experiments/prefix_strength_confirmation_C_20260913'
S104 = CV / 'experiments/typed_listing_decoder_C_20260913'
STEP104_TYPED_B0_DELTA = 0.0072169312169312145
SOURCES = {'B0': S98, 'incumbent': S98, 'stage1_only': S98, 'typed_B0': S104, 'resid_B0': OUT, 'resid_shuffled_B0': OUT, 'resid_structure_B0': OUT}
COMPARISONS = {'resid_B0_vs_B0': ('resid_B0', 'B0'), 'resid_B0_vs_typed_B0': ('resid_B0', 'typed_B0'),
               'resid_B0_vs_resid_shuffled_B0': ('resid_B0', 'resid_shuffled_B0'), 'resid_shuffled_B0_vs_B0': ('resid_shuffled_B0', 'B0'),
               'resid_structure_B0_vs_B0': ('resid_structure_B0', 'B0'), 'resid_B0_vs_resid_structure_B0': ('resid_B0', 'resid_structure_B0'),
               'typed_B0_vs_B0': ('typed_B0', 'B0'), 'resid_B0_vs_incumbent': ('resid_B0', 'incumbent'), 'stage1_only_vs_B0': ('stage1_only', 'B0')}


def main():
    assert not (OUT / 'results.json').exists(), 'Preserve completed evaluation'
    assert sha(PREREG) == PREREG_SHA
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    frozen = json.loads((OUT / 'predictions-frozen.json').read_text(encoding='utf-8'))
    assert frozen['registration_sha256'] == PREREG_SHA and frozen['protected_unchanged'] and frozen['codex_files_unchanged']
    assert frozen['audit_replay']['within_tolerance'] and not frozen['held_evidence_evaluation']
    assert sha(OUT / 'models.json') == frozen['models_sha256']
    for name, digest in frozen['scores'].items():
        assert sha(OUT / 'scores' / name) == digest, name
    frozen98 = json.loads((S98 / 'predictions-frozen.json').read_text(encoding='utf-8'))
    frozen104 = json.loads((S104 / 'predictions-frozen.json').read_text(encoding='utf-8'))
    b0_exact = all(sha(S98 / 'scores' / n) == h and frozen98['scores'][n] == h for n, h in frozen['base_scores_step98'].items())
    decoder_exact = all(sha(S104 / 'scores' / n) == h and frozen104['scores'][n] == h for n, h in frozen['decoder_scores_step104'].items())
    assert b0_exact and decoder_exact
    gate, protected = reg['gate'], check_protected()
    started = datetime.now(timezone.utc).isoformat()
    # every candidate score is hashed above; held truth is read only from here on
    evidence = pl.read_csv(ORIGINAL / 'data/development_evidence.csv').select('pair_id', 'hand_id')
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    families = dict(pl.read_csv(ORIGINAL / 'data/development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    rows = []
    for fold in (0, 1, 2):
        path = {a: src / 'scores' / f'fold{fold}-{a}.parquet' for a, src in SOURCES.items()}
        table = dict(pl.read_parquet(path['B0']).select('pair_id', 'table_id').unique().iter_rows())
        role = dict(pl.read_csv(S98 / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {a: per_pair_map(pl.read_parquet(p), truths) for a, p in path.items()}
        for comp, (cand, base) in COMPARISONS.items():
            for p in maps[base]:
                rows.append({'fold': fold, 'comparison': comp, 'pair_id': p, 'table_id': table[p], 'family': families[p], 'predicted': role[p],
                             'base_map5': maps[base][p], 'map5': maps[cand][p], 'delta': maps[cand][p] - maps[base][p]})
    frame = pl.DataFrame(rows)
    frame.write_parquet(OUT / 'pair_metrics.parquet')
    result = {'step': 106, 'author': 'Claude', 'evaluated_at': started, 'reused_outer': True, 'held_read_count_mine_after': 8, 'comparisons': {}}
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
        report['checks'], report['gate_pass'] = checks, all(checks.values())
        result['comparisons'][comp] = report
    cm = result['comparisons']
    result['controls'] = {'B0_scores_hash_exact_vs_step98': b0_exact, 'decoder_scores_hash_exact_vs_step104': decoder_exact,
                          'typed_B0_replays_step104': abs(cm['typed_B0_vs_B0']['mean_delta'] - STEP104_TYPED_B0_DELTA) <= 1e-9,
                          'residual_adds_over_decoder': cm['resid_B0_vs_typed_B0']['mean_delta'] > 0,
                          'aligned_attributes_beat_shuffled': cm['resid_B0_vs_resid_shuffled_B0']['mean_delta'] > 0}
    prim = cm['resid_B0_vs_B0']
    result['passed'] = bool(prim['gate_pass'] and all(result['controls'].values()))
    u = reg['prediction']['resid_B0_minus_B0_point_low80_high80']
    result['prediction_check'] = {'observed': prim['mean_delta'], 'point_low80_high80': u, 'inside80': u[1] <= prim['mean_delta'] <= u[2],
                                  'ci_lower_above_zero': prim['ci95'][0] > 0}
    result['protected_unchanged'] = check_protected() == protected
    write(OUT / 'results.json', result)
    for comp, r in cm.items():
        print(comp, round(r['mean_delta'], 6), [round(x, 6) for x in r['ci95']], 'folds', {k: round(v['mean_delta'], 6) for k, v in r['by_fold'].items()},
              'fam', {k[:5]: round(v, 6) for k, v in r['by_family'].items()}, 'halves', {k: round(v, 6) for k, v in r['halves'].items()},
              'better/worse', r['pairs_better'], r['pairs_worse'], 'gate', r['gate_pass'], flush=True)
    print('checks', prim['checks'], flush=True)
    print('controls', result['controls'], '| PASSED', result['passed'], '| prediction', result['prediction_check'], flush=True)


if __name__ == '__main__':
    main()
