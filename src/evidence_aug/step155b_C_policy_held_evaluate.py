"""Step 155 part B (Claude): the one held evaluation (evidence held read 10) of the frozen step-155 held scores, in the form of step 98b
(duplicate outer pools averaged per unique pair, pool bootstrap). Registration: experiments/rank_policy_grpo_C_20260914/registration155.json."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
from datetime import datetime, timezone
from evidence_aug.paths import ORIGINAL
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap

EXP = CV / 'experiments/rank_policy_grpo_C_20260914'
HOUT = EXP / 'held155'
REG = EXP / 'registration155.json'
ARMS = ('incumbent', 'dec_S_star_ci_untyped', 'G1_a25', 'G0_a25', 'G1_a10', 'G1_a100', 'G0_a100')
PRIMARY = 'G1_a25_vs_incumbent'
COMPARISONS = {'G1_a25_vs_incumbent': ('G1_a25', 'incumbent'), 'G0_a25_vs_incumbent': ('G0_a25', 'incumbent'),
               'G1_a10_vs_incumbent': ('G1_a10', 'incumbent'), 'G1_a100_vs_incumbent': ('G1_a100', 'incumbent'),
               'G0_a100_vs_incumbent': ('G0_a100', 'incumbent'), 'v1_vs_S_star_decoded_ci_untyped': ('incumbent', 'dec_S_star_ci_untyped')}


def summary(q):
    delta = dict(q.select('pair_id', 'delta').iter_rows())
    pool = dict(q.select('pair_id', 'table_id').iter_rows())
    return {'n_pairs': q.height, 'mean_delta': float(q['delta'].mean()), 'ci95': pool_bootstrap(delta, pool),
            'by_family': {f: float(q.filter(pl.col('family') == f)['delta'].mean()) for f in FAMILIES},
            'halves': {str(h): float(q.filter(pl.col('table_id').map_elements(half, return_dtype=pl.Int64) == h)['delta'].mean()) for h in (0, 1)},
            'pairs_better': int((q['delta'] > 1e-12).sum()), 'pairs_worse': int((q['delta'] < -1e-12).sum())}


def main():
    assert not (HOUT / 'results155.json').exists(), 'preserve the completed held read'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == reg['code']['evaluate_sha256']
    frozen = json.loads((HOUT / 'predictions-frozen.json').read_text(encoding='utf-8'))
    assert frozen['registration_sha256'] == sha(REG) and frozen['protected_unchanged'] and not frozen['held_evidence_evaluation']
    for name, digest in frozen['scores'].items():
        assert sha(HOUT / 'scores' / name) == digest, name
    rule = reg['decision']
    started = datetime.now(timezone.utc).isoformat()
    # every held score is hashed above; held truth is read only from here on
    evidence = pl.read_csv(ORIGINAL / 'data/development_evidence.csv').select('pair_id', 'hand_id')
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    families = dict(pl.read_csv(ORIGINAL / 'data/development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    rows, levels = [], {}
    for fold in (0, 1, 2):
        ref = pl.read_parquet(HOUT / 'scores' / f'fold{fold}-incumbent.parquet')
        table = dict(ref.select('pair_id', 'table_id').unique().iter_rows())
        maps = {a: per_pair_map(pl.read_parquet(HOUT / 'scores' / f'fold{fold}-{a}.parquet'), truths) for a in ARMS}
        levels[str(fold)] = {a: float(np.mean(list(m.values()))) for a, m in maps.items()}
        for comp, (cand, base) in COMPARISONS.items():
            for p in maps[base]:
                rows.append({'fold': fold, 'comparison': comp, 'pair_id': p, 'table_id': table[p], 'family': families[p],
                             'base_map5': maps[base][p], 'map5': maps[cand][p], 'delta': maps[cand][p] - maps[base][p]})
    frame = pl.DataFrame(rows)
    frame.write_parquet(HOUT / 'pair_metrics155.parquet')
    result = {'step': 155, 'author': 'Claude (session 9cdfac27)', 'evaluated_at': started, 'evidence_held_read_count_after': 10, 'map5_levels_by_fold': levels,
              'comparisons': {}, 'protected_unchanged_at_start': True}
    for comp in COMPARISONS:
        q = frame.filter(pl.col('comparison') == comp)
        byfold = {str(f): summary(q.filter(pl.col('fold') == f)) for f in (0, 1, 2)}
        unique = q.group_by('pair_id', 'table_id', 'family').agg(pl.col('delta').mean(), pl.col('base_map5').mean(), pl.col('map5').mean()).sort('pair_id')
        report = summary(unique)
        report.update({'unique_pool_count': unique['table_id'].n_unique(), 'fold_appearance_count': q.height, 'by_fold': byfold,
                       'unique_pair_base_map5': float(unique['base_map5'].mean()), 'unique_pair_map5': float(unique['map5'].mean())})
        result['comparisons'][comp] = report
    p = result['comparisons'][PRIMARY]
    folds_nonneg = sum(v['mean_delta'] >= 0 for v in p['by_fold'].values())
    build = (p['mean_delta'] >= rule['BUILD']['mean_min'] and p['ci95'][0] > rule['BUILD']['ci_lower_gt'] and folds_nonneg >= rule['BUILD']['folds_nonnegative_min']
             and min(p['by_family'].values()) >= rule['BUILD']['family_floor'] and min(p['halves'].values()) >= 0)
    confirmed = p['mean_delta'] >= .01 and p['ci95'][0] > 0 and folds_nonneg == 3 and build
    result['decision'] = {'folds_nonnegative': folds_nonneg, 'BUILD': bool(build), 'CONFIRMED': bool(confirmed),
                          'verdict': 'CONFIRMED' if confirmed else ('BUILD' if build else 'NO_BUILD')}
    u = reg['predictions']
    result['prediction_check'] = {k: {'observed': result['comparisons'][c]['mean_delta'], 'registered': u[k],
                                      'inside80': u[k][1] <= result['comparisons'][c]['mean_delta'] <= u[k][2]}
                                  for k, c in (('G1_a25_minus_incumbent', PRIMARY), ('v1_minus_S_star_decoded', 'v1_vs_S_star_decoded_ci_untyped'))}
    result['protected_unchanged'] = bool(check_protected())
    write(HOUT / 'results155.json', result)
    for comp, r in result['comparisons'].items():
        print(comp, round(r['mean_delta'], 6), [round(x, 6) for x in r['ci95']], 'n', r['n_pairs'], 'b/w', r['pairs_better'], r['pairs_worse'],
              'folds', {k: round(v['mean_delta'], 6) for k, v in r['by_fold'].items()}, 'fam', {k[:2]: round(v, 6) for k, v in r['by_family'].items()},
              'halves', {k: round(v, 6) for k, v in r['halves'].items()}, flush=True)
    print('levels', json.dumps(levels), flush=True)
    print('DECISION', json.dumps(result['decision']), json.dumps(result['prediction_check']), flush=True)


if __name__ == '__main__':
    main()
