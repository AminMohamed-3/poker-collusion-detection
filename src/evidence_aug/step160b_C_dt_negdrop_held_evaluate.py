"""Step 160 part B (Claude): the one held evaluation (evidence held read 11) of the frozen step-160 held scores, in the form of step 155b
(duplicate outer pools averaged per unique pair, pool bootstrap). Registration: experiments/dt_negdrop_held_C_20260915/registration160.json."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
from datetime import datetime, timezone
from evidence_aug.paths import ORIGINAL
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap

EXP = CV / 'experiments/dt_negdrop_held_C_20260915'
HOUT = EXP / 'held160'
REG = EXP / 'registration160.json'
DT = 'directed_transfer'
ARMS = ('incumbent', 'dec_S_star_ci_untyped', 'K0R_v1', 'K0R_noview')
PRIMARY = 'K0R_v1_vs_incumbent'
COMPARISONS = {PRIMARY: ('K0R_v1', 'incumbent'), 'K0R_noview_vs_S_star_decoded': ('K0R_noview', 'dec_S_star_ci_untyped')}


def mean_or_none(x):
    m = x.mean()
    return None if m is None else float(m)


def summary(q):
    delta = dict(q.select('pair_id', 'delta').iter_rows())
    pool = dict(q.select('pair_id', 'table_id').iter_rows())
    return {'n_pairs': q.height, 'mean_delta': float(q['delta'].mean()), 'ci95': pool_bootstrap(delta, pool),
            'by_family': {f: mean_or_none(q.filter(pl.col('family') == f)['delta']) for f in FAMILIES},
            'halves': {str(h): mean_or_none(q.filter(pl.col('table_id').map_elements(half, return_dtype=pl.Int64) == h)['delta']) for h in (0, 1)},
            'pairs_better': int((q['delta'] > 1e-12).sum()), 'pairs_worse': int((q['delta'] < -1e-12).sum())}


def main():
    assert not (HOUT / 'results160.json').exists(), 'preserve the completed held read'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    fix = json.loads((EXP / 'registration160_evaluator_fix.json').read_text(encoding='utf-8'))
    assert fix['registration_sha256'] == sha(REG) and fix['evaluate_sha256_registered'] == reg['code']['evaluate_sha256']
    assert sha(Path(__file__)) == fix['evaluate_sha256_fixed']
    frozen = json.loads((HOUT / 'predictions-frozen.json').read_text(encoding='utf-8'))
    assert frozen['registration_sha256'] == sha(REG) and frozen['protected_unchanged'] and not frozen['held_evidence_evaluation']
    for name, digest in frozen['scores'].items():
        assert sha(HOUT / 'scores' / name) == digest, name
    started = datetime.now(timezone.utc).isoformat()
    # every held score is hashed above; held truth is read only from here on
    evidence = pl.read_csv(ORIGINAL / 'data/development_evidence.csv').select('pair_id', 'hand_id')
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    families = dict(pl.read_csv(ORIGINAL / 'data/development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    rows, levels = [], {}
    for fold in (0, 1, 2):
        ref = pl.read_parquet(HOUT / 'scores' / f'fold{fold}-incumbent.parquet')
        table = dict(ref.select('pair_id', 'table_id').unique().iter_rows())
        roles = dict(pl.read_csv(HOUT / 'scores' / f'fold{fold}-roles.csv').iter_rows())
        maps = {a: per_pair_map(pl.read_parquet(HOUT / 'scores' / f'fold{fold}-{a}.parquet'), truths) for a in ARMS}
        levels[str(fold)] = {a: float(np.mean(list(m.values()))) for a, m in maps.items()}
        for comp, (cand, base) in COMPARISONS.items():
            for p in maps[base]:
                rows.append({'fold': fold, 'comparison': comp, 'pair_id': p, 'table_id': table[p], 'family': families[p], 'routed': roles[p],
                             'base_map5': maps[base][p], 'map5': maps[cand][p], 'delta': maps[cand][p] - maps[base][p]})
    frame = pl.DataFrame(rows)
    frame.write_parquet(HOUT / 'pair_metrics160.parquet')
    nondt = frame.filter(pl.col('routed') != DT)
    result = {'step': 160, 'author': 'Claude (session a82f9192)', 'evaluated_at': started, 'evidence_held_read_count_after': 11, 'map5_levels_by_fold': levels,
              'non_dt_routed_max_abs_delta': float(nondt['delta'].abs().max()), 'comparisons': {}}
    assert result['non_dt_routed_max_abs_delta'] == 0.0, 'only directed_transfer-routed lists may differ'
    for comp in COMPARISONS:
        q = frame.filter(pl.col('comparison') == comp)
        byfold = {str(f): summary(q.filter(pl.col('fold') == f)) for f in (0, 1, 2)}
        unique = q.group_by('pair_id', 'table_id', 'family', 'routed').agg(pl.col('delta').mean(), pl.col('base_map5').mean(), pl.col('map5').mean()).sort('pair_id')
        report = summary(unique)
        ud = unique.filter(pl.col('routed') == DT)
        report.update({'unique_pool_count': unique['table_id'].n_unique(), 'fold_appearance_count': q.height, 'by_fold': byfold,
                       'unique_pair_base_map5': float(unique['base_map5'].mean()), 'unique_pair_map5': float(unique['map5'].mean()),
                       'dt_routed_unique': summary(ud) | {'base_map5': float(ud['base_map5'].mean())},
                       'dt_routed_by_fold': {str(f): float(q.filter((pl.col('fold') == f) & (pl.col('routed') == DT))['delta'].mean()) for f in (0, 1, 2)}})
        result['comparisons'][comp] = report
    p = result['comparisons'][PRIMARY]
    d_dt = p['dt_routed_unique']['mean_delta']
    folds_nonneg = sum(v >= 0 for v in p['dt_routed_by_fold'].values())
    build = bool(d_dt > 0 and folds_nonneg >= 2)
    result['decision'] = {'dt_routed_unique_mean_delta': d_dt, 'dt_routed_folds_nonnegative': folds_nonneg, 'BUILD': build, 'verdict': 'BUILD' if build else 'NO_BUILD'}
    u = reg['predictions']
    result['prediction_check'] = {'dt_routed_unique_mean_delta': {'observed': d_dt, 'registered': u['dt_routed_unique_mean_delta'],
                                                                 'inside80': u['dt_routed_unique_mean_delta'][1] <= d_dt <= u['dt_routed_unique_mean_delta'][2]}}
    tp = reg['public_if_built']
    result['public_point_if_built'] = float(tp['scale'] * tp['E_weight'] * tp['dt_share_of_positive_pairs'] * d_dt) if build else None
    result['protected_unchanged'] = bool(check_protected())
    write(HOUT / 'results160.json', result)
    for comp, r in result['comparisons'].items():
        print(comp, round(r['mean_delta'], 6), [round(x, 6) for x in r['ci95']], 'n', r['n_pairs'], 'b/w', r['pairs_better'], r['pairs_worse'],
              '| DT routed', round(r['dt_routed_unique']['mean_delta'], 6), [round(x, 6) for x in r['dt_routed_unique']['ci95']], 'n', r['dt_routed_unique']['n_pairs'],
              'b/w', r['dt_routed_unique']['pairs_better'], r['dt_routed_unique']['pairs_worse'], 'folds', {k: round(v, 6) for k, v in r['dt_routed_by_fold'].items()},
              'halves', {k: round(v, 6) for k, v in r['dt_routed_unique']['halves'].items()}, flush=True)
    print('levels', json.dumps(levels), flush=True)
    print('DECISION', json.dumps(result['decision']), json.dumps(result['prediction_check']), 'public point', result['public_point_if_built'], flush=True)


if __name__ == '__main__':
    main()
