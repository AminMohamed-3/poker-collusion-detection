"""Step 5: single evaluation of the frozen six-arm predictions against held evidence truth.

This is the ONLY module that reads held-pool rows of data/development_evidence.csv.
It refuses to run unless predictions-frozen.json exists and every score file still matches its hash.
Outputs: results.json, pairs.csv (per pair-arm MAP@5), RESULTS.md
"""
import os
import sys
import json
import hashlib
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import EXP, ORIGINAL, SAVED_SCORES, sha, write_json, read_json, check_protected
from evidence_aug.frame import roles, FAMILIES
import numpy as np
import polars as pl

ARMS = ['original_only', 'control_88721', 'control_99643', 'control_110017', 'synthetic_capped', 'synthetic_all_causal']
CONTROLS = ['control_88721', 'control_99643', 'control_110017']
TIE_SEED = 931207
BOOT_SEED, BOOT_REPS = 20319, 2000


def ap5(ranking, truth):
    hits, total = 0, 0.0
    for rank, hand in enumerate(ranking, 1):
        if hand in truth:
            hits += 1
            total += hits / rank
    return total / min(5, len(truth)) if truth else 0.0


def half(table_id):
    return int(hashlib.sha256(('softplay-fast-v1:' + table_id).encode()).hexdigest(), 16) % 2


def per_pair_map(scored, truths):
    """scored: pair_id, hand_id, table_id, score sorted by pair_id, hand_id (tie draw follows that order)."""
    scored = scored.sort('pair_id', 'hand_id').with_columns(pl.Series('tie', np.random.default_rng(TIE_SEED).random(scored.height)))
    out = {}
    for key, frame in scored.group_by('pair_id'):
        order = frame.sort(['score', 'tie'], descending=[True, False])['hand_id'].head(5).to_list()
        out[key[0]] = ap5(order, truths[key[0]])
    return out


def pool_bootstrap(delta_by_pair, table_by_pair):
    pools = {}
    for p, d in delta_by_pair.items():
        pools.setdefault(table_by_pair[p], []).append(d)
    keys = sorted(pools)
    sums = np.array([sum(pools[k]) for k in keys]); ns = np.array([len(pools[k]) for k in keys])
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(keys), (BOOT_REPS, len(keys)))
    boot = sums[ix].sum(axis=1) / ns[ix].sum(axis=1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def main():
    frozen = read_json(EXP / 'predictions-frozen.json')
    assert frozen['freeze_sha256'] == sha(EXP / 'freeze.json')
    for name, digest in frozen['scores'].items():
        assert sha(EXP / 'scores' / name) == digest, name
    assert not (EXP / 'results.json').exists(), 'evaluation already recorded; preserve it'
    protected = check_protected()
    freeze = read_json(EXP / 'freeze.json')
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'behavior_family')
    held_pairs = pl.read_parquet(EXP / 'scores' / 'original_only-conditional_roles.parquet').select('pair_id', 'table_id').unique()
    assert held_pairs.height == 98
    # --- held truth is read here, after the freeze ---
    evidence = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id').join(held_pairs.select('pair_id'), on='pair_id', how='semi')
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    assert set(truths) == set(held_pairs['pair_id'])
    family = dict(labels.join(held_pairs, on='pair_id').select('pair_id', 'behavior_family').iter_rows())
    table = dict(held_pairs.iter_rows())
    results = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'held_truth_pairs': len(truths), 'held_listed_hands': evidence.height, 'routings': {}}
    rows = []
    for kind in ['conditional_roles', 'original_roles']:
        predicted = dict(roles(kind).iter_rows())
        routed = {p for p in family if predicted[p] == 'soft_play'}
        baseline = pl.read_parquet(SAVED_SCORES[kind]).join(held_pairs, on='pair_id')
        maps = {}
        for arm in ARMS:
            arm_scores = pl.read_parquet(EXP / 'scores' / f'{arm}-{kind}.parquet')
            composite = pl.concat([arm_scores.filter(pl.col('pair_id').is_in(sorted(routed))), baseline.filter(~pl.col('pair_id').is_in(sorted(routed))).select(arm_scores.columns)])
            assert composite.height == arm_scores.height
            maps[arm] = per_pair_map(composite, truths)
            rows += [{'routing': kind, 'arm': arm, 'pair_id': p, 'table_id': table[p], 'family': family[p], 'routed_to_arm': p in routed, 'map5': v} for p, v in maps[arm].items()]
        # unrouted diagnostic: arm applied to every held pair with the predicted family
        unrouted = {arm: per_pair_map(pl.read_parquet(EXP / 'scores' / f'{arm}-{kind}.parquet'), truths) for arm in ARMS}
        report = {'routed_pairs': len(routed), 'routed_true_families': {f: sum(1 for p in routed if family[p] == f) for f in FAMILIES}, 'arms': {}, 'comparisons': {}, 'unrouted_diagnostic': {}}
        for arm in ARMS:
            m = maps[arm]
            report['arms'][arm] = {'map5': float(np.mean(list(m.values()))),
                                   'by_true_family': {f: float(np.mean([m[p] for p in m if family[p] == f])) for f in FAMILIES},
                                   'routed_subset_map5': float(np.mean([m[p] for p in routed])) if routed else None,
                                   'by_half': {str(hh): float(np.mean([m[p] for p in m if half(table[p]) == hh])) for hh in [0, 1]}}
            report['unrouted_diagnostic'][arm] = {'map5': float(np.mean(list(unrouted[arm].values()))),
                                                  'by_true_family': {f: float(np.mean([unrouted[arm][p] for p in unrouted[arm] if family[p] == f])) for f in FAMILIES}}
        control_mean = {p: float(np.mean([maps[c][p] for c in CONTROLS])) for p in maps['original_only']}
        for arm in ARMS[1:]:
            comp = {}
            for base_name, base in [('original_only', maps['original_only']), ('control_mean', control_mean)]:
                delta = {p: maps[arm][p] - base[p] for p in base}
                comp[base_name] = {'delta': float(np.mean(list(delta.values()))),
                                   'by_true_family': {f: float(np.mean([delta[p] for p in delta if family[p] == f])) for f in FAMILIES},
                                   'by_half': {str(hh): float(np.mean([delta[p] for p in delta if half(table[p]) == hh])) for hh in [0, 1]},
                                   'pool_bootstrap_95': pool_bootstrap(delta, table)}
            g = freeze['gate']; a, b = comp['original_only'], comp['control_mean']
            comp['gate'] = {'overall_vs_original': a['delta'] >= g['overall_gain_vs_original_min'], 'overall_vs_control_mean': b['delta'] >= g['overall_gain_vs_control_mean_min'],
                            'true_soft_vs_original': a['by_true_family']['soft_play'] >= g['true_soft_gain_min'], 'true_soft_vs_control_mean': b['by_true_family']['soft_play'] >= g['true_soft_gain_min'],
                            'no_family_loss_vs_original': min(a['by_true_family'].values()) >= -g['true_family_loss_max'], 'no_family_loss_vs_control_mean': min(b['by_true_family'].values()) >= -g['true_family_loss_max'],
                            'halves_nonnegative_vs_original': min(a['by_half'].values()) >= 0, 'halves_nonnegative_vs_control_mean': min(b['by_half'].values()) >= 0}
            comp['gate']['passed'] = all(comp['gate'].values())
            report['comparisons'][arm] = comp
        results['routings'][kind] = report
    # oracle-family diagnostic (true family one-hots for every pair, no routing): never a candidate
    oracle = {}
    for arm in ARMS:
        m = per_pair_map(pl.read_parquet(EXP / 'scores' / f'{arm}-oracle_family.parquet'), truths)
        oracle[arm] = {'map5': float(np.mean(list(m.values()))), 'by_true_family': {f: float(np.mean([m[p] for p in m if family[p] == f])) for f in FAMILIES}}
    results['oracle_family_diagnostic'] = oracle
    results['limits'] = ['Reused 100-pool outer partition (457 pairs, 98 target queries); not untouched validation.',
                         'Synthetic positives are simulator interventions, not verified real-world suspiciousness.',
                         'Unlisted original hands and unselected synthetic hands are unknown, not confirmed negatives.',
                         'Full histories only; shortened-history checks are not part of this screen.']
    pl.DataFrame(rows).write_csv(EXP / 'pairs.csv')
    write_json(EXP / 'results.json', results)
    primary = results['routings']['conditional_roles']
    lines = ['# Evidence-augmentation screen results (Claude Version)', '', f"Evaluated {results['evaluated_at']}. Primary routing: conditional_roles. Routed (predicted soft-play) pairs: {primary['routed_pairs']} of 98; true families among routed: {primary['routed_true_families']}.", '',
             '| Arm | MAP@5 | transfer | soft | isolation | routed subset | half0 | half1 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r = primary['arms'][arm]; f = r['by_true_family']
        lines.append(f"| {arm} | {r['map5']:.5f} | {f['directed_transfer']:.5f} | {f['soft_play']:.5f} | {f['coordinated_isolation']:.5f} | {r['routed_subset_map5']:.5f} | {r['by_half']['0']:.5f} | {r['by_half']['1']:.5f} |")
    lines += ['', '| Arm | Δ vs original | 95% pool CI | Δ vs control mean | 95% pool CI | Δ soft vs orig | Δ soft vs ctrl | gate |', '|---|---:|---|---:|---|---:|---:|---|']
    for arm in ARMS[1:]:
        c = primary['comparisons'][arm]
        lines.append(f"| {arm} | {c['original_only']['delta']:+.5f} | [{c['original_only']['pool_bootstrap_95'][0]:+.4f}, {c['original_only']['pool_bootstrap_95'][1]:+.4f}] | {c['control_mean']['delta']:+.5f} | [{c['control_mean']['pool_bootstrap_95'][0]:+.4f}, {c['control_mean']['pool_bootstrap_95'][1]:+.4f}] | {c['original_only']['by_true_family']['soft_play']:+.5f} | {c['control_mean']['by_true_family']['soft_play']:+.5f} | {'PASS' if c['gate']['passed'] else 'FAIL'} |")
    lines += ['', 'Oracle-family diagnostic (true family one-hots, all pairs, no routing; never a candidate):', '', '| Arm | MAP@5 | transfer | soft | isolation |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        o = oracle[arm]; f = o['by_true_family']
        lines.append(f"| {arm} | {o['map5']:.5f} | {f['directed_transfer']:.5f} | {f['soft_play']:.5f} | {f['coordinated_isolation']:.5f} |")
    lines += ['', 'Limits: ' + ' '.join(results['limits'])]
    (EXP / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
