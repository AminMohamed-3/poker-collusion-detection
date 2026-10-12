"""Step 12: robustness diagnostic (not a gate) of the frozen chronology models on 65%-thinned held histories.

Evaluation pairs have shorter shared histories than development pairs (median 76 vs ~110 hands). The previous
agent's convention for this is hash-based thinning of each held pair's frame to 65% of its hands (HELDOUT_VIEW =
(0.65, 7365) in robust_pair_experiment.thin) with contexts rebuilt on the thinned frame. This script applies the
already-frozen fold models from step 10 (no refits) to such thinned frames, recomputing the chronology columns
on the thinned frame, and reports MAP@5 against the retained listed hands. Held truth for folds 0-2 has already
been read in step 11; this is a diagnostic on observed folds, recorded as such.
Output: Claude Version/experiments/chronology_20260910/shortened-robustness.json
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import CV, ORIGINAL, SRC, write_json
from evidence_aug.frame import score, family_columns, FAMILIES, context, SEQ, EQ
from evidence_aug.fold import load_fold, fold_roles
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap
from evidence_aug.step10_chronology_fit import add_chronology
sys.path.insert(0, str(SRC))
from robust_pair_experiment import thin, HELDOUT_VIEW
import numpy as np
import polars as pl
import joblib

OUT = CV / 'experiments' / 'chronology_20260910'
NAMES = ['baseline', 'chrono3', 'chrono1']


def rebuild_contexts(frame, features):
    events = [c[len('context_'):] for c in features if c.startswith('context_v2_')]
    caf = [c for c in features if c.startswith('ca_')]
    ctx = [c for c in frame.columns if c.startswith('context_')]
    out = frame.drop(ctx)
    for cols in [events, SEQ, EQ, caf]:
        out = context(out, cols)
    return out.sort('pair_id', 'hand_id')


def main():
    labels = dict(pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'behavior_family').iter_rows())
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    out = {'view': {'fraction': HELDOUT_VIEW[0], 'seed': HELDOUT_VIEW[1]}, 'diagnostic_not_gate': True, 'folds': {}}
    pooled = {n: {} for n in NAMES[1:]}; pooled_table = {}
    for fold in [0, 1, 2]:
        data, features, pools, identity = load_fold(fold)
        held = data.filter(pl.col('table_id').is_in(pools))
        hf = held.join(fold_roles(fold), on='pair_id', validate='m:1').sort('pair_id', 'hand_id')
        thinned = thin(hf.drop([c for c in ['chrono_rank', 'chrono_frac', 'n_hands'] if c in hf.columns]), *HELDOUT_VIEW)
        thinned = add_chronology(rebuild_contexts(thinned, features))
        table = dict(thinned.select('pair_id', 'table_id').unique().iter_rows())
        kept = set(thinned['hand_id'])
        truths = {k[0]: set(g['hand_id']) & kept for k, g in evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        with_truth = [p for p in table if truths[p]]
        maps = {}
        for name in NAMES:
            bundle = joblib.load(OUT / 'scores' / f'fold{fold}-{name}.joblib')
            s = score(bundle['model'], family_columns(thinned, 'predicted_behavior'), bundle['features'])
            maps[name] = per_pair_map(thinned.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', s)), truths)
        report = {'held_pairs': len(table), 'pairs_with_retained_truth': len(with_truth), 'rows_full': hf.height, 'rows_thinned': thinned.height, 'candidates': {}}
        for name in NAMES:
            m = maps[name]; delta = {p: m[p] - maps['baseline'][p] for p in with_truth}
            report['candidates'][name] = {'map5_pairs_with_truth': float(np.mean([m[p] for p in with_truth])), 'delta': float(np.mean(list(delta.values()))) if name != 'baseline' else 0.0,
                                          'delta_by_true_family': {f: float(np.mean([delta[p] for p in delta if labels[p] == f])) for f in FAMILIES},
                                          'pool_bootstrap_95': pool_bootstrap(delta, table) if name != 'baseline' else None}
            if name in pooled:
                pooled[name].update({f'{fold}:{p}': d for p, d in delta.items()}); pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in delta})
        out['folds'][str(fold)] = report
        print(json.dumps({'fold': fold, **{n: round(report['candidates'][n]['delta'], 4) for n in NAMES[1:]}, 'baseline_map5': round(report['candidates']['baseline']['map5_pairs_with_truth'], 4)}), flush=True)
    out['pooled'] = {n: {'delta': float(np.mean(list(pooled[n].values()))), 'pool_bootstrap_95': pool_bootstrap(pooled[n], pooled_table)} for n in NAMES[1:]}
    write_json(OUT / 'shortened-robustness.json', out)
    print(json.dumps(out['pooled']), flush=True)


if __name__ == '__main__':
    main()
