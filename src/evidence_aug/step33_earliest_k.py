"""Step 33: is the evidence list 'the first qualifying events, capped at 5'? Two label-process checks and a
pre-registered re-ranking rule on the frozen fold scores (OBSERVATIONS 14.11).

Check A (label process, all development evidence): is evidence_rank ordered by started_at within each pair?
Check B (frozen fold scores, folds 0-2): rule E_k = take the ranker's top-m hands, output the EARLIEST 5 of them.
  Pre-registered primary: chrono1 scores, m = 10. Baseline-score variant and m in {5, 7, 15, 20} reported as
  sensitivity only (no selection on them). Gate to build a candidate: primary beats chrono1 top-5 on every fold
  and pooled pool-bootstrap interval above zero.
Outputs: Claude Version/experiments/earliest_k_20260911/results.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap
import numpy as np
import polars as pl

CHRONO = CV / 'experiments' / 'chronology_20260910' / 'scores'
OUT = CV / 'experiments' / 'earliest_k_20260911'


def map5_from_lists(lists, truths):
    out = {}
    for p, hands in lists.items():
        rel = truths.get(p, set())
        if not rel:
            continue
        hits = 0; s = 0.0
        for i, h in enumerate(hands[:5]):
            if h in rel:
                hits += 1; s += hits / (i + 1)
        out[p] = s / min(len(rel), 5)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').select('hand_id', pl.col('started_at').cast(pl.Int64))
    ev = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').join(hands, on='hand_id')
    # Check A
    g = ev.sort('pair_id', 'evidence_rank').group_by('pair_id', maintain_order=True).agg(pl.col('started_at').alias('t'), pl.col('evidence_rank').alias('r'))
    mono_inc = 0; mono_dec = 0; n = 0
    for t in g['t'].to_list():
        if len(t) < 2:
            continue
        n += 1
        d = np.diff(np.array(t, dtype='datetime64[us]').astype(np.int64)) if not isinstance(t[0], (int, float)) else np.diff(np.array(t, dtype=float))
        mono_inc += bool((d > 0).all()); mono_dec += bool((d < 0).all())
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'check_A': {'pairs_with_2plus_evidence': n, 'rank_increasing_in_time': mono_inc, 'rank_decreasing_in_time': mono_dec,
                                                                              'share_increasing': mono_inc / max(n, 1)}}
    print(json.dumps(res['check_A']), flush=True)
    # Check B
    evidence_all = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    folds = {}
    pooled = {}; pooled_table = {}
    for fold in [0, 1, 2]:
        c1 = pl.read_parquet(CHRONO / f'fold{fold}-chrono1.parquet').join(hands, on='hand_id')
        base = pl.read_parquet(CHRONO / f'fold{fold}-baseline.parquet').join(hands, on='hand_id')
        table = dict(c1.select('pair_id', 'table_id').unique().iter_rows())
        truths = {k[0]: set(g['hand_id']) for k, g in evidence_all.filter(pl.col('pair_id').is_in(list(table))).group_by('pair_id')}
        ref = per_pair_map(c1.drop('started_at'), truths)
        rep = {'chrono1_top5': float(np.mean(list(ref.values()))), 'baseline_top5': float(np.mean(list(per_pair_map(base.drop('started_at'), truths).values()))), 'rules': {}}
        for name, df in [('chrono1', c1), ('baseline', base)]:
            for m in [5, 7, 10, 15, 20]:
                top = df.sort('pair_id', 'score', descending=[False, True]).group_by('pair_id', maintain_order=True).head(m)
                lists = {k[0]: g.sort('started_at')['hand_id'].to_list() for k, g in top.group_by('pair_id')}
                mm = map5_from_lists(lists, truths)
                d = {p: mm[p] - ref[p] for p in ref}
                rep['rules'][f'{name}_m{m}'] = {'map5': float(np.mean(list(mm.values()))), 'delta_vs_chrono1': float(np.mean(list(d.values()))), 'ci95': pool_bootstrap(d, table) if (name == 'chrono1' and m == 10) else None}
                if name == 'chrono1' and m == 10:
                    pooled.update({f'{fold}:{p}': v for p, v in d.items()}); pooled_table.update({f'{fold}:{p}': f'{fold}:{table[p]}' for p in d})
        folds[str(fold)] = rep
        print(json.dumps({'fold': fold, 'chrono1_top5': round(rep['chrono1_top5'], 4), **{k: round(v['map5'], 4) for k, v in rep['rules'].items()}}), flush=True)
    ci = pool_bootstrap(pooled, pooled_table)
    res['check_B'] = {'folds': folds, 'primary': 'chrono1_m10', 'pooled_delta': float(np.mean(list(pooled.values()))), 'pooled_ci95': ci,
                      'gate_passed': bool(ci[0] > 0 and all(folds[str(f)]['rules']['chrono1_m10']['delta_vs_chrono1'] > 0 for f in [0, 1, 2]))}
    write_json(OUT / 'results.json', res)
    print(json.dumps({k: v for k, v in res['check_B'].items() if k != 'folds'}), flush=True)


if __name__ == '__main__':
    main()
