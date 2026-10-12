"""Step 69: fold-0 60 %-prefix all-pairs build + frozen score (P4-B, APPROVED 2026-09-13).

Fold-0 port of step40_prefix_audit.build/score with the fold-0 screen assets
(NEXT_PLAN P4; .opencode/continuous/tasks/pair/p4bcd_spec.md section B):
  pools   reports/tabicl-screen/held-pairs.csv (100 tables, 457 labelled pairs)
  policy  reports/conditional-action-experiment/policy-screen/held-models.joblib
  ca_cols .../policy-screen/hand-features.parquet (caf column names)
  style   reports/conditional-action-experiment/public-action-style-counts.parquet
  bundle  reports/conditional-action-experiment/screen-results/conditional-pair.joblib (frozen scorer)
  saved   .../screen-results/pair-predictions.csv (full-length parity reference, asset-4)

Output: Claude Version/experiments/pu_pairs_20260911/fold0-prefix60/
  {identity.json, pairs.csv, batch_*-pairs.parquet, batch_*.json, scores.parquet}
CPU only, 2 threads, utf-8. Resume-safe (skips existing batch files).
Measurement only: no new lever is fitted here.
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
import sys, gc, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
from pu_pairs.build_allpairs_fold import (EXP, sha, all_pairs, base_features, v2_features, conditional_features,
                                          aggregate, tails, extra_aggregate)
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import matrix  # noqa: E402

FOLD = 0
PREFIX = 0.6
POLICY = ORIGINAL / 'reports' / 'conditional-action-experiment' / 'policy-screen' / 'held-models.joblib'
HF = ORIGINAL / 'reports' / 'conditional-action-experiment' / 'policy-screen' / 'hand-features.parquet'
STYLE = ORIGINAL / 'reports' / 'conditional-action-experiment' / 'public-action-style-counts.parquet'
BUNDLE = ORIGINAL / 'reports' / 'conditional-action-experiment' / 'screen-results' / 'conditional-pair.joblib'
SAVED = ORIGINAL / 'reports' / 'conditional-action-experiment' / 'screen-results' / 'pair-predictions.csv'
HELD = ORIGINAL / 'reports' / 'tabicl-screen' / 'held-pairs.csv'
DEFAULT_OUT = EXP / 'fold0-prefix60'


def fold0_pools():
    return sorted(pl.read_csv(HELD)['table_id'].unique().to_list())


def caf_columns():
    return [c for c in pl.read_parquet(HF, n_rows=1).columns if c.startswith('ca_')]


def build(prefix=PREFIX, pools_per_batch=2, out=DEFAULT_OUT):
    OUT = out
    OUT.mkdir(parents=True, exist_ok=True)
    pools = fold0_pools()
    hands_dev = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter(
        (pl.col('phase') == 'development') & pl.col('table_id').is_in(pools))
    hands_dev = hands_dev.sort('table_id', 'started_at').with_columns(
        (pl.int_range(pl.len()).over('table_id') / pl.len().over('table_id')).alias('_pos'))
    hands_all = hands_dev.filter(pl.col('_pos') < prefix).drop('_pos')
    write_json(OUT / 'identity.json', {
        'step': 'P4B_fold0_prefix60', 'fold': FOLD, 'prefix': prefix,
        'pools_per_batch': pools_per_batch, 'pools': pools, 'n_pools': len(pools),
        'hands_kept': hands_all.height, 'hands_dev': hands_dev.height,
        'policy_sha256': sha(POLICY), 'hand_features_sha256': None,
        'bundle_sha256': sha(BUNDLE), 'saved_sha256': sha(SAVED), 'held_sha256': sha(HELD),
        'style_counts_sha256': sha(STYLE),
        'code_sha256': sha(ORIGINAL / 'Claude Version' / 'src' / 'pu_pairs' / 'step69_fold0_prefix.py')})
    pairs = all_pairs(pools, hands_all)
    pairs.write_csv(OUT / 'pairs.csv')
    policy = joblib.load(POLICY)
    caf = caf_columns()
    counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
    nb = 0
    for i, start in enumerate(range(0, len(pools), pools_per_batch)):
        dest = OUT / f'batch_{i:03}-pairs.parquet'
        receipt = OUT / f'batch_{i:03}.json'
        if dest.exists() and receipt.exists():
            print(json.dumps({'cached': i}), flush=True)
            continue
        t0 = time.perf_counter()
        these = pools[start:start + pools_per_batch]
        h = hands_all.filter(pl.col('table_id').is_in(these))
        p = pairs.filter(pl.col('table_id').is_in(these)).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p)
        p = p.join(raw.select('pair_id').unique(), on='pair_id', how='semi')
        raw = v2_features(raw, p)
        names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
        events = [c for c in names if c.startswith('v2_')]
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(
            p.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        cas = []
        for pool in these:
            ca, _, _ = conditional_features(
                keys.filter(pl.col('table_id') == pool),
                h.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind'),
                policy, counts, caf)
            cas.append(ca)
            gc.collect()
        ca = pl.concat(cas)
        pairframe = aggregate(raw, names).join(
            tails(raw, events), on=['pair_id', 'table_id'], validate='1:1').join(
            extra_aggregate(ca, caf), on=['pair_id', 'table_id'], validate='1:1').sort('pair_id')
        assert pairframe.height == p.height, (pairframe.height, p.height)
        pairframe.write_parquet(dest)
        rep = {'batch': i, 'pools': these, 'pairs': pairframe.height, 'pair_hands': raw.height,
               'seconds': round(time.perf_counter() - t0, 1), 'sha256': sha(dest)}
        write_json(receipt, rep)
        print(json.dumps(rep), flush=True)
        nb += 1
        del raw, ca, cas, pairframe, keys
        gc.collect()
    print(json.dumps({'DONE': True, 'batches_built': nb}), flush=True)
    return pairs


def score(out=DEFAULT_OUT):
    OUT = out
    frame = pl.concat([pl.read_parquet(b) for b in sorted(OUT.glob('batch_*-pairs.parquet'))]).sort('pair_id')
    pairs = pl.read_csv(OUT / 'pairs.csv')
    frame = frame.join(pairs.select('pair_id', 'player_1', 'player_2', 'label', 'behavior_family'),
                       on='pair_id', validate='1:1')
    bundle = joblib.load(BUNDLE)
    proba = bundle['model'].predict_proba(matrix(frame, bundle['features']))
    scored = frame.select('pair_id', 'player_1', 'player_2', 'table_id', 'label', 'behavior_family',
                          'shared_hands').with_columns(
        pl.Series('risk', 1 - proba[:, 0]),
        pl.Series('predicted_class', np.argmax(proba[:, 1:], axis=1) + 1))
    scored.write_parquet(OUT / 'scores.parquet')
    print(json.dumps({'scored': scored.height, 'sha256': sha(OUT / 'scores.parquet')}), flush=True)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', type=float, default=PREFIX)
    ap.add_argument('--pools-per-batch', type=int, default=2)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--out', type=str, default='fold0-prefix60')
    args = ap.parse_args()
    out = EXP / args.out
    if not args.score_only:
        build(args.prefix, args.pools_per_batch, out)
    score(out)


if __name__ == '__main__':
    main()
