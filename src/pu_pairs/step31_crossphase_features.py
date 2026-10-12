"""Step 31: cross-phase pair features (label-free baseline for every evaluation pair).

Facts established 2026-09-11 (OBSERVATIONS 14.10): all 12,000 players sit at one table in BOTH phases; the
evaluation pair list excludes every pair with a disclosed-positive player; development pairs the model scores
>= .9 without a label are clean in evaluation (eval risk mean .004, none >= .5). Evaluation colluders are
therefore fresh plants among players with a clean development history, and each evaluation pair has ~3,000
development hands of the same two players as its own behavioural baseline. Symmetrically, each disclosed
development pair has ~2,000 evaluation hands in which (by the exclusion rule and the cross-phase check) it is
presumed clean. This script builds the base + v2 pair aggregates (features.py / advanced_features.py ports from
build_allpairs_fold.py, no policy stage) for four (pair list, hand phase) combinations:
  eval_on_eval  : 112,540 evaluation pairs, evaluation hands   (target phase)
  eval_on_dev   : 112,540 evaluation pairs, development hands  (baseline)
  lab_on_dev    : 1,860 labelled development pairs, development hands (target phase, training)
  lab_on_eval   : 1,860 labelled development pairs, evaluation hands  (baseline, training)
Usage: -m pu_pairs.step31_crossphase_features --hands development|evaluation   (two processes, one per phase)
Outputs: Claude Version/experiments/crossphase_20260911/{name}/batch_{i:03}.parquet + receipts, {name}.parquet merged.
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, gc, json, time, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
from pu_pairs.build_allpairs_fold import base_features, v2_features
sys.path.insert(0, str(ORIGINAL / 'src'))
import polars as pl
from robust_pair_experiment import aggregate  # noqa: E402

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'crossphase_20260911'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_lists(hands_phase):
    """(name, pairs frame with pair_id, player_1, player_2, table_id) for the given hand phase."""
    seats = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').select('hand_id', 'player_id')
    hands = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').select('hand_id', 'table_id', 'phase')
    ptab = seats.join(hands, on='hand_id').select('player_id', 'table_id', 'phase').unique().collect()
    tab = {ph: dict(ptab.filter(pl.col('phase') == ph).select('player_id', 'table_id').iter_rows()) for ph in ['development', 'evaluation']}
    ev = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2')
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
    out = []
    for name, frame in [('eval', ev), ('lab', lab)]:
        t = frame.with_columns(pl.col('player_1').replace_strict(tab[hands_phase], return_dtype=pl.String).alias('table_id'),
                               pl.col('player_2').replace_strict(tab[hands_phase], return_dtype=pl.String).alias('t2'))
        assert (t['table_id'] == t['t2']).all(), 'pair split across tables'
        out.append((f"{name}_on_{'dev' if hands_phase == 'development' else 'eval'}", t.drop('t2')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', choices=['development', 'evaluation'], required=True)
    ap.add_argument('--tables-per-batch', type=int, default=2)
    args = ap.parse_args()
    t00 = time.perf_counter()
    hands_all = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter(pl.col('phase') == args.hands)
    tables = sorted(set(hands_all['table_id']))
    lists = pair_lists(args.hands)
    for name, pairs in lists:
        (OUT / name).mkdir(parents=True, exist_ok=True)
        write_json(OUT / name / 'identity.json', {'hands_phase': args.hands, 'pairs': pairs.height, 'tables': len(tables), 'tables_per_batch': args.tables_per_batch,
                                                  'code_sha256': sha(ORIGINAL / 'Claude Version' / 'src' / 'pu_pairs' / 'step31_crossphase_features.py'),
                                                  'builder_sha256': sha(ORIGINAL / 'Claude Version' / 'src' / 'pu_pairs' / 'build_allpairs_fold.py'), 'stage': 'base + v2 aggregates (mean/std/max/q95), no policy'})
    batches = list(range(0, len(tables), args.tables_per_batch))
    for i, start in enumerate(batches):
        these = tables[start:start + args.tables_per_batch]
        h = hands_all.filter(pl.col('table_id').is_in(these))
        for name, pairs in lists:
            dest = OUT / name / f'batch_{i:03}.parquet'
            if dest.exists() and (OUT / name / f'batch_{i:03}.json').exists():
                continue
            t0 = time.perf_counter()
            p = pairs.filter(pl.col('table_id').is_in(these)).select('pair_id', 'player_1', 'player_2', 'table_id')
            if p.height == 0:
                pl.DataFrame({'pair_id': pl.Series([], dtype=pl.String)}).write_parquet(dest)
                write_json(OUT / name / f'batch_{i:03}.json', {'batch': i, 'tables': these, 'pairs': 0})
                continue
            raw = base_features(h, p)
            p2 = p.join(raw.select('pair_id').unique(), on='pair_id', how='semi')
            raw = v2_features(raw, p2)
            names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
            frame = aggregate(raw, names).sort('pair_id')
            frame.write_parquet(dest)
            rep = {'batch': i, 'tables': these, 'pairs_requested': p.height, 'pairs_built': frame.height, 'pair_hands': raw.height, 'columns': frame.width,
                   'seconds': round(time.perf_counter() - t0, 2), 'sha256': sha(dest)}
            write_json(OUT / name / f'batch_{i:03}.json', rep)
            print(json.dumps({k: v for k, v in rep.items() if k != 'sha256'} | {'name': name, 'elapsed': round(time.perf_counter() - t00, 1)}), flush=True)
            del raw, frame
            gc.collect()
    for name, pairs in lists:
        parts = [pl.read_parquet(OUT / name / f'batch_{i:03}.parquet') for i in range(len(batches))]
        parts = [x for x in parts if x.height]
        merged = pl.concat(parts, how='vertical_relaxed').sort('pair_id')
        merged.write_parquet(OUT / f'{name}.parquet')
        print(json.dumps({'merged': name, 'pairs': merged.height, 'of': pairs.height, 'columns': merged.width, 'sha256': sha(OUT / f'{name}.parquet')}), flush=True)
    print('done', round(time.perf_counter() - t00, 1), flush=True)


if __name__ == '__main__':
    main()
