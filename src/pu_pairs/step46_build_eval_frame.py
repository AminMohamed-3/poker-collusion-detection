"""Step 46 (label-free infrastructure): build the 897-feature EVALUATION pair frame that production inference computed
on the fly but never saved. Base aggregates (653) + tails come from outputs_v2/hands_evaluation per-hand frames with the
same aggregate()/tails() code as the training frame; the 132 conditional-action aggregates come from the inference package
(outputs_v11_conditional/inference-package/results/batch_*-pairs.parquet). Output: experiments/eval_frame_20260911/eval_frame.parquet.
Purpose: per-feature adversarial (development vs evaluation) drift scan and any later label-free scoring of evaluation pairs."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import polars as pl
import joblib
from pu_pairs.build_allpairs_fold import CONF
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import aggregate  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'eval_frame_20260911'
INF = ORIGINAL / 'outputs_v11_conditional' / 'inference-package' / 'results'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bundle = joblib.load(CONF / 'results' / 'fold1' / 'conditional-pair.joblib'); features = list(bundle['features'])
    batches = sorted((ORIGINAL / 'outputs_v2' / 'hands_evaluation').glob('batch_*.parquet'))
    parts = []
    for i, b in enumerate(batches):
        dest = OUT / f'{b.stem}-base.parquet'
        if dest.exists():
            parts.append(pl.read_parquet(dest)); continue
        raw = pl.read_parquet(b)
        names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
        events = [c for c in names if c.startswith('v2_')]
        agg = aggregate(raw, names).join(tails(raw, events), on=['pair_id', 'table_id'])
        agg.write_parquet(dest); parts.append(agg)
        print(json.dumps({'batch': b.name, 'rows': raw.height, 'pairs': agg.height, 'elapsed': round(time.time() - t0)}), flush=True)
    base = pl.concat(parts)
    ca = pl.concat([pl.read_parquet(p) for p in sorted(INF.glob('batch_*-pairs.parquet'))])
    frame = base.join(ca.drop('table_id'), on='pair_id', how='left', validate='1:1')
    missing = [f for f in features if f not in frame.columns]
    nulls = {c: int(frame[c].null_count()) for c in features if frame[c].null_count()}
    ev = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', pl.col('shared_hands').alias('sh_official'))
    chk = frame.select('pair_id', 'shared_hands').join(ev, on='pair_id', how='inner')
    info = {'pairs': frame.height, 'expected': 112540, 'missing_features': missing, 'features_with_nulls': nulls, 'shared_hands_mismatch': int((chk['shared_hands'] != chk['sh_official']).sum()),
            'ca_rows': ca.height, 'seconds': round(time.time() - t0)}
    assert frame.height == 112540 and not missing, info
    frame = frame.with_columns(pl.col([c for c in features if c in nulls]).fill_null(0)) if nulls else frame
    frame.select(['pair_id', 'table_id'] + [f for f in features if f not in ('pair_id', 'table_id')]).write_parquet(OUT / 'eval_frame.parquet')
    write_json(OUT / 'build-info.json', info)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
