"""Step 64e: final (all-labelled) fits of the length-matched views SHORT3 and ALL5 (step 64a) and their scores on the
evaluation frame, so that a candidate can use them without refitting. Same recipe as step 60 (3): training frame = all
1,860 labelled pairs with the final policy's ca features (FINAL_CA), thinned views per step 64a. Label-free on evaluation.
Outputs: experiments/viewdis_20260911/{tail_views_eval_frame.parquet, tail_views_models_full.joblib}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
import joblib
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs.step64a_tail_aggregate_views import view_frames, fit_multiclass, TRAIN_SETS, VD
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import matrix  # noqa: E402


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    bundle = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib'); features = list(bundle['features'])
    frames = view_frames(FINAL_CA, set(), features)
    log('frames', {str(k): v.height for k, v in frames.items()}, round(time.time() - t0))
    ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=list(dict.fromkeys(['pair_id', 'table_id', 'shared_hands'] + features)))
    out = ev.select('pair_id'); models = {}
    for name in ('SHORT3', 'ALL5'):
        train = pl.concat([frames[f] for f in TRAIN_SETS[name]]); m = fit_multiclass(train, features); models[name] = m
        out = out.with_columns(pl.Series(f'r_{name}', 1 - m.predict_proba(matrix(ev, features))[:, 0]))
        log(name, 'fitted on', train.height, 'rows; eval scored', round(time.time() - t0))
    out.write_parquet(VD / 'tail_views_eval_frame.parquet'); joblib.dump(models, VD / 'tail_views_models_full.joblib')
    ve = pl.read_parquet(VD / 'multiview_eval.parquet').select('pair_id', 'risk').join(out, on='pair_id')
    blk = ve.filter(pl.col('risk') >= .5)
    res = {'eval_rows': out.height, 'block': blk.height, 'block_median': {n: round(float(blk[f'r_{n}'].median()), 3) for n in ('SHORT3', 'ALL5')},
           'block_share_lt_.1': {n: round(float((blk[f'r_{n}'] < .1).mean()), 3) for n in ('SHORT3', 'ALL5')},
           'tail_ge_.5_count': {n: int((ve.filter(pl.col('risk') < .5)[f'r_{n}'] >= .5).sum()) for n in ('SHORT3', 'ALL5')},
           'tail_ge_.9_count': {n: int((ve.filter(pl.col('risk') < .5)[f'r_{n}'] >= .9).sum()) for n in ('SHORT3', 'ALL5')}, 'seconds': round(time.time() - t0)}
    write_json(VD / 'tail_views_eval_frame.json', res); log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
