"""Step 164 pre-check (Claude): label-free size check for idea 3 (length-matched training weights on the tail views), before any registration.
Builds the evaluation-mode training frames of draw k = 0 exactly as step159a (SXN3), then compares the shared_hands distribution of each view's
training rows (all rows, and labelled positives) with the evaluation pairs (all, tail risk < .5 of the STEP163 file, block >= .5).
Reports the importance-weight spread w = p_eval(bin) / p_train(bin) and its effective-sample-size ratio. No fits, no labels of evaluation pairs.
Output: experiments/length_weights_20260915/length_diag164.json"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV
import numpy as np, polars as pl, joblib
from pu_pairs.step53_view_disagreement import FINAL_CA
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded
from pu_pairs import step156b_C_SXN_views as N
from pu_pairs import step64a_tail_aggregate_views as S64

OUT = CV / 'experiments' / 'length_weights_20260915'
EDGES = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 170, 200, 250, 10 ** 6]


def hist(x):
    h, _ = np.histogram(np.asarray(x, dtype=float), bins=EDGES); return h / max(h.sum(), 1)


def q(x):
    return {str(p): float(np.percentile(np.asarray(x, dtype=float), p)) for p in (10, 25, 50, 75, 90)}


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); dest = OUT / 'length_diag164.json'; assert not dest.exists()
    N.RATIO = 8.7
    features = list(joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')['features'])
    c1, c2 = N.candidates(1, set()), N.candidates(2, set())
    cand = pl.concat([c1, c2.join(c1, on='pair_id', how='anti')])
    fr = view_frames_seeded(FINAL_CA, set(), features, 0)
    add, info = N.added_rows(cand, fr[1.0].height, 159900, features)
    ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet', columns=['pair_id', 'shared_hands'])
    sub = pl.read_csv(CV / 'outputs_step163_20260915' / 'submission_step163.csv', columns=['pair_id', 'risk_score'], schema_overrides={'risk_score': pl.Float64})
    ev = ev.join(sub, on='pair_id', validate='1:1')
    E = {'eval_all': ev['shared_hands'], 'eval_tail': ev.filter(pl.col('risk_score') < .5)['shared_hands'], 'eval_block': ev.filter(pl.col('risk_score') >= .5)['shared_hands']}
    res = {'added_info': info, 'edges': EDGES, 'eval': {k: {'n': len(v), 'q': q(v), 'hist': hist(v).round(4).tolist()} for k, v in E.items()}, 'views': {}}
    cols = ['pair_id', 'label', 'behavior_family', 'shared_hands']
    for name in ('SHORT3', 'ALL5', 'PROD'):
        tr = pl.concat([fr[f].select(cols) for f in S64.TRAIN_SETS[name]] + [add[L].select(cols) for L in N.ADD[name]], how='vertical_relaxed')
        pos = tr.filter(pl.col('behavior_family') != 'none')
        r = {'n': tr.height, 'n_pos': pos.height, 'q_all': q(tr['shared_hands']), 'q_pos': q(pos['shared_hands']),
             'hist_all': hist(tr['shared_hands']).round(4).tolist(), 'hist_pos': hist(pos['shared_hands']).round(4).tolist()}
        for ek in ('eval_all', 'eval_tail', 'eval_block'):
            pe = hist(E[ek])
            for which, x in (('all', tr['shared_hands']), ('pos', pos['shared_hands'])):
                pt = hist(x); b = np.digitize(np.asarray(x, dtype=float), EDGES[1:-1])
                w = np.where(pt[b] > 0, pe[b] / np.maximum(pt[b], 1e-12), 0.0)
                ess = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12) / len(w))
                r[f'w_{which}_vs_{ek}'] = {'ess_ratio': round(ess, 4), 'w_q10_50_90': [round(float(np.percentile(w, p)), 3) for p in (10, 50, 90)],
                                           'tv_distance': round(float(.5 * np.abs(pe - pt).sum()), 4)}
        res['views'][name] = r
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(dest, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
