"""Step 70: fold-0 views (P4-C, APPROVED 2026-09-13). Measurement only, no lever fitted.

Step-53 recipe on fold-0: training frame = production frame (labelled pairs, full +
70 % + 40 % hash-thinned views) from reports/conditional-action-experiment/policy-screen/hand-features.parquet
EXCLUDING the fold-0 held pools (reports/tabicl-screen/held-pairs.csv, 100 tables).
Refit FULL (897) + RAW (765 non-ca) with MODEL_PARAMS (seed 7329); score the eligible
prefix pairs (prefix shared_hands>=38) from step69 fold0-prefix60; R1_prefix via
dominance + noT rescoring with the frozen fold-0 bundle (step48/step61 pattern).

Persists (Claude Version/experiments/pu_pairs_20260911/fold0-prefix60/):
  view_FULL.joblib, view_RAW.joblib, fold0_views.parquet, views_receipt.json
CPU only, 2 threads, utf-8.
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
import sys, gc, json, time, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from pu_pairs.build_allpairs_fold import EXP, sha
from pu_pairs.step69_fold0_prefix import fold0_pools, caf_columns, POLICY, HF, STYLE, BUNDLE, DEFAULT_OUT
from pu_pairs.step53_view_disagreement import train_frame
from pu_pairs.step48_spillover_rescoring import dominance, make_scorer
from pu_pairs.step69_fold0_prefix import base_features, v2_features, conditional_features
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, MODEL_PARAMS, matrix  # noqa: E402

THR = 0.5
MIN_SH = 38


def log(*a):
    print(*a, flush=True)


def refit(out, feats, t0):
    pools = set(fold0_pools())
    train = train_frame(HF, pools, feats)
    log(json.dumps({'train_rows': train.height, 'seconds': round(time.time() - t0)}))
    y = np.array([LABELS[f] for f in train['behavior_family']])
    models = {}
    for name, cols in (('FULL', list(feats)), ('RAW', [f for f in feats if not f.startswith('ca_')])):
        models[name] = lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(train, cols), y)
        joblib.dump(models[name], out / f'view_{name}.joblib')
        log(json.dumps({'fitted': name, 'n_cols': len(cols), 'seconds': round(time.time() - t0)}))
    return models, train.height


def score_prefix(out, models, feats, t0):
    pref = pl.concat([pl.read_parquet(p) for p in sorted(out.glob('batch_*-pairs.parquet'))]).sort('pair_id')
    saved = pl.read_parquet(out / 'scores.parquet').select(
        'pair_id', 'player_1', 'player_2', 'table_id', 'label', 'behavior_family',
        'shared_hands', pl.col('risk').alias('risk_prefix'))
    cols_full = list(feats)
    cols_raw = [f for f in feats if not f.startswith('ca_')]
    p_full = models['FULL'].predict_proba(matrix(pref, cols_full))
    p_raw = models['RAW'].predict_proba(matrix(pref, cols_raw))
    sc = pref.select('pair_id').with_columns(
        pl.Series('r_FULL', 1 - p_full[:, 0]), pl.Series('r_RAW', 1 - p_raw[:, 0]))
    d = saved.join(sc, on='pair_id', how='left', validate='1:1')
    log(json.dumps({'scored': d.height, 'eligible_prefix38': int((d['shared_hands'] >= MIN_SH).sum()),
                    'seconds': round(time.time() - t0)}))
    return d


def block_frames(out, block, t0):
    """Per-hand raw+ca frames for block pairs (step61.build_frames pattern, fold-0 assets)."""
    fr, fc = out / 'prefix_fold0_raw.parquet', out / 'prefix_fold0_ca.parquet'
    if fr.exists() and fc.exists():
        raw, ca = pl.read_parquet(fr), pl.read_parquet(fc)
        if set(raw['pair_id']) == set(block['pair_id']):
            log(json.dumps({'block_frames': 'cached', 'pair_hands': raw.height}))
            return raw, ca
    policy = joblib.load(POLICY)
    caf = caf_columns()
    counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
    tables = sorted(set(block['table_id']))
    hd = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter(
        (pl.col('phase') == 'development') & pl.col('table_id').is_in(tables))
    hd = hd.sort('table_id', 'started_at').with_columns(
        (pl.int_range(pl.len()).over('table_id') / pl.len().over('table_id')).alias('_pos'))
    hands_all = hd.filter(pl.col('_pos') < 0.6).drop('_pos')
    raws, cas = [], []
    for i, tb in enumerate(tables):
        h = hands_all.filter(pl.col('table_id') == tb)
        p = block.filter(pl.col('table_id') == tb).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p)
        raw = v2_features(raw, p)
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(
            p.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        ca, _, _ = conditional_features(
            keys, h.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind'),
            policy, counts, caf)
        raws.append(raw)
        cas.append(ca)
        gc.collect()
        if i % 20 == 0:
            log(json.dumps({'block_table': i, 'of': len(tables), 'seconds': round(time.time() - t0)}))
    raw = pl.concat(raws, how='vertical_relaxed').sort('pair_id', 'hand_id')
    ca = pl.concat(cas, how='vertical_relaxed').sort('pair_id', 'hand_id')
    raw.write_parquet(fr)
    ca.write_parquet(fc)
    log(json.dumps({'block_frames': 'built', 'pair_hands': raw.height, 'seconds': round(time.time() - t0)}))
    return raw, ca


def r1_flags(out, d, bundle, feats, t0):
    block = d.filter(pl.col('risk_prefix') >= THR).select(
        'pair_id', 'player_1', 'player_2', 'table_id', pl.col('risk_prefix').alias('risk')).sort('pair_id')
    dom = dominance(block)
    raw, ca = block_frames(out, block, t0)
    assert raw.select('pair_id', 'hand_id').equals(ca.select('pair_id', 'hand_id'))
    caf = [c for c in ca.columns if c.startswith('ca_')]
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    events = [c for c in names if c.startswith('v2_')]
    score = make_scorer(bundle['model'], list(feats), names, events, caf)
    ids = block['pair_id'].to_list()
    hands = sorted(set(raw['hand_id']))
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet', columns=['hand_id', 'player_id']).filter(
        pl.col('hand_id').is_in(hands))
    dealt = defaultdict(set)
    for h_, p_ in seats.iter_rows():
        dealt[h_].add(p_)
    hands_of = dict(raw.group_by('pair_id').agg(pl.col('hand_id')).iter_rows())
    T_of = {}
    for pid, p1, p2 in block.select('pair_id', 'player_1', 'player_2').iter_rows():
        if pid in dom:
            T_of[pid] = dom[pid][2]
        else:
            cnt = defaultdict(int)
            for h_ in hands_of[pid]:
                for q in dealt[h_] - {p1, p2}:
                    cnt[q] += 1
            T_of[pid] = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    pairs_arr = raw['pair_id'].to_list()
    hands_arr = raw['hand_id'].to_list()
    isT = np.array([T_of[p] in dealt[h] for p, h in zip(pairs_arr, hands_arr)])
    noT, _ = score(raw.filter(pl.Series(~isT)), ca.filter(pl.Series(~isT)))
    rows = [{'pair_id': p, 'dominated': p in dom,
             'risk_noT': float(noT.get(p, float('nan')))} for p in ids]
    r1 = pl.DataFrame(rows).with_columns(
        (pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < 0.1)).alias('R1_prefix'))
    log(json.dumps({'block': len(ids), 'dominated': len(dom), 'R1_prefix': int(r1['R1_prefix'].sum()),
                    'seconds': round(time.time() - t0)}))
    return r1


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='fold0-prefix60')
    args = ap.parse_args()
    out = EXP / args.out
    bundle = joblib.load(BUNDLE)
    feats = list(bundle['features'])
    res = {'step': 'P4C_fold0_views', 'n_features': {'FULL': len(feats), 'RAW': len([f for f in feats if not f.startswith("ca_")])},
           'code_sha256': sha(ORIGINAL / 'Claude Version' / 'src' / 'pu_pairs' / 'step70_fold0_views.py'),
           'policy_sha256': sha(POLICY), 'bundle_sha256': sha(BUNDLE), 'held_sha256': sha(ORIGINAL / 'reports' / 'tabicl-screen' / 'held-pairs.csv')}
    models, n_train = refit(out, feats, t0)
    res['train_rows'] = n_train
    d = score_prefix(out, models, feats, t0)
    res['parity_r_FULL_vs_frozen_prefix'] = {
        'max_abs': round(float((d['r_FULL'] - d['risk_prefix']).abs().max()), 5)}
    r1 = r1_flags(out, d, bundle, feats, t0)
    d = d.join(r1, on='pair_id', how='left')
    d.write_parquet(out / 'fold0_views_noFull.parquet')
    res['pairs'] = d.height
    res['eligible_prefix38'] = int((d['shared_hands'] >= MIN_SH).sum())
    res['seconds'] = round(time.time() - t0)
    write_json(out / 'views_receipt.json', res)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
