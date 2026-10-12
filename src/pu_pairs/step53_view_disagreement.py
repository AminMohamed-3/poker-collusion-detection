"""Step 53 (label-free on evaluation; development labels only to validate): VIEW DISAGREEMENT as a false-positive
discriminator for the block. The production pair model uses 897 features = 765 raw pair statistics (base + v2 event
aggregates, tails) + 132 conditional-action (ca_) aggregates built on policy residuals. A genuine colluding pair expresses
in both views (12.x: undisclosed-high pairs match labelled positives on raw signatures and the chip ledger). A pair whose
production risk rests on one view only (raw-statistics view says null, or ca view says null) is a candidate false
positive of that view. Step 52's listing showed spill-over pairs (R1) with v13 ~.8 and raw-view score < .1.
Method: the exact production training frame (labelled pairs, full + 70 % + 40 % views, 4-class LightGBM, production
parameters) is refitted three times per fold: FULL (897, parity control vs the frozen fold model), RAW (765 non-ca), CA
(132 ca + shared_hands). Held-fold pairs (folds 1 and 2) are scored by each view; the evaluation frame is scored by the
same three views trained on all labelled pairs with the final policy's ca features.
Outputs: experiments/viewdis_20260911/{view_dev.parquet, view_eval.parquet, view_disagreement.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from pu_pairs.build_allpairs_fold import EXP, CONF, fold_pools, extra_aggregate
from pu_pairs.step43_prod_plus_pseudoneg import load_frames
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, TRAIN_VIEWS, MODEL_PARAMS, thin, aggregate, matrix  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

CV = ORIGINAL / 'Claude Version'
SP = CV / 'experiments' / 'spillover_20260911'
OUT = CV / 'experiments' / 'viewdis_20260911'
BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
FINAL_CA = ORIGINAL / 'outputs_v11_conditional' / 'policy' / 'development-hand-features.parquet'


def log(*a):
    print(*a, flush=True)


def train_frame(ca_path, held, features):
    """Production training frame: labelled pairs outside `held` tables, full + 70 % + 40 % hash-thinned views."""
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    ca = pl.read_parquet(ca_path)
    ca_features = [c for c in ca.columns if c.startswith('ca_')]
    combined = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left').with_columns(pl.col(ca_features).fill_null(0))
    assert combined.height == raw.height
    baseline_full = pl.read_parquet(ORIGINAL / 'outputs_v2' / 'pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    full = baseline_full.join(extra_aggregate(combined, ca_features), on=['pair_id', 'table_id']).sort('pair_id')
    views = []
    for fraction, seed in TRAIN_VIEWS:
        raw_sample = thin(raw, fraction, seed); added_sample = thin(combined, fraction, seed)
        view = aggregate(raw_sample, names).join(tails(raw_sample, events), on=['pair_id', 'table_id']).join(labels, on='pair_id')
        views.append(view.join(extra_aggregate(added_sample, ca_features), on=['pair_id', 'table_id']).select(full.columns).sort('pair_id'))
    held = sorted(held)
    train = pl.concat([full.filter(~pl.col('table_id').is_in(held))] + [v.filter(~pl.col('table_id').is_in(held)) for v in views])
    missing = [f for f in features if f not in train.columns]
    assert not missing, missing[:5]
    return train


def view_sets(features):
    return {'FULL': list(features), 'RAW': [f for f in features if not f.startswith('ca_')], 'CA': [f for f in features if f.startswith('ca_')] + ['shared_hands']}


def fit_views(train, features):
    y = np.array([LABELS[f] for f in train['behavior_family']])
    models = {}
    for name, cols in view_sets(features).items():
        models[name] = lgb.LGBMClassifier(**MODEL_PARAMS).fit(matrix(train, cols), y)
    return models


def score_views(models, frame, features):
    out = frame.select('pair_id')
    for name, cols in view_sets(features).items():
        p = models[name].predict_proba(matrix(frame, cols))
        out = out.with_columns(pl.Series(f'r_{name}', 1 - p[:, 0]))
    return out


def dist(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return {'n': 0}
    return {'n': int(len(x)), 'median': round(float(np.median(x)), 4), 'q10': round(float(np.quantile(x, .1)), 4), 'q25': round(float(np.quantile(x, .25)), 4),
            'share_lt_.05': round(float((x < .05).mean()), 4), 'share_lt_.1': round(float((x < .1).mean()), 4), 'share_lt_.5': round(float((x < .5).mean()), 4)}


def ap(y, s):
    o = np.argsort(-s, kind='stable'); y = y[o]
    if y.sum() == 0: return float('nan')
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / y.sum())


def analyse(df, name, flags):
    """df: pair_id, table_id, label, risk (frozen / v13), r_FULL, r_RAW, r_CA, shared_hands."""
    df = df.join(flags, on='pair_id', how='left').with_columns(pl.col('R1').fill_null(False), pl.col('dominated').fill_null(False), pl.col('R4any').fill_null(False))
    out = {'pairs': int(df.height), 'parity_FULL_vs_frozen': {'max_abs_diff': round(float((df['r_FULL'] - df['risk']).abs().max()), 5), 'spearman_block': round(float(df.filter(pl.col('risk') >= .5).select(pl.corr('r_FULL', 'risk', method='spearman')).item()), 4)}}
    blk = df.filter(pl.col('risk') >= .5)
    band = pl.when(pl.col('risk') >= .99).then(pl.lit('[.99,1]')).when(pl.col('risk') >= .9).then(pl.lit('[.9,.99)')).otherwise(pl.lit('[.5,.9)')).alias('band')
    blk = blk.with_columns(band)
    out['block'] = int(blk.height)
    for v in ('RAW', 'CA'):
        sec = {}
        for (lab,), g in blk.group_by('label'):
            sec[f'label={lab}'] = dist(g[f'r_{v}'].to_numpy())
        for (lab, b), g in blk.group_by('label', 'band'):
            sec[f'label={lab} band={b}'] = dist(g[f'r_{v}'].to_numpy())
        for (r,), g in blk.filter(pl.col('label') != 1).group_by('R1'):
            sec[f'unlabelled R1={r}'] = dist(g[f'r_{v}'].to_numpy())
        for (r,), g in blk.filter(pl.col('label') != 1).group_by('dominated'):
            sec[f'unlabelled dominated={r}'] = dist(g[f'r_{v}'].to_numpy())
        # candidate classes: block & view < t, not already R1
        for t in (.05, .1, .2, .5):
            sel = blk.filter(pl.col(f'r_{v}') < t)
            sec[f'block_and_{v}_lt_{t}'] = {'n': int(sel.height), 'positives': int((sel['label'] == 1).sum()), 'unlabelled': int((sel['label'] != 1).sum()),
                                           'in_R1': int(sel['R1'].sum()), 'not_R1': int((~sel['R1']).sum()), 'not_R1_by_band': sel.filter(~pl.col('R1')).group_by('band').len().sort('band').to_dicts()}
        out[v] = sec
    # both views low
    for t in (.1, .5):
        sel = blk.filter((pl.col('r_RAW') < t) & (pl.col('r_CA') < t))
        out[f'block_and_both_lt_{t}'] = {'n': int(sel.height), 'positives': int((sel['label'] == 1).sum()), 'in_R1': int(sel['R1'].sum())}
    # dev: labelled AP before / after demoting (view < t) block pairs to .4999
    if (df['label'] == 1).any():
        lab = df.filter(pl.col('label') >= 0)
        y = lab['label'].to_numpy().astype(float); s = lab['risk'].to_numpy().astype(float)
        res = {'before': round(ap(y, s), 5)}
        for v in ('RAW', 'CA'):
            for t in (.05, .1, .2):
                m = (lab['risk'].to_numpy() >= .5) & (lab[f'r_{v}'].to_numpy() < t)
                res[f'{v}_lt_{t}'] = round(ap(y, np.where(m, np.minimum(s, .4999), s)), 5)
        out['dev_labelled_AP_before_after'] = res
        # positives in block with low RAW: list
        out['positives_block_RAW_lt_.2'] = blk.filter((pl.col('label') == 1) & (pl.col('r_RAW') < .2)).select('pair_id', 'table_id', 'risk', 'r_RAW', 'r_CA', 'shared_hands', 'R1', 'dominated').to_dicts()
    # tail: how often is RAW high while frozen low (the reverse disagreement; missed positives?)
    tail = df.filter(pl.col('risk') < .1)
    out['tail_RAW_ge_.5'] = int((tail['r_RAW'] >= .5).sum()); out['tail_CA_ge_.5'] = int((tail['r_CA'] >= .5).sum())
    if (df['label'] == 1).any():
        out['tail_RAW_ge_.5_positives'] = int(((tail['r_RAW'] >= .5) & (tail['label'] == 1)).sum())
        out['tail_positives'] = int((tail['label'] == 1).sum())
    log(name, json.dumps({k: v for k, v in out.items() if k not in ('positives_block_RAW_lt_.2',)}, indent=None)[:6000])
    if 'positives_block_RAW_lt_.2' in out: log('positives with low RAW:', json.dumps(out['positives_block_RAW_lt_.2']))
    return out, df


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    bundle = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib'); features = list(bundle['features'])
    res = {'n_features': {k: len(v) for k, v in view_sets(features).items()}}
    # development: per fold, production frame with the fold policy, held pools scored
    devs = []
    for fold in (1, 2):
        train = train_frame(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', set(fold_pools(fold)), features)
        log(f'fold{fold} train rows', train.height, 'elapsed', round(time.time() - t0))
        models = fit_views(train, features)
        held = load_frames(f'fold{fold}', fold, features)
        sc = score_views(models, held, features)
        devs.append(held.select('pair_id', 'table_id', 'label', 'behavior_family', 'shared_hands', pl.col('risk_frozen').alias('risk')).join(sc, on='pair_id').with_columns(pl.lit(fold).alias('fold')))
        log(f'fold{fold} scored', held.height, 'elapsed', round(time.time() - t0))
    dev = pl.concat(devs).unique(subset=['pair_id'], keep='first', maintain_order=True)
    flags_dev = pl.read_parquet(SP / 'neighbour_dev_pairs.parquet').select('pair_id', 'dominated', 'R1', 'R4any').unique(subset=['pair_id'], keep='first')
    res['dev'], dev = analyse(dev, 'dev', flags_dev)
    dev.write_parquet(OUT / 'view_dev.parquet')
    # evaluation: all labelled pairs with the final policy's ca features
    train = train_frame(FINAL_CA, set(), features)
    log('full train rows', train.height, 'elapsed', round(time.time() - t0))
    models = fit_views(train, features)
    ev = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet')
    sc = score_views(models, ev, features)
    base = pl.read_csv(BASE).select('pair_id', pl.col('risk_score').alias('risk'))
    evd = ev.select('pair_id', 'table_id', 'shared_hands').join(base, on='pair_id', how='inner').with_columns(pl.lit(-1).alias('label')).join(sc, on='pair_id')
    flags_ev = pl.read_parquet(SP / 'neighbour_eval_pairs.parquet').select('pair_id', 'dominated', 'R1', 'R4any')
    res['eval'], evd = analyse(evd, 'eval', flags_ev)
    evd.write_parquet(OUT / 'view_eval.parquet')
    joblib.dump(models, OUT / 'view_models_full.joblib')
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'view_disagreement.json', res)
    log('done', res['seconds'])


if __name__ == '__main__':
    main()