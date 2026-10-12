"""Step 54 (development labels for validation only; evaluation side label-free): LENGTH RESPONSE of the three views.
Question (OBS 14.34): is 'RAW view < .1' on a block pair a property of false positives (spill-over, short-history noise)
or does the raw-statistics view simply lose genuine pairs when their histories are short? Evaluation histories are
~.75x development length (14.29) and the evaluation block is 38 % [.5,.9) pairs with median 53-64 shared hands.
Test: take the held block pairs of folds 1 and 2 (hand-level raw + ca features from step 48b), hash-thin every pair's
history to 70 / 50 / 35 % (two seeds each; the same label-independent thinning as the production training views),
re-aggregate exactly as production does and score FULL / RAW / CA (refit per fold as in step 53). Report, per group
(labelled positives; unlabelled [.99,1] = undisclosed positives by section 12; unlabelled lower-band not R1; R1), the
share with RAW < .1, FULL >= .5, CA >= .5 at each thinning. The evaluation block (909) is thinned the same way with the
all-labelled models to see how the RAW-low class grows with further shortening.
Decision rule (fixed in OBS 14.34): thinned labelled positives keep RAW >= .1 at >= 90 % down to ~40 shared hands ->
candidate D = C + RAW-low tier; > 25 % RAW < .1 -> the tier is length-confounded and dropped.
Outputs: experiments/viewdis_20260911/{length_response.json, length_response_dev.parquet, length_response_eval.parquet}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
from pu_pairs.build_allpairs_fold import CONF, fold_pools, extra_aggregate
from pu_pairs.step53_view_disagreement import train_frame, fit_views, score_views, SP, OUT
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import thin, aggregate  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

FRACTIONS = [(1.0, 0), (0.7, 11), (0.7, 12), (0.5, 21), (0.5, 22), (0.35, 31), (0.35, 32)]


def log(*a):
    print(*a, flush=True)


def band(c):
    return pl.when(pl.col(c) >= .99).then(pl.lit('[.99,1]')).when(pl.col(c) >= .9).then(pl.lit('[.9,.99)')).when(pl.col(c) >= .5).then(pl.lit('[.5,.9)')).otherwise(pl.lit('<.5'))


def thinned_scores(models, raw, ca, features):
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    events = [c for c in names if c.startswith('v2_')]
    caf = [c for c in ca.columns if c.startswith('ca_')]
    out = []
    for fraction, seed in FRACTIONS:
        r = raw if fraction == 1.0 else thin(raw, fraction, seed)
        c = ca if fraction == 1.0 else thin(ca, fraction, seed)
        assert r.select('pair_id', 'hand_id').sort('pair_id', 'hand_id').equals(c.select('pair_id', 'hand_id').sort('pair_id', 'hand_id'))
        frame = aggregate(r, names).join(tails(r, events), on=['pair_id', 'table_id']).join(extra_aggregate(c, caf), on=['pair_id', 'table_id'])
        sc = score_views(models, frame, features).join(frame.select('pair_id', pl.col('shared_hands').alias('sh_thin')), on='pair_id')
        out.append(sc.with_columns(pl.lit(fraction).alias('fraction'), pl.lit(seed).alias('seed')))
    return pl.concat(out)


def summarise(df, groups):
    res = {}
    for gname, mask in groups.items():
        sub = df.filter(mask)
        for (fraction,), g in sub.group_by('fraction', maintain_order=True):
            n_pairs = int(g['pair_id'].n_unique())
            res[f'{gname} | fraction={fraction}'] = {
                'pairs': n_pairs, 'rows': int(g.height), 'sh_median': float(g['sh_thin'].median()), 'sh_q10': float(g['sh_thin'].quantile(.1)),
                'RAW_lt_.1': round(float((g['r_RAW'] < .1).mean()), 4), 'RAW_lt_.5': round(float((g['r_RAW'] < .5).mean()), 4), 'RAW_median': round(float(g['r_RAW'].median()), 4),
                'FULL_ge_.5': round(float((g['r_FULL'] >= .5).mean()), 4), 'FULL_median': round(float(g['r_FULL'].median()), 4),
                'CA_ge_.5': round(float((g['r_CA'] >= .5).mean()), 4), 'CA_lt_.1': round(float((g['r_CA'] < .1).mean()), 4), 'CA_median': round(float(g['r_CA'].median()), 4)}
    return res


def main():
    t0 = time.time()
    bundle = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib'); features = list(bundle['features'])
    res = {}
    view_dev = pl.read_parquet(OUT / 'view_dev.parquet').select('pair_id', 'label', 'behavior_family', 'risk', 'shared_hands', 'R1', 'dominated', pl.col('r_RAW').alias('r_RAW_full_step53'))
    devs = []
    for fold in (1, 2):
        train = train_frame(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', set(fold_pools(fold)), features)
        models = fit_views(train, features)
        log(f'fold{fold} models fitted', round(time.time() - t0))
        raw = pl.read_parquet(SP / f'dev_fold{fold}_raw.parquet'); ca = pl.read_parquet(SP / f'dev_fold{fold}_ca.parquet')
        sc = thinned_scores(models, raw, ca, features).with_columns(pl.lit(fold).alias('fold'))
        devs.append(sc)
        log(f'fold{fold} thinned scores', sc.height, round(time.time() - t0))
    dev = pl.concat(devs)
    # keep each pair once (folds share 29 held tables): the fold that step 53 kept
    keep = view_dev.select('pair_id')
    dev = dev.join(keep, on='pair_id', how='inner').unique(subset=['pair_id', 'fraction', 'seed'], keep='first', maintain_order=True).join(view_dev, on='pair_id')
    dev = dev.with_columns(band('risk').alias('band'))
    # parity at fraction 1.0 with step 53 (same models refit; must be close)
    f1 = dev.filter(pl.col('fraction') == 1.0)
    res['parity_fraction1_vs_step53_RAW'] = {'max_abs_diff': round(float((f1['r_RAW'] - f1['r_RAW_full_step53']).abs().max()), 5), 'pairs': int(f1.height)}
    groups = {
        'labelled positives': pl.col('label') == 1,
        'labelled positives [.99,1]': (pl.col('label') == 1) & (pl.col('band') == '[.99,1]'),
        'labelled positives [.5,.99)': (pl.col('label') == 1) & (pl.col('band') != '[.99,1]'),
        'labelled positives shared_hands<=100': (pl.col('label') == 1) & (pl.col('shared_hands') <= 100),
        'labelled positives soft_play': (pl.col('label') == 1) & (pl.col('behavior_family') == 'soft_play'),
        'unlabelled [.99,1]': (pl.col('label') == -1) & (pl.col('band') == '[.99,1]'),
        'unlabelled [.9,.99) not R1': (pl.col('label') == -1) & (pl.col('band') == '[.9,.99)') & ~pl.col('R1'),
        'unlabelled [.5,.9) not R1': (pl.col('label') == -1) & (pl.col('band') == '[.5,.9)') & ~pl.col('R1'),
        'unlabelled R1': (pl.col('label') == -1) & pl.col('R1'),
        'unlabelled not R1 RAW_full<.1': (pl.col('label') == -1) & ~pl.col('R1') & (pl.col('r_RAW_full_step53') < .1),
        'unlabelled not R1 RAW_full>=.1 lower bands': (pl.col('label') == -1) & ~pl.col('R1') & (pl.col('r_RAW_full_step53') >= .1) & (pl.col('band') != '[.99,1]'),
    }
    res['dev'] = summarise(dev, groups)
    # transition matrix for positives: RAW>=.1 at full -> RAW<.1 at 35 %
    pos = dev.filter(pl.col('label') == 1)
    for fr in (0.7, 0.5, 0.35):
        g = pos.filter(pl.col('fraction') == fr)
        res[f'positives_RAW_lt_.1_at_{fr}_by_full_RAW'] = {'full_ge_.1': round(float((g.filter(pl.col('r_RAW_full_step53') >= .1)['r_RAW'] < .1).mean()), 4),
                                                            'full_lt_.1': round(float((g.filter(pl.col('r_RAW_full_step53') < .1)['r_RAW'] < .1).mean()), 4) if g.filter(pl.col('r_RAW_full_step53') < .1).height else None,
                                                            'sh_thin_median': float(g['sh_thin'].median())}
    dev.write_parquet(OUT / 'length_response_dev.parquet')
    log('dev done', round(time.time() - t0))
    # evaluation block: all-labelled models from step 53
    models = joblib.load(OUT / 'view_models_full.joblib')
    raw = pl.read_parquet(SP / 'eval_block_raw.parquet'); ca = pl.read_parquet(SP / 'eval_block_ca.parquet')
    view_ev = pl.read_parquet(OUT / 'view_eval.parquet').select('pair_id', 'risk', 'shared_hands', 'R1', 'dominated', pl.col('r_RAW').alias('r_RAW_full_step53'))
    ev = thinned_scores(models, raw, ca, features).join(view_ev, on='pair_id').with_columns(band('risk').alias('band'))
    f1 = ev.filter(pl.col('fraction') == 1.0)
    res['parity_eval_fraction1_vs_step53_RAW'] = {'max_abs_diff': round(float((f1['r_RAW'] - f1['r_RAW_full_step53']).abs().max()), 5), 'pairs': int(f1.height)}
    egroups = {
        'eval block': pl.lit(True),
        'eval [.99,1]': pl.col('band') == '[.99,1]',
        'eval [.9,.99) not R1': (pl.col('band') == '[.9,.99)') & ~pl.col('R1'),
        'eval [.5,.9) not R1': (pl.col('band') == '[.5,.9)') & ~pl.col('R1'),
        'eval R1': pl.col('R1'),
        'eval not R1 RAW_full<.1': ~pl.col('R1') & (pl.col('r_RAW_full_step53') < .1),
        'eval not R1 RAW_full>=.1 lower bands': ~pl.col('R1') & (pl.col('r_RAW_full_step53') >= .1) & (pl.col('band') != '[.99,1]'),
        'eval [.99,1] shared_hands<=60': (pl.col('band') == '[.99,1]') & (pl.col('shared_hands') <= 60),
    }
    res['eval'] = summarise(ev, egroups)
    ev.write_parquet(OUT / 'length_response_eval.parquet')
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'length_response.json', res)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
