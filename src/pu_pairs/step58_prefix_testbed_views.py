"""Step 58 (development labels for validation only): the 60 %-PREFIX TESTBED as a validation set for the block demotion tiers.
Step 40 built, for folds 1 and 2, the 897-feature frames of every held-pool pair on the first 60 % of each table's
development hands, and certified each pair by its frozen FULL-history risk: 'full-low' (< .1) prefix-block pairs are
length-induced false positives, 'full-high' (>= .5) are the frozen model's positives (mostly undisclosed colluders,
section 12), and the labelled positives are known. That is a far better testbed for a false-positive demotion rule than
the full-length development block (which holds almost no false positives): the prefix block of folds 1-2 holds 335
certified false positives against 146 labelled + 152 full-high pairs, at evaluation-like history length.
This step scores the prefix frames with the per-fold FULL / RAW / CA view refits of step 53 (same recipe, held pools
excluded) and asks: what share of the certified false positives has RAW < .05 / .1 / .5, what share of the positives,
and what does the RAW-low tier of candidate D do to the block's ranking on this testbed (AP with full-low as negatives,
labelled + full-high as positives; PU caveat: full-mid pairs [.1,.5) are excluded as undetermined)?
Outputs: experiments/viewdis_20260911/{prefix_testbed_views.json, prefix_testbed_views.parquet}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
import joblib
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs.step53_view_disagreement import train_frame, fit_views, score_views, OUT

VD = CV / 'experiments' / 'viewdis_20260911'


def log(*a):
    print(*a, flush=True)


def ap(y, s):
    o = np.argsort(-s, kind='stable'); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def main():
    t0 = time.time()
    bundle = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib'); features = list(bundle['features'])
    parts = []
    for fold in (1, 2):
        train = train_frame(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', set(fold_pools(fold)), features)
        models = fit_views(train, features)
        log(f'fold{fold} views fitted', round(time.time() - t0))
        pref = pl.concat([pl.read_parquet(p) for p in sorted((EXP / f'fold{fold}-prefix60').glob('batch_*-pairs.parquet'))])
        saved = pl.read_parquet(EXP / f'fold{fold}-prefix60' / 'scores.parquet').select('pair_id', 'label', 'behavior_family', pl.col('risk').alias('risk_prefix'))
        full = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet').select('pair_id', pl.col('risk').alias('risk_full'))
        sc = score_views(models, pref, features).join(pref.select('pair_id', 'table_id', 'shared_hands'), on='pair_id').join(saved, on='pair_id').join(full, on='pair_id', how='left')
        parts.append(sc.with_columns(pl.lit(fold).alias('fold')))
        log(f'fold{fold} prefix scored', sc.height, round(time.time() - t0))
    d = pl.concat(parts).unique(subset=['pair_id'], keep='first', maintain_order=True)
    d = d.with_columns(pl.when(pl.col('label') == 1).then(pl.lit('pos')).when(pl.col('label') == 0).then(pl.lit('neg'))
                       .when(pl.col('risk_full') >= .5).then(pl.lit('unl_full_high')).when(pl.col('risk_full') >= .1).then(pl.lit('unl_full_mid')).otherwise(pl.lit('unl_full_low')).alias('cls'))
    res = {'parity_r_FULL_vs_frozen_prefix': {'max_abs': round(float((d['r_FULL'] - d['risk_prefix']).abs().max()), 4),
                                               'spearman_block': None}}
    blk = d.filter(pl.col('risk_prefix') >= .5)
    from scipy.stats import spearmanr
    res['parity_r_FULL_vs_frozen_prefix']['spearman_block'] = round(float(spearmanr(blk['r_FULL'], blk['risk_prefix']).correlation), 4)
    bandp = pl.when(pl.col('risk_prefix') >= .99).then(pl.lit('[.99,1]')).when(pl.col('risk_prefix') >= .9).then(pl.lit('[.9,.99)')).when(pl.col('risk_prefix') >= .5).then(pl.lit('[.5,.9)')).when(pl.col('risk_prefix') >= .1).then(pl.lit('[.1,.5)')).otherwise(pl.lit('<.1'))
    d = d.with_columns(bandp.alias('band')); blk = d.filter(pl.col('risk_prefix') >= .5)
    # shares by class and band
    tab = {}
    for (cls, band), g in blk.group_by('cls', 'band', maintain_order=True):
        tab[f'{cls} | {band}'] = {'n': int(g.height), 'sh_median': float(g['shared_hands'].median()),
                                  'RAW<.05': round(float((g['r_RAW'] < .05).mean()), 3), 'RAW<.1': round(float((g['r_RAW'] < .1).mean()), 3), 'RAW<.5': round(float((g['r_RAW'] < .5).mean()), 3),
                                  'CA<.1': round(float((g['r_CA'] < .1).mean()), 3), 'CA<.5': round(float((g['r_CA'] < .5).mean()), 3), 'RAW_median': round(float(g['r_RAW'].median()), 3), 'CA_median': round(float(g['r_CA'].median()), 3)}
    for cls, g in blk.group_by('cls', maintain_order=True):
        cls = cls[0]
        tab[f'{cls} | block'] = {'n': int(g.height), 'sh_median': float(g['shared_hands'].median()),
                                 'RAW<.05': round(float((g['r_RAW'] < .05).mean()), 3), 'RAW<.1': round(float((g['r_RAW'] < .1).mean()), 3), 'RAW<.5': round(float((g['r_RAW'] < .5).mean()), 3),
                                 'CA<.1': round(float((g['r_CA'] < .1).mean()), 3), 'CA<.5': round(float((g['r_CA'] < .5).mean()), 3), 'RAW_median': round(float(g['r_RAW'].median()), 3), 'CA_median': round(float(g['r_CA'].median()), 3)}
    res['prefix_block_by_class_band'] = dict(sorted(tab.items()))
    # by shared hands, full-low vs positives
    sh = {}
    for cls in ('pos', 'unl_full_high', 'unl_full_low'):
        g = blk.filter(pl.col('cls') == cls).with_columns(pl.col('shared_hands').cut([50, 70, 100, 140]).cast(pl.Utf8).alias('shb'))
        sh[cls] = {str(k[0]): {'n': int(x.height), 'RAW<.05': round(float((x['r_RAW'] < .05).mean()), 3), 'RAW<.1': round(float((x['r_RAW'] < .1).mean()), 3)} for k, x in g.group_by('shb', maintain_order=True)}
    res['prefix_block_RAWlow_by_shared_hands'] = sh
    # ranking test on the certified subset of the block (full-mid excluded): positives = labelled pos + full-high, negatives = full-low
    cert = blk.filter(pl.col('cls').is_in(['pos', 'unl_full_high', 'unl_full_low']))
    y = (cert['cls'] != 'unl_full_low').to_numpy().astype(float)
    risk = cert['risk_prefix'].to_numpy(); raw = cert['r_RAW'].to_numpy(); ca = cert['r_CA'].to_numpy()
    def tiered(base, mask, key):
        s = base.copy(); s[mask] = .49 + .0099 * (key[mask] - key[mask].min()) / max(1e-9, key[mask].max() - key[mask].min()); return s
    orders = {'frozen prefix risk': risk, 'RAW alone': raw, 'CA alone': ca, 'logit(FULL)+logit(RAW)': np.log(np.clip(risk, 1e-6, 1 - 1e-6) / (1 - np.clip(risk, 1e-6, 1 - 1e-6))) + np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6))),
              'min(FULL,RAW)': np.minimum(risk, raw)}
    for thr in (.05, .1, .2, .5):
        orders[f'tier RAW<{thr} to bottom (by RAW)'] = tiered(risk, raw < thr, raw)
    res['certified_block_AP'] = {'n_pos': int(y.sum()), 'n_neg': int((1 - y).sum())}
    for k, s in orders.items():
        res['certified_block_AP'][k] = round(ap(y, s), 4)
    # same, labelled positives only as positives (full-high excluded), to remove the teacher's own bias
    cert2 = blk.filter(pl.col('cls').is_in(['pos', 'unl_full_low']))
    y2 = (cert2['cls'] == 'pos').to_numpy().astype(float); risk2 = cert2['risk_prefix'].to_numpy(); raw2 = cert2['r_RAW'].to_numpy()
    res['labelled_vs_full_low_AP'] = {'n_pos': int(y2.sum()), 'n_neg': int((1 - y2).sum()), 'frozen prefix risk': round(ap(y2, risk2), 4), 'RAW alone': round(ap(y2, raw2), 4)}
    for thr in (.05, .1, .2, .5):
        res['labelled_vs_full_low_AP'][f'tier RAW<{thr}'] = round(ap(y2, tiered(risk2, raw2 < thr, raw2)), 4)
    # what the tier moves: counts by class at each threshold
    res['tier_counts'] = {}
    for thr in (.05, .1, .2, .5):
        m = blk.filter(pl.col('r_RAW') < thr)
        # fixed at step 59: iter_rows yields (cls, len) row tuples, so the key is k, not k[0] (the original wrote first letters)
        res['tier_counts'][str(thr)] = {str(k): int(v) for k, v in m.group_by('cls').len().iter_rows()} if m.height else {}
        res['tier_counts'][str(thr)]['total'] = int(m.height)
    # tail: does RAW find fallen positives in prefix [.1,.5)?
    tail = d.filter((pl.col('risk_prefix') >= .1) & (pl.col('risk_prefix') < .5))
    res['prefix_tail_[.1,.5)'] = {str(k[0]): {'n': int(g.height), 'RAW>=.5': round(float((g['r_RAW'] >= .5).mean()), 3), 'RAW>=.9': round(float((g['r_RAW'] >= .9).mean()), 3), 'CA>=.5': round(float((g['r_CA'] >= .5).mean()), 3)} for k, g in tail.group_by('cls', maintain_order=True)}
    res['seconds'] = round(time.time() - t0)
    d.write_parquet(VD / 'prefix_testbed_views.parquet')
    write_json(VD / 'prefix_testbed_views.json', res)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
