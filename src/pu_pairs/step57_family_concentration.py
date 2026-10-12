"""Step 57 (label-free on evaluation; development labels for validation only): FAMILY CONCENTRATION of the 4-class pair
model as a false-positive discriminator that can act in the upper bands.
Idea: risk = 1 - p(null) says nothing about HOW the non-null mass is split over the three disclosed families. A genuine
pair of one family should put its non-null mass on that family (behaviour mAP .976 says the argmax is right); a pair that
is 'not null' because short-history noise pushed several weak signatures at once may spread its mass over families.
On full-length development the block is family-concentrated in every group (max family share median .96-.998, < .6 on
0 of 159 positives and 7 of 260 unlabelled block pairs), but the full-length development block holds almost no false
positives. This step measures the same quantity where the false positives live: (1) the 60 %-prefix frames of folds 1-2
(step 40: prefix-only-high pairs = length-induced false positives certified by the frozen full-history score < .1;
full-high = the frozen model's positives; labelled positives at prefix length), rescored with the frozen fold models;
(2) the 112,540 evaluation pairs scored with the production pair model (parity with the frozen v13 risk checked),
read by frozen band, R1 (spill-over) and tier 1 (RAW < .05).
Outputs: experiments/viewdis_20260911/{family_concentration.json, famconc_prefix_dev.parquet, famconc_eval.parquet}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
sys.path.insert(0, str(ORIGINAL / 'src'))
import numpy as np
import polars as pl
import joblib
from robust_pair_experiment import matrix  # noqa: E402
from pu_pairs.build_allpairs_fold import EXP, CONF  # noqa: E402

VD = CV / 'experiments' / 'viewdis_20260911'
SP = CV / 'experiments' / 'spillover_20260911'


def log(*a):
    print(*a, flush=True)


def band(c):
    return pl.when(pl.col(c) >= .99).then(pl.lit('[.99,1]')).when(pl.col(c) >= .9).then(pl.lit('[.9,.99)')).when(pl.col(c) >= .5).then(pl.lit('[.5,.9)')).when(pl.col(c) >= .1).then(pl.lit('[.1,.5)')).otherwise(pl.lit('<.1'))


def concentration(df):
    P = df.select('p1', 'p2', 'p3').to_numpy(); S = P.sum(1, keepdims=True); Pn = P / np.clip(S, 1e-12, None)
    ent = -(Pn * np.log(np.clip(Pn, 1e-12, 1))).sum(1) / np.log(3)
    srt = np.sort(Pn, axis=1)
    return df.with_columns(pl.Series('fam_conc', Pn.max(1)), pl.Series('fam_ent', ent), pl.Series('fam_margin', srt[:, 2] - srt[:, 1]),
                           pl.Series('fam_second', srt[:, 1]))


def score(bundle, frame):
    feats = list(bundle['features'])
    proba = bundle['model'].predict_proba(matrix(frame, feats))
    return frame.select('pair_id', 'table_id', 'shared_hands').with_columns(pl.Series('risk_s', 1 - proba[:, 0]), *[pl.Series(f'p{i}', proba[:, i]) for i in range(proba.shape[1])])


def stats(df, groups, cols=('fam_conc', 'fam_margin', 'fam_ent')):
    out = {}
    for name, mask in groups.items():
        g = df.filter(mask)
        if not g.height:
            out[name] = {'n': 0}; continue
        r = {'n': int(g.height)}
        for c in cols:
            r[c] = {'median': round(float(g[c].median()), 4), 'q10': round(float(g[c].quantile(.1)), 4), 'q25': round(float(g[c].quantile(.25)), 4)}
        for thr in (.5, .6, .7, .8, .9):
            r[f'conc<{thr}'] = round(float((g['fam_conc'] < thr).mean()), 4)
        for thr in (.3, .5, .7):
            r[f'margin<{thr}'] = round(float((g['fam_margin'] < thr).mean()), 4)
        out[name] = r
    return out


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(pos, neg, alternative='two-sided').statistic
    return round(float(u / (len(pos) * len(neg))), 4)


def main():
    t0 = time.time(); res = {}
    # ---------- (1) development prefix-60 frames, folds 1-2 ----------
    devs = []
    for fold in (1, 2):
        bundle = joblib.load(CONF / 'results' / f'fold{fold}' / 'conditional-pair.joblib')
        pref = pl.concat([pl.read_parquet(p) for p in sorted((EXP / f'fold{fold}-prefix60').glob('batch_*-pairs.parquet'))])
        sc = score(bundle, pref)
        saved = pl.read_parquet(EXP / f'fold{fold}-prefix60' / 'scores.parquet').select('pair_id', 'label', 'behavior_family', pl.col('risk').alias('risk_prefix_saved'))
        full = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet').select('pair_id', pl.col('risk').alias('risk_full'), pl.col('p1').alias('p1_full'), pl.col('p2').alias('p2_full'), pl.col('p3').alias('p3_full'))
        d = sc.join(saved, on='pair_id', how='inner').join(full, on='pair_id', how='left').with_columns(pl.lit(fold).alias('fold'))
        res[f'fold{fold}_prefix_parity_max_abs'] = round(float((d['risk_s'] - d['risk_prefix_saved']).abs().max()), 6)
        devs.append(d)
        log(f'fold{fold} prefix scored', d.height, round(time.time() - t0))
    dev = pl.concat(devs).unique(subset=['pair_id'], keep='first', maintain_order=True)
    dev = concentration(dev).with_columns(band('risk_s').alias('band_prefix'), band('risk_full').alias('band_full'))
    # full-history concentration of the same pairs, for the transition
    fullc = concentration(dev.select('pair_id', pl.col('p1_full').alias('p1'), pl.col('p2_full').alias('p2'), pl.col('p3_full').alias('p3')).drop_nulls()).select('pair_id', pl.col('fam_conc').alias('fam_conc_full'), pl.col('fam_margin').alias('fam_margin_full'))
    dev = dev.join(fullc, on='pair_id', how='left')
    hi = pl.col('risk_s') >= .5
    groups = {
        'prefix block: labelled positives': hi & (pl.col('label') == 1),
        'prefix block: labelled positives [.5,.9)': hi & (pl.col('label') == 1) & (pl.col('risk_s') < .9),
        'prefix block: labelled positives [.9,.99)': hi & (pl.col('label') == 1) & (pl.col('risk_s') >= .9) & (pl.col('risk_s') < .99),
        'prefix block: labelled positives [.99,1]': hi & (pl.col('label') == 1) & (pl.col('risk_s') >= .99),
        'prefix block: unlabelled full-high (full >= .5)': hi & (pl.col('label') == -1) & (pl.col('risk_full') >= .5),
        'prefix block: unlabelled full-high [.99,1] full': hi & (pl.col('label') == -1) & (pl.col('risk_full') >= .99),
        'prefix block: unlabelled full-mid [.1,.5)': hi & (pl.col('label') == -1) & (pl.col('risk_full') >= .1) & (pl.col('risk_full') < .5),
        'prefix block: unlabelled full-low (< .1) = length-induced FPs': hi & (pl.col('label') == -1) & (pl.col('risk_full') < .1),
        'prefix block: unlabelled full-low, prefix [.5,.9)': hi & (pl.col('label') == -1) & (pl.col('risk_full') < .1) & (pl.col('risk_s') < .9),
        'prefix block: unlabelled full-low, prefix [.9,1]': hi & (pl.col('label') == -1) & (pl.col('risk_full') < .1) & (pl.col('risk_s') >= .9),
        'prefix block: unlabelled full-high, prefix [.5,.9)': hi & (pl.col('label') == -1) & (pl.col('risk_full') >= .5) & (pl.col('risk_s') < .9),
        'prefix block: unlabelled full-high, prefix [.9,1]': hi & (pl.col('label') == -1) & (pl.col('risk_full') >= .5) & (pl.col('risk_s') >= .9),
        'prefix block: labelled negatives': hi & (pl.col('label') == 0),
        'prefix [.1,.5): labelled positives': (pl.col('risk_s') >= .1) & (pl.col('risk_s') < .5) & (pl.col('label') == 1),
        'prefix [.1,.5): unlabelled full-low': (pl.col('risk_s') >= .1) & (pl.col('risk_s') < .5) & (pl.col('label') == -1) & (pl.col('risk_full') < .1),
        'prefix [.1,.5): unlabelled full-high': (pl.col('risk_s') >= .1) & (pl.col('risk_s') < .5) & (pl.col('label') == -1) & (pl.col('risk_full') >= .5),
    }
    res['dev_prefix'] = stats(dev, groups)
    blk = dev.filter(hi)
    pos = blk.filter(pl.col('label') == 1); fph = blk.filter((pl.col('label') == -1) & (pl.col('risk_full') >= .5)); fpl = blk.filter((pl.col('label') == -1) & (pl.col('risk_full') < .1))
    res['dev_prefix_auc'] = {}
    for c in ('fam_conc', 'fam_margin', 'risk_s'):
        res['dev_prefix_auc'][c] = {'labelled_pos_vs_full_low': auc(pos[c].to_numpy(), fpl[c].to_numpy()), 'full_high_vs_full_low': auc(fph[c].to_numpy(), fpl[c].to_numpy()),
                                    'labelled_pos_vs_full_low_within_[.5,.9)': auc(pos.filter(pl.col('risk_s') < .9)[c].to_numpy(), fpl.filter(pl.col('risk_s') < .9)[c].to_numpy()),
                                    'full_high_vs_full_low_within_[.5,.9)': auc(fph.filter(pl.col('risk_s') < .9)[c].to_numpy(), fpl.filter(pl.col('risk_s') < .9)[c].to_numpy()),
                                    'full_high_vs_full_low_within_[.9,1]': auc(fph.filter(pl.col('risk_s') >= .9)[c].to_numpy(), fpl.filter(pl.col('risk_s') >= .9)[c].to_numpy())}
    # joint with risk: does concentration add to risk? logistic-free check: concentration quantile bins within the block
    res['dev_prefix_full_low_share_by_conc_bin'] = {}
    ub = blk.filter(pl.col('label') == -1).with_columns(pl.col('fam_conc').cut([.5, .6, .7, .8, .9, .95]).cast(pl.Utf8).alias('cb'))
    for (cb,), g in ub.group_by('cb', maintain_order=True):
        res['dev_prefix_full_low_share_by_conc_bin'][str(cb)] = {'n': int(g.height), 'full_low_share': round(float((g['risk_full'] < .1).mean()), 3), 'full_high_share': round(float((g['risk_full'] >= .5).mean()), 3), 'risk_prefix_median': round(float(g['risk_s'].median()), 3)}
    # transition: full-history concentration of the same pairs
    res['dev_prefix_block_conc_full_vs_prefix'] = {
        'labelled positives': {'conc_full_med': round(float(pos['fam_conc_full'].median()), 4), 'conc_prefix_med': round(float(pos['fam_conc'].median()), 4)},
        'unlabelled full-high': {'conc_full_med': round(float(fph['fam_conc_full'].median()), 4), 'conc_prefix_med': round(float(fph['fam_conc'].median()), 4)},
        'unlabelled full-low': {'conc_full_med': round(float(fpl['fam_conc_full'].median()), 4), 'conc_prefix_med': round(float(fpl['fam_conc'].median()), 4)}}
    dev.write_parquet(VD / 'famconc_prefix_dev.parquet')
    log('dev prefix done', round(time.time() - t0))
    # ---------- (2) evaluation, production model ----------
    bundle = joblib.load(ORIGINAL / 'outputs_v11_conditional' / 'pair-model.joblib')
    frame = pl.read_parquet(CV / 'experiments' / 'eval_frame_20260911' / 'eval_frame.parquet')
    ev = score(bundle, frame)
    resc = pl.read_parquet(SP / 'eval_rescoring.parquet').select('pair_id', 'risk_v13', 'dominated', 'risk_noT')
    ve = pl.read_parquet(VD / 'view_eval.parquet').select('pair_id', 'risk', 'r_RAW', 'R1', 'R4any')
    ev = ev.join(ve, on='pair_id', how='left').join(resc, on='pair_id', how='left')
    res['eval_parity_max_abs_risk'] = round(float((ev['risk_s'] - ev['risk']).abs().max()), 6)
    ev = concentration(ev).with_columns(band('risk').alias('band'))
    ev = ev.with_columns((pl.col('R1').fill_null(False)).alias('R1'), (~pl.col('R1').fill_null(False) & (pl.col('r_RAW') < .05) & (pl.col('risk') >= .5)).alias('T1'))
    eb = pl.col('risk') >= .5
    egroups = {
        'eval block': eb, 'eval block [.5,.9)': eb & (pl.col('risk') < .9), 'eval block [.9,.99)': eb & (pl.col('risk') >= .9) & (pl.col('risk') < .99), 'eval block [.99,1]': pl.col('risk') >= .99,
        'eval block R1': eb & pl.col('R1'), 'eval block T1': pl.col('T1'), 'eval block residual (not R1, not T1)': eb & ~pl.col('R1') & ~pl.col('T1'),
        'eval residual [.5,.9)': eb & ~pl.col('R1') & ~pl.col('T1') & (pl.col('risk') < .9), 'eval residual [.9,.99)': eb & ~pl.col('R1') & ~pl.col('T1') & (pl.col('risk') >= .9) & (pl.col('risk') < .99), 'eval residual [.99,1]': ~pl.col('R1') & ~pl.col('T1') & (pl.col('risk') >= .99),
        'eval residual shared_hands <= 60': eb & ~pl.col('R1') & ~pl.col('T1') & (pl.col('shared_hands') <= 60), 'eval residual shared_hands > 100': eb & ~pl.col('R1') & ~pl.col('T1') & (pl.col('shared_hands') > 100),
        'eval [.1,.5)': (pl.col('risk') >= .1) & (pl.col('risk') < .5), 'eval < .1': pl.col('risk') < .1,
    }
    res['eval'] = stats(ev, egroups)
    ev.select('pair_id', 'table_id', 'shared_hands', 'risk', 'risk_s', 'p0', 'p1', 'p2', 'p3', 'fam_conc', 'fam_ent', 'fam_margin', 'fam_second', 'band', 'R1', 'T1', 'R4any', 'r_RAW', 'dominated', 'risk_noT').write_parquet(VD / 'famconc_eval.parquet')
    res['seconds'] = round(time.time() - t0)
    write_json(VD / 'family_concentration.json', res)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
