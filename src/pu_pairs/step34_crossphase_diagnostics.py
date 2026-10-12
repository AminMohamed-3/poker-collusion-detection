"""Step 34: label-free diagnostics of the step-32 delta model before any candidate is considered for upload.

Question: the evaluation tail (v13 risk < .1) has 2,982 / 110,338 pairs with delta score >= .5 and 645 >= .9, while
labelled negatives (OOF) have q99 = .57 and q999 = .93. Roughly a 3-5x excess. Three label-free explanations are
tested here (none uses outer labels or the leaderboard):
  A. direction drift: training deltas are dev - eval, application deltas are eval - dev. If the simulator drifts
     between phases, clean pairs carry an offset mu in training and -mu in application. Test: OOF-score the
     REVERSED delta (eval - dev) of held-out labelled negatives (clean in both phases). If their >= .5 / >= .9
     rates match the evaluation tail rather than the forward-direction OOF, the excess is drift.
  B. per-feature phase drift: paired t statistic of (dev - eval) over labelled negatives, per feature.
  C. noise: delta noise scales with 1 / sqrt(shared hands). If high-delta tail pairs concentrate at low
     shared-hands counts (either phase), the excess is noise, not signal. Also per-table clustering of high-delta
     tail pairs versus a Poisson expectation.
  D. training-vs-application shared-hands distributions.
Outputs: experiments/crossphase_20260911/diagnostics.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
from pu_pairs.step32_crossphase_delta_model import EXP, PARAMS, feature_names, delta, X
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


def q(a, qs=(.5, .9, .99, .999)):
    a = np.asarray(a, dtype=float)
    return [float(v) for v in np.quantile(a, qs)] if len(a) else []


def main():
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    lab_dev = pl.read_parquet(EXP / 'lab_on_dev.parquet')
    lab_ev = pl.read_parquet(EXP / 'lab_on_eval.parquet')
    feats = feature_names(lab_dev)
    dcols = ['d_' + c for c in feats]
    fwd = delta(lab_dev, lab_ev, feats)
    rev = delta(lab_ev, lab_dev, feats)  # application direction for the labelled pairs
    tr = lab_dev.select('pair_id', 'table_id', pl.col('shared_hands').alias('sh_dev')).join(lab_ev.select('pair_id', pl.col('shared_hands').alias('sh_eval')), on='pair_id') \
        .join(fwd, on='pair_id').join(lab, on='pair_id').sort('pair_id')
    rv = tr.select('pair_id').join(rev, on='pair_id', how='left')
    assert rv['pair_id'].equals(tr['pair_id'])
    y = (tr['label'] == 1).to_numpy().astype(int)
    groups = tr['table_id'].to_numpy()
    oof_f = np.zeros(len(y)); oof_r = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=5).split(tr, y, groups):
        m = lgb.LGBMClassifier(**PARAMS).fit(X(tr[a], dcols), y[a])
        oof_f[b] = m.predict_proba(X(tr[b], dcols))[:, 1]
        oof_r[b] = m.predict_proba(X(rv[b], dcols))[:, 1]
    neg = y == 0; pos = y == 1
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'n_train': int(len(y)), 'n_pos': int(pos.sum()),
           'A_direction': {'forward_neg_q': q(oof_f[neg]), 'reversed_neg_q': q(oof_r[neg]),
                           'forward_neg_rate_ge_0.5': float((oof_f[neg] >= .5).mean()), 'reversed_neg_rate_ge_0.5': float((oof_r[neg] >= .5).mean()),
                           'forward_neg_rate_ge_0.9': float((oof_f[neg] >= .9).mean()), 'reversed_neg_rate_ge_0.9': float((oof_r[neg] >= .9).mean()),
                           'forward_pos_q': q(oof_f[pos], (.01, .1, .5)), 'reversed_pos_q': q(oof_r[pos], (.5, .9, .99)),
                           'eval_lo_rate_ge_0.5_reference': 2982 / 110338, 'eval_lo_rate_ge_0.9_reference': 645 / 110338}}
    print(json.dumps(res['A_direction']), flush=True)
    # B. per-feature paired drift over labelled negatives
    D = tr.filter(pl.col('label') == 0).select(dcols).to_numpy().astype(float)
    D = np.where(np.isfinite(D), D, np.nan)
    mu = np.nanmean(D, axis=0); sd = np.nanstd(D, axis=0) + 1e-12; n = np.sum(np.isfinite(D), axis=0)
    t = mu / (sd / np.sqrt(np.maximum(n, 1)))
    order = np.argsort(-np.abs(t))
    res['B_feature_drift'] = {'n_features': len(dcols), 'n_abs_t_gt_4': int((np.abs(t) > 4).sum()), 'n_abs_t_gt_8': int((np.abs(t) > 8).sum()),
                              'top': [{'feature': dcols[i], 't': float(t[i]), 'mean_over_sd': float(mu[i] / sd[i])} for i in order[:15]]}
    print(json.dumps({k: v for k, v in res['B_feature_drift'].items() if k != 'top'}), flush=True)
    print(json.dumps(res['B_feature_drift']['top'][:8]), flush=True)
    # C. evaluation tail: shared hands and clustering of high-delta pairs
    ev = pl.read_parquet(EXP / 'eval-scores.parquet')
    ev_t = pl.read_parquet(EXP / 'eval_on_eval.parquet').select('pair_id', 'table_id', pl.col('shared_hands').alias('sh_eval'))
    ev_b = pl.read_parquet(EXP / 'eval_on_dev.parquet').select('pair_id', pl.col('shared_hands').alias('sh_dev'))
    ev = ev.join(ev_t, on='pair_id', how='left').join(ev_b, on='pair_id', how='left')
    lo = ev.filter((pl.col('risk') < .1) & pl.col('delta_score').is_not_null())
    hi_d = lo.filter(pl.col('delta_score') >= .9)
    mid_d = lo.filter((pl.col('delta_score') >= .5) & (pl.col('delta_score') < .9))
    rest = lo.filter(pl.col('delta_score') < .5)
    def sh(frame):
        return {'n': frame.height, 'sh_eval_q': q(frame['sh_eval'].drop_nulls().to_numpy(), (.05, .25, .5, .75, .95)), 'sh_dev_q': q(frame['sh_dev'].drop_nulls().to_numpy(), (.05, .25, .5, .75, .95))}
    per_table = hi_d.group_by('table_id').len().sort('len', descending=True)
    lam = hi_d.height / 400
    res['C_tail'] = {'delta_ge_0.9': sh(hi_d), 'delta_0.5_to_0.9': sh(mid_d), 'delta_lt_0.5': sh(rest),
                     'hi_delta_tables_with_any': per_table.height, 'hi_delta_table_max': int(per_table['len'][0]) if per_table.height else 0,
                     'hi_delta_table_top10': per_table.head(10).to_dicts(), 'poisson_lambda_per_table': lam,
                     'hi_delta_within_same_table_as_block_pair': None}
    # do high-delta tail pairs share a player with a block pair (v13 >= .5)? (ring / spill-over check)
    pairs = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2')
    blk = pairs.join(ev.filter(pl.col('risk') >= .5).select('pair_id'), on='pair_id')
    blk_players = set(blk['player_1']) | set(blk['player_2'])
    hp = pairs.join(hi_d.select('pair_id'), on='pair_id')
    share = hp.filter(pl.col('player_1').is_in(list(blk_players)) | pl.col('player_2').is_in(list(blk_players))).height
    lp = pairs.join(lo.select('pair_id'), on='pair_id')
    share_all = lp.filter(pl.col('player_1').is_in(list(blk_players)) | pl.col('player_2').is_in(list(blk_players))).height
    res['C_tail']['hi_delta_sharing_player_with_block'] = {'n': share, 'rate': share / max(hi_d.height, 1), 'lo_baseline_rate': share_all / max(lo.height, 1)}
    print(json.dumps({k: v for k, v in res['C_tail'].items() if k != 'hi_delta_table_top10'}), flush=True)
    # D. shared-hands distributions, training vs application
    res['D_shared_hands'] = {'train_sh_dev_q': q(tr['sh_dev'].to_numpy(), (.05, .25, .5, .75, .95)), 'train_sh_eval_q': q(tr['sh_eval'].to_numpy(), (.05, .25, .5, .75, .95)),
                             'eval_sh_eval_q': q(ev['sh_eval'].drop_nulls().to_numpy(), (.05, .25, .5, .75, .95)), 'eval_sh_dev_q': q(ev['sh_dev'].drop_nulls().to_numpy(), (.05, .25, .5, .75, .95)),
                             'train_pos_sh_dev_q': q(tr.filter(pl.col('label') == 1)['sh_dev'].to_numpy(), (.05, .25, .5, .75, .95))}
    print(json.dumps(res['D_shared_hands']), flush=True)
    # E. how the two candidate orderings would move the tail (descriptive, label-free)
    tail = ev.filter((pl.col('risk') < .5) & pl.col('delta_score').is_not_null())
    r = tail['risk'].to_numpy(); d = tail['delta_score'].to_numpy()
    eps = 1e-6
    logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    ls = logit(r) + logit(d)
    top_n = 3000
    idx_ls = np.argsort(-ls)[:top_n]; idx_rank = np.argsort(-(0.5 * (np.argsort(np.argsort(r)) / len(r)) + 0.5 * (np.argsort(np.argsort(d)) / len(d))))[:top_n]
    res['E_orderings'] = {'tail_n': int(tail.height), 'logit_sum_top3000_risk_q': q(r[idx_ls], (.1, .5, .9)), 'logit_sum_top3000_delta_q': q(d[idx_ls], (.1, .5, .9)),
                          'rank_avg_top3000_risk_q': q(r[idx_rank], (.1, .5, .9)), 'rank_avg_top3000_delta_q': q(d[idx_rank], (.1, .5, .9)),
                          'overlap_top3000': int(len(set(idx_ls) & set(idx_rank))),
                          'risk_ge_0.1_pairs_in_tail': int((r >= .1).sum()), 'risk_ge_0.1_kept_in_logit_top3000': int((r[idx_ls] >= .1).sum()), 'risk_ge_0.1_kept_in_rank_top3000': int((r[idx_rank] >= .1).sum())}
    print(json.dumps(res['E_orderings']), flush=True)
    write_json(EXP / 'diagnostics.json', res)


if __name__ == '__main__':
    main()
