"""Step 17: training-only audit of the 'episodic activation window' hypothesis.

The competition page says coordination is episodic and 'a relationship is not necessarily active over the
whole timeline'. If planted evidence hands sit inside an activation window, then (a) the first listed hand
should be near the window start and (b) suspicious-looking hands should cluster in time, so a hand whose
time-neighbours also look suspicious is more likely listed. Uses the saved audit out-of-fold frame from
step 16 (no refit). Compares top-20 re-rankers: chronology vs local-density vs both.
Output: Claude Version/experiments/label_process_20260911/audit-window-density.json
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import write_json
from evidence_aug.frame import family_columns, FAMILIES
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step14_audit_truncation import fmean, OUT, TOP, CLF, FAMCOLS
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

WINDOWS = (3, 5, 10, 20)


def density_features(frame):
    """Per hand: count of other top-K candidates within +-w hands, mean OOF of the +-w time-neighbours, and
    the fraction of the pair's top-10 mass that lies before/after this hand."""
    frame = frame.sort('pair_id', 'chrono_rank')
    parts = []
    for key, g in frame.group_by('pair_id', maintain_order=True):
        c = g['chrono_rank'].to_numpy(); o = g['oof'].to_numpy(); r = g['oof_rank'].to_numpy()
        n = len(c); cols = {}
        top10 = (r <= 10).astype(float); top20 = (r <= 20).astype(float)
        for w in WINDOWS:
            d10 = np.zeros(n); d20 = np.zeros(n); mo = np.zeros(n)
            for i in range(n):
                m = (np.abs(c - c[i]) <= w) & (np.arange(n) != i)
                d10[i] = top10[m].sum(); d20[i] = top20[m].sum(); mo[i] = o[m].mean() if m.any() else -10.0
            cols[f'dens10_w{w}'] = d10; cols[f'dens20_w{w}'] = d20; cols[f'nbr_oof_w{w}'] = mo
        cum = np.cumsum(top10) - top10
        cols['top10_before'] = cum; cols['top10_after'] = top10.sum() - cum - top10
        # nearest other top-10 candidate distance
        idx = np.where(r <= 10)[0]
        nd = np.full(n, float(n))
        for i in range(n):
            others = idx[idx != i]
            if len(others):
                nd[i] = np.abs(c[others] - c[i]).min()
        cols['nearest_top10_dist'] = nd
        parts.append(g.with_columns([pl.Series(k, v) for k, v in cols.items()]))
    return pl.concat(parts).sort('pair_id', 'hand_id')


def main():
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    truths = {k[0]: set(g['hand_id']) for k, g in audit.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(audit.select('pair_id', 'behavior_family').unique().iter_rows())
    out = {'training_only': True, 'queries': int(audit['pair_id'].n_unique())}
    lst = audit.filter(pl.col('relevance') == 1)
    per = lst.group_by('pair_id').agg(pl.col('chrono_frac').min().alias('first'), pl.col('chrono_frac').max().alias('last'), pl.col('chrono_rank').min().alias('first_rank'),
                                      (pl.col('chrono_rank').max() - pl.col('chrono_rank').min()).alias('span_hands'), pl.col('n_hands').first(), pl.len().alias('n_listed'))
    q = [.1, .25, .5, .75, .9]
    out['first_listed_frac_quantiles'] = [float(v) for v in np.quantile(per['first'].to_numpy(), q)]
    out['first_listed_rank_quantiles'] = [float(v) for v in np.quantile(per['first_rank'].to_numpy(), q)]
    out['last_listed_frac_quantiles'] = [float(v) for v in np.quantile(per['last'].to_numpy(), q)]
    out['listed_span_over_history_quantiles'] = [float(v) for v in np.quantile((per['span_hands'] / (per['n_hands'] - 1)).to_numpy(), q)]
    # expected first-listed position if 5 hands were drawn uniformly over the history: n/(k+1)
    out['first_listed_frac_mean_vs_uniform_expectation'] = {'observed': fmean(per['first']), 'uniform_5_of_n': float(np.mean(1.0 / (per['n_listed'].to_numpy() + 1)))}
    out['listed_span_mean_vs_uniform_expectation'] = {'observed': fmean(per['span_hands'] / (per['n_hands'] - 1)), 'uniform_5_of_n': float(np.mean((per['n_listed'].to_numpy() - 1) / (per['n_listed'].to_numpy() + 1)))}
    # does the listed set look like a window? share of the pair's top-10 candidates that fall inside [first_listed, last_listed]
    joined = audit.join(per.select('pair_id', 'first', 'last'), on='pair_id')
    inside = joined.filter((pl.col('chrono_frac') >= pl.col('first')) & (pl.col('chrono_frac') <= pl.col('last')))
    out['top10_unlisted_inside_listed_span'] = {'inside_share_of_history': fmean(per['span_hands'] / (per['n_hands'] - 1)),
                                                 'share_of_unlisted_top10_inside': float((inside.filter((pl.col('relevance') == 0) & (pl.col('oof_rank') <= 10)).height) / max(1, joined.filter((pl.col('relevance') == 0) & (pl.col('oof_rank') <= 10)).height))}
    feat = density_features(audit)
    cands = feat.filter(pl.col('oof_rank') <= TOP).sort('pair_id', 'hand_id')
    fam = family_columns(cands, 'behavior_family')
    y = fam['relevance'].to_numpy(); groups = fam['table_id'].to_numpy()
    dens_cols = [c for c in cands.columns if c.startswith('dens') or c.startswith('nbr_oof') or c in ('top10_before', 'top10_after', 'nearest_top10_dist')]
    out['top20_auc_single'] = {c: float(roc_auc_score(y, fam[c].to_numpy())) for c in dens_cols + ['oof']}
    out['top20_auc_single']['chrono_frac_neg'] = float(roc_auc_score(y, -fam['chrono_frac'].to_numpy()))
    out['top20_listed_vs_unlisted_means'] = {c: {'listed': fmean(fam.filter(pl.col('relevance') == 1)[c]), 'unlisted': fmean(fam.filter(pl.col('relevance') == 0)[c])} for c in ['dens10_w5', 'dens10_w10', 'nbr_oof_w5', 'nearest_top10_dist', 'top10_before', 'top10_after']}
    base_map = per_pair_map(audit.select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')), truths)
    base = fmean(list(base_map.values()))
    tail = audit.filter(pl.col('oof_rank') > TOP).select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score'))
    S = ['oof', 'oof_rank']
    CH = ['chrono_rank', 'chrono_frac', 'n_hands']
    sets = {'score_only': S + FAMCOLS, 'chrono': S + CH + FAMCOLS, 'density': S + dens_cols + FAMCOLS, 'chrono_plus_density': S + CH + dens_cols + FAMCOLS,
            'chrono_plus_nbr_oof': S + CH + [f'nbr_oof_w{w}' for w in WINDOWS] + FAMCOLS, 'chrono_plus_before_after': S + CH + ['top10_before', 'top10_after', 'nearest_top10_dist'] + FAMCOLS}
    out['rerank_top20_map5'] = {}
    for name, cols in sets.items():
        X = np.nan_to_num(fam.select(cols).to_numpy().astype(np.float64))
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = lgb.LGBMClassifier(**CLF, random_state=11)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        head = fam.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred + 10.0))
        m = per_pair_map(pl.concat([head, tail.with_columns(pl.col('score') - 1e6)]), truths)
        s = {'map5': fmean(list(m.values())), 'delta_vs_oof': fmean(list(m.values())) - base, 'auc_top20': float(roc_auc_score(y, pred)),
             'by_family': {f: fmean([v for p, v in m.items() if labels[p] == f]) for f in FAMILIES}}
        out['rerank_top20_map5'][name] = s
        print(json.dumps({'rerank': name, 'map5': round(s['map5'], 4), 'delta': round(s['delta_vs_oof'], 4), 'auc': round(s['auc_top20'], 4)}), flush=True)
    write_json(OUT / 'audit-window-density.json', out)
    print(json.dumps({k: out[k] for k in out if k != 'rerank_top20_map5'}, indent=1))


if __name__ == '__main__':
    main()
