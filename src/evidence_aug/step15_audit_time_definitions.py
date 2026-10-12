"""Step 15: training-only audit of WHICH time definition the annotator responds to, and whether the
382-feature space holds more re-ranking signal than chronology alone.

Same audit pools as step 14 (labeled pools outside all three held folds), same 5-way pool-hashed
out-of-fold ranker. Candidate time definitions per hand:
  chrono_frac   position among the PAIR's shared hands (the confirmed chronology feature)
  table_frac    position among ALL hands of the table
  sess_idx      index of the pair's play session (gap between consecutive shared hands > threshold)
  sess_frac     position within that session
  hours         hours since the pair's first shared hand
  day           calendar day index since the table's first hand
Re-rankers over the top-20 out-of-fold candidates (pool-grouped 5-fold) compare these definitions,
and a full-feature second stage (382 features + chronology) tests whether more signal exists.
Output: Claude Version/experiments/label_process_20260911/audit-time-definitions.json
"""
import os
import sys
import json
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, FAMILIES
from evidence_aug.fold import load_fold, fold_pools
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step14_audit_truncation import split_of, fmean, OUT, SPLITS, TOP, CLF, FAMCOLS
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SESSION_GAP_MIN = 30.0


def add_time(data):
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet', columns=['hand_id', 'table_id', 'started_at'])
    table_rank = hands.with_columns(pl.col('started_at').rank('ordinal').over('table_id').cast(pl.Float64).sub(1).alias('table_rank'),
                                    pl.len().over('table_id').cast(pl.Float64).alias('table_n'),
                                    pl.col('started_at').min().over('table_id').alias('table_start'))
    out = data.join(table_rank.drop('table_id'), on='hand_id', validate='m:1')
    out = out.with_columns(
        pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'), pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'),
        (pl.col('table_rank') / (pl.col('table_n') - 1)).alias('table_frac'),
        ((pl.col('started_at') - pl.col('started_at').min().over('pair_id')).dt.total_seconds() / 3600.0).alias('hours'),
        ((pl.col('started_at') - pl.col('table_start')).dt.total_seconds() / 86400.0).floor().alias('day'))
    out = out.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac')).sort('pair_id', 'started_at')
    gap = (pl.col('started_at').diff().over('pair_id').dt.total_seconds() / 60.0).fill_null(0.0)
    out = out.with_columns(gap.alias('gap_min')).with_columns((pl.col('gap_min') > SESSION_GAP_MIN).cast(pl.Int64).cum_sum().over('pair_id').cast(pl.Float64).alias('sess_idx'))
    out = out.with_columns(pl.len().over('pair_id', 'sess_idx').cast(pl.Float64).alias('sess_n'), pl.col('chrono_rank').rank('ordinal').over('pair_id', 'sess_idx').cast(pl.Float64).sub(1).alias('sess_pos'),
                           pl.col('sess_idx').max().over('pair_id').alias('n_sess'))
    out = out.with_columns(pl.when(pl.col('sess_n') > 1).then(pl.col('sess_pos') / (pl.col('sess_n') - 1)).otherwise(0.0).alias('sess_frac'))
    return out.drop('started_at', 'table_start').sort('pair_id', 'hand_id')


def main():
    data, features, _, _ = load_fold(0)
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    audit = data.filter(~pl.col('table_id').is_in(sorted(held)))
    pools = sorted(set(audit['table_id']))
    evidence = training_relevance(set(pools))
    audit = add_time(attach_relevance(audit, evidence))
    out = {'training_only': True, 'audit_pools': len(pools), 'audit_queries': int(audit['pair_id'].n_unique()), 'session_gap_minutes': SESSION_GAP_MIN}
    gaps = audit.filter(pl.col('chrono_rank') > 0)['gap_min'].to_numpy()
    out['gap_minutes_quantiles_50_90_95_99'] = [float(q) for q in np.quantile(gaps, [.5, .9, .95, .99])]
    out['share_gaps_over_10_30_60_min'] = [float((gaps > t).mean()) for t in (10, 30, 60)]
    out['sessions_per_pair_mean'] = fmean(audit.group_by('pair_id').agg(pl.col('n_sess').first())['n_sess'] + 1)
    out['pair_span_hours_quantiles'] = [float(q) for q in np.quantile(audit.group_by('pair_id').agg(pl.col('hours').max())['hours'].to_numpy(), [.1, .5, .9])]
    out['table_hands_quantiles'] = [float(q) for q in np.quantile(audit.group_by('pair_id').agg(pl.col('table_n').first())['table_n'].to_numpy(), [.1, .5, .9])]
    lst = audit.filter(pl.col('relevance') == 1)
    out['listed_rate_by_session_index'] = {str(k): {'rate': fmean(audit.filter(pl.col('sess_idx') == k)['relevance']), 'rows': int(audit.filter(pl.col('sess_idx') == k).height)} for k in range(6)}
    out['listed_rate_by_day'] = {str(k): {'rate': fmean(audit.filter(pl.col('day') == k)['relevance']), 'rows': int(audit.filter(pl.col('day') == k).height)} for k in range(8)}
    out['listed_mean'] = {c: {'listed': fmean(lst[c]), 'unlisted': fmean(audit.filter(pl.col('relevance') == 0)[c])} for c in ['chrono_frac', 'table_frac', 'sess_idx', 'sess_frac', 'hours', 'day']}
    # out-of-fold ranker
    audit = audit.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
    oof = np.full(audit.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(audit['split'].to_numpy() == s)[0]
        model = fit_ranker(audit.filter(pl.col('split') != s), features)
        oof[idx] = score(model, family_columns(audit[idx], 'behavior_family'), features)
    assert np.isfinite(oof).all()
    audit = audit.with_columns(pl.Series('oof', oof)).with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').alias('oof_rank'))
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    labels = dict(audit.select('pair_id', 'behavior_family').unique().iter_rows())

    def summarize(scored):
        m = per_pair_map(scored, truths)
        return {'map5': fmean(list(m.values())), 'by_family': {f: fmean([v for p, v in m.items() if labels[p] == f]) for f in FAMILIES}}

    base = summarize(audit.select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')))
    out['oof_ranker'] = base
    cands = audit.filter(pl.col('oof_rank') <= TOP).sort('pair_id', 'hand_id')
    fam = family_columns(cands, 'behavior_family')
    y = fam['relevance'].to_numpy(); groups = fam['table_id'].to_numpy()
    out['top20_auc_single'] = {c: float(roc_auc_score(y, -fam[c].to_numpy() if c != 'oof' else fam[c].to_numpy())) for c in ['oof', 'chrono_frac', 'chrono_rank', 'table_frac', 'sess_idx', 'hours', 'day']}
    tail = audit.filter(pl.col('oof_rank') > TOP).select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score'))
    S = ['oof', 'oof_rank']
    sets = {'score_only': S + FAMCOLS,
            'pair_chrono': S + ['chrono_rank', 'chrono_frac', 'n_hands'] + FAMCOLS,
            'pair_frac_only': S + ['chrono_frac'] + FAMCOLS,
            'table_frac': S + ['table_frac', 'table_rank', 'table_n'] + FAMCOLS,
            'session': S + ['sess_idx', 'sess_frac', 'sess_n', 'n_sess'] + FAMCOLS,
            'hours_day': S + ['hours', 'day'] + FAMCOLS,
            'pair_chrono_plus_table': S + ['chrono_rank', 'chrono_frac', 'n_hands', 'table_frac', 'table_rank'] + FAMCOLS,
            'pair_chrono_plus_session': S + ['chrono_rank', 'chrono_frac', 'n_hands', 'sess_idx', 'sess_frac'] + FAMCOLS,
            'all_time': S + ['chrono_rank', 'chrono_frac', 'n_hands', 'table_frac', 'table_rank', 'sess_idx', 'sess_frac', 'hours', 'day'] + FAMCOLS,
            'full_features_plus_pair_chrono': S + ['chrono_rank', 'chrono_frac', 'n_hands'] + [f for f in features if f in fam.columns]}
    out['rerank_top20_map5'] = {}
    for name, cols in sets.items():
        X = fam.select(cols).to_numpy().astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = lgb.LGBMClassifier(**CLF, random_state=11)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        head = fam.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred + 10.0))
        s = summarize(pl.concat([head, tail.with_columns(pl.col('score') - 1e6)]))
        s['delta_vs_oof'] = s['map5'] - base['map5']
        s['auc_top20'] = float(roc_auc_score(y, pred))
        out['rerank_top20_map5'][name] = s
        print(json.dumps({'rerank': name, 'map5': round(s['map5'], 4), 'delta': round(s['delta_vs_oof'], 4), 'auc': round(s['auc_top20'], 4)}), flush=True)
    write_json(OUT / 'audit-time-definitions.json', out)
    print(json.dumps({k: out[k] for k in ['gap_minutes_quantiles_50_90_95_99', 'share_gaps_over_10_30_60_min', 'sessions_per_pair_mean', 'pair_span_hours_quantiles', 'table_hands_quantiles',
                                          'listed_rate_by_session_index', 'listed_rate_by_day', 'listed_mean', 'top20_auc_single']}, indent=1))


if __name__ == '__main__':
    main()
