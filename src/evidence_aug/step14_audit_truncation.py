"""Step 14: training-only audit of the evidence label process (truncation / chronological listing).

Audit set = labeled pools that are NOT held in any of the three known folds (fold0 screen, fold1/fold2
confirmation), so nothing learned here touches any held truth used by later confirmations.

Questions:
  1. Is the list capped at 5 and written in chronological order (evidence_rank == time order)?
  2. Do capped (5-hand) pairs list early hands while 3/4-hand pairs list hands spread over the history?
  3. Truncation signature: do unlisted hands AFTER the last listed hand score as high (out-of-fold ranker)
     as listed hands, i.e. are they 'interventions the annotator stopped listing'?
  4. First-k signature: listed rate as a function of the number of EARLIER high-scoring hands.
  5. How much MAP@5 do rule-based 'earliest-of-top-K' orderings and first-k re-rankers add over
     score-plus-chronology on the audit set?
Output: Claude Version/experiments/label_process_20260911/audit-truncation.json
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
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'label_process_20260911'
SPLITS = 5
TOP = 20
CLF = dict(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=40, learning_rate=.05, n_jobs=2, verbosity=-1)
FAMCOLS = ['family_' + f for f in FAMILIES]


def split_of(table):
    return int(hashlib.sha256(('audit-trunc-v1:' + table).encode()).hexdigest(), 16) % SPLITS


def add_chronology(data):
    hands = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').select('hand_id', 'started_at').join(data.select('hand_id').unique().lazy(), on='hand_id').collect()
    out = data.join(hands, on='hand_id', validate='m:1').with_columns(
        pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'), pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'))
    return out.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac')).drop('started_at').sort('pair_id', 'hand_id')


def fmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if len(x) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, features, _, _ = load_fold(0)
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    audit = data.filter(~pl.col('table_id').is_in(sorted(held)))
    pools = sorted(set(audit['table_id']))
    evidence_full = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv')
    evidence = training_relevance(set(pools))
    audit = add_chronology(attach_relevance(audit, evidence))
    ev = evidence_full.join(audit.select('pair_id').unique(), on='pair_id', how='semi')
    n_q = audit['pair_id'].n_unique()
    assert audit.filter(pl.col('relevance') == 1)['pair_id'].n_unique() == n_q
    out = {'training_only': True, 'audit_pools': len(pools), 'audit_queries': n_q, 'held_pools_excluded': len(held),
           'queries_by_family': {f: int(audit.filter(pl.col('behavior_family') == f)['pair_id'].n_unique()) for f in FAMILIES}}
    # 1. cap and chronological order of evidence_rank
    listed = audit.filter(pl.col('relevance') == 1).join(ev.select('pair_id', 'hand_id', 'evidence_rank'), on=['pair_id', 'hand_id'])
    counts = listed.group_by('pair_id').agg(pl.len().alias('n_listed'))
    out['n_listed_distribution'] = {str(k): int(v) for k, v in counts['n_listed'].value_counts().sort('n_listed').iter_rows()}
    order = listed.sort('pair_id', 'evidence_rank').group_by('pair_id', maintain_order=True).agg(pl.col('chrono_rank'), pl.col('evidence_rank'))
    asc = [bool(np.all(np.diff(np.asarray(c)) > 0)) for c in order['chrono_rank']]
    desc = [bool(np.all(np.diff(np.asarray(c)) < 0)) for c in order['chrono_rank']]
    rho = []
    for c in order['chrono_rank']:
        c = np.asarray(c, dtype=float)
        if len(c) > 2:
            rho.append(float(np.corrcoef(np.argsort(np.argsort(c)), np.arange(len(c)))[0, 1]))
    out['evidence_rank_is_chronological'] = {'pairs_strictly_ascending': int(sum(asc)), 'pairs_strictly_descending': int(sum(desc)), 'pairs': len(asc), 'mean_spearman_rank_vs_time': fmean(rho)}
    # 2. listed positions by n_listed
    listed = listed.join(counts, on='pair_id')
    per_pair = listed.group_by('pair_id').agg(pl.col('n_listed').first(), pl.col('chrono_frac').min().alias('first_frac'), pl.col('chrono_frac').max().alias('last_frac'),
                                              pl.col('chrono_frac').mean().alias('mean_frac'), pl.col('chrono_rank').max().alias('last_rank'), pl.col('n_hands').first(), pl.col('behavior_family').first())
    out['listed_position_by_n_listed'] = {}
    for n in sorted(set(per_pair['n_listed'])):
        g = per_pair.filter(pl.col('n_listed') == n)
        out['listed_position_by_n_listed'][str(n)] = {'pairs': g.height, 'mean_frac': fmean(g['mean_frac']), 'first_frac': fmean(g['first_frac']), 'last_frac': fmean(g['last_frac']),
                                                       'last_frac_quantiles': [float(q) for q in np.quantile(g['last_frac'].to_numpy(), [.1, .25, .5, .75, .9])], 'hands_after_last_listed_mean': fmean(g['n_hands'] - 1 - g['last_rank'])}
    out['listed_frac_quartile_rate_by_n_listed'] = {}
    for n in sorted(set(per_pair['n_listed'])):
        g = listed.filter(pl.col('n_listed') == n)
        out['listed_frac_quartile_rate_by_n_listed'][str(n)] = [float(((g['chrono_frac'] >= q / 4) & (g['chrono_frac'] < (q + 1) / 4 + (1 if q == 3 else 0))).mean()) for q in range(4)]
    out['listed_mean_frac_by_family'] = {f: fmean(per_pair.filter(pl.col('behavior_family') == f)['mean_frac']) for f in FAMILIES}
    # spacing between consecutive listed hands
    gaps = []
    for c in order['chrono_rank']:
        c = np.sort(np.asarray(c, dtype=float))
        gaps.extend(np.diff(c).tolist())
    gaps = np.asarray(gaps)
    out['gaps_between_listed_hands'] = {'mean': fmean(gaps), 'median': float(np.median(gaps)), 'adjacent_share': float((gaps == 1).mean()), 'quantiles_10_25_50_75_90': [float(q) for q in np.quantile(gaps, [.1, .25, .5, .75, .9])]}
    # out-of-fold ranker scores on the audit pools (5-way pool hash)
    audit = audit.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
    oof = np.full(audit.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(audit['split'].to_numpy() == s)[0]
        model = fit_ranker(audit.filter(pl.col('split') != s), features)
        oof[idx] = score(model, family_columns(audit[idx], 'behavior_family'), features)
        print(json.dumps({'split': s, 'scored_rows': int(len(idx))}), flush=True)
    assert np.isfinite(oof).all()
    audit = audit.with_columns(pl.Series('oof', oof)).with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').alias('oof_rank'))
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    labels = dict(audit.select('pair_id', 'behavior_family').unique().iter_rows())
    tables = dict(audit.select('pair_id', 'table_id').unique().iter_rows())

    def summarize(scored, name):
        m = per_pair_map(scored, truths)
        return {'map5': fmean(list(m.values())), 'by_family': {f: fmean([v for p, v in m.items() if labels[p] == f]) for f in FAMILIES}}, m

    base, base_map = summarize(audit.select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')), 'oof')
    out['oof_ranker'] = base
    lst = audit.filter(pl.col('relevance') == 1)
    out['listed_recall_at'] = {str(k): float((lst['oof_rank'] <= k).mean()) for k in [5, 10, 20, 40]}
    # 3. truncation signature: unlisted hands after the last listed hand vs listed hands (score and rank)
    audit = audit.join(per_pair.select('pair_id', 'last_rank', 'n_listed', pl.col('first_frac')), on='pair_id')
    after = audit.filter((pl.col('relevance') == 0) & (pl.col('chrono_rank') > pl.col('last_rank')))
    before = audit.filter((pl.col('relevance') == 0) & (pl.col('chrono_rank') < pl.col('last_rank')))
    out['truncation_signature'] = {}
    for n in sorted(set(per_pair['n_listed'])):
        L, A, B = lst.join(counts, on='pair_id').filter(pl.col('n_listed') == n), after.filter(pl.col('n_listed') == n), before.filter(pl.col('n_listed') == n)
        out['truncation_signature'][str(n)] = {
            'listed_mean_oof': fmean(L['oof']), 'unlisted_after_last_mean_oof': fmean(A['oof']), 'unlisted_before_last_mean_oof': fmean(B['oof']),
            'unlisted_after_last_top10_share': fmean(A['oof_rank'] <= 10) if A.height else None, 'unlisted_before_last_top10_share': fmean(B['oof_rank'] <= 10) if B.height else None,
            'unlisted_after_last_rows': A.height, 'unlisted_before_last_rows': B.height,
            'top10_hands_after_last_listed_per_pair': fmean(A.filter(pl.col('oof_rank') <= 10).group_by('pair_id').agg(pl.len()).join(per_pair.filter(pl.col('n_listed') == n).select('pair_id'), on='pair_id', how='right').fill_null(0)['len']) if A.height else None}
    # 4. first-k signature: among top-10 OOF hands, listed rate by number of EARLIER top-10 hands in the pair
    top10 = audit.filter(pl.col('oof_rank') <= 10).sort('pair_id', 'chrono_rank').with_columns(pl.int_range(pl.len()).over('pair_id').alias('n_earlier_top10'))
    out['listed_rate_by_n_earlier_top10'] = {str(k): {'rate': fmean(top10.filter(pl.col('n_earlier_top10') == k)['relevance']), 'rows': int(top10.filter(pl.col('n_earlier_top10') == k).height)} for k in range(10)}
    top20 = audit.filter(pl.col('oof_rank') <= 20).sort('pair_id', 'chrono_rank').with_columns(pl.int_range(pl.len()).over('pair_id').alias('n_earlier_top20'))
    out['listed_rate_by_n_earlier_top20'] = {str(k): fmean(top20.filter(pl.col('n_earlier_top20') == k)['relevance']) for k in range(20)}
    # 5a. rule-based orderings: take the top-K OOF hands, order them by time (earliest first), then the rest by score
    out['rules_map5'] = {}
    for K in [5, 6, 7, 8, 10, 12, 15, 20]:
        rule = audit.with_columns(pl.when(pl.col('oof_rank') <= K).then(1e6 - pl.col('chrono_rank')).otherwise(pl.col('oof')).alias('score'))
        s, m = summarize(rule.select('pair_id', 'hand_id', 'table_id', 'score'), f'earliest_of_top{K}')
        s['delta_vs_oof'] = s['map5'] - base['map5']
        out['rules_map5'][f'earliest_of_top{K}'] = s
    # 5b. learned re-rankers over the top-20 candidates (pool-grouped 5-fold), tail keeps ranker order
    cands = audit.filter(pl.col('oof_rank') <= TOP).sort('pair_id', 'hand_id')
    cands = cands.sort('pair_id', 'chrono_rank').with_columns(
        pl.int_range(pl.len()).over('pair_id').cast(pl.Float64).alias('n_earlier_top20'),
        (pl.col('oof_rank') <= 10).cast(pl.Float64).cum_sum().over('pair_id').sub((pl.col('oof_rank') <= 10).cast(pl.Float64)).alias('n_earlier_top10'),
        (pl.col('oof_rank') <= 5).cast(pl.Float64).cum_sum().over('pair_id').sub((pl.col('oof_rank') <= 5).cast(pl.Float64)).alias('n_earlier_top5'),
        pl.col('oof').cum_sum().over('pair_id').sub(pl.col('oof')).alias('cum_oof_before')).sort('pair_id', 'hand_id')
    # count of earlier candidates with a higher OOF score
    n_higher = np.zeros(cands.height)
    for key, g in cands.with_row_index('row').group_by('pair_id'):
        o = g['oof'].to_numpy(); c = g['chrono_rank'].to_numpy(); r = g['row'].to_numpy()
        for i in range(len(o)):
            n_higher[r[i]] = float(np.sum((c < c[i]) & (o > o[i])))
    cands = cands.with_columns(pl.Series('n_earlier_higher', n_higher))
    fam = family_columns(cands, 'behavior_family')
    FIRSTK = ['n_earlier_top20', 'n_earlier_top10', 'n_earlier_top5', 'n_earlier_higher', 'cum_oof_before']
    sets = {'score_only': ['oof', 'oof_rank'] + FAMCOLS,
            'score_plus_chrono': ['oof', 'oof_rank', 'chrono_rank', 'chrono_frac', 'n_hands'] + FAMCOLS,
            'score_plus_firstk': ['oof', 'oof_rank'] + FIRSTK + FAMCOLS,
            'score_plus_chrono_plus_firstk': ['oof', 'oof_rank', 'chrono_rank', 'chrono_frac', 'n_hands'] + FIRSTK + FAMCOLS}
    y = fam['relevance'].to_numpy(); groups = fam['table_id'].to_numpy()
    tail = audit.filter(pl.col('oof_rank') > TOP).select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score'))
    out['rerank_top20_map5'] = {}
    for name, cols in sets.items():
        X = fam.select(cols).to_numpy().astype(np.float64)
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = lgb.LGBMClassifier(**CLF, random_state=11)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        head = fam.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred + 10.0))
        s, m = summarize(pl.concat([head, tail.with_columns(pl.col('score') - 1e6)]), name)
        s['delta_vs_oof'] = s['map5'] - base['map5']
        s['delta_by_family_vs_oof'] = {f: s['by_family'][f] - base['by_family'][f] for f in FAMILIES}
        out['rerank_top20_map5'][name] = s
        print(json.dumps({'rerank': name, 'map5': s['map5'], 'delta': s['delta_vs_oof']}), flush=True)
    write_json(OUT / 'audit-truncation.json', out)
    print(json.dumps({k: out[k] for k in ['n_listed_distribution', 'evidence_rank_is_chronological', 'listed_position_by_n_listed', 'truncation_signature', 'listed_rate_by_n_earlier_top10', 'rules_map5']}, indent=1))


if __name__ == '__main__':
    main()
