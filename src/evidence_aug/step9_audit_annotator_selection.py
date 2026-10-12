"""Step 9: training-only audit of which fold-benefit-type hands the annotators actually list.

Fold-0 TRAINING pools only (no held truth). Out-of-fold ranker scores for the 274 training queries come from a
5-way pool-hashed cross-fit of the unchanged recipe. Among each query's top-20 out-of-fold candidates we ask
whether simple, non-ranker attributes (chronology within the pair, pot/loss magnitude, showdown) separate the
listed hands from the unlisted ones, and how much MAP@5 a pool-grouped re-ranker over the top-20 could add.
Output: Claude Version/experiments/evidence_augmentation_20260910/audit-annotator-selection.json
"""
import os
import sys
import json
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import EXP, ORIGINAL, write_json
from evidence_aug.frame import load_original, held_pools, training_relevance, attach_relevance, score, family_columns, FAMILIES
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step5_evaluate import per_pair_map
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

SPLITS = 5
TOP = 20
CLF = dict(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=40, learning_rate=.05, n_jobs=2, verbosity=-1)
ATTR = ['chrono_rank', 'chrono_frac', 'n_hands', 'pot_bb', 'net_bb_min', 'net_bb_gap', 'contribution_bb_max', 'hole_strength_max', 'hole_strength_min',
        'strong_fold_to_partner', 'folded_sum', 'players_at_showdown', 'went_to_showdown_sum', 'joint_win_share', 'opposite_outcomes']
FAMCOLS = ['family_' + f for f in FAMILIES]


def split_of(table):
    return int(hashlib.sha256(('audit-v1:' + table).encode()).hexdigest(), 16) % SPLITS


def main():
    data, features, caf = load_original()
    pools = held_pools()
    train = data.filter(~pl.col('table_id').is_in(pools))
    evidence = training_relevance(set(train['table_id']))
    train = attach_relevance(train, evidence)
    assert train['pair_id'].n_unique() == 274 and train.filter(pl.col('relevance') == 1)['pair_id'].n_unique() == 274
    hands = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').select('hand_id', 'started_at').join(train.select('hand_id').unique().lazy(), on='hand_id').collect()
    train = train.join(hands, on='hand_id', validate='m:1').with_columns(
        pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'), pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'))
    train = train.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'), pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
    # out-of-fold ranker scores
    oof = np.full(train.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(train['split'].to_numpy() == s)[0]
        model = fit_ranker(train.filter(pl.col('split') != s), features)
        oof[idx] = score(model, family_columns(train[idx], 'behavior_family'), features)
        print(json.dumps({'split': s, 'scored_rows': int(len(idx))}), flush=True)
    assert np.isfinite(oof).all()
    train = train.with_columns(pl.Series('oof', oof)).with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').alias('oof_rank'))
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    labels = dict(train.select('pair_id', 'behavior_family').unique().iter_rows())
    base_map = per_pair_map(train.select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')), truths)
    out = {'training_only': True, 'queries': 274, 'oof_map5': float(np.mean(list(base_map.values()))),
           'oof_map5_by_family': {f: float(np.mean([m for p, m in base_map.items() if labels[p] == f])) for f in FAMILIES}}
    listed = train.filter(pl.col('relevance') == 1)
    out['listed_recall_at'] = {str(k): float((listed['oof_rank'] <= k).mean()) for k in [5, 10, 20, 40]}
    cands = train.filter(pl.col('oof_rank') <= TOP).sort('pair_id', 'hand_id')
    lst, unl = cands.filter(pl.col('relevance') == 1), cands.filter(pl.col('relevance') == 0)
    out['top20_rows'] = {'listed': lst.height, 'unlisted': unl.height}
    out['top20_attribute_means'] = {a: {'listed': float(lst[a].mean()), 'unlisted': float(unl[a].mean())} for a in ATTR + ['oof', 'oof_rank']}
    out['top20_attribute_means_by_family'] = {f: {a: {'listed': float(lst.filter(pl.col('behavior_family') == f)[a].mean()), 'unlisted': float(unl.filter(pl.col('behavior_family') == f)[a].mean())}
                                                  for a in ['chrono_frac', 'pot_bb', 'net_bb_min', 'hole_strength_max', 'players_at_showdown']} for f in FAMILIES}
    out['listed_rate_by_chrono_quartile'] = {str(q): float(cands.filter((pl.col('chrono_frac') >= q / 4) & (pl.col('chrono_frac') < (q + 1) / 4 + (1 if q == 3 else 0)))['relevance'].mean()) for q in range(4)}
    cands = cands.with_columns(pl.col('pot_bb').rank('ordinal', descending=True).over('pair_id').alias('pot_rank_in_top20'))
    out['listed_rate_by_pot_rank_in_top20'] = {'1-5': float(cands.filter(pl.col('pot_rank_in_top20') <= 5)['relevance'].mean()), '6-20': float(cands.filter(pl.col('pot_rank_in_top20') > 5)['relevance'].mean())}
    # pool-grouped re-rankers over the top-20 candidates; the tail (rank > 20) keeps the ranker order below the head
    fam = family_columns(cands, 'behavior_family')
    sets = {'score_only': ['oof', 'oof_rank'] + FAMCOLS,
            'score_plus_chrono': ['oof', 'oof_rank', 'chrono_rank', 'chrono_frac', 'n_hands'] + FAMCOLS,
            'score_plus_magnitude': ['oof', 'oof_rank', 'pot_bb', 'net_bb_min', 'net_bb_gap', 'contribution_bb_max', 'players_at_showdown', 'went_to_showdown_sum', 'joint_win_share', 'opposite_outcomes'] + FAMCOLS,
            'score_plus_all_attributes': ['oof', 'oof_rank'] + ATTR + FAMCOLS}
    y = fam['relevance'].to_numpy(); groups = fam['table_id'].to_numpy()
    tail = train.filter(pl.col('oof_rank') > TOP).select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score'))
    out['rerank_top20_map5'] = {}
    for name, cols in sets.items():
        x = fam.select(cols).to_numpy().astype(np.float64); prob = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(GroupKFold(SPLITS).split(x, y, groups)):
            prob[te] = lgb.LGBMClassifier(**{**CLF, 'random_state': 4011 + k}).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
        assert np.isfinite(prob).all()
        full = pl.concat([fam.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', 1000.0 + prob)), tail])
        m = per_pair_map(full, truths)
        out['rerank_top20_map5'][name] = {'map5': float(np.mean(list(m.values()))), 'delta_vs_oof': float(np.mean([m[p] - base_map[p] for p in m])),
                                          'delta_by_family': {f: float(np.mean([m[p] - base_map[p] for p in m if labels[p] == f])) for f in FAMILIES}}
        print(json.dumps({name: out['rerank_top20_map5'][name]}), flush=True)
    write_json(EXP / 'audit-annotator-selection.json', out)
    print(json.dumps({k: v for k, v in out.items() if k in ['oof_map5', 'oof_map5_by_family', 'listed_recall_at', 'listed_rate_by_chrono_quartile', 'listed_rate_by_pot_rank_in_top20']}, indent=1))
    for a, v in out['top20_attribute_means'].items():
        print('%-28s listed=%9.4f unlisted=%9.4f' % (a, v['listed'], v['unlisted']))


if __name__ == '__main__':
    main()
