"""Step 21: score every pair of a held fold with the frozen fold pair model and audit the ranking at
realistic prevalence (training-only: the fold model never saw these pools; no label of an unlabelled pair
exists to tune against).

Parity guard: labelled pairs must reproduce reports/conditional-action-confirmation/results/fold{F}/
pair-predictions.csv (history=full, candidate=conditional) exactly, otherwise the pipeline port is wrong.
Output: Claude Version/experiments/pu_pairs_20260911/fold{F}/scores.parquet and audit.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
sys.path.insert(0, str(ORIGINAL / 'src'))
import numpy as np
import polars as pl
import joblib
from robust_pair_experiment import matrix  # noqa: E402
from pu_pairs.build_allpairs_fold import EXP, CONF  # noqa: E402


def average_precision(y, s, ids):
    """Official metric: mergesort on -score over rows sorted by pair_id (ties -> ascending pair_id)."""
    order0 = np.argsort(ids, kind='mergesort')
    y = y[order0]; s = s[order0]
    order = np.argsort(-s, kind='mergesort')
    ranked = y[order]
    tp = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    return float(np.sum((tp / ranks) * ranked) / max(1, ranked.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=1)
    args = ap.parse_args()
    out = EXP / f'fold{args.fold}'
    batches = sorted(out.glob('batch_*-pairs.parquet'))
    frame = pl.concat([pl.read_parquet(b) for b in batches]).sort('pair_id')
    pairs = pl.read_csv(out / 'pairs.csv')
    frame = frame.join(pairs.select('pair_id', 'player_1', 'player_2', 'label', 'behavior_family'), on='pair_id', validate='1:1')
    bundle = joblib.load(CONF / 'results' / f'fold{args.fold}' / 'conditional-pair.joblib')
    features = bundle['features']
    missing = [f for f in features if f not in frame.columns]
    assert not missing, missing[:10]
    proba = bundle['model'].predict_proba(matrix(frame, features))
    risk = 1 - proba[:, 0]
    pred = np.argmax(proba[:, 1:], axis=1) + 1
    scored = frame.select('pair_id', 'player_1', 'player_2', 'table_id', 'label', 'behavior_family', 'shared_hands').with_columns(
        pl.Series('risk', risk), pl.Series('predicted_class', pred), *[pl.Series(f'p{i}', proba[:, i]) for i in range(proba.shape[1])])
    scored.write_parquet(out / 'scores.parquet')
    # parity with Codex's saved fold predictions
    saved = pl.read_csv(CONF / 'results' / f'fold{args.fold}' / 'pair-predictions.csv').filter((pl.col('history') == 'full') & (pl.col('candidate') == 'conditional'))
    par = scored.filter(pl.col('label') >= 0).join(saved.select('pair_id', pl.col('risk').alias('saved_risk'), pl.col('predicted_class').alias('saved_class')), on='pair_id', how='inner')
    audit = {'fold': args.fold, 'batches': len(batches), 'pairs': scored.height, 'labelled_in_frame': int((scored['label'] >= 0).sum()), 'labelled_matched_to_saved': par.height,
             'parity_max_abs_risk_diff': float((par['risk'] - par['saved_risk']).abs().max()) if par.height else None,
             'parity_class_mismatches': int((par['predicted_class'] != par['saved_class']).sum()) if par.height else None}
    pos = scored.filter(pl.col('label') == 1)
    neg = scored.filter(pl.col('label') == 0)
    unl = scored.filter(pl.col('label') < 0)
    audit['counts'] = {'positives': pos.height, 'labelled_negatives': neg.height, 'unlabelled': unl.height}
    audit['ap_labelled_only'] = average_precision(np.r_[np.ones(pos.height), np.zeros(neg.height)], np.r_[pos['risk'].to_numpy(), neg['risk'].to_numpy()], np.r_[pos['pair_id'].to_numpy(), neg['pair_id'].to_numpy()])
    y_all = (scored['label'] == 1).to_numpy().astype(int)
    audit['ap_unlabelled_as_negative'] = average_precision(y_all, scored['risk'].to_numpy(), scored['pair_id'].to_numpy())
    for t in (.99, .9, .5, .1):
        audit[f'count_ge_{t}'] = {'positives': int((pos['risk'] >= t).sum()), 'labelled_negatives': int((neg['risk'] >= t).sum()), 'unlabelled': int((unl['risk'] >= t).sum())}
    audit['positive_risk_quantiles_10_25_50'] = [float(v) for v in np.quantile(pos['risk'].to_numpy(), [.1, .25, .5])] if pos.height else None
    med = float(np.median(pos['risk'].to_numpy())) if pos.height else 1.0
    audit['unlabelled_above_median_positive'] = int((unl['risk'] >= med).sum())
    audit['unlabelled_per_table_ge_0.5'] = float((unl['risk'] >= .5).sum() / max(1, scored['table_id'].n_unique()))
    # do high-scoring unlabelled pairs share a player with a labelled positive? (evaluation excludes such pairs)
    pos_players = set(pos['player_1']) | set(pos['player_2'])
    hi = unl.filter(pl.col('risk') >= .5).with_columns((pl.col('player_1').is_in(sorted(pos_players)) | pl.col('player_2').is_in(sorted(pos_players))).alias('shares_positive_player'))
    audit['unlabelled_ge_0.5_sharing_positive_player'] = int(hi['shares_positive_player'].sum()) if hi.height else 0
    # player concentration among high-scoring unlabelled pairs
    if hi.height:
        players = pl.concat([hi.select(pl.col('player_1').alias('p')), hi.select(pl.col('player_2').alias('p'))]).group_by('p').agg(pl.len().alias('n'))
        audit['unlabelled_ge_0.5_player_multiplicity'] = {'players': players.height, 'max_pairs_per_player': int(players['n'].max()), 'players_in_ge_3_pairs': int((players['n'] >= 3).sum())}
        audit['unlabelled_ge_0.5_predicted_class'] = {str(k): int(v) for k, v in hi.group_by('predicted_class').agg(pl.len()).iter_rows()}
    audit['top_unlabelled'] = hi.sort('risk', descending=True).head(25).select('pair_id', 'table_id', 'shared_hands', 'risk', 'predicted_class', 'shares_positive_player', 'p1', 'p2', 'p3').to_dicts()
    # positives outranked by unlabelled pairs: how many unlabelled pairs sit above each positive?
    unl_sorted = np.sort(unl['risk'].to_numpy())
    above = [int(len(unl_sorted) - np.searchsorted(unl_sorted, r, side='right')) for r in pos['risk'].to_numpy()]
    audit['unlabelled_above_each_positive_quantiles_50_75_90_max'] = [float(v) for v in np.quantile(above, [.5, .75, .9, 1.0])] if above else None
    write_json(out / 'audit.json', audit)
    print(json.dumps(audit, indent=1, default=str))


if __name__ == '__main__':
    main()
