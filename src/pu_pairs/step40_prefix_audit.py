"""Step 40: label-free short-history audit (OBSERVATIONS 14.23).

Question. Evaluation histories are ~2/3 as long as development ones (2,000 vs 3,000 hands per table) and the evaluation
block (risk >= .5) holds 2.27 pairs per table against 1.44 per table on full development histories (fold 1, all pairs)
even though the evaluation candidate list excludes every pair with a publicly labelled positive player. If shortening a
history can only remove true collusion signal, then any INCREASE in high-scoring pairs after truncation is a
false-positive population created by history length, and it would explain part of the public P loss (.938 vs local .99).

Method. Rebuild the exact production feature stack (build_allpairs_fold) for every pair of the held fold-1 pools using only
the first PREFIX of each table's development hands (by started_at), score with Codex's frozen fold-1 model (never saw these
pools), and compare against the full-history scores already saved (scores.parquet): counts >= .5 / .9 / .99 per table among
pairs with >= 38 shared hands in each view, labelled-negative crossings, labelled-positive recall, and the pair-level
transition table (full-history risk of the pairs that are high only on the prefix). No label of an unlabelled pair is used.
Output: experiments/pu_pairs_20260911/fold1-prefix60/{batch_*, scores.parquet, audit.json}
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, gc, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
from pu_pairs.build_allpairs_fold import (EXP, CONF, STYLE, sha, fold_policy, fold_pools, all_pairs, base_features, v2_features, conditional_features,
                                          aggregate, tails, extra_aggregate)
from pu_pairs.score_allpairs_fold import average_precision
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import matrix  # noqa: E402

MIN_SH = 38


def build(fold, prefix, out, pools_per_batch=2):
    out.mkdir(parents=True, exist_ok=True)
    pools = fold_pools(fold)
    hands_dev = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter((pl.col('phase') == 'development') & pl.col('table_id').is_in(pools))
    # first PREFIX of each table's development hands by started_at
    hands_dev = hands_dev.sort('table_id', 'started_at').with_columns((pl.int_range(pl.len()).over('table_id') / pl.len().over('table_id')).alias('_pos'))
    hands_all = hands_dev.filter(pl.col('_pos') < prefix).drop('_pos')
    write_json(out / 'identity.json', {'fold': fold, 'prefix': prefix, 'pools': pools, 'hands_kept': hands_all.height, 'hands_dev': hands_dev.height,
                                       'policy_sha256': sha(fold_policy(fold)), 'style_counts_sha256': sha(STYLE), 'code_sha256': sha(ORIGINAL / 'Claude Version' / 'src' / 'pu_pairs' / 'step40_prefix_audit.py')})
    pairs = all_pairs(pools, hands_all)
    pairs.write_csv(out / 'pairs.csv')
    policy = joblib.load(fold_policy(fold))
    caf = [c for c in pl.read_parquet(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', n_rows=1).columns if c.startswith('ca_')]
    counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
    for i, start in enumerate(range(0, len(pools), pools_per_batch)):
        dest = out / f'batch_{i:03}-pairs.parquet'
        if dest.exists():
            continue
        t0 = time.perf_counter()
        these = pools[start:start + pools_per_batch]
        h = hands_all.filter(pl.col('table_id').is_in(these))
        p = pairs.filter(pl.col('table_id').is_in(these)).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p)
        p = p.join(raw.select('pair_id').unique(), on='pair_id', how='semi')
        raw = v2_features(raw, p)
        names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
        events = [c for c in names if c.startswith('v2_')]
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(p.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        cas = []
        for pool in these:
            ca, _, _ = conditional_features(keys.filter(pl.col('table_id') == pool), h.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind'), policy, counts, caf)
            cas.append(ca); gc.collect()
        ca = pl.concat(cas)
        pairframe = aggregate(raw, names).join(tails(raw, events), on=['pair_id', 'table_id'], validate='1:1').join(extra_aggregate(ca, caf), on=['pair_id', 'table_id'], validate='1:1').sort('pair_id')
        assert pairframe.height == p.height
        pairframe.write_parquet(dest)
        print(json.dumps({'batch': i, 'pairs': pairframe.height, 'pair_hands': raw.height, 'seconds': round(time.perf_counter() - t0, 1)}), flush=True)
        del raw, ca, cas, pairframe, keys; gc.collect()
    return pairs


def score(fold, out):
    frame = pl.concat([pl.read_parquet(b) for b in sorted(out.glob('batch_*-pairs.parquet'))]).sort('pair_id')
    pairs = pl.read_csv(out / 'pairs.csv')
    frame = frame.join(pairs.select('pair_id', 'player_1', 'player_2', 'label', 'behavior_family'), on='pair_id', validate='1:1')
    bundle = joblib.load(CONF / 'results' / f'fold{fold}' / 'conditional-pair.joblib')
    proba = bundle['model'].predict_proba(matrix(frame, bundle['features']))
    scored = frame.select('pair_id', 'player_1', 'player_2', 'table_id', 'label', 'behavior_family', 'shared_hands').with_columns(pl.Series('risk', 1 - proba[:, 0]), pl.Series('predicted_class', np.argmax(proba[:, 1:], axis=1) + 1))
    scored.write_parquet(out / 'scores.parquet')
    return scored


def audit(fold, prefix, out, scored):
    full = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet').select('pair_id', pl.col('shared_hands').alias('shared_full'), pl.col('risk').alias('risk_full'))
    j = scored.join(full, on='pair_id', how='left')
    n_tables = scored['table_id'].n_unique()
    res = {'fold': fold, 'prefix': prefix, 'tables': n_tables, 'pairs': scored.height, 'pairs_matched_full': int(j['risk_full'].is_not_null().sum())}
    j = j.filter(pl.col('risk_full').is_not_null())
    res['shared_hands_median'] = {'prefix': float(j['shared_hands'].median()), 'full': float(j['shared_full'].median())}

    def counts(df, col, tag):
        pos = df.filter(pl.col('label') == 1); neg = df.filter(pl.col('label') == 0); unl = df.filter(pl.col('label') < 0)
        r = {'n': {'positives': pos.height, 'labelled_negatives': neg.height, 'unlabelled': unl.height}}
        for t in (.99, .9, .5, .1):
            r[f'ge_{t}'] = {'positives': int((pos[col] >= t).sum()), 'labelled_negatives': int((neg[col] >= t).sum()), 'unlabelled': int((unl[col] >= t).sum()), 'unlabelled_per_table': round(float((unl[col] >= t).sum()) / n_tables, 3)}
        r['ap_labelled'] = average_precision(np.r_[np.ones(pos.height), np.zeros(neg.height)], np.r_[pos[col].to_numpy(), neg[col].to_numpy()], np.r_[pos['pair_id'].to_numpy(), neg['pair_id'].to_numpy()]) if pos.height and neg.height else None
        r['ap_unlabelled_as_negative'] = average_precision((df['label'] == 1).to_numpy().astype(int), df[col].to_numpy(), df['pair_id'].to_numpy())
        return r
    # same pair universe: eligible in BOTH views (prefix shared >= 38); also each view's own eligibility
    both = j.filter(pl.col('shared_hands') >= MIN_SH)
    res['prefix_view_eligible_prefix38'] = counts(both, 'risk', 'prefix')
    res['full_view_same_pairs'] = counts(both, 'risk_full', 'full')
    res['full_view_eligible_full38'] = counts(j.filter(pl.col('shared_full') >= MIN_SH), 'risk_full', 'full')
    # transition table among eligible-in-prefix unlabelled pairs
    unl = both.filter(pl.col('label') < 0)
    hi_pre = unl.filter(pl.col('risk') >= .5)
    res['transition_unlabelled_prefix_ge_.5'] = {'n': hi_pre.height, 'full_ge_.5': int((hi_pre['risk_full'] >= .5).sum()), 'full_in_.1_.5': int(((hi_pre['risk_full'] >= .1) & (hi_pre['risk_full'] < .5)).sum()), 'full_lt_.1': int((hi_pre['risk_full'] < .1).sum()),
                                                'full_risk_quantiles_10_50_90': [float(v) for v in np.quantile(hi_pre['risk_full'].to_numpy(), [.1, .5, .9])] if hi_pre.height else None}
    hi_full = unl.filter(pl.col('risk_full') >= .5)
    res['transition_unlabelled_full_ge_.5'] = {'n': hi_full.height, 'prefix_ge_.5': int((hi_full['risk'] >= .5).sum()), 'prefix_lt_.1': int((hi_full['risk'] < .1).sum())}
    # by shared-hands bin (prefix view): unlabelled >= .5 rate
    bins = [(38, 55), (55, 70), (70, 100), (100, 150), (150, 1000)]
    res['unlabelled_ge_.5_rate_by_prefix_bin'] = {f'{a}_{b}': {'n': int(unl.filter((pl.col('shared_hands') >= a) & (pl.col('shared_hands') < b)).height), 'ge_.5': int(unl.filter((pl.col('shared_hands') >= a) & (pl.col('shared_hands') < b) & (pl.col('risk') >= .5)).height)} for a, b in bins}
    unl_full = j.filter((pl.col('shared_full') >= MIN_SH) & (pl.col('label') < 0))
    res['unlabelled_ge_.5_rate_by_full_bin'] = {f'{a}_{b}': {'n': int(unl_full.filter((pl.col('shared_full') >= a) & (pl.col('shared_full') < b)).height), 'ge_.5': int(unl_full.filter((pl.col('shared_full') >= a) & (pl.col('shared_full') < b) & (pl.col('risk_full') >= .5)).height)} for a, b in bins}
    # labelled positives: prefix vs full risk
    pos = both.filter(pl.col('label') == 1)
    res['positives_prefix_minus_full_risk_quantiles_10_50_90'] = [float(v) for v in np.quantile((pos['risk'] - pos['risk_full']).to_numpy(), [.1, .5, .9])] if pos.height else None
    res['positives_higher_on_prefix'] = int((pos['risk'] > pos['risk_full'] + .05).sum()) if pos.height else None
    # new prefix-only high scorers: family, shared hands
    new = hi_pre.filter(pl.col('risk_full') < .1)
    res['prefix_only_high_unlabelled'] = {'n': new.height, 'predicted_class': {str(k): int(v) for k, v in new.group_by('predicted_class').agg(pl.len()).iter_rows()} if new.height else {},
                                          'shared_hands_prefix_quantiles': [float(v) for v in np.quantile(new['shared_hands'].to_numpy(), [.1, .5, .9])] if new.height else None,
                                          'shared_full_quantiles': [float(v) for v in np.quantile(new['shared_full'].to_numpy(), [.1, .5, .9])] if new.height else None}
    write_json(out / 'audit.json', res)
    print(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--prefix', type=float, default=0.6)
    args = ap.parse_args()
    out = EXP / f'fold{args.fold}-prefix{int(round(args.prefix * 100))}'
    build(args.fold, args.prefix, out)
    scored = score(args.fold, out)
    audit(args.fold, args.prefix, out, scored)


if __name__ == '__main__':
    main()
