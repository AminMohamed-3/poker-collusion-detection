"""Step 45 (training-only, step-14 protocol): per-hand leave-one-out attribution of the production PAIR model as an
evidence re-ranking feature over the top-20 candidates of the 162 audit queries.

Idea (external research 14.27, attention-MIL proxy): the pair model decides "colluding" from aggregates over hands; the
hands whose removal lowers the pair's risk / family probability most are the hands the pair model leans on, which may
be the manipulated ones. The pair model is refitted with the audit pools EXCLUDED (production recipe otherwise), so the
attribution is out-of-sample for every audit query. Held pools and evaluation data are never touched.

Pre-registered gate (written before any result): paired MAP@5 delta of base+LOO vs base (score + chrono + first-k,
3 seeds, pool-grouped 5-fold) with pool-bootstrap 95 % CI excluding zero AND at least 30 of 162 queries moving.
Output: Claude Version/experiments/label_process_20260911/audit-loo-attribution.json (+ candidates parquet)."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
from evidence_aug.frame import training_relevance, attach_relevance, score, family_columns, FAMILIES
from evidence_aug.fold import load_fold, fold_pools
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step5_evaluate import per_pair_map, pool_bootstrap
from evidence_aug.step14_audit_truncation import split_of, add_chronology, SPLITS, TOP, CLF, FAMCOLS, fmean
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pu_pairs.build_allpairs_fold import CONF, extra_aggregate
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, TRAIN_VIEWS, MODEL_PARAMS, thin, aggregate, matrix  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'label_process_20260911'
SEEDS = [11, 12, 13]
LOO = ['d_risk', 'd_logit', 'd_fam', 'loo_rank', 'loo_z']
FIRSTK = ['n_earlier_top20', 'n_earlier_top10', 'n_earlier_top5', 'n_earlier_higher', 'cum_oof_before']
BASE = ['oof', 'oof_rank', 'chrono_rank', 'chrono_frac', 'n_hands'] + FIRSTK + FAMCOLS


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_audit():
    data, features, _, _ = load_fold(0)
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    audit = data.filter(~pl.col('table_id').is_in(sorted(held)))
    pools = sorted(set(audit['table_id']))
    evidence = training_relevance(set(pools))
    audit = add_chronology(attach_relevance(audit, evidence))
    audit = audit.with_columns(pl.col('table_id').map_elements(split_of, return_dtype=pl.Int64).alias('split'))
    oof = np.full(audit.height, np.nan)
    for s in range(SPLITS):
        idx = np.where(audit['split'].to_numpy() == s)[0]
        model = fit_ranker(audit.filter(pl.col('split') != s), features)
        oof[idx] = score(model, family_columns(audit[idx], 'behavior_family'), features)
        print(json.dumps({'ranker_split': s, 'scored_rows': int(len(idx))}), flush=True)
    assert np.isfinite(oof).all()
    audit = audit.with_columns(pl.Series('oof', oof)).with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').alias('oof_rank'))
    truths = {k[0]: set(g['hand_id']) for k, g in evidence.group_by('pair_id')}
    return audit, pools, truths


def pair_model_and_loo(audit, pools, cands):
    """Production-recipe pair model with the audit pools excluded; LOO attribution for the candidate hands."""
    t0 = time.time()
    bundle = joblib.load(CONF / 'results' / 'fold1' / 'conditional-pair.joblib'); features = list(bundle['features'])
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    raw = raw.join(labels.select('pair_id'), on='pair_id', how='semi')
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    ca = pl.read_parquet(CONF / 'conditional-fold1-v1' / 'policy-screen' / 'hand-features.parquet')
    ca_features = [c for c in ca.columns if c.startswith('ca_')]
    combined = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left').with_columns(pl.col(ca_features).fill_null(0))
    assert combined.height == raw.height
    baseline_full = pl.read_parquet(ORIGINAL / 'outputs_v2' / 'pairs_development.parquet').join(tails(raw, events), on=['pair_id', 'table_id']).join(labels, on='pair_id').sort('pair_id')
    full = baseline_full.join(extra_aggregate(combined, ca_features), on=['pair_id', 'table_id']).sort('pair_id')
    views = []
    for fraction, seed in TRAIN_VIEWS:
        raw_sample = thin(raw, fraction, seed); added_sample = thin(combined, fraction, seed)
        view = aggregate(raw_sample, names).join(tails(raw_sample, events), on=['pair_id', 'table_id']).join(labels, on='pair_id')
        views.append(view.join(extra_aggregate(added_sample, ca_features), on=['pair_id', 'table_id']).select(full.columns).sort('pair_id'))
    train = pl.concat([full.filter(~pl.col('table_id').is_in(pools))] + [v.filter(~pl.col('table_id').is_in(pools)) for v in views])
    y = np.array([LABELS[f] for f in train['behavior_family']]); X = matrix(train, features)
    assert not set(train['table_id']) & set(pools)
    model = lgb.LGBMClassifier(**MODEL_PARAMS); model.fit(X, y, sample_weight=np.full(train.height, 1 / 3))
    info = {'train_rows': int(train.height), 'train_positive_rows': int((y > 0).sum()), 'train_tables': int(train['table_id'].n_unique()), 'audit_pools_excluded': len(pools),
            'fit_seconds': round(time.time() - t0, 1)}
    print('pair model', json.dumps(info), flush=True)
    # ---- LOO variants for the candidate hands of the audit queries ----
    q = cands.select('pair_id').unique()
    raw_q = raw.join(q, on='pair_id', how='semi'); comb_q = combined.join(q, on='pair_id', how='semi')
    ca_cov = float(comb_q.join(ca.select('pair_id', 'hand_id').unique(), on=['pair_id', 'hand_id'], how='semi').height / comb_q.height)
    info['ca_hand_coverage_audit'] = ca_cov
    loo_keys = cands.select('pair_id', pl.col('hand_id').alias('loo_hand'))

    def variants(frame):
        fullv = frame.with_columns((pl.col('pair_id') + '#full').alias('vid'))
        loo = frame.join(loo_keys, on='pair_id', how='inner').filter(pl.col('hand_id') != pl.col('loo_hand')).with_columns((pl.col('pair_id') + '#' + pl.col('loo_hand')).alias('vid')).drop('loo_hand')
        return pl.concat([fullv, loo]).drop('pair_id').rename({'vid': 'pair_id'})

    rv = variants(raw_q); cv = variants(comb_q)
    agg = aggregate(rv, names).join(tails(rv, events), on=['pair_id', 'table_id']).join(extra_aggregate(cv, ca_features), on=['pair_id', 'table_id'])
    P = model.predict_proba(matrix(agg, features))
    agg = agg.select('pair_id', 'table_id').with_columns(pl.Series('risk', 1 - P[:, 0]), *[pl.Series(f'p{k}', P[:, k]) for k in range(P.shape[1])])
    agg = agg.with_columns(pl.col('pair_id').str.split('#').list.get(0).alias('base_pair'), pl.col('pair_id').str.split('#').list.get(1).alias('loo_hand'))
    fullr = agg.filter(pl.col('loo_hand') == 'full').select(pl.col('base_pair').alias('pair_id'), pl.col('risk').alias('risk_full'), *[pl.col(f'p{k}').alias(f'p{k}_full') for k in range(P.shape[1])])
    # parity: full variant via aggregate(raw) vs the precomputed production full frame
    prod = full.join(q, on='pair_id', how='semi').sort('pair_id')
    prod_risk = 1 - model.predict_proba(matrix(prod, features))[:, 0]
    fr = fullr.sort('pair_id')
    info['parity_full_variant_vs_production_frame_max_abs'] = float(np.max(np.abs(fr['risk_full'].to_numpy() - prod_risk)))
    info['audit_query_risk_full_quantiles'] = [float(x) for x in np.quantile(fr['risk_full'].to_numpy(), [.1, .5, .9])]
    loo = agg.filter(pl.col('loo_hand') != 'full').select(pl.col('base_pair').alias('pair_id'), pl.col('loo_hand').alias('hand_id'), 'risk', *[f'p{k}' for k in range(P.shape[1])]).join(fullr, on='pair_id')
    # predicted family class of the full pair (argmax over positive classes)
    pos_cols = [f'p{k}_full' for k in range(1, P.shape[1])]
    loo = loo.with_columns(pl.concat_list(pos_cols).list.arg_max().add(1).alias('fam_class'))
    fam_full = np.array([loo[f'p{k}_full'].to_numpy() for k in range(P.shape[1])]).T; fam_loo = np.array([loo[f'p{k}'].to_numpy() for k in range(P.shape[1])]).T
    fc = loo['fam_class'].to_numpy()
    d_fam = fam_full[np.arange(len(fc)), fc] - fam_loo[np.arange(len(fc)), fc]
    loo = loo.with_columns((pl.col('risk_full') - pl.col('risk')).alias('d_risk'), pl.Series('d_logit', logit(loo['risk_full'].to_numpy()) - logit(loo['risk'].to_numpy())), pl.Series('d_fam', d_fam))
    loo = loo.with_columns(pl.col('d_logit').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('loo_rank'),
                           ((pl.col('d_logit') - pl.col('d_logit').mean().over('pair_id')) / pl.col('d_logit').std().over('pair_id').fill_null(0).clip(lower_bound=1e-9)).alias('loo_z'))
    return loo.select('pair_id', 'hand_id', 'risk_full', 'risk', 'd_risk', 'd_logit', 'd_fam', 'loo_rank', 'loo_z'), info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gate = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'protocol': 'step-14 5b: top-20 OOF candidates, LGBM re-ranker, GroupKFold(5) by table, seeds 11/12/13 averaged per query, tail keeps ranker order',
            'accept_if': 'paired MAP@5 delta (base+LOO vs base) pool-bootstrap 95% CI excludes 0 AND >= 30 of 162 queries move', 'else': 'null; E stays closed'}
    write_json(OUT / 'audit-loo-attribution-gate.json', gate)
    audit, pools, truths = build_audit()
    labels = dict(audit.select('pair_id', 'behavior_family').unique().iter_rows()); tables = dict(audit.select('pair_id', 'table_id').unique().iter_rows())
    cands = audit.filter(pl.col('oof_rank') <= TOP).sort('pair_id', 'chrono_rank').with_columns(
        pl.int_range(pl.len()).over('pair_id').cast(pl.Float64).alias('n_earlier_top20'),
        (pl.col('oof_rank') <= 10).cast(pl.Float64).cum_sum().over('pair_id').sub((pl.col('oof_rank') <= 10).cast(pl.Float64)).alias('n_earlier_top10'),
        (pl.col('oof_rank') <= 5).cast(pl.Float64).cum_sum().over('pair_id').sub((pl.col('oof_rank') <= 5).cast(pl.Float64)).alias('n_earlier_top5'),
        pl.col('oof').cum_sum().over('pair_id').sub(pl.col('oof')).alias('cum_oof_before')).sort('pair_id', 'hand_id')
    n_higher = np.zeros(cands.height)
    for key, g in cands.with_row_index('row').group_by('pair_id'):
        o = g['oof'].to_numpy(); c = g['chrono_rank'].to_numpy(); r = g['row'].to_numpy()
        for i in range(len(o)):
            n_higher[r[i]] = float(np.sum((c < c[i]) & (o > o[i])))
    cands = cands.with_columns(pl.Series('n_earlier_higher', n_higher))
    loo, info = pair_model_and_loo(audit, pools, cands.select('pair_id', 'hand_id'))
    cands = cands.join(loo, on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert cands['d_logit'].null_count() == 0
    fam = family_columns(cands, 'behavior_family')
    fam.write_parquet(OUT / 'audit-loo-attribution-candidates.parquet')
    y = fam['relevance'].to_numpy(); groups = fam['table_id'].to_numpy()
    out = {'gate': gate, 'pair_model': info, 'audit_queries': int(audit['pair_id'].n_unique()), 'audit_pools': len(pools), 'candidate_rows': int(fam.height)}
    out['single_feature_auc_within_top20'] = {c: float(roc_auc_score(y, fam[c].to_numpy())) for c in LOO + ['oof', 'chrono_rank']}
    # listed rate by LOO rank bucket
    out['listed_rate_by_loo_rank'] = {str(k): fmean(fam.filter(pl.col('loo_rank') == k)['relevance']) for k in range(1, 21)}
    tail = audit.filter(pl.col('oof_rank') > TOP).select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')).with_columns(pl.col('score') - 1e6)
    base_m = per_pair_map(audit.select('pair_id', 'hand_id', 'table_id', pl.col('oof').alias('score')), truths)
    out['oof_ranker_map5'] = fmean(list(base_m.values()))
    sets = {'base': BASE, 'base_plus_loo': BASE + LOO, 'score_plus_loo': ['oof', 'oof_rank'] + LOO + FAMCOLS, 'base_plus_dlogit_only': BASE + ['d_logit']}
    maps = {}
    for name, cols in sets.items():
        X = fam.select(cols).to_numpy().astype(np.float64)
        per_seed = []
        for seed in SEEDS:
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
                clf = lgb.LGBMClassifier(**CLF, random_state=seed); clf.fit(X[tr], y[tr]); pred[te] = clf.predict_proba(X[te])[:, 1]
            head = fam.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', pred + 10.0))
            per_seed.append(per_pair_map(pl.concat([head, tail]), truths))
        m = {p: float(np.mean([s[p] for s in per_seed])) for p in per_seed[0]}
        maps[name] = m
        out[name] = {'map5': fmean(list(m.values())), 'by_family': {f: fmean([v for p, v in m.items() if labels[p] == f]) for f in FAMILIES}, 'per_seed_map5': [fmean(list(s.values())) for s in per_seed]}
        print(json.dumps({'rerank': name, 'map5': out[name]['map5'], 'per_seed': out[name]['per_seed_map5']}), flush=True)
    for name in ['base_plus_loo', 'score_plus_loo', 'base_plus_dlogit_only']:
        delta = {p: maps[name][p] - maps['base'][p] for p in maps['base']}
        out[name]['delta_vs_base'] = fmean(list(delta.values()))
        out[name]['delta_ci95_pool_bootstrap'] = pool_bootstrap(delta, tables)
        out[name]['queries_moved'] = int(sum(abs(d) > 1e-9 for d in delta.values())); out[name]['queries_up'] = int(sum(d > 1e-9 for d in delta.values())); out[name]['queries_down'] = int(sum(d < -1e-9 for d in delta.values()))
        out[name]['delta_by_family'] = {f: fmean([delta[p] for p in delta if labels[p] == f]) for f in FAMILIES}
    r = out['base_plus_loo']
    out['verdict'] = 'ACCEPT' if (r['delta_ci95_pool_bootstrap'][0] > 0 and r['queries_moved'] >= 30) else 'NULL'
    write_json(OUT / 'audit-loo-attribution.json', out)
    print(json.dumps({k: out[k] for k in ['single_feature_auc_within_top20', 'oof_ranker_map5', 'verdict']}), flush=True)
    for name in ['base_plus_loo', 'score_plus_loo', 'base_plus_dlogit_only']:
        print(name, json.dumps({k: out[name][k] for k in ['map5', 'delta_vs_base', 'delta_ci95_pool_bootstrap', 'queries_moved', 'queries_up', 'queries_down']}), flush=True)


if __name__ == '__main__':
    main()
