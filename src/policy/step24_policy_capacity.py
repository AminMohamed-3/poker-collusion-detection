"""Step 24: does a stronger 'normal play' policy model sharpen the per-hand evidence signal?

Training-only on labelled-pair action rows (Codex's action cache, reports/conditional-action-experiment/action-cache):
 - reference  = Codex's policy recipe (100 trees, 15 leaves, <=128 rows per pair, 100k cap) on confirmed-non-target rows of training pools
 - capacity   = same inputs, all confirmed-non-target rows of training pools, 600 trees / 63 leaves
Both score the held fold pools. Reported: multiclass log-loss on held negative rows (policy quality) and, on held
POSITIVE pairs, the within-pair ranking of listed (evidence) hands vs unlisted shared hands by per-hand surprise
aggregates (pooled AUC, mean per-pair AUC, recall@5 by surprise alone). No evidence labels enter any model.
Output: Claude Version/experiments/policy_20260911/step24-fold{F}.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score

CAE = ORIGINAL / 'reports' / 'conditional-action-experiment'
OUT = ORIGINAL / 'Claude Version' / 'experiments' / 'policy_20260911'


def hand_scores(rows, proba):
    y = rows['action_class'].to_numpy()
    surprise = -np.log(np.maximum(proba[np.arange(len(y)), y], 1e-5))
    facing = rows['facing_partner'].to_numpy()
    own = rows['own_strength'].to_numpy()
    resid = np.eye(4)[y] - proba
    d = rows.select('pair_id', 'hand_id').with_columns(
        pl.Series('surprise', surprise), pl.Series('surprise_facing', surprise * facing),
        pl.Series('fold_benefit', resid[:, 0] * facing * own), pl.Series('call_cost', resid[:, 2] * facing * (1 - own)),
        pl.Series('raise_weak', resid[:, 3] * (1 - own)))
    return d.group_by('pair_id', 'hand_id').agg(pl.col('surprise').max().alias('surprise_max'), pl.col('surprise').mean().alias('surprise_mean'), pl.col('surprise').sum().alias('surprise_sum'),
                                               pl.col('surprise_facing').max().alias('surprise_facing_max'), pl.col('fold_benefit').max().alias('fold_benefit_max'),
                                               pl.col('call_cost').max().alias('call_cost_max'), pl.col('raise_weak').max().alias('raise_weak_max'), pl.len().alias('n_actions'))


def rank_metrics(hs, evidence):
    """hs: per (pair, hand) scores for positive pairs; evidence: listed (pair, hand). Within-pair ranking quality."""
    hs = hs.join(evidence.with_columns(pl.lit(1).alias('listed')), on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('listed').fill_null(0))
    res = {}
    for col in [c for c in hs.columns if c not in ('pair_id', 'hand_id', 'listed', 'n_actions')]:
        aucs, rec5, rec10 = [], [], []
        for pid, g in hs.group_by('pair_id'):
            y = g['listed'].to_numpy(); s = g[col].to_numpy()
            if y.sum() == 0 or y.sum() == len(y): continue
            aucs.append(roc_auc_score(y, s))
            order = np.argsort(-s, kind='mergesort')
            rec5.append(y[order[:5]].sum() / y.sum()); rec10.append(y[order[:10]].sum() / y.sum())
        res[col] = {'mean_pair_auc': float(np.mean(aucs)), 'pooled_auc': float(roc_auc_score(hs['listed'].to_numpy(), hs[col].to_numpy())), 'recall_at_5': float(np.mean(rec5)), 'recall_at_10': float(np.mean(rec10)), 'pairs': len(aucs)}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, default=1); args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    plan = json.loads((CAE / 'policy-plan.json').read_text())
    inputs = plan['inputs']
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    evidence = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    held = set(pl.read_csv(ORIGINAL / 'reports' / 'tabicl-confirmation' / f'fold{args.fold}' / 'held-pairs.csv')['table_id'])
    data = pl.read_parquet(sorted((CAE / 'action-cache').glob('*.parquet'))).join(labels.select('pair_id', 'label', 'behavior_family'), on='pair_id')
    train_all = data.filter((pl.col('label') == 0) & ~pl.col('table_id').is_in(sorted(held)))
    rng = np.random.default_rng(55193)
    ref = train_all.with_columns(pl.Series('r', rng.random(train_all.height))).sort('pair_id', 'r').group_by('pair_id', maintain_order=True).head(128)
    if ref.height > 100000: ref = ref.sample(n=100000, seed=55193)
    held_neg = data.filter((pl.col('label') == 0) & pl.col('table_id').is_in(sorted(held)))
    held_pos = data.filter((pl.col('label') == 1) & pl.col('table_id').is_in(sorted(held)))
    res = {'fold': args.fold, 'held_pools': len(held), 'rows': {'reference_train': ref.height, 'capacity_train': train_all.height, 'held_neg': held_neg.height, 'held_pos': held_pos.height, 'held_pos_pairs': held_pos['pair_id'].n_unique()}}
    specs = {'reference': (ref, dict(plan['parameters'])), 'capacity': (train_all, dict(plan['parameters'], n_estimators=600, num_leaves=63, min_child_samples=40, learning_rate=.05, colsample_bytree=.8))}
    for name, (tr, params) in specs.items():
        params['n_jobs'] = 2
        t = time.perf_counter()
        clf = lgb.LGBMClassifier(**params); clf.fit(tr.select(inputs).to_numpy().astype(np.float32), tr['action_class'].to_numpy())
        pn = clf.predict_proba(held_neg.select(inputs).to_numpy().astype(np.float32)); pp = clf.predict_proba(held_pos.select(inputs).to_numpy().astype(np.float32))
        yn = held_neg['action_class'].to_numpy()
        r = {'seconds': round(time.perf_counter() - t, 1), 'held_neg_logloss': float(log_loss(yn, pn, labels=[0, 1, 2, 3])), 'held_neg_accuracy': float((pn.argmax(1) == yn).mean()),
             'held_neg_logloss_by_class': {str(k): float(-np.log(np.maximum(pn[yn == k, k], 1e-5)).mean()) for k in range(4)}}
        hs = hand_scores(held_pos, pp)
        r['evidence_ranking'] = rank_metrics(hs, evidence)
        hsl = hs.join(evidence.with_columns(pl.lit(1).alias('listed')), on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('listed').fill_null(0)).join(labels.select('pair_id', 'behavior_family'), on='pair_id')
        r['surprise_max_mean'] = {'listed': float(hsl.filter(pl.col('listed') == 1)['surprise_max'].mean()), 'unlisted': float(hsl.filter(pl.col('listed') == 0)['surprise_max'].mean())}
        r['by_family_mean_pair_auc_surprise_max'] = {}
        for fam, g in hsl.group_by('behavior_family'):
            a = [roc_auc_score(gg['listed'].to_numpy(), gg['surprise_max'].to_numpy()) for _, gg in g.group_by('pair_id') if 0 < gg['listed'].sum() < gg.height]
            r['by_family_mean_pair_auc_surprise_max'][fam[0]] = float(np.mean(a))
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != 'evidence_ranking'}), flush=True)
        for col, m in r['evidence_ranking'].items():
            print('   ', col, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
    write_json(OUT / f'step24-fold{args.fold}.json', res)


if __name__ == '__main__':
    main()
