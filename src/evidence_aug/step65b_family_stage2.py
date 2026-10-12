"""Step 65b: ablations of the step-65 finding (per-family second-stage evidence model with structural features lifts
coordinated-isolation MAP@5 .40 -> .56 on the audit set) and a hybrid evaluation with pool bootstrap.
Per family, GroupKFold(5, by table), 3 seeds, LGBMClassifier (step-14 CLF unless stated):
  F0  chrono + oof (+rank)                         -> is the gain the per-family split alone?
  F1  struct + chrono + oof                        -> step 65 (repeat)
  F1t F1 restricted to the top-20 ranker candidates -> is the all-hands scope needed?
  F3  struct + chrono                              -> without the 382-feature ranker
  F1c F1 with more capacity (num_leaves 15, 300 trees)
  P1c pooled F1 with more capacity + family columns -> can a pooled model learn it?
Hybrid H = per-family F1 for coordinated_isolation, pooled step-14 score+chrono top-20 re-ranker for the other two;
paired pool bootstrap vs the pooled re-ranker (.5447), per family and overall; gain features per family.
Training-only audit set (162 queries); development labels for validation only.
Output: experiments/label_process_20260911/audit-family-stage2.json"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import write_json
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import per_pair_map
from evidence_aug.step14_audit_truncation import OUT, TOP, CLF
from evidence_aug.step65_structural_evidence import oof_scores, summarize, SEEDS, log
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

CLF2 = dict(CLF, num_leaves=15, max_depth=5, n_estimators=300, min_child_samples=30)


def fam_oof(d, cols, y, groups, params, top_only=False):
    pred = np.full(d.height, -1e6)
    for fam in FAMILIES:
        mask = (d['behavior_family'] == fam).to_numpy()
        if top_only:
            mask = mask & (d['oof_rank'] <= TOP).to_numpy()
        X = d.select(cols).to_numpy().astype(np.float64)[mask]
        p = np.zeros(mask.sum())
        for seed in SEEDS:
            for tr, te in GroupKFold(n_splits=5).split(X, y[mask], groups[mask]):
                clf = lgb.LGBMClassifier(**params, random_state=seed); clf.fit(X[tr], y[mask][tr]); p[te] += clf.predict_proba(X[te])[:, 1] / len(SEEDS)
        pred[mask] = p
    return pred


def gain_features(d, cols, y, fam, params, k=12):
    mask = (d['behavior_family'] == fam).to_numpy()
    X = d.select(cols).to_numpy().astype(np.float64)[mask]
    clf = lgb.LGBMClassifier(**params, random_state=11); clf.fit(X, y[mask])
    g = clf.booster_.feature_importance('gain'); order = np.argsort(g)[::-1][:k]
    return [(cols[i], round(float(g[i] / g.sum()), 3)) for i in order]


def boot(a, b, ptab, pools, n=2000, seed=652):
    rng = np.random.default_rng(seed); by_pool = {}
    for p in a:
        by_pool.setdefault(ptab[p], []).append(b[p] - a[p])
    deltas = []
    for _ in range(n):
        pick = rng.choice(pools, len(pools), replace=True)
        vals = [v for t in pick for v in by_pool.get(t, [])]
        deltas.append(np.mean(vals))
    return {'mean_delta': float(np.mean([b[p] - a[p] for p in a])), 'ci95': [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
            'pairs_better': int(sum(b[p] > a[p] for p in a)), 'pairs_worse': int(sum(b[p] < a[p] for p in a)), 'n': len(a)}


def main():
    t0 = time.time()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet'); feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    d = d.with_columns(*[(pl.col('behavior_family') == f).cast(pl.Float64).alias('family_' + f) for f in FAMILIES])
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]; famcols = ['family_' + f for f in FAMILIES]; chrono = ['chrono_rank', 'chrono_frac', 'n_hands']; sc = ['oof', 'oof_rank']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows()); ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows()); pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float); groups = d['table_id'].to_numpy()
    key = d.select('pair_id', 'hand_id', 'table_id')
    res = {'training_only': True, 'queries': len(truths), 'sets': {}}
    # pooled step-14 re-ranker (reference)
    cands_mask = (d['oof_rank'] <= TOP).to_numpy()
    Xc = d.select(sc + chrono + famcols).to_numpy().astype(np.float64)[cands_mask]
    pc = np.full(d.height, -1e6); pc[cands_mask] = oof_scores(Xc, y[cands_mask], groups[cands_mask])
    pc = np.where(cands_mask, pc + 10.0, d['oof'].to_numpy() - 1e6)
    ref, ref_map = summarize(key.with_columns(pl.Series('score', pc)), truths, labels)
    res['sets']['pooled_score_chrono_top20'] = ref
    log('ref', json.dumps({k: (round(v, 4) if isinstance(v, float) else {f: round(x, 4) for f, x in v.items()}) for k, v in ref.items()}))
    maps = {'ref': ref_map}
    specs = {'F0_chrono_oof': (chrono + sc, CLF, False), 'F1_struct_chrono_oof': (struct + chrono + sc, CLF, False), 'F1t_top20_only': (struct + chrono + sc, CLF, True),
             'F3_struct_chrono': (struct + chrono, CLF, False), 'F1c_capacity': (struct + chrono + sc, CLF2, False)}
    for name, (cols, params, top_only) in specs.items():
        pred = fam_oof(d, cols, y, groups, params, top_only)
        if top_only:
            pred = np.where(cands_mask, pred + 10.0, d['oof'].to_numpy() - 1e6)
        s, m = summarize(key.with_columns(pl.Series('score', pred)), truths, labels); s['delta_vs_ref'] = s['map5'] - ref['map5']; s['delta_by_family'] = {f: s['by_family'][f] - ref['by_family'][f] for f in FAMILIES}
        res['sets'][name] = s; maps[name] = m
        log(name, json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}}))
    # pooled with capacity
    X = d.select(struct + chrono + sc + famcols).to_numpy().astype(np.float64)
    pred = oof_scores(X, y, groups) if False else None
    p = np.zeros(d.height)
    for seed in SEEDS:
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = lgb.LGBMClassifier(**CLF2, random_state=seed); clf.fit(X[tr], y[tr]); p[te] += clf.predict_proba(X[te])[:, 1] / len(SEEDS)
    s, m = summarize(key.with_columns(pl.Series('score', p)), truths, labels); s['delta_vs_ref'] = s['map5'] - ref['map5']; res['sets']['P1c_pooled_capacity'] = s; maps['P1c'] = m
    log('P1c_pooled_capacity', json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}}))
    # hybrids: CI from a per-family set, DT/SP from the reference
    res['hybrid'] = {}
    for name in ['F0_chrono_oof', 'F1_struct_chrono_oof', 'F1t_top20_only', 'F1c_capacity']:
        h = {p: (maps[name][p] if labels[p] == 'coordinated_isolation' else ref_map[p]) for p in ref_map}
        s = {'map5': float(np.mean(list(h.values()))), 'by_family': {f: float(np.mean([v for p, v in h.items() if labels[p] == f])) for f in FAMILIES}}
        s['delta_vs_ref'] = s['map5'] - ref['map5']
        s['bootstrap_all'] = boot(ref_map, h, ptab, pools)
        ci = {p: v for p, v in h.items() if labels[p] == 'coordinated_isolation'}
        s['bootstrap_CI_only'] = boot({p: ref_map[p] for p in ci}, ci, ptab, pools)
        res['hybrid'][name] = s
        log('hybrid', name, json.dumps({'map5': round(s['map5'], 4), 'delta': round(s['delta_vs_ref'], 4), 'ci95': [round(x, 4) for x in s['bootstrap_all']['ci95']], 'CI_only': [round(x, 4) for x in s['bootstrap_CI_only']['ci95']]}))
    # per-family gain features for F1
    res['gain_F1'] = {fam: gain_features(d, struct + chrono + sc, y, fam, CLF) for fam in FAMILIES}
    for fam in FAMILIES:
        log('gain', fam, res['gain_F1'][fam][:8])
    # listed-vs-unlisted structure for CI: listed rate by (pf_open_pair, pf_partner_fold, pf_others_fold>=k)
    ci = d.filter(pl.col('behavior_family') == 'coordinated_isolation')
    tab = ci.group_by('pf_open_pair', 'pf_partner_fold', 'pf_partner_none', (pl.col('pf_others_fold') >= 3).alias('others_fold_ge3')).agg(pl.len().alias('n'), pl.col('relevance').mean().alias('listed_rate'), pl.col('relevance').sum().alias('listed')).sort('listed_rate', descending=True)
    res['CI_listed_rate_by_structure'] = tab.to_dicts()
    log(tab)
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-family-stage2.json', res)


if __name__ == '__main__':
    main()
