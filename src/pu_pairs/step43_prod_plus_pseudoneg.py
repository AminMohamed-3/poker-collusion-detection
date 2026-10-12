"""Step 43: the production pair recipe refitted WITH certified pseudo-negatives at scale (OBSERVATIONS 14.24/14.25).

Why. Step 40 showed a false-positive population created by short histories (unlabelled >= .5 rises 59 -> 103 on identical
fold-1 pairs when histories are cut to 60 %). Step 41 (a second stage trained on one fold's prefix frames with ~63
positives) and step 42 (length-aware recalibration of the frozen score) both lost ranking quality on the undisclosed
positives. The remaining principled remedy keeps every production ingredient (all labelled pairs at full / 70 % / 40 %
views, 4-class LightGBM, 897 features, the same seed and parameters) and only ADDS negatives: unlabelled pairs of
pools outside the test fold whose full-history production risk is < NEG_MAX, at their full view and their 60 %-prefix
view. Nothing about any labelled pair changes; no label of an unlabelled pair is asserted beyond "the production model
itself is confident it is not collusion on the full history".

Test fold 1 (100 held pools, never seen by any model here). Pseudo-negatives come from fold-2 pools minus fold-1 pools
(71 tables; their labelled pairs are already in the fold-1 training pools, so adding their unlabelled pairs leaks nothing
about fold 1). Control = refit of the production recipe (parity-checked against Codex's saved fold-1 held predictions).

Metrics on fold-1 pools, full view and 60 %-prefix view, for control and each pseudo-weight arm:
  labelled AP (official stable), 1 %-prevalence AP, unlabelled >= .5 per table, prefix-only-high (prefix >= .5 and the
  SAME model's full-history risk < .1), top-K composition of unlabelled prefix ranking against (a) the frozen fold-1
  model's full-history risk and (b) the arm's own full-history risk, AUC of the prefix score for full-high vs full-low
  under both teachers, and prefix/full consistency (Jaccard of top-100 unlabelled sets).
Usage: step43_prod_plus_pseudoneg.py --test-fold 1 --neg-fold 2 --weights 0.3333 0.05
Output: experiments/pu_pairs_20260911/step43_test{F}.json (+ -scores.parquet)
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl
import joblib
import lightgbm as lgb
from pu_pairs.build_allpairs_fold import EXP, CONF, fold_pools, extra_aggregate
from pu_pairs.score_allpairs_fold import average_precision
from pu_pairs.step41_short_view_pu import weighted_ap
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, TRAIN_VIEWS, MODEL_PARAMS, thin, aggregate, matrix  # noqa: E402
from tail_pair_experiment import tails  # noqa: E402

MIN_SH = 38
NEG_MAX = 0.02
KEYS = ['pair_id', 'table_id', 'shared_hands']


def production_train_frame(test_fold, features):
    """Exact production training frame for the fold model: labelled pairs outside the held pools, full + 70 % + 40 % views."""
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    names = [c for c in raw.columns if c not in ['pair_id', 'hand_id', 'table_id']]
    events = [c for c in names if c.startswith('v2_')]
    ca = pl.read_parquet(CONF / f'conditional-fold{test_fold}-v1' / 'policy-screen' / 'hand-features.parquet')
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
    held = fold_pools(test_fold)
    train = pl.concat([full.filter(~pl.col('table_id').is_in(held))] + [v.filter(~pl.col('table_id').is_in(held)) for v in views])
    missing = [f for f in features if f not in train.columns]
    assert not missing, missing[:5]
    return train


def load_frames(dirname, fold, features):
    out = EXP / dirname
    frame = pl.concat([pl.read_parquet(b, columns=list(dict.fromkeys(KEYS + features))) for b in sorted(out.glob('batch_*-pairs.parquet'))]).sort('pair_id')
    pairs = pl.read_csv(out / 'pairs.csv').select('pair_id', 'label', 'behavior_family')
    frame = frame.join(pairs, on='pair_id', validate='1:1')
    frozen = pl.read_parquet(out / 'scores.parquet').select('pair_id', pl.col('risk').alias('risk_frozen'))
    frame = frame.join(frozen, on='pair_id', how='left')
    return frame.filter(pl.col('shared_hands') >= MIN_SH)


def ap_sets(score, pos, neg, pid):
    m = pos | neg
    return average_precision(pos[m].astype(int), score[m], pid[m])


def evaluate(full_f, pre_f, risk_full, risk_pre, teacher_full, n_tables):
    """full_f / pre_f: frames (same universe filtering applied separately); risks aligned to each frame."""
    r = {}
    for tag, f, s in (('full', full_f, risk_full), ('prefix', pre_f, risk_pre)):
        lab = f['label'].to_numpy(); pid = f['pair_id'].to_numpy(); pos = lab == 1; neg = lab == 0; unl = lab < 0
        d = {'ap_labelled': ap_sets(s, pos, neg, pid)}
        m = pos | neg; w = np.where(pos[m], 0.01 / pos[m].mean(), 0.99 / (1 - pos[m].mean()))
        d['ap_labelled_1pct'] = weighted_ap(pos[m].astype(int), s[m], pid[m], w)
        d['ap_pos_vs_all_unlabelled'] = ap_sets(s, pos, unl, pid)
        for t in (.9, .5):
            d[f'ge_{t}'] = {'positives': int((s[pos] >= t).sum()), 'labelled_negatives': int((s[neg] >= t).sum()), 'unlabelled': int((s[unl] >= t).sum()), 'unlabelled_per_table': round(float((s[unl] >= t).sum()) / n_tables, 3)}
        d['positives_risk_q10_q50'] = [float(v) for v in np.quantile(s[pos], [.1, .5])]
        r[tag] = d
    # prefix ranking of unlabelled pairs against two teachers: frozen full-history risk and the arm's own full-history risk
    pre_ids = pre_f['pair_id'].to_numpy(); lab = pre_f['label'].to_numpy(); unl = lab < 0
    own_full = dict(zip(full_f['pair_id'].to_numpy(), risk_full))
    own = np.array([own_full.get(p, np.nan) for p in pre_ids])
    for tname, tvals in (('frozen_teacher', teacher_full), ('own_teacher', own)):
        ok = unl & ~np.isnan(tvals)
        hi = ok & (tvals >= .5); lo = ok & (tvals < .1); mid = ok & ~hi & ~lo
        sc = np.where(ok, risk_pre, -np.inf); order = np.argsort(-sc, kind='mergesort')
        d = {'n_full_high': int(hi.sum()), 'n_full_low': int(lo.sum())}
        for K in (int(hi.sum()), 100, 200):
            top = order[:K]; d[f'top{K}'] = {'hi': int(hi[top].sum()), 'mid': int(mid[top].sum()), 'lo': int(lo[top].sum())}
        from sklearn.metrics import roc_auc_score
        d['auc_prefix_fullhi_vs_fulllo'] = round(float(roc_auc_score(np.r_[np.ones(hi.sum()), np.zeros(lo.sum())], np.r_[risk_pre[hi], risk_pre[lo]])), 4)
        hp = ok & (risk_pre >= .5)
        d['prefix_ge_.5_unlabelled'] = {'n': int(hp.sum()), 'full_ge_.5': int((hp & hi).sum()), 'full_in_.1_.5': int((hp & mid).sum()), 'full_lt_.1': int((hp & lo).sum())}
        d['full_high_recall_prefix_ge_.5'] = int((hi & (risk_pre >= .5)).sum())
        r[f'prefix_vs_{tname}'] = d
    # consistency: top-100 unlabelled sets, prefix vs own full
    top_pre = set(pre_ids[unl][np.argsort(-risk_pre[unl], kind='mergesort')[:100]])
    fl = full_f['label'].to_numpy() < 0; fid = full_f['pair_id'].to_numpy()
    top_full = set(fid[fl][np.argsort(-risk_full[fl], kind='mergesort')[:100]])
    r['jaccard_top100_unlabelled_prefix_vs_own_full'] = round(len(top_pre & top_full) / len(top_pre | top_full), 3)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-fold', type=int, default=1)
    ap.add_argument('--neg-fold', type=int, default=2)
    ap.add_argument('--weights', type=float, nargs='*', default=[1 / 3, 0.05])
    args = ap.parse_args()
    t0 = time.time()
    bundle = joblib.load(CONF / 'results' / f'fold{args.test_fold}' / 'conditional-pair.joblib'); features = list(bundle['features'])
    train = production_train_frame(args.test_fold, features)
    y_lab = np.array([LABELS[f] for f in train['behavior_family']]); w_lab = np.full(train.height, 1 / 3)
    X_lab = matrix(train, features)
    print('labelled training rows', train.height, 'positives', int((y_lab > 0).sum()), 'elapsed', round(time.time() - t0), flush=True)
    # test frames (fold pools)
    full_f = load_frames(f'fold{args.test_fold}', args.test_fold, features)
    pre_f = load_frames(f'fold{args.test_fold}-prefix60', args.test_fold, features)
    X_full = matrix(full_f, features); X_pre = matrix(pre_f, features)
    n_tables = full_f['table_id'].n_unique()
    teacher_full_map = dict(zip(full_f['pair_id'].to_numpy(), full_f['risk_frozen'].to_numpy()))
    teacher_full = np.array([teacher_full_map.get(p, np.nan) for p in pre_f['pair_id'].to_numpy()])
    # pseudo-negatives from neg-fold pools minus test pools
    held = set(fold_pools(args.test_fold))
    negs = []
    for dirname in (f'fold{args.neg_fold}', f'fold{args.neg_fold}-prefix60'):
        f = load_frames(dirname, args.neg_fold, features)
        full_scores = pl.read_parquet(EXP / f'fold{args.neg_fold}' / 'scores.parquet').select('pair_id', pl.col('risk').alias('risk_full'))
        f = f.join(full_scores, on='pair_id', how='left').filter(~pl.col('table_id').is_in(sorted(held)) & (pl.col('label') < 0) & (pl.col('risk_full') < NEG_MAX))
        negs.append(f.select(list(dict.fromkeys(KEYS + features))))
        print(dirname, 'pseudo-negatives', f.height, 'tables', f['table_id'].n_unique(), flush=True)
    neg = pl.concat(negs); X_neg = matrix(neg, features); y_neg = np.zeros(neg.height, dtype=int)
    res = {'test_fold': args.test_fold, 'neg_fold': args.neg_fold, 'neg_max': NEG_MAX, 'params': {k: v for k, v in MODEL_PARAMS.items()},
           'train_labelled_rows': int(train.height), 'train_positive_rows': int((y_lab > 0).sum()), 'pseudo_negative_rows': int(neg.height), 'pseudo_tables': int(neg['table_id'].n_unique()),
           'test': {'tables': n_tables, 'full_pairs': full_f.height, 'prefix_pairs': pre_f.height}, 'arms': {}}
    scores = pre_f.select('pair_id', 'table_id', 'label', 'shared_hands', pl.col('risk_frozen').alias('frozen_prefix'))
    scores_full = full_f.select('pair_id', pl.col('risk_frozen').alias('frozen_full'))
    # frozen model as reference arm
    res['arms']['frozen'] = evaluate(full_f, pre_f, full_f['risk_frozen'].to_numpy(), pre_f['risk_frozen'].to_numpy(), teacher_full, n_tables)
    print('frozen', json.dumps(res['arms']['frozen']), flush=True)
    arms = [('control', 0.0)] + [(f'pseudo_w{w:g}', w) for w in args.weights]
    for name, w in arms:
        t1 = time.time()
        if w > 0:
            X = np.vstack([X_lab, X_neg]); y = np.r_[y_lab, y_neg]; sw = np.r_[w_lab, np.full(neg.height, w)]
        else:
            X, y, sw = X_lab, y_lab, w_lab
        model = lgb.LGBMClassifier(**MODEL_PARAMS); model.fit(X, y, sample_weight=sw)
        rf = 1 - model.predict_proba(X_full)[:, 0]; rp = 1 - model.predict_proba(X_pre)[:, 0]
        r = evaluate(full_f, pre_f, rf, rp, teacher_full, n_tables); r['fit_seconds'] = round(time.time() - t1, 1); r['rows'] = int(len(y))
        if name == 'control':
            r['parity_max_abs_diff_vs_frozen_full'] = float(np.max(np.abs(rf - full_f['risk_frozen'].to_numpy())))
        res['arms'][name] = r
        scores = scores.with_columns(pl.Series(f'{name}_prefix', rp)); scores_full = scores_full.with_columns(pl.Series(f'{name}_full', rf))
        print(name, json.dumps(r), flush=True)
    res['seconds'] = round(time.time() - t0, 1)
    tag = f'step43_test{args.test_fold}_neg{args.neg_fold}'
    write_json(EXP / f'{tag}.json', res)
    scores.join(scores_full, on='pair_id', how='left').write_parquet(EXP / f'{tag}-scores.parquet')
    print('saved', tag, res['seconds'])


if __name__ == '__main__':
    main()
