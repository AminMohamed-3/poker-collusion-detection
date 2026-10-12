"""Step 41: second-stage short-view pair model with reliable pseudo-labels (OBSERVATIONS 14.24).

Premise (from step 40). The production pair model was fitted on ~1,400 labelled pairs (about 1,100 negatives). On short
histories it must extrapolate the negative tail from a few hundred examples, while evaluation has ~112,000 candidates
whose histories are ~2/3 as long as development. Step 40 measures whether truncating a history to its first 60 % creates
new high scorers among unlabelled pairs (false positives created by length). This step asks whether a model that sees
the negative tail AT SCALE fixes it: train on every pair of the held pools of one fold, on the 60 %-prefix feature stack,
with pseudo-labels taken from the FULL-history production score of the same pair (unlabelled with full risk < NEG_MAX
are negatives; labelled positives, and optionally unlabelled with full risk >= POS_MIN, are positives; everything
in between is dropped), then test on the prefix frames of ANOTHER fold whose pools are disjoint.

Honesty notes. (1) The fold-k teacher model never saw fold-k pools, so test-fold pseudo-labels are clean. (2) The train
fold's teacher DID see the test fold's labelled pairs (it was fitted on all other pools), so pseudo-POSITIVES in the
training set carry a faint distillation of test-fold labels; the arm `pu_negonly` uses pseudo-NEGATIVES only (labelled
positives are the only positives) and is the clean comparison for labelled AP. (3) Label-free metrics (unlabelled
counts per table, prefix-only high scorers with full risk < .1) do not depend on any test label.

Metrics on the test fold prefix view (pairs with >= 38 prefix shared hands), for each arm:
  ap_labelled              positives vs labelled negatives (official stable AP, pair_id order)
  ap_labelled_1pct         same, reweighted to 1 % positive prevalence
  ap_pseudo                positives + unlabelled(full >= POS_MIN)  vs  labelled negatives + unlabelled(full < NEG_MAX)
  ap_all_unlabelled_neg    labelled positives vs everything else (undisclosed positives count against every arm equally)
  counts >= .5/.9/.99      positives / labelled negatives / unlabelled (and unlabelled per table)
  prefix_only_high         unlabelled >= .5 whose full-history risk is < .1  (the false-positive indicator)
  full_high_recall         unlabelled with full-history risk >= .5 that the arm also scores >= .5
Arms: first_stage (frozen production model on the prefix frames), labelled_only (LightGBM on the train fold's labelled
prefix pairs only), pu_negonly, pu_both, pu_negonly_stacked (+ first-stage prefix risk as a feature).
Usage: step41_short_view_pu.py --train-fold 2 --test-fold 1 --prefix 60
Output: experiments/pu_pairs_20260911/step41_train{A}_test{B}_prefix{P}.json (+ per-pair scores parquet)
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
from pu_pairs.build_allpairs_fold import EXP, CONF
from pu_pairs.score_allpairs_fold import average_precision
from pu_pairs.step32_crossphase_delta_model import PARAMS
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import matrix  # noqa: E402

MIN_SH = 38
NEG_MAX = 0.02
POS_MIN = 0.90
SEEDS = [7329, 11, 12]


def weighted_ap(y, s, ids, w):
    """Official stable ranking (pair_id order, mergesort on -score) with class reweighting: precision = weighted tp / weighted rank."""
    order0 = np.argsort(ids, kind='mergesort'); y = y[order0]; s = s[order0]; w = w[order0]
    order = np.argsort(-s, kind='mergesort'); ranked = y[order]; wr = w[order]
    tp = np.cumsum(wr * ranked); tot = np.cumsum(wr)
    return float(np.sum((tp / tot) * wr * ranked) / max(1e-12, (wr * ranked).sum()))


def load_view(fold, prefix):
    out = EXP / f'fold{fold}-prefix{prefix}'
    frame = pl.concat([pl.read_parquet(b) for b in sorted(out.glob('batch_*-pairs.parquet'))]).sort('pair_id')
    pairs = pl.read_csv(out / 'pairs.csv').select('pair_id', 'player_1', 'player_2', 'label', 'behavior_family')
    frame = frame.join(pairs, on='pair_id', validate='1:1')
    first = pl.read_parquet(out / 'scores.parquet').select('pair_id', pl.col('risk').alias('risk_first'))
    full = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet').select('pair_id', pl.col('shared_hands').alias('shared_full'), pl.col('risk').alias('risk_full'))
    frame = frame.join(first, on='pair_id', how='left').join(full, on='pair_id', how='left')
    frame = frame.filter((pl.col('shared_hands') >= MIN_SH) & pl.col('risk_full').is_not_null() & pl.col('risk_first').is_not_null())
    return frame


def pseudo(frame, use_pos):
    lab = frame['label'].to_numpy(); full = frame['risk_full'].to_numpy()
    y = np.full(len(lab), -1)
    y[lab == 1] = 1
    y[lab == 0] = 0
    unl = lab < 0
    y[unl & (full < NEG_MAX)] = 0
    if use_pos:
        y[unl & (full >= POS_MIN)] = 1
    return y


def fit_predict(Xtr, ytr, Xte, seeds=SEEDS):
    preds = []
    for s in seeds:
        clf = lgb.LGBMClassifier(**{**PARAMS, 'random_state': s})
        clf.fit(Xtr, ytr)
        preds.append(clf.predict_proba(Xte)[:, 1])
    return np.mean(preds, axis=0)


def evaluate(frame, score, n_tables):
    lab = frame['label'].to_numpy(); full = frame['risk_full'].to_numpy(); pid = frame['pair_id'].to_numpy()
    pos = lab == 1; neg = lab == 0; unl = lab < 0
    r = {}
    m = pos | neg
    r['ap_labelled'] = average_precision(pos[m].astype(int), score[m], pid[m])
    w = np.where(pos[m], 0.01 / pos[m].mean(), 0.99 / (1 - pos[m].mean()))
    r['ap_labelled_1pct'] = weighted_ap(pos[m].astype(int), score[m], pid[m], w)
    ps = pos | (unl & (full >= POS_MIN)); ng = neg | (unl & (full < NEG_MAX)); m2 = ps | ng
    r['ap_pseudo'] = average_precision(ps[m2].astype(int), score[m2], pid[m2])
    r['ap_all_unlabelled_neg'] = average_precision(pos.astype(int), score, pid)
    for t in (.99, .9, .5, .1):
        r[f'ge_{t}'] = {'positives': int((score[pos] >= t).sum()), 'labelled_negatives': int((score[neg] >= t).sum()), 'unlabelled': int((score[unl] >= t).sum()), 'unlabelled_per_table': round(float((score[unl] >= t).sum()) / n_tables, 3)}
    hi = unl & (score >= .5)
    r['prefix_only_high'] = {'n_ge_.5': int(hi.sum()), 'full_lt_.1': int((hi & (full < .1)).sum()), 'full_in_.1_.5': int((hi & (full >= .1) & (full < .5)).sum()), 'full_ge_.5': int((hi & (full >= .5)).sum())}
    fh = unl & (full >= .5)
    r['full_high_recall'] = {'n': int(fh.sum()), 'ge_.5': int((fh & (score >= .5)).sum()), 'ge_.1': int((fh & (score >= .1)).sum())}
    r['positives_risk_quantiles_10_25_50'] = [float(v) for v in np.quantile(score[pos], [.1, .25, .5])]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-fold', type=int, default=2)
    ap.add_argument('--test-fold', type=int, default=1)
    ap.add_argument('--prefix', type=int, default=60)
    args = ap.parse_args()
    t0 = time.time()
    tr = load_view(args.train_fold, args.prefix); te = load_view(args.test_fold, args.prefix)
    # Codex's folds are independent GroupShuffleSplit draws, so held pools overlap: drop shared tables from the TRAIN side only
    shared = set(tr['table_id'].unique()) & set(te['table_id'].unique())
    n_train_tables_before = tr['table_id'].n_unique()
    tr = tr.filter(~pl.col('table_id').is_in(sorted(shared)))
    assert not set(tr['table_id'].unique()) & set(te['table_id'].unique())
    bundle = joblib.load(CONF / 'results' / f'fold{args.train_fold}' / 'conditional-pair.joblib')
    feats = list(bundle['features'])
    Xtr = matrix(tr, feats); Xte = matrix(te, feats)
    n_tables = te['table_id'].n_unique()
    res = {'train_fold': args.train_fold, 'test_fold': args.test_fold, 'prefix': args.prefix, 'neg_max': NEG_MAX, 'pos_min': POS_MIN, 'seeds': SEEDS,
           'train': {'pairs': tr.height, 'tables': tr['table_id'].n_unique(), 'tables_before_overlap_drop': n_train_tables_before, 'shared_tables_dropped': len(shared)}, 'test': {'pairs': te.height, 'tables': n_tables}, 'n_features': len(feats), 'arms': {}}
    scores = te.select('pair_id', 'table_id', 'label', 'shared_hands', 'shared_full', 'risk_full', 'risk_first')
    # arm 0: frozen first stage on prefix frames
    res['arms']['first_stage'] = evaluate(te, te['risk_first'].to_numpy(), n_tables)
    print('first_stage', json.dumps(res['arms']['first_stage']), flush=True)
    arms = {}
    lab_tr = tr['label'].to_numpy()
    y_lab = np.where(lab_tr == 1, 1, np.where(lab_tr == 0, 0, -1))
    arms['labelled_only'] = (y_lab, False)
    arms['pu_negonly'] = (pseudo(tr, use_pos=False), False)
    arms['pu_both'] = (pseudo(tr, use_pos=True), False)
    arms['pu_negonly_stacked'] = (pseudo(tr, use_pos=False), True)
    for name, (y, stacked) in arms.items():
        keep = y >= 0
        Xa = Xtr if not stacked else np.c_[Xtr, tr['risk_first'].to_numpy().astype(np.float32)]
        Xb = Xte if not stacked else np.c_[Xte, te['risk_first'].to_numpy().astype(np.float32)]
        t1 = time.time()
        pred = fit_predict(Xa[keep], y[keep], Xb)
        r = evaluate(te, pred, n_tables)
        r['train_rows'] = {'positives': int((y[keep] == 1).sum()), 'negatives': int((y[keep] == 0).sum())}
        r['fit_seconds'] = round(time.time() - t1, 1)
        res['arms'][name] = r
        scores = scores.with_columns(pl.Series(name, pred))
        print(name, json.dumps(r), flush=True)
    res['seconds'] = round(time.time() - t0, 1)
    tag = f'step41_train{args.train_fold}_test{args.test_fold}_prefix{args.prefix}'
    write_json(EXP / f'{tag}.json', res)
    scores.write_parquet(EXP / f'{tag}-scores.parquet')
    print('saved', tag, res['seconds'])


if __name__ == '__main__':
    main()
