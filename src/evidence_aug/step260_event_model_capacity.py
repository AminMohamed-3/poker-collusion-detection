"""STEP260: is the STEP254 MECHPLUS clean event model capacity-limited?  (development only, OOF on the step232 folds)

The registered STEP254 recipe is small (400 trees x 15 leaves, lr .03, one seed) because it was inherited from the
163-column STEP232 screen.  With 850 columns the same recipe may under-fit.  Arms (registration260.json written first):
    S3   the registered recipe, 3 seeds (232, 233, 234), probabilities averaged          (ensemble form: 3-seed bag)
    C1   1500 trees x 31 leaves, lr .015, min_child 20, colsample .3, seed 232             (ensemble form: single model)
    C1S3 = mean(C1, S3) -- no extra fit
Reference = oof_mechplus.npy (any .7607, DT .8853, SP .6626, CI .6671).
Gate G260: an arm is "worth a production re-score" iff any AP >= ref + .010 AND CI AP >= ref + .010 AND no family below ref.
Usage: python step260_event_model_capacity.py S3|C1
"""
from __future__ import annotations
import os
os.environ.setdefault('POLARS_MAX_THREADS', '3')
import gc, json, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, r'C:\Data\Projects\Money Making\work\poker\Claude Version\src')
from evidence_aug import step254_eval as E  # noqa: E402  (import-safe: main() is guarded)

OUT = E.CV + r'\experiments\step260_event_capacity_20260918'
NJ = int(os.environ.get('STEP260_NJOBS', '4'))
RECIPES = {
    'S3': [dict(E.P, n_jobs=NJ, random_state=s) for s in (232, 233, 234)],
    'C1': [dict(n_estimators=1500, num_leaves=31, learning_rate=.015, min_child_samples=20, subsample=.8, subsample_freq=1,
                colsample_bytree=.3, reg_lambda=5.0, n_jobs=NJ, verbosity=-1, random_state=232)],
}


def main() -> None:
    arm = sys.argv[1]
    assert os.path.exists(OUT + r'\registration260.json'), 'register first'
    h, feats = E.load()
    fx = pl.read_parquet(E.OUT + r'\step254_features.parquet')
    MECH = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
    h = h.join(fx, on=['pair_id', 'hand_id'], how='left')
    f2 = pl.read_parquet(E.S237 + r'\step237_features.parquet')
    h = h.join(f2, on=['pair_id', 'hand_id'], how='left')
    cols = feats + E.F237 + MECH
    assert len(cols) == 850 and h[MECH[0]].null_count() == 0 and h[E.F237[0]].null_count() == 0
    X = np.empty((h.height, len(cols)), np.float32)
    for i, c in enumerate(cols):
        X[:, i] = h[c].to_numpy()
    y_listed = h['listed'].to_numpy(); lbl = h['label'].to_numpy(); fold = h['fold'].to_numpy(); fam = h['behavior_family'].to_numpy()
    del h, fx, f2; gc.collect()
    ref = np.load(E.OUT + r'\oof_mechplus.npy')          # rows: any, DT, SP, CI
    res, oof = {}, []
    for hi, (name, posmask) in enumerate((('any', y_listed == 1), *[(f, (y_listed == 1) & (fam == f)) for f in E.FAMS])):
        pr = np.zeros(X.shape[0]); t0 = time.time()
        for P in RECIPES[arm]:
            for k in range(5):
                tr = (fold != k) & (posmask | (lbl == 0)); te = fold == k
                m = lgb.LGBMClassifier(**P).fit(X[tr], posmask[tr].astype(int))
                pr[te] += m.predict_proba(X[te])[:, 1] / len(RECIPES[arm])
                del m; gc.collect()
        ev = posmask | (lbl == 0); yy = posmask[ev].astype(int)
        out = {}
        for tag, ss in (('arm', pr[ev]), ('ref', ref[hi][ev]), ('mean_arm_ref', (pr[ev] + ref[hi][ev]) / 2)):
            neg = np.sort(ss[yy == 0])[::-1]
            out[tag] = {'auc': round(float(roc_auc_score(yy, ss)), 5), 'ap': round(float(average_precision_score(yy, ss)), 4),
                        **{f'tpr@{f}': round(float((ss[yy == 1] > neg[int(f * len(neg))]).mean()), 4) for f in (1e-4, 1e-3, 1e-2)}}
        out['seconds'] = round(time.time() - t0, 1)
        res[name] = out; oof.append(pr)
        print(arm, name, json.dumps(out), flush=True)
    np.save(OUT + f'\\oof_{arm}.npy', np.stack(oof))
    json.dump({arm: res}, open(OUT + f'\\results_{arm}.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
