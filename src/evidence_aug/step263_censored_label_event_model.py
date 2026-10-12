"""STEP263: Amin's list-length idea (2026-09-18), in the form the data supports.

Amin: "some of the evidence lists are 5, some 3, some 4 -- see if the model could fill in the 4th and 5th for the lists
where only 3 exist, and maybe train on that."
Premise check (scratchpad short_lists.py, recorded in the registration): the 32 SHORT lists (11 x 3, 21 x 4 hands) are
COMPLETE -- their unlisted hands carry +.19 / +.34 excess hands per pair above the honest q99.9, against **+4.17** for the 340
capped 5-hand lists.  So there is nothing to fill in on the short lists, but the measurement hands us two label sets the
event model currently throws away (step254_eval excludes every unlisted hand of a positive pair from training):
  HN   certified WITHIN-PAIR negatives: the 2,729 unlisted hands of the short-list pairs (a colluding pair's ordinary hands);
  PP   "fill in the 6th-13th": unlisted hands of the capped lists that a NESTED out-of-fold model puts above the honest
       q99.9 of the pair's own family head (about 97 % precise by the excess count) become extra positives.
Arms: HN, PP, HNPP on the registered MECHPLUS recipe (850 columns, E.P, step232 table folds).  Ensemble form: single model per
head and fold, exactly as the reference it is compared with (oof_mechplus.npy).
Nesting: the model that scores fold k may only see pseudo-labels of fold j chosen by a model trained WITHOUT folds j and k
(10 selection models per family head).  Evaluation positives are always the LISTED hands; negatives the confirmed
non-target pairs' hands.  Development phase only (asserted in E.load()).

Usage:  python step263_censored_label_event_model.py select | fit HN|PP|HNPP
"""
from __future__ import annotations
import os
os.environ.setdefault('POLARS_MAX_THREADS', '3')
import gc, itertools, json, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, r'C:\Data\Projects\Money Making\work\poker\Claude Version\src')
from evidence_aug import step254_eval as E  # noqa: E402

OUT = E.CV + r'\experiments\step263_censored_labels_20260918'
NJ = int(os.environ.get('STEP263_NJOBS', '3'))
P = dict(E.P, n_jobs=NJ)
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}


def load():
    h, feats = E.load()
    fx = pl.read_parquet(E.OUT + r'\step254_features.parquet')
    MECH = [c for c in fx.columns if c not in ('pair_id', 'hand_id')]
    h = h.join(fx, on=['pair_id', 'hand_id'], how='left').join(pl.read_parquet(E.S237 + r'\step237_features.parquet'), on=['pair_id', 'hand_id'], how='left')
    cols = feats + E.F237 + MECH
    assert len(cols) == 850 and h[MECH[0]].null_count() == 0 and h[E.F237[0]].null_count() == 0
    nl = h.group_by('pair_id').agg(pl.col('listed').sum().alias('n_listed'))
    h = h.join(nl, on='pair_id', how='left')
    X = np.empty((h.height, len(cols)), np.float32)
    for i, c in enumerate(cols):
        X[:, i] = h[c].to_numpy()
    meta = {k: h[k].to_numpy() for k in ('listed', 'label', 'fold', 'behavior_family', 'n_listed')}
    del h, fx; gc.collect()
    return X, meta


def select() -> None:
    X, M = load()
    listed, lbl, fold, fam, nl = M['listed'], M['label'], M['fold'], M['behavior_family'], M['n_listed']
    clean = lbl == 0
    out, log = {}, {}
    for f in E.FAMS:
        posmask = (listed == 1) & (fam == f)
        cand = (lbl == 1) & (listed == 0) & (nl == 5) & (fam == f)
        sel = np.zeros((5, X.shape[0]), bool)          # sel[k] = pseudo-positives usable by the model that scores fold k
        t0 = time.time()
        for j, k in itertools.combinations(range(5), 2):
            tr = (fold != j) & (fold != k) & (posmask | clean)
            m = lgb.LGBMClassifier(**P).fit(X[tr], posmask[tr].astype(int))
            held = (fold == j) | (fold == k)
            thr = float(np.quantile(m.predict_proba(X[held & clean])[:, 1], .999))     # honest q99.9 of THIS model, on its held-out clean hands
            for a, b in ((j, k), (k, j)):              # pseudo-labels of fold a, usable by the model that scores fold b
                ix = np.flatnonzero(cand & (fold == a))
                p = m.predict_proba(X[ix])[:, 1]
                sel[b, ix[p > thr]] = True
            del m; gc.collect()
        out[SHORT[f]] = sel
        log[SHORT[f]] = {'candidates': int(cand.sum()), 'pairs': None, 'selected_per_scoring_fold': sel.sum(1).tolist(), 'listed_positives': int(posmask.sum()), 'seconds': round(time.time() - t0, 1)}
        print('select', f, json.dumps(log[SHORT[f]]), flush=True)
    np.savez_compressed(OUT + r'\pp_selection.npz', **out)
    json.dump(log, open(OUT + r'\selection263.json', 'w', encoding='utf-8'), indent=1)


def fit(arm: str) -> None:
    X, M = load()
    listed, lbl, fold, fam, nl = M['listed'], M['label'], M['fold'], M['behavior_family'], M['n_listed']
    clean = lbl == 0
    hn = (lbl == 1) & (listed == 0) & (nl > 0) & (nl < 5) if 'HN' in arm else np.zeros(X.shape[0], bool)
    S = np.load(OUT + r'\pp_selection.npz') if 'PP' in arm else None
    ref = np.load(E.OUT + r'\oof_mechplus.npy')
    res, oof = {}, []
    heads = (('any', None), *[(f, f) for f in E.FAMS])
    for hi, (name, f) in enumerate(heads):
        posmask = (listed == 1) if f is None else (listed == 1) & (fam == f)
        pr = np.full(X.shape[0], np.nan); t0 = time.time(); n_pp = []
        for k in range(5):
            if S is None:
                pp = np.zeros(X.shape[0], bool)
            else:
                pp = S[SHORT[f]][k] if f is not None else (S['DT'][k] | S['SP'][k] | S['CI'][k])
            assert not (pp & (fold == k)).any() and not (pp & posmask).any()
            y = posmask | pp
            tr = (fold != k) & (y | clean | hn)
            m = lgb.LGBMClassifier(**P).fit(X[tr], y[tr].astype(int))
            pr[fold == k] = m.predict_proba(X[fold == k])[:, 1]
            n_pp.append(int((pp & (fold != k)).sum()))
            del m; gc.collect()
        ev = posmask | clean; yy = posmask[ev].astype(int)
        out = {'pseudo_positives_per_fold_model': n_pp, 'hard_negatives': int(hn.sum())}
        for tag, ss in (('arm', pr[ev]), ('ref', ref[hi][ev])):
            neg = np.sort(ss[yy == 0])[::-1]
            out[tag] = {'auc': round(float(roc_auc_score(yy, ss)), 5), 'ap': round(float(average_precision_score(yy, ss)), 4),
                        **{f'tpr@{q}': round(float((ss[yy == 1] > neg[int(q * len(neg))]).mean()), 4) for q in (1e-4, 1e-3, 1e-2)}}
        # within-pair discrimination where the truth is COMPLETE: listed vs unlisted hands of the short-list pairs (OOF)
        sp = (lbl == 1) & (nl > 0) & (nl < 5) if f is None else (lbl == 1) & (nl > 0) & (nl < 5) & (fam == f)
        if sp.any() and listed[sp].sum() > 0:
            out['short_pairs_within_AUC'] = {'arm': round(float(roc_auc_score(listed[sp], pr[sp])), 4), 'ref': round(float(roc_auc_score(listed[sp], ref[hi][sp])), 4),
                                             'arm_AP': round(float(average_precision_score(listed[sp], pr[sp])), 4), 'ref_AP': round(float(average_precision_score(listed[sp], ref[hi][sp])), 4), 'n_listed': int(listed[sp].sum())}
        # and on the capped lists: listed vs unlisted inside the pair (unlisted holds censored events: chronology decides, not event-ness)
        fp = (lbl == 1) & (nl == 5) if f is None else (lbl == 1) & (nl == 5) & (fam == f)
        out['full_pairs_within_AUC'] = {'arm': round(float(roc_auc_score(listed[fp], pr[fp])), 4), 'ref': round(float(roc_auc_score(listed[fp], ref[hi][fp])), 4)}
        out['seconds'] = round(time.time() - t0, 1)
        res[name] = out; oof.append(pr)
        print(arm, name, json.dumps(out), flush=True)
    np.save(OUT + f'\\oof_{arm}.npy', np.stack(oof))
    json.dump({arm: res}, open(OUT + f'\\results_{arm}.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    assert os.path.exists(OUT + r'\registration263.json'), 'register first'
    if sys.argv[1] == 'select':
        select()
    else:
        fit(sys.argv[2])
