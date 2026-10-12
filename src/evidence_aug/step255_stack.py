"""STEP255 step 4 -- metrics, diversity readouts, the nested stack, the permutation control and the
G1 verdict. Reads the base predictions written by step255_arms.py / step255_torch.py.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import json
import time

import numpy as np
import polars as pl
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = r'C:\Data\Projects\Money Making\work\poker'
OUT = ROOT + r'\Claude Version\experiments\step255_sequence_view_20260917'
S232 = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
MODELS = ('any',) + FAMS
EPS = 1e-6


def lg(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def metrics(pos, elig, p, tag=''):
    yy = pos[elig].astype(int)
    ss = p[elig]
    neg = np.sort(ss[yy == 0])[::-1]
    out = {'auc': round(float(roc_auc_score(yy, ss)), 5),
           'ap': round(float(average_precision_score(yy, ss)), 4),
           'n_pos': int(yy.sum())}
    for fpr in (1e-4, 1e-3, 1e-2):
        out[f'tpr@{fpr:g}'] = round(float((ss[yy == 1] > neg[int(fpr * len(neg))]).mean()), 4)
    return out


def main() -> None:
    t0 = time.time()
    order = pl.read_parquet(OUT + r'\row_order255.parquet').with_row_index('rid')
    oof = pl.read_parquet(S232)
    j = order.join(oof, on=['pair_id', 'hand_id'], how='left').sort('rid')
    fold = j['fold'].to_numpy()
    lbl = j['label'].to_numpy()
    listed = j['listed'].to_numpy()
    fam = j['behavior_family'].to_numpy()
    pair = j['pair_id'].to_numpy()
    N = len(fold)
    honest = lbl == 0

    P = {}
    for f in ('preds255_plus.npz', 'preds255_N_lr_N_gbm.npz', 'preds255_T_gru.npz',
              'preds255_T_tx.npz'):
        if os.path.exists(OUT + '\\' + f):
            z = np.load(OUT + '\\' + f)
            for k in z.files:
                P[k] = z[k]
    seq_arms = [a for a in ('N_lr', 'N_gbm') if f'loo_{a}_any' in P]
    neural = [a for a in ('T_gru', 'T_tx') if f'loo_{a}_any' in P]
    has_T = bool(neural)
    print('arms present:', ['plus'] + seq_arms + neural, flush=True)

    posmask = {m: (listed == 1) if m == 'any' else ((listed == 1) & (fam == m)) for m in MODELS}
    elig = {m: posmask[m] | honest for m in MODELS}

    # ------------------------------------------------ within-pair permutation of the sequence arms
    rng = np.random.default_rng(1255)
    idx = np.arange(N)
    o = np.argsort(pair, kind='stable')
    starts = np.flatnonzero(np.r_[True, pair[o][1:] != pair[o][:-1]])
    for a, b in zip(starts, np.r_[starts[1:], N]):
        blk = o[a:b].copy()
        rng.shuffle(blk)
        idx[o[a:b]] = blk

    # ------------------------------------------------ stacks
    def stack(arms, use_lto, perm=False):
        p_out = {}
        for m in MODELS:
            pr = np.full(N, np.nan)
            for k in range(5):
                tr = np.flatnonzero(elig[m] & (fold != k))
                te = np.flatnonzero(fold == k)
                cols_tr, cols_te = [], []
                for a in arms:
                    if use_lto and f'lto_{a}_{m}' in P:
                        v = P[f'lto_{a}_{m}'][k]
                    else:
                        v = P[f'loo_{a}_{m}']
                    w = P[f'loo_{a}_{m}']
                    if perm and a != 'plus':
                        v, w = v[idx], w[idx]
                    cols_tr.append(lg(v[tr]))
                    cols_te.append(lg(w[te]))
                Z = np.column_stack(cols_tr)
                assert np.isfinite(Z).all(), (m, k, arms)
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z, posmask[m][tr].astype(int))
                pr[te] = lr.predict_proba(np.column_stack(cols_te))[:, 1]
            p_out[m] = pr
        return p_out

    arms_lr = ['plus'] + seq_arms
    S_lr = stack(arms_lr, True)
    S_perm = stack(arms_lr, True, perm=True)
    S_lrT = stack(arms_lr + neural, True) if has_T else None
    avg = {}
    for m in MODELS:
        sq = np.mean([lg(P[f'loo_{a}_{m}']) for a in seq_arms + neural], axis=0)
        z = .5 * lg(P[f'loo_plus_{m}']) + .5 * sq
        avg[m] = 1 / (1 + np.exp(-z))
    print('stacks built', f'{time.time() - t0:.0f}s', flush=True)

    # ------------------------------------------------ readouts
    named = {'plus': {m: P[f'loo_plus_{m}'] for m in MODELS}}
    for a in seq_arms + neural:
        named[a] = {m: P[f'loo_{a}_{m}'] for m in MODELS}
    named['STACK_lr'] = S_lr
    if S_lrT:
        named['STACK_lrT'] = S_lrT
    named['STACK_avg'] = avg
    named['PERM'] = S_perm

    res = {}
    for a, d in named.items():
        pmax = np.max(np.stack([d[f] for f in FAMS]), axis=0)
        q99 = float(np.quantile(pmax[honest], .99))
        r = {}
        for m in MODELS:
            r[m] = metrics(posmask[m], elig[m], d[m])
            sel = posmask[m]
            r[m]['invisible_share'] = round(float((pmax[sel] < q99).mean()), 4)
        # diversity vs plus
        for m in MODELS:
            if a == 'plus':
                continue
            lo_a, lo_p = lg(d[m]), lg(named['plus'][m])
            r[m]['spearman_vs_plus_listed'] = round(
                float(spearmanr(lo_a[listed == 1], lo_p[listed == 1]).statistic), 3)
            hs = np.flatnonzero(honest)
            r[m]['spearman_vs_plus_honest'] = round(
                float(spearmanr(lo_a[hs], lo_p[hs]).statistic), 3)
        r['_honest_q99_fmax'] = round(q99, 6)
        res[a] = r
        print(a, {m: (r[m]['auc'], r[m]['ap']) for m in MODELS}, flush=True)

    # per-fold deltas for G1
    gate = {}
    for name in ('STACK_lr', 'PERM', 'STACK_avg') + (('STACK_lrT',) if S_lrT else ()):
        g = {}
        for m in MODELS:
            per = []
            for k in range(5):
                s = elig[m] & (fold == k)
                if posmask[m][s].sum() < 5:
                    per.append(None)
                    continue
                a1 = average_precision_score(posmask[m][s], named[name][m][s])
                a0 = average_precision_score(posmask[m][s], named['plus'][m][s])
                per.append(round(float(a1 - a0), 4))
            g[m] = {'delta_ap': round(res[name][m]['ap'] - res['plus'][m]['ap'], 4),
                    'delta_auc': round(res[name][m]['auc'] - res['plus'][m]['auc'], 5),
                    'per_fold': per,
                    'folds_positive': int(sum(1 for x in per if x is not None and x > 0))}
        gate[name] = g
    g1 = gate['STACK_lr']
    verdict = {
        'any_ap_delta': g1['any']['delta_ap'], 'bar': 0.025,
        'ci_ap_delta': g1['coordinated_isolation']['delta_ap'], 'ci_bar': 0.030,
        'folds_positive_any': g1['any']['folds_positive'], 'folds_bar': 4,
        'perm_any_ap_delta': gate['PERM']['any']['delta_ap'], 'perm_bar': 0.005,
    }
    verdict['pass'] = bool(verdict['any_ap_delta'] >= .025 and verdict['ci_ap_delta'] >= .030
                           and verdict['folds_positive_any'] >= 4
                           and verdict['perm_any_ap_delta'] <= .005)
    print('G1', verdict, flush=True)

    # ------------------------------------------------ top n-grams per family (descriptive + fitted)
    X = sparse.load_npz(OUT + r'\ngrams255.npz').tocsr()
    Xb = X.copy()
    Xb.data[:] = 1.
    vocab = pl.read_parquet(OUT + r'\vocab255.parquet')
    text = vocab['text'].to_list()
    hon_rate = np.asarray(Xb[np.flatnonzero(honest)].sum(axis=0)).ravel() / max(honest.sum(), 1)
    tops = {}
    for m in MODELS:
        sel = np.flatnonzero(posmask[m])
        rate = np.asarray(Xb[sel].sum(axis=0)).ravel() / len(sel)
        cnt = np.asarray(Xb[sel].sum(axis=0)).ravel()
        lo = (np.log((rate + .002) / (hon_rate + .002)))
        score = lo * np.sqrt(np.minimum(cnt, 400))
        keep = cnt >= 25
        idxs = np.argsort(np.where(keep, score, -1e9))[::-1][:20]
        tops[m] = [{'ngram': text[i], 'listed_rate': round(float(rate[i]), 4),
                    'honest_rate': round(float(hon_rate[i]), 4),
                    'lift': round(float(rate[i] / max(hon_rate[i], 1e-5)), 2),
                    'n_listed': int(cnt[i])} for i in idxs]
    print('top n-grams computed', f'{time.time() - t0:.0f}s', flush=True)

    # ------------------------------------------------ save
    cols = {'pair_id': j['pair_id'], 'hand_id': j['hand_id'], 'table_id': j['table_id'],
            'fold': j['fold'], 'label': j['label'], 'listed': j['listed'],
            'behavior_family': j['behavior_family']}
    out = pl.DataFrame(cols)
    add = []
    for a, d in named.items():
        for m in MODELS:
            add.append(pl.Series(f'p_{a}_{m}', d[m].astype(np.float32)))
    out = out.with_columns(add)
    out.write_parquet(OUT + r'\hand_oof_step255.parquet')
    json.dump({'arms': res, 'gate': gate, 'G1': verdict, 'top_ngrams': tops,
               'seq_arms': seq_arms, 'has_T': bool(has_T),
               'seconds': round(time.time() - t0, 1)},
              open(OUT + r'\results255.json', 'w', encoding='utf-8'), indent=1)
    print('DONE', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
