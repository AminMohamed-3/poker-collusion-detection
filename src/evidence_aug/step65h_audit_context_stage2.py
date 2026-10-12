"""Step 65h: training-only audit of cross-candidate (type, time) context on top of the held-confirmed stage-2 (step 65c/65f).
The listing rule is (type, time)-sorted with cap 5 (OBS 14.16): a coordinated showdown-type hand is listed only if fewer than
five fold-type / earlier showdown-type coordinated hands precede it. The stage-2 sees each candidate alone (pair_sd, chrono
columns) and cannot count the candidates that precede it in that order. Variants, all per family, top-20 stage-1 candidates,
GroupKFold(5, by table), 3 seeds, step-14 CLF, tail keeps stage-1 order, 162 audit queries:
  ref        struct + chrono3 + oof, oof_rank                                    (= F1t_chrono3, step 65g)
  ctx_hard   ref + hard counts among the top-20: type flag, n showdown/fold-type, n earlier, n earlier same type,
             position in the (type, time) order among the top-20, earlier-same-type among top-10, rank of oof within type
  ctx_soft   ref + expected counts: the same quantities weighted by a nested (inner GroupKFold) pass-1 stage-2 probability
  ctx_both   ref + both
  r382       ref + the 382 stage-1 ranker features (per family; can a per-family model re-weight them?)
  r382_sel   ref + the 15 ranker features with the largest |AUC-.5| within the top-20 on the TRAINING fold (nested selection)
  top30      ref applied to the top-30 candidates instead of the top-20 (scope check)
Also: per-family within-top-20 single-feature AUC scan of the 382 ranker + structural features (descriptive).
Output: experiments/label_process_20260911/audit-context-stage2.json"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import write_json
from evidence_aug.frame import FAMILIES
from evidence_aug.fold import load_fold
from evidence_aug.step14_audit_truncation import OUT, TOP, CLF
from evidence_aug.step65_structural_evidence import summarize, SEEDS, log
from evidence_aug.step65b_family_stage2 import boot
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HARD = ['type_sd', 'n_sd', 'n_fold', 'n_earlier', 'n_earlier_same_type', 'pos_type_time', 'n_earlier_same_type_top10', 'oof_rank_within_type', 'frac_earlier_same_type']
SOFT = ['p_self', 'p_rank', 'soft_total', 'soft_n_sd', 'soft_n_fold', 'soft_n_earlier', 'soft_n_earlier_same_type', 'soft_pos_type_time', 'soft_pos_over_cap']


def fit(X, y, seed):
    clf = lgb.LGBMClassifier(**CLF, random_state=seed)
    clf.fit(X, y)
    return clf


def arrays(sub, idx=None):
    a = {'pid': sub['pair_id'].to_numpy(), 'cr': sub['chrono_rank'].to_numpy().astype(float), 'orank': sub['oof_rank'].to_numpy().astype(float),
         'sd': (sub['pair_sd'].to_numpy() > 0).astype(float), 'oof': sub['oof'].to_numpy().astype(float)}
    return a if idx is None else {k: v[idx] for k, v in a.items()}


def hard_features(a):
    """a: arrays of one family's candidates (pid, cr, orank, sd, oof). Counts among the given rows per pair."""
    pid, cr, orank, sd, oof = a['pid'], a['cr'], a['orank'], a['sd'], a['oof']
    out = np.zeros((len(pid), len(HARD)))
    for p in np.unique(pid):
        ix = np.where(pid == p)[0]
        s, c, r, o = sd[ix], cr[ix], orank[ix], oof[ix]
        n_sd, n_fold = s.sum(), (1 - s).sum()
        for k, i in enumerate(ix):
            earlier = c < c[k]; same = s == s[k]
            n_e = earlier.sum(); n_es = (earlier & same).sum()
            pos = (n_fold if s[k] > 0 else 0.0) + n_es
            n_es10 = (earlier & same & (r <= 10)).sum()
            rk_type = (same & (o > o[k])).sum() + 1
            out[i] = [s[k], n_sd, n_fold, n_e, n_es, pos, n_es10, rk_type, n_es / max(same.sum() - 1, 1)]
    return out


def soft_features(a, p):
    pid, cr, sd = a['pid'], a['cr'], a['sd']
    out = np.zeros((len(pid), len(SOFT)))
    for q in np.unique(pid):
        ix = np.where(pid == q)[0]
        s, c, pp = sd[ix], cr[ix], p[ix]
        tot, n_sd, n_fold = pp.sum(), (pp * s).sum(), (pp * (1 - s)).sum()
        for k, i in enumerate(ix):
            earlier = c < c[k]; same = s == s[k]
            se = (pp * earlier).sum(); ses = (pp * earlier * same).sum()
            pos = (n_fold if s[k] > 0 else 0.0) + ses
            prank = (pp > pp[k]).sum() + 1
            out[i] = [pp[k], prank, tot, n_sd, n_fold, se, ses, pos, max(pos - 5.0 + 1.0, 0.0)]
    return out


def auc_select(X, y, k):
    scores = []
    for j in range(X.shape[1]):
        x = X[:, j]
        scores.append(abs(roc_auc_score(y, x) - .5) if len(np.unique(x)) > 1 and 0 < y.sum() < len(y) else 0.0)
    return np.argsort(scores)[::-1][:k]


def fam_oof_ctx(d, cols, y, groups, top=TOP, hard=False, soft=False, extra=None, select=0):
    """extra: numpy matrix aligned with d (e.g. the 382 ranker features) appended to cols; select>0: nested top-k AUC selection from extra."""
    pred = np.full(d.height, -1e6)
    for fam in FAMILIES:
        mask = ((d['behavior_family'] == fam) & (d['oof_rank'] <= top)).to_numpy()
        sub = d.filter(pl.Series(mask)); arr = arrays(sub)
        X = sub.select(cols).to_numpy().astype(np.float64); ys = y[mask]; gs = groups[mask]
        E = extra[mask] if extra is not None else None
        if hard:
            X = np.hstack([X, hard_features(arr)])
        p = np.zeros(len(ys))
        for seed in SEEDS:
            for tr, te in GroupKFold(n_splits=5).split(X, ys, gs):
                Xtr, Xte = X[tr], X[te]
                if E is not None:
                    if select:
                        keep = auc_select(E[tr], ys[tr], select); Xtr = np.hstack([Xtr, E[tr][:, keep]]); Xte = np.hstack([Xte, E[te][:, keep]])
                    else:
                        Xtr = np.hstack([Xtr, E[tr]]); Xte = np.hstack([Xte, E[te]])
                if soft:
                    p_tr = np.zeros(len(tr))
                    for itr, ite in GroupKFold(n_splits=5).split(Xtr, ys[tr], gs[tr]):
                        p_tr[ite] = fit(Xtr[itr], ys[tr][itr], seed).predict_proba(Xtr[ite])[:, 1]
                    p_te = fit(Xtr, ys[tr], seed).predict_proba(Xte)[:, 1]
                    Xtr = np.hstack([Xtr, soft_features(arrays(sub, tr), p_tr)]); Xte = np.hstack([Xte, soft_features(arrays(sub, te), p_te)])
                p[te] += fit(Xtr, ys[tr], seed).predict_proba(Xte)[:, 1] / len(SEEDS)
        pred[mask] = p
    return pred


def main():
    t0 = time.time()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet'); feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]; sc = ['oof', 'oof_rank']; chrono = ['chrono_rank', 'chrono_frac', 'n_hands']
    ref_cols = struct + chrono + sc
    data, features, _, _ = load_fold(0)
    rk = data.select(['pair_id', 'hand_id'] + [f for f in features if not f.startswith('family_') and f in data.columns])
    rcols = [c for c in rk.columns if c not in ('pair_id', 'hand_id')]
    d = d.join(rk.with_columns(pl.lit(1).alias('_hit')), on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert d['_hit'].null_count() == 0, 'ranker features missing for audit rows'
    d = d.drop('_hit')
    null_share = float(d.select(pl.sum_horizontal(pl.col(rcols).is_null().sum())).item()) / (d.height * len(rcols))
    log('ranker features joined', len(rcols), 'null share', round(null_share, 4), round(time.time() - t0))
    extra = np.nan_to_num(d.select(rcols).to_numpy().astype(np.float64), nan=-999.0)
    d = d.with_columns(pl.col(rcols).fill_null(-999.0))
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows()); ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows()); pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float); groups = d['table_id'].to_numpy(); key = d.select('pair_id', 'hand_id', 'table_id')
    res = {'training_only': True, 'queries': len(truths), 'n_ranker_features': len(rcols), 'sets': {}, 'auc_top20': {}}
    # descriptive: per-family single-feature AUC within the top-20 (ranker + structural + hard context)
    res['single_auc_top20'] = {}
    hard_all = np.zeros((d.height, len(HARD)))
    for fam in FAMILIES:
        mask = ((d['behavior_family'] == fam) & (d['oof_rank'] <= TOP)).to_numpy()
        hard_all[mask] = hard_features(arrays(d.filter(pl.Series(mask))))
        g = d.filter(pl.Series(mask)); yy = g['relevance'].to_numpy()
        table = {}
        for c in rcols + struct + chrono + sc:
            x = g[c].to_numpy().astype(float)
            if len(np.unique(x)) > 1:
                table[c] = round(float(roc_auc_score(yy, x)), 3)
        for j, c in enumerate(HARD):
            x = hard_all[mask][:, j]
            if len(np.unique(x)) > 1:
                table[c] = round(float(roc_auc_score(yy, x)), 3)
        top = sorted(table.items(), key=lambda kv: -abs(kv[1] - .5))[:25]
        res['single_auc_top20'][fam] = {'n_rows': int(mask.sum()), 'n_listed': int(yy.sum()), 'top25': top, 'context': {c: table.get(c) for c in HARD}}
        log(fam, 'top single AUCs within top-20:', top[:12])
    specs = {'ref': dict(), 'ctx_hard': dict(hard=True), 'ctx_soft': dict(soft=True), 'ctx_both': dict(hard=True, soft=True),
             'r382': dict(extra=extra), 'r382_sel': dict(extra=extra, select=15), 'top30': dict(top=30)}
    maps = {}
    for name, kw in specs.items():
        t1 = time.time()
        pred = fam_oof_ctx(d, ref_cols, y, groups, **kw)
        top = kw.get('top', TOP); cands = (d['oof_rank'] <= top).to_numpy()
        pred = np.where(cands, pred + 10.0, d['oof'].to_numpy() - 1e6)
        s, m = summarize(key.with_columns(pl.Series('score', pred)), truths, labels); maps[name] = m
        s['auc_within_top20'] = {}
        for fam in FAMILIES:
            mask = ((d['behavior_family'] == fam) & (d['oof_rank'] <= TOP)).to_numpy()
            s['auc_within_top20'][fam] = round(float(roc_auc_score(y[mask], pred[mask])), 4)
        mask = (d['oof_rank'] <= TOP).to_numpy()
        s['auc_within_top20']['pooled_by_pair'] = round(float(np.mean([roc_auc_score(y[mask & (d['pair_id'] == p).to_numpy()], pred[mask & (d['pair_id'] == p).to_numpy()]) for p in truths if 0 < y[mask & (d['pair_id'] == p).to_numpy()].sum() < (mask & (d['pair_id'] == p).to_numpy()).sum()])), 4)
        s['seconds'] = round(time.time() - t1); res['sets'][name] = s
        log(name, json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}, 'auc20': s['auc_within_top20'], 'sec': s['seconds']}))
    res['paired_vs_ref'] = {}
    for name in specs:
        if name == 'ref':
            continue
        b = boot(maps['ref'], maps[name], ptab, pools)
        b['by_family'] = {f: round(float(np.mean([maps[name][p] - maps['ref'][p] for p in maps['ref'] if labels[p] == f])), 4) for f in FAMILIES}
        res['paired_vs_ref'][name] = b
        log(f'{name} minus ref', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}))
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-context-stage2.json', res)
    log('done', res['seconds'])


if __name__ == '__main__':
    main()
