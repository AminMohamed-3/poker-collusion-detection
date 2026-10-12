"""Step 64c: HAND-level pooled-evidence view + evaluation of every step-64 view on the recall tail (folds 1-2, 60 % prefix).
Architecture change. The production model aggregates each pair's hands into 897 statistics (means / std / max / q95 over
the whole shared history) and classifies the aggregate. For a SHORT pair those statistics are noisy and the trees fall
back to the majority leaf (the tail positives of 14.38 have 11-82 hands). The classical alternative for "constant per-hand
rate" collusion (14.20) is a hand-level classifier whose per-hand log-likelihood ratios are SUMMED over the pair: the
sequential-test statistic, whose evidence grows with the number of hands and stays comparable across lengths.
  HAND model: LightGBM on the per-hand features (163 raw + 33 fold-policy ca), label = the pair's label (binary) or family
  (4-class), trained on the hands of labelled pairs outside the fold's held pools; scored on the prefix hands of the
  step-64b population; pooled per pair: mean logit, LLR sum (logit - prior logit), top-5 mean, fraction > .5, per-family
  LLR sums and their max.
Evaluation (development labels, validation only): (1) within-block AUC positives vs certified false positives for every new
view (64a aggregate views + hand pools); (2) where the 18 tail positives (prefix risk < .5) land under each view against the
label-free null background (percentile among the tail sample / among all testbed pairs); (3) cross-fold learned blends
(step 60 protocol, l2 = 10) at floor .5 and floor .1, definitions A and C, base set = 8 views + log sh + R1 vs + each new
view; (4) OPEN AP: positives vs every pair of the population with the tail sample re-weighted to the full tail, i.e. the
closest local analogue of the competition ranking, for the frozen risk, the base blend and each extended blend.
Outputs: experiments/viewdis_20260911/{hand_view_prefix.parquet, tail_views_eval.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
import lightgbm as lgb
from pu_pairs.build_allpairs_fold import CONF, EXP, fold_pools
from pu_pairs.step60_multiview_blend import logreg, predict, lg, ap, auc
sys.path.insert(0, str(ORIGINAL / 'src'))
from robust_pair_experiment import LABELS, FAMILIES  # noqa: E402

VD = CV / 'experiments' / 'viewdis_20260911'
HAND_PARAMS = dict(n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=50, colsample_bytree=.8, subsample=.8, subsample_freq=1,
                   reg_lambda=5, n_jobs=2, verbosity=-1, random_state=6403, deterministic=True, force_col_wise=True)
VIEWS = ['FULL', 'RAW', 'CA', 'BASE', 'OPP', 'V2', 'MEAN', 'TAILS']
AGG_NEW = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'DT', 'SP', 'CI', 'FAMMAX']
HAND_NEW = ['H_mean', 'H_llr', 'H_top5', 'H_frac', 'H_any', 'H_fammax']
L2 = 10.0
DEFS = {'A': (['pos', 'undisclosed'], ['FP', 'undisclosed_R1']), 'C': (['pos'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg'])}


def log(*a):
    print(*a, flush=True)


def hand_train(fold):
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted((ORIGINAL / 'outputs_v2' / 'hands_development').glob('batch_*.parquet')))
    ca = pl.read_parquet(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet')
    caf = [c for c in ca.columns if c.startswith('ca_')]
    d = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], how='left', validate='1:1').with_columns(pl.col(caf).fill_null(0)).join(labels, on='pair_id')
    held = set(fold_pools(fold)); d = d.filter(~pl.col('table_id').is_in(sorted(held)))
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    return d, names + caf


def pool(scored, prior_pos, prior_fam):
    """scored: pair_id, hand_id, p (binary positive prob), p0..p3 (4-class)."""
    z = lg(scored['p'].to_numpy()); zp = float(np.log(prior_pos / (1 - prior_pos)))
    P = np.column_stack([scored[f'p{k}'].to_numpy() for k in range(4)]); P = np.clip(P, 1e-6, 1)
    fam_llr = {f'llr_{k}': np.log(P[:, k] / P[:, 0]) - np.log(prior_fam[k] / prior_fam[0]) for k in range(1, 4)}
    any_llr = np.log((1 - P[:, 0]) / P[:, 0]) - np.log((1 - prior_fam[0]) / prior_fam[0])
    s = scored.select('pair_id').with_columns(pl.Series('z', z), pl.Series('llr', z - zp), pl.Series('hit', (z > 0).astype(float)), pl.Series('any', any_llr), *[pl.Series(k, v) for k, v in fam_llr.items()])
    g = s.group_by('pair_id').agg(pl.len().alias('H_n'), pl.col('z').mean().alias('H_mean'), pl.col('llr').sum().alias('H_llr'), pl.col('z').top_k(5).mean().alias('H_top5'),
                                  pl.col('hit').mean().alias('H_frac'), pl.col('any').sum().alias('H_any'), *[pl.col(k).sum().alias(k) for k in fam_llr])
    return g.with_columns(pl.max_horizontal('llr_1', 'llr_2', 'llr_3').alias('H_fammax'))


def hand_view(t0):
    parts, info = [], {}
    for fold in (1, 2):
        d, feats = hand_train(fold)
        X = d.select(feats).to_numpy().astype(np.float32); X = np.nan_to_num(X)
        yb = (d['label'] == 1).to_numpy().astype(int); yf = np.array([LABELS[f] for f in d['behavior_family']])
        mb = lgb.LGBMClassifier(**HAND_PARAMS).fit(X, yb); mf = lgb.LGBMClassifier(**HAND_PARAMS).fit(X, yf)
        prior_pos = float(yb.mean()); prior_fam = np.bincount(yf, minlength=4) / len(yf)
        log(f'fold{fold} hand models fitted on {len(yb)} hands ({int(yb.sum())} positive-pair hands), {len(feats)} features', round(time.time() - t0))
        raw = pl.read_parquet(VD / f'prefix_fold{fold}_ext_raw.parquet'); ca = pl.read_parquet(VD / f'prefix_fold{fold}_ext_ca.parquet')
        pre = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], validate='1:1')
        Xp = np.nan_to_num(pre.select(feats).to_numpy().astype(np.float32))
        sc = pre.select('pair_id', 'hand_id').with_columns(pl.Series('p', mb.predict_proba(Xp)[:, 1]))
        pf = mf.predict_proba(Xp); sc = sc.with_columns(*[pl.Series(f'p{k}', pf[:, k]) for k in range(4)])
        g = pool(sc, prior_pos, prior_fam).with_columns(pl.lit(fold).alias('fold_new'))
        parts.append(g); info[fold] = {'train_hands': int(len(yb)), 'prior_pos_hand': round(prior_pos, 4), 'scored_pairs': g.height, 'scored_hands': sc.height}
        log(f'fold{fold} hand view pooled', g.height, round(time.time() - t0))
    out = pl.concat(parts); out.write_parquet(VD / 'hand_view_prefix.parquet')
    return out, info


def wap(y, s, w):
    """Weighted AP: precision at each positive = cum positive weight / cum weight (weights re-scale the tail sample)."""
    o = np.argsort(-s, kind='stable'); y = y[o]; w = w[o]
    cw = np.cumsum(w); cp = np.cumsum(w * y)
    return float(((cp / cw) * w * y).sum() / max(1e-9, (w * y).sum()))


def X_of(c, fs):
    cols = []
    for v in fs:
        if v == 'sh': cols.append(np.log(c['shared_hands'].to_numpy().astype(float)))
        elif v == 'R1': cols.append(c['R1_prefix'].fill_null(False).to_numpy().astype(float))
        elif v == 'FULL': cols.append(lg(c['risk_prefix'].to_numpy()))
        elif v.startswith('H_'): cols.append(c[v].to_numpy().astype(float))
        else: cols.append(lg(c[f'r_{v}'].to_numpy()))
    return np.column_stack(cols)


def crossfold(c, y, fs):
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2; tab = c['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
    X = X_of(c, fs); pred = np.full(len(y), np.nan)
    m1 = logreg(X[in1], y[in1], L2); pred[in2 | insh] = predict(m1, X[in2 | insh])
    m2 = logreg(X[in2], y[in2], L2); pred[in1] = predict(m2, X[in1])
    return pred, (m1, m2), (in1, in2, insh)


def main():
    t0 = time.time(); res = {}
    hv, res['hand_fit'] = hand_view(t0)
    mp = pl.read_parquet(VD / 'multiview_prefix.parquet')
    tv = pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).drop('fold_new')
    hv1 = hv.sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).drop('fold_new')
    pops = pl.concat([pl.read_parquet(VD / f'prefix_fold{k}_ext_population.parquet').select('pair_id', 'why') for k in (1, 2)]).unique(subset=['pair_id'], keep='first')
    r1 = pl.read_parquet(VD / 'prefix_rescoring.parquet').select('pair_id', 'R1_prefix').unique(subset=['pair_id'], keep='first')
    d = mp.join(tv, on='pair_id', how='left').join(hv1, on='pair_id', how='left').join(pops, on='pair_id', how='left').join(r1, on='pair_id', how='left').with_columns(pl.col('R1_prefix').fill_null(False))
    n_tail_total = int((mp['risk_prefix'] < .1).sum()); n_tail_sample = int((d['why'] == 'tail_sample').sum())
    res['sizes'] = {'testbed_pairs': mp.height, 'tail_lt_.1': n_tail_total, 'tail_sample': n_tail_sample, 'hand_view_pairs': int(d['H_llr'].is_not_null().sum())}
    # (1) within-block separation
    blk = d.filter(pl.col('risk_prefix') >= .5); sep = {}
    for v in VIEWS + AGG_NEW + HAND_NEW:
        col = 'risk_prefix' if v == 'FULL' else (v if v.startswith('H_') else f'r_{v}')
        g = {c: blk.filter((pl.col('cls') == c) & pl.col(col).is_not_null())[col].to_numpy() for c in ('pos', 'undisclosed', 'FP', 'mid', 'ambiguous')}
        sep[v] = {'AUC_pos_vs_FP': round(auc(g['pos'], g['FP']), 4), 'AUC_undisc_vs_FP': round(auc(g['undisclosed'], g['FP']), 4), 'AUC_pos_vs_amb': round(auc(g['pos'], g['ambiguous']), 4), 'n': {k: len(x) for k, x in g.items()}}
    res['block_separation'] = sep
    # (2) tail positives against the null background
    tailpos = d.filter((pl.col('label') == 1) & (pl.col('risk_prefix') < .5))
    negs_all = d.filter(pl.col('label').is_null() | (pl.col('label') == 0))
    negs_h = negs_all.filter(pl.col('H_llr').is_not_null())
    tl = {'n_tail_pos': tailpos.height, 'pairs': tailpos.select('pair_id', 'shared_hands', 'risk_prefix', 'behavior_family').to_dicts()}
    for v in ['FULL'] + AGG_NEW + HAND_NEW:
        col = 'risk_prefix' if v == 'FULL' else (v if v.startswith('H_') else f'r_{v}')
        ref = (negs_h if v.startswith('H_') else negs_all)[col].drop_nulls().to_numpy(); ref = np.sort(ref)
        x = tailpos[col].to_numpy()
        pct = np.searchsorted(ref, x, side='left') / len(ref)
        tl[v] = {'pctile_median': round(float(np.nanmedian(pct)), 4), 'n_above_p99': int(np.nansum(pct >= .99)), 'n_above_p999': int(np.nansum(pct >= .999)), 'n_above_p90': int(np.nansum(pct >= .9)),
                 'pctiles': [round(float(p), 4) for p in pct], 'ref_n': int(len(ref))}
    res['tail_positives'] = tl
    # (3) cross-fold blends and (4) open AP
    base_fs = VIEWS + ['sh', 'R1']
    ext_sets = {'base': base_fs, **{f'+{v}': base_fs + [v] for v in AGG_NEW + HAND_NEW}, '+SHORT3+ALL5': base_fs + ['SHORT3', 'ALL5'], '+H_llr+H_fammax': base_fs + ['H_llr', 'H_fammax'],
                '+all agg new': base_fs + AGG_NEW, '+all hand': base_fs + HAND_NEW, '+all new': base_fs + AGG_NEW + HAND_NEW}
    out = {}
    for floor in (.5, .1):
        for dn, (posset, negset) in DEFS.items():
            c = d.filter((pl.col('risk_prefix') >= floor) & pl.col('cls').is_in(posset + negset) & pl.col('H_llr').is_not_null() & pl.col('r_SHORT3').is_not_null())
            y = c['cls'].is_in(posset).to_numpy().astype(float)
            r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'frozen': round(ap(y, c['risk_prefix'].to_numpy()), 4)}
            for name, fs in ext_sets.items():
                pred, (m1, m2), (in1, in2, insh) = crossfold(c, y, fs)
                r[name] = {'pooled': round(ap(y, pred), 4), 'f2sh': round(ap(y[in2 | insh], pred[in2 | insh]), 4), 'f1': round(ap(y[in1], pred[in1]), 4),
                           'w1': {v: round(float(w), 2) for v, w in zip(fs, m1[0]) if v not in VIEWS + ['sh']}, 'w2': {v: round(float(w), 2) for v, w in zip(fs, m2[0]) if v not in VIEWS + ['sh']}}
            out[f'floor {floor} def {dn}'] = r
    res['blends'] = out
    # (4) OPEN AP on the hand-view population: positives vs everything else, tail sample re-weighted; blend fitted on def C floor .1 rows, applied to all
    cpop = d.filter(pl.col('H_llr').is_not_null() & pl.col('r_SHORT3').is_not_null())
    ypop = (cpop['label'] == 1).to_numpy().astype(float)
    wpop = np.where((cpop['why'] == 'tail_sample').to_numpy(), n_tail_total / max(1, n_tail_sample), 1.0)
    fit_rows = ((cpop['risk_prefix'] >= .1) & cpop['cls'].is_in(DEFS['C'][0] + DEFS['C'][1])).to_numpy()
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2; tab = cpop['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
    op = {'n_pos': int(ypop.sum()), 'n_rows': int(len(ypop)), 'weighted_negatives': round(float(wpop[ypop == 0].sum())), 'frozen': round(wap(ypop, cpop['risk_prefix'].to_numpy(), wpop), 4),
          'frozen_f1': round(wap(ypop[in1], cpop['risk_prefix'].to_numpy()[in1], wpop[in1]), 4), 'frozen_f2sh': round(wap(ypop[in2 | insh], cpop['risk_prefix'].to_numpy()[in2 | insh], wpop[in2 | insh]), 4)}
    open_pred = cpop.select('pair_id', 'table_id', 'label', 'cls', 'why', 'risk_prefix', 'risk_full', 'shared_hands').with_columns(pl.Series('weight', wpop))
    for name, fs in ext_sets.items():
        X = X_of(cpop, fs); yc = ypop.copy(); pred = np.full(len(ypop), np.nan)
        m1 = logreg(X[in1 & fit_rows], yc[in1 & fit_rows], L2); pred[in2 | insh] = predict(m1, X[in2 | insh])
        m2 = logreg(X[in2 & fit_rows], yc[in2 & fit_rows], L2); pred[in1] = predict(m2, X[in1])
        open_pred = open_pred.with_columns(pl.Series(f'open_{name}', pred))
        op[name] = {'open_AP': round(wap(ypop, pred, wpop), 4), 'f1': round(wap(ypop[in1], pred[in1], wpop[in1]), 4), 'f2sh': round(wap(ypop[in2 | insh], pred[in2 | insh], wpop[in2 | insh]), 4),
                    'tail_pos_rank_pct': [round(float((pred[(ypop == 0)] >= pred[i]).astype(float) @ wpop[ypop == 0] / wpop[ypop == 0].sum()), 4) for i in np.where((ypop == 1) & (cpop['risk_prefix'].to_numpy() < .5))[0]]}
    res['open_AP'] = op
    open_pred.write_parquet(VD / 'tail_views_open_pred.parquet')
    res['seconds'] = round(time.time() - t0)
    write_json(VD / 'tail_views_eval.json', res)
    log(json.dumps({k: res[k] for k in ('sizes', 'block_separation', 'tail_positives')}, indent=1)[:6000])
    log(json.dumps(res['blends'], indent=1)[:8000]); log(json.dumps(res['open_AP'], indent=1)[:6000])


if __name__ == '__main__':
    main()
