"""Step 66: EVIDENCE-SIGNATURE view for the pair side (development labels for validation only; label-free at evaluation).
Idea. The evidence ranker (listed vs unlisted hands inside positive pairs) recognises what a PLANTED hand looks like
(within-top-20 AUC ~.91 after the structural features), a much cleaner target than the HAND view of step 64c (label =
"hand belongs to a positive pair", 90 % of which are ordinary hands; null, block AUC .57). If every positive pair carries
~6-8 planted hands and a length-induced false positive carries none, the per-pair aggregate of a planted-hand classifier
is a view that does not depend on history length the way the 897 aggregates do.
  PLANT model per fold k in {1, 2}: LightGBM classifier on per-hand features (163 raw + 33 fold-policy ca + ~72 structural
  + chrono_frac), rows = all hands of the labelled POSITIVE pairs outside fold k's held pools, label = listed in
  development_evidence (the evidence truth of the training pools only; held pools never read).
  Scored on the step-64b prefix population of fold k (2,880 / 2,974 pairs, 60 % prefix); pooled per pair: max p, top-5
  mean, top-8 sum, count > .5, fraction > .5, LLR sum, and the same for logits.
Evaluation = the step-64c protocol: (1) within-block AUC positives vs certified false positives; (2) tail positives'
percentiles; (3) cross-fold learned blends (step 60 protocol, l2 = 10) at floor .5 / .1, definitions A / C, base = 8 views
+ log sh + R1 (M2) and base + SHORT3 + ALL5 (M3 design) vs + each E view; (4) OPEN AP with the tail re-weighted.
Outputs: experiments/viewdis_20260911/{evidence_view_prefix.parquet, evidence_view_eval.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
from evidence_aug.step65_structural_evidence import build
import numpy as np
import polars as pl
import lightgbm as lgb
from pu_pairs.build_allpairs_fold import fold_pools
from pu_pairs.step60_multiview_blend import logreg, predict, lg, ap, auc
from pu_pairs.step64c_hand_view import hand_train, wap, VIEWS, L2, DEFS, VD

PLANT_PARAMS = dict(n_estimators=300, learning_rate=.05, num_leaves=15, max_depth=5, min_child_samples=40, colsample_bytree=.5, subsample=.8, subsample_freq=1,
                    reg_lambda=5, n_jobs=2, verbosity=-1, random_state=6601, deterministic=True, force_col_wise=True)
E_NEW = ['E_max', 'E_top5', 'E_top8sum', 'E_cnt50', 'E_frac50', 'E_llr', 'E_ztop5']
NOSTRUCT = os.environ.get('STEP66_NOSTRUCT') == '1'  # ablation: raw + ca + chrono_frac only
NOCA = os.environ.get('STEP66_NOCA') == '1'  # ablation: raw + chrono_frac only (no policy ca features, no struct)
SUFFIX = '_rawonly' if NOCA else ('_nostruct' if NOSTRUCT else '')


def log(*a):
    print(*a, flush=True)


def chrono(frame):
    hands = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet', columns=['hand_id', 'started_at'])
    out = frame.join(hands, on='hand_id', how='left')
    assert out['started_at'].null_count() == 0
    out = out.with_columns(pl.col('started_at').rank('ordinal').over('pair_id').cast(pl.Float64).sub(1).alias('chrono_rank'), pl.len().over('pair_id').cast(pl.Float64).alias('n_hands'))
    return out.with_columns((pl.col('chrono_rank') / pl.max_horizontal(pl.col('n_hands') - 1, 1.0)).alias('chrono_frac')).drop('started_at', 'chrono_rank', 'n_hands')


def with_struct(frame, pair_players):
    feat = build(frame.select('pair_id', 'hand_id'), pair_players)
    feat = feat.rename({c: f's_{c}' for c in feat.columns if c not in ('pair_id', 'hand_id')})  # 's_' prefix: raw already has pot_bb
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    out = frame.join(feat, on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert out[struct[0]].null_count() == 0
    return out, struct


def pool(scored, prior):
    z = lg(scored['p'].to_numpy()); zp = float(np.log(prior / (1 - prior)))
    s = scored.select('pair_id').with_columns(pl.Series('p', scored['p'].to_numpy()), pl.Series('z', z), pl.Series('llr', z - zp), pl.Series('hit', (scored['p'].to_numpy() > .5).astype(float)))
    return s.group_by('pair_id').agg(pl.len().alias('E_n'), pl.col('p').max().alias('E_max'), pl.col('p').top_k(5).mean().alias('E_top5'), pl.col('p').top_k(8).sum().alias('E_top8sum'),
                                     pl.col('hit').sum().alias('E_cnt50'), pl.col('hit').mean().alias('E_frac50'), pl.col('llr').sum().alias('E_llr'), pl.col('z').top_k(5).mean().alias('E_ztop5'))


def evidence_view(t0):
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    listed = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id').with_columns(pl.lit(1.0).alias('relevance'))
    parts, info = [], {}
    for fold in (1, 2):
        d, feats = hand_train(fold)
        d = d.filter(pl.col('label') == 1).join(listed, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('relevance').fill_null(0.0))
        d = chrono(d)
        if NOCA: feats = [f for f in feats if not f.startswith('ca_')]
        if NOSTRUCT or NOCA: struct = []
        else: d, struct = with_struct(d, labels.select('pair_id', 'player_1', 'player_2'))
        cols = feats + struct + ['chrono_frac']
        X = np.nan_to_num(d.select(cols).to_numpy().astype(np.float32)); y = d['relevance'].to_numpy().astype(int)
        m = lgb.LGBMClassifier(**PLANT_PARAMS).fit(X, y)
        prior = float(y.mean())
        gain = sorted(zip(cols, m.booster_.feature_importance('gain')), key=lambda kv: -kv[1])[:12]; tot = sum(v for _, v in gain) or 1.0
        log(f'fold{fold} plant model: {len(y)} hands of {d["pair_id"].n_unique()} positive pairs, {int(y.sum())} listed, {len(cols)} features', round(time.time() - t0))
        raw = pl.read_parquet(VD / f'prefix_fold{fold}_ext_raw.parquet'); ca = pl.read_parquet(VD / f'prefix_fold{fold}_ext_ca.parquet')
        pop = pl.read_parquet(VD / f'prefix_fold{fold}_ext_population.parquet').select('pair_id', 'player_1', 'player_2')
        pre = raw.join(ca.drop('table_id'), on=['pair_id', 'hand_id'], validate='1:1')
        pre = chrono(pre)
        if not (NOSTRUCT or NOCA):
            pre, struct2 = with_struct(pre, pop); assert struct2 == struct
        Xp = np.nan_to_num(pre.select(cols).to_numpy().astype(np.float32))
        sc = pre.select('pair_id', 'hand_id').with_columns(pl.Series('p', m.predict_proba(Xp)[:, 1]))
        g = pool(sc, prior).with_columns(pl.lit(fold).alias('fold_new'))
        parts.append(g)
        info[fold] = {'train_hands': int(len(y)), 'train_listed': int(y.sum()), 'train_pairs': int(d['pair_id'].n_unique()), 'prior_listed': round(prior, 4), 'scored_pairs': g.height, 'scored_hands': sc.height,
                      'top_gain': [(k, round(float(v) / tot, 3)) for k, v in gain]}
        log(f'fold{fold} evidence view pooled', g.height, round(time.time() - t0), json.dumps(info[fold]['top_gain']))
    out = pl.concat(parts); out.write_parquet(VD / f'evidence_view_prefix{SUFFIX}.parquet')
    return out, info


def X_of(c, fs):
    cols = []
    for v in fs:
        if v == 'sh': cols.append(np.log(c['shared_hands'].to_numpy().astype(float)))
        elif v == 'R1': cols.append(c['R1_prefix'].fill_null(False).to_numpy().astype(float))
        elif v == 'FULL': cols.append(lg(c['risk_prefix'].to_numpy()))
        elif v.startswith('E_') or v.startswith('H_'): cols.append(c[v].to_numpy().astype(float))
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
    ev, res['plant_fit'] = evidence_view(t0)
    mp = pl.read_parquet(VD / 'multiview_prefix.parquet')
    tv = pl.read_parquet(VD / 'tail_views_prefix.parquet').sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).drop('fold_new')
    ev1 = ev.sort('pair_id', 'fold_new').unique(subset=['pair_id'], keep='first', maintain_order=True).drop('fold_new')
    pops = pl.concat([pl.read_parquet(VD / f'prefix_fold{k}_ext_population.parquet').select('pair_id', 'why') for k in (1, 2)]).unique(subset=['pair_id'], keep='first')
    r1 = pl.read_parquet(VD / 'prefix_rescoring.parquet').select('pair_id', 'R1_prefix').unique(subset=['pair_id'], keep='first')
    d = mp.join(tv, on='pair_id', how='left').join(ev1, on='pair_id', how='left').join(pops, on='pair_id', how='left').join(r1, on='pair_id', how='left').with_columns(pl.col('R1_prefix').fill_null(False))
    n_tail_total = int((mp['risk_prefix'] < .1).sum()); n_tail_sample = int((d['why'] == 'tail_sample').sum())
    res['sizes'] = {'testbed_pairs': mp.height, 'tail_lt_.1': n_tail_total, 'tail_sample': n_tail_sample, 'evidence_view_pairs': int(d['E_llr'].is_not_null().sum())}
    blk = d.filter(pl.col('risk_prefix') >= .5); sep = {}
    for v in ['FULL', 'RAW', 'SHORT3', 'ALL5'] + E_NEW:
        col = 'risk_prefix' if v == 'FULL' else (v if v.startswith('E_') else f'r_{v}')
        g = {c: blk.filter((pl.col('cls') == c) & pl.col(col).is_not_null())[col].to_numpy() for c in ('pos', 'undisclosed', 'FP', 'mid', 'ambiguous')}
        sep[v] = {'AUC_pos_vs_FP': round(auc(g['pos'], g['FP']), 4), 'AUC_undisc_vs_FP': round(auc(g['undisclosed'], g['FP']), 4), 'AUC_pos_vs_amb': round(auc(g['pos'], g['ambiguous']), 4), 'n': {k: len(x) for k, x in g.items()}}
    res['block_separation'] = sep
    # by-family separation of the E view inside the block
    res['block_separation_by_family'] = {}
    for fam in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
        pos = blk.filter((pl.col('cls') == 'pos') & (pl.col('behavior_family') == fam) & pl.col('E_llr').is_not_null())
        res['block_separation_by_family'][fam] = {v: round(auc(pos[v].to_numpy(), blk.filter((pl.col('cls') == 'FP') & pl.col(v).is_not_null())[v].to_numpy()), 4) for v in ('E_top5', 'E_llr', 'E_max')} | {'n_pos': pos.height}
    tailpos = d.filter((pl.col('label') == 1) & (pl.col('risk_prefix') < .5))
    negs_all = d.filter(pl.col('label').is_null() | (pl.col('label') == 0)); negs_e = negs_all.filter(pl.col('E_llr').is_not_null())
    tl = {'n_tail_pos': tailpos.height}
    for v in ['FULL', 'SHORT3'] + E_NEW:
        col = 'risk_prefix' if v == 'FULL' else (v if v.startswith('E_') else f'r_{v}')
        ref = np.sort((negs_e if v.startswith('E_') else negs_all)[col].drop_nulls().to_numpy()); x = tailpos[col].to_numpy()
        pct = np.searchsorted(ref, x, side='left') / len(ref)
        tl[v] = {'pctile_median': round(float(np.nanmedian(pct)), 4), 'n_above_p99': int(np.nansum(pct >= .99)), 'n_above_p90': int(np.nansum(pct >= .9)), 'pctiles': [round(float(p), 4) for p in pct]}
    res['tail_positives'] = tl
    base_fs = VIEWS + ['sh', 'R1']; m3_fs = base_fs + ['SHORT3', 'ALL5']
    ext_sets = {'base(M2)': base_fs, 'M3': m3_fs, **{f'M3+{v}': m3_fs + [v] for v in E_NEW}, 'M3+E_top5+E_llr': m3_fs + ['E_top5', 'E_llr'], 'M3+all E': m3_fs + E_NEW, 'M2+E_top5': base_fs + ['E_top5']}
    out = {}
    for floor in (.5, .1):
        for dn, (posset, negset) in DEFS.items():
            c = d.filter((pl.col('risk_prefix') >= floor) & pl.col('cls').is_in(posset + negset) & pl.col('E_llr').is_not_null() & pl.col('r_SHORT3').is_not_null())
            y = c['cls'].is_in(posset).to_numpy().astype(float)
            r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'frozen': round(ap(y, c['risk_prefix'].to_numpy()), 4)}
            for name, fs in ext_sets.items():
                pred, (m1, m2), (in1, in2, insh) = crossfold(c, y, fs)
                r[name] = {'pooled': round(ap(y, pred), 4), 'f2sh': round(ap(y[in2 | insh], pred[in2 | insh]), 4), 'f1': round(ap(y[in1], pred[in1]), 4),
                           'w1': {v: round(float(w), 2) for v, w in zip(fs, m1[0]) if v not in VIEWS + ['sh']}, 'w2': {v: round(float(w), 2) for v, w in zip(fs, m2[0]) if v not in VIEWS + ['sh']}}
            out[f'floor {floor} def {dn}'] = r
    res['blends'] = out
    cpop = d.filter(pl.col('E_llr').is_not_null() & pl.col('r_SHORT3').is_not_null())
    ypop = (cpop['label'] == 1).to_numpy().astype(float)
    wpop = np.where((cpop['why'] == 'tail_sample').to_numpy(), n_tail_total / max(1, n_tail_sample), 1.0)
    fit_rows = ((cpop['risk_prefix'] >= .1) & cpop['cls'].is_in(DEFS['C'][0] + DEFS['C'][1])).to_numpy()
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2; tab = cpop['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
    op = {'n_pos': int(ypop.sum()), 'n_rows': int(len(ypop)), 'frozen': round(wap(ypop, cpop['risk_prefix'].to_numpy(), wpop), 4)}
    for name, fs in ext_sets.items():
        X = X_of(cpop, fs); pred = np.full(len(ypop), np.nan)
        m1 = logreg(X[in1 & fit_rows], ypop[in1 & fit_rows], L2); pred[in2 | insh] = predict(m1, X[in2 | insh])
        m2 = logreg(X[in2 & fit_rows], ypop[in2 & fit_rows], L2); pred[in1] = predict(m2, X[in1])
        op[name] = {'open_AP': round(wap(ypop, pred, wpop), 4), 'f1': round(wap(ypop[in1], pred[in1], wpop[in1]), 4), 'f2sh': round(wap(ypop[in2 | insh], pred[in2 | insh], wpop[in2 | insh]), 4)}
    res['open_AP'] = op
    res['seconds'] = round(time.time() - t0)
    write_json(VD / f'evidence_view_eval{SUFFIX}.json', res)
    log(json.dumps({k: res[k] for k in ('sizes', 'block_separation', 'block_separation_by_family')}, indent=1)[:5000])
    log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'pctiles'} if isinstance(v, dict) else v for k, v in res['tail_positives'].items()}, indent=1)[:3000])
    log(json.dumps(res['blends'], indent=1)[:9000]); log(json.dumps(res['open_AP'], indent=1)[:3000])


if __name__ == '__main__':
    main()
