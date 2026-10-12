"""Step 61: re-scoring of the 60 %-PREFIX block (folds 1-2) under hand drops, at evaluation-like history length.
Two purposes. (1) Validate the spill-over rule R1 (dominated & risk_noT < .1; candidate C, inside the measured candidate S)
on certified data: at the prefix, every block pair is certified by its full-history risk (< .1 = length-induced false
positive, >= .9 = positive or undisclosed positive), so the R1 flag computed AT THE PREFIX (dominance from the prefix
scores, T = the dominator's other player) can be cross-tabulated against the certification. (2) A risk-STABILITY feature:
re-score each block pair after dropping 30 % / 50 % of its hands at random (3 seeds each); a pair whose risk rests on a
few hands (false positive) should be less stable than a genuine colluder whose collusion is a constant per-hand rate
(14.20). Both are label-free on evaluation (the same quantities exist for the evaluation block from step 48: risk_noT,
R1, risk_rand_*), so anything that validates here can be used in a candidate.
Per-hand frames of the prefix block pairs are rebuilt with the fold policy (as step 48b, restricted to the first 60 % of
each table's development hands), parity with the saved prefix risk is asserted, frames are cached.
Outputs: experiments/viewdis_20260911/{prefix_fold{k}_raw.parquet, prefix_fold{k}_ca.parquet, prefix_rescoring.parquet, prefix_rescoring.json}."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time, gc
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
import numpy as np
import polars as pl
import joblib
from pu_pairs.build_allpairs_fold import CONF, STYLE, EXP, fold_policy, fold_pools, base_features, v2_features, conditional_features
from pu_pairs.step48_spillover_rescoring import dominance, make_scorer
from pu_pairs.step60_multiview_blend import logreg, predict, lg, ap, auc, NEW_VIEWS

VD = CV / 'experiments' / 'viewdis_20260911'
PREFIX = .6
THR = .5
SEEDS = (11, 12, 13)


def log(*a):
    print(*a, flush=True)


def prefix_hands(tables):
    hd = pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet').filter((pl.col('phase') == 'development') & pl.col('table_id').is_in(tables))
    hd = hd.sort('table_id', 'started_at').with_columns((pl.int_range(pl.len()).over('table_id') / pl.len().over('table_id')).alias('_pos'))
    return hd.filter(pl.col('_pos') < PREFIX).drop('_pos')


def build_frames(fold, block, t0):
    fr, fc = VD / f'prefix_fold{fold}_raw.parquet', VD / f'prefix_fold{fold}_ca.parquet'
    if fr.exists() and fc.exists():
        raw, ca = pl.read_parquet(fr), pl.read_parquet(fc)
        if set(raw['pair_id']) == set(block['pair_id']):
            return raw, ca
    policy = joblib.load(fold_policy(fold))
    caf = [c for c in pl.read_parquet(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', n_rows=1).columns if c.startswith('ca_')]
    counts = {r[0]: np.array(r[1:], dtype=np.int64) for r in pl.read_parquet(STYLE).iter_rows()}
    tables = sorted(set(block['table_id']))
    hands_all = prefix_hands(tables)
    raws, cas = [], []
    for i, t in enumerate(tables):
        h = hands_all.filter(pl.col('table_id') == t)
        p = block.filter(pl.col('table_id') == t).select('pair_id', 'player_1', 'player_2', 'table_id')
        raw = base_features(h, p); raw = v2_features(raw, p)
        keys = raw.select('pair_id', 'hand_id', 'table_id').join(p.select('pair_id', 'player_1', 'player_2'), on='pair_id', validate='m:1')
        ca, _, _ = conditional_features(keys, h.select('hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind'), policy, counts, caf)
        raws.append(raw); cas.append(ca); gc.collect()
        if i % 20 == 0:
            log(json.dumps({'fold': fold, 'table': i, 'of': len(tables), 'seconds': round(time.time() - t0)}))
    raw = pl.concat(raws, how='vertical_relaxed').sort('pair_id', 'hand_id'); ca = pl.concat(cas, how='vertical_relaxed').sort('pair_id', 'hand_id')
    raw.write_parquet(fr); ca.write_parquet(fc)
    return raw, ca


def run_fold(fold, t0):
    scores = pl.read_parquet(EXP / f'fold{fold}-prefix60' / 'scores.parquet')
    block = scores.filter(pl.col('risk') >= THR).sort('pair_id'); ids = block['pair_id'].to_list()
    dom = dominance(block)
    raw, ca = build_frames(fold, block, t0)
    assert raw.select('pair_id', 'hand_id').equals(ca.select('pair_id', 'hand_id'))
    caf = [c for c in ca.columns if c.startswith('ca_')]
    bundle = joblib.load(CONF / 'results' / f'fold{fold}' / 'conditional-pair.joblib'); model = bundle['model']; features = list(bundle['features'])
    names = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id')]; events = [c for c in names if c.startswith('v2_')]
    score = make_scorer(model, features, names, events, caf)
    full, n_full = score(raw, ca)
    saved = dict(block.select('pair_id', 'risk').iter_rows())
    diff = np.array([abs(full[p] - saved[p]) for p in ids])
    parity = {'max_abs_diff': float(diff.max()), 'n_gt_1e-6': int((diff > 1e-6).sum())}
    log(json.dumps({'fold': fold, 'parity': parity, 'block': len(ids), 'dominated': len(dom), 'pair_hands': raw.height, 'seconds': round(time.time() - t0)}))
    hands = sorted(set(raw['hand_id']))
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet', columns=['hand_id', 'player_id']).filter(pl.col('hand_id').is_in(hands))
    dealt = defaultdict(set)
    for h_, p_ in seats.iter_rows():
        dealt[h_].add(p_)
    hands_of = dict(raw.group_by('pair_id').agg(pl.col('hand_id')).iter_rows())
    T_of, kind_of = {}, {}
    for pid, p1, p2 in block.select('pair_id', 'player_1', 'player_2').iter_rows():
        if pid in dom:
            T_of[pid], kind_of[pid] = dom[pid][2], 'dominator_partner'
        else:
            cnt = defaultdict(int)
            for h_ in hands_of[pid]:
                for q in dealt[h_] - {p1, p2}:
                    cnt[q] += 1
            T_of[pid], kind_of[pid] = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0], 'max_copresent'
    pairs_arr = raw['pair_id'].to_list(); hands_arr = raw['hand_id'].to_list()
    isT = np.array([T_of[p] in dealt[h] for p, h in zip(pairs_arr, hands_arr)])
    noT, n_noT = score(raw.filter(pl.Series(~isT)), ca.filter(pl.Series(~isT)))
    rows_of = defaultdict(list)
    for i, p in enumerate(pairs_arr):
        rows_of[p].append(i)
    runs = {}
    for frac in (.3, .5):
        for seed in SEEDS:
            rs = np.random.default_rng(seed + 100 * fold + int(frac * 100)); keep = np.ones(raw.height, dtype=bool)
            for p, rows in rows_of.items():
                d = int(round(frac * len(rows)))
                if d > 0:
                    keep[rs.choice(rows, size=d, replace=False)] = False
            runs[(frac, seed)] = score(raw.filter(pl.Series(keep)), ca.filter(pl.Series(keep)))[0]
    log(json.dumps({'fold': fold, 'variants_done': True, 'seconds': round(time.time() - t0)}))
    share_T = {p: float(np.mean(isT[rows_of[p]])) for p in ids}
    out = []
    for p in ids:
        r = {'fold': fold, 'pair_id': p, 'risk_prefix': saved[p], 'risk_rescored': full[p], 'n_hands': n_full[p], 'dominated': p in dom, 'dom_risk': dom[p][0] if p in dom else None, 'dom_pair': dom[p][1] if p in dom else None,
             'T_kind': kind_of[p], 'share_hands_with_T': share_T[p], 'n_noT': n_noT.get(p, 0), 'risk_noT': noT.get(p, float('nan'))}
        for frac in (.3, .5):
            v = [runs[(frac, s)].get(p, float('nan')) for s in SEEDS]
            r[f'rand{int(frac*100)}_mean'] = float(np.nanmean(v)); r[f'rand{int(frac*100)}_min'] = float(np.nanmin(v)); r[f'rand{int(frac*100)}_max'] = float(np.nanmax(v))
        out.append(r)
    return pl.DataFrame(out), parity


def analyse(d):
    d = d.with_columns((pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .1)).alias('R1_prefix'),
                       (pl.col('risk_prefix') - pl.col('rand30_min')).alias('drop30_min'), (pl.col('risk_prefix') - pl.col('rand50_min')).alias('drop50_min'),
                       (pl.col('risk_prefix') - pl.col('rand30_mean')).alias('drop30_mean'), (pl.col('risk_prefix') - pl.col('rand50_mean')).alias('drop50_mean'))
    res = {}
    ct = d.group_by('cls', 'R1_prefix').len().sort('cls', 'R1_prefix')
    res['R1_prefix_by_class'] = {f"{c} R1={r}": int(n) for c, r, n in ct.iter_rows()}
    ct = d.group_by('cls', 'dominated').len().sort('cls', 'dominated')
    res['dominated_by_class'] = {f"{c} dom={r}": int(n) for c, r, n in ct.iter_rows()}
    res['R1_prefix_vs_R1_full'] = {f"prefix={a} full={b}": int(n) for a, b, n in d.group_by('R1_prefix', 'R1_full').len().sort('R1_prefix', 'R1_full').iter_rows()}
    g = {c: d.filter(pl.col('cls') == c) for c in ('pos', 'undisclosed', 'undisclosed_R1', 'ambiguous', 'mid', 'FP')}
    res['medians'] = {c: {k: round(float(x[k].median()), 3) for k in ('risk_prefix', 'risk_noT', 'rand30_mean', 'rand30_min', 'rand50_mean', 'rand50_min', 'drop30_min', 'drop50_min', 'share_hands_with_T', 'n_hands')} for c, x in g.items() if x.height}
    res['AUC_pos_vs_FP'] = {k: round(auc(-g['pos'][k].to_numpy(), -g['FP'][k].to_numpy()), 4) for k in ('drop30_min', 'drop50_min', 'drop30_mean', 'drop50_mean')}
    res['AUC_pos_vs_FP'].update({k: round(auc(g['pos'][k].to_numpy(), g['FP'][k].to_numpy()), 4) for k in ('rand30_min', 'rand50_min', 'rand30_mean', 'rand50_mean', 'risk_noT', 'risk_prefix')})
    res['AUC_undisclosed_vs_FP'] = {k: round(auc(g['undisclosed'][k].to_numpy(), g['FP'][k].to_numpy()), 4) for k in ('rand30_min', 'rand50_min', 'rand50_mean', 'risk_noT', 'risk_prefix')}
    # within-band: does stability separate at matched frozen risk?
    wb = {}
    for lo, hi in ((.5, .9), (.9, .99), (.99, 1.01)):
        b = d.filter((pl.col('risk_prefix') >= lo) & (pl.col('risk_prefix') < hi))
        P = b.filter(pl.col('cls').is_in(['pos', 'undisclosed'])); N = b.filter(pl.col('cls') == 'FP')
        wb[f'[{lo},{hi})'] = {'n_pos': P.height, 'n_FP': N.height, **{k: round(auc(P[k].to_numpy(), N[k].to_numpy()), 3) for k in ('rand30_min', 'rand50_min', 'rand50_mean', 'risk_noT', 'r_RAW') if P.height and N.height}}
    res['within_band_AUC_posund_vs_FP'] = wb
    # cross-fold learned blends with the new features added
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    views = ['FULL', 'RAW', 'CA'] + NEW_VIEWS
    def X_of(c, fs):
        cols = []
        for v in fs:
            if v == 'sh': cols.append(np.log(c['shared_hands'].to_numpy().astype(float)))
            elif v == 'FULL': cols.append(lg(c['risk_prefix'].to_numpy()))
            elif v == 'noT': cols.append(lg(c['risk_noT'].fill_nan(0).to_numpy()))
            elif v == 'R1': cols.append(c['R1_prefix'].to_numpy().astype(float))
            elif v == 'dom': cols.append(c['dominated'].to_numpy().astype(float))
            elif v.startswith('rand'): cols.append(lg(c[v].to_numpy()))
            else: cols.append(lg(c[f'r_{v}'].to_numpy()))
        return np.column_stack(cols)
    sets = {'views+sh': views + ['sh'], 'views+sh+noT': views + ['sh', 'noT'], 'views+sh+R1': views + ['sh', 'R1'], 'views+sh+noT+dom': views + ['sh', 'noT', 'dom'],
            'views+sh+rand50_min': views + ['sh', 'rand50_min'], 'views+sh+rand30_min+rand50_min': views + ['sh', 'rand30_min', 'rand50_min'], 'views+sh+noT+rand50_min': views + ['sh', 'noT', 'rand50_min'],
            'FULL+RAW+sh+noT': ['FULL', 'RAW', 'sh', 'noT']}
    defs = {'A': (['pos', 'undisclosed'], ['FP', 'undisclosed_R1']), 'C': (['pos'], ['FP', 'undisclosed_R1', 'ambiguous', 'mid', 'neg'])}
    bl = {}
    for dn, (posset, negset) in defs.items():
        c = d.filter(pl.col('cls').is_in(posset + negset)); y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy()
        risk = c['risk_prefix'].to_numpy(); raw = c['r_RAW'].to_numpy()
        in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
        r = {'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'frozen': round(ap(y, risk), 4), 'fixed w=.5': round(ap(y, lg(risk) + .5 * lg(raw)), 4),
             'fixed w=.5 + R1 bottom': round(ap(y, np.where(c['R1_prefix'].to_numpy(), -100 + lg(c['risk_noT'].fill_nan(0).to_numpy()), lg(risk) + .5 * lg(raw))), 4)}
        for sn, fs in sets.items():
            X = X_of(c, fs); pred = np.full(len(y), np.nan)
            m1 = logreg(X[in1], y[in1], 10.0); pred[in2 | insh] = predict(m1, X[in2 | insh]); m2 = logreg(X[in2], y[in2], 10.0); pred[in1] = predict(m2, X[in1])
            r[sn] = {'pooled_AP': round(ap(y, pred), 4), 'AP_test_f2sh': round(ap(y[in2 | insh], pred[in2 | insh]), 4), 'AP_test_f1': round(ap(y[in1], pred[in1]), 4),
                     'w1': {v: round(float(w), 2) for v, w in zip(fs, m1[0])}, 'w2': {v: round(float(w), 2) for v, w in zip(fs, m2[0])}}
        r['fixed_same_rows'] = [round(ap(y[in2 | insh], (lg(risk) + .5 * lg(raw))[in2 | insh]), 4), round(ap(y[in1], (lg(risk) + .5 * lg(raw))[in1]), 4)]
        bl[dn] = r
    res['blends'] = bl
    return d, res


def main():
    t0 = time.time()
    parts, par = [], {}
    for fold in (1, 2):
        df, parity = run_fold(fold, t0); parts.append(df); par[fold] = parity
    d = pl.concat(parts).unique(subset=['pair_id'], keep='first', maintain_order=True)
    mv = pl.read_parquet(VD / 'multiview_prefix.parquet')
    d = d.join(mv.drop('fold'), on='pair_id', how='left')
    d, res = analyse(d)
    res['parity'] = par; res['n_block_unique'] = d.height; res['seconds'] = round(time.time() - t0)
    d.write_parquet(VD / 'prefix_rescoring.parquet'); write_json(VD / 'prefix_rescoring.json', res)
    log(json.dumps({k: v for k, v in res.items()}, indent=1))


if __name__ == '__main__':
    main()
