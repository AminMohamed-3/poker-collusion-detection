"""STEP262 step 4: the registered arms on the tail harness.

registration262.json 'arms' / 'gates'.  Greedy in the fixed order T1 MECHLGB, T2 MECHKEY (T3 is run by
step262_arms.py t3 only if the clock allows); CONTROL PLUSLGB and the NOISE control are measured
against the same base stack and reported, never built on.

Every arm produces ONE type column on the scale of the ten FINAL2 columns (registration 'scale_map')
and is measured by src/pu_pairs/step262_repro_tail.py, i.e. by the code path that reproduced the
recorded reading exactly.  Development labels only.

Usage:  python step262_arms.py
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
import sys, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl, lightgbm as lgb, joblib               # noqa: E402
from pu_pairs import step262_repro_tail as H                            # noqa: E402
from pu_pairs import step159c_C_combine_dev as S159                     # noqa: E402
from pu_pairs.build_allpairs_fold import fold_pools, CONF, EXP          # noqa: E402
from pu_pairs.step151c_C_SX_views_dev import view_frames_seeded         # noqa: E402
sys.path.insert(0, str(ROOT + 'src'))
from robust_pair_experiment import MODEL_PARAMS                         # noqa: E402

OUT = H.OUT
SURF = H.SURF
AGG = ['cnt_q999', 'cnt_q99', 'top1_logit', 'top3_logit', 'top5_logit', 'noisy_or']
DRAWS = 8
NBOOT = 2000
NOISE_SEED = 262001


def log(*a):
    print(*a, flush=True)


def inputs(d, arm):
    """the registered 7 inputs: log1p on the counts and the noisy-or, the top-k logits raw, log n_hands."""
    return np.column_stack([
        np.log1p(d[f'{arm}_cnt_q999'].to_numpy().astype(float)),
        np.log1p(d[f'{arm}_cnt_q99'].to_numpy().astype(float)),
        d[f'{arm}_top1_logit'].to_numpy().astype(float),
        d[f'{arm}_top3_logit'].to_numpy().astype(float),
        d[f'{arm}_top5_logit'].to_numpy().astype(float),
        np.log1p(d[f'{arm}_noisy_or'].to_numpy().astype(float)),
        np.log(d['n_hands'].to_numpy().astype(float)),
    ])


def fit_scores(dev, tr, arm):
    """8 draws per fold; returns raw type value = mean over draws of lg(p).  Fold-matched held pools."""
    raw = np.full(dev.height, np.nan)
    devf = dev['fold'].to_numpy()
    info = {}
    for fold in (1, 2):
        held = set(fold_pools(fold))
        te = devf == fold
        Xte = inputs(dev.filter(pl.Series(te)), arm)
        acc = np.zeros(int(te.sum()))
        for k in range(DRAWS):
            t = tr.filter((pl.col('k') == k) & ~pl.col('table_id').is_in(sorted(held)))
            y = (t['label'] == 1).to_numpy().astype(int)
            m = lgb.LGBMClassifier(**{**MODEL_PARAMS, 'random_state': 7329 + k, 'n_jobs': 1}).fit(inputs(t, arm), y)
            acc += S159.lg(m.predict_proba(Xte)[:, 1])
            if k == 0:
                info[f'fold{fold}'] = {'n_train_rows': int(t.height), 'n_train_pairs': int(t['pair_id'].n_unique()),
                                       'n_pos_rows': int(y.sum()), 'n_scored': int(te.sum())}
        raw[te] = acc / DRAWS
    assert np.isfinite(raw).all()
    return raw, info


def fit_scores_t3(dev, tr, n_extra=8):
    """T3 MECHLGB+: T1 plus the n_extra highest-gain production frame columns of the fold's draw-0 fit.
    Training columns come from step151c.view_frames_seeded (the PRODUCTION thinned frames, same seeds
    as the aggregates); scoring columns from the surface's own pair frames."""
    features = list(joblib.load(ROOT + 'outputs_v11_conditional/pair-model.joblib')['features'])
    raw = np.full(dev.height, np.nan)
    devf, devs = dev['fold'].to_numpy(), dev['surface'].to_numpy()
    info = {}
    for fold in (1, 2):
        held = set(fold_pools(fold))
        pref = {p: pl.concat([pl.read_parquet(x, columns=list(dict.fromkeys(['pair_id'] + features)))
                              for x in sorted((EXP / f'fold{fold}-{p}').glob('batch_*-pairs.parquet'))])
                for p in SURF}
        acc = {}
        cols8 = None
        for k in range(DRAWS):
            fr = view_frames_seeded(CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet',
                                    held, features, k)
            parts = []
            for frac in (1.0, 0.7, 0.4):
                parts.append(tr.filter((pl.col('k') == k) & (pl.col('frac') == frac))
                             .join(fr[frac].select(['pair_id'] + features), on='pair_id', how='inner'))
            T = pl.concat(parts)
            del fr
            y = (T['label'] == 1).to_numpy().astype(int)
            if cols8 is None:
                m0 = lgb.LGBMClassifier(**{**MODEL_PARAMS, 'random_state': 7329, 'n_jobs': 1}).fit(
                    T.select(features).to_numpy(), y)
                g = m0.booster_.feature_importance(importance_type='gain')
                cols8 = [features[i] for i in sorted(np.argsort(-g, kind='stable')[:n_extra])]
                info[f'fold{fold}'] = {'extra_columns': cols8, 'n_train_rows': int(T.height),
                                       'n_pos_rows': int(y.sum())}
            Xtr = np.column_stack([inputs(T, 'mechplus'), T.select(cols8).to_numpy().astype(float)])
            m = lgb.LGBMClassifier(**{**MODEL_PARAMS, 'random_state': 7329 + k, 'n_jobs': 1}).fit(Xtr, y)
            for p in SURF:
                sel = (devf == fold) & (devs == p)
                if not sel.any():
                    continue
                rows = dev.filter(pl.Series(sel))
                j = rows.join(pref[p].select(['pair_id'] + cols8), on='pair_id', how='left', validate='1:1', maintain_order='left')
                assert j.height == rows.height and j[cols8[0]].null_count() == 0, (fold, p)
                Xte = np.column_stack([inputs(j, 'mechplus'), j.select(cols8).to_numpy().astype(float)])
                acc[(fold, p)] = acc.get((fold, p), 0.0) + S159.lg(m.predict_proba(Xte)[:, 1])
        for p in SURF:
            sel = (devf == fold) & (devs == p)
            if sel.any():
                raw[sel] = acc[(fold, p)] / DRAWS
    assert np.isfinite(raw).all()
    return raw, info


def scale_map(D):
    """per surface: the mean of the ten FINAL2 columns' means / sds on that surface's D rows."""
    M = {}
    for p in SURF:
        d = D[p]
        mu = float(np.mean([float(d[c].mean()) for c in H.BASE]))
        sd = float(np.mean([float(d[c].std()) for c in H.BASE]))
        M[p] = (mu, sd)
    return M


def typecol(dev, raw, name, M):
    """z-score the raw value within each surface, then map onto the FINAL2 columns' scale."""
    out = np.full(dev.height, np.nan)
    surf = dev['surface'].to_numpy()
    for p in SURF:
        m = surf == p
        v = raw[m]
        z = (v - v.mean()) / v.std()
        out[m] = z * M[p][1] + M[p][0]
    assert np.isfinite(out).all()
    return dev.select('pair_id', 'fold', 'surface').with_columns(pl.Series(name, out))


def boot(D, colsA, colsB, n=NBOOT, seed=262):
    """paired bootstrap over held pools: resample TABLES with replacement, stratified by the row's fold."""
    pre = {}
    for p in SURF:
        d = D[p]
        s = d['risk'].to_numpy().astype(float)
        lab = d['label'].to_numpy()
        rf = d['risk_full'].fill_null(0).to_numpy().astype(float)
        yp = ((lab == 1) | ((lab == -1) & (rf >= .9))).astype(int)
        kA = d.select(pl.mean_horizontal(colsA)).to_series().to_numpy()
        kB = d.select(pl.mean_horizontal(colsB)).to_series().to_numpy()
        tab = d['table_id'].to_numpy()
        fo = d['fold'].to_numpy()
        order = np.argsort(tab, kind='stable')
        starts = np.flatnonzero(np.r_[True, tab[order][1:] != tab[order][:-1]])
        rows = [order[a:b] for a, b in zip(starts, np.r_[starts[1:], len(order)])]
        tfold = np.array([fo[r[0]] for r in rows])
        pre[p] = (s, yp, kA, kB, rows, tfold)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for b in range(n):
        acc = 0.0
        for p in SURF:
            s, yp, kA, kB, rows, tfold = pre[p]
            pick = []
            for f in (1, 2):
                idx = np.flatnonzero(tfold == f)
                if len(idx):
                    pick.append(rng.choice(idx, size=len(idx), replace=True))
            pick = np.concatenate(pick)
            ix = np.concatenate([rows[i] for i in pick])
            ss, yy = s[ix], yp[ix]
            blk = ss < .5
            acc += S159.ap(yy, S159.permute(ss, kA[ix], blk)) - S159.ap(yy, S159.permute(ss, kB[ix], blk))
        out[b] = acc / len(SURF)
    return {'P_gt_0': float((out > 0).mean()), 'p05': float(np.percentile(out, 5)),
            'p50': float(np.percentile(out, 50)), 'p95': float(np.percentile(out, 95)),
            'mean': float(out.mean()), 'n': int(n)}


def rank_moves(D, colsA, colsB, lowdose):
    """mean rank change inside the tail (negative = promoted toward the top)."""
    res = {}
    for p in SURF:
        d = D[p]
        s = d['risk'].to_numpy().astype(float)
        blk = s < .5
        kA = d.select(pl.mean_horizontal(colsA)).to_series().to_numpy()
        kB = d.select(pl.mean_horizontal(colsB)).to_series().to_numpy()
        idx = np.flatnonzero(blk)
        rA = np.empty(len(idx), np.int64); rA[np.argsort(-kA[idx], kind='mergesort')] = np.arange(len(idx))
        rB = np.empty(len(idx), np.int64); rB[np.argsort(-kB[idx], kind='mergesort')] = np.arange(len(idx))
        mv = (rA - rB).astype(float)
        r1 = d['R1_full'].fill_null(False).to_numpy()[idx]
        lab = d['label'].to_numpy()[idx]
        pid = np.array(d['pair_id'].to_list())[idx]
        ld = np.isin(pid, list(lowdose))
        res[p] = {'n_tail': int(len(idx)),
                  'R1_pairs': int(r1.sum()), 'mean_move_R1': float(mv[r1].mean()) if r1.any() else None,
                  'mean_move_nonR1': float(mv[~r1].mean()),
                  'n_lowdose_pos': int(ld.sum()), 'mean_move_lowdose_pos': float(mv[ld].mean()) if ld.any() else None,
                  'n_labelled_pos': int((lab == 1).sum()),
                  'mean_move_labelled_pos': float(mv[lab == 1].mean()) if (lab == 1).any() else None}
    return res


def lowdose_pairs():
    ev = pl.read_csv(DATA + 'development_evidence.csv').select('pair_id', 'hand_id').unique()
    n = ev.group_by('pair_id').len()
    lab = pl.read_csv(DATA + 'development_labels.csv').select('pair_id', 'label')
    j = n.join(lab, on='pair_id', how='inner').filter((pl.col('label') == 1) & pl.col('len').is_in([3, 4]))
    return set(j['pair_id'].to_list())


def main():
    t0 = time.time()
    dest = OUT + 'results262.json'
    assert not os.path.exists(dest), dest
    reg = OUT + 'registration262.json'
    import hashlib
    rec = {'registration_sha256': hashlib.sha256(open(reg, 'rb').read()).hexdigest(),
           'repro262': json.load(open(OUT + 'repro262.json', encoding='utf-8'))['gate_passes'],
           'base_stack': H.BASE}
    assert rec['repro262'], 'the FIRST GATE did not pass'

    dev = pl.read_parquet(OUT + 'aggregates262.parquet')
    tr = pl.read_parquet(OUT + 'train_aggs262.parquet')
    V = H.stack_V('arch3')
    D0 = H.build_D(V)
    M = scale_map(D0)
    rec['scale_map'] = {p: {'mu': M[p][0], 'sd': M[p][1]} for p in M}
    for p in SURF:
        assert D0[p].height == dev.filter(pl.col('surface') == p).height

    cols = []
    rec['arms'] = {}
    raws = {}
    for name, arm, kind in (('MECH_T1', 'mechplus', 'lgb'), ('MECH_T2', 'mechplus', 'key'),
                            ('PLUS_T1', 'plus', 'lgb'), ('NOISE_T1', 'mechplus', 'noise')):
        if kind == 'lgb':
            raw, info = fit_scores(dev, tr, arm)
        elif kind == 'key':
            raw, info = dev[f'{arm}_top3_logit'].to_numpy().astype(float), {'fit': 'none'}
        else:
            d2 = dev.clone()
            surf = dev['surface'].to_numpy()
            perm = np.arange(dev.height)
            for si, p in enumerate(SURF):
                m = np.flatnonzero(surf == p)
                perm[m] = m[np.random.default_rng(NOISE_SEED + si).permutation(len(m))]
            acols = [f'{arm}_{a}' for a in AGG] + ['n_hands']
            d2 = d2.with_columns([pl.Series(c, dev[c].to_numpy()[perm]) for c in acols])
            raw, info = fit_scores(d2, tr, arm)
            info['permutation_seeds'] = [NOISE_SEED + i for i in range(len(SURF))]
        raws[name] = raw
        cols.append(typecol(dev, raw, name, M))
        rec['arms'][name] = {'fit': info}
        log('arm', name, 'built', round(time.time() - t0, 1), 's')

    built = ['MECH_T1', 'MECH_T2', 'PLUS_T1', 'NOISE_T1']
    try:
        raw, info = fit_scores_t3(dev, tr)
        cols.append(typecol(dev, raw, 'MECH_T3', M))
        rec['arms']['MECH_T3'] = {'fit': info}
        built.append('MECH_T3')
        log('arm MECH_T3 built', round(time.time() - t0, 1), 's')
    except Exception as e:                                     # registered as conditional
        rec['arms']['MECH_T3'] = {'NOT_RUN': repr(e)[:400]}
        log('arm MECH_T3 NOT RUN', repr(e)[:300])

    D = H.build_D(V, extra=cols)
    for p in SURF:
        for c in built:
            assert D[p][c].null_count() == 0, (p, c)
    ld = lowdose_pairs()
    rec['n_lowdose_positive_pairs'] = len(ld)

    cur = H.measure(D, H.BASE)
    rec['FINAL2_16_abs'] = cur
    stack = list(H.BASE)
    rec['greedy'] = []
    for t in [x for x in ('MECH_T1', 'MECH_T2', 'MECH_T3') if x in built]:
        cand = H.measure(D, stack + [t])
        r = S159.rule({p: {'a': cand[p], 'b': cur[p]} for p in cand}, 'a', 'b')
        r['type'] = t
        r['abs'] = cand
        r['per_surface'] = {p: {m: cand[p][m] - cur[p][m] for m in cand[p]} for p in cand}
        r['bootstrap'] = boot(D, stack + [t], stack)
        r['soft_rule_S'] = bool(r['bootstrap']['P_gt_0'] >= .75)
        r['rank_moves'] = rank_moves(D, stack + [t], stack, ld)
        rec['greedy'].append(r)
        log(t, json.dumps({k: v for k, v in r.items() if k not in ('per_surface', 'abs', 'rank_moves')}))
        if r['passes']:
            stack.append(t)
            cur = cand
    rec['stack_after_greedy'] = stack
    rec['kept_types'] = [c for c in stack if c not in H.BASE]

    rec['controls'] = {}
    for t in ('PLUS_T1', 'NOISE_T1'):
        cand = H.measure(D, H.BASE + [t])
        r = S159.rule({p: {'a': cand[p], 'b': rec['FINAL2_16_abs'][p]} for p in cand}, 'a', 'b')
        r['type'] = t
        r['abs'] = cand
        r['per_surface'] = {p: {m: cand[p][m] - rec['FINAL2_16_abs'][p][m] for m in cand[p]} for p in cand}
        r['bootstrap'] = boot(D, H.BASE + [t], H.BASE)
        r['rank_moves'] = rank_moves(D, H.BASE + [t], H.BASE, ld)
        rec['controls'][t] = r
        log(t, json.dumps({k: v for k, v in r.items() if k not in ('per_surface', 'abs', 'rank_moves')}))

    rec['correlations'] = {}
    for p in SURF:
        d = D[p]
        base_key = d.select(pl.mean_horizontal(H.BASE)).to_series().to_numpy()
        rec['correlations'][p] = {c: float(np.corrcoef(base_key, d[c].to_numpy())[0, 1]) for c in built}
        rec['correlations'][p]['MECH_T1_vs_PLUS_T1'] = float(np.corrcoef(d['MECH_T1'].to_numpy(), d['PLUS_T1'].to_numpy())[0, 1])

    if rec['kept_types']:
        fin = H.measure(D, stack)
        rec['FINALM_minus_FINAL2_16'] = S159.rule({p: {'a': fin[p], 'b': rec['FINAL2_16_abs'][p]} for p in fin}, 'a', 'b')
        rec['prediction_vs_step251_file'] = {'point': .65 * .37 * rec['FINALM_minus_FINAL2_16']['mean_comp'],
                                             'band80_halfwidth': .0025,
                                             'honest_note': 'the three tail reads were +.0015 / +.0015 / +.0009 and the read SE is about .00065'}
    else:
        rec['FINALM_minus_FINAL2_16'] = None
        rec['prediction_vs_step251_file'] = {'point': 0.0, 'band80_halfwidth': 0.0}
    rec['build_authorised'] = bool(rec['kept_types'] and
                                   all(g['soft_rule_S'] for g in rec['greedy'] if g['passes']) and
                                   not rec['controls']['NOISE_T1']['passes'])
    rec['seconds'] = round(time.time() - t0, 1)
    json.dump(rec, open(dest, 'w', encoding='utf-8'), indent=1)
    log('KEPT', rec['kept_types'], 'BUILD', rec['build_authorised'], rec['seconds'], 's')


if __name__ == '__main__':
    main()
