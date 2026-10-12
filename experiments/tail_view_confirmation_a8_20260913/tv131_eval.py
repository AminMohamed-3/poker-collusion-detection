"""STEP131 (poker-a8): registered readings of the tail re-order arms on prefix-40/80 (PREDECLARE sha256 ed03060e...).
First a harness check on prefix-60, already read by poker-11's m3p_tail/m3p_boot: this code must reproduce those numbers,
else abort before any registered reading. Development labels, validation only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np, polars as pl
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/'
O, X, VD = C + 'tail_view_confirmation_a8_20260913/', C + 'pu_pairs_20260911/', C + 'viewdis_20260911/'
S11 = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s111/'
REG = 'ed03060eaa3ad14edb53db8f90dd5737e819cc66b55b9113612e8d0f939d7b0c'
assert hashlib.sha256(open(O + 'PREDECLARE_step131_tail_view_confirmation.md', 'rb').read()).hexdigest() == REG
V16 = ['RAW', 'CA', 'BASE', 'OPP', 'V2', 'MEAN', 'TAILS', 'SHORT3', 'ALL5', 'PROD', 'LIN30', 'LIN300', 'DT', 'SP', 'CI', 'FAMMAX']
BANDS = np.array([0, .01, .1, .3, .5, .7, .9, .95, .99, 1.01])
rd = lambda o: json.loads(json.dumps(o), parse_float=lambda x: round(float(x), 5))


def dump(obj):
    json.dump(obj, open(O + 'tv131_eval.json', 'w', encoding='utf-8'), indent=1)


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(1, int(yy.sum())))


def permute(s, z, blk):
    s2 = s.copy(); idx = np.where(blk)[0]; s2[idx[np.argsort(-z[idx], kind='mergesort')]] = np.sort(s[idx])[::-1]
    return s2


rep = json.load(open(O + 'views_run.json', encoding='utf-8'))['replication_prefix60_max_abs_diff']
gate = all(v <= 1e-6 for k, v in rep.items() if k.startswith('r_')) and len([k for k in rep if k.startswith('r_')]) == 16 and rep['tail_views_rows'] > 0 and rep['multiview_rows'] > 0
if not gate:
    dump({'registration_sha256': REG, 'gate_replication': False, 'replication': rep, 'evaluated': False}); sys.exit(print('ABORT: replication gate failed'))
R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
VIEWS = pl.read_parquet(O + 'views_prefix40_60_80.parquet')


def surface(p):
    parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'shared_hands', 'risk'])
             .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
             .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
    vw = VIEWS.filter(pl.col('surface') == p).drop('surface')
    return (pl.concat(parts).join(vw, on=['pair_id', 'fold'], how='left').unique(subset=['pair_id'], keep='first', maintain_order=True)
            .filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))


def readings(d, p):
    s, lab, fold, tab = d['risk'].to_numpy().astype(float), d['label'].to_numpy(), d['fold'].to_numpy(), d['table_id'].to_numpy()
    yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
    yp, yc = (yr.astype(bool) | (unl & (rf >= .9))).astype(int), (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
    b = np.digitize(s, BANDS) - 1; L, U = np.bincount(b, weights=yr * 1.0, minlength=9), np.bincount(b, weights=unl * 1.0, minlength=9)
    c = max(L[i] / (L[i] + U[i]) for i in range(9) if L[i] + U[i] >= 40); q = np.clip(L * (1 / c - 1) / np.maximum(1, U), 0, 1)[b]
    rng, draws = np.random.default_rng(9), []
    for _ in range(30):
        y = yr.copy(); y[unl] = (rng.random(int(unl.sum())) < q[unl]).astype(int); draws.append(y)
    T, LAB = s < .5, {'RAW': yr, 'PP90': yp, 'PPcls': yc}
    K = {'ALL5': lg(d['r_ALL5']), 'P': lg(d['r_SHORT3']) + lg(d['r_ALL5']), 'E': np.mean([lg(s)] + [lg(d[f'r_{v}']) for v in V16], axis=0)}
    full = lambda sc: {**{k: ap(y, sc) for k, y in LAB.items()}, 'SCAR': float(np.mean([ap(y, sc) for y in draws]))}
    B, A = full(s), {k: permute(s, z, T) for k, z in K.items()}
    out = {'n': len(s), 'n_tail': int(T.sum()), 'pos_RAW': int(yr.sum()), 'pos_PP90': int(yp.sum()), 'pos_PPcls': int(yc.sum()), 'tail_pos_RAW': int(yr[T].sum()),
           'tail_pos_PP90': int(yp[T].sum()), 'tail_pos_PPcls': int(yc[T].sum()), 'c': float(c), 'label_values': sorted(set(lab.tolist())),
           'risk_full_null': d['risk_full'].null_count(), 'presumed_with_R1_null': int((unl & (rf >= .9) & d['R1_full'].is_null().to_numpy()).sum()),
           'view_nulls': {v: d[f'r_{v}'].null_count() for v in V16}, 'key_ties_in_tail': {k: int(T.sum() - len(np.unique(z[T]))) for k, z in K.items()},
           'prod_vs_risk_max_abs': float(np.max(np.abs(d['r_PROD'].to_numpy() - s))), 'base': B}
    for an, sc in A.items():
        F = full(sc)
        out[an] = {'delta': {k: F[k] - B[k] for k in F}, 'fold_PP90': {int(k): ap(yp[fold == k], sc[fold == k]) - ap(yp[fold == k], s[fold == k]) for k in (1, 2)}}
    rb, d5 = np.random.default_rng(13), []
    for _ in range(5):
        y = yr.copy(); y[unl] = (rb.random(int(unl.sum())) < q[unl]).astype(int); d5.append(y)
    ev = lambda sc, idx: {'PP90': ap(yp[idx], sc[idx]), 'PPcls': ap(yc[idx], sc[idx]), 'SCAR5': float(np.mean([ap(y[idx], sc[idx]) for y in d5]))}
    ai = np.arange(len(s)); b_all = ev(s, ai)
    for an in A:
        out[an]['delta_SCAR5'] = ev(A[an], ai)['SCAR5'] - b_all['SCAR5']
    ut = np.unique(tab); by = {t: np.where(tab == t)[0] for t in ut}; bt = {an: {k: [] for k in ('PP90', 'PPcls', 'SCAR5')} for an in A}
    for _ in range(200):
        idx = np.concatenate([by[t] for t in rb.choice(ut, len(ut), replace=True)]); b0 = ev(s, idx)
        for an in A:
            o = ev(A[an], idx)
            for k in o:
                bt[an][k].append(o[k] - b0[k])
    for an in A:
        out[an]['boot'] = {k: [float(np.quantile(v, x)) for x in (.05, .5, .95)] + [float(np.mean(np.array(v) > 0))] for k, v in bt[an].items()}
    print(p, 'read', json.dumps(rd({k: out[k] for k in ('n', 'n_tail', 'tail_pos_PP90', 'c', 'prod_vs_risk_max_abs', 'base', 'P', 'E', 'ALL5')})), flush=True)
    return out, yc


ref_t = json.load(open(S11 + 'm3p_tail.json', encoding='utf-8'))['tail_views']; ref_b = json.load(open(S11 + 'm3p_boot.json', encoding='utf-8'))
d60 = surface('prefix60')
mp = pl.read_parquet(VD + 'multiview_prefix.parquet', columns=['pair_id', 'table_id', 'shared_hands', 'label', 'risk_prefix', 'risk_full', 'fold', 'cls']).filter(pl.col('shared_hands') >= 38).sort('pair_id')
chk = {'rows_mine': d60.height, 'rows_ref': mp.height, 'same_pairs': d60['pair_id'].to_list() == mp['pair_id'].to_list()}
ok = chk['same_pairs']
if ok:
    chk.update(fold_eq=bool((d60['fold'].cast(pl.Int64) == mp['fold'].cast(pl.Int64)).all()), label_eq=bool((d60['label'] == mp['label']).all()),
               table_eq=bool((d60['table_id'] == mp['table_id']).all()), risk_max_abs=float((d60['risk'] - mp['risk_prefix']).abs().max()),
               risk_full_max_abs=float((d60['risk_full'] - mp['risk_full']).abs().max()))
    o60, yc60 = readings(d60, 'prefix60')
    chk['PPcls_eq_cls'] = bool((yc60 == ((mp['label'].to_numpy() == 1) | (mp['cls'].to_numpy() == 'undisclosed')).astype(int)).all())
    tol = lambda a, b_, t=6e-5: abs(a - b_) <= t
    for an, rk, bk in (('P', 'SHORT3+ALL5', 'tailS3A5'), ('ALL5', 'ALL5', 'tailALL5')):
        mine = {**o60[an]['delta'], 'PP90_f1': o60[an]['fold_PP90'][1], 'PP90_f2': o60[an]['fold_PP90'][2]}
        rt, rbt, rdl = ref_t[rk]['tail'], ref_b['boot'][bk], ref_b['delta_vs_M3'][bk]
        chk[f'{an}_tail_match'] = all(tol(mine[k], rt[k]) for k in rt)
        chk[f'{an}_boot_match'] = all(tol(o60[an]['boot'][k][i], rbt[k][i]) for k in rbt for i in range(3)) and all(tol(o60[an]['boot'][k][3], rbt[k][3], 6e-4) for k in rbt)
        chk[f'{an}_full_delta_match'] = tol(mine['PP90'], rdl['PP90']) and tol(mine['PPcls'], rdl['PPcls']) and tol(o60[an]['delta_SCAR5'], rdl['SCAR5'])
    ok = (chk['fold_eq'] and chk['label_eq'] and chk['table_eq'] and chk['risk_max_abs'] <= 1e-9 and chk['risk_full_max_abs'] <= 1e-9 and chk['PPcls_eq_cls']
          and all(chk[f'{a}_{m}_match'] for a in ('P', 'ALL5') for m in ('tail', 'boot', 'full_delta')))
    chk['prefix60_readings'] = o60
res = {'registration_sha256': REG, 'gate_replication': gate, 'replication': rep, 'harness_check_prefix60': chk, 'harness_ok': bool(ok)}
print('HARNESS', ok, json.dumps(rd({k: v for k, v in chk.items() if k != 'prefix60_readings'})), flush=True)
if not ok:
    dump(res); sys.exit(print('ABORT: harness check failed, no registered reading made'))
res['surfaces'] = {p: readings(surface(p), p)[0] for p in ('prefix40', 'prefix80')}
S = res['surfaces']; res['verdict'] = {}
for an in ('P', 'E'):
    c1 = all(S[p][an]['delta']['PP90'] > 0 for p in S); c2 = all(S[p][an]['delta']['PPcls'] > 0 and S[p][an]['delta']['SCAR'] > 0 for p in S)
    c3 = all(v >= -.002 for p in S for v in S[p][an]['fold_PP90'].values()); c4 = all(S[p][an]['boot']['PP90'][3] >= .80 for p in S)
    res['verdict'][an] = {'C1': c1, 'C2': c2, 'C3': c3, 'C4': c4, 'PASS': c1 and c2 and c3 and c4}
dump(res)
print('VERDICT', json.dumps(res['verdict']), flush=True)
