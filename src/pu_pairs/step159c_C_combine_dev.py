"""Step 159c (Claude): parts S and K of outputs_step159_20260915/prediction_step159.json (sha a77dcb64), after the STEP157, STEP158 and
part N registered readings exist. Part S: the five-type family F (SXN / SXB / SXN3) and whether XGB3/ET/MLP stay (FINAL16), plus the
dev composite of FINAL16 - SXN16 on the STEP131 surfaces. Part K: the step 64f block testbed with SHORT3/ALL5 from F's 16 draws against
SX's 16 draws. Output outputs_step159_20260915/combination159.json. Development labels only; no evaluation labels."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, hashlib
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import testbed, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap as ap_b, logreg, predict
from pu_pairs.step62_multiview_candidate import L2, DEFS
from pu_pairs.build_allpairs_fold import fold_pools

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
REG = 'a77dcb643b129e50b956d4d95a889eb1c5c7b4080ab6dc0bae38011ee53a2252'
assert hashlib.sha256(open(C + 'outputs_step159_20260915/prediction_step159.json', 'rb').read()).hexdigest() == REG
DIRS = {'SX': (E + 'tail_keys_20260914/', 'sx_dev_fold'), 'SXN': (E + 'tail_keys_20260915_neg/', 'sxn_dev_fold'),
        'SXB': (E + 'bagged_views_20260915/', 'sxb_dev_fold'), 'SXN3': (E + 'tail_keys_20260915_neg3/', 'sxn3_dev_fold'),
        'ARCH': (E + 'arch_views_20260915/', 'arch_dev_neg1_fold')}
VT = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']; NEW_VT = ['XGB3', 'ET', 'MLP']
FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}
K = 16


def lg(x):
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(1, int(yy.sum())))


def permute(s, z, blk):
    s2 = s.copy(); idx = np.where(blk)[0]; s2[idx[np.argsort(-z[idx], kind='mergesort')]] = np.sort(s[idx])[::-1]; return s2


def bmap(sc, lab, fam, pred):
    return float(np.mean([ap(((lab == 1) & (fam == FAM[c])).astype(int), np.where(pred == c, sc, 0.0)) for c in (1, 2, 3)]))


def load(name, surface=None):
    folder, prefix = DIRS[name]
    d = pl.concat([pl.read_parquet(folder + f) for f in sorted(os.listdir(folder)) if f.startswith(prefix) and f.endswith('.parquet')]).filter(pl.col('k') < K)
    if surface: d = d.filter(pl.col('surface') == surface)
    return d


def type_means(name, vts):
    d = load(name); assert d.select('fold', 'k').unique().height == 2 * K
    d = d.with_columns([pl.Series(f'L_{v}', lg(d[f'r_{v}'])) for v in vts])
    return d.group_by('pair_id', 'fold', 'surface').agg([pl.col(f'L_{v}').mean().alias(f'{name}_{v}') for v in vts])


def rule(S, a, b):
    d90 = [S[p][a]['PP90'] - S[p][b]['PP90'] for p in S]; dcl = [S[p][a]['PPcls'] - S[p][b]['PPcls'] for p in S]
    r = {'mean_dPP90': float(np.mean(d90)), 'per_surface_dPP90': [round(x, 5) for x in d90], 'mean_dPPcls': float(np.mean(dcl)),
         'mean_comp': float(np.mean([.7 * (S[p][a]['PP90'] - S[p][b]['PP90']) + .1 * (S[p][a]['B'] - S[p][b]['B']) for p in S]))}
    r['passes'] = bool(np.mean(d90) > 0 and np.min(d90) >= -.0005 and np.mean(dcl) > 0)
    return r


def main():
    r157 = json.load(open(DIRS['ARCH'][0] + 'arch_dev_eval.json', encoding='utf-8'))['rule']
    r158 = json.load(open(DIRS['SXB'][0] + 'sxb_dev_eval.json', encoding='utf-8'))['rule']
    rN = json.load(open(DIRS['SXN3'][0] + 'sxn3_dev_eval.json', encoding='utf-8'))['rule']
    p157, p158, pN = r157['SA16_REPLACES_INC16'], r158['SXB16_REPLACES_SXN16'], rN['SXN3_16_REPLACES_SXN16']
    cand = {n: c for n, ok, c in (('SXB', p158, r158['mean_comp_SXB16_minus_SXN16']), ('SXN3', pN, rN['mean_comp_SXN3_16_minus_SXN16'])) if ok}
    F = max(cand, key=cand.get) if cand else 'SXN'
    out = {'registration_sha256': REG, 'STEP157_pass': p157, 'STEP158_pass': p158, 'partN_pass': pN, 'family_candidates_comp_vs_SXN16': cand, 'F': F}

    # Part S on the STEP131 tail surfaces
    V = type_means('SXN', VT)
    if F != 'SXN': V = V.join(type_means(F, VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    if p157: V = V.join(type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    keys = {'SXN16': [f'SXN_{v}' for v in VT], 'F16': [f'{F}_{v}' for v in VT]}
    if p157: keys['FINAL_candidate16'] = keys['F16'] + [f'ARCH_{v}' for v in NEW_VT]
    V = V.with_columns([pl.mean_horizontal(c).alias(k) for k, c in keys.items()])
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    S = {}
    for p in ('prefix40', 'prefix60', 'prefix80'):
        parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        d = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
             .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
        assert all(d[k].null_count() == 0 for k in keys)
        s, lab = d['risk'].to_numpy().astype(float), d['label'].to_numpy()
        fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
        yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
        yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
        T = s < .5
        S[p] = {}
        for k in keys:
            sc = permute(s, d[k].to_numpy(), T)
            S[p][k] = {'PP90': ap(yp, sc), 'PPcls': ap(yc, sc), 'RAW': ap(yr, sc), 'B': bmap(sc, lab, fam, pred)}
    out['tail_abs'] = S
    if p157 and F == 'SXN':
        include_new, out['FINAL_vs_F_rule'] = True, 'F = SXN: STEP157 SA16 rule decides (passed)'
    elif p157:
        rr = rule(S, 'FINAL_candidate16', 'F16'); include_new = rr['passes']; out['FINAL_vs_F_rule'] = rr
    else:
        include_new = False
    fin = 'FINAL_candidate16' if include_new else 'F16'
    out['include_new_types'] = include_new
    out['FINAL'] = {'five_type_family': F, 'new_types': NEW_VT if include_new else []}
    out['FINAL16_minus_SXN16'] = rule(S, fin, 'SXN16')
    tail_changes = (F != 'SXN') or include_new
    print('F', F, 'include_new', include_new, 'FINAL16-SXN16', json.dumps(out['FINAL16_minus_SXN16']), flush=True)

    # Part K: block testbed
    SXd = load('SX', 'prefix60')
    k0 = SXd.filter(pl.col('k') == 0).select('pair_id', 'fold', pl.col('r_SHORT3').alias('k0_S3'), pl.col('r_ALL5').alias('k0_A5'))
    agg = {}
    for name in ('SX', F):
        dd = SXd if name == 'SX' else load(name, 'prefix60')
        assert dd.select('fold', 'k').unique().height == 2 * K
        agg[name] = (dd.with_columns(pl.Series('L_S3', lg(dd['r_SHORT3'])), pl.Series('L_A5', lg(dd['r_ALL5']))).group_by('pair_id', 'fold')
                     .agg(pl.col('L_S3').mean().alias(f'{name}_S3'), pl.col('L_A5').mean().alias(f'{name}_A5')))
    d = testbed()
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2
    blk = {}
    for dn, (posset, negset) in DEFS.items():
        c = d.filter(pl.col('cls').is_in(posset + negset))
        j = c.select('pair_id', 'r_SHORT3', 'r_ALL5').join(k0, on='pair_id', how='left')
        j = j.with_columns(((pl.col('k0_S3') - pl.col('r_SHORT3')).abs() + (pl.col('k0_A5') - pl.col('r_ALL5')).abs()).alias('dif')).sort('pair_id', 'dif').unique(subset=['pair_id'], keep='first', maintain_order=True)
        gate = float(j['dif'].max()); assert gate == 0.0, gate
        c = c.join(j.select('pair_id', pl.col('fold').alias('mfold')), on='pair_id', how='left')
        for name in agg:
            c = c.join(agg[name].rename({'fold': 'mfold'}), on=['pair_id', 'mfold'], how='left')
        y = c['cls'].is_in(posset).to_numpy().astype(float); tab = c['table_id'].to_numpy()
        in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab])
        insh = np.array([t in shared for t in tab])

        def cf(c2):
            Xm = X_of(c2, FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT'); pr = np.full(len(y), np.nan)
            m1 = logreg(Xm[in1], y[in1], L2); pr[in2 | insh] = predict(m1, Xm[in2 | insh]); m2 = logreg(Xm[in2], y[in2], L2); pr[in1] = predict(m2, Xm[in1])
            return {'pooled': ap_b(y, pr), 'test_f2sh': ap_b(y[in2 | insh], pr[in2 | insh]), 'test_f1': ap_b(y[in1], pr[in1])}
        r = {'n': len(y), 'n_pos': int(y.sum()), 'gate_k0': gate}
        for name in agg:
            assert c[f'{name}_S3'].null_count() == 0
            r[f'M3x16_{name}'] = cf(c.with_columns(pl.Series('r_SHORT3', sig(c[f'{name}_S3'])), pl.Series('r_ALL5', sig(c[f'{name}_A5']))))
        if F != 'SX':
            r['dAP_F_minus_SX'] = {kk: r[f'M3x16_{F}'][kk] - r['M3x16_SX'][kk] for kk in ('pooled', 'test_f2sh', 'test_f1')}
        blk[dn] = r
        print('block', dn, json.dumps({k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
    out['block'] = blk
    dA, dC = blk['A']['dAP_F_minus_SX'], blk['C']['dAP_F_minus_SX']
    kpass = bool(np.mean([dA['pooled'], dC['pooled']]) > 0 and dA['test_f2sh'] >= -.001 and dA['test_f1'] >= -.001)
    out['partK'] = {'mean_pooled_dAP_A_C': float(np.mean([dA['pooled'], dC['pooled']])), 'defA_test_cells': [dA['test_f2sh'], dA['test_f1']], 'passes': kpass}
    tail_part = .37 * out['FINAL16_minus_SXN16']['mean_comp'] if tail_changes else 0.0
    block_part = .7 * .9 * out['partK']['mean_pooled_dAP_A_C'] if kpass else 0.0
    out['build_file'] = bool(tail_changes or kpass)
    out['prediction_vs_90472'] = {'point': .65 * (tail_part + block_part), 'tail_part_unscaled': tail_part, 'block_part_unscaled': block_part,
                                  'band80_halfwidth': float(np.sqrt((.0025 if tail_changes else 0) ** 2 + (.0015 if kpass else 0) ** 2))}
    json.dump(out, open(C + 'outputs_step159_20260915/combination159.json', 'w', encoding='utf-8'), indent=1)
    print('PARTK', json.dumps(out['partK']), 'BUILD', out['build_file'], 'PRED', json.dumps(out['prediction_vs_90472']))


if __name__ == '__main__':
    main()
