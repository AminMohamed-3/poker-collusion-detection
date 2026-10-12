"""Step 187 (Claude): the STEP185 question asked of the PAIR tail key, which carries .7 of the composite instead of .2.
STEP185 found that STEP174 threw away +.0074 of decoded evidence MAP by SELECTING one network architecture out of twelve instead of
AVERAGING them, and that its worst arm (N07, -.00469) sat inside its best composition. The production pair key was assembled the same way,
and two places in it are selections rather than averages: SXN3_FAMMAX is a MAX over the DT, SP and CI family views -- a max is a selection
operator, and those three views are cached but never enter the mean on their own -- and ARCH2_LGBB3 was dropped by STEP163's forward
selection at -.000292 dPP90.
Arms widen the type set; the block (risk >= .5) keeps its production order throughout and only the tail key changes, exactly step183 at
w = 0. CTRL_NOISE (a seeded normal column) must not help and CTRL_DROP (production minus its strongest added type) must hurt, or the
surface is not measuring what it claims to. PRIMARY METRIC IS RAW, labelled positives only: PP90/PPcls pseudo-positives come from
risk_full >= .9 and so reward agreement with the incumbent, which is the bias step183 was written to dodge and STEP180 was burned by.
SXB arms are DEV-ONLY -- that pool has no evaluation-side parquet, so a win there is information, not a shippable file.
Development labels only, no uploads. Usage: step187_C_key_widening.py REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step159c_C_combine_dev as S159

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
OUT = E + 'key_widening_20260916/'
REG = OUT + 'registration187.json'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')   # as step163b / step183
SURFACES = ('prefix40', 'prefix60', 'prefix80')
SXN3_ALL = ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX', 'DT', 'SP', 'CI']
ARCH_ALL = ['XGB3', 'ET', 'MLP']
ARCH2_ALL = ['MLP2', 'HGB5', 'LGBB3']
P = lambda pre, xs: [f'{pre}_{x}' for x in xs]
PROD = P('SXN3', ['SHORT3', 'ALL5', 'LIN30', 'LIN300', 'FAMMAX']) + P('ARCH', ARCH_ALL) + P('ARCH2', ['MLP2', 'HGB5'])
FAM3 = P('SXN3', ['DT', 'SP', 'CI'])
ARMS = {
    'PROD10': PROD,
    'P_FAM3': PROD + FAM3,
    'P_LGBB3': PROD + ['ARCH2_LGBB3'],
    'P_FAM3_LGBB3': PROD + FAM3 + ['ARCH2_LGBB3'],
    'SWAP_FAMMAX': [c for c in PROD if c != 'SXN3_FAMMAX'] + FAM3,
    'ALL14': P('SXN3', SXN3_ALL) + P('ARCH', ARCH_ALL) + P('ARCH2', ARCH2_ALL),
    'CTRL_NOISE': PROD + ['NOISE'],
    'CTRL_DROP': [c for c in PROD if c != 'ARCH2_MLP2'],
}
SXB_ARMS = {'PROD10_SXB': PROD + P('SXB', SXN3_ALL), 'ALL14_SXB': P('SXN3', SXN3_ALL) + P('ARCH', ARCH_ALL) + P('ARCH2', ARCH2_ALL) + P('SXB', SXN3_ALL)}


def zrank(x):
    r = np.empty(len(x), float)
    order = np.argsort(x, kind='mergesort')
    r[order] = np.arange(len(x), dtype=float)
    return (r - r.mean()) / r.std()


def reassign(s, blk_key, tail_key, blk):
    """Production form: the block keeps its value multiset re-assigned in blk_key order, the tail in tail_key order."""
    out = np.empty_like(s)
    for mask, key in ((blk, blk_key), (~blk, tail_key)):
        idx = np.flatnonzero(mask)
        order = idx[np.lexsort((idx, -key[idx]))]
        out[order] = np.sort(s[idx])[::-1]
    return out


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[1]
    from evidence_aug.step165_C_e2e_audit import sha
    from evidence_aug.paths import check_protected
    from pathlib import Path
    assert sha(Path(REG)) == reg_sha, 'registration hash mismatch'
    reg = json.loads(Path(REG).read_text(encoding='utf-8'))
    assert reg['production_key']['columns'] == PROD
    dest = Path(OUT + 'results187.json')
    assert not dest.exists(), 'preserve a completed read'
    prot = check_protected()

    V = (S159.type_means('SXN3', SXN3_ALL)
         .join(S159.type_means('ARCH', ARCH_ALL), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
         .join(S159.type_means('ARCH2', ARCH2_ALL), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1'))
    arms = dict(ARMS)
    sxb_note = 'not attempted'
    try:
        V = V.join(S159.type_means('SXB', SXN3_ALL), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
        arms.update(SXB_ARMS)
        sxb_note = 'joined; SXB arms are DEV-ONLY (no evaluation-side parquet exists for this pool)'
    except Exception as e:
        sxb_note = f'unavailable: {type(e).__name__}: {e}'
    print('SXB', sxb_note, flush=True)

    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    D = {}
    for p in SURFACES:
        parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        D[p] = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
                .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
        D[p] = D[p].with_columns(pl.Series('NOISE', np.random.default_rng(1870 + SURFACES.index(p)).standard_normal(D[p].height)))

    def measure(cols):
        S = {}
        for p, d in D.items():
            key = d.select(pl.mean_horizontal(cols)).to_series().to_numpy()
            assert np.isfinite(key).all(), cols
            s, lab = d['risk'].to_numpy().astype(float), d['label'].to_numpy()
            fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
            yr, unl = (lab == 1).astype(int), lab == -1
            rf = d['risk_full'].fill_null(0).to_numpy().astype(float); r1 = d['R1_full'].fill_null(False).to_numpy()
            yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int)
            yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
            blk = s >= .5
            sc = reassign(s, zrank(s), zrank(key), blk)
            S[p] = {'PP90': S159.ap(yp, sc), 'PPcls': S159.ap(yc, sc), 'RAW': S159.ap(yr, sc), 'B': S159.bmap(sc, lab, fam, pred),
                    'n': int(len(s)), 'raw_pos': int(yr.sum())}
        return S

    R = lambda x: round(float(x), 6)
    base = measure(PROD)
    ref = json.load(open(E + 'length_extrapolation_20260916/results183.json', encoding='utf-8'))['production_abs']
    gate = {p: {m: R(base[p][m] - ref[p][m]) for m in ('RAW', 'PP90', 'PPcls', 'B')} for p in SURFACES}
    gate_ok = all(abs(gate[p][m]) < 1e-9 for p in SURFACES for m in ('RAW', 'PP90'))
    print('GATE vs step183 production_abs', json.dumps(gate), 'ok', gate_ok, flush=True)

    res = {}
    for name, cols in arms.items():
        cur = measure(cols)
        d_raw = [cur[p]['RAW'] - base[p]['RAW'] for p in SURFACES]
        d_90 = [cur[p]['PP90'] - base[p]['PP90'] for p in SURFACES]
        res[name] = {'n_types': len(cols), 'mean_dRAW': R(np.mean(d_raw)), 'per_surface_dRAW': [R(x) for x in d_raw],
                     'mean_dPP90': R(np.mean(d_90)), 'per_surface_dPP90': [R(x) for x in d_90],
                     'mean_dPPcls': R(np.mean([cur[p]['PPcls'] - base[p]['PPcls'] for p in SURFACES])),
                     'mean_dB': R(np.mean([cur[p]['B'] - base[p]['B'] for p in SURFACES])),
                     'dev_only': bool(name in SXB_ARMS)}
        print(f'{name:14s} n={len(cols):2d} dRAW {np.mean(d_raw):+.6f} per {[round(x, 5) for x in d_raw]} '
              f'dPP90 {np.mean(d_90):+.6f}  {round(time.perf_counter() - t0)}s', flush=True)

    noise = res['CTRL_NOISE']['mean_dRAW']
    cands = {k: v for k, v in res.items() if k not in ('PROD10', 'CTRL_NOISE', 'CTRL_DROP')}
    fin = max(cands, key=lambda k: cands[k]['mean_dRAW'])
    g = cands[fin]
    passes = bool(g['mean_dRAW'] > 0 and min(g['per_surface_dRAW']) >= -.0005 and g['mean_dPP90'] >= 0 and g['mean_dRAW'] > noise)
    build = bool(passes and g['mean_dRAW'] >= .003 and not g['dev_only'])
    out = {'registration_sha256': reg_sha, 'author': 'claude',
           'gates': {'PROD10_vs_step183_production_abs': gate, 'gate_ok': gate_ok, 'SXB': sxb_note,
                     'CTRL_NOISE_mean_dRAW': noise, 'CTRL_DROP_mean_dRAW': res['CTRL_DROP']['mean_dRAW'],
                     'controls_behave': bool(noise <= 0.0005 and res['CTRL_DROP']['mean_dRAW'] < 0)},
           'production_abs': {p: {m: R(base[p][m]) for m in ('PP90', 'PPcls', 'RAW', 'B')} for p in SURFACES},
           'population': {p: {k: base[p][k] for k in ('n', 'raw_pos')} for p in SURFACES},
           'arms': dict(sorted(res.items(), key=lambda kv: -kv[1]['mean_dRAW'])),
           'final_arm': fin, 'final': g,
           'rule': 'mean dRAW > 0 AND every surface dRAW >= -.0005 AND mean dPP90 >= 0 AND beats CTRL_NOISE',
           'passes': passes, 'build_bar': 'mean dRAW >= +.003 and not dev-only', 'build_justified': build,
           'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    assert gate_ok, 'PROD10 must reproduce step183 production_abs'
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('FINAL', fin, json.dumps(g), 'PASSES', passes, 'BUILD', build, flush=True)


if __name__ == '__main__':
    main()
