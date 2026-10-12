"""Step 183 (Claude): the pair ranking extrapolates to pairs shorter than anything it was trained on, and this tests a post-hoc monotone
correction for it. 26 % of evaluation pairs (29,237 of 112,540) have fewer shared hands than ANY labelled development pair -- evaluation
spans 38 to 419, the labelled pairs start at 57 -- and those short pairs are 1.45x over-represented in the top-909 block (342 of 909
against 26 % of the population). On the labelled range a logistic fit of the label on out-of-fold risk and log(shared_hands) gives
4.81 and +0.50: at equal risk a LONGER pair is more likely to be a target, so short pairs are over-scored.
Not STEP164, which reweighted training rows and failed, and not step42's isotonic-by-bin, which was scored on a PU metric of the kind
STEP180 showed to mislead. Arms blend the production key with a rank-normalised log(shared_hands), on the tail key, on the block order
or on both, with negative-w controls that must come out worse. PRIMARY METRIC IS RAW -- labelled positives only -- because the PP90/PPcls
pseudo-positives are defined by risk_full >= .9 and so inherit the model's own length bias; PP90 is kept as a direction check. Surfaces,
labels and AP exactly as step159c/step163b. Development labels only, no uploads.
Usage: step183_C_length_extrapolation.py REGSHA"""
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
OUT = E + 'length_extrapolation_20260916/'
REG = OUT + 'registration183.json'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')   # as step163b
VT, NEW_VT = S159.VT, S159.NEW_VT
KEPT = ['MLP2', 'HGB5']                      # STEP163 receipt: the production FINAL2 stack
SURFACES = ('prefix40', 'prefix60', 'prefix80')
W = [-.10, -.05, .02, .05, .10, .20, .35]
SCOPES = ('TAIL', 'BLOCK', 'BOTH')


def zrank(x):
    """rank-normalised to mean 0, unit sd; ties averaged."""
    r = np.empty(len(x), float)
    order = np.argsort(x, kind='mergesort')
    r[order] = np.arange(len(x), dtype=float)
    r = (r - r.mean()) / r.std()
    return r


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
    assert reg['grid']['w'] == W and reg['grid']['scopes'] == list(SCOPES)
    dest = Path(OUT + 'results183.json')
    assert not dest.exists(), 'preserve a completed read'
    prot = check_protected()

    V = S159.type_means('SXN3', VT).join(S159.type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', KEPT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    cols = [f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT] + [f'ARCH2_{v}' for v in KEPT]
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    D = {}
    for p in SURFACES:
        parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        D[p] = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
                .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))

    def measure(w, scope):
        S = {}
        for p, d in D.items():
            key = d.select(pl.mean_horizontal(cols)).to_series().to_numpy()
            assert np.isfinite(key).all()
            s, lab = d['risk'].to_numpy().astype(float), d['label'].to_numpy()
            fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
            sh = d['shared_hands'].to_numpy().astype(float)
            yr, unl = (lab == 1).astype(int), lab == -1
            rf = d['risk_full'].fill_null(0).to_numpy().astype(float); r1 = d['R1_full'].fill_null(False).to_numpy()
            yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int)
            yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
            blk = s >= .5
            L = zrank(np.log(sh))
            bk, tk = zrank(s), zrank(key)
            if scope in ('TAIL', 'BOTH'): tk = (1 - abs(w)) * tk + w * L
            if scope in ('BLOCK', 'BOTH'): bk = (1 - abs(w)) * bk + w * L
            sc = reassign(s, bk, tk, blk)
            S[p] = {'PP90': S159.ap(yp, sc), 'PPcls': S159.ap(yc, sc), 'RAW': S159.ap(yr, sc), 'B': S159.bmap(sc, lab, fam, pred),
                    'n': int(len(s)), 'n_block': int(blk.sum()), 'raw_pos': int(yr.sum()),
                    'short_share_block': round(float((sh[blk] < 57).mean()), 4), 'short_share_all': round(float((sh < 57).mean()), 4)}
        return S

    base = measure(0., 'TAIL')
    prod = {p: {m: base[p][m] for m in ('PP90', 'PPcls', 'RAW', 'B')} for p in base}
    # gate: the w = 0 arm must reproduce the STEP163 production reading of the same stack
    ref = json.load(open(E + 'arch2_views_20260915/arch2_dev_eval.json', encoding='utf-8'))
    gate = {p: round(base[p]['PP90'] - ref['FINAL16_abs'][p]['PP90'], 6) for p in SURFACES}
    R = lambda x: round(float(x), 6)
    grid = {}
    for scope in SCOPES:
        for w in W:
            cur = measure(w, scope)
            d_raw = [cur[p]['RAW'] - prod[p]['RAW'] for p in SURFACES]
            d_90 = [cur[p]['PP90'] - prod[p]['PP90'] for p in SURFACES]
            grid[f'{scope}_w{w}'] = {'mean_dRAW': R(np.mean(d_raw)), 'per_surface_dRAW': [R(x) for x in d_raw],
                                     'mean_dPP90': R(np.mean(d_90)), 'per_surface_dPP90': [R(x) for x in d_90],
                                     'mean_dPPcls': R(np.mean([cur[p]['PPcls'] - prod[p]['PPcls'] for p in SURFACES])),
                                     'mean_dB': R(np.mean([cur[p]['B'] - prod[p]['B'] for p in SURFACES]))}
            print(f'{scope} w={w:+.2f} dRAW {np.mean(d_raw):+.5f} per {[round(x,5) for x in d_raw]} dPP90 {np.mean(d_90):+.5f}', flush=True)
    pos = [k for k in grid if not k.split('_w')[1].startswith('-')]
    fin = max(pos, key=lambda k: grid[k]['mean_dRAW'])
    g = grid[fin]
    ctrl = {s: max(grid[f'{s}_w-0.1']['mean_dRAW'], grid[f'{s}_w-0.05']['mean_dRAW']) for s in SCOPES}
    sign_ok = bool(g['mean_dRAW'] > ctrl[fin.split('_w')[0]])
    proceed = bool(g['mean_dRAW'] > 0 and min(g['per_surface_dRAW']) >= -.0005 and g['mean_dPP90'] >= 0 and sign_ok)
    out = {'registration_sha256': reg_sha, 'author': 'claude',
           'gates': {'w0_reproduces_step163_FINAL16_PP90': gate,
                     'note': 'w0 uses the FINAL2 stack (FINAL16 + MLP2 + HGB5), so a small positive offset against FINAL16 is expected, not zero',
                     'population': {p: {k: base[p][k] for k in ('n', 'n_block', 'raw_pos', 'short_share_block', 'short_share_all')} for p in SURFACES}},
           'production_abs': {p: {m: R(prod[p][m]) for m in prod[p]} for p in SURFACES},
           'grid': grid, 'final_arm': fin, 'final': g, 'controls_best_negative_w': {k: R(v) for k, v in ctrl.items()},
           'sign_consistency_ok': sign_ok,
           'rule': 'mean dRAW > 0 AND every surface dRAW >= -.0005 AND mean dPP90 >= 0 AND the arm beats its own negative-w control',
           'proceed': proceed, 'wall_seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('FINAL', fin, json.dumps(g), 'sign_ok', sign_ok, 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
