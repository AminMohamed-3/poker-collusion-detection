"""Step 191 (Claude): average the STEP177 block-blend pool instead of selecting from it.

STEP177 selected among these same 20 cells and failed honestly (G -0.002135). STEP185 and
STEP190 apply one operation to that class of failure: average, do not pick. Both of those were
on the evidence axis (weight 0.2). This is the first on the PAIR axis (weight 0.7).

Everything about the surface is STEP177's and unchanged: the same 20 cells, the same cross-fold
structure (f1 tables predicted by the f2 fit and vice versa), the same def A AP metric on the
step64f prefix-60 certified testbed, the same cross-group selection weighted by def A positives.
The single-cell deltas are gated to reproduce results177.json exactly, so the only new thing
here is the ensembling.

Usage: REGSHA
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib, time
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np
from pu_pairs.step64f_M3_candidate import testbed, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap, logreg, predict
from pu_pairs.step62_multiview_candidate import DEFS
from pu_pairs.build_allpairs_fold import fold_pools
from pu_pairs.step156a_C_block_M3x import dev_views32

E = C + 'experiments/'
REG = E + 'block_blend_hp_20260915/registration191.json'
REF177 = E + 'block_blend_hp_20260915/results177.json'
DEST = E + 'block_blend_hp_20260915/results191.json'
L2S, ROWS, FEATS = (1., 3., 10., 30., 100.), ('A', 'C'), {'M3': FS_M3, 'M3noT': FS_M3 + ['noT']}
INC = 'l2_10_A_M3'
COMBINERS = ('MEAN', 'ZMEAN', 'RANKMEAN')


def cells():
    out = [(f'l2_{int(l)}_{r}_{f}', l, r, f) for l in L2S for r in ROWS for f in FEATS]
    assert len(out) == 20 and INC in [c[0] for c in out]
    return out


def combine(name, mat):
    """mat: cells x rows, no NaN. Returns one vector whose ORDER is the ensemble's."""
    if name == 'MEAN':
        return mat.mean(0)
    if name == 'ZMEAN':
        sd = mat.std(axis=1, keepdims=True)
        return np.where(sd > 0, (mat - mat.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0), 0.0).mean(0)
    if name == 'RANKMEAN':
        r = np.empty_like(mat)
        for i in range(mat.shape[0]):
            o = np.argsort(mat[i], kind='mergesort')
            r[i, o] = np.arange(mat.shape[1], dtype=float)
        return r.mean(0)
    raise ValueError(name)


def main():
    t0 = time.time()
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(REG, 'rb').read()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not os.path.exists(DEST), 'preserve a completed read'
    d, gate_fold = dev_views32(testbed())
    assert gate_fold == 0.0, gate_fold
    f1, f2 = set(fold_pools(1)), set(fold_pools(2))
    shared = f1 & f2
    tab = d['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab])
    in2 = np.array([t in f2 and t not in shared for t in tab])
    insh = np.array([t in shared for t in tab])
    assert (in1 | in2 | insh).all()
    G_ = {'f1': in1, 'f2sh': in2 | insh}
    cert = {D: d['cls'].is_in(DEFS[D][0] + DEFS[D][1]).to_numpy() for D in ROWS}
    ypos = {D: d['cls'].is_in(DEFS[D][0]).to_numpy().astype(float) for D in ROWS}
    Xs = {f: X_of(d, fs, 'risk_prefix', 'R1_prefix', 'risk_noT') for f, fs in FEATS.items()}

    PRED, res = {}, {}

    def score(pred):
        R = {}
        for D in ROWS:
            R[f'{D}_pooled'] = ap(ypos[D][cert[D]], pred[cert[D]])
            for g, m in G_.items():
                R[f'{D}_{g}'] = ap(ypos[D][cert[D] & m], pred[cert[D] & m])
        R['score_pooled'] = R['A_pooled']
        for g in G_:
            R[f'score_{g}'] = R[f'A_{g}']
        return R

    for name, l2, r, f in cells():
        X = Xs[f]; tr = cert[r]; pred = np.full(d.height, np.nan)
        m1 = logreg(X[in1 & tr], ypos[r][in1 & tr], l2); pred[in2 | insh] = predict(m1, X[in2 | insh])
        m2 = logreg(X[in2 & tr], ypos[r][in2 & tr], l2); pred[in1] = predict(m2, X[in1])
        assert np.isfinite(pred).all(), name
        PRED[name] = pred
        res[name] = score(pred)

    # ---- gates ----
    ref = json.load(open(C + 'outputs_step156_20260915/block_screen.json', encoding='utf-8'))
    for D, cell in (('A', INC), ('C', 'l2_10_C_M3')):
        assert res[cell][f'{D}_pooled'] == ref[D]['M3x32']['pooled'] \
            and res[cell][f'{D}_f2sh'] == ref[D]['M3x32']['test_f2sh'] \
            and res[cell][f'{D}_f1'] == ref[D]['M3x32']['test_f1'], (D, cell)
    rd = lambda x: round(float(x), 6)
    r177 = json.load(open(REF177, encoding='utf-8'))['grid_delta_vs_incumbent']
    worst = 0.0
    for c in r177:
        for k, v in r177[c].items():
            worst = max(worst, abs(rd(res[c][k] - res[INC][k]) - v))
    print(f'GATE vs results177 grid: worst abs {worst}', flush=True)
    assert worst == 0.0, 'STEP177 replay is not exact'

    ids = [c[0] for c in cells()]
    POOLS = {
        'ALL20': ids,
        'L2_5_A_M3': [f'l2_{int(l)}_A_M3' for l in L2S],
        'ROWS_A_10': [c for c in ids if '_A_' in c],
        'FEAT_M3_10': [c for c in ids if c.endswith('_M3')],
        'NOT_INC_19': [c for c in ids if c != INC],
    }
    arms = {}
    for pn, cs in POOLS.items():
        mat = np.array([PRED[c] for c in cs])
        for cb in COMBINERS:
            nm = f'{pn}|{cb}'
            res[nm] = score(combine(cb, mat))
            arms[nm] = {'pool': pn, 'combiner': cb, 'n_cells': len(cs),
                        **{k: rd(res[nm][k] - res[INC][k]) for k in ('A_pooled', 'A_f1', 'A_f2sh', 'C_pooled', 'C_f1', 'C_f2sh')}}
            print(f'{nm:20s} n={len(cs):3d} A_pooled {arms[nm]["A_pooled"]:+.6f} '
                  f'f1 {arms[nm]["A_f1"]:+.6f} f2sh {arms[nm]["A_f2sh"]:+.6f} '
                  f'C_pooled {arms[nm]["C_pooled"]:+.6f}', flush=True)

    npos = {g: int(ypos['A'][cert['A'] & m].sum()) for g, m in G_.items()}
    cand = list(arms) + [INC]
    held = {}
    for on, off in (('f1', 'f2sh'), ('f2sh', 'f1')):
        pick = max(cand, key=lambda c: (res[c][f'score_{on}'], c == INC, -cand.index(c)))
        held[on] = {'arm': pick, 'selection_delta': rd(res[pick][f'score_{on}'] - res[INC][f'score_{on}']),
                    'held_group': off, 'held_delta': rd(res[pick][f'score_{off}'] - res[INC][f'score_{off}'])}
        print(f'cross-group {on}: picked {pick} sel {held[on]["selection_delta"]:+.6f} '
              f'held {held[on]["held_delta"]:+.6f}', flush=True)
    G = (held['f1']['held_delta'] * npos['f2sh'] + held['f2sh']['held_delta'] * npos['f1']) / (npos['f1'] + npos['f2sh'])

    fin = max(arms, key=lambda c: (res[c]['score_pooled'], -list(arms).index(c)))
    fv = {k: rd(res[fin][k] - res[INC][k]) for k in res[INC]}
    build = bool(G > 0 and fv['A_f1'] > 0 and fv['A_f2sh'] > 0 and fv['score_pooled'] > 0
                 and min(fv[f'{D}_{g}'] for D in ROWS for g in G_) >= -.002)
    print(f'G {rd(G):+.6f}  best arm {fin}  build_rule_met {build}', flush=True)

    out = {'step': 191, 'registration_sha256': reg_sha, 'gate_fold_match': gate_fold,
           'gate_results177_replay_worst_abs': worst, 'def_A_positives_by_group': npos,
           'incumbent': {k: rd(v) for k, v in res[INC].items()},
           'single_cells': {c: {k: rd(res[c][k] - res[INC][k]) for k in res[INC]} for c in ids},
           'arms': arms, 'cross_group': held, 'G_honest': rd(G), 'best_arm': fin,
           'best_vs_incumbent': fv, 'build_rule_met': build, 'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(DEST, 'w', encoding='utf-8'), indent=1)
    print('wrote', DEST, flush=True)


if __name__ == '__main__':
    main()
