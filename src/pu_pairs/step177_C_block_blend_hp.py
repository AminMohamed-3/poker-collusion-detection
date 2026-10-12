"""Step 177 (Claude): H3 of the user's hyperparameter search, the block blend. Production orders the 909 block pairs by M3x: a logistic blend
(step60 logreg, Newton, standardised inputs) of FS_M3 with SHORT3/ALL5 as 32-draw logit means, l2 = 10, fitted on def A certified rows of
the step 64f prefix-60 testbed. l2, the training-row definition and the noT feature were fixed at step 60-64 and never searched.
Grid (20 cells): l2 in {1, 3, 10, 30, 100} x rows in {A, C} x features in {FS_M3, FS_M3 + noT}. Incumbent l2_10_A_M3 (reproduces
block_screen.json M3x32 def A exactly). Cross-fold as step 64f: f1-only tables are predicted by the fit on f2-only rows, f2-only and shared
tables by the fit on f1-only rows. Score of a group = def A AP on the group's def A certified rows (def A counts undisclosed high-risk
pairs as positives, as PP90 does); def C AP is a guard only (a C-trained cell would otherwise gain from matching its own definition).
Honest selection: pick on one group, score on the other; G = held deltas weighted by the held group's def A positives. Final = best pooled
def A AP. Proceed iff G > 0, final != incumbent, pooled def A delta > 0 and every def x group delta (A and C) >= -.002.
Gates: l2_10_A_M3 def A and l2_10_C_M3 def C replay block_screen.json M3x32 exactly.
Output experiments/block_blend_hp_20260915/results177.json. Usage: REGSHA"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib, time
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs.step64f_M3_candidate import testbed, FS_M3
from pu_pairs.step63_softR1_candidate import X_of
from pu_pairs.step60_multiview_blend import ap, logreg, predict
from pu_pairs.step62_multiview_candidate import DEFS
from pu_pairs.build_allpairs_fold import fold_pools
from pu_pairs.step156a_C_block_M3x import dev_views32

E = C + 'experiments/'
REG = C + 'outputs_step177_20260915/prediction_step177.json'
DEST = E + 'block_blend_hp_20260915/results177.json'
L2S, ROWS, FEATS = (1., 3., 10., 30., 100.), ('A', 'C'), {'M3': FS_M3, 'M3noT': FS_M3 + ['noT']}
INC = 'l2_10_A_M3'


def cells():
    out = [(f'l2_{int(l)}_{r}_{f}', l, r, f) for l in L2S for r in ROWS for f in FEATS]
    assert len(out) == 20 and INC in [c[0] for c in out]
    return out


def main():
    t0 = time.time()
    reg_sha = sys.argv[1]
    assert hashlib.sha256(open(REG, 'rb').read()).hexdigest() == reg_sha
    reg = json.load(open(REG, encoding='utf-8'))
    assert hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest() == reg['runner_sha256']
    os.makedirs(os.path.dirname(DEST), exist_ok=True); assert not os.path.exists(DEST)
    d, gate_fold = dev_views32(testbed()); assert gate_fold == 0.0
    f1, f2 = set(fold_pools(1)), set(fold_pools(2)); shared = f1 & f2; tab = d['table_id'].to_numpy()
    in1 = np.array([t in f1 and t not in shared for t in tab]); in2 = np.array([t in f2 and t not in shared for t in tab]); insh = np.array([t in shared for t in tab])
    assert (in1 | in2 | insh).all()
    G_ = {'f1': in1, 'f2sh': in2 | insh}
    cert = {D: d['cls'].is_in(DEFS[D][0] + DEFS[D][1]).to_numpy() for D in ROWS}
    ypos = {D: d['cls'].is_in(DEFS[D][0]).to_numpy().astype(float) for D in ROWS}
    Xs = {f: X_of(d, fs, 'risk_prefix', 'R1_prefix', 'risk_noT') for f, fs in FEATS.items()}
    res = {}
    for name, l2, r, f in cells():
        X = Xs[f]; tr = cert[r]; pred = np.full(d.height, np.nan)
        m1 = logreg(X[in1 & tr], ypos[r][in1 & tr], l2); pred[in2 | insh] = predict(m1, X[in2 | insh])
        m2 = logreg(X[in2 & tr], ypos[r][in2 & tr], l2); pred[in1] = predict(m2, X[in1])
        R = {}
        for D in ROWS:
            R[f'{D}_pooled'] = ap(ypos[D][cert[D]], pred[cert[D]])
            for g, m in G_.items():
                R[f'{D}_{g}'] = ap(ypos[D][cert[D] & m], pred[cert[D] & m])
        R['score_pooled'] = R['A_pooled']
        for g in G_:
            R[f'score_{g}'] = R[f'A_{g}']
        res[name] = R
    ref = json.load(open(C + 'outputs_step156_20260915/block_screen.json', encoding='utf-8'))
    for D, cell in (('A', INC), ('C', 'l2_10_C_M3')):
        assert res[cell][f'{D}_pooled'] == ref[D]['M3x32']['pooled'] and res[cell][f'{D}_f2sh'] == ref[D]['M3x32']['test_f2sh'] and res[cell][f'{D}_f1'] == ref[D]['M3x32']['test_f1']
    ids = [c[0] for c in cells()]
    npos = {g: int(ypos['A'][cert['A'] & m].sum()) for g, m in G_.items()}
    rd = lambda x: round(float(x), 6)
    held = {}
    for on, off in (('f1', 'f2sh'), ('f2sh', 'f1')):
        pick = max(ids, key=lambda c: (res[c][f'score_{on}'], c == INC, -ids.index(c)))
        held[on] = {'cell': pick, 'selection_delta': rd(res[pick][f'score_{on}'] - res[INC][f'score_{on}']), 'held_group': off, 'held_delta': rd(res[pick][f'score_{off}'] - res[INC][f'score_{off}'])}
    G = (held['f1']['held_delta'] * npos['f2sh'] + held['f2sh']['held_delta'] * npos['f1']) / (npos['f1'] + npos['f2sh'])
    fin = max(ids, key=lambda c: (res[c]['score_pooled'], c == INC, -ids.index(c)))
    fv = {k: rd(res[fin][k] - res[INC][k]) for k in res[INC]}
    proceed = bool(G > 0 and fin != INC and fv['score_pooled'] > 0 and min(fv[f'{D}_{g}'] for D in ROWS for g in G_) >= -.002)
    out = {'registration_sha256': reg_sha, 'gate_fold_match': gate_fold, 'gate_incumbent_replays_block_screen': True, 'def_A_positives_by_group': npos,
           'incumbent': {k: rd(v) for k, v in res[INC].items()}, 'cross_group': held, 'G_honest': rd(G), 'final_cell': fin, 'final_vs_incumbent': fv, 'proceed': proceed,
           'grid_delta_vs_incumbent': {c: {k: rd(v - res[INC][k]) for k, v in res[c].items()} for c in ids}, 'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(DEST, 'w', encoding='utf-8'), indent=1)
    print('G', out['G_honest'], json.dumps(held), 'final', fin, json.dumps({k: fv[k] for k in ('score_pooled', 'A_pooled', 'C_pooled', 'score_f1', 'score_f2sh')}), 'PROCEED', proceed, flush=True)


if __name__ == '__main__':
    main()
