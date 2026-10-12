"""STEP246 post-hoc diagnostics.  Changes no decision; the registered read is results_step246.json.

(A) Why results_step246.json says VOID.  Its only failing gate is `S_star_v1_replay_exact`
    (6.98e-10 instead of exactly 0.0).  step221/step235 accumulate the S* view scores in TWO slot
    accumulators (`accS = [zeros, zeros]`, one for the view maker and one for the pad maker) and add
    them at the end; the STEP246 runner used ONE accumulator so that view averaging could be carried
    per (member, family, list) instead of per arm.  The two orders differ only in floating-point
    association.  This script recomputes the same 40 S* view scores and reports both orders against
    D['V1'], which settles whether the gate failure is an artifact of the accumulator or a real
    reproduction failure.  (The step221 runner already reproduced 0.0 on the same data in the STEP246
    stage-1 gate, results_step246gate.json.)

(B) What the production stage-2 trees actually DO with the STEP237 block.  For each family the stage-2
    LightGBM is refit on the reference list and on the reference list + the 25 columns (ac.CLF has
    colsample_bytree and subsample at their defaults of 1.0, so a feature that never wins a split
    leaves the model bit-identical).  Reports, per family: the share of total split gain taken by the
    block, how many of the 25 columns are ever used, each used column's gain share and rank, and which
    of the six outsider-card columns are used at all.

Development phase only (asserted by the loader).  Reads no evaluation row.  Builds no file that any
decision depends on.  Never uploads.
Usage: python step246_D_diagnostics.py
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step246_20260917'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
R = lambda x: round(float(x), 6)


def main():
    reg = json.loads((OUT / 'registration246.json').read_text(encoding='utf-8'))
    B25 = reg['block']['B25']
    OUT6 = reg['block']['dropped_for_B19']
    t0 = time.perf_counter()
    D = S165.load()
    q0 = D['Q'].select(K)
    for rel, cols in reg['feature_files'].items():
        f = pl.read_parquet(CV / rel).select(K + cols)
        D['Q'] = D['Q'].join(f, on=K, how='left', validate='1:1', maintain_order='left')
    assert D['Q'].select(K).equals(q0)
    d, Q = D['d'], D['Q']
    from evidence_aug.step165b_C_lp_v1_level import build_views

    base = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    REFADD = reg['lists']['ref']
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    ac = D['ac']

    # ---------------------------------------------------------------- (A) the S* replay, both orders
    MS, FOLDS = {}, {}
    for f in FAMS:
        ix = np.flatnonzero(isf[f])
        q = Q[ix]
        yy = q['relevance'].to_numpy().astype(int)
        FOLDS[f] = (ix, list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), yy, q['table_id'].to_numpy())))
        z = q.select(base[f]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(FOLDS[f][1]):
            MS[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[tr], yy[tr])

    def score_sstar(v):
        p = D['SS'].copy()
        for f in FAMS:
            ix, folds = FOLDS[f]
            zv = v[ix].select(base[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds):
                p[D['QIX'][ix[te]]] = MS[(f, k)].predict_proba(zv[te])[:, 1] + 10
        return p

    assert float(np.abs(score_sstar(Q) - D['SS']).max()) == 0.0
    views, pads = build_views(D)
    two = [np.zeros(len(d)), np.zeros(len(d))]
    one = np.zeros(len(d))
    nv = 0
    for slot, (maker, seeds) in enumerate(((views, (1241, 2241)), (pads, (1242, 2242)))):
        for s in seeds:
            for v in maker(.5, 10, s):
                sc = score_sstar(v)
                two[slot] += sc
                one += sc
                nv += 1
    assert nv == 40, nv
    v_two = (two[0] + two[1] + D['SS']) / (nv + 1)
    v_one = (one + D['SS']) / (nv + 1)
    A = {'views': nv,
         'step221_two_slot_accumulator_max_abs_vs_D_V1': float(np.abs(v_two - D['V1']).max()),
         'step246_single_accumulator_max_abs_vs_D_V1': float(np.abs(v_one - D['V1']).max()),
         'max_abs_between_the_two_orders': float(np.abs(v_two - v_one).max()),
         'relative_size': float(np.abs(v_two - v_one).max() / np.abs(D['V1']).max()),
         'verdict': None}
    A['verdict'] = ('the failing gate is floating-point association in the accumulator, not a '
                    'reproduction failure' if A['step221_two_slot_accumulator_max_abs_vs_D_V1'] == 0.0
                    else 'REAL REPRODUCTION FAILURE -- investigate')
    print(json.dumps(A, indent=1), flush=True)

    # ---------------------------------------------------------------- (B) does stage 2 use the block?
    y = Q['relevance'].to_numpy().astype(int)
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    qpairs = Q['pair_id'].to_numpy()
    folds_q = [(np.flatnonzero(np.isin(qpairs, upairs[tr])), np.flatnonzero(np.isin(qpairs, upairs[te])))
               for tr, te in outer]
    B = {}
    for f in FAMS:
        ref_cols = list(base[f]) + list(REFADD.get(f, []))
        plus_cols = ref_cols + B25
        z = Q.select(plus_cols).to_numpy().astype(float)
        zr = Q.select(ref_cols).to_numpy().astype(float)
        gain = np.zeros(len(plus_cols))
        same = 0
        nused_ref, nused_plus = set(), set()
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            # no network columns here: this asks only which per-hand columns the trees reach for
            mp = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[a], y[a])
            mr = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(zr[a], y[a])
            g = mp.booster_.feature_importance('gain')
            gain += g
            nused_plus |= {plus_cols[i] for i in np.flatnonzero(g)}
            gr = mr.booster_.feature_importance('gain')
            nused_ref |= {ref_cols[i] for i in np.flatnonzero(gr)}
            b = te[isf[f][te]]
            if len(b) and np.abs(mp.predict_proba(z[b])[:, 1] - mr.predict_proba(zr[b])[:, 1]).max() == 0.0:
                same += 1
        tot = gain.sum()
        blockg = {c: R(gain[len(ref_cols) + i] / tot) for i, c in enumerate(B25)}
        order = sorted(range(len(plus_cols)), key=lambda i: -gain[i])
        rank = {c: order.index(len(ref_cols) + i) + 1 for i, c in enumerate(B25) if gain[len(ref_cols) + i] > 0}
        B[f] = {
            'n_reference_columns': len(ref_cols), 'n_columns_with_the_block': len(plus_cols),
            'block_gain_share': R(sum(gain[len(ref_cols):]) / tot),
            'block_columns_ever_used': sorted(c for c in B25 if blockg[c] > 0),
            'n_block_columns_used': int(sum(1 for c in B25 if blockg[c] > 0)),
            'outsider_card_columns_used': sorted(c for c in OUT6 if blockg[c] > 0),
            'per_column_gain_share': dict(sorted(blockg.items(), key=lambda t: -t[1])),
            'rank_among_all_columns': dict(sorted(rank.items(), key=lambda t: t[1])),
            'reference_columns_ever_used': len(nused_ref),
            'columns_ever_used_with_block': len(nused_plus),
            'folds_where_the_block_changed_nothing': same}
        print(f, json.dumps({k: B[f][k] for k in ('block_gain_share', 'n_block_columns_used',
                                                  'outsider_card_columns_used',
                                                  'reference_columns_ever_used',
                                                  'folds_where_the_block_changed_nothing')}), flush=True)
    res = {'note': 'post-hoc diagnostics; no registered decision depends on them',
           'A_sstar_replay': A, 'B_block_usage': B, 'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'diagnostics246.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('wrote diagnostics246.json  %ds' % round(time.perf_counter() - t0), flush=True)


if __name__ == '__main__':
    main()
