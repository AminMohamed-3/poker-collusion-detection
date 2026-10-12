"""STEP196: train on the LLM annotator's judgements, which failed their own calibration gate.

STEP195's gate failed at -6.0 sigma (annotator 0.2842 vs production 0.4435 over 292 pairs). Amin
asked to explore the lane anyway, so this measures it instead of arguing about it.

What is used is NOT the annotator's answers as answers -- ranks 4 and 5 are already known, and
training on a guess at a known label is strictly worse than training on the label. What is used is
its MISSES: 177 hands it judged as strong evidence that sit outside the true top 5, inside the
audit's candidate set. Those are candidates for the censored positives, since 340 of 372
development lists are capped at exactly 5 while 32 are shorter -- so 5 is a cap, not a quota.

Labels enter stage-2 training only, with sample_weight w. Evaluation is always against the true
relevance. The joint network is fitted ONCE and shared by all four arms, so the only thing that
varies is the stage-2 training signal.

Registration outputs_step196_20260916/registration196.json. Audit only. Never uploads.

Usage: python step196_C_annotator_labels_train.py REGSHA
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step196_20260916'
REG = OUT / 'registration196.json'
S195 = CV / 'outputs_step195_20260916'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
WEIGHTS = {'W025': 0.25, 'W050': 0.5, 'W100': 1.0}


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    t0 = time.perf_counter()

    D = S165.load()
    d, Q = D['d'], D['Q']
    y_true = Q['relevance'].to_numpy().astype(int)
    qpos = {(p, h): i for i, (p, h) in enumerate(zip(Q['pair_id'].to_list(), Q['hand_id'].to_list()))}

    key = json.loads((S195 / 'ANSWER_KEY_calib.json').read_text(encoding='utf-8'))['hidden_by_pair']
    ans = json.loads((S195 / 'calib/answers.json').read_text(encoding='utf-8'))
    ans = ans.get('answers', ans)
    add = []
    for pid, v in ans.items():
        order = key[pid]['hand_order']
        for x in v[:2]:
            i = qpos.get((pid, order[int(x)]))
            if i is not None and y_true[i] == 0:
                add.append(i)
    add = np.array(sorted(set(add)))
    print(f'added positives: {len(add)} (registration says 177)', flush=True)
    assert len(add) == 177, f'coverage changed: {len(add)}'
    assert (y_true[add] == 0).all(), 'an added row was already positive'

    X, C = S165.matrices(D)
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'],
                 d['relevance'].to_numpy(), pairs, tables)
    jobs = []
    for k, (tr, te) in enumerate(outer):
        for s in S165.SEEDS_INNER_OUTER:
            jobs.append((('outer', k), 'joint', s, rowmask(tr), rowmask(te)))
        for j, (itr, ite) in enumerate(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
            for s in S165.SEEDS_INNER:
                jobs.append((('inner', k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    print(f'X {X.shape}  {len(jobs)} network fits (shared by all arms)', flush=True)
    with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    print(f'network refits done {round(time.perf_counter()-t0)}s', flush=True)

    nq = len(Q)
    AF_te, AF_in = {k: [] for k in range(5)}, {k: {} for k in range(5)}
    for jtag, m, s, rows, tq, p, S, trace, sec in out:
        if jtag[0] == 'outer':
            AF_te[jtag[1]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
        else:
            AF_in[jtag[1]].setdefault(jtag[2], []).append((tq, S165.aligned_features(D, rows, tq, S, p)))
    qpairs = Q['pair_id'].to_numpy()
    folds_q, EA, EB = [], {}, np.full((nq, len(S165.NN_COLS)), np.nan)
    for k, (tr, te) in enumerate(outer):
        trq = np.flatnonzero(np.isin(qpairs, upairs[tr]))
        teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
        ea = np.full((nq, len(S165.NN_COLS)), np.nan)
        for j, lst in AF_in[k].items():
            ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
        eb = np.full((nq, len(S165.NN_COLS)), np.nan)
        eb[AF_te[k][0][0]] = np.mean([a for _, a in AF_te[k]], axis=0)
        folds_q.append((trq, teq))
        EA[k] = ea[trq]
        EB[teq] = eb[teq]
    assert np.isfinite(EB).all()

    ac = D['ac']
    colsf = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    Zf = {f: Q.select(colsf[f]).to_numpy().astype(float) for f in FAMS}

    def run(w):
        y = y_true.copy()
        sw = np.ones(nq)
        if w is not None:
            y[add] = 1
            sw[add] = w
        p = D['SS'].copy()
        for f in FAMS:
            z = Zf[f]
            for k, (tr, te) in enumerate(folds_q):
                a = tr[isf[f][tr]]
                m = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                    np.column_stack([z[a], EA[k][isf[f][tr]]]), y[a], sample_weight=sw[a])
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = m.predict_proba(np.column_stack([z[b], EB[b]]))[:, 1] + 10
        return p

    scores = {'control': run(None)}
    for name, w in WEIGHTS.items():
        scores[name] = run(w)
    print(f'all arms scored {round(time.perf_counter()-t0)}s', flush=True)

    dec = D['dec']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    kf = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    M = {}
    for arm, s in scores.items():
        _, m = summarize(kf.with_columns(pl.Series('score', dec(s))), truths, fams)
        M[arm] = np.array([m[q] for q in qids])
    hv = np.array([half(ptabq[q]) for q in qids])
    R = lambda x: round(float(x), 6)
    names = list(WEIGHTS)
    A = np.array([M[n] - M['control'] for n in names])

    # honest cross-half selection: pick on one half, read on the other
    held = {}
    for h in (0, 1):
        on, off = hv == h, hv != h
        best = int(np.argmax(A[:, on].mean(1)))
        held[h] = {'selected_on_half': h, 'arm': names[best],
                   'held_half_delta': R(A[best, off].mean()), 'held_half_n': int(off.sum()),
                   'selection_half_delta': R(A[best, on].mean())}
    G = (held[0]['held_half_delta'] * held[0]['held_half_n']
         + held[1]['held_half_delta'] * held[1]['held_half_n']) / len(qids)

    res = {'registration_sha256': reg_sha, 'level': 'LP original pool (u=0), 162-query audit',
           'n_queries': len(qids), 'added_positives': int(len(add)),
           'map5_control': R(M['control'].mean()),
           'arms': {n: {'map5': R(M[n].mean()), 'delta': R((M[n] - M['control']).mean()),
                        'half0': R((M[n] - M['control'])[hv == 0].mean()),
                        'half1': R((M[n] - M['control'])[hv == 1].mean()),
                        'positive_on_both_halves': bool((M[n] - M['control'])[hv == 0].mean() > 0
                                                        and (M[n] - M['control'])[hv == 1].mean() > 0)}
                    for n in names},
           'cross_half': held, 'G_honest': R(G),
           'gates': {'control_reproduces_194': {'expected': 0.69, 'here': R(M['control'].mean()),
                                                'tolerance': 0.002,
                                                'passed': bool(abs(M['control'].mean() - 0.69) <= 0.002)},
                     'added_count_is_177': bool(len(add) == 177)},
           'bar': {'corrected_build_bar_decoded': 0.0073, 'clears_bar': bool(G >= 0.0073)},
           'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'results196.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
