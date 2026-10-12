"""Step 165b (Claude): the registered if_pass of STEP165. LP (S* LightGBM stage 2 + 7 aligned stage-1 outputs of the end-to-end network)
passed ER1 against S*g on the network folds. Here both are taken to the v1 level on the training-only 162-query audit: each fold model
is averaged over the original pool and the same 40 label-free pool views as v1 (20 thinned: seeds 1241/2241; 20 padded: 1242/2242;
step-124 recipe). The network outputs are held at their original-pool values in every view (a production simplification: the stage-1
outputs are fixed, as the stage-1 LightGBM score is). Gates: the 250 network fits and both LightGBM stage 2s reproduce STEP165's saved
scores exactly; the S* per-family fold models over these views reproduce the saved v1 exactly.
Primary: decode(v1LP) vs decode(v1g) under ER1. No held folds, no evaluation pairs. Never uploads. Usage: run REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '1'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from evidence_aug import step165_C_e2e_audit as S165

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/e2e_evidence_C_20260915'
REG = EXP / 'registration165b.json'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
POOLF4 = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof_rank']


def build_views(D):
    from evidence_aug.frame import context
    d, Q = D['d'], D['Q']
    ctxc = [c for c in D['rcols'] if c.startswith('context_')]; src = [c[8:] for c in ctxc]
    assert len(ctxc) == 136 and all(s in d.columns for s in src)
    POOL = d.select(K + ['relevance', 'chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))
    CMAX = float(POOL['chrono_rank'].max()) + 1
    keep = [c for c in Q.columns if c not in POOLF4 + ctxc]

    def finish(parts, m):
        a = pl.concat(parts).with_columns(pl.concat_str([pl.col('pair_id'), pl.col('copy').cast(pl.String)], separator='#').alias('pc'))
        a = a.with_columns((pl.col('chrono_rank').rank('ordinal').over('pc') - 1).cast(pl.Float64).alias('chrono_rank'),
                           pl.len().over('pc').cast(pl.Float64).alias('n_hands'),
                           pl.col('oof').rank('ordinal', descending=True).over('pc').cast(pl.Float64).alias('oof_rank'))
        a = a.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
        a = context(a.rename({'pair_id': 'true_pair'}).rename({'pc': 'pair_id'}), src).drop('pair_id').rename({'true_pair': 'pair_id'})
        a = a.filter(pl.col('top0')).select(K + ['copy'] + POOLF4 + ctxc)
        a = a.join(Q.select(keep), on=K, how='left', validate='m:1')
        out = []
        for c_ in range(m):
            v = a.filter(pl.col('copy') == c_).sort(K); assert v.select(K).equals(Q.select(K)); out.append(v)
        return out

    def views(u, m, seed):
        top0, parts = POOL['top0'].to_numpy(), []
        for c_ in range(m):
            r = np.random.default_rng([seed, c_]).random(POOL.height)
            parts.append(POOL.filter(pl.Series(~((r < u) & ~top0))).with_columns(pl.lit(c_).alias('copy')))
        return finish(parts, m)

    def pads(u, m, seed):
        top0, parts = POOL['top0'].to_numpy(), []
        for c_ in range(m):
            r = np.random.default_rng([seed, c_]).random(POOL.height)
            pick = (r < u) & ~top0
            dupl = POOL.filter(pl.Series(pick)).with_columns((pl.col('chrono_rank') + CMAX * 10).alias('chrono_rank'), pl.lit(False).alias('top0'))
            parts.append(pl.concat([POOL, dupl]).with_columns(pl.lit(c_).alias('copy')))
        return finish(parts, m)

    return views, pads


def main():
    t0 = time.perf_counter()
    reg_sha = sys.argv[2]
    assert S165.sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert S165.sha(Path(__file__)) == reg['code']['runner_sha256'] and S165.sha(CV / 'src/evidence_aug/step165_C_e2e_audit.py') == reg['code']['step165_runner_sha256']
    assert not (EXP / 'results165b.json').exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    D = S165.load()
    X, C = S165.matrices(D)
    d, Q = D['d'], D['Q']
    saved = pl.read_parquet(EXP / 'audit_scores165.parquet').sort(K); assert saved.select(K).equals(d.select(K))
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True); ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], d['relevance'].to_numpy(), pairs, tables)
    jobs = []
    for k, (tr, te) in enumerate(outer):
        for s in S165.SEEDS_INNER_OUTER:
            jobs.append((('outer', k), 'joint', s, rowmask(tr), rowmask(te)))
        for j, (itr, ite) in enumerate(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
            for s in S165.SEEDS_INNER:
                jobs.append((('inner', k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    with Pool(14, initializer=S165.pool_init, initargs=init_args) as pool:
        out = pool.map(S165.fit_job, jobs, chunksize=1)
    print('network refits', len(out), round(time.perf_counter() - t0), 's', flush=True)
    nq = len(Q)
    AF_te, AF_in = {k: [] for k in range(5)}, {k: {} for k in range(5)}
    for tag, m, s, rows, tq, p, S, trace, sec in out:
        if tag[0] == 'outer':
            AF_te[tag[1]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
        else:
            AF_in[tag[1]].setdefault(tag[2], []).append((tq, S165.aligned_features(D, rows, tq, S, p)))
    qpairs = Q['pair_id'].to_numpy()
    folds_q, EA, EB = [], {}, np.full((nq, len(S165.NN_COLS)), np.nan)
    for k, (tr, te) in enumerate(outer):
        trq = np.flatnonzero(np.isin(qpairs, upairs[tr])); teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
        ea = np.full((nq, len(S165.NN_COLS)), np.nan)
        for j, lst in AF_in[k].items():
            ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
        eb = np.full((nq, len(S165.NN_COLS)), np.nan)
        eb[AF_te[k][0][0]] = np.mean([a for _, a in AF_te[k]], axis=0)
        assert np.isfinite(ea[trq]).all() and np.isfinite(eb[teq]).all()
        folds_q.append((trq, teq)); EA[k] = ea[trq]; EB[teq] = eb[teq]
    assert np.isfinite(EB).all()
    ac = D['ac']
    y = Q['relevance'].to_numpy().astype(int)
    colsf = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols'] for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    MG, ML = {}, {}
    for f in FAMS:
        z = Q.select(colsf[f]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds_q):
            a = tr[isf[f][tr]]
            MG[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[a], y[a])
            ML[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(np.column_stack([z[a], EA[k][isf[f][tr]]]), y[a])

    def score_view(v, models, with_nn):
        p = D['SS'].copy()
        for f in FAMS:
            z = v.select(colsf[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds_q):
                b = te[isf[f][te]]
                if not len(b):
                    continue
                zb = np.column_stack([z[b], EB[b]]) if with_nn else z[b]
                p[D['QIX'][b]] = models[(f, k)].predict_proba(zb)[:, 1] + 10
        return p
    G0, L0 = score_view(Q, MG, False), score_view(Q, ML, True)
    gate_g, gate_l = float(np.abs(G0 - saved['S_star_g'].to_numpy()).max()), float(np.abs(L0 - saved['LP'].to_numpy()).max())
    print('replay S*g', gate_g, 'LP', gate_l, flush=True)
    assert gate_g == 0.0 and gate_l == 0.0
    # S* per-family fold models (step-124 FOLDS) for the view-construction gate
    MS, FOLDS = {}, {}
    for f in FAMS:
        ix = np.flatnonzero(isf[f]); q = Q[ix]; yy = q['relevance'].to_numpy().astype(int)
        FOLDS[f] = (ix, list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), yy, q['table_id'].to_numpy())))
        z = q.select(colsf[f]).to_numpy().astype(float)
        for k, (tr, te) in enumerate(FOLDS[f][1]):
            MS[(f, k)] = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(z[tr], yy[tr])

    def score_sstar(v):
        p = D['SS'].copy()
        for f in FAMS:
            ix, folds = FOLDS[f]
            zv = v[ix].select(colsf[f]).to_numpy().astype(float)
            for k, (tr, te) in enumerate(folds):
                p[D['QIX'][ix[te]]] = MS[(f, k)].predict_proba(zv[te])[:, 1] + 10
        return p
    views, pads = build_views(D)
    acc = {n: [np.zeros(len(d)), np.zeros(len(d))] for n in ('S', 'G', 'L')}  # [thin sum, pad sum]
    nv = [0, 0]
    for slot, maker, seeds in ((0, views, (1241, 2241)), (1, pads, (1242, 2242))):
        for s in seeds:
            for v in maker(.5, 10, s):
                acc['S'][slot] += score_sstar(v); acc['G'][slot] += score_view(v, MG, False); acc['L'][slot] += score_view(v, ML, True)
                nv[slot] += 1
            print('views', nv, round(time.perf_counter() - t0), 's', flush=True)
    V1r = (acc['S'][0] + acc['S'][1] + D['SS']) / (nv[0] + nv[1] + 1)
    gate_v1 = float(np.abs(V1r - D['V1']).max())
    print('v1 replay max abs diff', gate_v1, flush=True)
    assert gate_v1 == 0.0
    V1G = (acc['G'][0] + acc['G'][1] + G0) / (nv[0] + nv[1] + 1)
    V1L = (acc['L'][0] + acc['L'][1] + L0) / (nv[0] + nv[1] + 1)
    dec = D['dec']
    R = lambda x: round(float(x), 6)

    def cmp(A0, B0):
        c = ac.evaluate(d, {'incumbent': A0, 'arm': B0})['comparisons']['arm']
        return {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                'fam': {kk[:2]: R(x) for kk, x in c['by_family'].items()}, 'halves': {kk: R(x) for kk, x in c['halves'].items()}, 'ac_gate': bool(c['gate_pass'])}
    DV1, DG, DL = dec(D['V1']), dec(V1G), dec(V1L)
    comp = {'primary_dec_v1LP_vs_dec_v1g': cmp(DG, DL), 'dec_v1LP_vs_dec_v1_saved': cmp(DV1, DL), 'dec_v1g_vs_dec_v1_saved': cmp(DV1, DG),
            'raw_v1LP_vs_v1g': cmp(V1G, V1L)}
    r = reg['rule']; c = comp['primary_dec_v1LP_vs_dec_v1g']
    passes = c['mean'] > 0 and c['ci95'][0] > r['ci_lower_floor'] and min(c['fam'].values()) >= r['family_floor']
    yv = d['relevance'].to_numpy()[D['QIX']]
    ll = lambda s: R(-np.mean(yv * np.log(np.clip(s[D['QIX']] - 10, 1e-6, 1 - 1e-6)) + (1 - yv) * np.log(1 - np.clip(s[D['QIX']] - 10, 1e-6, 1 - 1e-6))))
    maps = {n: R(ac.evaluate(d, {'incumbent': s})['sets']['incumbent']['map5']) for n, s in (('dec_v1', DV1), ('dec_v1g', DG), ('dec_v1LP', DL))}
    d.select(K).with_columns(pl.Series('v1g', V1G), pl.Series('v1LP', V1L), pl.Series('dec_v1g', DG), pl.Series('dec_v1LP', DL)).write_parquet(EXP / 'audit_scores165b.parquet')
    assert check_protected() == prot
    res = {'registration_sha256': reg_sha, 'gates': {'S_star_g_replay': gate_g, 'LP_replay': gate_l, 'v1_replay': gate_v1, 'views': nv},
           'decoded_map5': maps, 'logloss_top20': {'v1': ll(D['V1']), 'v1g': ll(V1G), 'v1LP': ll(V1L)}, 'comparisons': comp, 'ER1_pass': bool(passes),
           'scores_sha256': S165.sha(EXP / 'audit_scores165b.parquet'), 'wall_seconds': round(time.perf_counter() - t0, 1)}
    (EXP / 'results165b.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
