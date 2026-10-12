"""Step 165 (Claude): end-to-end evidence network on the training-only 162-query audit (user idea 2026-09-15: train the evidence
stages end to end instead of separately, with a medium MLP layer so stage 1 passes more than one number to stage 2).
Arms (module step165_e2e_mlp): NE joint end-to-end network; N2 the same network trained in two separate phases (control);
NB probability mean of S* and NE; LP the production LightGBM stage 2 (S* columns) plus aligned NE stage-1 outputs, nested cross-fitted.
Cross-fitting: one GroupKFold(5) by table over all 162 queries for the networks; LP and its reference S*g use the same folds.
Decoding = the production typed decoder with T1 off for coordinated isolation. No held folds, no evaluation pairs. Never uploads.
Usage: dry (in-sample timing/trace on outer fold 0, no test prediction) | run REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '1'
import sys, json, time, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold

CV, CR = POKER / 'Claude Version', POKER / 'Codex Resume'
EXP = CV / 'experiments/e2e_evidence_C_20260915'
REG = EXP / 'registration165.json'
K = ['pair_id', 'hand_id']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
CI = FAMS[2]
POOLF = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
SEEDS_OUTER = [16500 + i for i in range(20)]
SEEDS_INNER = [16600 + i for i in range(10)]
NN_COLS = ['nn_own_c', 'nn_own_rank', 'nn_DT_c', 'nn_SP_c', 'nn_CI_c', 'nn_p2', 'nn_p2_rank']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load():
    import audit_common as ac
    from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align
    d, shared, rcols, _ = ac.load_current()
    cs = pl.read_parquet(CR / 'research/contestable_state_20260912/features.parquet').sort(K)
    cs_cols = [c for c in cs.columns if c not in K]
    pf = pl.read_parquet(CR / 'research/retrospective_strength_20260912/prefix.parquet').sort(K)
    pf_cols = json.load(open(CR / 'research/retrospective_strength_20260912/freeze.json', encoding='utf-8'))['features']
    assert len(cs_cols) == 29 and pf.columns == K + pf_cols
    top = (d['oof_rank'] <= 20).to_numpy(); QIX = np.flatnonzero(top)
    Q = d[QIX].join(cs, on=K, validate='1:1').join(pf, on=K, validate='1:1').sort(K)
    assert Q.select(K).equals(d[QIX].select(K))
    v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
    assert v.select(K).equals(d.select(K))
    fr = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
    assert fr.select(K).equals(d.select(K))
    t1 = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
    fam = d['behavior_family'].to_numpy()
    typed = t1 & (fam != CI)
    dec = lambda p: decode(fr.with_columns(pl.Series('score', p)), typed)
    struct = [c for c in shared if c not in POOLF]
    return dict(ac=ac, d=d, shared=shared, rcols=rcols, struct=struct, cs_cols=cs_cols, pf_cols=pf_cols, Q=Q, QIX=QIX, top=top,
                SS=v['S_star'].to_numpy(), V1=v['v1'].to_numpy(), dec=dec, fam=fam)


def matrices(D):
    d, Q = D['d'], D['Q']
    Xc = D['rcols'] + D['struct']
    X = np.column_stack([d.select(Xc).to_numpy().astype(np.float64), d['chrono_frac'].to_numpy(), np.log(d['n_hands'].to_numpy())] +
                        [(D['fam'] == f).astype(float) for f in FAMS])
    C = np.column_stack([Q.select(D['cs_cols'] + D['pf_cols']).to_numpy().astype(np.float64), Q['chrono_frac'].to_numpy()])
    assert np.isfinite(X).all() and np.isfinite(C).all()
    return X, C


def standardize(A, rows):
    mu, sd = A[rows].mean(0), A[rows].std(0)
    sd = np.where(sd > 1e-9, sd, 1.)
    return np.clip((A - mu) / sd, -6, 6)


def make_batch(M, D, X, C, pairs_mask_rows):
    """Batch over rows (sorted, whole pairs) with top-20 rows mapped into it."""
    rows = np.flatnonzero(pairs_mask_rows)
    pos = np.full(len(pairs_mask_rows), -1); pos[rows] = np.arange(len(rows))
    tq = np.flatnonzero(pairs_mask_rows[D['QIX']])
    T = pos[D['QIX'][tq]]
    famidx = np.array([FAMS.index(f) for f in D['fam'][rows]])
    return M.Batch(X[rows], famidx, D['d']['relevance'].to_numpy()[rows], D['d']['pair_id'].to_numpy()[rows], T, C[tq]), rows, tq


def fit_job(job):
    from evidence_aug import step165_e2e_mlp as M
    tag, mode, seed, train_mask, pred_mask = job
    D = G
    rows_tr = np.flatnonzero(train_mask)
    tabs = np.unique(G['tab'][rows_tr])
    vt = np.random.default_rng([seed, 7]).choice(tabs, size=max(1, int(round(len(tabs) * M.CONFIG['val_share_tables']))), replace=False)
    val_mask = train_mask & np.isin(G['tab'], vt)
    fit_mask = train_mask & ~val_mask
    Xs = standardize(G['X'], rows_tr)
    Cs = standardize(G['C'], np.flatnonzero(train_mask[D['QIX']]))
    Btr, _, _ = make_batch(M, D, Xs, Cs, fit_mask)
    Bv, _, _ = make_batch(M, D, Xs, Cs, val_mask)
    t0 = time.perf_counter()
    P, trace = M.train(Btr, mode, seed, Bv)
    Bte, rows, tq = make_batch(M, D, Xs, Cs, pred_mask)
    p, S = M.predict(P, Bte)
    return tag, mode, seed, rows, tq, p.astype(np.float64), S.astype(np.float64), [float(x) for x in trace], round(time.perf_counter() - t0, 2)


G = {}


def pool_init(Xa, Ca, QIX, fam, rel, pid, tab):
    import polars as _pl
    G.update(X=Xa, C=Ca, QIX=QIX, fam=fam, tab=tab, d=_pl.DataFrame({'relevance': rel, 'pair_id': pid}))


def aligned_features(D, rows, tq, S, p):
    """Within-pair centred family stage-1 scores and ranks for top-20 rows, from all-hand scores S (rows) and stage-2 p (tq)."""
    pid = D['d']['pair_id'].to_numpy()[rows]
    famidx = np.array([FAMS.index(f) for f in D['fam'][rows]])
    f = pl.DataFrame({'pair_id': pid, 'own': S[np.arange(len(rows)), famidx], 'DT': S[:, 0], 'SP': S[:, 1], 'CI': S[:, 2]})
    f = f.with_columns([(pl.col(c) - pl.col(c).mean().over('pair_id')).alias(c + '_c') for c in ('own', 'DT', 'SP', 'CI')] +
                       [((pl.col('own').rank('ordinal', descending=True).over('pair_id') - 1) / pl.len().over('pair_id')).alias('own_rank')])
    pos = np.full(len(D['d']), -1); pos[rows] = np.arange(len(rows))
    sub = f[pos[D['QIX'][tq]]]
    tp = pl.DataFrame({'pair_id': D['Q']['pair_id'].to_numpy()[tq], 'p': p}).with_columns(
        ((pl.col('p').rank('ordinal', descending=True).over('pair_id') - 1) / 19.).alias('pr'))
    return np.column_stack([sub['own_c'], sub['own_rank'], sub['DT_c'], sub['SP_c'], sub['CI_c'], p, tp['pr']])


def lgb_stage2(D, folds_q, extra=None):
    """Per-family LightGBM stage 2 (ac.CLF, random_state 11) on S* columns (+ extra top-20 columns), cross-fitted by folds_q
    (list of (train_q, test_q) top-20 row index arrays). extra: dict fold -> (train_matrix, test_matrix) aligned with those rows."""
    import lightgbm as lgb
    ac, Q = D['ac'], D['Q']
    pred = np.full(len(Q), np.nan)
    for f in FAMS:
        cols = D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols']
        z = Q.select(cols).to_numpy().astype(float); y = Q['relevance'].to_numpy().astype(int); isf = (Q['behavior_family'] == f).to_numpy()
        for k, (tr, te) in enumerate(folds_q):
            a, b = tr[isf[tr]], te[isf[te]]
            if not len(b):
                continue
            Xa, Xb = z[a], z[b]
            if extra is not None:
                ea, eb = extra[k]
                Xa = np.column_stack([Xa, ea[isf[tr]]]); Xb = np.column_stack([Xb, eb[isf[te]]])
            m = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(Xa, y[a])
            pred[b] = m.predict_proba(Xb)[:, 1]
    assert np.isfinite(pred).all()
    return pred


def sstar_replay(D):
    """The saved S* uses per-family GroupKFold(5) on top-20 rows; reproduce it exactly."""
    Q = D['Q']; folds = []
    out = np.full(len(Q), np.nan)
    import lightgbm as lgb
    for f in FAMS:
        ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy()); q = Q[ix]
        y = q['relevance'].to_numpy().astype(int); g = q['table_id'].to_numpy()
        cols = D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols']
        z = q.select(cols).to_numpy().astype(float)
        for tr, te in GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g):
            out[ix[te]] = lgb.LGBMClassifier(**D['ac'].CLF, random_state=11).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
    full = D['SS'].copy(); full[D['QIX']] = out + 10
    return float(np.abs(full - D['SS']).max())


def full_scores(D, p_top):
    s = D['SS'].copy(); s[D['QIX']] = 10 + p_top
    return s


def main():
    mode = sys.argv[1]
    t0 = time.perf_counter()
    D = load()
    X, C = matrices(D)
    d = D['d']
    pairs = d['pair_id'].to_numpy(); tables = d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    init_args = (X.astype(np.float32), C.astype(np.float32), D['QIX'], D['fam'], d['relevance'].to_numpy(), pairs, tables)
    print('data', X.shape, C.shape, 'pairs', len(upairs), 'load s', round(time.perf_counter() - t0, 1), flush=True)
    if mode == 'dry':
        pool_init(*init_args)
        tr, te = outer[0]
        for m in ('joint', 'two_stage'):
            r = fit_job(('dry', m, SEEDS_OUTER[0], rowmask(tr), rowmask(tr)))
            print('dry', m, 'trace [first, best val, best epoch] per phase', [round(x, 4) for x in r[7]], 'fit s', r[8], flush=True)
            y = d['relevance'].to_numpy()[D['QIX'][r[4]]]
            pp = np.clip(r[5], 1e-6, 1 - 1e-6)
            print('   in-sample top-20 logloss', round(float(-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))), 4), 'mean p', round(float(pp.mean()), 4), flush=True)
        return
    reg_sha = sys.argv[2]
    assert sha(REG) == reg_sha
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert sha(CV / 'src/evidence_aug/step165_e2e_mlp.py') == reg['code']['module_sha256'] and sha(Path(__file__)) == reg['code']['runner_sha256']
    for name, v in reg['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], name
    assert not (EXP / 'results165.json').exists(), 'preserve a completed read'
    from evidence_aug import step165_e2e_mlp as M
    assert json.loads(json.dumps(M.CONFIG)) == reg['method']['config']
    from evidence_aug.paths import check_protected
    prot = check_protected()
    rep = sstar_replay(D)
    print('S* replay max abs diff', rep, flush=True)
    assert rep == 0.0
    jobs = []
    for k, (tr, te) in enumerate(outer):
        for s in SEEDS_OUTER:
            jobs.append((('outer', k), 'joint', s, rowmask(tr), rowmask(te)))
            jobs.append((('outer', k), 'two_stage', s, rowmask(tr), rowmask(te)))
        sub = list(GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr]))
        for j, (itr, ite) in enumerate(sub):
            for s in SEEDS_INNER:
                jobs.append((('inner', k, j), 'joint', s, rowmask(tr[itr]), rowmask(tr[ite])))
    # the outer-fold training rows also need NE outputs for themselves? no: LP training rows use inner OOF outputs only
    print('jobs', len(jobs), flush=True)
    with Pool(14, initializer=pool_init, initargs=init_args) as pool:
        out = pool.map(fit_job, jobs, chunksize=1)
    nq = len(D['Q'])
    P = {m: np.zeros((len(SEEDS_OUTER), nq)) for m in ('joint', 'two_stage')}
    AF_te = {k: [] for k in range(5)}
    AF_in = {k: {} for k in range(5)}
    secs, traces = 0., {'joint': [], 'two_stage': [], 'inner': []}
    for tag, m, s, rows, tq, p, S, trace, sec in out:
        secs += sec
        if tag[0] == 'outer':
            P[m][SEEDS_OUTER.index(s)][tq] = p
            traces[m].append(trace)
            if m == 'joint' and s in SEEDS_INNER_OUTER:
                AF_te[tag[1]].append((tq, aligned_features(D, rows, tq, S, p)))
        else:
            AF_in[tag[1]].setdefault(tag[2], []).append((tq, aligned_features(D, rows, tq, S, p)))
            traces['inner'].append(trace)
    NE, N2 = P['joint'].mean(0), P['two_stage'].mean(0)
    # LP: LightGBM training rows get inner-OOF aligned outputs (seed mean), test rows the outer outputs (first 10 outer seeds)
    folds_q, extra = [], {}
    qpairs = D['Q']['pair_id'].to_numpy()
    for k, (tr, te) in enumerate(outer):
        trq = np.flatnonzero(np.isin(qpairs, upairs[tr])); teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
        ea = np.full((nq, len(NN_COLS)), np.nan)
        for j, lst in AF_in[k].items():
            assert len(lst) == len(SEEDS_INNER)
            ea[lst[0][0]] = np.mean([a for _, a in lst], axis=0)
        eb = np.full((nq, len(NN_COLS)), np.nan)
        assert len(AF_te[k]) == len(SEEDS_INNER_OUTER)
        eb[AF_te[k][0][0]] = np.mean([a for _, a in AF_te[k]], axis=0)
        assert np.isfinite(ea[trq]).all() and np.isfinite(eb[teq]).all()
        folds_q.append((trq, teq)); extra[k] = (ea[trq], eb[teq])
    SG = lgb_stage2(D, folds_q)
    LP = lgb_stage2(D, folds_q, extra)
    ac, dec = D['ac'], D['dec']
    S = {'S_star': D['SS'], 'NE': full_scores(D, NE), 'N2': full_scores(D, N2), 'NB': full_scores(D, .5 * (D['SS'][D['QIX']] - 10) + .5 * NE),
         'S_star_g': full_scores(D, SG), 'LP': full_scores(D, LP), 'v1': D['V1']}
    DEC = {k: dec(v) for k, v in S.items()}
    R = lambda x: round(float(x), 6)
    yv = d['relevance'].to_numpy()[D['QIX']]

    def logloss(s):
        v = np.clip(s[D['QIX']] - 10, 1e-6, 1 - 1e-6)
        return R(-np.mean(yv * np.log(v) + (1 - yv) * np.log(1 - v)))

    def cmp(a, b, A, B):
        c = ac.evaluate(d, {'incumbent': A, b: B})['comparisons'][b]
        return {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                'fam': {kk[:2]: R(x) for kk, x in c['by_family'].items()}, 'halves': {kk: R(x) for kk, x in c['halves'].items()}, 'ac_gate': bool(c['gate_pass'])}

    maps = {k: R(ac.evaluate(d, {'incumbent': v})['sets']['incumbent']['map5']) for k, v in DEC.items()}
    raw = {k: R(ac.evaluate(d, {'incumbent': v})['sets']['incumbent']['map5']) for k, v in S.items()}
    comp = {'NE_dec_vs_S_star_dec': cmp('S', 'NE', DEC['S_star'], DEC['NE']), 'N2_dec_vs_S_star_dec': cmp('S', 'N2', DEC['S_star'], DEC['N2']),
            'NE_dec_vs_N2_dec': cmp('N2', 'NE', DEC['N2'], DEC['NE']), 'NB_dec_vs_S_star_dec': cmp('S', 'NB', DEC['S_star'], DEC['NB']),
            'LP_dec_vs_S_star_g_dec': cmp('SG', 'LP', DEC['S_star_g'], DEC['LP']), 'S_star_g_dec_vs_S_star_dec': cmp('S', 'SG', DEC['S_star'], DEC['S_star_g']),
            'NE_raw_vs_S_star_raw': cmp('S', 'NEr', S['S_star'], S['NE']), 'v1_dec_vs_S_star_dec_reference': cmp('S', 'V1', DEC['S_star'], DEC['v1'])}
    rule = reg['rule']
    def passes(c):
        return c['mean'] > 0 and c['ci95'][0] > rule['ci_lower_floor'] and min(c['fam'].values()) >= rule['family_floor']
    verdict = {a: passes(comp[c]) for a, c in (('NE', 'NE_dec_vs_S_star_dec'), ('NB', 'NB_dec_vs_S_star_dec'), ('LP', 'LP_dec_vs_S_star_g_dec'))}
    for kk, v in comp.items():
        print(kk, json.dumps(v), flush=True)
    print('decoded MAP', json.dumps(maps), 'raw MAP', json.dumps(raw), flush=True)
    EXP.mkdir(parents=True, exist_ok=True)
    d.select(K).with_columns(*[pl.Series(k, v) for k, v in S.items()], *[pl.Series('dec_' + k, v) for k, v in DEC.items()]).write_parquet(EXP / 'audit_scores165.parquet')
    assert check_protected() == prot
    tr_ = {m: np.array(v) for m, v in traces.items()}
    res = {'registration_sha256': reg_sha, 'S_star_replay_max_abs_diff': rep, 'decoded_map5': maps, 'raw_map5': raw,
           'logloss_top20': {k: logloss(v) for k, v in S.items()}, 'comparisons': comp, 'rule_pass': verdict,
           'train_trace_first_last_mean': {m: [R(v[:, 0].mean()), R(v[:, -1].mean())] for m, v in tr_.items()},
           'fits': len(out), 'fit_cpu_seconds': round(secs, 1), 'wall_seconds': round(time.perf_counter() - t0, 1),
           'scores_sha256': sha(EXP / 'audit_scores165.parquet'), 'held_reads': 11}
    (EXP / 'results165.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('RULE', json.dumps(verdict), 'wall', res['wall_seconds'], flush=True)


SEEDS_INNER_OUTER = SEEDS_OUTER[:10]

if __name__ == '__main__':
    main()
