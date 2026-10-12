"""STEP201: augmentation as a TRAINING CURRICULUM, not as extra rows in one fit.

Five augmentation attempts have already failed on this data (STEP109-123 thinning-train / role swap
/ padding, STEP134 stage-2 copies, STEP136 generation, STEP137 pseudo-labels). Every one of them
changed the COMPOSITION of a single training phase. None of them staged it.

step165_e2e_mlp.adam_phase(P, B, mode, keys, Bv) mutates the parameter dict in place and restores
the early-stopped best, so a second phase continues the SAME weights on a different batch.
Pretrain-then-finetune is therefore natively expressible and has never been run here.

    control  one joint phase on real data -- the incumbent M.train
    A        joint on AUGMENTED to convergence, then joint on REAL      (Amin's first trial)
    B        joint on REAL+AUGMENTED concatenated, then joint on REAL   (Amin's second trial)
    C        exactly A, but augmented labels SHUFFLED WITHIN PAIR       (THE CONTROL)
    D        augmented only, no real phase -- bounds the standalone value

C is the point of the design. If A gains and C gains equally, the warm-up is weight initialisation
or regularisation and transfers nothing about collusion. The pair (A, C) is the evidence, not A.

The augmentation is the step-124 recipe v1 already averages over: thinned views drop non-top-20 pool
rows with probability u, padded views duplicate them. It reaches the network through exactly three
things -- chrono_frac, log(n_hands), and the 136 context_ columns in rcols -- because struct excludes
POOLF. Everything else, and every label, is untouched.

GATE BEFORE ANY ARM RUNS: the d-shaped augmented frame, restricted to top-20 rows, must reproduce
step165b's Q-shaped view EXACTLY for the same seed and copy. build_views is imported for that
comparison, never retyped. Same discipline that caught STEP200's hand_order and STEP197's
accumulation.

Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step201_C_augmentation_curriculum.py REGSHA
"""
import os
import sys
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
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
OUT = CV / 'outputs_step201_20260917'
REG = OUT / 'registration201.json'
SCRATCH = Path('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
               'a82f9192-7916-45cc-9e3b-dc4040b8b623/scratchpad/step201_aug')
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
POOLF4 = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof_rank']
ARMS = ['control', 'A_WARMUP_AUG_THEN_REAL', 'B_MIXED_THEN_REAL_SWEEP',
        'C_SHUFFLED_AUG_WARMUP', 'D_AUG_ONLY']


def build_aug_frames(D, seeds_thin=(1241,), seeds_pad=(1242,), m=2, u=.5):
    """d-SHAPED views: the step165b recipe with the top0 filter removed so non-top-20 rows survive.

    Returns a list of polars frames, each a full alternative development frame for the network.
    """
    from evidence_aug.frame import context
    d, Q = D['d'], D['Q']
    ctxc = [c for c in D['rcols'] if c.startswith('context_')]
    src = [c[8:] for c in ctxc]
    assert len(ctxc) == 136 and all(s in d.columns for s in src)
    POOL = d.select(K + ['relevance', 'chrono_rank', 'oof', 'oof_rank'] + src).with_columns(
        pl.col('oof_rank').le(20).alias('top0'))
    CMAX = float(POOL['chrono_rank'].max()) + 1
    keep = [c for c in d.columns if c not in POOLF4 + ctxc]

    def finish(parts):
        a = pl.concat(parts).with_columns(
            pl.concat_str([pl.col('pair_id'), pl.col('copy').cast(pl.String)], separator='#').alias('pc'))
        a = a.with_columns((pl.col('chrono_rank').rank('ordinal').over('pc') - 1).cast(pl.Float64).alias('chrono_rank'),
                           pl.len().over('pc').cast(pl.Float64).alias('n_hands'),
                           pl.col('oof').rank('ordinal', descending=True).over('pc').cast(pl.Float64).alias('oof_rank'))
        a = a.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
        a = context(a.rename({'pair_id': 'true_pair'}).rename({'pc': 'pair_id'}), src).drop('pair_id').rename({'true_pair': 'pair_id'})
        a = a.select(K + ['copy', 'top0'] + POOLF4 + ctxc)
        return a.join(d.select(keep), on=K, how='left', validate='m:1')

    out = []
    for maker, seeds in (('thin', seeds_thin), ('pad', seeds_pad)):
        for s in seeds:
            parts = []
            top0 = POOL['top0'].to_numpy()
            for c_ in range(m):
                r = np.random.default_rng([s, c_]).random(POOL.height)
                if maker == 'thin':
                    parts.append(POOL.filter(pl.Series(~((r < u) & ~top0))).with_columns(pl.lit(c_).alias('copy')))
                else:
                    dupl = POOL.filter(pl.Series((r < u) & ~top0)).with_columns(
                        (pl.col('chrono_rank') + CMAX * 10).alias('chrono_rank'), pl.lit(False).alias('top0'))
                    parts.append(pl.concat([POOL, dupl]).with_columns(pl.lit(c_).alias('copy')))
            a = finish(parts)
            for c_ in range(m):
                out.append((maker, s, c_, a.filter(pl.col('copy') == c_).sort(K)))
    return out


GA = {}


def save_bundle(name, arrays, dtype=np.float32):
    """X and C as their own .npy so workers can memory-map them instead of each holding a copy.

    float32, because that is what the incumbent standardizes. step165_C_e2e_audit.matrices returns
    float64, but every caller (step165 main, step200) hands pool_init `X.astype(np.float32)`, so
    S165.standardize runs in float32 and Batch's own float32 cast is a no-op. Storing float64 here
    moves the standardisation into float64 and the control arm stops being the incumbent: measured,
    it leaves the trace identical to 4 digits and the same best epoch but moves predictions by up to
    4.9e-2 and the 162-query decoded MAP by -0.0067. See gate201_control_resolution.json.
    """
    X, C, QIX, fam, rel, pid, tab = arrays
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.save(SCRATCH / f'{name}_X.npy', X.astype(dtype))
    np.save(SCRATCH / f'{name}_C.npy', C.astype(dtype))
    np.savez(SCRATCH / f'{name}_meta.npz', QIX=QIX, fam=fam.astype(str), rel=rel, pid=pid.astype(str),
             tab=tab.astype(str))
    return name


def load_bundle(name):
    m = np.load(SCRATCH / f'{name}_meta.npz', allow_pickle=False)
    return {'X': np.load(SCRATCH / f'{name}_X.npy', mmap_mode='r'),
            'C': np.load(SCRATCH / f'{name}_C.npy', mmap_mode='r'),
            'QIX': m['QIX'], 'fam': m['fam'], 'rel': m['rel'], 'pid': m['pid'], 'tab': m['tab']}


def pool_init2(names):
    for group, lst in names.items():
        GA[group] = [load_bundle(n) for n in lst] if isinstance(lst, list) else load_bundle(lst)


def _batch(M, G, mask, Xs, Cs):
    """make_batch against an arbitrary G bundle (real or augmented)."""
    rows = np.flatnonzero(mask)
    pos = np.full(len(mask), -1)
    pos[rows] = np.arange(len(rows))
    tq = np.flatnonzero(mask[G['QIX']])
    T = pos[G['QIX'][tq]]
    famidx = np.array([FAMS.index(f) for f in G['fam'][rows]])
    return M.Batch(Xs[rows], famidx, G['rel'][rows], G['pid'][rows], T, Cs[tq])


def _split(G, seed, M, train_pairs):
    """the incumbent's own val rule, applied to whichever frame G is."""
    mask = np.isin(G['pid'], train_pairs)
    rows_tr = np.flatnonzero(mask)
    tabs = np.unique(G['tab'][rows_tr])
    vt = np.random.default_rng([seed, 7]).choice(
        tabs, size=max(1, int(round(len(tabs) * M.CONFIG['val_share_tables']))), replace=False)
    val = mask & np.isin(G['tab'], vt)
    fit = mask & ~val
    Xs = S165.standardize(G['X'], rows_tr)
    Cs = S165.standardize(G['C'], np.flatnonzero(mask[G['QIX']]))
    return _batch(M, G, fit, Xs, Cs), _batch(M, G, val, Xs, Cs), Xs, Cs


def fit_job2(job):
    from evidence_aug import step165_e2e_mlp as M
    tag, arm, seed, train_pairs, pred_pairs = job
    trunk = ['W1', 'b1', 'W2', 'b2', 'U', 'c']
    head = ['V1', 'e1', 'v2', 'e2']
    keys = trunk + head
    R = GA['real']
    Btr, Bv, Xs, Cs = _split(R, seed, M, train_pairs)
    P = M.init(Btr.X.shape[1], Btr.C.shape[1], seed, Btr.X.dtype)
    trace = []
    pool_aug = GA['shuf'] if arm == 'C_SHUFFLED_AUG_WARMUP' else GA['aug']
    A = pool_aug[seed % len(pool_aug)]

    if arm in ('A_WARMUP_AUG_THEN_REAL', 'C_SHUFFLED_AUG_WARMUP', 'D_AUG_ONLY'):
        aBtr, aBv, _, _ = _split(A, seed, M, train_pairs)
        trace += M.adam_phase(P, aBtr, 'joint', keys, aBv)
        if arm != 'D_AUG_ONLY':
            trace += M.adam_phase(P, Btr, 'joint', keys, Bv)
    elif arm == 'B_MIXED_THEN_REAL_SWEEP':
        # Batch derives its pair-block index from a `pair` array it does not keep, and assumes rows
        # are CONTIGUOUS by pair. Concatenating two Batch objects would interleave pairs and
        # silently corrupt first/gid/tfirst. So the mixed frame is built and SORTED BY PAIR at build
        # time and enters here as an ordinary bundle.
        Mx = GA['mixed'][seed % len(GA['mixed'])]
        mBtr, _, _, _ = _split(Mx, seed, M, train_pairs)
        trace += M.adam_phase(P, mBtr, 'joint', keys, Bv)
        trace += M.adam_phase(P, Btr, 'joint', keys, Bv)
    else:
        trace += M.adam_phase(P, Btr, 'joint', keys, Bv)

    pmask = np.isin(R['pid'], pred_pairs)
    rows = np.flatnonzero(pmask)
    tq = np.flatnonzero(pmask[R['QIX']])
    Bte = _batch(M, R, pmask, Xs, Cs)
    p, S = M.predict(P, Bte)
    return tag, arm, seed, rows, tq, p.astype(np.float64), S.astype(np.float64), [float(x) for x in trace]


def frame_to_arrays(D, a, expect_Q):
    """X, C and row bookkeeping from an augmented (or mixed) frame.

    cs_cols and pf_cols live on Q only and are NOT pool-derived -- thinning and padding cannot
    change them -- so C's feature block is taken from Q unchanged and only chrono_frac is the
    recomputed part. `a` must already be sorted so that rows are contiguous by pair.
    """
    Q = D['Q']
    Xc = D['rcols'] + D['struct']
    fam = a['behavior_family'].to_numpy()
    X = np.column_stack([a.select(Xc).to_numpy().astype(np.float64),
                         a['chrono_frac'].to_numpy(), np.log(a['n_hands'].to_numpy())] +
                        [(fam == f).astype(float) for f in FAMS])
    QIX = np.flatnonzero(a['top0'].to_numpy())
    top = a.filter(pl.col('top0'))
    if expect_Q:
        assert top.select(K).equals(Q.select(K)), 'augmented top-20 rows do not align with Q'
        assert np.array_equal(top['relevance'].to_numpy(), Q['relevance'].to_numpy()), \
            'augmented labels differ from their source rows'
    qmap = {k: i for i, k in enumerate(Q.select(K).iter_rows())}
    sel = np.array([qmap[k] for k in top.select(K).iter_rows()])
    base = Q.select(D['cs_cols'] + D['pf_cols']).to_numpy().astype(np.float64)[sel]
    C = np.column_stack([base, top['chrono_frac'].to_numpy()])
    assert np.isfinite(X).all() and np.isfinite(C).all()
    assert len(QIX) == len(C)
    return (X, C, QIX, fam, a['relevance'].to_numpy(),
            a['pair_id'].to_numpy(), a['table_id'].to_numpy())


def main():
    reg_sha = sys.argv[1]
    # argv[2] names the output file. The first run of this step FAILED its registered control gate
    # (0.683308 against 0.690 +- 0.002) because save_bundle stored float64 while the incumbent
    # standardizes float32; that read is preserved VOID in results201.json and the corrected run
    # writes results201_rerun_float32.json. The registration is unchanged -- same experiment, fixed
    # implementation.
    outname = sys.argv[2] if len(sys.argv) > 2 else 'results201.json'
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    assert not (OUT / outname).exists(), 'preserve a completed read'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.step165b_C_lp_v1_level import build_views
    from evidence_aug.paths import check_protected
    prot = check_protected()
    t0 = time.perf_counter()

    D = S165.load()
    d, Q = D['d'], D['Q']

    # ---- GATE: the d-shaped view must reproduce step165b's Q-shaped view exactly ---------------
    views_fn, _pads_fn = build_views(D)
    ref = views_fn(.5, 2, 1241)
    frames = build_aug_frames(D)
    gate = {}
    for kind, s, c_, a in frames:
        if kind != 'thin' or s != 1241:
            continue
        mine = a.filter(pl.col('top0')).sort(K)
        theirs = ref[c_].sort(K)
        assert mine.select(K).equals(theirs.select(K)), 'view row sets differ'
        cols = [c for c in POOLF4 if c in theirs.columns] + \
               [c for c in theirs.columns if c.startswith('context_')]
        assert len(cols) >= 140, cols
        mx = mine.select(cols).to_numpy().astype(float)
        tx = theirs.select(cols).to_numpy().astype(float)
        gate['thin_1241_copy' + str(c_)] = float(np.abs(mx - tx).max())
    print('view reconstruction gate (max abs diff vs step165b):', gate, flush=True)
    assert gate and max(gate.values()) == 0.0, 'd-shaped view does not reproduce step165b exactly'
    del ref, views_fn, _pads_fn

    # ---- bundles --------------------------------------------------------------------------------
    Xr, Cr = S165.matrices(D)
    save_bundle('real', (Xr, Cr, D['QIX'], D['fam'], d['relevance'].to_numpy(),
                         d['pair_id'].to_numpy(), d['table_id'].to_numpy()))
    del Xr, Cr
    dtop = d.with_columns(pl.Series('top0', np.isin(np.arange(d.height), D['QIX'])))
    common = list(dict.fromkeys(
        K + ['top0', 'chrono_frac', 'n_hands', 'behavior_family', 'relevance', 'table_id']
        + D['rcols'] + D['struct']))
    aug_names, shuf_names, mixed_names = [], [], []
    for i, (kind, s, c_, a) in enumerate(frames):
        arr = frame_to_arrays(D, a, expect_Q=True)
        save_bundle('aug' + str(i), arr)
        aug_names.append('aug' + str(i))
        # shuffled-label twin: relevance permuted WITHIN pair, everything else identical
        rel, pid = arr[4].copy(), arr[5]
        rng = np.random.default_rng([2011, i])
        for p in np.unique(pid):
            ix = np.flatnonzero(pid == p)
            rel[ix] = rng.permutation(rel[ix])
        assert rel.sum() == arr[4].sum(), 'shuffle changed the positive count'
        assert not np.array_equal(rel, arr[4]), 'shuffle was a no-op'
        save_bundle('shuf' + str(i), (arr[0], arr[1], arr[2], arr[3], rel, arr[5], arr[6]))
        shuf_names.append('shuf' + str(i))
        del arr
        # mixed: real rows + augmented rows, SORTED BY PAIR so Batch's block index stays valid
        mx = pl.concat([dtop.select(common), a.select(common)]).sort('pair_id', 'hand_id')
        save_bundle('mixed' + str(i), frame_to_arrays(D, mx, expect_Q=False))
        mixed_names.append('mixed' + str(i))
        del mx
        print('  bundles for %s/%d/%d done %ds' % (kind, s, c_, round(time.perf_counter() - t0)),
              flush=True)
    del frames, dtop

    # ---- folds ------------------------------------------------------------------------------------
    pairs, tables = d['pair_id'].to_numpy(), d['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    outer = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    jobs = []
    for arm in ARMS:
        for k_, (tr, te) in enumerate(outer):
            for s in S165.SEEDS_INNER_OUTER:
                jobs.append((('outer', k_), arm, s, upairs[tr], upairs[te]))
            for j, (itr, ite) in enumerate(
                    GroupKFold(n_splits=4).split(np.zeros(len(tr)), groups=ptab[tr])):
                for s in S165.SEEDS_INNER:
                    jobs.append((('inner', k_, j), arm, s, upairs[tr[itr]], upairs[tr[ite]]))
    print('%d fits over %d arms' % (len(jobs), len(ARMS)), flush=True)
    names = {'real': 'real', 'aug': aug_names, 'shuf': shuf_names, 'mixed': mixed_names}
    with Pool(14, initializer=pool_init2, initargs=(names,)) as pool:
        out = pool.map(fit_job2, jobs, chunksize=1)
    print('all fits done %ds' % round(time.perf_counter() - t0), flush=True)

    # ---- per-arm LP chain ---------------------------------------------------------------------------
    ac = D['ac']
    nq = len(Q)
    y = Q['relevance'].to_numpy().astype(int)
    qpairs = Q['pair_id'].to_numpy()
    colsf = {f: D['shared'] + (D['rcols'] if f != CI else []) + D['cs_cols'] + D['pf_cols']
             for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    Zf = {f: Q.select(colsf[f]).to_numpy().astype(float) for f in FAMS}
    scores, traces = {}, {}
    for arm in ARMS:
        AF_te, AF_in = {k_: [] for k_ in range(5)}, {k_: {} for k_ in range(5)}
        tr_all = []
        for tag, a_, s, rows, tq, p, S, trace in out:
            if a_ != arm:
                continue
            tr_all.append(trace)
            if tag[0] == 'outer':
                AF_te[tag[1]].append((tq, S165.aligned_features(D, rows, tq, S, p)))
            else:
                AF_in[tag[1]].setdefault(tag[2], []).append(
                    (tq, S165.aligned_features(D, rows, tq, S, p)))
        folds_q, EA, EB = [], {}, np.full((nq, len(S165.NN_COLS)), np.nan)
        for k_, (tr, te) in enumerate(outer):
            trq = np.flatnonzero(np.isin(qpairs, upairs[tr]))
            teq = np.flatnonzero(np.isin(qpairs, upairs[te]))
            ea = np.full((nq, len(S165.NN_COLS)), np.nan)
            for j, lst in AF_in[k_].items():
                ea[lst[0][0]] = np.mean([x for _, x in lst], axis=0)
            eb = np.full((nq, len(S165.NN_COLS)), np.nan)
            eb[AF_te[k_][0][0]] = np.mean([x for _, x in AF_te[k_]], axis=0)
            folds_q.append((trq, teq))
            EA[k_] = ea[trq]
            EB[teq] = eb[teq]
        assert np.isfinite(EB).all(), arm + ': EB incomplete'
        p = D['SS'].copy()
        for f in FAMS:
            z = Zf[f]
            for k_, (tr, te) in enumerate(folds_q):
                aix = tr[isf[f][tr]]
                m = lgb.LGBMClassifier(**ac.CLF, random_state=11).fit(
                    np.column_stack([z[aix], EA[k_][isf[f][tr]]]), y[aix])
                b = te[isf[f][te]]
                if len(b):
                    p[D['QIX'][b]] = m.predict_proba(np.column_stack([z[b], EB[b]]))[:, 1] + 10
        scores[arm] = p
        L = max(len(u) for u in tr_all)
        traces[arm] = [round(float(x), 4) for x in
                       np.mean(np.array([t for t in tr_all if len(t) == L]), axis=0)]
        print('  %-26s scored  %ds' % (arm, round(time.perf_counter() - t0)), flush=True)

    dec = D['dec']
    truths = {k_[0]: set(g['hand_id'])
              for k_, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    kf = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabq[q]) for q in qids])
    R = lambda x: round(float(x), 6)
    MM = {}
    for arm, s in scores.items():
        _, mm = summarize(kf.with_columns(pl.Series('score', dec(s))), truths, fams)
        MM[arm] = np.array([mm[q] for q in qids])

    res = {'registration_sha256': reg_sha, 'level': 'LP original pool (u=0), 162-query audit',
           'view_reconstruction_gate': gate, 'n_queries': len(qids),
           'map5_control': R(MM['control'].mean()),
           'phase_traces_mean_first_bestval_bestepoch': traces, 'arms': {}}
    for arm in ARMS[1:]:
        delta = MM[arm] - MM['control']
        res['arms'][arm] = {
            'map5': R(MM[arm].mean()), 'decoded_delta': R(delta.mean()),
            'half0': R(delta[hv == 0].mean()), 'half1': R(delta[hv == 1].mean()),
            'positive_on_both_halves': bool(delta[hv == 0].mean() > 0 and delta[hv == 1].mean() > 0),
            'per_family': {f: R(delta[[fams[q] == f for q in qids]].mean()) for f in FAMS}}
    A_ = res['arms']['A_WARMUP_AUG_THEN_REAL']['decoded_delta']
    C_ = res['arms']['C_SHUFFLED_AUG_WARMUP']['decoded_delta']
    res['THE_CONTROL'] = {
        'A_minus_C': R(A_ - C_),
        'curriculum_carries_information': bool(A_ - C_ > 0.004),
        'reading': ('if A and C move together the warm-up is weight initialisation or '
                    'regularisation and transfers nothing about collusion. The PAIR is the '
                    'evidence, not A.')}
    res['gates'] = {'control_reproduces_0690': {
        'here': R(MM['control'].mean()), 'tolerance': 0.002,
        'passed': bool(abs(MM['control'].mean() - 0.690) <= 0.002)}}
    res['bar'] = {'corrected_build_bar_decoded': 0.0073, 'clears_bar': bool(A_ >= 0.0073)}
    ok = res['arms']['A_WARMUP_AUG_THEN_REAL']['positive_on_both_halves']
    info = res['THE_CONTROL']['curriculum_carries_information']
    res['verdict'] = ('CURRICULUM_WORKS' if (ok and info and res['bar']['clears_bar'])
                      else 'REAL_BUT_UNDER_BAR' if (ok and info)
                      else 'CLOSED')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / outname).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
