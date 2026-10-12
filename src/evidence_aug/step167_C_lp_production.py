"""Step 167 (Claude): exploratory production of LP evidence (STEP165/165b/165c) on the .90563 file. Never uploads.
LP = the step-108 S* per-family LightGBM stage 2 (same columns, CLF, seeds 11-13) plus the 7 aligned stage-1 outputs of the STEP165 joint
network (step165_e2e_mlp CONFIG). Registered before any fit: outputs_step167_20260915/prediction_step167.json.
Phases (resumable from receipts):
  fit    development frame (CSP.development_frame + contestable state + prefix strength, as step 108/149); gate: S* refit equals the saved
         step-108 models; network: 10 full fits (seeds 16500-16509) on every development list, and 5-fold table OOF x 10 seeds (16600-16609)
         for the LightGBM training rows; LP family models on those OOF outputs.
  score  per evaluation batch (STEP167_ONLY=i,j,...): step-128 frame assembly; network outputs on the original pool (mean of the 10 full fits'
         aligned outputs), held fixed in every view; S* and LP scored on the original pool and the 40 step-128 views; gate: S* original and
         v1 equal the STEP128 saved top-20 scores exactly.
  build  typed decode (T1 off for predicted coordinated isolation, the STEP148 rule); gate: decode(v1) replays the base evidence lists of all
         16,060 selected pairs; CSVs on the STEP163 file (.90563) and on the STEP169 file (same evidence swap). Risk and behaviour byte-identical.
Usage: step167_C_lp_production.py fit|score|build REGSHA"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import gc, json, time
from pathlib import Path

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import joblib
import numpy as np
import polars as pl
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from evidence_aug.paths import sha, check_protected, write_json
from evidence_aug.frame import FAMILIES
from evidence_aug import step165_C_e2e_audit as S165

OUT = POKER / 'Claude Version/outputs_step167_20260915'
WORK = OUT / 'work'
PREREG = OUT / 'prediction_step167.json'
V1 = POKER / 'Claude Version/outputs_v1_view_averaged_exploratory_20260913'
NN = list(S165.NN_COLS)
FAMS = S165.FAMS
SEEDS_FULL = list(S165.SEEDS_INNER_OUTER)
SEEDS_OOF = list(S165.SEEDS_INNER)
POOLF = ['chrono_rank', 'chrono_frac', 'n_hands', 's1', 's1_rank']


def modules():
    from evidence_aug import step108_C_stacked_decoder_production as S108
    from evidence_aug import step128_C_v1_view_averaged_production as S128
    return S108, S128


def audit_columns():
    import audit_common as ac
    d, shared, rcols, _ = ac.load_current()
    struct = [c for c in shared if c not in ('chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank')]
    return rcols, struct


def xmatrix(frame, rcols, struct, famcol):
    """STEP165 matrices(): rcols (nulls/NaN -> -999), struct, chrono_frac, log n_hands, family one-hot."""
    R = frame.select(rcols).fill_null(-999.).to_numpy().astype(np.float64)
    R[np.isnan(R)] = -999.
    fam = frame[famcol].to_numpy()
    X = np.column_stack([R, frame.select(struct).to_numpy().astype(np.float64), frame['chrono_frac'].to_numpy(), np.log(frame['n_hands'].to_numpy())] +
                        [(fam == f).astype(float) for f in FAMS])
    assert np.isfinite(X).all()
    return X, fam


def cmatrix(top, S108):
    C = np.column_stack([top.select(S108.STATE + S108.RS).to_numpy().astype(np.float64), top['chrono_frac'].to_numpy()])
    assert np.isfinite(C).all()
    return C


def stats(A):
    """S165.standardize statistics over all rows (same indexing, so the same floating-point result)."""
    rows = np.arange(len(A))
    mu, sd = A[rows].mean(0), A[rows].std(0)
    return mu, np.where(sd > 1e-9, sd, 1.)


def apply(A, st):
    return np.clip((A - st[0]) / st[1], -6, 6)


def fit_full(job):
    """S165.fit_job for all development rows, returning parameters and standardisation statistics instead of predictions."""
    from evidence_aug import step165_e2e_mlp as M
    seed = job
    G = S165.G
    train_mask = np.ones(len(G['X']), bool)
    tabs = np.unique(G['tab'])
    vt = np.random.default_rng([seed, 7]).choice(tabs, size=max(1, int(round(len(tabs) * M.CONFIG['val_share_tables']))), replace=False)
    val_mask = np.isin(G['tab'], vt)
    xs, cs = stats(G['X']), stats(G['C'])
    assert np.array_equal(apply(G['X'], xs), S165.standardize(G['X'], np.flatnonzero(train_mask)))
    Xs, Cs = apply(G['X'], xs), apply(G['C'], cs)
    Btr, _, _ = S165.make_batch(M, G, Xs, Cs, ~val_mask)
    Bv, _, _ = S165.make_batch(M, G, Xs, Cs, val_mask)
    P, trace = M.train(Btr, 'joint', seed, Bv)
    return seed, P, xs, cs, [float(x) for x in trace]


def check_reg(reg_sha):
    assert sha(PREREG) == reg_sha, 'registration changed'
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    for k, v in reg['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], k
    return reg


def phase_fit(reg_sha):
    S108, _ = modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    mp, fp = WORK / 'lp-models.joblib', WORK / 'fit.json'
    if fp.exists():
        r = json.loads(fp.read_text(encoding='utf-8')); assert r['registration_sha256'] == reg_sha and r['models_sha256'] == sha(mp)
        return r
    t0 = time.perf_counter()
    assert sha(Path(__file__)) == json.loads(PREREG.read_text(encoding='utf-8'))['code']['runner_sha256'], 'runner changed before the fit'
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    rcols, struct = audit_columns()
    for f in FAMS:
        cols = bundle['cols'][f]
        assert cols[-61:] == S108.STATE + S108.RS
        assert set(cols[:-61]) == set(struct + POOLF + (rcols if f != CSP.CI else [])), f
    data, _, _, _, _ = CSP.development_frame()
    players = pl.read_csv(CSP.ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players); assert state.select(KEYS).equals(requested)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    pref, _ = S108.S98.prefix_for_keys(requested)
    keys0 = data.select(KEYS)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left'); assert data.select(KEYS).equals(keys0)
    data = data.sort(KEYS)
    for f in FAMS:
        top = data.filter((pl.col('behavior_family') == f) & (pl.col('s1_rank') <= TOP))
        refit, _, _ = CSP.stage2_fit(data, bundle['cols'][f], f, CSP.CLF)
        np.testing.assert_array_equal(CSP.stage2_predict(refit, top, bundle['cols'][f]), CSP.stage2_predict(bundle['models'][f], top, bundle['cols'][f]))
    print('S* refit equals the saved step-108 models; development rows', data.height, 'pairs', data['pair_id'].n_unique(), round(time.perf_counter() - t0, 1), flush=True)
    X, fam = xmatrix(data, rcols, struct, 'behavior_family')
    top = (data['s1_rank'] <= TOP).to_numpy(); QIX = np.flatnonzero(top)
    Q = data[QIX]
    C = cmatrix(Q, S108)
    pairs, tables = data['pair_id'].to_numpy(), data['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True); ptab = tables[pfirst]
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rel = data['relevance'].to_numpy()
    init_args = (X.astype(np.float32), C.astype(np.float32), QIX, fam, rel, pairs, tables)
    jobs = [(('oof', k), 'joint', s, rowmask(tr), rowmask(te)) for k, (tr, te) in enumerate(folds) for s in SEEDS_OOF]
    with Pool(12, initializer=S165.pool_init, initargs=init_args) as pool:
        full = pool.map(fit_full, SEEDS_FULL, chunksize=1)
        oof = pool.map(S165.fit_job, jobs, chunksize=1)
    Dd = {'d': data.select('pair_id'), 'fam': fam, 'QIX': QIX, 'Q': Q.select('pair_id')}
    acc = np.zeros((len(QIX), len(NN))); cnt = np.zeros(len(QIX))
    for tag, m, s, rows, tq, p, S, trace, sec in oof:
        acc[tq] += S165.aligned_features(Dd, rows, tq, S, p); cnt[tq] += 1
    assert (cnt == len(SEEDS_OOF)).all()
    E = acc / cnt[:, None]
    Eall = np.full((data.height, len(NN)), np.nan); Eall[QIX] = E
    data = data.with_columns([pl.Series(c, Eall[:, j]) for j, c in enumerate(NN)])
    models, cols, receipts = {}, {}, []
    for f in FAMS:
        cols[f] = bundle['cols'][f] + NN
        models[f], nq, npos = CSP.stage2_fit(data, cols[f], f, CSP.CLF)
        receipts.append({'family': f, 'training_pairs': int(nq), 'training_listed': int(npos), 'n_columns': len(cols[f])})
    net = [{'seed': s, 'P': P, 'xstats': xs, 'cstats': cs, 'trace': tr} for s, P, xs, cs, tr in full]
    joblib.dump({'models': models, 'cols': cols, 'net': net, 'rcols': rcols, 'struct': struct}, mp)
    restored = joblib.load(mp)
    for f in FAMS:
        t = data.filter((pl.col('behavior_family') == f) & (pl.col('s1_rank') <= TOP))
        np.testing.assert_array_equal(CSP.stage2_predict(models[f], t, cols[f]), CSP.stage2_predict(restored['models'][f], t, cols[f]))
    y = rel[QIX]
    r = {'registration_sha256': reg_sha, 'models_sha256': sha(mp), 'dev_rows': data.height, 'dev_pairs': int(len(upairs)), 'dev_top20_rows': int(len(QIX)),
         'X_shape': list(X.shape), 'C_shape': list(C.shape), 'sstar_refit_equals_step108': True, 'lp_receipts': receipts,
         'full_fit_best_epochs': [tr[2] for _, _, _, _, tr in full], 'oof_fits': len(oof),
         'oof_nn_p2_auc_top20': float(roc_auc_score(y, E[:, 5])),
         'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(fp, r)
    print('fit', json.dumps(r), flush=True)
    return r


def net_features(lpm, frame, top, S108):
    """Mean over the full fits of the STEP165 aligned outputs for the top-20 rows of one assembled part frame (sorted by keys)."""
    from evidence_aug import step165_e2e_mlp as M
    X, fam = xmatrix(frame, lpm['rcols'], lpm['struct'], 'predicted_behavior')
    QIX = np.flatnonzero(top)
    tfr = frame[QIX]
    C = cmatrix(tfr, S108)
    pid = frame['pair_id'].to_numpy()
    famidx = np.array([FAMS.index(f) for f in fam])
    D = {'d': frame.select('pair_id'), 'fam': fam, 'QIX': QIX, 'Q': tfr.select('pair_id')}
    acc = np.zeros((len(QIX), len(NN)))
    for n in lpm['net']:
        B = M.Batch(apply(X, n['xstats']), famidx, np.zeros(len(pid)), pid, QIX, apply(C, n['cstats']))
        p, S = M.predict(n['P'], B)
        acc += S165.aligned_features(D, np.arange(len(pid)), np.arange(len(QIX)), S.astype(np.float64), p.astype(np.float64))
    out = acc / len(lpm['net'])
    assert np.isfinite(out).all()
    return out


def score_batch(index, path, bundle, lpm, ctx, rec128, reg_sha):
    S108, S128 = modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    name = path.stem
    receipt, sout = WORK / 'batches' / (name + '.json'), WORK / 'batches' / (name + '-lp-scores.parquet')
    if receipt.exists():
        r = json.loads(receipt.read_text(encoding='utf-8')); assert r['registration_sha256'] == reg_sha and r['files'][sout.name] == sha(sout)
        return r
    t0 = time.perf_counter()
    frames, src = S128.assemble(index, path, bundle, ctx)
    pool = S128.CHRONO_FIELDS + ['context_' + c for c in src]
    requested = pl.concat([f for _, f in frames]).filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, ctx['players']); assert state.select(KEYS).equals(requested)
    prefix, swaps = S108.prefix_rows(requested, S108.pair_inputs(requested, ctx['players']))
    print(json.dumps({'batch': name, 'prepared_s': round(time.perf_counter() - t0, 1)}), flush=True)
    pieces, nviews, u0 = [], 0, 0
    for offset, frame in frames:
        frame = frame.join(state, on=KEYS, how='left', validate='1:1').sort(KEYS)
        keys0 = frame.select(KEYS)
        frame = frame.join(prefix, on=KEYS, how='left', validate='1:1', maintain_order='left'); assert frame.select(KEYS).equals(keys0)
        top = (frame['s1_rank'] <= TOP).to_numpy()
        nnf = net_features(lpm, frame, top, S108)
        tf = frame.filter(pl.Series(top)).sort(KEYS)
        assert tf.select(KEYS).equals(frame.filter(pl.Series(top)).select(KEYS))
        tf = tf.with_columns([pl.Series(c, nnf[:, j]) for j, c in enumerate(NN)])
        assert tf.select(S108.STATE + S108.RS).null_count().sum_horizontal().item() == 0
        fams = tf['predicted_behavior'].to_numpy()

        def score(t, models, cols):
            o = np.full(t.height, np.nan)
            for f in FAMILIES:
                m = fams == f
                if m.any():
                    o[m] = CSP.stage2_predict(models[f], t.filter(pl.Series(m)), cols[f])
            assert np.isfinite(o).all()
            return o
        ss, lp = score(tf, bundle['models'], bundle['cols']), score(tf, lpm['models'], lpm['cols'])
        acc_ss, acc_lp = ss.copy(), lp.copy()
        ref = tf.select(KEYS + pool)
        for kind in ('thin', 'pad'):
            assert S128.view(frame, src, pool, kind, 0., np.random.default_rng(0)).equals(ref), f'u=0 {kind} view does not replay {name} part {offset}'
        u0 += 1
        rest = tf.drop(pool)
        for k, kind in ((1, 'thin'), (2, 'pad')):
            for c in range(S128.NV):
                v = S128.view(frame, src, pool, kind, S128.U, np.random.default_rng([128, k, index, offset, c]))
                t = rest.join(v, on=KEYS, how='left', validate='1:1', maintain_order='left')
                assert t.select(KEYS).equals(tf.select(KEYS)) and np.array_equal(t['s1_rank'].to_numpy(), tf['s1_rank'].to_numpy())
                acc_ss += score(t, bundle['models'], bundle['cols']); acc_lp += score(t, lpm['models'], lpm['cols']); nviews += 1
                del v, t
        pieces.append(tf.select(KEYS).with_columns(pl.Series('ss', ss), pl.Series('v1', acc_ss / (2 * S128.NV + 1)),
                                                   pl.Series('ss_lp', lp), pl.Series('v1_lp', acc_lp / (2 * S128.NV + 1)),
                                                   *[pl.Series(c, nnf[:, j]) for j, c in enumerate(NN)]))
        print(json.dumps({'batch': name, 'part': offset, 'top_rows': tf.height, 'views': nviews, 's': round(time.perf_counter() - t0, 1)}), flush=True)
        del frame, tf, rest; gc.collect()
    bf = pl.concat(pieces).sort(KEYS)
    top_file = V1 / 'batches' / (name + '-top20-scores.parquet')
    assert sha(top_file) == rec128['files'][top_file.name]
    saved = pl.read_parquet(top_file).sort(KEYS)
    assert bf.select(KEYS).equals(saved.select(KEYS)) and bf.select(KEYS).equals(requested)
    assert np.array_equal(bf['ss'].to_numpy(), saved['ss'].to_numpy()), 'S* original pool does not replay STEP128'
    assert np.array_equal(bf['v1'].to_numpy(), saved['v1'].to_numpy()), 'v1 does not replay STEP128'
    bf.write_parquet(sout)
    dv = bf['v1_lp'].to_numpy() - bf['v1'].to_numpy()
    r = {'registration_sha256': reg_sha, 'files': {sout.name: sha(sout)}, 'pairs': bf['pair_id'].n_unique(), 'top20_rows': bf.height, 'views_scored': nviews,
         'u0_view_replay_exact_parts': u0, 'step128_ss_v1_replay_exact': True, 'prefix_swap_checks': swaps,
         'runner_sha256': sha(Path(__file__)), 'v1_lp_mean_abs_change': float(np.abs(dv).mean()), 'v1_lp_corr': float(np.corrcoef(bf['v1_lp'].to_numpy(), bf['v1'].to_numpy())[0, 1]),
         'seconds': round(time.perf_counter() - t0, 1)}
    write_json(receipt, r)
    print(json.dumps({k: r[k] for k in ('pairs', 'top20_rows', 'views_scored', 'v1_lp_mean_abs_change', 'v1_lp_corr', 'seconds')}), flush=True)
    del frames, state, prefix, bf; gc.collect()
    return r


def phase_score(reg_sha):
    S108, S128 = modules()
    CSP = S108.CSP
    fit = json.loads((WORK / 'fit.json').read_text(encoding='utf-8'))
    assert fit['registration_sha256'] == reg_sha and fit['models_sha256'] == sha(WORK / 'lp-models.joblib')
    lpm = joblib.load(WORK / 'lp-models.joblib')
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    rec128 = json.loads((V1 / 'receipt.json').read_text(encoding='utf-8'))
    selected = pl.read_parquet(CSP.ROOT / 'reports/replay-v13/selected-pairs.parquet'); assert selected.height == 16060
    ctx = {'source': pl.read_csv(CSP.BEST, infer_schema=False), 'selected': selected,
           'added': selected.join(pl.read_parquet(CSP.ROOT / 'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id'), on='pair_id', how='anti'),
           'players': pl.read_csv(CSP.ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']), 'chrono': joblib.load(CSP.STAGE1),
           'structural_receipt': json.loads((CSP.FS2 / 'eval_structural/receipt.json').read_text()),
           'caf': joblib.load(CSP.CONDITIONAL / 'evidence-model.joblib')['conditional_features']}
    batches = sorted((CSP.ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet')); assert len(batches) == 20
    only = [int(x) for x in os.environ.get('STEP167_ONLY', '').split(',') if x]
    for i, p in enumerate(batches):
        if only and i not in only:
            continue
        rj = V1 / 'batches' / (p.stem + '.json'); assert sha(rj) == rec128['batch_receipts'][p.stem]
        score_batch(i, p, bundle, lpm, ctx, json.loads(rj.read_text(encoding='utf-8')), reg_sha)


def phase_build(reg_sha, reg):
    S108, _ = modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    from evidence_aug import step104_C_typed_decoder_build as S104
    from validation import validate_submission
    t0 = time.perf_counter()
    prot = check_protected()
    EV = list(CSP.EVIDENCE_COLUMNS)
    base = pl.read_csv(POKER / reg['inputs']['base_step163']['path'], infer_schema=False)
    alt_path = POKER / reg['inputs']['base_step169']['path']
    alt = pl.read_csv(alt_path, infer_schema=False)
    assert alt.select(['pair_id', 'predicted_behavior'] + EV).equals(base.select(['pair_id', 'predicted_behavior'] + EV)), 'STEP169 evidence differs from STEP163'
    ci = base.filter(pl.col('predicted_behavior') == 'coordinated_isolation').select('pair_id')
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())
    pmap = {p: (a, b) for p, a, b in pl.read_csv(CSP.ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    news, recs = [], []
    for i in range(20):
        name, tb = 'batch_%03d' % i, time.perf_counter()
        rj = WORK / 'batches' / (name + '.json'); sc = WORK / 'batches' / (name + '-lp-scores.parquet')
        r = json.loads(rj.read_text(encoding='utf-8')); assert r['registration_sha256'] == reg_sha and r['files'][sc.name] == sha(sc)
        top = pl.read_parquet(sc).sort(KEYS)
        sel = top.select('pair_id').unique().sort('pair_id')
        bf = top.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'), (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1'), (pl.col('v1_lp') + 10).alias('v1_lp'))
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        cif = fc.select('pair_id').join(ci.with_columns(pl.lit(True).alias('f')), on='pair_id', how='left', maintain_order='left')['f'].fill_null(False).to_numpy()
        typed = t1 & ~cif
        inc = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), typed)))
        chk = base.join(sel, on='pair_id', how='semi').select(inc.columns).sort('pair_id')
        assert inc.equals(chk), 'v1 replay of the base evidence failed in ' + name
        new = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['v1_lp'].to_numpy())), typed)))
        assert new.select('pair_id').equals(inc.select('pair_id'))
        new.write_parquet(WORK / 'batches' / (name + '-lp-decoded.parquet'))
        news.append(new)
        sets = [len(set(u[1:]) - set(o[1:])) for u, o in zip(new.rows(), inc.rows())]
        fams = [fam[x[0]] for x in new.rows()]
        recs.append({'batch': name, 'selected_pairs': new.height, 'T1_rows_used': int(typed.sum()), 'lists_changed': int(sum(u != o for u, o in zip(new.rows(), inc.rows()))),
                     'top5_set_change_mean': round(float(np.mean(sets)), 4),
                     'top5_set_change_by_family': {f: round(float(np.mean([s for s, g in zip(sets, fams) if g == f])), 4) for f in sorted(set(fams))},
                     's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    repl = pl.concat(news).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060
    outs = {}
    for tag, b, bpath in (('on_step163', base, POKER / reg['inputs']['base_step163']['path']), ('on_step169', alt, alt_path)):
        rest = b.join(repl.select('pair_id'), on='pair_id', how='anti')
        swap = b.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(repl, on='pair_id', validate='1:1')
        result = b.select('pair_id').join(pl.concat([rest, swap.select(b.columns)]), on='pair_id', validate='1:1', maintain_order='left')
        assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(b.select('pair_id', 'risk_score', 'predicted_behavior'))
        csv = OUT / f'submission_step167_lp_{tag}.csv'
        assert not csv.exists()
        result.write_csv(csv)
        A, B = bpath.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
        diff = [k for k in range(len(A)) if A[k] != B[k]] if len(A) == len(B) else None
        assert diff is not None and {A[k].split(b',')[0].decode() for k in diff} <= set(repl['pair_id'])
        for k in diff:
            assert A[k].split(b',')[:3] == B[k].split(b',')[:3]
        val = validate_submission(csv, CSP.ROOT / 'data/evaluation_pairs.csv', CSP.ROOT / 'data/hands.parquet', CSP.ROOT / 'data/seats.parquet', CSP.ROOT / 'data/sample_submission.csv')
        assert not val.errors, val
        outs[tag] = {'output': str(csv), 'output_sha256': sha(csv), 'base_sha256': sha(bpath), 'rows_changed_vs_base': len(diff), 'validation': str(val)[:300]}
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'outputs': outs, 'batches': recs, 'lists_changed': int(sum(r['lists_changed'] for r in recs)),
           'top5_set_change_mean': float(np.average([r['top5_set_change_mean'] for r in recs], weights=[r['selected_pairs'] for r in recs])),
           'protected_unchanged': True, 'no_submission_actions': True, 'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'receipt_step167.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('DONE', json.dumps({'outputs': outs, 'lists_changed': rec['lists_changed'], 'top5_set_change_mean': rec['top5_set_change_mean']}), flush=True)


def main():
    phase, reg_sha = sys.argv[1], sys.argv[2]
    reg = check_reg(reg_sha)
    (WORK / 'batches').mkdir(parents=True, exist_ok=True)
    if phase == 'fit':
        phase_fit(reg_sha)
    elif phase == 'score':
        phase_score(reg_sha)
    elif phase == 'build':
        phase_build(reg_sha, reg)


if __name__ == '__main__':
    main()
