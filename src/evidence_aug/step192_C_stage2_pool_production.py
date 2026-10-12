"""STEP192: production build of a stage-2 POOL at the frozen stage-1 incumbent.

The whole point of this build is that the expensive axis is not the one that pays. Production
scoring cost is polars view construction, which is per STAGE-1 member; every stage-2
configuration is scored inside the SAME view, on about 391 rows, so a pool of N configurations
costs one view pass plus N LightGBM predicts per view instead of N view passes.

So there is exactly ONE stage-1 member here -- the STEP167 incumbent, frozen, its full fits
reused from disk and never refitted -- and the ensemble lives entirely on the stage-2 axis.

Phases:
  fit    reconstruct the STEP167 out-of-fold network features, then fit one stage-2 bundle per
         configuration on them. Gated: the C00_incumbent bundle must predict IDENTICALLY to the
         stage-2 models STEP167 saved, which proves the reconstruction is exact.
  score  one pass over the 20 evaluation batches. Per view, score every configuration and
         accumulate. Emits v1_lp (the C00 column, for gating) and v1_lp_ens (the pool mean).
         Gated: v1_lp must equal STEP167's stored v1_lp for every batch, and ss/v1 must replay
         STEP128, exactly as STEP167 gates them.
  build  swap the decoded ensemble evidence into the base file. Identical to STEP188's build
         except that the ensemble arrives as one column rather than a mean over members.

Usage: PHASE REGSHA
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug import step167_C_lp_production as S167
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import check_protected

sha, write_json = S165.sha, S167.write_json
CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step192_20260916'
REG = OUT / 'registration192.json'
WORK = OUT / 'work'
INC_WORK = CV / 'outputs_step167_20260915/work'
NN = list(S165.NN_COLS)
FAMS = S165.FAMS
INC_CFG = 'C00_incumbent'


def read_reg(reg_sha):
    assert sha(REG) == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert reg['stage1'] == 'INC_REUSED_FROM_STEP167'
    assert reg['stage2_pool'][0] == INC_CFG, 'the incumbent configuration must lead the pool so the gate has something to check'
    return reg


def phase_fit(reg, reg_sha):
    """STEP167's phase_fit, but the stage-2 section is repeated once per configuration.

    The stage-1 full fits are NOT recomputed: they are loaded from STEP167's own bundle. Only the
    out-of-fold features E have to be rebuilt, because STEP167 saved models and not the frame.
    """
    from multiprocessing import Pool
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from evidence_aug import step172b_C_hp_search as S172B
    S108, _ = S167.modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    mp, fp = WORK / 'pool-models.joblib', WORK / 'fit.json'
    if fp.exists():
        r = json.loads(fp.read_text(encoding='utf-8'))
        assert r['registration_sha256'] == reg_sha and r['models_sha256'] == sha(mp)
        print('fit already on disk', flush=True)
        return r
    t0 = time.perf_counter()
    WORK.mkdir(parents=True, exist_ok=True)
    pool_cfgs = list(reg['stage2_pool'])
    allc = dict(S172B.generate_configs())
    for c in pool_cfgs:
        assert c in allc, c

    inc_bundle = joblib.load(INC_WORK / 'lp-models.joblib')
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    rcols, struct = S167.audit_columns()
    assert inc_bundle['rcols'] == rcols and inc_bundle['struct'] == struct, 'STEP167 column definition drifted'

    data, _, _, _, _ = CSP.development_frame()
    players = pl.read_csv(CSP.ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players)
    assert state.select(KEYS).equals(requested)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    pref, _ = S108.S98.prefix_for_keys(requested)
    keys0 = data.select(KEYS)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(keys0)
    data = data.sort(KEYS)

    X, fam = S167.xmatrix(data, rcols, struct, 'behavior_family')
    top = (data['s1_rank'] <= TOP).to_numpy()
    QIX = np.flatnonzero(top)
    Q = data[QIX]
    C = S167.cmatrix(Q, S108)
    pairs, tables = data['pair_id'].to_numpy(), data['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True)
    ptab = tables[pfirst]
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rel = data['relevance'].to_numpy()
    init_args = (X.astype(np.float32), C.astype(np.float32), QIX, fam, rel, pairs, tables)
    jobs = [(('oof', k), 'joint', s, rowmask(tr), rowmask(te))
            for k, (tr, te) in enumerate(folds) for s in S167.SEEDS_OOF]
    # STEP167's phase_fit issues TWO pool.map calls inside one Pool(12): the ten full fits first,
    # then the fifty out-of-fold fits. Running only the second one produced out-of-fold features
    # that missed STEP167's stage-2 models by 0.150 (registration192 amendment_1), so the sequence
    # is reproduced exactly here. The recomputed full fits are discarded for scoring -- the bundle
    # on disk is used for that -- but they are gated against STEP167's recorded best epochs, which
    # turns a reused artefact into a proven-identical one.
    with Pool(12, initializer=S165.pool_init, initargs=init_args) as pool:
        full = pool.map(S167.fit_full, S167.SEEDS_FULL, chunksize=1)
        oof = pool.map(S165.fit_job, jobs, chunksize=1)
    inc_fit = json.loads((INC_WORK / 'fit.json').read_text(encoding='utf-8'))
    got_ep = [tr[2] for _, _, _, _, tr in full]
    print(f'full fit best epochs: mine {got_ep}', flush=True)
    print(f'                      STEP167 {inc_fit["full_fit_best_epochs"]}', flush=True)
    gate_ep = got_ep == list(inc_fit['full_fit_best_epochs'])
    print(f'GATE stage-1 full fits reproduce STEP167: {gate_ep}', flush=True)
    Dd = {'d': data.select('pair_id'), 'fam': fam, 'QIX': QIX, 'Q': Q.select('pair_id')}
    acc = np.zeros((len(QIX), len(NN)))
    cnt = np.zeros(len(QIX))
    for tag, m, s, rows, tq, p, S, trace, sec in oof:
        acc[tq] += S165.aligned_features(Dd, rows, tq, S, p)
        cnt[tq] += 1
    assert (cnt == len(S167.SEEDS_OOF)).all()
    E = acc / cnt[:, None]
    Eall = np.full((data.height, len(NN)), np.nan)
    Eall[QIX] = E
    data = data.with_columns([pl.Series(c, Eall[:, j]) for j, c in enumerate(NN)])
    from sklearn.metrics import roc_auc_score as _auc
    my_auc = float(_auc(rel[QIX], E[:, 5]))
    print(f'out-of-fold features rebuilt  {round(time.perf_counter() - t0)}s', flush=True)
    print(f'  X {list(X.shape)} vs STEP167 {inc_fit["X_shape"]} | C {list(C.shape)} vs {inc_fit["C_shape"]} | oof {len(oof)} vs {inc_fit["oof_fits"]}', flush=True)
    print(f'  GATE oof AUC: mine {my_auc!r} vs STEP167 {inc_fit["oof_nn_p2_auc_top20"]!r} '
          f'-> delta {my_auc - inc_fit["oof_nn_p2_auc_top20"]:+.3e}', flush=True)

    cols = {f: bundle['cols'][f] + NN for f in FAMS}
    models, receipts = {}, []
    for ci, cname in enumerate(pool_cfgs):
        # The incumbent member is fitted with CSP.CLF verbatim -- literally what STEP167 ran -- so the
        # gate below is a true identity check. Every other member differs from it only in hyperparameters.
        # CSP.stage2_fit always bags SEEDS = (11, 12, 13); the audit's v1_full bags 5 seeds for stochastic
        # configurations and 1 for deterministic ones. That audit/production difference in bagging depth is
        # pre-existing (it already applies to the incumbent) and is declared in the registration.
        params = dict(CSP.CLF) if cname == INC_CFG else dict(allc[cname], n_jobs=CSP.CLF['n_jobs'])
        mk = {}
        for f in FAMS:
            mk[f], nq, npos = CSP.stage2_fit(data, cols[f], f, params)
            if ci == 0:
                receipts.append({'family': f, 'training_pairs': int(nq), 'training_listed': int(npos),
                                 'n_columns': len(cols[f])})
        models[cname] = mk
        print(f'  fitted stage-2 {ci + 1}/{len(pool_cfgs)} {cname}  {round(time.perf_counter() - t0)}s', flush=True)

    # ---- THE GATE: the incumbent configuration must reproduce STEP167's saved stage-2 exactly ----
    worst = 0.0
    for f in FAMS:
        t = data.filter((pl.col('behavior_family') == f) & (pl.col('s1_rank') <= TOP))
        a = CSP.stage2_predict(models[INC_CFG][f], t, cols[f])
        b = CSP.stage2_predict(inc_bundle['models'][f], t, cols[f])
        worst = max(worst, float(np.abs(a - b).max()))
    print(f'GATE incumbent stage-2 reproduces STEP167: worst abs {worst}', flush=True)
    assert worst == 0.0, 'the out-of-fold reconstruction is not exact; do NOT build on this'

    joblib.dump({'models': models, 'cols': cols, 'net': inc_bundle['net'],
                 'rcols': rcols, 'struct': struct, 'pool': pool_cfgs}, mp)
    y = rel[QIX]
    r = {'registration_sha256': reg_sha, 'models_sha256': sha(mp), 'pool': pool_cfgs,
         'n_configs': len(pool_cfgs), 'dev_rows': data.height, 'dev_pairs': int(len(upairs)),
         'dev_top20_rows': int(len(QIX)), 'lp_receipts': receipts,
         'incumbent_stage2_replay_worst_abs': worst,
         'stage1_net_reused_from': str(INC_WORK / 'lp-models.joblib'),
         'stage1_net_sha256': sha(INC_WORK / 'lp-models.joblib'),
         'oof_fits': len(oof), 'oof_nn_p2_auc_top20': float(roc_auc_score(y, E[:, 5])),
         'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(fp, r)
    print('fit ' + json.dumps({k: r[k] for k in ('n_configs', 'oof_nn_p2_auc_top20', 'seconds')}), flush=True)
    return r


def score_batch(index, path, bundle, pm, ctx, rec128, reg_sha):
    """STEP167.score_batch with the single LP bundle replaced by the pool.

    The view loop is unchanged; only the inner scoring is repeated per configuration, which is
    where the whole cost argument lives.
    """
    import gc
    S108, S128 = S167.modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    name = path.stem
    receipt = WORK / 'batches' / (name + '.json')
    sout = WORK / 'batches' / (name + '-pool-scores.parquet')
    if receipt.exists():
        r = json.loads(receipt.read_text(encoding='utf-8'))
        assert r['registration_sha256'] == reg_sha and r['files'][sout.name] == sha(sout)
        return r
    t0 = time.perf_counter()
    pool_cfgs = pm['pool']
    frames, src = S128.assemble(index, path, bundle, ctx)
    poolf = S128.CHRONO_FIELDS + ['context_' + c for c in src]
    requested = pl.concat([f for _, f in frames]).filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, ctx['players'])
    assert state.select(KEYS).equals(requested)
    prefix, swaps = S108.prefix_rows(requested, S108.pair_inputs(requested, ctx['players']))
    print(json.dumps({'batch': name, 'prepared_s': round(time.perf_counter() - t0, 1)}), flush=True)
    pieces, nviews, u0 = [], 0, 0
    for offset, frame in frames:
        frame = frame.join(state, on=KEYS, how='left', validate='1:1').sort(KEYS)
        keys0 = frame.select(KEYS)
        frame = frame.join(prefix, on=KEYS, how='left', validate='1:1', maintain_order='left')
        assert frame.select(KEYS).equals(keys0)
        top = (frame['s1_rank'] <= TOP).to_numpy()
        nnf = S167.net_features(pm, frame, top, S108)
        tf = frame.filter(pl.Series(top)).sort(KEYS)
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

        ss = score(tf, bundle['models'], bundle['cols'])
        lp = {c: score(tf, pm['models'][c], pm['cols']) for c in pool_cfgs}
        acc_ss = ss.copy()
        acc_lp = {c: lp[c].copy() for c in pool_cfgs}
        ref = tf.select(KEYS + poolf)
        for kind in ('thin', 'pad'):
            assert S128.view(frame, src, poolf, kind, 0., np.random.default_rng(0)).equals(ref), \
                f'u=0 {kind} view does not replay {name} part {offset}'
        u0 += 1
        rest = tf.drop(poolf)
        for k, kind in ((1, 'thin'), (2, 'pad')):
            for c in range(S128.NV):
                v = S128.view(frame, src, poolf, kind, S128.U, np.random.default_rng([128, k, index, offset, c]))
                t = rest.join(v, on=KEYS, how='left', validate='1:1', maintain_order='left')
                assert t.select(KEYS).equals(tf.select(KEYS)) and np.array_equal(t['s1_rank'].to_numpy(), tf['s1_rank'].to_numpy())
                acc_ss += score(t, bundle['models'], bundle['cols'])
                for cn in pool_cfgs:
                    acc_lp[cn] += score(t, pm['models'][cn], pm['cols'])
                nviews += 1
                del v, t
        nv = 2 * S128.NV + 1
        v1_lp_inc = acc_lp[INC_CFG] / nv
        v1_ens = np.mean([acc_lp[cn] / nv for cn in pool_cfgs], axis=0)
        pieces.append(tf.select(KEYS).with_columns(
            pl.Series('ss', ss), pl.Series('v1', acc_ss / nv),
            pl.Series('v1_lp', v1_lp_inc), pl.Series('v1_lp_ens', v1_ens)))
        print(json.dumps({'batch': name, 'part': offset, 'top_rows': tf.height, 'views': nviews,
                          's': round(time.perf_counter() - t0, 1)}), flush=True)
        del frame, tf, rest
        gc.collect()

    bf = pl.concat(pieces).sort(KEYS)
    top_file = S167.V1 / 'batches' / (name + '-top20-scores.parquet')
    assert sha(top_file) == rec128['files'][top_file.name]
    saved = pl.read_parquet(top_file).sort(KEYS)
    assert bf.select(KEYS).equals(saved.select(KEYS)) and bf.select(KEYS).equals(requested)
    assert np.array_equal(bf['ss'].to_numpy(), saved['ss'].to_numpy()), 'S* original pool does not replay STEP128'
    assert np.array_equal(bf['v1'].to_numpy(), saved['v1'].to_numpy()), 'v1 does not replay STEP128'
    # ---- THE END-TO-END GATE: the incumbent column must equal STEP167's own production output ----
    inc_file = INC_WORK / 'batches' / (name + '-lp-scores.parquet')
    inc_rec = json.loads((INC_WORK / 'batches' / (name + '.json')).read_text(encoding='utf-8'))
    assert inc_rec['files'][inc_file.name] == sha(inc_file), 'STEP167 batch receipt does not match its file'
    inc = pl.read_parquet(inc_file).sort(KEYS)
    assert inc.select(KEYS).equals(bf.select(KEYS))
    gate_inc = float(np.abs(inc['v1_lp'].to_numpy() - bf['v1_lp'].to_numpy()).max())
    assert gate_inc == 0.0, f'incumbent v1_lp does not replay STEP167 in {name}: {gate_inc}'
    bf.write_parquet(sout)
    dv = bf['v1_lp_ens'].to_numpy() - bf['v1_lp'].to_numpy()
    r = {'registration_sha256': reg_sha, 'files': {sout.name: sha(sout)}, 'pairs': bf['pair_id'].n_unique(),
         'top20_rows': bf.height, 'views_scored': nviews, 'n_configs': len(pool_cfgs),
         'u0_view_replay_exact_parts': u0, 'step128_ss_v1_replay_exact': True,
         'step167_v1_lp_replay_max_abs': gate_inc, 'prefix_swap_checks': swaps,
         'runner_sha256': sha(Path(__file__)),
         'ens_minus_inc_mean_abs': float(np.abs(dv).mean()),
         'ens_inc_corr': float(np.corrcoef(bf['v1_lp_ens'].to_numpy(), bf['v1_lp'].to_numpy())[0, 1]),
         'seconds': round(time.perf_counter() - t0, 1)}
    write_json(receipt, r)
    print(json.dumps({k: r[k] for k in ('pairs', 'views_scored', 'n_configs', 'step167_v1_lp_replay_max_abs',
                                        'ens_minus_inc_mean_abs', 'ens_inc_corr', 'seconds')}), flush=True)
    del frames, state, prefix, bf
    gc.collect()
    return r


def phase_score(reg, reg_sha):
    """STEP167.phase_score with the single LP bundle replaced by the pool. ctx is built identically."""
    S108, _ = S167.modules()
    CSP = S108.CSP
    fit = json.loads((WORK / 'fit.json').read_text(encoding='utf-8'))
    assert fit['registration_sha256'] == reg_sha and fit['models_sha256'] == sha(WORK / 'pool-models.joblib')
    pm = joblib.load(WORK / 'pool-models.joblib')
    assert pm['pool'] == list(reg['stage2_pool'])
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    (WORK / 'batches').mkdir(parents=True, exist_ok=True)
    rec128 = json.loads((S167.V1 / 'receipt.json').read_text(encoding='utf-8'))
    selected = pl.read_parquet(CSP.ROOT / 'reports/replay-v13/selected-pairs.parquet')
    assert selected.height == 16060
    ctx = {'source': pl.read_csv(CSP.BEST, infer_schema=False), 'selected': selected,
           'added': selected.join(pl.read_parquet(CSP.ROOT / 'reports/evidence-public-calibration/selected-pairs.parquet').select('pair_id'), on='pair_id', how='anti'),
           'players': pl.read_csv(CSP.ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']),
           'chrono': joblib.load(CSP.STAGE1),
           'structural_receipt': json.loads((CSP.FS2 / 'eval_structural/receipt.json').read_text()),
           'caf': joblib.load(CSP.CONDITIONAL / 'evidence-model.joblib')['conditional_features']}
    batches = sorted((CSP.ROOT / 'outputs_v2/hands_evaluation').glob('batch_*.parquet'))
    assert len(batches) == 20
    only = [int(x) for x in os.environ.get('STEP192_BATCHES', '').split(',') if x]
    for i, p in enumerate(batches):
        if only and i not in only:
            continue
        rj = S167.V1 / 'batches' / (p.stem + '.json')
        assert sha(rj) == rec128['batch_receipts'][p.stem]
        score_batch(i, p, bundle, pm, ctx, json.loads(rj.read_text(encoding='utf-8')), reg_sha)


def phase_build(reg, reg_sha):
    """STEP188's build, with the ensemble arriving as one column instead of a mean over members."""
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    from evidence_aug import step104_C_typed_decoder_build as S104
    from validation import validate_submission
    t0 = time.perf_counter()
    prot = check_protected()
    EV = list(CSP.EVIDENCE_COLUMNS)
    basep = POKER / reg['base']['path']
    assert sha(basep) == reg['base']['sha256'], 'base file changed'
    base = pl.read_csv(basep, infer_schema=False)
    ci = base.filter(pl.col('predicted_behavior') == 'coordinated_isolation').select('pair_id')
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())
    pmap = {p: (a, b) for p, a, b in pl.read_csv(CSP.ROOT / 'data/evaluation_pairs.csv',
                                                 columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    news, recs = [], []
    for i in range(20):
        name, tb = 'batch_%03d' % i, time.perf_counter()
        rj = WORK / 'batches' / (name + '.json')
        sc = WORK / 'batches' / (name + '-pool-scores.parquet')
        r = json.loads(rj.read_text(encoding='utf-8'))
        assert r['files'][sc.name] == sha(sc), name
        assert r['step167_v1_lp_replay_max_abs'] == 0.0, name
        ref = pl.read_parquet(sc).sort(KEYS)
        bf = ref.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'),
                        (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1')) \
                .with_columns(pl.Series('v1_ens', ref['v1_lp_ens'].to_numpy() + 10))
        sel = bf.select('pair_id').unique().sort('pair_id')
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        cif = fc.select('pair_id').join(ci.with_columns(pl.lit(True).alias('f')), on='pair_id',
                                        how='left', maintain_order='left')['f'].fill_null(False).to_numpy()
        typed = t1 & ~cif
        inc = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), typed)))
        chk = base.join(sel, on='pair_id', how='semi').select(inc.columns).sort('pair_id')
        assert inc.equals(chk), 'v1 replay of the base evidence failed in ' + name
        new = S108.expand(CSP.rank(bf, S104.decode(fc.with_columns(pl.Series('score', bf['v1_ens'].to_numpy())), typed)))
        assert new.select('pair_id').equals(inc.select('pair_id'))
        news.append(new)
        sets = [len(set(u[1:]) - set(o[1:])) for u, o in zip(new.rows(), inc.rows())]
        fams = [fam[x[0]] for x in new.rows()]
        recs.append({'batch': name, 'selected_pairs': new.height, 'T1_rows_used': int(typed.sum()),
                     'lists_changed': int(sum(u != o for u, o in zip(new.rows(), inc.rows()))),
                     'top5_set_change_mean': round(float(np.mean(sets)), 4),
                     'top5_set_change_by_family': {f: round(float(np.mean([s for s, g in zip(sets, fams) if g == f])), 4)
                                                   for f in sorted(set(fams))},
                     's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    repl = pl.concat(news).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060
    rest = base.join(repl.select('pair_id'), on='pair_id', how='anti')
    swap = base.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(repl, on='pair_id', validate='1:1')
    result = base.select('pair_id').join(pl.concat([rest, swap.select(base.columns)]), on='pair_id',
                                         validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(
        base.select('pair_id', 'risk_score', 'predicted_behavior'))
    csv = OUT / reg['output_name']
    assert not csv.exists(), 'preserve a built file'
    result.write_csv(csv)
    A, B = basep.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
    assert len(A) == len(B)
    diff = [k for k in range(len(A)) if A[k] != B[k]]
    assert {A[k].split(b',')[0].decode() for k in diff} <= set(repl['pair_id'])
    for k in diff:
        assert A[k].split(b',')[:3] == B[k].split(b',')[:3], 'risk or behaviour changed'
    val = validate_submission(csv, CSP.ROOT / 'data/evaluation_pairs.csv', CSP.ROOT / 'data/hands.parquet',
                              CSP.ROOT / 'data/seats.parquet', CSP.ROOT / 'data/sample_submission.csv')
    assert not val.errors, val
    assert check_protected() == prot
    fit = json.loads((WORK / 'fit.json').read_text(encoding='utf-8'))
    rec = {'registration_sha256': reg_sha, 'stage1': 'INC reused from STEP167, never refitted',
           'stage2_pool': fit['pool'], 'n_configs': fit['n_configs'],
           'combiner': 'SCORE_mean over the stage-2 pool inside one view pass, decoded once',
           'output': str(csv), 'output_sha256': sha(csv), 'base_sha256': sha(basep),
           'rows_changed_vs_base': len(diff), 'validation': str(val)[:300], 'batches': recs,
           'lists_changed': int(sum(r['lists_changed'] for r in recs)),
           'top5_set_change_mean': float(np.average([r['top5_set_change_mean'] for r in recs],
                                                    weights=[r['selected_pairs'] for r in recs])),
           'protected_unchanged': True, 'no_submission_actions': True,
           'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'receipt_step192.json', rec)
    print('DONE ' + json.dumps({k: rec[k] for k in ('output_sha256', 'rows_changed_vs_base', 'lists_changed',
                                                    'top5_set_change_mean', 'seconds')}), flush=True)


def main():
    phase, reg_sha = sys.argv[1], sys.argv[2]
    reg = read_reg(reg_sha)
    OUT.mkdir(parents=True, exist_ok=True)
    if phase == 'fit':
        phase_fit(reg, reg_sha)
    elif phase == 'score':
        phase_score(reg, reg_sha)
    elif phase == 'build':
        phase_build(reg, reg_sha)
    else:
        raise SystemExit('unknown phase ' + phase)


if __name__ == '__main__':
    main()
