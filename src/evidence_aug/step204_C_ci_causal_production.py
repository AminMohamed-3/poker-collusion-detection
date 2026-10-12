"""STEP204: production build carrying the CI causal-fold columns, optionally with the STEP188 ensemble.

The change is contained: coordinated_isolation's STAGE-2 column list gains four columns. The network
input stays at 453 and the S* score v1 is untouched, so STEP128's ss and v1 replay gates still bind
exactly as they did for STEP167 and STEP188 -- only v1_lp moves.

Four phases:
  features  compute the four columns for the EVALUATION top-20 keys. GATE: the same code, run on the
            development frame, must reproduce outputs_step194_20260916/causal_fold_features.parquet
            value-for-value. That is what proves the evaluation-side computation is the measured one.
  fit       per member, STEP167 phase_fit with cols[CI] extended by the four columns. The network is
            unchanged, so this differs from STEP188's fit ONLY in the stage-2 column list.
  score     per member, delegated to S167.phase_score with WORK repointed and CSP.stage2_predict
            WRAPPED so the four columns are joined on demand. Nothing else in the scoring path is
            touched -- the wrapper is a no-op for the S* bundle, whose column lists do not contain
            them, which is why ss and v1 still replay STEP128 exactly.
  build     average v1_lp over members by (pair_id, hand_id), typed-decode once, swap the evidence of
            the 16,060 selected pairs into the STEP169 base. Copied from STEP188 phase_build.

Why the wrapper rather than an edited score_batch: stage2_predict does frame.select(cols), so joining
the columns when and only when cols asks for them leaves every other code path byte-identical. A
hand-copied score_batch would risk the drift that gate197_sstar_resolution.json warns about.

Never uploads. Amin decides.

Usage: step204_C_ci_causal_production.py features|fit|score|build REGSHA
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import json
import time
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
from evidence_aug import step165_C_e2e_audit as S165
from evidence_aug import step167_C_lp_production as S167
from evidence_aug import step188_C_ensemble_production as S188
from evidence_aug.step194_C_causal_fold_features import causal_features, NEW

OUT = POKER / 'Claude Version/outputs_step204_20260917'
REG = OUT / 'registration204.json'
DEVFEAT = POKER / 'Claude Version/outputs_step194_20260916/causal_fold_features.parquet'
EVALFEAT = OUT / 'causal_fold_eval.parquet'
NN = list(S165.NN_COLS)
FAMS = S165.FAMS


def read_reg(reg_sha):
    assert sha(REG) == reg_sha, 'registration changed'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == reg['code']['runner_sha256'], 'runner changed after registration'
    return reg


def member_dir(name, reg):
    r = (reg.get('reuse') or {}).get(name)
    return (POKER / r) if r else (OUT / 'work' / name)


def compute_features(keys, pmap, root):
    """causal_features over (pair_id, hand_id) keys. The per-hand action list and the pair's own two
    players are all it needs; `causal_features` itself is IMPORTED, never retyped."""
    hids = keys['hand_id'].unique().to_list()
    acts = (pl.scan_parquet(root / 'data/actions.parquet')
            .filter(pl.col('hand_id').is_in(hids) & (pl.col('street') == 'preflop'))
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call')
            .sort('hand_id', 'action_no').collect())
    arows = {}
    for hid, _, st, pid, a, amt, tc in acts.iter_rows():
        arows.setdefault(hid, []).append((st, pid, a, amt, tc))
    rows = []
    for pid, hid in keys.iter_rows():
        f = causal_features(arows.get(hid, []), *pmap[pid])
        f['pair_id'], f['hand_id'] = pid, hid
        rows.append(f)
    out = pl.DataFrame(rows).select(['pair_id', 'hand_id'] + NEW)
    assert out.height == keys.height
    assert np.isfinite(out.select(NEW).to_numpy()).all(), 'non-finite feature'
    return out, acts.height


def phase_features(reg, reg_sha):
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    t0 = time.perf_counter()
    prot = check_protected()

    # ---- GATE: reproduce the measured development features with this same code path --------------
    frame, _, _, _, _ = CSP.development_frame()
    dkeys = frame.select(KEYS).unique().sort(KEYS)
    dlab = pl.read_csv(CSP.ROOT / 'data/development_labels.csv',
                       columns=['pair_id', 'player_1', 'player_2'])
    dmap = {r['pair_id']: (r['player_1'], r['player_2']) for r in dlab.iter_rows(named=True)}
    mine, nact = compute_features(dkeys, dmap, CSP.ROOT)
    ref = pl.read_parquet(DEVFEAT).sort(KEYS)
    mine_s = mine.sort(KEYS)
    assert mine_s.select(KEYS).equals(ref.select(KEYS)), 'development key set differs'
    gate = {c: float(np.abs(mine_s[c].to_numpy() - ref[c].to_numpy()).max()) for c in NEW}
    print('development reproduction gate', json.dumps(gate), flush=True)
    assert max(gate.values()) == 0.0, 'this code does not reproduce the measured STEP194 features'

    # ---- the evaluation keys, taken from STEP128's stored top-20 score files ---------------------
    V1 = S167.V1
    tops = sorted((V1 / 'batches').glob('batch_*-top20-scores.parquet'))
    assert len(tops) == 20, len(tops)
    # the top-level STEP128 receipt holds a sha per BATCH RECEIPT; each batch receipt then holds the
    # shas of its own parquets. Verify both links so the keys are provably STEP128's.
    rec128 = json.loads((V1 / 'receipt.json').read_text(encoding='utf-8'))
    ekeys = []
    for p in tops:
        stem = p.name.split('-')[0]
        rj = V1 / 'batches' / (stem + '.json')
        assert sha(rj) == rec128['batch_receipts'][stem], rj.name
        assert sha(p) == json.loads(rj.read_text(encoding='utf-8'))['files'][p.name], p.name
        ekeys.append(pl.read_parquet(p).select(KEYS))
    ekeys = pl.concat(ekeys).unique().sort(KEYS)
    pmap = {p: (a, b) for p, a, b in pl.read_csv(
        CSP.ROOT / 'data/evaluation_pairs.csv',
        columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    assert set(ekeys['pair_id'].unique()) <= set(pmap), 'a scored pair is not in evaluation_pairs'
    print('%d evaluation (pair,hand) rows over %d pairs, %d hands  %ds'
          % (ekeys.height, ekeys['pair_id'].n_unique(), ekeys['hand_id'].n_unique(),
             round(time.perf_counter() - t0)), flush=True)
    ev, enact = compute_features(ekeys, pmap, CSP.ROOT)
    ev = ev.sort(KEYS)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.write_parquet(EVALFEAT)
    r = {'registration_sha256': reg_sha,
         'development_reproduction_gate_max_abs': gate,
         'development_rows': dkeys.height, 'development_preflop_actions': nact,
         'eval_rows': ev.height, 'eval_pairs': int(ekeys['pair_id'].n_unique()),
         'eval_hands': int(ekeys['hand_id'].n_unique()), 'eval_preflop_actions': enact,
         'eval_feature_sha256': sha(EVALFEAT),
         'describe': {c: {'mean': float(ev[c].mean()), 'max': float(ev[c].max())} for c in NEW},
         'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    assert check_protected() == prot
    write_json(OUT / 'receipt_features.json', r)
    print(json.dumps(r, indent=1), flush=True)


def patch_stage2_predict(CSP, KEYS):
    """Join the four columns on demand, and ONLY when the requested column list asks for them.

    A no-op for the S* bundle (its column lists do not contain them), which is exactly why ss and v1
    still replay STEP128 after this is installed.
    """
    lookup = pl.read_parquet(EVALFEAT).select(KEYS + NEW)
    orig = CSP.stage2_predict
    state = {'joins': 0, 'rows': 0}

    def patched(models, frame, cols):
        need = [c for c in NEW if c in cols]
        if need and not set(need) <= set(frame.columns):
            frame = frame.join(lookup, on=KEYS, how='left', validate='m:1')
            n = frame.select(need).null_count().sum_horizontal().item()
            assert n == 0, 'unmatched evaluation rows in the causal-fold join: %d' % n
            state['joins'] += 1
            state['rows'] += frame.height
        return orig(models, frame, cols)

    CSP.stage2_predict = patched
    return orig, state


def phase_fit(reg, member):
    """STEP188 phase_fit with cols[CI] extended by the four columns. Network untouched."""
    S108, _ = S167.modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    cfg = dict(reg['members'][member])
    W = member_dir(member, reg)
    (W / 'batches').mkdir(parents=True, exist_ok=True)
    mp, fp = W / 'lp-models.joblib', W / 'fit.json'
    if fp.exists():
        r = json.loads(fp.read_text(encoding='utf-8'))
        assert r['models_sha256'] == sha(mp)
        print('fit already complete for', member, flush=True)
        return r
    t0 = time.perf_counter()
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    rcols, struct = S167.audit_columns()
    for f in FAMS:
        cols = bundle['cols'][f]
        assert cols[-61:] == S108.STATE + S108.RS
        assert set(cols[:-61]) == set(struct + S167.POOLF + (rcols if f != CSP.CI else [])), f
    data, _, _, _, _ = CSP.development_frame()
    players = pl.read_csv(CSP.ROOT / 'data/development_labels.csv',
                          columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players)
    assert state.select(KEYS).equals(requested)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    pref, _ = S108.S98.prefix_for_keys(requested)
    keys0 = data.select(KEYS)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(keys0)
    # the four columns, from the MEASURED development parquet
    keys1 = data.select(KEYS)
    data = data.join(pl.read_parquet(DEVFEAT).select(KEYS + NEW), on=KEYS, how='left',
                     validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(keys1), 'causal-fold join reordered the development frame'
    assert data.select(NEW).null_count().sum_horizontal().item() == 0, 'unmatched development rows'
    data = data.sort(KEYS)
    for f in FAMS:
        top = data.filter((pl.col('behavior_family') == f) & (pl.col('s1_rank') <= TOP))
        refit, _, _ = CSP.stage2_fit(data, bundle['cols'][f], f, CSP.CLF)
        np.testing.assert_array_equal(
            CSP.stage2_predict(refit, top, bundle['cols'][f]),
            CSP.stage2_predict(bundle['models'][f], top, bundle['cols'][f]))
    print(member, 'S* refit equals step-108; rows', data.height, 'pairs',
          data['pair_id'].n_unique(), round(time.perf_counter() - t0, 1), flush=True)
    X, fam = S167.xmatrix(data, rcols, struct, 'behavior_family')
    assert X.shape[1] == 453, 'network input changed: %s' % (X.shape,)
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
    jobs = [(cfg, (('oof', k), 'joint', s, rowmask(tr), rowmask(te)))
            for k, (tr, te) in enumerate(folds) for s in S167.SEEDS_OOF]
    with Pool(12, initializer=S165.pool_init, initargs=init_args) as pool:
        full = pool.map(S188.fit_full_cfg, [(cfg, s) for s in S167.SEEDS_FULL], chunksize=1)
        oof = pool.map(S188.fit_job_cfg, jobs, chunksize=1)
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
    models, cols, receipts = {}, {}, []
    for f in FAMS:
        extra = NEW if f == CSP.CI else []
        cols[f] = bundle['cols'][f] + extra + NN
        models[f], nq, npos = CSP.stage2_fit(data, cols[f], f, CSP.CLF)
        receipts.append({'family': f, 'training_pairs': int(nq), 'training_listed': int(npos),
                         'n_columns': len(cols[f]), 'causal_fold_columns': len(extra)})
    assert len(cols[CSP.CI]) == len(bundle['cols'][CSP.CI]) + len(NEW) + len(NN)
    for f in FAMS:
        if f != CSP.CI:
            assert cols[f] == bundle['cols'][f] + NN, f + ' column list changed'
    net = [{'seed': s, 'P': P, 'xstats': xs, 'cstats': cs, 'trace': tr} for s, P, xs, cs, tr in full]
    joblib.dump({'models': models, 'cols': cols, 'net': net, 'rcols': rcols, 'struct': struct}, mp)
    y = rel[QIX]
    r = {'member': member, 'config': cfg, 'registration_sha256': sha(REG),
         'models_sha256': sha(mp), 'dev_rows': data.height, 'dev_pairs': int(len(upairs)),
         'dev_top20_rows': int(len(QIX)), 'sstar_refit_equals_step108': True,
         'ci_stage2_columns_added': NEW, 'lp_receipts': receipts,
         'full_fit_best_epochs': [tr[2] for _, _, _, _, tr in full], 'oof_fits': len(oof),
         'oof_nn_p2_auc_top20': float(roc_auc_score(y, E[:, 5])),
         'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(fp, r)
    print('fit', json.dumps({k: r[k] for k in
                             ('member', 'config', 'oof_nn_p2_auc_top20', 'seconds')}), flush=True)
    return r


def phase_score(reg, member):
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    W = member_dir(member, reg)
    assert (W / 'fit.json').exists(), 'fit the member first'
    orig, state = patch_stage2_predict(CSP, KEYS)
    old = S167.WORK
    try:
        S167.WORK = W
        S167.phase_score(json.loads(
            (W / 'fit.json').read_text(encoding='utf-8'))['registration_sha256'])
    finally:
        S167.WORK = old
        CSP.stage2_predict = orig
    print('causal-fold joins performed: %d over %d rows' % (state['joins'], state['rows']),
          flush=True)
    assert state['joins'] > 0, 'the causal-fold columns were never joined -- scoring ignored them'


def phase_build(reg, reg_sha):
    """STEP188 phase_build, unchanged in substance: average v1_lp over members, decode once, swap."""
    S108, _ = S167.modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    from evidence_aug import step104_C_typed_decoder_build as S104
    from validation import validate_submission
    t0 = time.perf_counter()
    prot = check_protected()
    members = list(reg['members'])
    EV = list(CSP.EVIDENCE_COLUMNS)
    basep = POKER / reg['base']['path']
    assert sha(basep) == reg['base']['sha256'], 'base file changed'
    base = pl.read_csv(basep, infer_schema=False)
    ci = base.filter(pl.col('predicted_behavior') == 'coordinated_isolation').select('pair_id')
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())
    pmap = {p: (a, b) for p, a, b in pl.read_csv(
        CSP.ROOT / 'data/evaluation_pairs.csv',
        columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    news, recs = [], []
    for i in range(20):
        name, tb = 'batch_%03d' % i, time.perf_counter()
        tops = []
        for m in members:
            W = member_dir(m, reg)
            rj = W / 'batches' / (name + '.json')
            sc = W / 'batches' / (name + '-lp-scores.parquet')
            r = json.loads(rj.read_text(encoding='utf-8'))
            assert r['files'][sc.name] == sha(sc), (m, name)
            tops.append(pl.read_parquet(sc).sort(KEYS))
        ref = tops[0]
        for t in tops[1:]:
            assert t.select(KEYS).equals(ref.select(KEYS)), 'member key order differs in ' + name
            assert np.array_equal(t['v1'].to_numpy(), ref['v1'].to_numpy()), \
                'v1 must not depend on the member'
        ens = np.mean([t['v1_lp'].to_numpy() for t in tops], axis=0)
        bf = ref.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'),
                        (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1')
                        ).with_columns(pl.Series('v1_ens', ens + 10))
        sel = bf.select('pair_id').unique().sort('pair_id')
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id',
                                                  pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        cif = fc.select('pair_id').join(ci.with_columns(pl.lit(True).alias('f')), on='pair_id',
                                        how='left', maintain_order='left')['f'
                                                                           ].fill_null(False).to_numpy()
        typed = t1 & ~cif
        inc = S108.expand(CSP.rank(bf, S104.decode(
            fc.with_columns(pl.Series('score', bf['v1'].to_numpy())), typed)))
        chk = base.join(sel, on='pair_id', how='semi').select(inc.columns).sort('pair_id')
        assert inc.equals(chk), 'v1 replay of the base evidence failed in ' + name
        new = S108.expand(CSP.rank(bf, S104.decode(
            fc.with_columns(pl.Series('score', bf['v1_ens'].to_numpy())), typed)))
        assert new.select('pair_id').equals(inc.select('pair_id'))
        news.append(new)
        sets = [len(set(u[1:]) - set(o[1:])) for u, o in zip(new.rows(), inc.rows())]
        fams = [fam[x[0]] for x in new.rows()]
        recs.append({'batch': name, 'selected_pairs': new.height, 'T1_rows_used': int(typed.sum()),
                     'lists_changed': int(sum(u != o for u, o in zip(new.rows(), inc.rows()))),
                     'top5_set_change_mean': round(float(np.mean(sets)), 4),
                     'top5_set_change_by_family': {
                         f: round(float(np.mean([s for s, g in zip(sets, fams) if g == f])), 4)
                         for f in sorted(set(fams))},
                     's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    repl = pl.concat(news).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060
    rest = base.join(repl.select('pair_id'), on='pair_id', how='anti')
    swap = base.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(
        repl, on='pair_id', validate='1:1')
    result = base.select('pair_id').join(pl.concat([rest, swap.select(base.columns)]),
                                         on='pair_id', validate='1:1', maintain_order='left')
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
    val = validate_submission(csv, CSP.ROOT / 'data/evaluation_pairs.csv',
                              CSP.ROOT / 'data/hands.parquet', CSP.ROOT / 'data/seats.parquet',
                              CSP.ROOT / 'data/sample_submission.csv')
    assert not val.errors, val
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'members': members,
           'combiner': 'SCORE_mean over member v1_lp, decoded once',
           'ci_stage2_columns_added': NEW,
           'eval_feature_sha256': sha(EVALFEAT),
           'output': str(csv), 'output_sha256': sha(csv), 'base_sha256': sha(basep),
           'rows_changed_vs_base': len(diff), 'validation': str(val)[:300], 'batches': recs,
           'lists_changed': int(sum(r['lists_changed'] for r in recs)),
           'top5_set_change_mean': float(np.average(
               [r['top5_set_change_mean'] for r in recs],
               weights=[r['selected_pairs'] for r in recs])),
           'protected_unchanged': True, 'no_submission_actions': True,
           'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'receipt_step204.json', rec)
    print('DONE', json.dumps({k: rec[k] for k in
                              ('output_sha256', 'rows_changed_vs_base', 'lists_changed',
                               'top5_set_change_mean', 'seconds')}), flush=True)


def main():
    phase, reg_sha = sys.argv[1], sys.argv[2]
    reg = read_reg(reg_sha)
    OUT.mkdir(parents=True, exist_ok=True)
    if phase == 'features':
        phase_features(reg, reg_sha)
        return
    if phase == 'build':
        phase_build(reg, reg_sha)
        return
    only = [m for m in os.environ.get('STEP204_MEMBERS', '').split(',') if m] or list(reg['members'])
    for m in only:
        assert m in reg['members'], m
        if phase == 'fit':
            phase_fit(reg, m)
        elif phase == 'score':
            phase_score(reg, m)


if __name__ == '__main__':
    main()
