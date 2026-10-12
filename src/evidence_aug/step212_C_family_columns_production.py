"""STEP212 production builder: step207_C_family_columns_production.py generalised (step207 itself is left
untouched because its registered build is running and checks its own hash).

Additions over step207:
  * evaluation builders 'role_contest' (step210 cs_ split) and 'role_events' (step211 v2 split), both using the
    pool sign from 'roles' exactly as 'role_strength' does, and each reproducing its development parquet first;
  * evaluation builder 'role_equity' (step215 eq_ split, same sign; its build returns a third value, ignored);
  * evaluation builder 'outsider' (step222: Codex's linked outsider-card block, unsigned, top-20 keys only);
  * receipts named by the registration's `step`; the registration file is whatever path is passed;
  * the row-by-row check is against `compare_to`, and the families allowed to differ are named in the
    registration (`families_changed_vs_compare`).

Never uploads. Amin decides.
Usage: step212_C_family_columns_production.py REGPATH features|fit|score|build REGSHA
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
from evidence_aug import step204_C_ci_causal_production as S204

CV = POKER / 'Claude Version'
NN = list(S165.NN_COLS)
FAMS = S165.FAMS
KEYS = ['pair_id', 'hand_id']
STEP204_EVAL = CV / 'outputs_step204_20260917/causal_fold_eval.parquet'
SIGNED = ('role_strength', 'role_contest', 'role_events', 'role_equity')
STEP204_EVAL_SHA = '22ab6a425182ab53edaf05baf29ee0154a242e6da62d1023780163e07cd42333'


REGPATH = {}


def load(regp, reg_sha):
    assert sha(regp) == reg_sha, 'registration changed'
    REGPATH['p'] = regp
    reg = json.loads(regp.read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == reg['code']['runner_sha256'], 'runner changed after registration'
    return reg, regp.parent


def extra_for(reg, f, CI):
    return (S204.NEW if f == CI else []) + list(reg['extra_columns'].get(f, []))


def eval_lookup(reg, out):
    """All evaluation-side extra columns in one frame keyed by (pair, hand)."""
    lk = pl.read_parquet(STEP204_EVAL).select(KEYS + S204.NEW)
    for name, rel in reg['eval_builders'].items():
        cols = reg['eval_columns'][name]
        lk = lk.join(pl.read_parquet(out / rel).select(KEYS + cols), on=KEYS, how='left', validate='1:1')
    assert lk.null_count().sum_horizontal().item() == 0
    return lk


def evaluation_pools(ev):
    """Every evaluation-phase hand both listed players sat in, per pair. Must equal shared_hands for every pair."""
    players = sorted(set(ev['player_1'].to_list()) | set(ev['player_2'].to_list()))
    eh = pl.scan_parquet(POKER / 'data/hands.parquet').filter(pl.col('phase') == 'evaluation').select('hand_id')
    st = (pl.scan_parquet(POKER / 'data/seats.parquet').filter(pl.col('player_id').is_in(players))
          .join(eh, on='hand_id', how='semi').select('hand_id', 'player_id').collect())
    a = ev.select('pair_id', 'player_1', 'player_2')
    pool = (a.join(st.rename({'player_id': 'player_1'}), on='player_1')
            .join(st.rename({'player_id': 'player_2', 'hand_id': 'h2'}),
                  left_on=['player_2', 'hand_id'], right_on=['player_2', 'h2'], how='inner')
            .select(KEYS).unique().sort(KEYS))
    n = dict(pool.group_by('pair_id').len().iter_rows())
    assert all(n.get(p, 0) == s for p, s in ev.select('pair_id', 'shared_hands').iter_rows()), \
        'pool size differs from shared_hands'
    return pool


def phase_features(reg, out, reg_sha):
    from evidence_aug import step205_C_causal_completion as S205
    from evidence_aug import step206_C_dt_direction as S206
    from evidence_aug import step206c_C_dt_roles as S206c
    from evidence_aug import step208_C_dt_role_strength as S208
    from evidence_aug import step210_C_dt_role_contest as S210
    from evidence_aug import step211_C_dt_role_events as S211
    from evidence_aug import step215_C_dt_role_equity as S215
    from evidence_aug import step222_C_outsider_linked as S222
    MODS = {'role_strength': S208, 'role_contest': S210, 'role_events': S211, 'role_equity': S215}
    t0 = time.perf_counter()
    prot = check_protected()
    assert sha(STEP204_EVAL) == STEP204_EVAL_SHA
    ekeys = pl.read_parquet(STEP204_EVAL).select(KEYS).sort(KEYS)   # STEP128's keys, receipt-verified in STEP204
    assert ekeys.height == 321200
    ev = pl.read_csv(POKER / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2', 'shared_hands'])
    ev = ev.join(ekeys.select('pair_id').unique(), on='pair_id', how='semi')
    assert ev.height == 16060
    pmap = {p: (a, b) for p, a, b, _ in ev.iter_rows()}
    dlab = pl.read_csv(POKER / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    dmap = {p: (a, b) for p, a, b in dlab.iter_rows()}
    dkeys = pl.read_parquet(CV / 'outputs_step194_20260916/causal_fold_features.parquet').select(KEYS)
    rec = {'registration_sha256': reg_sha, 'gates': {}}
    tmp = out / 'tmp_features'
    tmp.mkdir(parents=True, exist_ok=True)
    dev_roles = None
    for name in reg['eval_builders']:
        cols = reg['eval_columns'][name]
        ref_path = CV / reg['dev_features_by_builder'][name]
        ref = pl.read_parquet(ref_path).sort(KEYS)
        if name in SIGNED:
            assert dev_roles is not None, 'roles must be built before ' + name
            mod = MODS[name]
            mine = mod.build(dev_roles.select(KEYS + ['dt_sign']), dmap, 'development',
                                tmp / ('dev_' + name + '.parquet'))[0]
            mine = mine.sort(KEYS)
            cols = [c for c in mine.columns if c not in KEYS]
        elif name == 'completion':
            mine = S205.build_features(dkeys, dmap, 'development', tmp / 'dev_completion.parquet').sort(KEYS)
        elif name == 'outsider':
            mine = S222.build(dkeys, dmap, 'development', tmp / 'dev_outsider.parquet').sort(KEYS)
        elif name == 'direction':
            mine = S206.build_direction(dkeys, dmap, 'development', tmp / 'dev_direction.parquet').sort(KEYS)
        elif name == 'roles':
            mine = S206c.build_roles(dkeys, dmap, 'development', tmp / 'dev_roles.parquet').sort(KEYS)
            dev_roles = mine
            cols = ['dt_sign'] + S206c.ROLE
        else:
            raise SystemExit('unknown builder ' + name)
        assert mine.select(KEYS).equals(ref.select(KEYS))
        g = {c: float(np.abs(mine[c].to_numpy() - ref[c].to_numpy()).max()) for c in cols}
        rec['gates'][name] = g
        print(name, 'development reproduction', json.dumps(g), flush=True)
        assert max(g.values()) == 0.0, name + ' does not reproduce the measured development features'
    pool = None
    eval_roles = None
    for name, rel in reg['eval_builders'].items():
        dest = out / rel
        assert not dest.exists(), 'preserve a built evaluation feature file'
        if name == 'completion':
            S205.build_features(ekeys, pmap, 'evaluation', dest)
        elif name == 'outsider':
            S222.build(ekeys, pmap, 'evaluation', dest)
        elif name in SIGNED:
            assert eval_roles is not None, 'roles must be built before ' + name
            mod = MODS[name]
            mod.build(eval_roles.select(KEYS + ['dt_sign']), pmap, 'evaluation', dest)
        else:
            # direction and roles need the whole evaluation pool of each pair
            if pool is None:
                pool = evaluation_pools(ev)
                assert ekeys.join(pool, on=KEYS, how='anti').height == 0, 'a top-20 row is outside its pool'
                rec['evaluation_pool_rows'] = pool.height
            if name == 'direction':
                S206.build_direction(pool, pmap, 'evaluation', dest, keep_keys=ekeys)
            elif name == 'roles':
                eval_roles = S206c.build_roles(pool, pmap, 'evaluation', dest, keep_keys=ekeys)
                assert eval_roles.select(KEYS).equals(ekeys)
            else:
                raise SystemExit('unknown builder ' + name)
        rec[name + '_sha256'] = sha(dest)
    if {'direction', 'roles'} <= set(reg['eval_builders']):
        # the two files must agree on every pair's direction
        dr = pl.read_parquet(out / reg['eval_builders']['direction'])
        ro = pl.read_parquet(out / reg['eval_builders']['roles']).select(KEYS + ['dt_sign', 'dt_donor_put_bb'])
        j = dr.join(ro, on=KEYS, validate='1:1')
        raw = ((j['B'] - j['A']) / 2).to_numpy()
        nz = raw != 0
        bad = int((np.sign(j['dt_flow_along_bb'].to_numpy()[nz] / raw[nz]) != j['dt_sign'].to_numpy()[nz]).sum())
        rec['evaluation_sign_disagreements'] = bad
        assert bad == 0, 'direction and roles disagree on the sign'
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    write_json(out / ('receipt_features_%s.json' % reg['step'].lower()), rec)
    print(json.dumps(rec, indent=1), flush=True)


def patch(CSP, lookup, cols_all):
    orig = CSP.stage2_predict
    state = {'joins': 0}

    def patched(models, frame, cols):
        need = [c for c in cols_all if c in cols]
        if need and not set(need) <= set(frame.columns):
            frame = frame.join(lookup.select(KEYS + [c for c in cols_all if c not in frame.columns]),
                               on=KEYS, how='left', validate='m:1')
            assert frame.select(need).null_count().sum_horizontal().item() == 0, 'unmatched evaluation rows'
            state['joins'] += 1
        return orig(models, frame, cols)

    CSP.stage2_predict = patched
    return orig, state


def phase_fit(reg, out, member):
    S108, _ = S167.modules()
    CSP, TOP = S108.CSP, S108.TOP
    cfg = dict(reg['members'][member])
    W = out / 'work' / member
    (W / 'batches').mkdir(parents=True, exist_ok=True)
    mp, fp = W / 'lp-models.joblib', W / 'fit.json'
    if fp.exists():
        print('fit already complete for', member, flush=True)
        return
    t0 = time.perf_counter()
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    rcols, struct = S167.audit_columns()
    data, _, _, _, _ = CSP.development_frame()
    players = pl.read_csv(CSP.ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    pref, _ = S108.S98.prefix_for_keys(requested)
    keys0 = data.select(KEYS)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(keys0)
    allx = []
    for rel, cols in reg['dev_features'].items():
        k1 = data.select(KEYS)
        data = data.join(pl.read_parquet(CV / rel).select(KEYS + cols), on=KEYS, how='left',
                         validate='1:1', maintain_order='left')
        assert data.select(KEYS).equals(k1)
        allx += cols
    assert data.select(allx).null_count().sum_horizontal().item() == 0
    data = data.sort(KEYS)
    for f in FAMS:
        top = data.filter((pl.col('behavior_family') == f) & (pl.col('s1_rank') <= TOP))
        refit, _, _ = CSP.stage2_fit(data, bundle['cols'][f], f, CSP.CLF)
        np.testing.assert_array_equal(CSP.stage2_predict(refit, top, bundle['cols'][f]),
                                      CSP.stage2_predict(bundle['models'][f], top, bundle['cols'][f]))
    X, fam = S167.xmatrix(data, rcols, struct, 'behavior_family')
    assert X.shape[1] == 453
    QIX = np.flatnonzero((data['s1_rank'] <= TOP).to_numpy())
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
    E = acc / cnt[:, None]
    Eall = np.full((data.height, len(NN)), np.nan)
    Eall[QIX] = E
    data = data.with_columns([pl.Series(c, Eall[:, j]) for j, c in enumerate(NN)])
    models, cols, receipts = {}, {}, []
    for f in FAMS:
        cols[f] = bundle['cols'][f] + extra_for(reg, f, CSP.CI) + NN
        models[f], nq, npos = CSP.stage2_fit(data, cols[f], f, CSP.CLF)
        receipts.append({'family': f, 'n_columns': len(cols[f]), 'extra': extra_for(reg, f, CSP.CI)})
    net = [{'seed': s, 'P': P, 'xstats': xs, 'cstats': cs, 'trace': tr} for s, P, xs, cs, tr in full]
    joblib.dump({'models': models, 'cols': cols, 'net': net, 'rcols': rcols, 'struct': struct}, mp)
    # the STEP204 fit of the same member, for the record: identical best epochs mean an identical network
    s204 = json.loads((CV / 'outputs_step204_20260917/work' / member / 'fit.json').read_text(encoding='utf-8'))
    ep = [tr[2] for _, _, _, _, tr in full]
    r = {'member': member, 'config': cfg, 'registration_sha256': sha(REGPATH['p']),
         'models_sha256': sha(mp), 'lp_receipts': receipts, 'full_fit_best_epochs': ep,
         'same_best_epochs_as_step204': ep == s204['full_fit_best_epochs'],
         'oof_nn_p2_auc_top20': float(roc_auc_score(rel[QIX], E[:, 5])),
         'step204_oof_nn_p2_auc_top20': s204['oof_nn_p2_auc_top20'],
         'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(fp, r)
    print('fit', json.dumps({k: r[k] for k in ('member', 'same_best_epochs_as_step204', 'oof_nn_p2_auc_top20',
                                               'step204_oof_nn_p2_auc_top20', 'seconds')}), flush=True)


def phase_score(reg, out, member):
    S108, _ = S167.modules()
    CSP = S108.CSP
    W = out / 'work' / member
    assert (W / 'fit.json').exists(), 'fit the member first'
    cols_all = sorted({c for f in FAMS for c in extra_for(reg, f, CSP.CI)})
    orig, state = patch(CSP, eval_lookup(reg, out), cols_all)
    old = S167.WORK
    try:
        S167.WORK = W
        S167.phase_score(json.loads((W / 'fit.json').read_text(encoding='utf-8'))['registration_sha256'])
    finally:
        S167.WORK = old
        CSP.stage2_predict = orig
    print('extra-column joins performed: %d' % state['joins'], flush=True)
    assert state['joins'] > 0


def phase_build(reg, out, reg_sha):
    """STEP204 phase_build with the member directories under this step, then the row-by-row check."""
    reg2 = dict(reg)
    old = S204.OUT, S204.EVALFEAT
    try:
        S204.OUT = out
        S204.EVALFEAT = STEP204_EVAL
        S204.phase_build(reg2, reg_sha)
    finally:
        S204.OUT, S204.EVALFEAT = old
    src = out / 'receipt_step204.json'
    src.rename(out / ('receipt_build_%s.json' % reg['step'].lower()))
    new = pl.read_csv(out / reg['output_name'], infer_schema=False)
    ref = pl.read_csv(CV / reg['compare_to']['path'], infer_schema=False)
    assert sha(CV / reg['compare_to']['path']) == reg['compare_to']['sha256']
    ev = [c for c in new.columns if c not in ('pair_id', 'risk_score', 'predicted_behavior')]
    fam = new['predicted_behavior'].to_list()
    ra, rb = new.select(ev).rows(), ref.select(ev).rows()
    diff = [i for i in range(new.height) if ra[i] != rb[i]]
    by = {}
    for i in diff:
        by[fam[i]] = by.get(fam[i], 0) + 1
    changed_fams = set(reg['families_changed_vs_compare'])
    res = {'output_sha256': sha(out / reg['output_name']),
           'rows_differing_from_compare': len(diff), 'by_predicted_behavior': by,
           'only_changed_families_differ': set(by) <= changed_fams,
           'risk_behaviour_identical_to_compare': new.select('pair_id', 'risk_score', 'predicted_behavior').equals(
               ref.select('pair_id', 'risk_score', 'predicted_behavior')),
           'evidence_cells_changed': sum(sum(x != y for x, y in zip(ra[i], rb[i])) for i in diff)}
    res['compare_to'] = reg['compare_to']
    write_json(out / ('verify_%s_vs_compare.json' % reg['step'].lower()), res)
    print(json.dumps(res, indent=1), flush=True)
    # STEP204 refit the networks and reproduced STEP188's DT/SP evidence exactly, so an unchanged
    # family that moves here means the build is not what was audited
    assert res['only_changed_families_differ'], 'an unchanged family moved'
    assert res['risk_behaviour_identical_to_compare']


def main():
    regp, phase, reg_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    reg, out = load(regp, reg_sha)
    if phase == 'features':
        phase_features(reg, out, reg_sha)
        return
    if phase == 'build':
        phase_build(reg, out, reg_sha)
        return
    only = [m for m in os.environ.get('STEP_MEMBERS', '').split(',') if m] or list(reg['members'])
    for m in only:
        assert m in reg['members'], m
        if phase == 'fit':
            phase_fit(reg, out, m)
        elif phase == 'score':
            phase_score(reg, out, m)


if __name__ == '__main__':
    main()
