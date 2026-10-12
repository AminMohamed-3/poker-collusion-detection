"""STEP251 production builder: STEP222 + the three STEP237 fold-shape columns on directed_transfer (DTfp3).

This is step212_C_family_columns_production.py's logic copied into a new file (step212 is hash-registered and
its own registered build has finished, so it is never edited), with three changes:

  1. a new evaluation-side builder 'invisible' -- the STEP237 block through step246_invisible_features, which
     drives STEP237's own hand_tables / pairhand_features read-only so the development and evaluation numbers
     cannot diverge.  Its development reproduction gate is an inner join against STEP237's own table
     (experiments/step237_invisible_events_20260917/step237_features.parquet, 223,749 rows) on the 45,129
     production development rows, max abs exactly 0.0, run BEFORE any evaluation key is touched;
  2. the five evaluation feature files STEP222 already built are REUSED: they are copied into this step's
     folder and their sha256 is checked against the registration, because they are the same functions of the
     same evaluation data and rebuilding them would only risk a difference.  STEP222's own features receipt
     is recorded as their provenance;
  3. WORKERS (default 6) instead of step212's hard-wired 12, and phase_score skips a member whose 20 batches
     are already complete -- step212 asserts joins > 0 and therefore cannot be re-run on a finished member.

Only directed_transfer may change against STEP222: the three columns enter the directed_transfer stage-2
column list and nothing else, so soft_play and coordinated_isolation evidence, the risk score and the
predicted behaviour must all come out byte-identical, which phase_build asserts.

Evaluation-phase hands are asserted for every evaluation feature row and development-phase hands for every
development row; nothing is ever joined across phases by player or pair identity.  Never uploads.
Usage: step251_P_production.py REGPATH features|fit|score|build REGSHA
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import json
import shutil
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
STEP204_EVAL_SHA = '22ab6a425182ab53edaf05baf29ee0154a242e6da62d1023780163e07cd42333'
WORKERS = int(os.environ.get('STEP251_WORKERS', '6'))

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


def phase_features(reg, out, reg_sha):
    """Reuse STEP222's five evaluation feature files; gate and build only the new 'invisible' block."""
    from evidence_aug import step246_invisible_features as S246
    t0 = time.perf_counter()
    prot = check_protected()
    assert sha(STEP204_EVAL) == STEP204_EVAL_SHA
    ekeys = pl.read_parquet(STEP204_EVAL).select(KEYS).sort(KEYS)
    assert ekeys.height == 321200
    ev = pl.read_csv(POKER / 'data/evaluation_pairs.csv',
                     columns=['pair_id', 'player_1', 'player_2', 'shared_hands'])
    ev = ev.join(ekeys.select('pair_id').unique(), on='pair_id', how='semi')
    assert ev.height == 16060
    pmap = {p: (a, b) for p, a, b, _ in ev.iter_rows()}
    dlab = pl.read_csv(POKER / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    dmap = {p: (a, b) for p, a, b in dlab.iter_rows()}
    dkeys = pl.read_parquet(CV / 'outputs_step194_20260916/causal_fold_features.parquet').select(KEYS)
    assert dkeys.height == 45129
    rec = {'registration_sha256': reg_sha, 'reused': {}, 'built': {}, 'gates': {}}

    # ---- 1. the five STEP222 evaluation feature files, copied and sha-checked
    for name, src in reg['reuse_eval_features'].items():
        rel = reg['eval_builders'][name]
        dest = out / rel
        s = CV / src['path']
        assert sha(s) == src['sha256'], 'STEP222 evaluation feature file changed: ' + name
        if dest.exists():
            assert sha(dest) == src['sha256'], 'a different copy is already here: ' + name
        else:
            shutil.copyfile(s, dest)
            assert sha(dest) == src['sha256'], 'copy differs from source: ' + name
        f = pl.read_parquet(dest)
        assert f.select(KEYS).sort(KEYS).equals(ekeys), name + ' does not carry the evaluation keys'
        assert f.select(reg['eval_columns'][name]).null_count().sum_horizontal().item() == 0
        rec['reused'][name] = {'from': src['path'], 'sha256': src['sha256'], 'rows': f.height}
        print('reused %-14s %s  %s' % (name, f.shape, src['sha256'][:16]), flush=True)
    rec['step222_features_receipt_sha256'] = sha(CV / reg['reuse_provenance']['step222_features_receipt'])

    # ---- 2. the new block: development gate first, then the evaluation build
    tmp = out / 'tmp_features'
    tmp.mkdir(parents=True, exist_ok=True)
    nm = S246.names()
    ref = pl.read_parquet(CV / reg['dev_features_by_builder']['invisible'])
    assert sha(CV / reg['dev_features_by_builder']['invisible']) == \
        reg['code']['feature_file_sha256'][reg['dev_features_by_builder']['invisible']]
    dev_path = tmp / 'dev_invisible.parquet'
    if dev_path.exists():
        mine = pl.read_parquet(dev_path)
    else:
        mine = S246.build(dkeys, dmap, 'development', dev_path, workers=WORKERS)
    j = mine.join(ref, on=KEYS, how='inner', validate='1:1', suffix='_ref')
    assert j.height == mine.height == 45129, 'gate rows missing'
    g = {c: float(np.abs(j[c].to_numpy().astype('float64')
                         - j[c + '_ref'].to_numpy().astype('float64')).max()) for c in nm}
    rec['gates']['invisible_development_reproduction'] = {'rows': j.height, 'max_abs': max(g.values()),
                                                          'per_column': g}
    print('invisible development reproduction max abs', max(g.values()), 'on', j.height, 'rows', flush=True)
    assert max(g.values()) == 0.0, 'the invisible block does not reproduce step237_features.parquet'

    dest = out / reg['eval_builders']['invisible']
    if dest.exists():
        print('evaluation invisible file already built', flush=True)
        ef = pl.read_parquet(dest)
    else:
        ef = S246.build(ekeys, pmap, 'evaluation', dest, workers=WORKERS)
    assert ef.select(KEYS).sort(KEYS).equals(ekeys), 'the evaluation build lost keys'
    assert ef.select(reg['eval_columns']['invisible']).null_count().sum_horizontal().item() == 0
    v = ef.select(reg['eval_columns']['invisible']).to_numpy()
    assert np.isfinite(v).all()
    rec['built']['invisible'] = {'rows': ef.height, 'sha256': sha(dest),
                                 'columns_used': reg['eval_columns']['invisible'],
                                 'nonzero_share': {c: round(float((ef[c] != 0).mean()), 4)
                                                   for c in reg['eval_columns']['invisible']},
                                 'dev_nonzero_share': {c: round(float((mine[c] != 0).mean()), 4)
                                                       for c in reg['eval_columns']['invisible']}}
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    write_json(out / ('receipt_features_%s.json' % reg['step'].lower()), rec)
    print(json.dumps({k: v for k, v in rec.items() if k != 'gates'}, indent=1), flush=True)


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
    players = pl.read_csv(CSP.ROOT / 'data/development_labels.csv',
                          columns=['pair_id', 'player_1', 'player_2'])
    requested = data.filter(pl.col('s1_rank') <= TOP).select(KEYS).unique().sort(KEYS)
    state, _ = CSP.build_for_keys(requested, players)
    data = data.join(state, on=KEYS, how='left', validate='1:1')
    pref, _ = S108.S98.prefix_for_keys(requested)
    keys0 = data.select(KEYS)
    data = data.join(pref, on=KEYS, how='left', validate='1:1', maintain_order='left')
    assert data.select(KEYS).equals(keys0)
    allx = []
    for rel, cols in reg['dev_features'].items():
        assert sha(CV / rel) == reg['code']['feature_file_sha256'][rel], rel
        k1 = data.select(KEYS)
        data = data.join(pl.read_parquet(CV / rel).select(KEYS + cols), on=KEYS, how='left',
                         validate='1:1', maintain_order='left')
        assert data.select(KEYS).equals(k1)
        allx += cols
    assert len(set(allx)) == len(allx), 'a development column is supplied twice'
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
    with Pool(WORKERS, initializer=S165.pool_init, initargs=init_args) as pool:
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
        assert len(set(cols[f])) == len(cols[f]), ('duplicate stage-2 column', f)
        models[f], nq, npos = CSP.stage2_fit(data, cols[f], f, CSP.CLF)
        receipts.append({'family': f, 'n_columns': len(cols[f]), 'extra': extra_for(reg, f, CSP.CI)})
    net = [{'seed': s, 'P': P, 'xstats': xs, 'cstats': cs, 'trace': tr} for s, P, xs, cs, tr in full]
    joblib.dump({'models': models, 'cols': cols, 'net': net, 'rcols': rcols, 'struct': struct}, mp)
    # the networks do not see the extra family columns, so the fit must land on STEP204's epochs exactly
    s204 = json.loads((CV / 'outputs_step204_20260917/work' / member / 'fit.json').read_text(encoding='utf-8'))
    ep = [tr[2] for _, _, _, _, tr in full]
    r = {'member': member, 'config': cfg, 'registration_sha256': sha(REGPATH['p']),
         'models_sha256': sha(mp), 'lp_receipts': receipts, 'full_fit_best_epochs': ep,
         'same_best_epochs_as_step204': ep == s204['full_fit_best_epochs'],
         'oof_nn_p2_auc_top20': float(roc_auc_score(rel[QIX], E[:, 5])),
         'step204_oof_nn_p2_auc_top20': s204['oof_nn_p2_auc_top20'],
         'workers': WORKERS,
         'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(fp, r)
    print('fit', json.dumps({k: r[k] for k in ('member', 'same_best_epochs_as_step204', 'oof_nn_p2_auc_top20',
                                               'step204_oof_nn_p2_auc_top20', 'seconds')}), flush=True)


def phase_score(reg, out, member):
    S108, _ = S167.modules()
    CSP = S108.CSP
    W = out / 'work' / member
    assert (W / 'fit.json').exists(), 'fit the member first'
    done = len(list((W / 'batches').glob('batch_*.json')))
    if done == 20:
        print('score already complete for %s (20/20 batches); skipping' % member, flush=True)
        return
    cols_all = sorted({c for f in FAMS for c in extra_for(reg, f, CSP.CI)})
    orig, state = patch(CSP, eval_lookup(reg, out), cols_all)
    old = S167.WORK
    try:
        S167.WORK = W
        S167.phase_score(json.loads((W / 'fit.json').read_text(encoding='utf-8'))['registration_sha256'])
    finally:
        S167.WORK = old
        CSP.stage2_predict = orig
    print('extra-column joins performed: %d (batches already done before this call: %d)'
          % (state['joins'], done), flush=True)
    assert state['joins'] > 0, 'nothing was scored; the extra columns never reached stage 2'


def phase_build(reg, out, reg_sha):
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
    write_json(out / ('verify_%s_vs_step222.json' % reg['step'].lower()), res)
    print(json.dumps(res, indent=1), flush=True)
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
