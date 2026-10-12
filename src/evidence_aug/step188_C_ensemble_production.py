"""Step 188 (Claude): production build of the STEP185/186 evidence ensemble. Exploratory, never uploads.
STEP185 measured that averaging the stage-1 network across architectures beats selecting one of them (G -.00046 -> +.00277, final
+.00736, CI [+.0025,+.0130]) and that the right combiner is SCORE_mean -- average the per-member stage-2 outputs, not the stage-1 feature
columns. This runs the STEP167 production pipeline once per registered member and averages v1_lp across members before the typed decode.
Structure, and why it is shaped this way:
  fit    per member. NOT a call into step167.phase_fit: the fits run in a multiprocessing Pool and on Windows the children re-import
         step165_e2e_mlp with its DEFAULT CONFIG, so an architecture set in the parent would be silently ignored and every member would
         come out as the incumbent. The config is therefore carried in the job tuple and applied inside the worker, exactly as STEP174 does.
  score  per member, delegated to step167.phase_score with WORK repointed at the member directory. Safe to delegate because
         step165_e2e_mlp.predict/forward read layer shapes from the parameter dict and never consult CONFIG, so scoring is
         architecture-agnostic, and score_batch is sequential rather than pooled.
  build  average v1_lp over members by (pair_id, hand_id), typed-decode once, swap the evidence of the 16,060 selected pairs into the
         STEP169 base. The v1 (S*) replay gate is unchanged and still exact, since v1 does not depend on the member.
Gates: every member's fit reproduces the step-108 S* models; every batch replays STEP128's ss and v1 exactly; decode(v1) reproduces the
base file's evidence lists; risk and behaviour columns stay byte-identical. Usage: step188_C_ensemble_production.py fit|score|build REGSHA"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import json, time
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

OUT = POKER / 'Claude Version/outputs_step188_20260916'
REG = OUT / 'registration188.json'
NN = list(S165.NN_COLS)
FAMS = S165.FAMS
BASE_CONFIG = {'h1': 64, 'medium': 16, 'h2': 32, 'epochs_max': 400, 'patience': 40, 'lr': 1e-3, 'wd': 1e-2,
               'lam_list2': .5, 'w_stage1': 1., 'clip': 6., 'val_share_tables': .2}


def set_config(cfg):
    """BASE_CONFIG must equal the module's own default, or a member would silently be fitted against a different incumbent."""
    from evidence_aug import step165_e2e_mlp as M
    assert dict(M.CONFIG) == BASE_CONFIG or set_config.applied, 'step165_e2e_mlp.CONFIG default has drifted from BASE_CONFIG'
    set_config.applied = True
    M.CONFIG.clear(); M.CONFIG.update(BASE_CONFIG); M.CONFIG.update(cfg)
    return M


set_config.applied = False


def fit_full_cfg(job):
    """S167.fit_full with the member's architecture applied INSIDE the worker."""
    cfg, seed = job
    set_config(cfg)
    return S167.fit_full(seed)


def fit_job_cfg(job):
    """S165.fit_job with the member's architecture applied INSIDE the worker."""
    cfg, inner = job
    set_config(cfg)
    return S165.fit_job(inner)


def member_dir(name, reg):
    """A member may REUSE an existing production work directory instead of being refitted. The incumbent member is exactly
    STEP167's production run -- same architecture, same seeds, same fold structure -- so its lp-models.joblib and its twenty
    scored batches are reused verbatim rather than recomputed, which halves the build."""
    r = (reg.get('reuse') or {}).get(name)
    return (POKER / r) if r else (OUT / 'work' / name)


def read_reg(reg_sha):
    assert sha(REG) == reg_sha, 'registration changed'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == reg['code']['runner_sha256'], 'runner changed after registration'
    return reg


def phase_fit(reg, member):
    """STEP167 phase_fit, member architecture carried into the workers. Writes member_dir/lp-models.joblib + fit.json."""
    cfg = dict(reg['members'][member])
    W = member_dir(member, reg); (W / 'batches').mkdir(parents=True, exist_ok=True)
    if (reg.get('reuse') or {}).get(member):
        print('member', member, 'is REUSED from', W, '- not refitted'); return json.loads((W / 'fit.json').read_text(encoding='utf-8'))
    mp, fp = W / 'lp-models.joblib', W / 'fit.json'
    if fp.exists():
        r = json.loads(fp.read_text(encoding='utf-8')); assert r['models_sha256'] == sha(mp)
        print('fit already complete for', member, flush=True)
        return r
    t0 = time.perf_counter()
    S108, _ = S167.modules()
    CSP, KEYS, TOP = S108.CSP, S108.KEYS, S108.TOP
    bundle = joblib.load(S108.OUT / 'stage2-models.joblib')
    rcols, struct = S167.audit_columns()
    for f in FAMS:
        cols = bundle['cols'][f]
        assert cols[-61:] == S108.STATE + S108.RS
        assert set(cols[:-61]) == set(struct + S167.POOLF + (rcols if f != CSP.CI else [])), f
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
    print(member, 'S* refit equals step-108; rows', data.height, 'pairs', data['pair_id'].n_unique(), round(time.perf_counter() - t0, 1), flush=True)
    X, fam = S167.xmatrix(data, rcols, struct, 'behavior_family')
    top = (data['s1_rank'] <= TOP).to_numpy(); QIX = np.flatnonzero(top)
    Q = data[QIX]
    C = S167.cmatrix(Q, S108)
    pairs, tables = data['pair_id'].to_numpy(), data['table_id'].to_numpy()
    upairs, pfirst = np.unique(pairs, return_index=True); ptab = tables[pfirst]
    rowmask = lambda sel: np.isin(pairs, upairs[sel])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(upairs)), groups=ptab))
    rel = data['relevance'].to_numpy()
    init_args = (X.astype(np.float32), C.astype(np.float32), QIX, fam, rel, pairs, tables)
    jobs = [(cfg, (('oof', k), 'joint', s, rowmask(tr), rowmask(te))) for k, (tr, te) in enumerate(folds) for s in S167.SEEDS_OOF]
    with Pool(12, initializer=S165.pool_init, initargs=init_args) as pool:
        full = pool.map(fit_full_cfg, [(cfg, s) for s in S167.SEEDS_FULL], chunksize=1)
        oof = pool.map(fit_job_cfg, jobs, chunksize=1)
    Dd = {'d': data.select('pair_id'), 'fam': fam, 'QIX': QIX, 'Q': Q.select('pair_id')}
    acc = np.zeros((len(QIX), len(NN))); cnt = np.zeros(len(QIX))
    for tag, m, s, rows, tq, p, S, trace, sec in oof:
        acc[tq] += S165.aligned_features(Dd, rows, tq, S, p); cnt[tq] += 1
    assert (cnt == len(S167.SEEDS_OOF)).all()
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
    y = rel[QIX]
    r = {'member': member, 'config': cfg, 'registration_sha256': sha(REG), 'models_sha256': sha(mp),
         'dev_rows': data.height, 'dev_pairs': int(len(upairs)), 'dev_top20_rows': int(len(QIX)),
         'sstar_refit_equals_step108': True, 'lp_receipts': receipts,
         'full_fit_best_epochs': [tr[2] for _, _, _, _, tr in full], 'oof_fits': len(oof),
         'oof_nn_p2_auc_top20': float(roc_auc_score(y, E[:, 5])),
         'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(fp, r)
    print('fit', json.dumps({k: r[k] for k in ('member', 'config', 'oof_nn_p2_auc_top20', 'seconds')}), flush=True)
    return r


def phase_score(reg, member):
    """Delegate to STEP167 scoring with WORK repointed; predict() is architecture-agnostic so no CONFIG is needed here."""
    W = member_dir(member, reg)
    if (reg.get('reuse') or {}).get(member):
        n = len(sorted((W / 'batches').glob('*-lp-scores.parquet')))
        print('member', member, 'REUSED:', n, 'scored batches already on disk'); assert n == 20; return
    assert (W / 'fit.json').exists(), 'fit the member first'
    old = S167.WORK
    try:
        S167.WORK = W
        S167.phase_score(json.loads((W / 'fit.json').read_text(encoding='utf-8'))['registration_sha256'])
    finally:
        S167.WORK = old


def phase_build(reg, reg_sha):
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
    pmap = {p: (a, b) for p, a, b in pl.read_csv(CSP.ROOT / 'data/evaluation_pairs.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    news, recs = [], []
    for i in range(20):
        name, tb = 'batch_%03d' % i, time.perf_counter()
        tops = []
        for m in members:
            W = member_dir(m, reg)
            rj, sc = W / 'batches' / (name + '.json'), W / 'batches' / (name + '-lp-scores.parquet')
            r = json.loads(rj.read_text(encoding='utf-8')); assert r['files'][sc.name] == sha(sc), (m, name)
            tops.append(pl.read_parquet(sc).sort(KEYS))
        ref = tops[0]
        for t in tops[1:]:
            assert t.select(KEYS).equals(ref.select(KEYS)), 'member key order differs in ' + name
            assert np.array_equal(t['v1'].to_numpy(), ref['v1'].to_numpy()), 'v1 must not depend on the member'
        ens = np.mean([t['v1_lp'].to_numpy() for t in tops], axis=0)
        bf = ref.select('pair_id', 'hand_id', pl.lit(0.0).alias('tie'), (pl.col('ss') + 10).alias('ss'), (pl.col('v1') + 10).alias('v1')).with_columns(pl.Series('v1_ens', ens + 10))
        sel = bf.select('pair_id').unique().sort('pair_id')
        fc, ties = S108.chronology_eval(bf.select('pair_id', 'hand_id', pl.col('ss').alias('score')))
        assert fc.select(KEYS).equals(bf.select(KEYS)) and ties == 0
        t1 = S104.align(fc, S104.type_attributes(fc.select(KEYS), pmap))
        cif = fc.select('pair_id').join(ci.with_columns(pl.lit(True).alias('f')), on='pair_id', how='left', maintain_order='left')['f'].fill_null(False).to_numpy()
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
                     'top5_set_change_by_family': {f: round(float(np.mean([s for s, g in zip(sets, fams) if g == f])), 4) for f in sorted(set(fams))},
                     's': round(time.perf_counter() - tb, 1)})
        print(json.dumps(recs[-1]), flush=True)
    repl = pl.concat(news).sort('pair_id')
    assert repl.height == repl['pair_id'].n_unique() == 16060
    rest = base.join(repl.select('pair_id'), on='pair_id', how='anti')
    swap = base.join(repl.select('pair_id'), on='pair_id', how='semi').drop(EV).join(repl, on='pair_id', validate='1:1')
    result = base.select('pair_id').join(pl.concat([rest, swap.select(base.columns)]), on='pair_id', validate='1:1', maintain_order='left')
    assert result.select('pair_id', 'risk_score', 'predicted_behavior').equals(base.select('pair_id', 'risk_score', 'predicted_behavior'))
    csv = OUT / reg['output_name']
    assert not csv.exists(), 'preserve a built file'
    result.write_csv(csv)
    A, B = basep.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
    assert len(A) == len(B)
    diff = [k for k in range(len(A)) if A[k] != B[k]]
    assert {A[k].split(b',')[0].decode() for k in diff} <= set(repl['pair_id'])
    for k in diff:
        assert A[k].split(b',')[:3] == B[k].split(b',')[:3], 'risk or behaviour changed'
    val = validate_submission(csv, CSP.ROOT / 'data/evaluation_pairs.csv', CSP.ROOT / 'data/hands.parquet', CSP.ROOT / 'data/seats.parquet', CSP.ROOT / 'data/sample_submission.csv')
    assert not val.errors, val
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'members': members, 'combiner': 'SCORE_mean over member v1_lp, decoded once',
           'output': str(csv), 'output_sha256': sha(csv), 'base_sha256': sha(basep), 'rows_changed_vs_base': len(diff),
           'validation': str(val)[:300], 'batches': recs, 'lists_changed': int(sum(r['lists_changed'] for r in recs)),
           'top5_set_change_mean': float(np.average([r['top5_set_change_mean'] for r in recs], weights=[r['selected_pairs'] for r in recs])),
           'protected_unchanged': True, 'no_submission_actions': True,
           'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'receipt_step188.json', rec)
    print('DONE', json.dumps({k: rec[k] for k in ('output_sha256', 'rows_changed_vs_base', 'lists_changed', 'top5_set_change_mean', 'seconds')}), flush=True)


def main():
    phase, reg_sha = sys.argv[1], sys.argv[2]
    reg = read_reg(reg_sha)
    OUT.mkdir(parents=True, exist_ok=True)
    if phase == 'build':
        phase_build(reg, reg_sha)
        return
    only = [m for m in os.environ.get('STEP188_MEMBERS', '').split(',') if m] or list(reg['members'])
    for m in only:
        assert m in reg['members'], m
        if phase == 'fit':
            phase_fit(reg, m)
        elif phase == 'score':
            phase_score(reg, m)


if __name__ == '__main__':
    main()
