"""Step 160 part A (Claude): label-free build for the held folds 0-2 check of route 1: directed_transfer stage 2 trained with fewer unlisted
negatives (STEP144 arm K0R: in every capped training list, as many random top-20 negatives dropped as the list has top-20 negatives after its
last listed hand).
1. Audit replay: the K0R mask rebuilt here reproduces STEP144's audit mask counts and its directed_transfer K0R_noview scores exactly.
2. Per held fold: rebuilds the step-98 training frame (inner OOF stage 1), refits S*_f directed_transfer and checks it replays the saved booster on
   the held rows exactly; draws the K0R mask from the training lists only; fits DT-K0R_f with the same parameters.
3. Scores the step-155 held frame with the same 40 label-free views under S*_f and under S*_f with DT-K0R_f, and decodes with T1 off for predicted
   CI. The incumbent and dec_S_star_ci_untyped files must replay held155 exactly. Held truth is never read here (step160b reads it once).
Registration: experiments/dt_negdrop_held_C_20260915/registration160.json. STEP160_SMOKE=1: audit replay, fold 0 with 4 views (no-view replay only), no writes."""
import os
for _k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
sys.path.insert(0, str(CV / 'src/evidence_aug'))
import time
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from evidence_aug.frame import training_relevance, attach_relevance, family_columns, score
from evidence_aug.fold import fold_roles
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology
from evidence_aug.step14_audit_truncation import split_of, SPLITS
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align
from evidence_aug import step155_C_policy_held_build as S155
from contestable_state_features import FEATURES as STATE_FEATURES
import retrospective_strength_features as RSF

EXP = CV / 'experiments/dt_negdrop_held_C_20260915'
HOUT = EXP / 'held160'
REG = EXP / 'registration160.json'
S144 = CV / 'experiments/cap_censoring_a8_20260913'
K, CI, DT = ['pair_id', 'hand_id'], 'coordinated_isolation', 'directed_transfer'
STATE, RS = list(STATE_FEATURES), list(RSF.FEATURES)
SMOKE = os.environ.get('STEP160_SMOKE') == '1'
EVID = pl.read_csv(ROOT / 'data/development_evidence.csv').select(K + ['evidence_rank'])


def k0r(frame, top):
    """STEP144 masks on training rows. frame: all hands of the training queries with relevance from their own lists; top: their top-20 rows
    sorted by K. AMB = top-20 negatives after the list's last listed hand in capped (5-hand) lists; K0R = as many random top-20 negatives of
    the same query, drawn with seed [1392, index of the query in the sorted query ids of frame]."""
    ev = EVID.join(frame.select('pair_id').unique(), on='pair_id', how='semi')  # only the lists of the frame's own (training) queries
    a = frame.select(K + ['chrono_rank', 'relevance']).join(ev, on=K, how='left', validate='1:1')
    assert ev.height == int(a['evidence_rank'].is_not_null().sum())
    assert int(a['evidence_rank'].is_not_null().sum()) == int(frame['relevance'].sum())
    ls = a.filter(pl.col('evidence_rank').is_not_null()).sort('pair_id', 'evidence_rank').group_by('pair_id', maintain_order=True).agg(
        pl.len().alias('L'), pl.col('chrono_rank').last().alias('t_cut'))
    qm = top.select(K + ['relevance', 'chrono_rank']).join(ls, on='pair_id', how='left', validate='m:1')
    assert qm.select(K).equals(top.select(K)) and qm['L'].null_count() == 0
    amb = ((qm['relevance'] == 0) & (qm['L'] == 5) & (qm['chrono_rank'] > qm['t_cut'])).to_numpy()
    pid, rel, cap = top['pair_id'].to_numpy(), top['relevance'].to_numpy(), (qm['L'] == 5).to_numpy()
    r0 = np.zeros(len(pid), bool)
    for qi, q in enumerate(sorted(set(frame['pair_id'].unique().to_list()))):
        rows = np.flatnonzero(pid == q)
        n = int(amb[rows].sum()) if len(rows) else 0
        if n and cap[rows[0]]:
            r0[np.random.default_rng([1392, qi]).choice(rows[rel[rows] == 0], n, replace=False)] = True
    assert int(r0.sum()) == int(amb.sum()) and not (r0 & (rel == 1)).any()
    return amb, r0


def audit_replay():
    t0 = time.perf_counter()
    d, shared, rcols, _ = load_current()
    cs = pl.read_parquet(HERE / 'research/contestable_state_20260912/features.parquet').sort(K)
    pf = pl.read_parquet(HERE / 'research/retrospective_strength_20260912/prefix.parquet').sort(K)
    cs_cols = [c for c in cs.columns if c not in K]
    pf_cols = json.loads((HERE / 'research/retrospective_strength_20260912/freeze.json').read_text(encoding='utf-8'))['features']
    assert pf.columns == K + pf_cols and len(cs_cols) == 29
    Q = d.filter(pl.col('oof_rank') <= 20).join(cs, on=K, validate='1:1').join(pf, on=K, validate='1:1').sort(K)
    QIX = np.flatnonzero((d['oof_rank'] <= 20).to_numpy())
    assert d[QIX].select(K).equals(Q.select(K))
    amb, r0 = k0r(d, Q)
    fam = Q['behavior_family'].to_numpy()
    cnt = lambda m: {f: int(m[fam == f].sum()) for f in FAMILIES}
    assert cnt(amb) == {CI: 309, DT: 573, 'soft_play': 320} and cnt(r0) == cnt(amb), (cnt(amb), cnt(r0))
    saved144 = json.loads((S144 / 'results_step144.json').read_text(encoding='utf-8'))
    assert sha(S144 / 'audit_scores_step144.parquet') == saved144['scores_sha256']
    saved = pl.read_parquet(S144 / 'audit_scores_step144.parquet').sort(K)
    assert saved.select(K).equals(d.select(K))
    ix = np.flatnonzero(fam == DT)
    q = Q[ix]
    y, g = q['relevance'].to_numpy().astype(int), q['table_id'].to_numpy()
    z = q.select(shared + rcols + cs_cols + pf_cols).to_numpy().astype(float)
    p = np.full(len(ix), np.nan)
    dropped = []
    for tr, te in GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, g):
        keep = ~r0[ix[tr]]
        dropped.append(int((~keep).sum()))
        p[te] = lgb.LGBMClassifier(**CLF, random_state=11).fit(z[tr][keep], y[tr][keep]).predict_proba(z[te])[:, 1] + 10
    ref = saved['K0R_noview'].to_numpy()[QIX][ix]
    rec = {'mask_counts_AMB': cnt(amb), 'mask_counts_K0R': cnt(r0), 'dt_K0R_noview_exact': bool(np.array_equal(p, ref)),
           'dt_K0R_noview_max_abs_diff': float(np.abs(p - ref).max()), 'dt_dropped_by_fold': dropped,
           'dt_dropped_matches_step144': dropped == [saved144['mask_info']['K0R'][f'{DT}|{k}']['dropped'] for k in range(5)],
           'seconds': round(time.perf_counter() - t0, 1)}
    print('audit replay', json.dumps(rec), flush=True)
    assert rec['dt_K0R_noview_exact'] and rec['dt_dropped_matches_step144'], rec
    return rec


def fold_run(fold, structural, struct, nviews):
    t0 = time.perf_counter()
    data, features, pools, identity = load_fold(fold)
    data = add_chronology(data).join(structural, on=K, validate='1:1').sort(K)
    held = data.filter(pl.col('table_id').is_in(pools)).sort(K)
    train = data.filter(~pl.col('table_id').is_in(pools)).sort(K)
    del data
    # training frame exactly as step 98 (training lists only)
    evidence = training_relevance(set(train['table_id']))
    assert not set(evidence['pair_id']) & set(held['pair_id'])
    train = attach_relevance(train, evidence).sort(K)
    split = np.asarray([split_of(p) for p in train['table_id']])
    st = np.full(train.height, np.nan)
    for s in range(SPLITS):
        te = np.flatnonzero(split == s)
        inner = fit_ranker(train.filter(pl.Series(split != s)), features + ['chrono_frac'])
        st[te] = score(inner, family_columns(train[te], 'behavior_family'), features + ['chrono_frac'])
    assert np.isfinite(st).all()
    assert np.array_equal(train['pot_bb'].to_numpy().astype(np.float32), train['pot_bb_right'].to_numpy().astype(np.float32))
    train = train.with_columns(pl.Series('oof', st), pl.col('pot_bb_right').alias('pot_bb'))
    train = train.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    shared = struct + ['chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
    extra = [f for f in features if not f.startswith('family_') and f not in struct]
    train = train.with_columns(pl.col(extra).fill_null(-999.))
    state = pl.read_parquet(S155.CODEX_CONF / f'fold{fold}-state-features.parquet')
    pref = pl.read_parquet(S155.S98 / f'fold{fold}-prefix-features.parquet')
    top = train.filter(pl.col('oof_rank') <= 20).join(state, on=K, how='left', validate='1:1').join(pref, on=K, how='left', validate='1:1').sort(K)
    amb, r0 = k0r(train, top)
    cols = lambda f: shared + (extra if f != CI else []) + STATE + RS
    dtm = (top['behavior_family'] == DT).to_numpy()
    x, y = top.filter(pl.Series(dtm)).select(cols(DT)).to_numpy().astype(float), top.filter(pl.Series(dtm))['relevance'].to_numpy()
    assert np.isfinite(x).all()
    refit = lgb.LGBMClassifier(**CLF, random_state=11).fit(x, y)
    keep = ~r0[dtm]
    cand = lgb.LGBMClassifier(**CLF, random_state=11).fit(x[keep], y[keep])
    train_rec = {'train_pairs': train['pair_id'].n_unique(), 'dt_train_queries': top.filter(pl.Series(dtm))['pair_id'].n_unique(), 'dt_top20_rows': int(dtm.sum()),
                 'dt_negatives': int((y == 0).sum()), 'dt_dropped': int((~keep).sum()), 'dt_kept_negative_share': float(((y == 0) & keep).sum() / (y == 0).sum()),
                 'all_family_amb': {f: int(amb[(top['behavior_family'] == f).to_numpy()].sum()) for f in FAMILIES}}
    del train
    # held frame exactly as step 155
    held = held.join(fold_roles(fold), on='pair_id', validate='m:1').sort(K)
    s1 = pl.read_parquet(S155.S98 / 'scores' / f'fold{fold}-stage1_only.parquet').sort(K)
    assert s1.select(K).equals(held.select(K))
    sh = s1['score'].to_numpy()
    assert np.array_equal(held['pot_bb'].to_numpy().astype(np.float32), held['pot_bb_right'].to_numpy().astype(np.float32))
    held = held.with_columns(pl.Series('oof', sh), pl.col('pot_bb_right').alias('pot_bb'))
    held = held.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    held = held.with_columns(pl.col(extra).fill_null(-999.))
    ctxc = [c for c in extra if c.startswith('context_')]
    src = [c[8:] for c in ctxc]
    assert len(ctxc) == 136 and all(s in held.columns for s in src)
    Q = held.filter(pl.col('oof_rank') <= 20).join(state, on=K, how='left', validate='1:1').join(pref, on=K, how='left', validate='1:1').sort(K)
    QIX = np.flatnonzero((held['oof_rank'] <= 20).to_numpy())
    assert held[QIX].select(K).equals(Q.select(K))
    fams = Q['predicted_behavior'].to_numpy()
    boosters = {f: lgb.Booster(model_file=str(S155.S98 / 'models' / f'fold{fold}' / f'{f}-S_star.txt')) for f in FAMILIES}
    hx = Q.filter(pl.Series(fams == DT)).select(cols(DT)).to_numpy().astype(float)
    refit_exact = bool(np.array_equal(refit.predict_proba(hx)[:, 1], boosters[DT].predict(hx, num_threads=2)))
    assert refit_exact, f'fold {fold}: S*_f directed_transfer refit does not replay the saved booster'
    arms = {'S': boosters, 'R': dict(boosters, **{DT: cand.booster_})}

    def score_frame(qf, bs):
        p = np.full(Q.height, np.nan)
        for f in FAMILIES:
            m = fams == f
            if m.any():
                xx = qf.filter(pl.Series(m)).select(cols(f)).to_numpy().astype(float)
                assert np.isfinite(xx).all()
                p[m] = bs[f].predict(xx, num_threads=2) + 10
        return p
    SS = {a: score_frame(Q, bs) for a, bs in arms.items()}
    saved = pl.read_parquet(S155.S98 / 'scores' / f'fold{fold}-S_star.parquet').sort(K)
    assert saved.select(K).equals(held.select(K))
    fam_rows = ~np.isnan(SS['S'])
    assert np.array_equal(SS['S'][fam_rows], saved['score'].to_numpy()[QIX][fam_rows]), f'fold {fold} S* held replay differs'
    assert np.array_equal(SS['R'][fams != DT], SS['S'][fams != DT])
    POOL = held.select(K + ['chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))
    TH, PD = {a: np.zeros(Q.height) for a in arms}, {a: np.zeros(Q.height) for a in arms}
    for s in S155.THIN_SEEDS:
        for v in S155.pool_views(POOL, Q, src, ctxc, 'thin', S155.U, nviews, s):
            qv = S155.with_view(Q, v, ctxc)
            for a, bs in arms.items():
                TH[a] += score_frame(qv, bs)
    for s in S155.PAD_SEEDS:
        for v in S155.pool_views(POOL, Q, src, ctxc, 'pad', S155.U, nviews, s):
            qv = S155.with_view(Q, v, ctxc)
            for a, bs in arms.items():
                PD[a] += score_frame(qv, bs)
    nv = 2 * len(S155.THIN_SEEDS) * nviews
    V1 = {a: (TH[a] + PD[a] + SS[a]) / (nv + 1) for a in arms}
    full = {}
    for name, arr in (('ss_S', SS['S']), ('v1_S', V1['S']), ('ss_R', SS['R']), ('v1_R', V1['R'])):
        f_ = saved['score'].to_numpy().copy()
        f_[QIX[fam_rows]] = arr[fam_rows]
        full[name] = f_
    frame = held.select(K + ['table_id', 'predicted_behavior'])
    fr = chronology(frame.select(K + ['table_id']).with_columns(pl.Series('score', full['ss_S'])))
    assert fr.select(K).equals(frame.select(K))
    attrs = pl.read_parquet(S155.S104 / f'fold{fold}-type-attributes.parquet')
    assert fr.filter(pl.col('score') > 5).select(K).join(attrs, on=K, how='anti').height == 0
    t1 = align(fr, attrs)
    famf = frame['predicted_behavior'].to_numpy()
    typed = t1 & (famf != CI)
    out = {'dec_S_star_ci_untyped': decode(fr, typed), 'incumbent': decode(fr.with_columns(pl.Series('score', full['v1_S'])), typed),
           'K0R_noview': decode(fr.with_columns(pl.Series('score', full['ss_R'])), typed), 'K0R_v1': decode(fr.with_columns(pl.Series('score', full['v1_R'])), typed)}
    replay = {}
    for name in (('dec_S_star_ci_untyped',) if SMOKE else ('incumbent', 'dec_S_star_ci_untyped')):
        if True:
            ref = pl.read_parquet(S155.HOUT / 'scores' / f'fold{fold}-{name}.parquet')
            assert ref.select(K).equals(frame.select(K))
            replay[name] = bool(np.array_equal(ref['score'].to_numpy(), out[name]))
        assert all(replay.values()), replay
    pid = frame['pair_id'].to_numpy()
    dtp = famf == DT

    def top5_change(a, b):
        ch = []
        for p_ in np.unique(pid[dtp & (a > 5)]):
            m = pid == p_
            ch.append(set(np.argsort(-a[m])[:5]) != set(np.argsort(-b[m])[:5]))
        return float(np.mean(ch)) if ch else 0.0
    dtq = QIX[fam_rows & (fams == DT)]
    rec = {'fold': fold, 'identity': identity, 'training': train_rec, 'held_pairs': held['pair_id'].n_unique(), 'held_dt_routed_pairs': len(np.unique(pid[dtp])),
           'S_star_dt_refit_replays_saved_booster': refit_exact, 'held155_replay_exact': replay, 'views': nv,
           'dt_top5_set_changed_share_v1': top5_change(out['K0R_v1'], out['incumbent']), 'dt_top5_set_changed_share_noview': top5_change(out['K0R_noview'], out['dec_S_star_ci_untyped']),
           'dt_mean_prob_S': float((full['v1_S'][dtq] - 10).mean()), 'dt_mean_prob_R': float((full['v1_R'][dtq] - 10).mean()),
           'dt_v1_corr': float(np.corrcoef(full['v1_S'][dtq], full['v1_R'][dtq])[0, 1]), 'seconds': round(time.perf_counter() - t0, 1)}
    print('fold', fold, json.dumps({k: v for k, v in rec.items() if k != 'identity'}), flush=True)
    return frame, out, rec


def main():
    start = time.perf_counter()
    if not SMOKE:
        reg = json.loads(REG.read_text(encoding='utf-8'))
        for name, v in reg['inputs'].items():
            assert sha(ROOT / v['path']) == v['sha256'], name
        assert sha(Path(__file__)) == reg['code']['build_sha256']
        assert not (HOUT / 'predictions-frozen.json').exists(), 'preserve a completed freeze'
        HOUT.mkdir(parents=True, exist_ok=True); (HOUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    replay = audit_replay()
    structural = pl.read_parquet(S155.STRUCT_FILE)
    struct = [c for c in structural.columns if c not in K]
    receipts = []
    for fold in ((0,) if SMOKE else (0, 1, 2)):
        frame, out, rec = fold_run(fold, structural, struct, 1 if SMOKE else S155.M)
        if not SMOKE:
            for name, p in out.items():
                assert np.isfinite(p).all() and np.array_equal(p > 5, out['incumbent'] > 5)
                frame.select(K + ['table_id']).with_columns(pl.Series('score', p)).write_parquet(HOUT / 'scores' / f'fold{fold}-{name}.parquet')
            frame.select('pair_id', 'predicted_behavior').unique().sort('pair_id').write_csv(HOUT / 'scores' / f'fold{fold}-roles.csv')
        receipts.append(rec)
    assert check_protected() == protected
    if SMOKE:
        print('SMOKE done', round(time.perf_counter() - start, 1)); return
    write(HOUT / 'predictions-frozen.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'registration_sha256': sha(REG), 'audit_replay': replay, 'fold_receipts': receipts,
        'scores': {p.name: sha(p) for p in sorted((HOUT / 'scores').iterdir())}, 'protected_unchanged': True,
        'held_evidence_evaluation': False, 'seconds': round(time.perf_counter() - start, 1)})
    print('PREDICTIONS FROZEN', round(time.perf_counter() - start, 1), flush=True)


if __name__ == '__main__':
    main()
