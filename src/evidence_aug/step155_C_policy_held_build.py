"""Step 155 part A (Claude): label-free build for the held folds 0-2 check of the STEP154 blend (G1 policy at 25% inside each list).
1. Replays the audit v1 scores exactly with a generic copy of the step-124 view recipe (S* audit fold models refit, 20 thinned + 20 padded views).
2. Per held fold: rebuilds the step-98 held frame, replays the saved S*_f held scores from the saved boosters, scores the same 40 label-free
   views of the held pools (v1_f), decodes with T1 off for predicted coordinated_isolation, trains G1/G0 on all 162 audit lists
   (10 seeds each, STEP152 config) and writes one score file per arm. Held truth is never read here (step155b reads it once).
Registration: experiments/rank_policy_grpo_C_20260914/registration155.json. STEP155_SMOKE=1: fold 0, 2 views, no audit replay, no writes."""
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
from evidence_aug.frame import context
from evidence_aug.fold import fold_roles
from evidence_aug.step10_chronology_fit import add_chronology
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align, less_than_five
from evidence_aug import step152_rank_policy as RP
from contestable_state_features import FEATURES as STATE_FEATURES
import retrospective_strength_features as RSF

EXP = CV / 'experiments/rank_policy_grpo_C_20260914'
HOUT = EXP / 'held155'
REG = EXP / 'registration155.json'
S98 = CV / 'experiments/prefix_strength_confirmation_C_20260913'
S104 = CV / 'experiments/typed_listing_decoder_C_20260913'
CODEX_CONF = HERE / 'experiments/contestable_state_confirmation_20260912'
STRUCT_FILE = CV / 'experiments/family_stage2_20260911/dev-structural-features.parquet'
K = ['pair_id', 'hand_id']
CI = 'coordinated_isolation'
POOLF = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof_rank']
STATE, RS = list(STATE_FEATURES), list(RSF.FEATURES)
THIN_SEEDS, PAD_SEEDS, U, M = (1241, 2241), (1242, 2242), .5, 10
SEEDS = [15200 + i for i in range(10)]
BLENDS = {'G1_a25': ('G1', .25), 'G0_a25': ('G0', .25), 'G1_a10': ('G1', .10), 'G1_a100': ('G1', 1.), 'G0_a100': ('G0', 1.)}
SMOKE = os.environ.get('STEP155_SMOKE') == '1'


def lgt(x, eps):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def pool_views(POOL, Q, src, ctxc, kind, u, m, seed):
    """step-124 views / pads(train=False): m frames aligned to Q (sorted by K) holding POOLF + ctxc recomputed on the label-free view."""
    top0, parts = POOL['top0'].to_numpy(), []
    cmax = float(POOL['chrono_rank'].max()) + 1
    for c in range(m):
        r = np.random.default_rng([seed, c]).random(POOL.height)
        pick = (r < u) & ~top0
        if kind == 'thin':
            parts.append(POOL.filter(pl.Series(~pick)).with_columns(pl.lit(c).alias('copy')))
        else:
            dupl = POOL.filter(pl.Series(pick)).with_columns((pl.col('chrono_rank') + cmax * 10).alias('chrono_rank'), pl.lit(False).alias('top0'))
            parts.append(pl.concat([POOL, dupl]).with_columns(pl.lit(c).alias('copy')))
    a = pl.concat(parts).with_columns(pl.concat_str([pl.col('pair_id'), pl.col('copy').cast(pl.String)], separator='#').alias('pc'))
    a = a.with_columns((pl.col('chrono_rank').rank('ordinal').over('pc') - 1).cast(pl.Float64).alias('chrono_rank'),
                       pl.len().over('pc').cast(pl.Float64).alias('n_hands'),
                       pl.col('oof').rank('ordinal', descending=True).over('pc').cast(pl.Float64).alias('oof_rank'))
    a = a.with_columns((pl.col('chrono_rank') / (pl.col('n_hands') - 1)).alias('chrono_frac'))
    a = context(a.rename({'pair_id': 'true_pair'}).rename({'pc': 'pair_id'}), src).drop('pair_id').rename({'true_pair': 'pair_id'})
    a = a.filter(pl.col('top0')).select(K + ['copy'] + POOLF + ctxc)
    out = []
    for c in range(m):
        v = a.filter(pl.col('copy') == c).drop('copy').sort(K)
        assert v.select(K).equals(Q.select(K)) and np.array_equal(v['oof_rank'].to_numpy(), Q['oof_rank'].to_numpy())
        out.append(v)
    return out


def with_view(Q, v, ctxc):
    return Q.drop(POOLF + ctxc).join(v, on=K, how='left', validate='1:1', maintain_order='left')


def audit_replay():
    """Bit-level replay of experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet (S_star and v1) with pool_views."""
    t0 = time.perf_counter()
    d, shared, rcols, _ = load_current()
    top = (d['oof_rank'] <= 20).to_numpy()
    ctxc = [c for c in rcols if c.startswith('context_')]
    src = [c[8:] for c in ctxc]
    cs = pl.read_parquet(HERE / 'research/contestable_state_20260912/features.parquet').sort(K)
    pf = pl.read_parquet(HERE / 'research/retrospective_strength_20260912/prefix.parquet').sort(K)
    cs_cols, pf_cols = [c for c in cs.columns if c not in K], [c for c in pf.columns if c not in K]
    Q = d.filter(pl.col('oof_rank') <= 20).join(cs, on=K, validate='1:1').join(pf, on=K, validate='1:1').sort(K)
    QIX = np.flatnonzero(top)
    assert d[QIX].select(K).equals(Q.select(K))
    cols = lambda f: shared + (rcols if f != CI else []) + cs_cols + pf_cols
    MD, SS = {}, np.zeros(Q.height)
    folds = {}
    for f in FAMILIES:
        ix = np.flatnonzero((Q['behavior_family'] == f).to_numpy())
        q = Q[ix]; y = q['relevance'].to_numpy().astype(int)
        folds[f] = (ix, list(GroupKFold(n_splits=5).split(np.zeros((len(ix), 1)), y, q['table_id'].to_numpy())))
        z = q.select(cols(f)).to_numpy().astype(float)
        for k, (tr, te) in enumerate(folds[f][1]):
            MD[(f, k)] = lgb.LGBMClassifier(**CLF, random_state=11).fit(z[tr], y[tr])
            SS[ix[te]] = MD[(f, k)].predict_proba(z[te])[:, 1] + 10
    saved = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
    assert saved.select(K).equals(d.select(K))
    ss_saved, v1_saved = saved['S_star'].to_numpy()[QIX], saved['v1'].to_numpy()[QIX]
    assert np.array_equal(SS, ss_saved), 'audit S* refit differs'

    def score_view(v):
        p = SS.copy(); qv = with_view(Q, v, ctxc)
        for f in FAMILIES:
            ix, fk = folds[f]
            zv = qv[ix].select(cols(f)).to_numpy().astype(float)
            for k, (tr, te) in enumerate(fk):
                p[ix[te]] = MD[(f, k)].predict_proba(zv[te])[:, 1] + 10
        return p
    POOL = d.select(K + ['chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))
    zero = pool_views(POOL, Q, src, ctxc, 'pad', 0., 1, 0)[0]
    zero_diff = max(float((zero[c].cast(pl.Float64) - Q[c].cast(pl.Float64)).abs().max()) for c in POOLF + ctxc)
    TH = np.zeros(Q.height); PD = np.zeros(Q.height)
    for s in THIN_SEEDS:
        for v in pool_views(POOL, Q, src, ctxc, 'thin', U, M, s):
            TH += score_view(v)
    for s in PAD_SEEDS:
        for v in pool_views(POOL, Q, src, ctxc, 'pad', U, M, s):
            PD += score_view(v)
    V1 = (TH + PD + SS) / (2 * len(THIN_SEEDS) * M + 1)
    rec = {'S_star_exact': True, 'v1_exact': bool(np.array_equal(V1, v1_saved)), 'v1_max_abs_diff': float(np.abs(V1 - v1_saved).max()),
           'u0_pad_view_max_abs_diff': zero_diff, 'seconds': round(time.perf_counter() - t0, 1)}
    print('audit replay', json.dumps(rec), flush=True)
    assert rec['v1_max_abs_diff'] <= 1e-12, rec
    return rec


def held_fold(fold, structural, struct, nviews):
    t0 = time.perf_counter()
    data, features, pools, identity = load_fold(fold)
    data = add_chronology(data).join(structural, on=K, validate='1:1').sort(K)
    held = data.filter(pl.col('table_id').is_in(pools)).sort(K)
    del data
    held = held.join(fold_roles(fold), on='pair_id', validate='m:1').sort(K)
    s1 = pl.read_parquet(S98 / 'scores' / f'fold{fold}-stage1_only.parquet').sort(K)
    assert s1.select(K).equals(held.select(K))
    sh = s1['score'].to_numpy()
    assert np.array_equal(held['pot_bb'].to_numpy().astype(np.float32), held['pot_bb_right'].to_numpy().astype(np.float32))
    held = held.with_columns(pl.Series('oof', sh), pl.col('pot_bb_right').alias('pot_bb'))
    held = held.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    shared = struct + ['chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
    extra = [f for f in features if not f.startswith('family_') and f not in struct]
    held = held.with_columns(pl.col(extra).fill_null(-999.))
    ctxc = [c for c in extra if c.startswith('context_')]
    src = [c[8:] for c in ctxc]
    assert len(ctxc) == 136 and all(s in held.columns for s in src)
    state = pl.read_parquet(CODEX_CONF / f'fold{fold}-state-features.parquet')
    pref = pl.read_parquet(S98 / f'fold{fold}-prefix-features.parquet')
    Q = held.filter(pl.col('oof_rank') <= 20).join(state, on=K, how='left', validate='1:1').join(pref, on=K, how='left', validate='1:1').sort(K)
    QIX = np.flatnonzero((held['oof_rank'] <= 20).to_numpy())
    assert held[QIX].select(K).equals(Q.select(K))
    fams = Q['predicted_behavior'].to_numpy()
    cols = lambda f: shared + (extra if f != CI else []) + STATE + RS
    boosters = {f: lgb.Booster(model_file=str(S98 / 'models' / f'fold{fold}' / f'{f}-S_star.txt')) for f in FAMILIES}

    def score_frame(qf):
        p = np.full(Q.height, np.nan)
        for f in FAMILIES:
            m = fams == f
            if m.any():
                x = qf.filter(pl.Series(m)).select(cols(f)).to_numpy().astype(float)
                assert np.isfinite(x).all()
                p[m] = boosters[f].predict(x, num_threads=2) + 10
        return p
    SS = score_frame(Q)
    saved = pl.read_parquet(S98 / 'scores' / f'fold{fold}-S_star.parquet').sort(K)
    assert saved.select(K).equals(held.select(K))
    fam_rows = ~np.isnan(SS)
    assert np.array_equal(SS[fam_rows], saved['score'].to_numpy()[QIX][fam_rows]), f'fold {fold} S* held replay differs'
    POOL = held.select(K + ['chrono_rank', 'oof', 'oof_rank'] + src).with_columns(pl.col('oof_rank').le(20).alias('top0'))
    zero = pool_views(POOL, Q, src, ctxc, 'pad', 0., 1, 0)[0]
    zc = with_view(Q, zero, ctxc)
    zero_exact = bool(np.array_equal(score_frame(zc)[fam_rows], SS[fam_rows]))
    TH = np.zeros(Q.height); PD = np.zeros(Q.height)
    for s in THIN_SEEDS:
        for v in pool_views(POOL, Q, src, ctxc, 'thin', U, nviews, s):
            TH += score_frame(with_view(Q, v, ctxc))
    for s in PAD_SEEDS:
        for v in pool_views(POOL, Q, src, ctxc, 'pad', U, nviews, s):
            PD += score_frame(with_view(Q, v, ctxc))
    V1 = (TH + PD + SS) / (2 * len(THIN_SEEDS) * nviews + 1)
    ss_full, v1_full = saved['score'].to_numpy().copy(), saved['score'].to_numpy().copy()
    v1_full[QIX[fam_rows]] = V1[fam_rows]
    frame = held.select(K + ['table_id', 'predicted_behavior'])
    rec = {'fold': fold, 'held_pairs': held['pair_id'].n_unique(), 'top20_rows': int(len(QIX)), 'stage2_rows': int(fam_rows.sum()),
           'S_star_held_replay_exact': True, 'u0_view_rescore_exact': zero_exact, 'views': 2 * len(THIN_SEEDS) * nviews,
           'mean_abs_v1_minus_S_star': float(np.abs(V1[fam_rows] - SS[fam_rows]).mean()), 'identity': identity,
           'seconds_views': round(time.perf_counter() - t0, 1)}
    assert zero_exact
    del held, Q, POOL
    return frame, ss_full, v1_full, rec


def build_features(rows_ix, v1, ss, inc, typed, cr, fam):
    """STEP152 list features (same order and transforms as step152 load()) for lists whose 20 candidates carry stage-2 scores."""
    Q = len(rows_ix)
    X, base = np.zeros((Q, 20, 13)), np.zeros((Q, 20))
    for qi, ix in enumerate(rows_ix):
        p, ps, t = v1[ix] - 10, ss[ix] - 10, typed[ix]
        order = np.lexsort((cr[ix], ~t))
        q = np.empty(20); q[order] = less_than_five(p[order])
        prior = np.empty(20); prior[order] = np.r_[0, np.cumsum(p[order])[:-1]]
        tpos = np.empty(20); tpos[np.argsort(cr[ix], kind='stable')] = np.arange(20) / 19
        prank = np.empty(20); prank[np.argsort(-p, kind='stable')] = np.arange(20) / 19
        f = fam[ix[0]]
        X[qi] = np.column_stack([lgt(p, 1e-6), lgt(p * q, 1e-6), q, t, tpos, prank, prior / 5, lgt(ps, 1e-6) - lgt(p, 1e-6),
                                 np.full(20, f == FAMILIES[0]), np.full(20, f == FAMILIES[1]), np.full(20, f == FAMILIES[2]),
                                 np.full(20, p.sum() / 5), np.full(20, t.sum() / 20)])
        base[qi] = lgt(inc[ix] - 10, 1e-15)
    return X, base


def z(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)


def main():
    start = time.perf_counter()
    if not SMOKE:
        reg = json.loads(REG.read_text(encoding='utf-8'))
        for name, v in reg['inputs'].items():
            assert sha(ROOT / v['path']) == v['sha256'], name
        assert sha(Path(__file__)) == reg['code']['build_sha256'] and sha(CV / 'src/evidence_aug/step152_rank_policy.py') == reg['code']['policy_module_sha256']
        assert not (HOUT / 'predictions-frozen.json').exists(), 'preserve a completed freeze'
        HOUT.mkdir(exist_ok=True); (HOUT / 'scores').mkdir(exist_ok=True)
    protected = check_protected()
    replay = audit_replay() if not SMOKE else {'skipped': 'smoke'}
    # policy: all 162 audit lists, the STEP152 features and config
    import step152_C_rank_policy_grpo_audit as S152
    A = S152.load()
    Xa, ba, ra, na = A['X'], A['base'], A['rel'], A['n_norm']
    mu, sd = Xa.reshape(-1, 13).mean(0), Xa.reshape(-1, 13).std(0) + 1e-9
    seeds = SEEDS if not SMOKE else SEEDS[:1]
    policies = {arm: [RP.train(arm, ba, (Xa - mu) / sd, ra, na, s)[0] for s in seeds] for arm in ('G1', 'G0')}
    print('policies trained', round(time.perf_counter() - start, 1), flush=True)
    structural = pl.read_parquet(STRUCT_FILE)
    struct = [c for c in structural.columns if c not in K]
    players = {r[0]: (r[1], r[2]) for r in pl.read_csv(ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    receipts = []
    for fold in ((0,) if SMOKE else (0, 1, 2)):
        frame, ss_full, v1_full, rec = held_fold(fold, structural, struct, 1 if SMOKE else M)
        fr = chronology(frame.select(K + ['table_id']).with_columns(pl.Series('score', ss_full)))
        assert fr.select(K).equals(frame.select(K))
        attrs = pl.read_parquet(S104 / f'fold{fold}-type-attributes.parquet')
        need = fr.filter(pl.col('score') > 5).select(K)
        assert need.join(attrs, on=K, how='anti').height == 0, 'type attributes missing for stage-2 rows'
        t1 = align(fr, attrs)
        fam = frame['predicted_behavior'].to_numpy()
        typed = t1 & (fam != CI)
        dec_ss = decode(fr, typed)
        inc = decode(fr.with_columns(pl.Series('score', v1_full)), typed)
        top = np.flatnonzero(v1_full > 5)
        pid = frame['pair_id'].to_numpy()[top]
        _, first = np.unique(pid, return_index=True)
        rows_ix = [top[g] for g in np.split(np.arange(len(top)), first[1:])]
        assert all(len(ix) == 20 for ix in rows_ix)
        cr = fr['chrono_rank'].to_numpy()
        X, base = build_features(rows_ix, v1_full, ss_full, inc, typed, cr, fam)
        Xs = (X - mu) / sd
        S = {arm: np.mean([pol.forward(base, Xs) for pol in policies[arm]], axis=0) for arm in policies}
        arms = {'dec_S_star_ci_untyped': dec_ss, 'incumbent': inc}
        changed = {}
        for name, (arm, a) in BLENDS.items():
            blend = (1 - a) * z(base) + a * z(S[arm])
            full = inc.copy()
            full[np.concatenate(rows_ix)] = 10 + blend.reshape(-1) / 1000
            arms[name] = full
            changed[name] = float(np.mean([set(np.argsort(-b)[:5]) != set(np.argsort(-o)[:5]) for b, o in zip(blend, base)]))
        rec.update({'lists_with_policy': len(rows_ix), 'T1_rows_typed': int(typed[top].sum()), 'top5_set_changed_share': changed,
                    'seconds': round(time.perf_counter() - start, 1)})
        print('fold', fold, json.dumps({k: v for k, v in rec.items() if k != 'identity'}), flush=True)
        if not SMOKE:
            for name, p in arms.items():
                assert np.isfinite(p).all() and np.array_equal(p > 5, inc > 5)
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
