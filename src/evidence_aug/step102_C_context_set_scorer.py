"""Step 102 [C]: Codex IDEA_BANK D, a learned five-hand set scorer, as a cross-fitted context block on B0.

Each top-20 candidate gets nine features relating it to the five best other candidates of its query under cross-fitted B0
scores: provisional rank and score gaps, chronological distance to that five, and leaf agreement with that five in the B0
trees. Training rows take provisional scores from inner group folds inside the outer training rows; held rows take them from
the fold's own B0 model. Controls: shuffled history, random anchor, rank-only stacking, a residual scorer, four pure-noise
blocks and stage1_only. The context loader, replication and summaries come from the step 100 script.
Registration: experiments/context_set_scorer_C_20260913/prediction-context-set-scorer.json (PREREG_SHA).

Modes:
  dry        a fresh scratch directory; the whole pipeline with a stand-in evaluator that computes no metric
  official   the single run into the experiment directory; refuses to overwrite
"""
import sys
sys.dont_write_bytecode = True
import os
for _k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Data/Projects/Money Making/work/poker')
CV = ROOT / 'Claude Version'
CODEX = ROOT / 'Codex Resume'
sys.path.insert(0, str(CV / 'src'))
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import evidence_aug.step100_C_outlier_exposure_density as s100
from evidence_aug.paths import check_protected

ac = s100.ac
CLF, FAMILIES = s100.CLF, s100.FAMILIES
EXP_DIR = CV / 'experiments' / 'context_set_scorer_C_20260913'
PREREG = EXP_DIR / 'prediction-context-set-scorer.json'
PREREG_SHA = 'cb6ce8aa0e68cb2617d455213620f8c79c9469d55ccc14cd4731bf9e65dea50a'
STEP100 = CV / 'src' / 'evidence_aug' / 'step100_C_outlier_exposure_density.py'
STEP100_SHA = 'cb289111c6e8ce3b3e4d571857f6e1f40aa131591904e121db54e61c96b8f5b0'
SCRATCH = Path('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s102')
AUDIT_NAME = 'audit-C-context-set-scorer.json'
FEATS = ('prov_rank', 'prov_z', 'prov_gap5', 'chrono_min', 'chrono_near3', 'chrono_centre', 'chrono_soft', 'leaf_mean', 'leaf_max')
VARIANTS = ('real', 'shuffled_history', 'random_anchor')
NOISE = ('B0_noise0', 'B0_noise1', 'B0_noise2', 'B0_noise3')
ARMS = ('incumbent', 'B0', 'S_star', 'B0_ctx', 'B0_ctx_shufhist', 'B0_ctx_randanchor', 'B0_rank', 'B0_resid') + NOISE + ('stage1_only',)
GATE = {'primary_mean_delta_min': .010, 'primary_ci95_lower_gt': 0., 'primary_family_min': -.010, 'primary_halves_min': 0.,
        'ctx_minus_shuffled_history_mean_gt': 0., 'primary_mean_gt_max_abs_noise_shift': True,
        'negative_control_ci95_upper_lt': 0.}
K_ANCHOR, NEAR, TAU = 5, 3, 5.


def fit(X, y, init=None):
    """CLF with random_state 11 (bit-identical across seeds: no bagging or column sampling); the model and its string."""
    m = lgb.LGBMClassifier(**CLF, random_state=11)
    if init is None:
        m.fit(X, y)
    else:
        m.fit(X, y, init_score=init)
    return m, m.booster_.model_to_string()


def replays(m, s, X):
    return bool(np.array_equal(m.predict(X, raw_score=True), lgb.Booster(model_str=s).predict(X, raw_score=True, num_threads=2)))


def agreement(m, X, pid):
    """Per query: the 20 x 20 fraction of trees in which two candidates share a leaf, in the model that scored them."""
    L = m.booster_.predict(X, pred_leaf=True)
    out = {}
    for p in np.unique(pid):
        q = L[pid == p]
        out[p] = (q[:, None, :] == q[None, :, :]).mean(axis=2)
    return out, int(L.shape[1])


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def query_features(zq, za, cq, nh, A):
    """Nine features for the 20 candidates of one query: zq real logits, za anchor logits, cq chronology, A leaf agreement."""
    order = np.argsort(-zq, kind='stable')
    rank = np.empty(20)
    rank[order] = np.arange(1, 21)
    zz = (zq - zq.mean()) / max(float(zq.std()), 1e-6)
    gap5 = zq - zq[order[K_ANCHOR - 1]]
    order_a = np.argsort(-za, kind='stable')
    out = np.empty((20, len(FEATS)))
    anchors = np.empty((20, K_ANCHOR), dtype=np.int64)
    for i in range(20):
        a = order_a[order_a != i][:K_ANCHOR]
        rest = np.flatnonzero(np.arange(20) != i)
        dist = np.abs(cq[i] - cq[a])
        soft = float(np.sum(softmax(za[rest]) * np.exp(-np.abs(cq[i] - cq[rest]) / TAU)))
        out[i] = [rank[i], zz[i], gap5[i], dist.min(), (dist <= NEAR).sum(), abs(cq[i] - np.median(cq[a])) / nh[i], soft,
                  A[i, a].mean(), A[i, a].max()]
        anchors[i] = a
    return out, anchors, rank


def perms(F):
    """Fixed within-query permutations for the shuffled-history and random-anchor variants (query index = sorted position)."""
    qs = np.unique(F['pid'])
    hist = {p: np.random.default_rng([102, F['fi'], qi]).permutation(20) for qi, p in enumerate(qs)}
    anch = {p: np.random.default_rng([102, 1, F['fi'], qi]).permutation(20) for qi, p in enumerate(qs)}
    return hist, anch


def context_fold(F, k, hist, anch):
    """Provisional logits and leaf agreement for every family row under outer fold k, then the nine features per variant."""
    tr, te = F['folds'][k]
    x, y, g, pid, hid = F['x'], F['y'], F['g'], F['pid'], F['hid']
    n = len(y)
    z, role, agr = np.full(n, np.nan), np.full(n, -9), {}
    c4 = bool(not set(g[tr]) & set(g[te]) and not set(hid[te]) & set(hid[tr]))
    m, s = fit(x[tr], y[tr])
    z[te] = m.predict(x[te], raw_score=True)
    a, trees = agreement(m, x[te], pid[te])
    agr.update(a)
    role[te] = -1
    p_held = m.predict_proba(x[te])[:, 1]
    rec = {'outer_fold': k, 'outer_model_sha': s100.h(s), 'outer_replay': replays(m, s, x[te]), 'outer_trees': trees, 'inner': []}
    te_set = set(te.tolist())
    for j, (itr, ite) in enumerate(GroupKFold(n_splits=5).split(x[tr], y[tr], g[tr])):
        a_tr, a_te = tr[itr], tr[ite]
        c4 &= bool(not set(g[a_tr]) & set(g[a_te]) and not set(hid[a_te]) & set(hid[a_tr]) and not set(a_tr.tolist()) & te_set)
        mi, si = fit(x[a_tr], y[a_tr])
        z[a_te] = mi.predict(x[a_te], raw_score=True)
        a, trees_i = agreement(mi, x[a_te], pid[a_te])
        agr.update(a)
        role[a_te] = j
        rec['inner'].append({'inner_fold': j, 'train_rows': int(len(a_tr)), 'held_rows': int(len(a_te)),
                             'replay': replays(mi, si, x[a_te]), 'trees': trees_i, 'sha': s100.h(si)})
    c4 &= bool(np.all(role[tr] >= 0) and np.all(role[te] == -1))
    cr = x[:, F['blocks']['stage1_chrono'][0]]
    nh = x[:, F['blocks']['stage1_chrono'][2]]
    feats = {v: np.full((n, len(FEATS)), np.nan) for v in VARIANTS}
    c3, nh_const = bool(np.all(np.isfinite(z))), True
    for p in np.unique(pid):
        ix = np.flatnonzero(pid == p)
        A = agr[p]
        c4 &= bool(len(set(role[ix].tolist())) == 1)
        c3 &= bool(len(ix) == 20 and A.shape == (20, 20) and np.all((A >= 0) & (A <= 1)))
        c3 &= bool(np.array_equal(np.sort(hist[p]), np.arange(20)) and np.array_equal(np.sort(anch[p]), np.arange(20)))
        nh_const &= bool(np.all(nh[ix] == nh[ix][0]))
        for v in VARIANTS:
            za = z[ix][anch[p]] if v == 'random_anchor' else z[ix]
            cv = cr[ix][hist[p]] if v == 'shuffled_history' else cr[ix]
            fq, anchors, rank = query_features(z[ix], za, cv, nh[ix], A)
            c3 &= bool(np.array_equal(np.sort(rank), np.arange(1, 21)))
            c3 &= bool(all(i not in anchors[i] and len(set(anchors[i].tolist())) == K_ANCHOR for i in range(20)))
            feats[v][ix] = fq
    c3 &= bool(all(np.all(np.isfinite(feats[v])) for v in VARIANTS))
    rec.update({'C3': c3, 'C4': c4, 'n_hands_constant_within_query': nh_const,
                'feature_means_real_all_rows': [round(float(v), 6) for v in feats['real'].mean(axis=0)]})
    return z, feats, p_held, rec


def build_context(ctx):
    t0 = time.perf_counter()
    cx, recs, frames = {}, [], []
    for f in FAMILIES:
        F = ctx['fam'][f]
        hist, anch = perms(F)
        for k in range(5):
            z, feats, p_held, rec = context_fold(F, k, hist, anch)
            cx[(f, k)] = (z, feats, p_held)
            recs.append({'family': f, **rec})
            n = len(z)
            held = np.zeros(n, dtype=bool)
            held[F['folds'][k][1]] = True
            for v in VARIANTS:
                cols = {c: feats[v][:, i] for i, c in enumerate(FEATS)}
                frames.append(pl.DataFrame({'family': [f] * n, 'outer_fold': [k] * n, 'variant': [v] * n, 'pair_id': F['pid'],
                                            'hand_id': F['hid'], 'held': held, 'provisional_logit': z, **cols}))
        print(f'  context {f}: {round(time.perf_counter() - t0, 1)}s', flush=True)
    summary = {'outer_fits': len(recs), 'inner_fits': sum(len(r['inner']) for r in recs),
               'replay_exact_all': all(r['outer_replay'] and all(i['replay'] for i in r['inner']) for r in recs),
               'C3': all(r['C3'] for r in recs), 'C4': all(r['C4'] for r in recs),
               'n_hands_constant_within_query': all(r['n_hands_constant_within_query'] for r in recs),
               'seconds': round(time.perf_counter() - t0, 1)}
    return cx, recs, pl.concat(frames), summary


def noise_blocks(ctx):
    return {(f, name): np.random.default_rng([102, 9, r, ctx['fam'][f]['fi']]).normal(size=(len(ctx['fam'][f]['y']), 9))
            for f in FAMILIES for r, name in enumerate(NOISE)}


def arm_inputs(F, f, name, z, feats, noise):
    """Feature matrix and init score of one arm for one family-fold; only the residual scorer has an init score."""
    if name == 'incumbent':
        return F['x_inc'], None
    if name == 'B0':
        return F['x'], None
    if name == 'S_star':
        return F['x_star'], None
    if name == 'B0_resid':
        return feats['real'], z
    if name == 'B0_rank':
        return np.column_stack([F['x'], feats['real'][:, :3]]), None
    if name in NOISE:
        return np.column_stack([F['x'], noise[(f, name)]]), None
    v = {'B0_ctx': 'real', 'B0_ctx_shufhist': 'shuffled_history', 'B0_ctx_randanchor': 'random_anchor'}[name]
    return np.column_stack([F['x'], feats[v]]), None


def run_arm(ctx, name, cx, noise):
    """Stage 2 as in steps 97-101 (CLF, random_state 11; top 20 get p + 10, tail oof - 1e6) on each arm's inputs."""
    t0 = time.perf_counter()
    d, top = ctx['d'], ctx['top']
    pred = d['oof'].to_numpy().copy() - 1e6
    fills = np.zeros(d.height, dtype=int)
    fits = []
    for f in FAMILIES:
        F = ctx['fam'][f]
        for k, (tr, te) in enumerate(F['folds']):
            assert not set(F['g'][tr]) & set(F['g'][te]) and not set(F['hid'][te]) & set(F['hid'][tr])
            z, feats, _ = cx[(f, k)]
            X, init = arm_inputs(F, f, name, z, feats, noise)
            m, s = fit(X[tr], F['y'][tr], None if init is None else init[tr])
            if init is None:
                p = m.predict_proba(X[te])[:, 1]
            else:
                p = 1. / (1. + np.exp(-(init[te] + m.predict(X[te], raw_score=True))))
            pred[F['ix'][te]] = p + 10
            fills[F['ix'][te]] += 1
            fits.append({'family': f, 'fold': k, 'n_features': int(X.shape[1]), 'replay': replays(m, s, X[te]), 'sha': s100.h(s)})
    rec = {'arm': name, 'fits': len(fits), 'replay_exact_all': all(x['replay'] for x in fits),
           'top20_filled_once': bool(np.all(fills[top] == 1) and np.all(fills[~top] == 0)),
           'tail_is_oof_minus_1e6': bool(np.array_equal(pred[~top], d['oof'].to_numpy()[~top] - 1e6)),
           'n_features_by_family': {x['family']: x['n_features'] for x in fits},
           'model_sha_by_family_fold': {f"{x['family']}:{x['fold']}": x['sha'] for x in fits},
           'seconds': round(time.perf_counter() - t0, 1)}
    print(f"  fit {name}: {rec['seconds']}s replay={rec['replay_exact_all']} filled_once={rec['top20_filled_once']}", flush=True)
    return pred, rec


def provisional_parity(ctx, cx, recs, preds, fits):
    """C8: the outer provisional model is the B0 arm model, and its held probabilities give the B0 arm scores bit-exactly."""
    res, ok = {}, True
    for r in recs:
        f, k = r['family'], r['outer_fold']
        F = ctx['fam'][f]
        rows = F['ix'][F['folds'][k][1]]
        same_model = r['outer_model_sha'] == fits['B0']['model_sha_by_family_fold'][f'{f}:{k}']
        same_score = bool(np.array_equal(cx[(f, k)][2] + 10, preds['B0'][rows]))
        res[f'{f}:{k}'] = {'same_model_string': same_model, 'held_scores_bit_exact': same_score}
        ok &= same_model and same_score
    return res, bool(ok)


def write_frozen(ctx, out, preds, cframe):
    sp, cp = out / 'scores-arms.parquet', out / 'context-features.parquet'
    assert not sp.exists() and not cp.exists()
    ctx['d'].select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(n, preds[n]) for n in ARMS]).write_parquet(sp)
    back = pl.read_parquet(sp)
    assert back.select(s100.KEYS).equals(ctx['keys']) and all(np.array_equal(back[n].to_numpy(), preds[n]) for n in ARMS)
    cframe.write_parquet(cp)
    return {'scores-arms.parquet': s100.fsha(sp), 'context-features.parquet': s100.fsha(cp)}


def evaluate_all(ctx, preds, EV):
    d = ctx['d']

    def cmp(cand, ref):
        return s100.summ(EV(d, {'incumbent': preds[ref], cand: preds[cand]}), cand)
    ev = {'primary': cmp('B0_ctx', 'B0'), 'mechanism_shuffled_history': cmp('B0_ctx', 'B0_ctx_shufhist'),
          'negative_control': cmp('stage1_only', 'B0'), 'noise': {n: cmp(n, 'B0') for n in NOISE}}
    pairs = (('B0_ctx', 'B0_ctx_randanchor'), ('B0_ctx', 'B0_rank'), ('B0_rank', 'B0'), ('B0_resid', 'B0'),
             ('B0_ctx_shufhist', 'B0'), ('B0_ctx_randanchor', 'B0'), ('S_star', 'B0'), ('B0', 'incumbent'))
    ev['descriptive'] = {f'{c}_vs_{r}': cmp(c, r) for c, r in pairs}
    return ev


def gate_checks(ev, checks):
    p = ev['primary']
    noise_max = max(abs(v['mean_delta']) for v in ev['noise'].values())
    g = {'primary_mean_delta_min': p['mean_delta'] >= GATE['primary_mean_delta_min'],
         'primary_ci95_lower_gt': p['ci95'][0] > GATE['primary_ci95_lower_gt'],
         'primary_family_min': min(p['by_family'].values()) >= GATE['primary_family_min'],
         'primary_halves_min': min(p['halves'].values()) >= GATE['primary_halves_min'],
         'ctx_minus_shuffled_history_mean_gt': ev['mechanism_shuffled_history']['mean_delta'] > GATE['ctx_minus_shuffled_history_mean_gt'],
         'primary_mean_gt_max_abs_noise_shift': p['mean_delta'] > noise_max,
         'negative_control_ci95_upper_lt': ev['negative_control']['ci95'][1] < GATE['negative_control_ci95_upper_lt']}
    des = ev['descriptive']
    c7 = {'S_star_vs_B0': des['S_star_vs_B0']['mean_delta'], 'B0_vs_incumbent': des['B0_vs_incumbent']['mean_delta']}
    c7_ok = abs(c7['S_star_vs_B0'] - .019211) <= 1.5e-6 and abs(c7['B0_vs_incumbent'] - .033932) <= 1.5e-6
    allc = {k: bool(v) for k, v in checks.items()}
    allc['C7'] = bool(c7_ok)
    return {k: bool(v) for k, v in g.items()}, dict(sorted(allc.items())), c7, noise_max


def prediction_check(reg, ev, noise_max):
    pr = reg['prediction']

    def band(key, obs):
        pt, lo, hi = pr[key]
        return {'observed': obs, 'point_low80_high80': [pt, lo, hi], 'inside80': bool(lo <= obs <= hi)}
    p, des = ev['primary'], ev['descriptive']
    return {'primary': band('B0_ctx_minus_B0_point_low80_high80', p['mean_delta']),
            'primary_mean_ge_010': bool(p['mean_delta'] >= .010), 'P_primary_mean_ge_010': pr['P_primary_mean_ge_010'],
            'P_gate_pass': pr['P_gate_pass'],
            'ctx_minus_shufhist': band('ctx_minus_shufhist_point_low80_high80', ev['mechanism_shuffled_history']['mean_delta']),
            'rank_minus_B0': band('B0_rank_minus_B0_point_low80_high80', des['B0_rank_vs_B0']['mean_delta']),
            'resid_minus_B0': band('B0_resid_minus_B0_point_low80_high80', des['B0_resid_vs_B0']['mean_delta']),
            'max_abs_noise_shift': band('max_abs_noise_shift_point_low80_high80', noise_max)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dry', 'official'])
    mode = ap.parse_args().mode
    t_start = time.perf_counter()
    raw = PREREG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PREREG_SHA, 'registration changed'
    reg = json.loads(raw.decode('utf-8'))
    assert reg['step'] == 102 and reg['gate'] == GATE and tuple(reg['design']['arms']) == ARMS
    assert tuple(reg['design']['feature_order']) == FEATS and s100.fsha(STEP100) == STEP100_SHA
    if mode == 'dry':
        out = SCRATCH / ('dry_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        out.mkdir(parents=True)
        EV = s100.fake_evaluate
    else:
        out = EXP_DIR
        for n in ('freeze.json', 'scores-arms.parquet', 'context-features.parquet', 'predictions-frozen.json', AUDIT_NAME):
            assert not (out / n).exists(), n
        EV = ac.evaluate
    prot0, snap0 = check_protected(), s100.codex_snapshot()
    ctx = s100.load_context()
    print('context', json.dumps(ctx['info'], default=s100.jdefault), flush=True)
    src = {str(p): s100.fsha(p) for p in [Path(__file__), PREREG, STEP100, CODEX / 'src/audit_common.py', s100.STEP99]}
    freeze = {'created_utc': s100.utc(), 'step': 102, 'author': 'Claude', 'mode': mode, 'registration_sha256': PREREG_SHA,
              'source_sha256': src, 'protected': prot0, 'gate': GATE, 'arms': ARMS, 'features': FEATS, 'context': ctx['info'],
              'metric_computed_before_freeze': False}
    (out / 'freeze.json').write_text(json.dumps(freeze, indent=2, default=s100.jdefault) + '\n', encoding='utf-8')
    cx, crecs, cframe, csum = build_context(ctx)
    print('context block', json.dumps(csum), flush=True)
    noise = noise_blocks(ctx)
    preds, fits = {}, {}
    for name in ARMS:
        if name == 'stage1_only':
            preds[name] = ctx['d']['oof'].to_numpy().copy()
            continue
        preds[name], fits[name] = run_arm(ctx, name, cx, noise)
    rep = s100.replicate(ctx, preds)
    c8, c8_ok = provisional_parity(ctx, cx, crecs, preds, fits)
    print('replication', json.dumps(rep), 'C8', c8_ok, flush=True)
    digests = write_frozen(ctx, out, preds, cframe)
    prot1, snap1 = check_protected(), s100.codex_snapshot()
    changed = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
    checks = {'C0': rep['C0']['bit_exact_all_rows'], 'C1': rep['C1']['bit_exact_all_rows'], 'C2': rep['C2']['bit_exact_all_rows'],
              'C3': csum['C3'], 'C4': csum['C4'],
              'C5': csum['replay_exact_all'] and all(r['replay_exact_all'] and r['top20_filled_once'] and r['tail_is_oof_minus_1e6']
                                                     for r in fits.values()),
              'C6': prot0 == prot1 and not changed, 'C8': c8_ok}
    frozen = {'created_utc': s100.utc(), 'mode': mode, 'registration_sha256': PREREG_SHA, 'freeze_sha256': s100.fsha(out / 'freeze.json'),
              'outputs_sha256': digests, 'context_summary': csum, 'context_receipts': crecs, 'stage2_receipts': fits,
              'replication': rep, 'provisional_parity_C8': c8, 'checks_before_metric': checks,
              'C4_note': 'per fold: pool-disjoint outer and inner folds, no held hand among training hands, one inner held fold per query',
              'codex_changed_entries': changed, 'codex_entries_checked': len(snap1),
              'seconds': round(time.perf_counter() - t_start, 1), 'metric_computed': False}
    (out / 'predictions-frozen.json').write_text(json.dumps(frozen, indent=2, default=s100.jdefault) + '\n', encoding='utf-8')
    print('PREDICTIONS FROZEN', frozen['seconds'], json.dumps(checks), flush=True)
    ev = evaluate_all(ctx, preds, EV)
    g, allc, c7, noise_max = gate_checks(ev, checks)
    passed = bool(all(g.values()) and all(allc.values()))
    audit = {'step': 102, 'author': 'Claude', 'mode': mode, 'evaluated_utc': s100.utc(), 'registration_sha256': PREREG_SHA,
             'predictions_frozen_sha256': s100.fsha(out / 'predictions-frozen.json'),
             'surface': 'training-only audit, 162 queries; no outer fold, public score or evaluation-set file read',
             'evaluation': ev, 'noise_max_abs_shift': noise_max, 'gate': GATE, 'gate_checks': g, 'checks_C0_C8': allc,
             'C7_values': c7, 'passed': passed, 'decision': reg['decision']['if_pass' if passed else 'if_fail'],
             'prediction_check': prediction_check(reg, ev, noise_max), 'held_read_count_mine': 6,
             'protected_unchanged': prot0 == check_protected(), 'seconds': round(time.perf_counter() - t_start, 1)}
    (out / AUDIT_NAME).write_text(json.dumps(audit, indent=2, default=s100.jdefault) + '\n', encoding='utf-8')
    p = ev['primary']
    print('PRIMARY B0_ctx vs B0', round(p['mean_delta'], 6), [round(x, 6) for x in p['ci95']], 'halves', p['halves'],
          'families', p['by_family'], 'better/worse', p['pairs_better'], p['pairs_worse'], flush=True)
    for k in ('mechanism_shuffled_history', 'negative_control'):
        print(k, round(ev[k]['mean_delta'], 6), [round(x, 6) for x in ev[k]['ci95']], flush=True)
    for k, v in list(ev['noise'].items()) + list(ev['descriptive'].items()):
        print('descriptive', k, round(v['mean_delta'], 6), [round(x, 6) for x in v['ci95']], flush=True)
    print('GATE', json.dumps(g), 'CHECKS', json.dumps(allc), 'PASSED', passed, 'out', str(out), flush=True)


if __name__ == '__main__':
    main()
