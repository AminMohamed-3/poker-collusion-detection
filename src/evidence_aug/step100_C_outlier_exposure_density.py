"""Step 100 [C]: Codex IDEA_BANK #1 on the training-only audit, on top of B0.

Serial outlier exposure then real-inlier density. Per family and audit fold, a small LightGBM projector learns real
unlisted top-20 rows against block-swapped copies, where each swapped block is an exact block of another inlier hand of
the same pair. Its per-tree leaf outputs are the embedding. The kNN distance to real inliers of the fitting set,
cross-fitted by inner pool folds, becomes one within-pair percentile field appended to B0. Controls at the same dose:
matched real resampling, random-feature perturbation, identity-space density, chronology-only density and stage1_only.
Registration: experiments/outlier_exposure_density_C_20260913/prediction-outlier-exposure-density.json (PREREG_SHA).

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
sys.path.insert(0, str(CODEX / 'src'))
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import audit_common as ac
from evidence_aug.paths import check_protected
from evidence_aug.fold import fold_pools

CLF = ac.CLF
FAMILIES = ac.FAMILIES
CI = 'coordinated_isolation'
RES = CODEX / 'research'
EXP_DIR = CV / 'experiments' / 'outlier_exposure_density_C_20260913'
PREREG = EXP_DIR / 'prediction-outlier-exposure-density.json'
PREREG_SHA = '09d89b9742f9e7cf053fd63e694f2eca539367219e759b223adc15ac4096e334'
SCRATCH = Path('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s100')
STEP99 = CV / 'experiments' / 'improver_blend_C_20260913' / 'scores-blends.parquet'
AUDIT_NAME = 'audit-C-outlier-exposure-density.json'
KEYS = ['pair_id', 'hand_id']
CHRONO = ['chrono_rank', 'chrono_frac', 'n_hands', 'oof', 'oof_rank']
PROJ = {'n_estimators': 50, 'num_leaves': 7, 'max_depth': 3, 'learning_rate': 0.1, 'min_child_samples': 20,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'random_state': 11, 'n_jobs': 2, 'verbosity': -1}
K_NN = 10
VARIANTS = ('exposure', 'matched', 'random', 'identity', 'chrono')   # variant_index is the position
PROJECTED = ('exposure', 'matched', 'random')
ARMS = ('incumbent', 'B0', 'B0_od_exposure', 'B0_od_matched', 'B0_od_random', 'B0_od_identity', 'B0_od_chrono',
        'S_star', 'S_star_od_exposure', 'stage1_only')
GATE = {'primary_mean_delta_min': .010, 'primary_ci95_lower_gt': 0., 'primary_family_min': -.010, 'primary_halves_min': 0.,
        'exposure_minus_matched_real_mean_gt': 0., 'exposure_minus_random_feature_mean_gt': 0.,
        'exposure_minus_chronology_mean_gt': 0., 'negative_control_ci95_upper_lt': 0.}


def utc():
    return datetime.now(timezone.utc).isoformat()


def fsha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def h(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def dump(path, obj):
    assert not path.exists(), f'refusing to overwrite {path}'
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return fsha(path)


def codex_snapshot():
    files = sorted((CODEX / 'src').glob('*.py')) + sorted((CODEX / 'src').glob('__pycache__/*'))
    files += [RES / 'contestable_state_20260912/features.parquet', RES / 'retrospective_strength_20260912/prefix.parquet',
              RES / 'retrospective_strength_20260912/freeze.json']
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}


def fake_evaluate(d, pred):
    """Dry mode only: built from prediction names; reads no label and computes no metric."""
    names = [n for n in pred if n != 'incumbent']
    comp = {n: {'mean_delta': 0., 'ci95': [-.01, .01], 'halves': {'0': 0., '1': 0.}, 'by_family': {f: 0. for f in FAMILIES},
                'pairs_better': 0, 'pairs_worse': 0, 'n': 162} for n in names}
    return {'audit_only': True, 'sets': {n: {'map5': .5} for n in pred}, 'comparisons': comp, 'passed': False}


def load_context():
    """The step 97 surface: B0 = base + cs_ fields, S_star = B0 + the prefix_strength block, same per-family folds."""
    t0 = time.perf_counter()
    d, shared, rcols, saved = ac.load_current()
    keys = d.select(KEYS)
    top = (d['oof_rank'] <= 20).to_numpy()
    expected = d.filter(pl.col('oof_rank') <= 20).select(KEYS)
    outer = set().union(*(set(fold_pools(f)) for f in (0, 1, 2)))
    assert not set(d['table_id']) & outer and d['pair_id'].n_unique() == 162 and int(top.sum()) == 3240
    assert shared[-5:] == CHRONO and saved.select(KEYS).equals(keys)
    cs = pl.read_parquet(RES / 'contestable_state_20260912/features.parquet').sort(KEYS)
    cs_cols = [c for c in cs.columns if c not in KEYS]
    assert len(cs_cols) == 29 and all(c.startswith('cs_') for c in cs_cols) and expected.equals(cs.select(KEYS))
    pre = pl.read_parquet(RES / 'retrospective_strength_20260912/prefix.parquet').sort(KEYS)
    pcols = json.loads((RES / 'retrospective_strength_20260912/freeze.json').read_text(encoding='utf-8'))['features']
    assert pre.columns == KEYS + pcols and pre.select(KEYS).equals(expected) and not set(pcols) & set(d.columns)
    pf = keys.join(pre, on=KEYS, how='left', validate='1:1').sort(KEYS)
    assert pf.select(KEYS).equals(keys)
    prefix = pf.select(pcols).to_numpy()
    dx = d.join(cs, on=KEYS, how='left', validate='1:1').sort(KEYS)
    assert dx.select(KEYS).equals(keys)
    fam = {}
    for fi, f in enumerate(FAMILIES):
        ix = np.flatnonzero(((d['behavior_family'] == f) & (d['oof_rank'] <= 20)).to_numpy())
        q = dx[ix]
        base = shared + (rcols if f != CI else [])
        cols = base + cs_cols
        assert len(set(cols)) == len(cols)
        x_inc = q.select(base).to_numpy().astype(float)
        x = q.select(cols).to_numpy().astype(float)
        y = q['relevance'].to_numpy().astype(int)
        g = q['table_id'].to_numpy()
        pid = q['pair_id'].to_numpy()
        assert np.isfinite(x).all() and np.array_equal(x[:, :len(base)], x_inc)
        folds = list(GroupKFold(n_splits=5).split(x_inc, y, g))
        for tr, te in folds:
            assert not set(g[tr]) & set(g[te])
        pos = {c: i for i, c in enumerate(cols)}
        blocks = {'struct': [pos[c] for c in shared if c not in CHRONO], 'stage1_chrono': [pos[c] for c in CHRONO]}
        if f != CI:
            blocks['rcols'] = [pos[c] for c in rcols]
        blocks['cs'] = [pos[c] for c in cs_cols]
        assert sorted(i for b in blocks.values() for i in b) == list(range(len(cols)))
        sizes = pl.DataFrame({'p': pid}).group_by('p').len()['len']
        assert int(sizes.min()) == 20 and int(sizes.max()) == 20
        fam[f] = {'fi': fi, 'ix': ix, 'x': x, 'x_inc': x_inc, 'x_star': np.column_stack([x, prefix[ix]]), 'y': y, 'g': g,
                  'pid': pid, 'hid': q['hand_id'].to_numpy(), 'folds': folds, 'order': tuple(blocks),
                  'blocks': {b: np.array(v, dtype=np.int64) for b, v in blocks.items()}}
    info = {'rows': d.height, 'top20_rows': int(top.sum()), 'queries': d['pair_id'].n_unique(), 'outer_pools_checked': len(outer),
            'shared_columns': len(shared), 'rcols': len(rcols), 'cs_fields': len(cs_cols), 'prefix_fields': len(pcols),
            'top20_rows_by_family': {f: len(fam[f]['ix']) for f in FAMILIES},
            'columns_by_family': {f: int(fam[f]['x'].shape[1]) for f in FAMILIES},
            'block_order_and_sizes': {f: [[b, len(fam[f]['blocks'][b])] for b in fam[f]['order']] for f in FAMILIES},
            'training_inliers_by_family_fold': {f: [int((fam[f]['y'][tr] == 0).sum()) for tr, _ in fam[f]['folds']] for f in FAMILIES},
            'load_seconds': round(time.perf_counter() - t0, 1)}
    return {'d': d, 'keys': keys, 'top': top, 'saved': saved, 'fam': fam, 'info': info}


def rng_for(fi, k, j, v):
    return np.random.default_rng([100, fi, k, j, v])


def perturb(F, fit, variant, rng):
    """One copy per fitting inlier row (fit indexes the family arrays); returns the copies and their provenance for C3."""
    R = F['x'][fit]
    n = len(fit)
    if variant == 'matched':
        pick = rng.integers(0, n, size=n)
        return R[pick].copy(), {'pick': pick}
    order = F['order']
    bsel = rng.integers(0, len(order), size=n)
    P = R.copy()
    donor = np.full(n, -1, dtype=np.int64)
    if variant == 'exposure':
        pid = F['pid'][fit]
        mates = {}
        for i, p in enumerate(pid):
            mates.setdefault(p, []).append(i)
        for i in range(n):
            m = [t for t in mates[pid[i]] if t != i]
            if m:
                donor[i] = m[int(rng.integers(0, len(m)))]
                c = F['blocks'][order[bsel[i]]]
                P[i, c] = R[donor[i], c]
    else:
        assert variant == 'random'
        mu, sd = R.mean(axis=0), R.std(axis=0)
        for i in range(n):
            c = F['blocks'][order[bsel[i]]]
            P[i, c] = rng.normal(mu[c], sd[c])
    return P, {'block': bsel, 'donor': donor}


def legality(F, fit, P, prov, variant):
    R = F['x'][fit]
    if variant == 'matched':
        return {'legal': bool(np.array_equal(P, R[prov['pick']])), 'unchanged_copies': 0}
    ok, unchanged = True, 0
    pid = F['pid'][fit]
    for i in range(len(fit)):
        c = F['blocks'][F['order'][prov['block'][i]]]
        rest = np.ones(R.shape[1], dtype=bool)
        rest[c] = False
        ok &= bool(np.array_equal(P[i, rest], R[i, rest]))
        if variant == 'exposure':
            dn = int(prov['donor'][i])
            if dn < 0:
                unchanged += 1
                ok &= bool(np.array_equal(P[i], R[i]))
            else:
                ok &= bool(dn != i and pid[dn] == pid[i] and np.array_equal(P[i, c], R[dn, c]))
    return {'legal': ok, 'unchanged_copies': unchanged}


def leaf_values(b):
    info = b.dump_model()['tree_info']
    tab = np.zeros((len(info), max(t['num_leaves'] for t in info)))
    for t, ti in enumerate(info):
        for leaf in range(ti['num_leaves']):
            tab[t, leaf] = b.get_leaf_output(t, leaf)
    return tab


def embed(b, tab, X):
    leaves = np.asarray(b.predict(X, pred_leaf=True, num_threads=2)).astype(np.int64)
    return tab[np.arange(tab.shape[0])[None, :], leaves]


def knn_mean(ref, qry, k=K_NN):
    rr = np.einsum('ij,ij->i', ref, ref)
    out = np.empty(len(qry))
    for s in range(0, len(qry), 256):
        qq = qry[s:s + 256]
        d2 = np.einsum('ij,ij->i', qq, qq)[:, None] + rr[None, :] - 2. * (qq @ ref.T)
        np.maximum(d2, 0., out=d2)
        out[s:s + 256] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, :k]).mean(axis=1)
    return out


def od_scores(F, variant, fit, qry, rng, rec):
    """kNN distance of the qry rows to the real inliers in fit, in the variant space. C4 is asserted here."""
    X = F['x']
    assert len(fit) > K_NN and not F['y'][fit].any() and not set(F['g'][fit]) & set(F['g'][qry])
    R = X[fit]
    rec['fits'] += 1
    if variant in PROJECTED:
        P, prov = perturb(F, fit, variant, rng)
        leg = legality(F, fit, P, prov, variant)
        m = lgb.LGBMClassifier(**PROJ).fit(np.vstack([R, P]), np.r_[np.zeros(len(R), dtype=int), np.ones(len(P), dtype=int)])
        s = m.booster_.model_to_string()
        tab = leaf_values(m.booster_)
        er, eq = embed(m.booster_, tab, R), embed(m.booster_, tab, X[qry])
        b = lgb.Booster(model_str=s)
        rec['replay_exact'] &= bool(np.array_equal(eq, embed(b, leaf_values(b), X[qry])))
        gap = float(np.max(np.abs(eq.sum(axis=1) - m.booster_.predict(X[qry], raw_score=True))))
        rec['max_sum_gap'] = max(rec['max_sum_gap'], gap)
        rec['legal'] &= leg['legal']
        rec['unchanged_copies'] += leg['unchanged_copies']
        rec['trees'].add(int(tab.shape[0]))
        rec['projector_sha256'].append(h(s))
        return knn_mean(er, eq)
    cols = np.arange(X.shape[1]) if variant == 'identity' else F['blocks']['stage1_chrono'][:3]
    Z, Q = R[:, cols], X[qry][:, cols]
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    keep = sd > 0
    rec['dims'].add(int(keep.sum()))
    return knn_mean((Z[:, keep] - mu[keep]) / sd[keep], (Q[:, keep] - mu[keep]) / sd[keep])


def od_fold(F, k, variant, rec):
    """Held rows: fit on all training-pool inliers. Training rows: fit on the inliers of the other inner pool splits."""
    tr, te = F['folds'][k]
    y, g = F['y'], F['g']
    vi = VARIANTS.index(variant)
    raw = np.full(len(F['ix']), np.nan)
    raw[te] = od_scores(F, variant, tr[y[tr] == 0], te, rng_for(F['fi'], k, 5, vi), rec)
    for j, (itr, ite) in enumerate(GroupKFold(n_splits=5).split(tr.reshape(-1, 1), y[tr], g[tr])):
        rows = tr[itr]
        raw[tr[ite]] = od_scores(F, variant, rows[y[rows] == 0], tr[ite], rng_for(F['fi'], k, j, vi), rec)
    assert np.isfinite(raw).all()
    pct = pl.DataFrame({'p': F['pid'], 'v': raw}).select(pl.col('v').rank('average').over('p'))['v'].to_numpy() / 20.
    return raw, pct


def build_od(ctx):
    od, frames, recs = {}, [], {}
    for v in VARIANTS:
        t0 = time.perf_counter()
        rec = {'fits': 0, 'legal': True, 'unchanged_copies': 0, 'replay_exact': True, 'max_sum_gap': 0., 'trees': set(),
               'dims': set(), 'projector_sha256': []}
        for f in FAMILIES:
            F = ctx['fam'][f]
            for k in range(5):
                raw, pct = od_fold(F, k, v, rec)
                od[(v, f, k)] = pct
                n = len(raw)
                frames.append(pl.DataFrame({'pair_id': F['pid'], 'hand_id': F['hid'], 'family': [f] * n, 'fold': [k] * n,
                                            'variant': [v] * n, 'raw_knn_distance': raw, 'within_pair_pct': pct}))
        rec.update(trees=sorted(rec['trees']), dims=sorted(rec['dims']), projector_sha256=h('|'.join(rec['projector_sha256'])),
                   seconds=round(time.perf_counter() - t0, 1))
        if v not in PROJECTED:
            rec.update(legal=None, replay_exact=None, max_sum_gap=None, projector_sha256=None)
        assert rec['fits'] == 90, rec['fits']
        recs[v] = rec
        print(f"  od {v}: fits={rec['fits']} legal={rec['legal']} replay={rec['replay_exact']} unchanged={rec['unchanged_copies']} "
              f"trees={rec['trees']} dims={rec['dims']} gap={rec['max_sum_gap']} {rec['seconds']}s", flush=True)
    return od, pl.concat(frames), recs


def run_arm(ctx, name, od):
    """Stage 2 as step 97 run_recipe: per family and fold, ac.CLF with random_state 11; top 20 get p + 10, tail oof - 1e6."""
    t0 = time.perf_counter()
    d, top = ctx['d'], ctx['top']
    pred = d['oof'].to_numpy().copy() - 1e6
    fills = np.zeros(d.height, dtype=int)
    fits = []
    base = 'x_star' if name.startswith('S_star') else ('x_inc' if name == 'incumbent' else 'x')
    variant = name.split('_od_')[1] if '_od_' in name else None
    for f in FAMILIES:
        F = ctx['fam'][f]
        for k, (tr, te) in enumerate(F['folds']):
            z = F[base] if variant is None else np.column_stack([F[base], od[(variant, f, k)]])
            m = lgb.LGBMClassifier(**CLF, random_state=11).fit(z[tr], F['y'][tr])
            p = m.predict_proba(z[te])[:, 1]
            s = m.booster_.model_to_string()
            ok = bool(np.array_equal(p, lgb.Booster(model_str=s).predict(z[te], num_threads=2)))
            pred[F['ix'][te]] = p + 10
            fills[F['ix'][te]] += 1
            fits.append((f, int(z.shape[1]), ok, h(s)))
    rec = {'arm': name, 'fits': len(fits), 'replay_exact_all': all(x[2] for x in fits),
           'top20_filled_once': bool(np.all(fills[top] == 1) and np.all(fills[~top] == 0)),
           'tail_is_oof_minus_1e6': bool(np.array_equal(pred[~top], d['oof'].to_numpy()[~top] - 1e6)),
           'n_features_by_family': {x[0]: x[1] for x in fits}, 'model_strings_sha256': h('|'.join(x[3] for x in fits)),
           'seconds': round(time.perf_counter() - t0, 1)}
    print(f"  fit {name}: {rec['seconds']}s replay={rec['replay_exact_all']} filled_once={rec['top20_filled_once']}", flush=True)
    return pred, rec


def replicate(ctx, preds):
    """C0-C2: exact agreement with the saved Codex incumbent and the frozen step 99 B0 and S_star columns."""
    assert fsha(STEP99).startswith('24f5f302')
    s99 = pl.read_parquet(STEP99).sort(KEYS)
    assert s99.select(KEYS).equals(ctx['keys'])
    ref = {'C0': ('incumbent', ctx['saved']['incumbent'].to_numpy()), 'C1': ('B0', s99['B0'].to_numpy()),
           'C2': ('S_star', s99['S_star'].to_numpy())}
    return {c: {'arm': a, 'bit_exact_all_rows': bool(np.array_equal(preds[a], b)), 'max_abs_diff': float(np.max(np.abs(preds[a] - b)))}
            for c, (a, b) in ref.items()}


def write_frozen(ctx, out, preds, od_frame):
    sp, op = out / 'scores-arms.parquet', out / 'od-features.parquet'
    assert not sp.exists() and not op.exists()
    ctx['d'].select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(n, preds[n]) for n in ARMS]).write_parquet(sp)
    back = pl.read_parquet(sp)
    assert back.select(KEYS).equals(ctx['keys']) and all(np.array_equal(back[n].to_numpy(), preds[n]) for n in ARMS)
    od_frame.write_parquet(op)
    return {'scores-arms.parquet': fsha(sp), 'od-features.parquet': fsha(op)}


def summ(res, name):
    c = res['comparisons'][name]
    return {'map5': res['sets'][name]['map5'], 'reference_map5': res['sets']['incumbent']['map5'], 'mean_delta': c['mean_delta'],
            'ci95': c['ci95'], 'halves': c['halves'], 'by_family': c['by_family'], 'pairs_better': c['pairs_better'],
            'pairs_worse': c['pairs_worse'], 'n': c['n']}


def evaluate_all(ctx, preds, EV):
    d = ctx['d']

    def cmp(cand, ref):
        return summ(EV(d, {'incumbent': preds[ref], cand: preds[cand]}), cand)
    ev = {'primary': cmp('B0_od_exposure', 'B0'),
          'mechanism_matched_real': cmp('B0_od_exposure', 'B0_od_matched'),
          'mechanism_random_feature': cmp('B0_od_exposure', 'B0_od_random'),
          'chronology_alone': cmp('B0_od_exposure', 'B0_od_chrono'),
          'negative_control': cmp('stage1_only', 'B0')}
    pairs = (('B0_od_identity', 'B0'), ('B0_od_matched', 'B0'), ('B0_od_random', 'B0'), ('B0_od_chrono', 'B0'),
             ('S_star_od_exposure', 'S_star'), ('S_star', 'B0'), ('B0', 'incumbent'))
    ev['descriptive'] = {f'{c}_vs_{r}': cmp(c, r) for c, r in pairs}
    return ev


def gate_checks(ev, checks):
    p = ev['primary']
    g = {'primary_mean_delta_min': p['mean_delta'] >= GATE['primary_mean_delta_min'],
         'primary_ci95_lower_gt': p['ci95'][0] > GATE['primary_ci95_lower_gt'],
         'primary_family_min': min(p['by_family'].values()) >= GATE['primary_family_min'],
         'primary_halves_min': min(p['halves'].values()) >= GATE['primary_halves_min'],
         'exposure_minus_matched_real_mean_gt': ev['mechanism_matched_real']['mean_delta'] > GATE['exposure_minus_matched_real_mean_gt'],
         'exposure_minus_random_feature_mean_gt': ev['mechanism_random_feature']['mean_delta'] > GATE['exposure_minus_random_feature_mean_gt'],
         'exposure_minus_chronology_mean_gt': ev['chronology_alone']['mean_delta'] > GATE['exposure_minus_chronology_mean_gt'],
         'negative_control_ci95_upper_lt': ev['negative_control']['ci95'][1] < GATE['negative_control_ci95_upper_lt']}
    des = ev['descriptive']
    c7 = {'S_star_vs_B0': des['S_star_vs_B0']['mean_delta'], 'B0_vs_incumbent': des['B0_vs_incumbent']['mean_delta']}
    c7_ok = abs(c7['S_star_vs_B0'] - .019211) <= 1.5e-6 and abs(c7['B0_vs_incumbent'] - .033932) <= 1.5e-6
    return {k: bool(v) for k, v in g.items()}, dict({k: bool(v) for k, v in checks.items()}, C7=bool(c7_ok)), c7


def prediction_check(reg, ev):
    pr = reg['prediction']
    pt, lo, hi = pr['B0_od_exposure_minus_B0_point_low80_high80']
    mpt, mlo, mhi = pr['exposure_minus_matched_real_point_low80_high80']
    obs, mo = ev['primary']['mean_delta'], ev['mechanism_matched_real']['mean_delta']
    return {'primary_observed': obs, 'primary_point_low80_high80': [pt, lo, hi], 'primary_inside80': bool(lo <= obs <= hi),
            'primary_mean_ge_010': bool(obs >= .010), 'P_primary_mean_ge_010': pr['P_primary_mean_ge_010'], 'P_gate_pass': pr['P_gate_pass'],
            'exposure_minus_matched_observed': mo, 'exposure_minus_matched_point_low80_high80': [mpt, mlo, mhi],
            'exposure_minus_matched_inside80': bool(mlo <= mo <= mhi),
            'identity_minus_B0_observed': ev['descriptive']['B0_od_identity_vs_B0']['mean_delta'],
            'identity_minus_B0_point': pr['B0_od_identity_minus_B0_point']}


def jdefault(o):
    return o.item() if hasattr(o, 'item') else str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dry', 'official'])
    mode = ap.parse_args().mode
    t_start = time.perf_counter()
    raw = PREREG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PREREG_SHA, 'registration changed'
    reg = json.loads(raw.decode('utf-8'))
    assert reg['step'] == 100 and all(reg['gate'][k] == v for k, v in GATE.items())
    assert reg['design']['projector']['params'] == PROJ and reg['design']['density']['k'] == K_NN and tuple(reg['design']['arms']) == ARMS
    if mode == 'dry':
        out = SCRATCH / ('dry_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        out.mkdir(parents=True)
        EV = fake_evaluate
    else:
        out = EXP_DIR
        for n in ('freeze.json', 'scores-arms.parquet', 'od-features.parquet', 'predictions-frozen.json', AUDIT_NAME):
            assert not (out / n).exists(), n
        EV = ac.evaluate
    prot0, snap0 = check_protected(), codex_snapshot()
    ctx = load_context()
    print('context', json.dumps(ctx['info'], default=jdefault), flush=True)
    src = {str(p): fsha(p) for p in [Path(__file__), PREREG, CODEX / 'src/audit_common.py', STEP99,
                                     RES / 'contestable_state_20260912/features.parquet',
                                     RES / 'retrospective_strength_20260912/prefix.parquet',
                                     RES / 'retrospective_strength_20260912/freeze.json']}
    freeze = {'created_utc': utc(), 'step': 100, 'author': 'Claude', 'mode': mode, 'registration_sha256': PREREG_SHA,
              'source_sha256': src, 'protected': prot0, 'gate': GATE, 'projector': PROJ, 'k': K_NN, 'variants': VARIANTS,
              'arms': ARMS, 'context': ctx['info'], 'metric_computed_before_freeze': False}
    (out / 'freeze.json').write_text(json.dumps(freeze, indent=2, default=jdefault) + '\n', encoding='utf-8')
    od, od_frame, recs = build_od(ctx)
    preds, fits = {}, {}
    for name in ARMS:
        if name == 'stage1_only':
            preds[name] = ctx['d']['oof'].to_numpy().copy()
        else:
            preds[name], fits[name] = run_arm(ctx, name, od)
    rep = replicate(ctx, preds)
    print('replication', json.dumps(rep), flush=True)
    digests = write_frozen(ctx, out, preds, od_frame)
    prot1, snap1 = check_protected(), codex_snapshot()
    changed = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
    checks = {'C0': rep['C0']['bit_exact_all_rows'], 'C1': rep['C1']['bit_exact_all_rows'], 'C2': rep['C2']['bit_exact_all_rows'],
              'C3': all(recs[v]['legal'] for v in PROJECTED), 'C4': True,
              'C5': all(recs[v]['replay_exact'] for v in PROJECTED) and all(
                  r['replay_exact_all'] and r['top20_filled_once'] and r['tail_is_oof_minus_1e6'] for r in fits.values()),
              'C6': prot0 == prot1 and not changed}
    frozen = {'created_utc': utc(), 'mode': mode, 'registration_sha256': PREREG_SHA, 'freeze_sha256': fsha(out / 'freeze.json'),
              'outputs_sha256': digests, 'od_receipts': recs, 'stage2_receipts': fits, 'replication': rep, 'checks_C0_C6': checks,
              'C4_note': 'asserted inside every projector and density fit (a failure stops the run)',
              'codex_changed_entries': changed, 'codex_entries_checked': len(snap1),
              'seconds': round(time.perf_counter() - t_start, 1), 'metric_computed': False}
    (out / 'predictions-frozen.json').write_text(json.dumps(frozen, indent=2, default=jdefault) + '\n', encoding='utf-8')
    print('PREDICTIONS FROZEN', frozen['seconds'], json.dumps(checks), flush=True)
    ev = evaluate_all(ctx, preds, EV)
    g, allc, c7 = gate_checks(ev, checks)
    passed = bool(all(g.values()) and all(allc.values()))
    audit = {'step': 100, 'author': 'Claude', 'mode': mode, 'evaluated_utc': utc(), 'registration_sha256': PREREG_SHA,
             'predictions_frozen_sha256': fsha(out / 'predictions-frozen.json'),
             'surface': 'training-only audit, 162 queries; no outer fold, public score or evaluation-set file read',
             'evaluation': ev, 'gate': GATE, 'gate_checks': g, 'checks_C0_C7': allc, 'C7_values': c7, 'passed': passed,
             'decision': reg['decision']['if_pass' if passed else 'if_fail'], 'prediction_check': prediction_check(reg, ev),
             'held_read_count_mine': 6, 'protected_unchanged': prot0 == check_protected(),
             'seconds': round(time.perf_counter() - t_start, 1)}
    (out / AUDIT_NAME).write_text(json.dumps(audit, indent=2, default=jdefault) + '\n', encoding='utf-8')
    p = ev['primary']
    print('PRIMARY B0_od_exposure vs B0', round(p['mean_delta'], 6), [round(x, 6) for x in p['ci95']], 'halves', p['halves'],
          'families', p['by_family'], 'better/worse', p['pairs_better'], p['pairs_worse'], flush=True)
    for k in ('mechanism_matched_real', 'mechanism_random_feature', 'chronology_alone', 'negative_control'):
        print(k, round(ev[k]['mean_delta'], 6), [round(x, 6) for x in ev[k]['ci95']], flush=True)
    for k, v in ev['descriptive'].items():
        print('descriptive', k, round(v['mean_delta'], 6), [round(x, 6) for x in v['ci95']], flush=True)
    print('GATE', json.dumps(g), 'CHECKS', json.dumps(allc), 'PASSED', passed, 'out', str(out), flush=True)


if __name__ == '__main__':
    main()
