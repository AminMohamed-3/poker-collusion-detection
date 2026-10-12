"""Step 101 [C]: Codex IDEA_BANK #4, physical-hand exposure weighting, on the training-only audit on top of B0.

For every family model and audit fold, a hand's multiplicity m counts that fold's training rows (the family top-20 rows of the
training pools) with the same hand_id. The row weight m**-0.5 is renormalised within each training query, so every query keeps
its row count as total weight. Controls: a within-query permutation of the same weights, an inverse-multiplicity diagnostic,
S* with the same weights, and stage1_only. The context loader, replication and summaries come from the step 100 script.
Registration: experiments/physical_hand_weighting_C_20260913/prediction-physical-hand-weighting.json (PREREG_SHA).

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
import evidence_aug.step100_C_outlier_exposure_density as s100
from evidence_aug.paths import check_protected

ac = s100.ac
CLF, FAMILIES = s100.CLF, s100.FAMILIES
EXP_DIR = CV / 'experiments' / 'physical_hand_weighting_C_20260913'
PREREG = EXP_DIR / 'prediction-physical-hand-weighting.json'
PREREG_SHA = '13898bac981976f286d2a2e4d559420800d2f0976a21926013a543859fbf4b1a'
STEP100 = CV / 'src' / 'evidence_aug' / 'step100_C_outlier_exposure_density.py'
STEP100_SHA = 'cb289111c6e8ce3b3e4d571857f6e1f40aa131591904e121db54e61c96b8f5b0'
INVENTORY = EXP_DIR / 'inventory-no-fit.json'
SCRATCH = Path('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s101')
AUDIT_NAME = 'audit-C-physical-hand-weighting.json'
ARMS = ('incumbent', 'B0', 'B0_w_sqrt', 'B0_w_inv', 'B0_w_perm', 'S_star', 'S_star_w_sqrt', 'stage1_only')
WEIGHTED = ('B0_w_sqrt', 'B0_w_inv', 'B0_w_perm', 'S_star_w_sqrt')
GATE = {'primary_mean_delta_min': .010, 'primary_ci95_lower_gt': 0., 'primary_family_min': -.010, 'primary_halves_min': 0.,
        'primary_ci_family_min': 0., 'sqrt_minus_permuted_mean_gt': 0., 'negative_control_ci95_upper_lt': 0.}
EXPECTED_UNIT = {'directed_transfer': [0], 'soft_play': [0, 1, 2, 3, 4], 'coordinated_isolation': [1]}


def weights_for(F, tr, kind, k):
    """Training-row weights of one family model and fold. Counts see the training rows only (C4)."""
    hid, pid = F['hid'][tr], F['pid'][tr]
    _, inv, cnt = np.unique(hid, return_inverse=True, return_counts=True)
    m = cnt[inv.reshape(-1)].astype(float)
    fr = pl.DataFrame({'p': pid, 'w': m ** (-1. if kind == 'inv' else -.5)})
    fr = fr.with_columns(pl.col('w').sum().over('p').alias('s'), pl.col('w').count().over('p').alias('n'))
    w = (fr['w'] * fr['n'] / fr['s']).to_numpy()
    touched = np.unique(pid[m > 1])
    if kind == 'perm':
        rng = np.random.default_rng([101, F['fi'], k])
        w = w.copy()
        for p in touched:
            ix = np.flatnonzero(pid == p)
            w[ix] = w[ix][rng.permutation(len(ix))]
    return w, m, touched


def check_weights(F, tr, kind, w, touched, sqrt_w):
    """C3 for one fit: finite and positive, query totals kept, untouched queries exactly 1, permutation keeps each multiset."""
    pid = F['pid'][tr]
    ok = bool(len(w) == len(tr) and np.all(np.isfinite(w)) and np.all(w > 0))
    agg = pl.DataFrame({'p': pid, 'w': w}).group_by('p').agg(pl.col('w').sum().alias('s'), pl.col('w').count().alias('n'))
    ok &= bool(np.max(np.abs(agg['s'].to_numpy() - agg['n'].to_numpy())) <= 1e-9)
    ts = set(touched.tolist())
    untouched = np.array([p not in ts for p in pid])
    ok &= bool(np.all(w[untouched] == 1.))
    if kind == 'perm':
        for p in touched:
            ix = np.flatnonzero(pid == p)
            ok &= bool(np.array_equal(np.sort(w[ix]), np.sort(sqrt_w[ix])))
    return ok


def run_arm(ctx, name):
    """Stage 2 as in steps 97 and 100 (ac.CLF, random_state 11; top 20 get p + 10, tail oof - 1e6), weighted for _w_ arms."""
    t0 = time.perf_counter()
    d, top = ctx['d'], ctx['top']
    pred = d['oof'].to_numpy().copy() - 1e6
    fills = np.zeros(d.height, dtype=int)
    base = 'x_star' if name.startswith('S_star') else ('x_inc' if name == 'incumbent' else 'x')
    kind = name.split('_w_')[1] if '_w_' in name else None
    fits, frames, c3 = [], [], True
    for f in FAMILIES:
        F = ctx['fam'][f]
        for k, (tr, te) in enumerate(F['folds']):
            assert not set(F['g'][tr]) & set(F['g'][te]) and not set(F['hid'][te]) & set(F['hid'][tr])
            z = F[base]
            m = lgb.LGBMClassifier(**CLF, random_state=11)
            unit = None
            if kind is None:
                m.fit(z[tr], F['y'][tr])
            else:
                w, mult, touched = weights_for(F, tr, kind, k)
                c3 &= check_weights(F, tr, kind, w, touched, weights_for(F, tr, 'sqrt', k)[0])
                m.fit(z[tr], F['y'][tr], sample_weight=w)
                unit = bool(np.all(w == 1.))
                n = len(tr)
                frames.append(pl.DataFrame({'arm': [name] * n, 'family': [f] * n, 'fold': [k] * n, 'pair_id': F['pid'][tr],
                                            'hand_id': F['hid'][tr], 'multiplicity': mult.astype(np.int64), 'weight': w}))
            p = m.predict_proba(z[te])[:, 1]
            s = m.booster_.model_to_string()
            ok = bool(np.array_equal(p, lgb.Booster(model_str=s).predict(z[te], num_threads=2)))
            pred[F['ix'][te]] = p + 10
            fills[F['ix'][te]] += 1
            fits.append({'family': f, 'fold': k, 'n_features': int(z.shape[1]), 'replay': ok, 'unit': unit, 'sha': s100.h(s)})
    rec = {'arm': name, 'fits': len(fits), 'replay_exact_all': all(x['replay'] for x in fits),
           'top20_filled_once': bool(np.all(fills[top] == 1) and np.all(fills[~top] == 0)),
           'tail_is_oof_minus_1e6': bool(np.array_equal(pred[~top], d['oof'].to_numpy()[~top] - 1e6)),
           'weights_C3': bool(c3) if kind else None,
           'unit_family_folds': {f: [x['fold'] for x in fits if x['family'] == f and x['unit']] for f in FAMILIES} if kind else None,
           'n_features_by_family': {x['family']: x['n_features'] for x in fits},
           'model_strings_sha256': s100.h('|'.join(x['sha'] for x in fits)), 'seconds': round(time.perf_counter() - t0, 1)}
    wall = pl.concat(frames) if frames else None
    if wall is not None:
        rec['weight_summary'] = {'training_rows': wall.height, 'rows_weight_not_1': int((wall['weight'] != 1.).sum()),
                                 'rows_multiplicity_gt_1': int((wall['multiplicity'] > 1).sum()),
                                 'min': float(wall['weight'].min()), 'max': float(wall['weight'].max())}
    print(f"  fit {name}: {rec['seconds']}s replay={rec['replay_exact_all']} filled_once={rec['top20_filled_once']} "
          f"C3={rec['weights_C3']} unit={rec['unit_family_folds']}", flush=True)
    return pred, rec, wall


def unit_parity(ctx, preds, fits):
    """C8: where every training weight of a family-fold is 1, the weighted arm equals its unweighted base on those held rows."""
    out, ok = {}, True
    for name in WEIGHTED:
        base = 'S_star' if name.startswith('S_star') else 'B0'
        u = fits[name]['unit_family_folds']
        res = {}
        for f, ks in u.items():
            F = ctx['fam'][f]
            for k in ks:
                rows = F['ix'][F['folds'][k][1]]
                res[f'{f}:{k}'] = bool(np.array_equal(preds[name][rows], preds[base][rows]))
        out[name] = {'base': base, 'unit_family_folds': u, 'matches_registered_expectation': u == EXPECTED_UNIT, 'bit_exact': res}
        ok &= bool(res) and all(res.values()) and u == EXPECTED_UNIT
    return out, bool(ok)


def write_frozen(ctx, out, preds, wframe):
    sp, wp = out / 'scores-arms.parquet', out / 'weights.parquet'
    assert not sp.exists() and not wp.exists()
    ctx['d'].select('pair_id', 'hand_id', 'table_id').with_columns(*[pl.Series(n, preds[n]) for n in ARMS]).write_parquet(sp)
    back = pl.read_parquet(sp)
    assert back.select(s100.KEYS).equals(ctx['keys']) and all(np.array_equal(back[n].to_numpy(), preds[n]) for n in ARMS)
    wframe.write_parquet(wp)
    return {'scores-arms.parquet': s100.fsha(sp), 'weights.parquet': s100.fsha(wp)}


def evaluate_all(ctx, preds, EV):
    d = ctx['d']

    def cmp(cand, ref):
        return s100.summ(EV(d, {'incumbent': preds[ref], cand: preds[cand]}), cand)
    ev = {'primary': cmp('B0_w_sqrt', 'B0'), 'mechanism_permuted': cmp('B0_w_sqrt', 'B0_w_perm'),
          'negative_control': cmp('stage1_only', 'B0')}
    pairs = (('B0_w_inv', 'B0'), ('B0_w_perm', 'B0'), ('S_star_w_sqrt', 'S_star'), ('S_star', 'B0'), ('B0', 'incumbent'))
    ev['descriptive'] = {f'{c}_vs_{r}': cmp(c, r) for c, r in pairs}
    return ev


def gate_checks(ev, checks):
    p = ev['primary']
    g = {'primary_mean_delta_min': p['mean_delta'] >= GATE['primary_mean_delta_min'],
         'primary_ci95_lower_gt': p['ci95'][0] > GATE['primary_ci95_lower_gt'],
         'primary_family_min': min(p['by_family'].values()) >= GATE['primary_family_min'],
         'primary_halves_min': min(p['halves'].values()) >= GATE['primary_halves_min'],
         'primary_ci_family_min': p['by_family']['coordinated_isolation'] >= GATE['primary_ci_family_min'],
         'sqrt_minus_permuted_mean_gt': ev['mechanism_permuted']['mean_delta'] > GATE['sqrt_minus_permuted_mean_gt'],
         'negative_control_ci95_upper_lt': ev['negative_control']['ci95'][1] < GATE['negative_control_ci95_upper_lt']}
    des = ev['descriptive']
    c7 = {'S_star_vs_B0': des['S_star_vs_B0']['mean_delta'], 'B0_vs_incumbent': des['B0_vs_incumbent']['mean_delta']}
    c7_ok = abs(c7['S_star_vs_B0'] - .019211) <= 1.5e-6 and abs(c7['B0_vs_incumbent'] - .033932) <= 1.5e-6
    allc = {k: bool(v) for k, v in checks.items()}
    allc['C7'] = bool(c7_ok)
    return {k: bool(v) for k, v in g.items()}, dict(sorted(allc.items())), c7


def prediction_check(reg, ev):
    pr = reg['prediction']

    def band(key, obs):
        pt, lo, hi = pr[key]
        return {'observed': obs, 'point_low80_high80': [pt, lo, hi], 'inside80': bool(lo <= obs <= hi)}
    p = ev['primary']
    return {'primary': band('B0_w_sqrt_minus_B0_point_low80_high80', p['mean_delta']),
            'primary_mean_ge_010': bool(p['mean_delta'] >= .010), 'P_primary_mean_ge_010': pr['P_primary_mean_ge_010'],
            'P_gate_pass': pr['P_gate_pass'],
            'sqrt_minus_perm': band('sqrt_minus_perm_point_low80_high80', ev['mechanism_permuted']['mean_delta']),
            'inv_minus_B0': band('B0_w_inv_minus_B0_point_low80_high80', ev['descriptive']['B0_w_inv_vs_B0']['mean_delta']),
            'soft_play_primary_delta': p['by_family']['soft_play'], 'soft_play_exactly_zero': bool(p['by_family']['soft_play'] == 0.),
            'P_soft_play_primary_delta_exactly_zero': pr['P_soft_play_primary_delta_exactly_zero']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dry', 'official'])
    mode = ap.parse_args().mode
    t_start = time.perf_counter()
    raw = PREREG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PREREG_SHA, 'registration changed'
    reg = json.loads(raw.decode('utf-8'))
    assert reg['step'] == 101 and reg['gate'] == GATE and tuple(reg['design']['arms']) == ARMS
    assert s100.fsha(STEP100) == STEP100_SHA and INVENTORY.exists()
    if mode == 'dry':
        out = SCRATCH / ('dry_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        out.mkdir(parents=True)
        EV = s100.fake_evaluate
    else:
        out = EXP_DIR
        for n in ('freeze.json', 'scores-arms.parquet', 'weights.parquet', 'predictions-frozen.json', AUDIT_NAME):
            assert not (out / n).exists(), n
        EV = ac.evaluate
    prot0, snap0 = check_protected(), s100.codex_snapshot()
    ctx = s100.load_context()
    print('context', json.dumps(ctx['info'], default=s100.jdefault), flush=True)
    src = {str(p): s100.fsha(p) for p in [Path(__file__), PREREG, STEP100, CODEX / 'src/audit_common.py', s100.STEP99, INVENTORY]}
    freeze = {'created_utc': s100.utc(), 'step': 101, 'author': 'Claude', 'mode': mode, 'registration_sha256': PREREG_SHA,
              'source_sha256': src, 'protected': prot0, 'gate': GATE, 'arms': ARMS, 'context': ctx['info'],
              'metric_computed_before_freeze': False}
    (out / 'freeze.json').write_text(json.dumps(freeze, indent=2, default=s100.jdefault) + '\n', encoding='utf-8')
    preds, fits, frames = {}, {}, []
    for name in ARMS:
        if name == 'stage1_only':
            preds[name] = ctx['d']['oof'].to_numpy().copy()
            continue
        preds[name], fits[name], wall = run_arm(ctx, name)
        if wall is not None:
            frames.append(wall)
    rep = s100.replicate(ctx, preds)
    c8, c8_ok = unit_parity(ctx, preds, fits)
    print('replication', json.dumps(rep), 'C8', c8_ok, flush=True)
    digests = write_frozen(ctx, out, preds, pl.concat(frames))
    prot1, snap1 = check_protected(), s100.codex_snapshot()
    changed = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
    checks = {'C0': rep['C0']['bit_exact_all_rows'], 'C1': rep['C1']['bit_exact_all_rows'], 'C2': rep['C2']['bit_exact_all_rows'],
              'C3': all(fits[n]['weights_C3'] for n in WEIGHTED), 'C4': True,
              'C5': all(r['replay_exact_all'] and r['top20_filled_once'] and r['tail_is_oof_minus_1e6'] for r in fits.values()),
              'C6': prot0 == prot1 and not changed, 'C8': c8_ok}
    frozen = {'created_utc': s100.utc(), 'mode': mode, 'registration_sha256': PREREG_SHA, 'freeze_sha256': s100.fsha(out / 'freeze.json'),
              'outputs_sha256': digests, 'stage2_receipts': fits, 'replication': rep, 'unit_parity_C8': c8,
              'checks_before_metric': checks,
              'C4_note': 'asserted in every fit: pool-disjoint folds, no held hand among training hands, counts from training rows only',
              'codex_changed_entries': changed, 'codex_entries_checked': len(snap1),
              'seconds': round(time.perf_counter() - t_start, 1), 'metric_computed': False}
    (out / 'predictions-frozen.json').write_text(json.dumps(frozen, indent=2, default=s100.jdefault) + '\n', encoding='utf-8')
    print('PREDICTIONS FROZEN', frozen['seconds'], json.dumps(checks), flush=True)
    ev = evaluate_all(ctx, preds, EV)
    g, allc, c7 = gate_checks(ev, checks)
    passed = bool(all(g.values()) and all(allc.values()))
    audit = {'step': 101, 'author': 'Claude', 'mode': mode, 'evaluated_utc': s100.utc(), 'registration_sha256': PREREG_SHA,
             'predictions_frozen_sha256': s100.fsha(out / 'predictions-frozen.json'),
             'surface': 'training-only audit, 162 queries; no outer fold, public score or evaluation-set file read',
             'evaluation': ev, 'gate': GATE, 'gate_checks': g, 'checks_C0_C8': allc, 'C7_values': c7, 'passed': passed,
             'decision': reg['decision']['if_pass' if passed else 'if_fail'], 'prediction_check': prediction_check(reg, ev),
             'held_read_count_mine': 6, 'protected_unchanged': prot0 == check_protected(),
             'seconds': round(time.perf_counter() - t_start, 1)}
    (out / AUDIT_NAME).write_text(json.dumps(audit, indent=2, default=s100.jdefault) + '\n', encoding='utf-8')
    p = ev['primary']
    print('PRIMARY B0_w_sqrt vs B0', round(p['mean_delta'], 6), [round(x, 6) for x in p['ci95']], 'halves', p['halves'],
          'families', p['by_family'], 'better/worse', p['pairs_better'], p['pairs_worse'], flush=True)
    for k in ('mechanism_permuted', 'negative_control'):
        print(k, round(ev[k]['mean_delta'], 6), [round(x, 6) for x in ev[k]['ci95']], flush=True)
    for k, v in ev['descriptive'].items():
        print('descriptive', k, round(v['mean_delta'], 6), [round(x, 6) for x in v['ci95']], flush=True)
    print('GATE', json.dumps(g), 'CHECKS', json.dumps(allc), 'PASSED', passed, 'out', str(out), flush=True)


if __name__ == '__main__':
    main()
