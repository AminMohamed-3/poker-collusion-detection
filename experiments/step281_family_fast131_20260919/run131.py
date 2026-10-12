"""STEP281: finish Codex's family-specific action-policy Fast131 screen.

STATE WHEN PICKED UP (2026-09-19 19:34 UTC). Codex's `goal_top5_20260919/foundation_pair/action_policy_family_fast`
completed its CPU stage at 18:42 UTC and stopped: CPU_RECEIPT.json records 18 new family policy fits (3 families x
3 nested pipelines x 2 outer folds) plus 6 reused normal policies in 179.7 s, with `reload_all3_exact`,
`normal_membership_and_probabilities_exact` and `first128_exact` all true, and both Fast131 contexts written.
MANIFEST.json still says `status: prepared_no_fit`. The directory has `prepare.py` and `cpu_generate.py` but no
`gpu_run.py` and no `evaluate.py`, so the scoring stage and the verdict were never produced. That is what this
script finishes. Codex's own process is alive but has written nothing for ~52 minutes (RUNNING.md has it awaiting
the user's reply on a replacement VM's SSH key), and the GPU is idle at 165 MiB of 4096.

NOTHING IN CODEX'S TREE IS WRITTEN. Its contexts, receipts, checkpoint, vendored TabPFN runtime and frozen
reference predictions are read read-only; every output of this run lands in this directory.

THE EXPERIMENT (Codex's PROTOCOL.md, quoted): "Use the exact six nested pipelines, table folds, cap128 action
rows, 38 inputs, nuisance predictions, and pair-direction-equal weights from action_policy_fast. Fit only three
family policies per pipeline... For each family, compute held actual-action log(P_family/P_normal), mean within
direction and then equally across both directions. Append exactly three columns to the original Fast128 contexts.
Later fit two Fast models, four members seed35935, with fresh replay. Primary comparison is fixed 80% original
Fast +20% Fast131 versus the already frozen 80% original Fast +20% Fast129 binary-policy arm on identical keys.
Original Fast is secondary."

So the question is whether splitting the pooled target action-policy LLR into three family-specific LLRs beats the
pooled one. The pooled Fast129 arm already passed at delta_joint +0.000236168749 (folds +0.000124233514 /
+0.000325940326, table95 [0.0000289551, 0.0005126997], logloss 0.0352701 -> 0.0294290).

MODES (each a fresh process, so the replay is a genuine re-run, as in Codex's design):
    --fit      two TabPFN-v3.5-fast fits, 4 members, seed 35935, 256-row query chunks -> fast131_predictions.npz
    --replay   refit fold 0 in a new process and require bit-exact equality with the stored fold-0 block
    --eval     the three arms scored on identical keys -> RESULTS.json

The metric helpers (`ap`, `met`, `rank2`, the .7/.1 joint, the table bootstrap, the 7/12 self-test) are copied
verbatim from `action_policy_fast/evaluate.py` so that the primary comparison is like-for-like with the frozen
Fast129 number rather than a re-implementation of it.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
G = ROOT / 'goal_top5_20260919'
FAM = G / 'foundation_pair/action_policy_family_fast'
F129 = G / 'foundation_pair/action_policy_fast'
SCREEN = G / 'foundation_pair/screen'
T35 = G / 'tabpfn35'
CK = T35 / 'tabpfn-v3.5-fast-20260909.safetensors'
CK_SHA = '14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550'
SEED = 35935
PRED = H / 'fast131_predictions.npz'
REPLAY = H / 'fast131_replay_fold0.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(*a):
    print(*a, flush=True)


def check_inputs():
    """Codex's receipts must describe the files on disk, or the port is reading something else."""
    cr = json.loads((FAM / 'CPU_RECEIPT.json').read_text())
    assert cr['manifest_sha256'] == sha(FAM / 'MANIFEST.json'), 'family manifest changed since the CPU stage'
    assert cr['new_family_fits'] == 18 and cr['reused_normal_fits'] == 6, cr
    for z in cr['folds']:
        f = z['fold']
        assert sha(FAM / ('fast131_context_fold%d.npz' % f)) == z['context_sha256'], \
            'fast131 context fold %d does not match the CPU receipt' % f
        assert z['first128_exact'] is True, 'fold %d does not nest the original Fast128 block' % f
        for e in z['fits']:
            if 'reload_all3_exact' in e:
                assert e['reload_all3_exact'] and e['normal_membership_and_probabilities_exact'], e
    fr = json.loads((F129 / 'FAST129_RECEIPT.json').read_text())
    rr = json.loads((F129 / 'REPLAY_RECEIPT.json').read_text())
    assert rr['exact'] and rr['predictions_sha256'] == fr['predictions_sha256'], 'frozen Fast129 arm is not clean'
    assert sha(F129 / 'fast129_predictions.npz') == fr['predictions_sha256'], 'Fast129 predictions moved'
    assert sha(CK) == CK_SHA, 'TabPFN fast checkpoint is not the pinned one'
    log('inputs verified: 18 family fits, both contexts match receipt, first128 nested, frozen Fast129 clean')
    return cr


def one(path, np, torch, TabPFNClassifier):
    """one fold: fit on the fold's training block, predict the held block in fixed 256-row chunks."""
    st = time.monotonic()
    d = np.load(path, allow_pickle=False)
    torch.cuda.reset_peak_memory_stats()
    m = TabPFNClassifier(model_path=str(CK), device='cuda', n_estimators=4, auto_scale_n_estimators=False,
                         fit_mode='low_memory', memory_saving_mode=True, random_state=SEED,
                         n_preprocessing_jobs=1)
    m.fit(d['X_train'], d['y_train'])
    q = np.concatenate([m.predict_proba(d['X_held'][i:i + 256]) for i in range(0, len(d['X_held']), 256)])
    assert np.array_equal(m.classes_, np.arange(4)) and q.shape == (len(d['X_held']), 4)
    assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all() and np.allclose(q.sum(1), 1, atol=1e-5)
    peak = int(torch.cuda.max_memory_reserved())
    assert peak < 3 * 1024 ** 3, 'exceeded the 3 GiB reserve'
    el = time.monotonic() - st
    assert el < 300, 'fold exceeded the 300 s cap'
    log('  %s  rows %d  %.2fs  peak %.3f GiB' % (path.name, len(q), el, peak / 1024 ** 3))
    return q, peak


def do_fit(replay: bool):
    sys.path.insert(0, str(T35))
    import runtime35                                                              # noqa: F401
    import numpy as np
    import torch
    from tabpfn import TabPFNClassifier
    assert torch.cuda.is_available(), 'no CUDA device'
    check_inputs()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.7)
    st = time.monotonic()
    if replay:
        assert PRED.exists() and not REPLAY.exists(), 'replay needs the fit to exist and itself not to'
        q, peak = one(FAM / 'fast131_context_fold0.npz', np, torch, TabPFNClassifier)
        np.savez_compressed(REPLAY, p=q)
        base = np.load(PRED, allow_pickle=False)
        exact = bool(np.array_equal(q, base['fold0']))
        assert exact, 'FRESH REPLAY IS NOT BIT-EXACT: the fit is not reproducible'
        (H / 'REPLAY_RECEIPT.json').write_text(json.dumps(
            {'predictions_sha256': sha(PRED), 'replay_sha256': sha(REPLAY), 'exact': exact,
             'seconds': round(time.monotonic() - st, 3), 'peak_reserved_bytes': peak}, indent=2) + '\n')
        log('REPLAY EXACT: fold 0 reproduces bit-for-bit in a fresh process')
        return
    assert not PRED.exists(), 'predictions already exist; refusing to overwrite'
    a = {}
    peaks = []
    for f in (0, 1):
        q, peak = one(FAM / ('fast131_context_fold%d.npz' % f), np, torch, TabPFNClassifier)
        a['fold%d' % f] = q
        peaks.append(peak)
        gc.collect()
        torch.cuda.empty_cache()
    assert time.monotonic() - st < 600, 'exceeded the 600 s budget'
    np.savez_compressed(PRED, **a)
    (H / 'FAST131_RECEIPT.json').write_text(json.dumps(
        {'predictions_sha256': sha(PRED), 'seconds': round(time.monotonic() - st, 3),
         'peak_reserved_bytes': max(peaks), 'seed': SEED, 'n_estimators': 4,
         'checkpoint_sha256': CK_SHA}, indent=2) + '\n')
    log('FIT DONE %.1fs  ->  %s' % (time.monotonic() - st, PRED.name))


# ---------------------------------------------------------------- evaluation
def ap(y, s, k, np):
    """average precision, Codex's implementation verbatim (stable key sort then score sort)."""
    y = np.asarray(y, bool)
    z = np.argsort(k, kind='stable')
    o = z[np.argsort(-s[z], kind='stable')]
    q = y[o]
    return float((np.cumsum(q) / (np.arange(len(q)) + 1))[q].mean()) if q.any() else 0.


def do_eval():
    import site
    site.addsitedir(str(ROOT / '.venv/Lib/site-packages'))
    import numpy as np
    import scipy.stats as st

    def rank2(x):
        return np.rint(2 * st.rankdata(x, method='average')).astype(np.int64)

    def met(y, s, route, k):
        return {'pair_ap': ap(y > 0, s, k, np),
                'behavior_map': float(np.mean([ap(y == c, s * (route == c), k, np) for c in (1, 2, 3)]))}

    def joint(z):
        return .7 * z['pair_ap'] + .1 * z['behavior_map']

    assert not (H / 'RESULTS.json').exists(), 'RESULTS.json already exists'
    cr = check_inputs()
    fr = json.loads((H / 'FAST131_RECEIPT.json').read_text())
    rr = json.loads((H / 'REPLAY_RECEIPT.json').read_text())
    assert rr['exact'] and rr['predictions_sha256'] == fr['predictions_sha256'] == sha(PRED), 'receipts disagree'
    assert abs(ap([1, 0, 1], np.array([.5, .5, .2]), np.array(['b', 'a', 'c']), np) - 7 / 12) < 1e-12, \
        'the AP implementation fails its own self-test'

    old = np.load(SCREEN / 'pf35_predictions.npz', allow_pickle=False)
    p129 = np.load(F129 / 'fast129_predictions.npz', allow_pickle=False)
    p131 = np.load(PRED, allow_pickle=False)
    rep = np.load(REPLAY, allow_pickle=False)
    assert set(p131.files) == {'fold0', 'fold1'} and np.array_equal(np.asarray(rep['p']), p131['fold0'])
    ctx = [np.load(FAM / ('fast131_context_fold%d.npz' % f), allow_pickle=False) for f in (0, 1)]
    ys, ks, ts, fs, oo, c9, c1 = [], [], [], [], [], [], []
    for f in (0, 1):
        e = np.load(SCREEN / ('evaluator_fold%d.npz' % f), allow_pickle=False)
        assert len(np.unique(e['pair_id'])) == len(e['pair_id']) and np.array_equal(e['pair_id'], np.sort(e['pair_id']))
        assert np.array_equal(e['pair_id'], ctx[f]['pair_id']), 'fold %d keys do not line up' % f
        for q, nm in ((p129['fold%d' % f], 'fast129'), (p131['fold%d' % f], 'fast131')):
            q = np.asarray(q)
            assert q.shape == (len(e['pair_id']), 4) and np.isfinite(q).all() and np.allclose(q.sum(1), 1, atol=1e-5)
        ys.append(e['y']); ks.append(e['pair_id']); ts.append(e['table_id'])
        fs.append(np.full(len(e['y']), f))
        oo.append(old['known_fold%d' % f]); c9.append(p129['fold%d' % f]); c1.append(p131['fold%d' % f])
    y = np.concatenate(ys); k = np.concatenate(ks); t = np.concatenate(ts); ff = np.concatenate(fs)
    a = np.concatenate(oo).astype(np.float64)
    b9 = np.concatenate(c9).astype(np.float64)
    b1 = np.concatenate(c1).astype(np.float64)
    ro = 1.0 - a[:, 0]
    r9 = 1.0 - b9[:, 0]
    r1 = 1.0 - b1[:, 0]
    route = 1 + np.argmax(a[:, 1:], 1)                      # behaviour route frozen on the original, as in Codex
    ARMS = {'orig_fast': ro,
            'fast129_80_20': 4 * rank2(ro) + rank2(r9),
            'fast131_80_20': 4 * rank2(ro) + rank2(r1)}
    M = {nm: met(y, s, route, k) for nm, s in ARMS.items()}
    log('')
    log('%-16s %12s %14s %12s' % ('arm', 'pair_AP', 'behaviour_MAP', 'joint'))
    for nm in ARMS:
        log('%-16s %12.9f %14.9f %12.9f' % (nm, M[nm]['pair_ap'], M[nm]['behavior_map'], joint(M[nm])))

    tabs = sorted(np.unique(t)); ixs = [np.flatnonzero(t == q) for q in tabs]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([ixs[j] for j in rng.integers(0, len(tabs), len(tabs))]) for _ in range(1000)]

    def contrast(cand, base):
        d = joint(met(y, cand, route, k)) - joint(met(y, base, route, k))
        folds = {str(f): joint(met(y[ff == f], cand[ff == f], route[ff == f], k[ff == f]))
                 - joint(met(y[ff == f], base[ff == f], route[ff == f], k[ff == f])) for f in (0, 1)}
        bt = [joint(met(y[ix], cand[ix], route[ix], k[ix])) - joint(met(y[ix], base[ix], route[ix], k[ix]))
              for ix in draws]
        return {'delta_joint': float(d), 'fold_deltas': folds,
                'bootstrap95': [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))],
                'bootstrap_p_gt_0': float(np.mean(np.array(bt) > 0)),
                'gate': bool(d > 0 and all(v > 0 for v in folds.values()))}

    PRIMARY = contrast(ARMS['fast131_80_20'], ARMS['fast129_80_20'])
    SECOND = contrast(ARMS['fast131_80_20'], ARMS['orig_fast'])
    REF129 = contrast(ARMS['fast129_80_20'], ARMS['orig_fast'])
    ll = {nm: float(-np.log(np.maximum(m[np.arange(len(y)), y], 1e-12)).mean())
          for nm, m in (('orig_fast', a), ('fast129', b9), ('fast131', b1))}
    fam = {str(c): {nm: ap(y == c, s, k, np) for nm, s in ARMS.items()} for c in (1, 2, 3)}
    res = {'step': 'STEP281', 'what': 'family-specific action-policy Fast131, finishing Codex action_policy_family_fast',
           'cpu_stage': {'new_family_fits': cr['new_family_fits'], 'reused_normal_fits': cr['reused_normal_fits'],
                         'seconds': cr['seconds']},
           'fit': fr, 'replay': rr, 'metrics': M,
           'PRIMARY_fast131_vs_frozen_fast129': PRIMARY,
           'secondary_fast131_vs_orig_fast': SECOND,
           'reference_fast129_vs_orig_fast': REF129,
           'reference_fast129_recorded_by_codex': {'delta_joint': 0.00023616874921550401,
                                                   'fold_deltas': {'0': 0.00012423351364643231,
                                                                   '1': 0.0003259403262608762}},
           'risk_correlation': {'fast131_vs_orig': float(np.corrcoef(ro, r1)[0, 1]),
                                'fast131_vs_fast129': float(np.corrcoef(r9, r1)[0, 1])},
           'family_ap': fam, 'multiclass_logloss': ll}
    (H / 'RESULTS.json').write_text(json.dumps(res, indent=2) + '\n')
    log('')
    log('REFERENCE  fast129 vs orig  delta %+.9f  folds %+.9f / %+.9f   (Codex recorded %+.9f)'
        % (REF129['delta_joint'], REF129['fold_deltas']['0'], REF129['fold_deltas']['1'], 0.000236168749))
    log('SECONDARY  fast131 vs orig  delta %+.9f  folds %+.9f / %+.9f  CI [%+.9f, %+.9f]  gate %s'
        % (SECOND['delta_joint'], SECOND['fold_deltas']['0'], SECOND['fold_deltas']['1'],
           SECOND['bootstrap95'][0], SECOND['bootstrap95'][1], SECOND['gate']))
    log('PRIMARY    fast131 vs fast129  delta %+.9f  folds %+.9f / %+.9f  CI [%+.9f, %+.9f]  P(>0) %.4f  GATE %s'
        % (PRIMARY['delta_joint'], PRIMARY['fold_deltas']['0'], PRIMARY['fold_deltas']['1'],
           PRIMARY['bootstrap95'][0], PRIMARY['bootstrap95'][1], PRIMARY['bootstrap_p_gt_0'], PRIMARY['gate']))
    log('logloss   orig %.9f   fast129 %.9f   fast131 %.9f' % (ll['orig_fast'], ll['fast129'], ll['fast131']))
    log('wrote RESULTS.json')


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('mode', choices=['fit', 'replay', 'eval'])
    x = ap_.parse_args()
    if x.mode == 'fit':
        do_fit(False)
    elif x.mode == 'replay':
        do_fit(True)
    else:
        do_eval()
