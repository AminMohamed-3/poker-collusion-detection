"""step 94 [C].  A training-only clean-label mask from the (type, time) listing rule of OBS 14.16.

author: claude (agent)

QUESTION (pre-registered): does a training-only clean-label mask built from the (type, time)
listing rule of OBS 14.16 raise stage-2 audit MAP@5 over the refit identity of the incumbent
hybrid?

REGISTRATION: experiments/clean_label_mask_C_20260913/prediction-clean-label-mask.json, written
before any fit.  Its sha256 is verified at entry and at exit; this file never writes it.  The
salts, thresholds and registered counts are read from it and asserted equal to the constants
below, so a constant cannot drift from what was registered.

ARMS
  ref  the step 88 C1 refit identity: the stored production stage-1 `oof` of audit-oof.parquet,
       the 5-way pool split under the salt, stage 2 refit by the IMPORTED step74.fit_predict
       (step 14 CLF, TOP 20, SPLITS 5, seeds 11/12/13), r382 design for true directed_transfer
       and soft_play, ref design for true coordinated_isolation
  A0   ref with the type-rule AMBIGUOUS rows removed from every training index
  A1   ref with LightGBM sample_weight .5 on type-rule AMBIGUOUS training rows, 1.0 on the rest
  T0   ref with the T0 time-rule rows removed from every training index
step85b.hybrid() and step 88's hybrid_incumbent() / route() / frame_of() are closures inside
main(): they can be neither imported nor handed a fit function, so step 88's bodies are copied
below (step 88 showed that path reproduces step85b.hybrid() in 12/12 cells).  The masked arms
call fit_predict_masked, a line-for-line copy of step74.fit_predict with two optional additions
that touch only the TRAINING index of each (family, split) fit: a row filter on (pair_id,
hand_id) and a per-row sample_weight.  Test rows, the metric and the features are untouched and
listed rows are never masked.  A0 and T0 really drop rows: a LightGBM weight of 0 would still
count towards min_child_samples.

MASK RULES (registered).  n_listed counts every listed hand of the pair, chrono_rank is the stored
column of audit-oof.parquet, type_showdown = (pair_sd == 2) and fold type otherwise.  For an
unlisted row of pair q:
  n_listed(q) < 5                            clean negative
  n_listed(q) == 5, list holds a showdown    fold-type clean; showdown-type AMBIGUOUS iff its
                                             chrono_rank > the latest listed showdown-type one
  n_listed(q) == 5, list all fold-type       showdown-type AMBIGUOUS; fold-type AMBIGUOUS iff its
                                             chrono_rank > the latest listed fold-type one
  T0: unlisted rows with chrono_rank > the latest listed chrono_rank, in pairs with n_listed == 5.

CONTROLS.  C1 ref at audit-trunc-v1 within 5e-5 of .615934.  C2 the masked path with an empty
mask reproduces the ref predictions, max abs diff 0.  C3 the 14.16 listed type counts and the
registered top-20 design counts reproduce.  C4 no audit table lies in a held fold pool (step 88's
check).  C5 check_protected() identical at entry and exit.  C6 the weighted path with every
weight 1.0 reproduces ref MAP@5 at 6 dp and the ref predictions within 1e-9.  C1, C2, C4 or C6
failing stops the run (registered).  C3 failing also stops it, before any fit: a mask that does
not reproduce the registered counts is not the registered mask.  A stopped run writes its
controls and no arm result.

GATE (A0 only, registered).  At audit-trunc-v1: mean per-pair delta >= +.005 AND boot ci95 lower
bound > 0 AND every true-family mean delta >= -.01 AND both table halves
(int(sha256(table_id), 16) % 2) have mean delta >= 0; AND A0 - ref >= 0 at audit-resplit-b and
audit-resplit-c.  Every comparison uses the unrounded value.

TRAINING ONLY: the 162-pair audit surface.  load_fold(0) is read for feature columns only, as
steps 74, 85b and 88 do.  No held-fold truth, no evaluation label, no public score, no submission.

Output: experiments/clean_label_mask_C_20260913/audit-C-clean-label-mask.json
        experiments/clean_label_mask_C_20260913/audit-C-clean-label-mask-per-pair-ap.parquet
"""
from __future__ import annotations

import hashlib
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import ctypes
import gc
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl

from evidence_aug.fold import fold_pools, load_fold
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import CV, check_protected, read_json, sha, write_json
from evidence_aug.step14_audit_truncation import CLF, OUT as LP, SPLITS, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step74_length_shift import (CHRONO3, CI, S1, SEEDS as STEP74_SEEDS, build_d0,
                                              evaluate, fit_predict, frame_at)
from evidence_aug.step85_C_stage1_ensemble import SALTS, pair_slices, resplit
from evidence_aug.step88_C_joint_refit import (ANCHOR_A3, INCUMBENT_ROUTING, S2_DESIGNS,
                                               S2_SEEDS_FULL)

OUTDIR = CV / 'experiments' / 'clean_label_mask_C_20260913'
PRED = OUTDIR / 'prediction-clean-label-mask.json'
PRED_SHA = '3f6bce76ced875395a4637ba13fbde4abeda75660b4460c4cdac0b0aec957a78'
OUT_JSON = OUTDIR / 'audit-C-clean-label-mask.json'
OUT_PP = OUTDIR / 'audit-C-clean-label-mask-per-pair-ap.parquet'
INPUT_SHA = {
    'audit-oof.parquet': 'df3e91a96b9355c675cf3109142f1e60ccfb45d10697f551d1b02ce1cccbeda1',
    'audit-structural-features.parquet':
        '7c4f2b4ad20986c2bf6b538d5741d02f8593194b5babd52050fa75bc68af2f4f'}
PRIMARY = 'audit-trunc-v1'
MASKED_ARMS = ('A0', 'A1', 'T0')
A1_WEIGHT = .5
C1_TARGET, C1_TOL = .615934, 5e-5
C6_PRED_TOL = 1e-9
GATE = {'delta_min': .005, 'ci95_lower_gt': 0.0, 'family_floor': -.01,
        'both_table_halves_nonnegative': True, 'other_salts_nonnegative': True}
BOOT_N, BOOT_SEED = 2000, 652
# OBS 14.16's listed type counts: the same numbers as step77.CTRL_TYPE_SHARE and the registration
CTRL_TYPE_SHARE = {('coordinated_isolation', 'fold'): 210, ('coordinated_isolation', 'showdown'): 5,
                   ('directed_transfer', 'fold'): 271, ('directed_transfer', 'showdown'): 84,
                   ('soft_play', 'fold'): 157, ('soft_play', 'showdown'): 67}


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------
def peak_mb():
    """peak working set of this process in MB via the Windows API; None if unavailable."""
    try:
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                    'PagefileUsage', 'PeakPagefileUsage')]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 2 ** 20)
    except Exception:  # noqa: BLE001 -- a memory readout is diagnostic and must never stop the run
        return None


def table_half(table_id):
    """the registered table halves"""
    return int(hashlib.sha256(table_id.encode()).hexdigest(), 16) % 2


def str_keys(d):
    """tuple-keyed counts -> 'a|b' keys for JSON"""
    return {f'{k[0]}|{k[1]}': v for k, v in d.items()}


def total(ledger, field):
    """one ledger field summed over designs and families"""
    return int(sum(r[field] for g in ledger.values() for r in g.values()))


# ------------------------------------------------------------------------------------------------
# the registered mask
# ------------------------------------------------------------------------------------------------
def clean_label_masks(m):
    """boolean row flags of the registered rules for a frame sorted by (pair_id, hand_id).

    `m` carries the STORED relevance and chrono_rank of audit-oof.parquet and pair_sd of
    audit-structural-features.parquet.  Returns (amb, t0, t_all, list_types):
      amb         type-rule AMBIGUOUS rows (arms A0 and A1)
      t0          T0 time rule: unlisted, chrono_rank > latest listed chrono_rank, n_listed == 5
      t_all       the same time rule over ALL pairs: the registration's uncapped draft, reported
      list_types  pairs per 'family|n_listed=k|any_showdown=b', as the registration counts them
    n_listed and every 'latest listed' run over ALL listed hands of the pair, not the top 20."""
    pid = m['pair_id'].to_numpy()
    listed = m['relevance'].to_numpy() == 1
    sd = m['pair_sd'].to_numpy() == 2
    cr = m['chrono_rank'].to_numpy()
    fam = m['behavior_family'].to_numpy()
    amb, t0, t_all = (np.zeros(m.height, dtype=bool) for _ in range(3))
    list_types = {}
    slices = pair_slices(pid)
    assert len(slices) == len(set(pid.tolist())), 'pairs are not contiguous'
    for a, b in slices:
        assert (pid[a:b] == pid[a]).all() and (fam[a:b] == fam[a]).all(), 'pair slice is mixed'
        lis, sdt, chr_ = listed[a:b], sd[a:b], cr[a:b]
        unl = ~lis
        n_listed = int(lis.sum())
        assert 1 <= n_listed <= 5, f'n_listed={n_listed} is outside the registered rule'
        latest = chr_[lis].max()
        t_all[a:b] = unl & (chr_ > latest)
        any_sd = bool((lis & sdt).any())
        k = f'{fam[a]}|n_listed={n_listed}|any_showdown={any_sd}'
        list_types[k] = list_types.get(k, 0) + 1
        if n_listed < 5:
            continue                                        # every unlisted row is a clean negative
        t0[a:b] = unl & (chr_ > latest)
        if any_sd:
            latest_sd = chr_[lis & sdt].max()
            amb[a:b] = unl & sdt & (chr_ > latest_sd)       # unlisted fold-type rows stay clean
        else:
            latest_fold = chr_[lis & ~sdt].max()
            amb[a:b] = unl & (sdt | (~sdt & (chr_ > latest_fold)))
    assert not (amb & listed).any() and not (t0 & listed).any(), 'a listed row was masked'
    return amb, t0, t_all, dict(sorted(list_types.items()))


def design_counts(m, amb, t0, t_all):
    """C3: the listed type counts of 14.16 over ALL listed rows, and the registered design counts
    over the top-20 (stored oof_rank) unlisted rows of each true family."""
    fam = m['behavior_family'].to_numpy()
    listed = m['relevance'].to_numpy() == 1
    sd = m['pair_sd'].to_numpy() == 2
    top = m['oof_rank'].to_numpy() <= TOP
    types, counts, uncapped = {}, {}, {}
    for f in FAMILIES:
        types[(f, 'fold')] = int((listed & (fam == f) & ~sd).sum())
        types[(f, 'showdown')] = int((listed & (fam == f) & sd).sum())
        u = (fam == f) & top & ~listed
        counts[f] = {'unlisted': int(u.sum()), 'ambiguous': int((u & amb).sum()),
                     'late': int((u & t0).sum()), 'overlap': int((u & amb & t0).sum()),
                     'type_only': int((u & amb & ~t0).sum()),
                     'time_only': int((u & t0 & ~amb).sum())}
        uncapped[f] = {'late': int((u & t_all).sum()), 'time_only': int((u & t_all & ~amb).sum())}
    totals = {k: sum(counts[f][k] for f in FAMILIES)
              for k in ('unlisted', 'ambiguous', 'late', 'overlap')}
    return types, counts, uncapped, totals


# ------------------------------------------------------------------------------------------------
# the masked stage-2 fit: step74.fit_predict line for line, plus two optional additions
# ------------------------------------------------------------------------------------------------
def _ledger_row():
    return {'fits': 0, 'skipped_splits': [], 'training_rows_before_mask': 0, 'removed': 0,
            'training_rows_used': 0, 'in_weight_set': 0, 'reweighted': 0}


def fit_predict_masked(dtr, dte, cols, topk, seeds=STEP74_SEEDS, params=CLF, drop=None,
                       weight=None, ledger=None):
    """per-family, 5-way pool-hash OOF: fit on the training tables' rows of `dtr`, predict `dte`.

    Additions to step74.fit_predict, both on the TRAINING index only and both off by default:
      drop    set of (pair_id, hand_id): rows of `dtr` in it leave the training index of every
              (family, split) fit -- removed, not zero-weighted (min_child_samples counts rows)
      weight  (set of (pair_id, hand_id), w): LightGBM sample_weight w on the training rows in the
              set and 1.0 on every other training row
      ledger  dict filled per family with the training rows actually removed / reweighted, summed
              over the splits that were fit
    `te`, the prediction vector, the metric and the feature matrices are never touched.  Listed
    rows are never in `drop`, so the `ytr[tr].sum() == 0` guard cannot change."""
    pred = np.full(dte.height, -1e6)
    ftr, fte = dtr['behavior_family'].to_numpy(), dte['behavior_family'].to_numpy()
    str_, ste = dtr['split'].to_numpy(), dte['split'].to_numpy()
    ytr = dtr['relevance'].to_numpy().astype(float)
    ktr = (dtr['oof_rank'] <= topk).to_numpy()
    kte = (dte['oof_rank'] <= topk).to_numpy()
    Xtr_all = dtr.select(cols).to_numpy().astype(np.float64)
    Xte_all = dte.select(cols).to_numpy().astype(np.float64)
    keys = list(zip(dtr['pair_id'].to_list(), dtr['hand_id'].to_list()))           # [addition]
    drop_tr = None if drop is None else np.array([k in drop for k in keys], dtype=bool)
    in_w = w_tr = None
    if weight is not None:
        in_w = np.array([k in weight[0] for k in keys], dtype=bool)
        w_tr = np.where(in_w, float(weight[1]), 1.0)
    for fam in FAMILIES:
        for s in range(SPLITS):
            tr = np.where((ftr == fam) & ktr & (str_ != s))[0]
            te = np.where((fte == fam) & kte & (ste == s))[0]
            n_before = len(tr)                                                     # [addition]
            if drop_tr is not None:                                                # [addition 1]
                tr = tr[~drop_tr[tr]]
            if len(te) == 0 or len(tr) == 0 or ytr[tr].sum() == 0:
                if ledger is not None:                                             # [addition]
                    ledger.setdefault(fam, _ledger_row())['skipped_splits'].append(s)
                continue
            if ledger is not None:                                                 # [addition]
                row = ledger.setdefault(fam, _ledger_row())
                row['fits'] += 1
                row['training_rows_before_mask'] += n_before
                row['removed'] += n_before - len(tr)
                row['training_rows_used'] += len(tr)
                if w_tr is not None:
                    row['in_weight_set'] += int(in_w[tr].sum())
                    row['reweighted'] += int((w_tr[tr] != 1.0).sum())
            acc = np.zeros(len(te))
            for seed in seeds:
                clf = lgb.LGBMClassifier(**params, random_state=seed)
                if w_tr is None:
                    clf.fit(Xtr_all[tr], ytr[tr])
                else:                                                              # [addition 2]
                    clf.fit(Xtr_all[tr], ytr[tr], sample_weight=w_tr[tr])
                acc += clf.predict_proba(Xte_all[te])[:, 1] / len(seeds)
            pred[te] = acc
    return pred


# ------------------------------------------------------------------------------------------------
# step 88's refit-identity path, copied (closures inside step88.main cannot be imported)
# ------------------------------------------------------------------------------------------------
def frame_of(d2, oof_vec, split):
    """step88 frame_of: a stage-1 vector and a salt's split on the step-74 frame, oof_rank redone"""
    dd = d2.with_columns(pl.Series('oof', oof_vec), pl.Series('split', split))
    return dd.with_columns(pl.col('oof').rank('ordinal', descending=True).over('pair_id')
                           .cast(pl.UInt32).alias('oof_rank'))


def hybrid(dd, design, famv, truths, labels, fit):
    """step88 hybrid_incumbent + route, with the stage-2 fit function supplied.

    fit(dd_tr, dd_te, cols, design_name) is step74.fit_predict for ref and fit_predict_masked for
    the masked arms.  Both designs are fit on every family and split, routed per TRUE family
    (INCUMBENT_ROUTING: r382 for transfer and soft play, ref for isolation) and scored once."""
    preds = {k: fit(dd, dd, design[k], k) for k in S2_DESIGNS}
    pred = np.full(dd.height, -1e6)
    for f in FAMILIES:
        m = famv == f
        pred[m] = preds[INCUMBENT_ROUTING[f]][m]
    _, pp = evaluate(dd, pred, TOP, truths, labels)
    return float(np.mean(list(pp.values()))), pp, preds


def by_family(pp, labels):
    return {f: float(np.mean([v for q, v in pp.items() if labels[q] == f])) for f in FAMILIES}


def paired(ref_pp, arm_pp, labels, ptab):
    """mean per-pair delta arm - ref, overall, per true family and per registered table half"""
    dl = {q: arm_pp[q] - ref_pp[q] for q in ref_pp}
    halves = {}
    for h in (0, 1):
        qs = [q for q in dl if table_half(ptab[q]) == h]
        halves[str(h)] = {'mean_delta': float(np.mean([dl[q] for q in qs])) if qs else None,
                          'pairs': len(qs), 'tables': len({ptab[q] for q in qs})}
    return {'mean_delta': float(np.mean(list(dl.values()))),
            'by_family': {f: float(np.mean([v for q, v in dl.items() if labels[q] == f]))
                          for f in FAMILIES},
            'halves': halves,
            'pairs_better': int(sum(v > 0 for v in dl.values())),
            'pairs_worse': int(sum(v < 0 for v in dl.values())), 'n': len(dl)}


def stop_run(res, why, t0, prot_in):
    """a stop control failed: write the controls, quote no arm result, exit non-zero"""
    prot_out = check_protected()
    res['verdict'] = {'label': 'STOPPED', 'why': why, 'arm_results_written': False}
    res['protected_identical'] = bool(prot_in == prot_out)
    res['controls']['C5_protected'] = {'identical': res['protected_identical'],
                                       'pass': res['protected_identical']}
    res['prediction_sha256_exit'] = sha(PRED)
    res['prediction_sha256_exit_ok'] = res['prediction_sha256_exit'] == PRED_SHA
    res['seconds'] = round(time.time() - t0)
    write_json(OUT_JSON, res)
    log('STOPPED:', why)
    raise SystemExit(3)


# ------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    # ---- the registration and the inputs, verified before anything else ------------------------
    sha_in = sha(PRED)
    if sha_in != PRED_SHA:
        raise SystemExit(f'ABORT: prediction sha256 {sha_in} != registered {PRED_SHA}')
    reg = read_json(PRED)
    inputs = {k: sha(LP / k) for k in INPUT_SHA}
    if reg['inputs_sha256'] != INPUT_SHA or inputs != INPUT_SHA:
        raise SystemExit(f"ABORT: input sha256 {inputs} != registered {reg['inputs_sha256']}")
    assert [s.split(' ')[0] for s in reg['salts']] == list(SALTS) and SALTS[0] == PRIMARY
    assert reg['controls']['C1_refit_identity_target'] == C1_TARGET
    assert {k: reg['gate_A0'][k] for k in GATE} == GATE, 'gate constants differ from registration'
    assert reg['gate_A0']['bootstrap'].startswith(
        f'step65b_family_stage2.boot(a, b, ptab, pools, n={BOOT_N}, seed={BOOT_SEED})')
    assert tuple(S2_SEEDS_FULL) == tuple(STEP74_SEEDS) == (11, 12, 13) and (TOP, SPLITS) == (20, 5)
    assert INCUMBENT_ROUTING == {'directed_transfer': 'r382', 'soft_play': 'r382',
                                 'coordinated_isolation': 'ref'}
    prot_in = check_protected()
    log('=' * 100)
    log('STEP 94 [C].  CLEAN-LABEL MASK FROM THE (TYPE, TIME) LISTING RULE: A0 / A1 / T0 vs REF')
    log('=' * 100)
    log('prediction sha256 verified:', sha_in)
    log('inputs sha256 verified:', json.dumps(inputs))

    res = {'author': 'claude (agent)', 'step': 94, 'training_only': True,
           'held_folds_touched': False, 'proposed_for_submission': False,
           'prediction_file': 'experiments/clean_label_mask_C_20260913/'
                              'prediction-clean-label-mask.json',
           'prediction_sha256_entry': sha_in, 'prediction_sha256_entry_ok': True,
           'inputs_sha256': inputs, 'inputs_sha256_ok': True,
           'gate_text_registered': reg['gate_text'],
           'implementation': {
               'ref': 'imported step74.fit_predict inside step 88 hybrid_incumbent + route (copied)',
               'masked_arms': 'fit_predict_masked: step74.fit_predict line for line plus a '
                              'training-row filter on (pair_id, hand_id) and a per-row sample_weight',
               'stage1': 'the stored oof of audit-oof.parquet at every salt, oof_rank recomputed '
                         'by frame_of exactly as step 88',
               'mask_scope': 'the training index of every fit_predict call inside the hybrid, both '
                             'designs, every family and split; never test rows, the metric or a '
                             'feature; listed rows never masked',
               'stage2_seeds': list(S2_SEEDS_FULL), 'top': TOP, 'splits': SPLITS, 'clf': CLF,
               'routing': INCUMBENT_ROUTING, 'A1_weight': A1_WEIGHT,
               'gate_comparisons': 'unrounded values'},
           'environment': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                           'polars': pl.__version__, 'numpy': np.__version__,
                           'threads': {k: os.environ.get(k) for k in
                                       ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS')}},
           'controls': {}, 'frame_checks': {}}

    # ---- frames: step 88's construction; C4 exactly as step 88 checks it -----------------------
    audit = pl.read_parquet(LP / 'audit-oof.parquet')
    data, features, _, _ = load_fold(0)          # FEATURE COLUMNS only, as steps 74 / 85b / 88
    d = (audit.select('pair_id', 'hand_id')
         .join(data, on=['pair_id', 'hand_id'], how='left', validate='1:1')
         .join(audit.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance',
                            'chrono_rank', 'chrono_frac', 'n_hands',
                            pl.col('oof').alias('oof_saved')),
               on=['pair_id', 'hand_id'], validate='1:1')
         .sort('pair_id', 'hand_id'))
    held = set()
    for f in (0, 1, 2):
        held |= set(fold_pools(f))
    overlap = sorted(set(d['table_id']) & held)
    res['controls']['C4_no_held_overlap'] = {
        'held_pools_folds_0_1_2': len(held), 'audit_tables': int(d['table_id'].n_unique()),
        'overlap_tables': len(overlap), 'pass': len(overlap) == 0,
        'how': "held = union of fold_pools(0), fold_pools(1), fold_pools(2); "
               "not (set(d['table_id']) & held), with d built as step 88 builds it"}
    res['frame_checks']['step88_d_suffixed_columns_equal_audit'] = {
        c: bool((d[c] == d[f'{c}_right']).all()) for c in ('table_id', 'behavior_family')
        if f'{c}_right' in d.columns}
    d = d.select('pair_id', 'hand_id', 'table_id', 'behavior_family', 'relevance', 'oof_saved')
    del data, features
    gc.collect()
    log('C4', json.dumps(res['controls']['C4_no_held_overlap']), '| peak MB', peak_mb())
    if not res['controls']['C4_no_held_overlap']['pass']:
        stop_run(res, 'C4 failed: an audit table lies in a held fold pool', t0, prot_in)

    d0, struct, rcols = build_d0()
    d2 = frame_at(d0, 1.0)
    del d0
    gc.collect()
    assert d.select('pair_id', 'hand_id').equals(d2.select('pair_id', 'hand_id')), 'frame keys differ'
    famv = d['behavior_family'].to_numpy()
    truths = {k[0]: set(g['hand_id'])
              for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    design = {'r382': struct + CHRONO3 + S1 + rcols, 'ref': struct + CHRONO3 + S1}
    log('frames', json.dumps({'rows': d.height, 'pairs': len(truths), 'pools': len(pools),
                              'r382_cols': len(design['r382']), 'ref_cols': len(design['ref'])}),
        '| peak MB', peak_mb())

    # ---- C3: the registered mask and its counts, before any fit --------------------------------
    sdf = pl.read_parquet(LP / 'audit-structural-features.parquet',
                          columns=['pair_id', 'hand_id', 'pair_sd'])
    m = (audit.select('pair_id', 'hand_id', 'behavior_family', 'relevance', 'chrono_rank',
                      'oof_rank')
         .join(sdf, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id'))
    assert m.select('pair_id', 'hand_id').equals(d.select('pair_id', 'hand_id')), 'mask keys differ'
    amb, late, late_all, list_types = clean_label_masks(m)
    types, counts, uncapped, totals = design_counts(m, amb, late, late_all)
    c3 = {'listed_type_counts': str_keys(types),
          'listed_type_counts_registered': str_keys(CTRL_TYPE_SHARE),
          'listed_type_counts_match': types == CTRL_TYPE_SHARE,
          'design_counts_top20_unlisted': counts,
          'design_counts_top20_unlisted_match': counts == reg['design_counts_top20_unlisted'],
          'also_reproduced_not_a_stop_condition': {
              'time_rule_uncapped_draft_counts': uncapped == reg['time_rule_uncapped_draft_counts'],
              'design_totals': totals == reg['design_totals'],
              'list_types': list_types == reg['list_types']},
          'type_definition': 'type_showdown = (pair_sd == 2), fold type otherwise',
          'top20_definition': 'stored oof_rank <= 20 of audit-oof.parquet'}
    c3['pass'] = bool(c3['listed_type_counts_match'] and c3['design_counts_top20_unlisted_match'])
    res['controls']['C3_type_and_design_counts'] = c3
    res['frame_checks'].update({
        'behavior_family_d_equals_audit': bool((famv == m['behavior_family'].to_numpy()).all()),
        'chrono_rank_frame_equals_stored': bool(
            (d2['chrono_rank'].to_numpy() == m['chrono_rank'].to_numpy()).all()),
        'oof_rank_frame_equals_stored': bool(
            (d2['oof_rank'].to_numpy() == m['oof_rank'].to_numpy()).all()),
        'pair_sd_frame_equals_features': bool(
            (d2['pair_sd'].to_numpy() == m['pair_sd'].to_numpy()).all())})
    keys = list(zip(m['pair_id'].to_list(), m['hand_id'].to_list()))
    amb_set = {k for k, x in zip(keys, amb) if x}
    t0_set = {k for k, x in zip(keys, late) if x}
    res['mask_rows_all_ranks'] = {
        f: {'type_ambiguous': int((amb & (famv == f)).sum()),
            'T0_time_rule': int((late & (famv == f)).sum())} for f in FAMILIES}
    log('C3', json.dumps({k: c3[k] for k in ('pass', 'listed_type_counts_match',
                                             'design_counts_top20_unlisted_match',
                                             'also_reproduced_not_a_stop_condition')}))
    if not c3['pass']:
        stop_run(res, 'C3 failed: the mask does not reproduce the registered counts', t0, prot_in)

    # ---- the primary salt: ref, then C1, C2 and C6, all before any arm -------------------------
    oof_saved = d['oof_saved'].to_numpy()

    def fit_ref(a, b, cols, dname):
        return fit_predict(a, b, cols, TOP, seeds=S2_SEEDS_FULL)        # the imported step 74 fit

    def fit_masked(drop=None, weight=None, ledger=None):
        def fit(a, b, cols, dname):
            lg = None if ledger is None else ledger.setdefault(dname, {})
            return fit_predict_masked(a, b, cols, TOP, seeds=S2_SEEDS_FULL, drop=drop,
                                      weight=weight, ledger=lg)
        return fit

    frames = {PRIMARY: frame_of(d2, oof_saved, resplit(d, PRIMARY)['split'].to_numpy())}
    dd = frames[PRIMARY]
    t1 = time.time()
    a_ref, pp_ref, preds_ref = hybrid(dd, design, famv, truths, labels, fit_ref)
    _, ppr = evaluate(dd, preds_ref['r382'], TOP, truths, labels)
    _, ppf = evaluate(dd, preds_ref['ref'], TOP, truths, labels)
    c1 = {'ref_map5_6dp': round(a_ref, 6), 'ref_map5': a_ref, 'target': C1_TARGET,
          'abs_deviation': abs(a_ref - C1_TARGET), 'tolerance': C1_TOL,
          'pass': bool(abs(a_ref - C1_TARGET) < C1_TOL), 'step88_ANCHOR_A3': ANCHOR_A3,
          'route_equals_step85b_per_pair_recombination':
              {q: (ppf[q] if labels[q] == CI else ppr[q]) for q in ppr} == pp_ref}
    res['controls']['C1_refit_identity'] = c1
    log('C1', json.dumps(c1), f'({time.time() - t1:.0f} s, peak MB {peak_mb()})')
    if not c1['pass']:
        stop_run(res, 'C1 failed: ref does not reproduce the step 88 refit identity', t0, prot_in)

    t1 = time.time()
    led_c2 = {}
    a_c2, pp_c2, preds_c2 = hybrid(dd, design, famv, truths, labels,
                                   fit_masked(drop=set(), ledger=led_c2))
    mad2 = max(float(np.max(np.abs(preds_c2[k] - preds_ref[k]))) for k in S2_DESIGNS)
    c2 = {'max_abs_pred_diff': mad2, 'map5_6dp': round(a_c2, 6),
          'per_pair_ap_identical': pp_c2 == pp_ref, 'rows_removed': total(led_c2, 'removed'),
          'fits': total(led_c2, 'fits'), 'pass': bool(mad2 == 0.0)}
    res['controls']['C2_empty_mask_identity'] = c2
    log('C2', json.dumps(c2), f'({time.time() - t1:.0f} s)')
    if not c2['pass']:
        stop_run(res, 'C2 failed: the empty mask does not reproduce the ref predictions', t0,
                 prot_in)

    t1 = time.time()
    led_c6 = {}
    a_c6, pp_c6, preds_c6 = hybrid(dd, design, famv, truths, labels,
                                   fit_masked(weight=(amb_set, 1.0), ledger=led_c6))
    mad6 = max(float(np.max(np.abs(preds_c6[k] - preds_ref[k]))) for k in S2_DESIGNS)
    c6 = {'map5_6dp': round(a_c6, 6), 'ref_map5_6dp': round(a_ref, 6),
          'map5_equal_at_6dp': round(a_c6, 6) == round(a_ref, 6), 'max_abs_pred_diff': mad6,
          'pred_tolerance': C6_PRED_TOL, 'per_pair_ap_identical': pp_c6 == pp_ref,
          'training_rows_in_weight_set': total(led_c6, 'in_weight_set'),
          'training_rows_with_weight_not_1': total(led_c6, 'reweighted'),
          'what_it_is': 'the A1 code path with w = 1.0: sample_weight is passed to every LightGBM '
                        'fit and equals 1.0 on every training row'}
    c6['pass'] = bool(c6['map5_equal_at_6dp'] and mad6 <= C6_PRED_TOL)
    res['controls']['C6_unit_weight_identity'] = c6
    log('C6', json.dumps(c6), f'({time.time() - t1:.0f} s)')
    if not c6['pass']:
        stop_run(res, 'C6 failed: unit weights do not reproduce ref', t0, prot_in)

    # ---- the arms at every salt, each compared with the ref of the same salt -------------------
    arm_spec = {'A0': dict(drop=amb_set), 'A1': dict(weight=(amb_set, A1_WEIGHT)),
                'T0': dict(drop=t0_set)}
    refs = {PRIMARY: (a_ref, pp_ref)}
    per_pair = {(PRIMARY, 'ref'): pp_ref, (PRIMARY, 'C2_empty_mask'): pp_c2,
                (PRIMARY, 'C6_unit_weight'): pp_c6}
    del preds_ref, preds_c2, preds_c6
    res['by_salt'], res['training_rows_masked'] = {}, {}
    for salt in SALTS:
        if salt not in frames:
            frames[salt] = frame_of(d2, oof_saved, resplit(d, salt)['split'].to_numpy())
            t1 = time.time()
            a, pp, _ = hybrid(frames[salt], design, famv, truths, labels, fit_ref)
            refs[salt] = (a, pp)
            per_pair[(salt, 'ref')] = pp
            log(f'{salt} ref', json.dumps({'map5': round(a, 6), 'sec': round(time.time() - t1)}))
        a_r, pp_r = refs[salt]
        row = {'ref': {'map5': a_r, 'by_family': by_family(pp_r, labels)}}
        res['training_rows_masked'][salt] = {}
        for arm in MASKED_ARMS:
            t1 = time.time()
            led = {}
            a, pp, _ = hybrid(frames[salt], design, famv, truths, labels,
                              fit_masked(ledger=led, **arm_spec[arm]))
            per_pair[(salt, arm)] = pp
            pr = paired(pp_r, pp, labels, ptab)
            row[arm] = {'map5': a, 'map5_minus_ref': a - a_r, 'by_family': by_family(pp, labels),
                        'mean_per_pair_delta': pr['mean_delta'], 'family_delta': pr['by_family'],
                        'halves': pr['halves'], 'pairs_better': pr['pairs_better'],
                        'pairs_worse': pr['pairs_worse'], 'n': pr['n']}
            if salt == PRIMARY:
                row[arm]['boot'] = boot(pp_r, pp, ptab, pools, n=BOOT_N, seed=BOOT_SEED)
            res['training_rows_masked'][salt][arm] = led
            log(f'{salt} {arm}', json.dumps({'map5': round(a, 6),
                                             'delta': round(pr['mean_delta'], 6),
                                             'removed': total(led, 'removed'),
                                             'reweighted': total(led, 'reweighted'),
                                             'sec': round(time.time() - t1), 'peak_mb': peak_mb()}))
        res['by_salt'][salt] = row
        del frames[salt]
        gc.collect()

    # ---- the registered A0 gate ----------------------------------------------------------------
    p = res['by_salt'][PRIMARY]['A0']
    bb = p['boot']
    halves = {h: v['mean_delta'] for h, v in p['halves'].items()}
    others = {s: res['by_salt'][s]['A0']['mean_per_pair_delta'] for s in SALTS if s != PRIMARY}
    gate = {'salt': PRIMARY,
            'mean_delta': bb['mean_delta'], 'delta_min': GATE['delta_min'],
            'delta_ok': bool(bb['mean_delta'] >= GATE['delta_min']),
            'ci95': bb['ci95'], 'ci95_lower_gt': GATE['ci95_lower_gt'],
            'ci_ok': bool(bb['ci95'][0] > GATE['ci95_lower_gt']),
            'family_delta': p['family_delta'], 'family_floor': GATE['family_floor'],
            'family_ok': bool(all(v >= GATE['family_floor'] for v in p['family_delta'].values())),
            'table_halves_delta': halves,
            'halves_ok': bool(all(v is not None and v >= 0 for v in halves.values())),
            'other_salts_delta': others,
            'other_salts_ok': bool(all(v >= 0 for v in others.values())),
            'boot_mean_equals_paired_mean': bb['mean_delta'] == p['mean_per_pair_delta'],
            'comparisons_use': 'unrounded values'}
    gate['PASS'] = bool(gate['delta_ok'] and gate['ci_ok'] and gate['family_ok']
                        and gate['halves_ok'] and gate['other_salts_ok'])
    gate['verdict'] = 'PASS' if gate['PASS'] else 'FAIL'
    res['gate_A0'] = gate
    res['boot_primary'] = {arm: res['by_salt'][PRIMARY][arm]['boot'] for arm in MASKED_ARMS}
    res['halves_primary'] = {arm: res['by_salt'][PRIMARY][arm]['halves'] for arm in MASKED_ARMS}
    summ = {}
    for s in SALTS:
        summ[s] = {}
        for arm in MASKED_ARMS:
            led = res['training_rows_masked'][s][arm]
            summ[s][arm] = {f: {k: led['r382'][f][k] for k in
                                ('removed', 'reweighted', 'training_rows_before_mask',
                                 'training_rows_used', 'fits')} for f in FAMILIES}
            summ[s][arm]['identical_in_both_designs'] = bool(led['r382'] == led['ref'])
    res['training_rows_masked_summary'] = summ
    res['training_rows_masked_summary_note'] = (
        'per family, summed over the splits that were fit, for ONE fit_predict call; the hybrid '
        'makes two (r382 and ref designs) with identical training indices, compared in '
        'identical_in_both_designs.  removed = training rows dropped (A0, T0); reweighted = '
        'training rows fit with sample_weight .5 (A1).  Full per-design ledgers are in '
        'training_rows_masked.')

    # ---- per-pair AP of every arm and salt -----------------------------------------------------
    rows = []
    for (s, arm), pp in per_pair.items():
        rp = refs[s][1]
        for q in sorted(pp):
            rows.append((s, arm, q, ptab[q], labels[q], table_half(ptab[q]), float(pp[q]),
                         float(rp[q]), float(pp[q] - rp[q])))
    pl.DataFrame(rows, orient='row', schema=[
        ('salt', pl.String), ('arm', pl.String), ('pair_id', pl.String), ('table_id', pl.String),
        ('behavior_family', pl.String), ('table_half', pl.Int64), ('ap', pl.Float64),
        ('ref_ap', pl.Float64), ('delta_vs_ref', pl.Float64)]).write_parquet(OUT_PP)
    res['per_pair_ap_file'] = OUT_PP.name
    res['per_pair_ap_arms'] = sorted(f'{s}|{arm}' for s, arm in per_pair)

    # ---- exit checks ---------------------------------------------------------------------------
    prot_out = check_protected()
    res['protected_identical'] = bool(prot_in == prot_out)
    res['controls']['C5_protected'] = {'entry': prot_in, 'exit': prot_out,
                                       'identical': res['protected_identical'],
                                       'pass': res['protected_identical']}
    res['controls']['stop_controls_pass'] = {
        k: res['controls'][k]['pass'] for k in ('C1_refit_identity', 'C2_empty_mask_identity',
                                                'C4_no_held_overlap', 'C6_unit_weight_identity')}
    res['inputs_sha256_exit_ok'] = {k: sha(LP / k) for k in INPUT_SHA} == INPUT_SHA
    sha_out = sha(PRED)
    res['prediction_sha256_exit'] = sha_out
    res['prediction_sha256_exit_ok'] = sha_out == PRED_SHA
    res['peak_working_set_mb'] = peak_mb()
    res['seconds'] = round(time.time() - t0)
    if sha_out != PRED_SHA:
        res['gate_A0']['verdict'] = 'ABORTED: the prediction file changed during the run'
        write_json(OUT_JSON, res)
        raise SystemExit('ABORT: prediction sha256 changed during the run')
    write_json(OUT_JSON, res)

    # ---- printout ------------------------------------------------------------------------------
    def fx(v):
        return 'None' if v is None else f'{v:+.6f}'

    cc = res['controls']
    log()
    log(f"  C1 ref MAP@5 {cc['C1_refit_identity']['ref_map5_6dp']:.6f} vs {C1_TARGET} (|dev| "
        f"{cc['C1_refit_identity']['abs_deviation']:.2e}) pass={cc['C1_refit_identity']['pass']}")
    log(f"  C2 empty mask max|dpred| {cc['C2_empty_mask_identity']['max_abs_pred_diff']} "
        f"pass={cc['C2_empty_mask_identity']['pass']}")
    log(f"  C3 type counts {c3['listed_type_counts_match']}, design counts "
        f"{c3['design_counts_top20_unlisted_match']} pass={c3['pass']}")
    log(f"  C4 held overlap {cc['C4_no_held_overlap']['overlap_tables']} "
        f"pass={cc['C4_no_held_overlap']['pass']}")
    log(f"  C5 protected identical={res['protected_identical']}")
    log(f"  C6 unit weights MAP@5 {cc['C6_unit_weight_identity']['map5_6dp']:.6f} max|dpred| "
        f"{cc['C6_unit_weight_identity']['max_abs_pred_diff']:.2e} "
        f"pass={cc['C6_unit_weight_identity']['pass']}")
    log()
    log('  MAP@5 BY SALT AND ARM  (brackets: mean per-pair delta vs the ref of the same salt)')
    for s in SALTS:
        r = res['by_salt'][s]
        log(f"    {s:16s} ref {r['ref']['map5']:.6f}  " + '  '.join(
            f"{arm} {r[arm]['map5']:.6f} ({fx(r[arm]['mean_per_pair_delta'])})" for arm in MASKED_ARMS))
    log()
    for arm in MASKED_ARMS:
        r = res['by_salt'][PRIMARY][arm]
        log(f"    {arm} @ {PRIMARY}: fam " + ' '.join(f'{f[:2]} {fx(v)}' for f, v in r['family_delta'].items())
            + '  halves ' + ' '.join(f"{h}:{fx(v['mean_delta'])}" for h, v in r['halves'].items())
            + f"  boot {fx(r['boot']['mean_delta'])} [{fx(r['boot']['ci95'][0])}, "
              f"{fx(r['boot']['ci95'][1])}] better {r['boot']['pairs_better']} "
              f"worse {r['boot']['pairs_worse']} n {r['boot']['n']}")
    log()
    log(f"  A0 GATE: delta {fx(gate['mean_delta'])} ok={gate['delta_ok']} | ci95 lower "
        f"{fx(gate['ci95'][0])} ok={gate['ci_ok']} | families ok={gate['family_ok']} | halves "
        f"ok={gate['halves_ok']} | other salts ok={gate['other_salts_ok']}")
    log(f"  VERDICT A0: {gate['verdict']}   ({res['seconds']} s, peak MB {res['peak_working_set_mb']})")
    log(f'  wrote {OUT_JSON}')
    log(f'  wrote {OUT_PP}')
    log('DONE')


if __name__ == '__main__':
    main()
