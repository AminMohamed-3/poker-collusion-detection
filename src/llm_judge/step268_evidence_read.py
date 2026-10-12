"""STEP268 D -- the EVIDENCE read for the encoder heads: STEP264's fusion check plus arm M4.

A thin wrapper.  It IMPORTS src/llm_judge/step264_fusion_check.py and does not modify it: the
harness, the STEP251 base, the registered MECHPLUS REF factor, the 57-query / 20-candidate scope,
the MAP@5 machinery, the 2,000-draw paired bootstrap (seed 264) and the score-file contract are
all that module's.  What this file adds:

  M4   THE REPLACEMENT ARM.  Score vector = BASE with the 1,140 candidate rows REPLACED by
       10 + p_llm (the same 10 + p encoding `fuse` uses), so the decoder orders those candidates
       by the head's probability alone.  Reported as M4 - M1 and M4 - M0 through the imported
       `dstat`, on the same per-query bootstrap draws.
       M2 (the fused ladder), M3 and the standalone AP block come from `run_one` unchanged.

  GATES, exactly as registration268.json states them:
       "evidence": gate 2 alongside   M2 - M1 >= +.005, P >= .75, both halves >= 0 vs the NULL row
                   gate versus        M4 - M1 >= 0,     P >= .75, both halves
       The NULL row is the Uniform(0,100) third factor of fusion_check_validation.json (written
       by step264_fusion_check.py --validate); "both halves >= 0 vs the NULL row" is read as
       h0 and h1 each at or above the NULL run's h0 / h1 at the same w3.  Every w3 of the grid is
       reported; nothing is selected.

  and it runs score_heldout_eval.py's `evaluate(load(...))` on the SUBSET score file, so the
  hand-level AP / AUC / within-pair / stacking read comes out of the same command.

USAGE
  python step268_evidence_read.py --scope57 <heads/h1_scope57_scores.jsonl> \\
                                  --subset  <heads/h1_subset_scores.jsonl> \\
                                  [--tag NAME] [--w3 .1,.2,.3] [--out eval/evidence_read.json]
  python step268_evidence_read.py --selftest      # runs the whole thing on STEP264's
                                                  # base_nothink_scores.jsonl and writes
                                                  # eval/evidence_read_selftest.json
Development phase only (folds 3-4).  Reads only; builds no submission; uploads nothing.
"""
from __future__ import annotations

import os
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'

import argparse
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = r'C:\Data\Projects\Money Making\work\poker'
CV = ROOT + r'\Claude Version'
if CV + r'\src' not in sys.path:
    sys.path.insert(0, CV + r'\src')

import step264_fusion_check as FC                                              # noqa: E402
import score_heldout_eval as SE                                                # noqa: E402

EXP = CV + r'\experiments\step268_qwen06b_encoder_heads_20260918'
EVAL = EXP + r'\eval'
S264 = CV + r'\experiments\step264_llm_hand_judge_20260918'
VALIDATION = S264 + r'\fusion_check_validation.json'

GATE2_MIN = .005          # M2 - M1
GATE2_P = .75
GATE_VS_MIN = .0          # M4 - M1
GATE_VS_P = .75


def m4_arm(ctx, p_llm):
    """BASE with the candidate rows replaced by 10 + p_llm - the decoder sees only the head."""
    s = ctx['BASE'].copy()
    qix = np.asarray(ctx['QIX'])
    s[qix[ctx['cand_rows']]] = 10.0 + p_llm[ctx['cand_rows']]
    return ctx['mapof'](s)


def null_row(w3s):
    """The Uniform(0,100) third factor from step264_fusion_check.py --validate, if it is there."""
    if not os.path.exists(VALIDATION):
        return None, 'fusion_check_validation.json is not present; run step264_fusion_check.py ' \
                     '--validate to get the NULL row'
    v = json.load(open(VALIDATION, encoding='utf-8'))
    n = v.get('runs', {}).get('NULL', {}).get('M2_minus_M1', {})
    got = {k: n[k] for k in n if any(k.endswith('w3=%g' % w) for w in w3s)}
    return (got, None) if got else (None, 'no NULL row at this w3 grid')


def gates(res, w3s, null):
    """registration268 'evidence': gate 2 alongside and gate versus, evaluated mechanically."""
    out = {'rule_alongside': 'M2 - M1 >= +%.3f, P(>0) >= %.2f, both halves >= the NULL row'
                             % (GATE2_MIN, GATE2_P),
           'rule_versus': 'M4 - M1 >= %.3f, P(>0) >= %.2f, both halves >= 0'
                          % (GATE_VS_MIN, GATE_VS_P),
           'null_row_source': VALIDATION if null else None,
           'alongside': {}, 'versus': {}}
    for sname in ('dt_sp', 'all_families'):
        for w3 in w3s:
            k = '%s@w3=%g' % (sname, w3)
            d = res['M2_minus_M1'][k]
            nb = (null or {}).get(k)
            c = {'delta>=+%.3f' % GATE2_MIN: d['delta'] >= GATE2_MIN,
                 'P_gt_0>=%.2f' % GATE2_P: d['P_gt_0'] >= GATE2_P,
                 'h0>=null': (d['h0'] >= nb['h0']) if nb else None,
                 'h1>=null': (d['h1'] >= nb['h1']) if nb else None}
            out['alongside'][k] = {
                'delta': d['delta'], 'P_gt_0': d['P_gt_0'], 'h0': d['h0'], 'h1': d['h1'],
                'null_delta': nb['delta'] if nb else None,
                'conditions': {kk: (None if vv is None else bool(vv)) for kk, vv in c.items()},
                'verdict': ('UNDECIDABLE_no_null_row' if any(v is None for v in c.values())
                            else ('PASS' if all(c.values()) else 'FAIL'))}
    d = res['M4_minus_M1']
    c = {'delta>=%.3f' % GATE_VS_MIN: d['delta'] >= GATE_VS_MIN,
         'P_gt_0>=%.2f' % GATE_VS_P: d['P_gt_0'] >= GATE_VS_P,
         'h0>=0': d['h0'] >= 0, 'h1>=0': d['h1'] >= 0}
    out['versus'] = {'delta': d['delta'], 'P_gt_0': d['P_gt_0'], 'h0': d['h0'], 'h1': d['h1'],
                     'conditions': {k: bool(v) for k, v in c.items()},
                     'verdict': 'PASS' if all(c.values()) else 'FAIL'}
    verdicts = [b['verdict'] for b in out['alongside'].values()]
    out['gate2_any_w3_PASS'] = 'PASS' in verdicts
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope57', default=None, help='score file over the 57x20 audit candidates')
    ap.add_argument('--subset', default=None, help='score file over the 2,595 held-out subset')
    ap.add_argument('--w3', default='.1,.2,.3')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default=EVAL + r'\evidence_read.json')
    ap.add_argument('--selftest', action='store_true',
                    help="run on STEP264's base_nothink_scores.jsonl (same file both sides)")
    a = ap.parse_args(argv)
    w3s = [float(x) for x in a.w3.split(',') if x.strip()]
    assert w3s and all(0 < w < 1 for w in w3s), 'the w3 grid must be strictly inside (0, 1)'
    if a.selftest:
        a.scope57 = a.scope57 or S264 + r'\base_nothink_scores.jsonl'
        a.subset = a.subset or S264 + r'\base_nothink_scores.jsonl'
        a.tag = a.tag or 'SELFTEST_base_nothink'
        a.out = EVAL + r'\evidence_read_selftest.json'
    assert a.scope57, 'pass --scope57 heads/h1_scope57_scores.jsonl (or --selftest)'
    os.makedirs(EVAL, exist_ok=True)
    tag = a.tag or os.path.splitext(os.path.basename(a.scope57))[0]
    t0 = time.time()

    ctx = FC.load_context()
    res = FC.run_one(ctx, a.scope57, w3s, tag)

    # ---- arm M4, built on the SAME p_llm vector run_one used
    conf, _mal, _n = FC.read_scores(a.scope57)
    p_llm, hit, _vals = FC.llm_factor(ctx, conf)
    M4 = m4_arm(ctx, p_llm)
    hv, fq, IDX = ctx['hv'], ctx['fq'], ctx['IDX']
    res['M4_level'] = FC.lvl(M4, hv)
    res['M4_minus_M1'] = FC.dstat(M4 - ctx['M1'], hv, fq, IDX)
    res['M4_minus_M0'] = FC.dstat(M4 - ctx['M0'], hv, fq, IDX)
    res['M4_definition'] = ('score vector = BASE (v1_DTfp3) with the 57x20 candidate rows set to '
                            '10 + p_llm; every non-candidate row untouched.  The decoder then '
                            'orders the candidates by the head alone - a REPLACEMENT arm, not a '
                            'fusion.')
    res['M4_candidate_coverage'] = round(float(hit.mean()), 6)

    null, note = null_row(w3s)
    res['gates'] = gates(res, w3s, null)
    if note:
        res['gates']['note'] = note

    print('\n' + tag)
    print(FC.table(res, w3s))
    print('  M4 (replacement)  %.5f   M4-M1 %+.5f (P>0 %.3f, h0 %+.5f, h1 %+.5f)   '
          'M4-M0 %+.5f' % (res['M4_level']['map5'], res['M4_minus_M1']['delta'],
                           res['M4_minus_M1']['P_gt_0'], res['M4_minus_M1']['h0'],
                           res['M4_minus_M1']['h1'], res['M4_minus_M0']['delta']))
    print('  gate versus (M4): ' + res['gates']['versus']['verdict']
          + '   gate 2 alongside (M2): '
          + ('PASS at some w3' if res['gates']['gate2_any_w3_PASS'] else 'FAIL at every w3'))

    # ---- the hand-level read on the subset file
    if a.subset:
        try:
            res['heldout_eval'] = SE.evaluate(SE.load(a.subset))
            res['heldout_eval_file'] = a.subset
            print('  heldout eval (%s): %s' % (os.path.basename(a.subset),
                                               json.dumps(res['heldout_eval']
                                                          .get('event_detection', {}))))
        except AssertionError as e:                                       # noqa: PERF203
            res['heldout_eval_error'] = str(e)
            print('  heldout eval FAILED:', e)

    res['wall_seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    print('\nwrote', a.out, 'in', res['wall_seconds'], 's')


if __name__ == '__main__':
    main()
