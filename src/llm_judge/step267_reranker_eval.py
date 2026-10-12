"""STEP267: read a TEXT-RERANKER score file through the STEP264 audit fusion harness.

This is step264_fusion_check.py's ladder (M0 / M1 / M2(w3) / M3(w3), 57 audit queries, folds 3-4,
MECHPLUS REF as the second factor) PLUS the two things registration267.json asks for that the
STEP264 script does not compute:

  M4_alone   the "VERSUS" arm: the reranker's probability ALONE through the production decoder on
             the top-20 rows.  The score vector is BASE with the 1,140 candidate rows replaced by
             10 + p_rr, so stage 1 is untouched and only the top-20 candidates are re-ordered.
             This is rank-identical to M3 at w3 = 1 (prior_shift is strictly monotone in p), which
             is why registration267.json calls it "M3 with w3=1".
             Reported as M4_level, M4_minus_M1, M4_minus_M0 through the same dstat bootstrap.
  gates      gate2_alongside and gate_versus of registration267.json, evaluated MECHANICALLY per
             w3, with the NULL-row reference values copied from fusion_check_validation.json.

step264_fusion_check.py is IMPORTED, never modified: the context, the fusion form, the decoder,
the MAP@5 machinery and the bootstrap indices are literally its own.

    python step267_reranker_eval.py --scores a.jsonl b.jsonl --out eval/result.json
                                    [--w3 .1,.2,.3] [--tags A,B] [--subset-eval sub.jsonl]

--subset-eval additionally runs score_heldout_eval.py's own machinery (AP/AUC vs mechplus on the
2,595-row held-out subset) and folds its JSON into the result.

Development-phase data only.  Reads only; writes one JSON under experiments/step267_.../eval/.
"""
from __future__ import annotations

import os
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import step264_fusion_check as F                                               # noqa: E402

CV = Path(F.POKER) / 'Claude Version'
EXP267 = CV / 'experiments/step267_text_rerankers_20260918'
REG267 = EXP267 / 'registration267.json'
VALID = F.EXP / 'fusion_check_validation.json'

DT, SP, CI = F.DT, F.SP, F.CI
SHORT = F.SHORT

# registration267.json, gates block, transcribed as numbers so they can be checked mechanically
GATE2_MIN_DELTA = .005
GATE2_MIN_P = .75
GATE_VERSUS_MIN_DELTA = .0
GATE_VERSUS_MIN_P = .75


# --------------------------------------------------------------------------------- M4 (alone)
def m4_vector(ctx, p_llm):
    """BASE with the 57x20 candidate rows replaced by 10 + p_rr: only those rows re-order."""
    s = ctx['BASE'].copy()
    s[ctx['QIX'][ctx['cand_rows']]] = 10.0 + p_llm[ctx['cand_rows']]
    return s


def gate_strings(res, w3s):
    """PASS/FAIL for the two registered gates, per w3 and scope, mechanically."""
    null_ref = {}
    if VALID.exists():
        v = json.loads(VALID.read_text(encoding='utf-8'))
        null_ref = {k: val['delta'] for k, val in
                    v.get('runs', {}).get('NULL', {}).get('M2_minus_M1', {}).items()}
    out = {'rule_gate2': ('M2 - M1 >= +%.3f with bootstrap P(>0) >= %.2f at a reported w3 and '
                          'both halves >= 0' % (GATE2_MIN_DELTA, GATE2_MIN_P)),
           'rule_gate_versus': ('M4 - M1 >= %.0f with P(>0) >= %.2f and both halves >= 0'
                                % (GATE_VERSUS_MIN_DELTA, GATE_VERSUS_MIN_P)),
           'null_reference_from_fusion_check_validation': null_ref,
           'gate2_alongside': {}, 'gate2_any_w3_PASS': False}
    for sname in ('dt_sp', 'all_families'):
        for w3 in w3s:
            k = '%s@w3=%g' % (sname, w3)
            d = res['M2_minus_M1'][k]
            ok = (d['delta'] >= GATE2_MIN_DELTA and d['P_gt_0'] >= GATE2_MIN_P
                  and d['h0'] >= 0 and d['h1'] >= 0)
            out['gate2_alongside'][k] = {
                'delta': d['delta'], 'P_gt_0': d['P_gt_0'], 'h0': d['h0'], 'h1': d['h1'],
                'null_row_delta': null_ref.get(k),
                'above_null_by': (round(d['delta'] - null_ref[k], 6) if k in null_ref else None),
                'verdict': 'PASS' if ok else 'FAIL',
                'why': ('all four conditions met' if ok else '; '.join(
                    x for x in [
                        None if d['delta'] >= GATE2_MIN_DELTA
                        else 'delta %+.5f < %+.3f' % (d['delta'], GATE2_MIN_DELTA),
                        None if d['P_gt_0'] >= GATE2_MIN_P
                        else 'P(>0) %.3f < %.2f' % (d['P_gt_0'], GATE2_MIN_P),
                        None if d['h0'] >= 0 else 'h0 %+.5f < 0' % d['h0'],
                        None if d['h1'] >= 0 else 'h1 %+.5f < 0' % d['h1']] if x))}
            if ok and sname == 'dt_sp':
                out['gate2_any_w3_PASS'] = True
    d4 = res['M4_minus_M1']
    ok4 = (d4['delta'] >= GATE_VERSUS_MIN_DELTA and d4['P_gt_0'] >= GATE_VERSUS_MIN_P
           and d4['h0'] >= 0 and d4['h1'] >= 0)
    out['gate_versus'] = {
        'delta': d4['delta'], 'P_gt_0': d4['P_gt_0'], 'h0': d4['h0'], 'h1': d4['h1'],
        'verdict': 'PASS' if ok4 else 'FAIL',
        'why': ('all four conditions met' if ok4 else '; '.join(
            x for x in [
                None if d4['delta'] >= GATE_VERSUS_MIN_DELTA
                else 'delta %+.5f < %+.0f' % (d4['delta'], GATE_VERSUS_MIN_DELTA),
                None if d4['P_gt_0'] >= GATE_VERSUS_MIN_P
                else 'P(>0) %.3f < %.2f' % (d4['P_gt_0'], GATE_VERSUS_MIN_P),
                None if d4['h0'] >= 0 else 'h0 %+.5f < 0' % d4['h0'],
                None if d4['h1'] >= 0 else 'h1 %+.5f < 0' % d4['h1']] if x))}
    return out


def standalone_extra(ctx, p_llm):
    """AP and AUC of the reranker alone on the 57x20 candidates, overall and by family."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = ctx['cand_rel']
    s = p_llm[ctx['cand_rows']]
    base = ctx['BASE'][ctx['QIX']][ctx['cand_rows']]
    own = np.array([ctx['mech'][f][r] for f, r in zip(ctx['cand_fam'], ctx['cand_rows'])])
    out = {'overall': {'n': int(len(y)), 'n_listed': int(y.sum()),
                       'rr_ap': round(float(average_precision_score(y, s)), 5),
                       'rr_auc': round(float(roc_auc_score(y, s)), 5),
                       'base_v1_ap': round(float(average_precision_score(y, base)), 5),
                       'base_v1_auc': round(float(roc_auc_score(y, base)), 5),
                       'mechplus_own_family_ap': round(float(average_precision_score(y, own)), 5),
                       'mechplus_own_family_auc': round(float(roc_auc_score(y, own)), 5)},
           'by_family': {}}
    for f in (DT, SP, CI):
        m = ctx['cand_fam'] == f
        if y[m].sum() and (1 - y[m]).sum():
            out['by_family'][SHORT[f]] = {
                'n': int(m.sum()), 'n_listed': int(y[m].sum()),
                'rr_ap': round(float(average_precision_score(y[m], s[m])), 5),
                'rr_auc': round(float(roc_auc_score(y[m], s[m])), 5),
                'base_v1_ap': round(float(average_precision_score(y[m], base[m])), 5),
                'mechplus_ap': round(float(average_precision_score(y[m], own[m])), 5)}
    return out


def run_file(ctx, path, w3s, tag):
    """step264_fusion_check.run_one PLUS M4 and the gates.  One read of the score file."""
    res = F.run_one(ctx, path, w3s, tag)
    conf, _mal, _n = F.read_scores(path)
    p_llm, _hit, _vals = F.llm_factor(ctx, conf)
    hv, fq, IDX = ctx['hv'], ctx['fq'], ctx['IDX']
    M4 = ctx['mapof'](m4_vector(ctx, p_llm))
    res['M4_level'] = F.lvl(M4, hv)
    res['M4_minus_M1'] = F.dstat(M4 - ctx['M1'], hv, fq, IDX)
    res['M4_minus_M0'] = F.dstat(M4 - ctx['M0'], hv, fq, IDX)
    res['M4_definition'] = ('BASE with the 1,140 top-20 candidate rows replaced by 10 + p_rr; '
                            'rank-identical to M3 at w3 = 1 because prior_shift is monotone')
    res['standalone'] = standalone_extra(ctx, p_llm)
    res['gates'] = gate_strings(res, w3s)
    return res


def table(res, w3s):
    L = [F.table(res, w3s)]
    L.append('  M4 (reranker ALONE through the decoder) %.5f   M4-M1 %+.5f  P>0 %.3f  '
             'halves %+.5f / %+.5f   M4-M0 %+.5f'
             % (res['M4_level']['map5'], res['M4_minus_M1']['delta'],
                res['M4_minus_M1']['P_gt_0'], res['M4_minus_M1']['h0'],
                res['M4_minus_M1']['h1'], res['M4_minus_M0']['delta']))
    o = res['standalone']['overall']
    L.append('  standalone on the 57x20: rr AP %.4f AUC %.4f | base v1 AP %.4f | '
             'mechplus own-family AP %.4f'
             % (o['rr_ap'], o['rr_auc'], o['base_v1_ap'], o['mechplus_own_family_ap']))
    g = res['gates']
    best = max(w3s)
    L.append('  GATE2 dt_sp@w3=%g: %s (%s)'
             % (best, g['gate2_alongside']['dt_sp@w3=%g' % best]['verdict'],
                g['gate2_alongside']['dt_sp@w3=%g' % best]['why']))
    L.append('  GATE2 any reported w3 on dt_sp: %s'
             % ('PASS' if g['gate2_any_w3_PASS'] else 'FAIL'))
    L.append('  GATE_VERSUS: %s (%s)' % (g['gate_versus']['verdict'], g['gate_versus']['why']))
    return '\n'.join(L)


def subset_eval(path):
    """score_heldout_eval.py's own machinery on a score file over the 2,595-row subset."""
    import score_heldout_eval as SHE
    return SHE.evaluate(SHE.load(path))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True)
    ap.add_argument('--out', default=str(EXP267 / 'eval/result.json'))
    ap.add_argument('--w3', default='.1,.2,.3')
    ap.add_argument('--tags', default=None,
                    help='comma-separated labels, one per --scores file')
    ap.add_argument('--subset-eval', default=None, dest='subset_eval')
    a = ap.parse_args(argv)
    w3s = [float(x) for x in a.w3.split(',') if x.strip()]
    assert w3s and all(0 < w < 1 for w in w3s), 'the w3 grid must be strictly inside (0, 1)'
    tags = ([t.strip() for t in a.tags.split(',')] if a.tags
            else [Path(p).stem for p in a.scores])
    assert len(tags) == len(a.scores), '--tags must have one label per --scores file'

    t0 = time.time()
    ctx = F.load_context()
    out = {'step': 'STEP267 reranker read through the STEP264 audit fusion harness',
           'registration': str(REG267), 'w3_grid': w3s,
           'harness': 'src/llm_judge/step264_fusion_check.py (imported unchanged)',
           'scope': {'queries': F.N_QUERIES, 'candidates': int(len(ctx['cand_rows'])),
                     'folds': [3, 4]},
           'M0': F.lvl(ctx['M0'], ctx['hv']), 'M1': F.lvl(ctx['M1'], ctx['hv']),
           'runs': {}}
    lines = []
    for p, tg in zip(a.scores, tags):
        r = run_file(ctx, p, w3s, tg)
        out['runs'][tg] = r
        lines.append(tg + '\n' + table(r, w3s))
        print('\n' + lines[-1], flush=True)
    if a.subset_eval:
        out['subset_eval'] = {'file': a.subset_eval, 'result': subset_eval(a.subset_eval)}
        print('\nsubset eval (score_heldout_eval.py machinery) on', a.subset_eval, flush=True)
        print(json.dumps(out['subset_eval']['result'].get('event_detection'), indent=1),
              flush=True)
    out['table'] = lines
    out['wall_seconds'] = round(time.time() - t0, 1)
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, 'w', encoding='utf-8'), indent=1)
    print('\nwrote', op, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
