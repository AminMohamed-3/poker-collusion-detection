"""STEP264 - THE THIRD-FACTOR AUDIT FUSION CHECK for the LLM hand judge.

score_heldout_eval.py answers "does the LLM rank hands?".  This script answers the only question
that can justify a production step: "does the LLM move MAP@5 ON TOP OF the production evidence
fusion?"  It is a zero-fit check - nothing is trained, nothing is tuned; the w3 grid is REPORTED,
never selected.

SCOPE.  keys_heldout.parquet's AUDIT rows are the UNLISTED stage-1 top-20 candidates of the 57
audit queries whose table falls in folds 3-4, and the POS rows of those same 57 pairs are the
LISTED ones.  Together they are the COMPLETE top-20 candidate list of 57 of the 162 audit
queries (exactly 20 per query, asserted).  MAP@5 is therefore computable for those 57 queries
from the LLM scores alone, through the production decoder.

THE LADDER (per-query MAP@5 over the 57, pooled, by half(table_id), paired bootstrap over
queries, 2,000 draws, seed 264):

  M0        base v1_DTfp3 (STEP251 audit scores) - the unfused stack
  M1        M0 + MECHPLUS, p' = p^(1-w) * prior_shift(p_evt, F=10)^w, w = .25 on the
            directed_transfer rows and w = .25 on the soft_play rows, coordinated_isolation
            untouched.  This is the STEP259b production recipe and IT IS THE BAR.
  M2(w3)    M1 further fused with the LLM factor at w3, same DT/SP scope
            ("all families" variant, which also fuses the CI rows, reported SEPARATELY)
  M3(w3)    M0 + the LLM factor only, no MECHPLUS - reference, not a candidate

The only number that can justify a step is M2 - M1.

THE MECHPLUS FACTOR IS THE REGISTERED **REF** ARRAY: experiments/step254_mechanical_tensor_
20260917/oof_mechplus.npy, rows 1 (DT) and 2 (SP), joined on (pair_id, hand_id) through
hand_oof_step254.parquet.  It is byte-identical to step266's oof_REF.npy (asserted here) and it
is the arm whose composition reproduces the registered +.011147 on the 162 queries (STEP266 gate
G2).  The step266 arms A4/G4/M4/S3/SINGLE_* are NESTED 3-fold models built to measure the
fold-averaging penalty; they are a diagnostic of the averaging path, not a production-faithful
event probability, and STEP266's own report says the production models each see 4/5 of the
development data "like REF".  So REF is the correct audit bar for a third factor; using G4 would
set the bar about .0022 too low and flatter the LLM.

USAGE
  python step264_fusion_check.py --scores heldout_scores.jsonl [--out fusion_check264.json]
                                 [--w3 .1,.2,.3] [--label-tag RUN_NAME]
  python step264_fusion_check.py --validate      # ORACLE / NULL / MECHPLUS-as-LLM, one harness load

Rows of the score file are the ones score_heldout_eval.py reads:
{"uid","mean_conf","k"[,"confidences","verdict_counts","malformed"]}, mean_conf 50 when
malformed.  p_llm = clip(mean_conf/100, .01, .99).  A top-20 candidate with no row in the score
file gets a NEUTRAL p = .5 and is counted; the coverage share is reported.

Development-phase data only (folds 3-4 of the step232 table folds).  Reads only; never uploads.
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
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

from evidence_aug import step165_C_e2e_audit as S165                          # noqa: E402
from evidence_aug.step5_evaluate import half                                  # noqa: E402
from evidence_aug.step65_structural_evidence import summarize                 # noqa: E402

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/step264_llm_hand_judge_20260918'
SYN = EXP / 'synthetic'
KEYS = EXP / 'keys_heldout.parquet'
S251 = CV / 'outputs_step251_20260917/audit_scores_step251.parquet'
S254 = CV / 'experiments/step254_mechanical_tensor_20260917'
MECH_NPY = S254 / 'oof_mechplus.npy'
MECH_PARQUET = S254 / 'hand_oof_step254.parquet'
REF_CHECK = CV / 'experiments/step266_fold_average_calibration_20260918/oof_REF.npy'

K, FAMS = S165.K, S165.FAMS
DT, SP, CI = 'directed_transfer', 'soft_play', 'coordinated_isolation'
SHORT = {DT: 'DT', SP: 'SP', CI: 'CI'}
W_MECH = .25
F_SHIFT = 10.0
N_BOOT = 2000
BOOT_SEED = 264
N_QUERIES = 57
N_CANDIDATES = 20


# --------------------------------------------------------------------------- fusion primitives
def prior_shift(p, F=F_SHIFT):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * float(F)
    return o / (1 + o)


def fuse(vec, QIX, pvec, mask, w):
    """p' = p^(1-w) * prior_shift(pvec, F=10)^w on the masked top-20 rows.  Exactly the STEP259
    production form (step266_fusion_check.fuse) with an explicit row mask."""
    p_top = vec[QIX] - 10.0
    pe = prior_shift(pvec)
    newp = p_top.copy()
    newp[mask] = (np.clip(p_top[mask], 1e-9, 1.0) ** (1 - w)
                  * np.clip(pe[mask], 1e-9, 1 - 1e-9) ** w)
    s = vec.copy()
    s[QIX] = 10.0 + newp
    return s


# --------------------------------------------------------------------------- context
def load_context() -> dict:
    """The audit harness, the STEP251 base, the registered MECHPLUS REF factor and the 57-query
    restriction.  Takes about a minute (the harness load dominates)."""
    t0 = time.perf_counter()
    D = S165.load()
    d, Q, QIX, dec = D['d'], D['Q'], D['QIX'], D['dec']

    sc = pl.read_parquet(S251)
    assert sc.select(K).equals(d.select(K)), 'STEP251 audit scores are not in the harness row order'
    BASE = sc['v1_DTfp3'].to_numpy().astype(np.float64).copy()

    # ---- MECHPLUS: the REGISTERED REF array (see the module docstring)
    reg = np.load(MECH_NPY)
    assert reg.shape[0] == 4
    if REF_CHECK.exists():
        assert np.array_equal(reg, np.load(REF_CHECK)), 'step266 oof_REF.npy != oof_mechplus.npy'
    oof = pl.read_parquet(MECH_PARQUET, columns=K + ['mechplus_p_any', 'mechplus_p_DT',
                                                     'mechplus_p_SP', 'mechplus_p_CI'])
    for i, nm in enumerate(('any', 'DT', 'SP', 'CI')):                # parquet order == npy order
        assert np.allclose(oof['mechplus_p_' + nm].to_numpy(), reg[i], atol=1e-7), nm
    oof = oof.with_columns(pl.Series('REF_DT', reg[1]), pl.Series('REF_SP', reg[2]),
                           pl.Series('REF_CI', reg[3]))
    j = Q.select(K + ['behavior_family', 'relevance']).join(
        oof.select(K + ['REF_DT', 'REF_SP', 'REF_CI']), on=K, how='left', validate='1:1',
        maintain_order='left')
    assert j.select(K).equals(Q.select(K))
    assert sum(j[c].null_count() for c in ('REF_DT', 'REF_SP', 'REF_CI')) == 0, 'MECHPLUS gap'
    mech = {DT: j['REF_DT'].to_numpy().astype(np.float64),
            SP: j['REF_SP'].to_numpy().astype(np.float64),
            CI: j['REF_CI'].to_numpy().astype(np.float64)}

    # ---- the 57 audit queries of keys_heldout.parquet
    keys = pl.read_parquet(KEYS)
    apairs = sorted(set(keys.filter(pl.col('source_class') == 'AUDIT')['pair_id'].to_list()))
    assert len(apairs) == N_QUERIES, ('expected %d audit pairs, got %d' % (N_QUERIES, len(apairs)))
    qfam_all = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    assert set(apairs) <= set(qfam_all), 'an audit pair of keys_heldout is not an audit query'
    assert set(keys.filter(pl.col('pair_id').is_in(apairs))['fold'].to_list()) <= {3, 4}

    qpid = Q['pair_id'].to_numpy()
    in57 = np.isin(qpid, apairs)
    assert int(in57.sum()) == N_QUERIES * N_CANDIDATES, \
        ('top-20 candidates of the 57 queries: expected %d rows, got %d'
         % (N_QUERIES * N_CANDIDATES, int(in57.sum())))
    cnt = Q.filter(pl.col('pair_id').is_in(apairs)).group_by('pair_id').len()
    bad = cnt.filter(pl.col('len') != N_CANDIDATES)
    assert bad.height == 0, 'queries without exactly 20 candidates: ' + str(bad.rows())

    # every candidate must be a held-out prompt (structural: the package must cover the scope)
    cand = Q.filter(pl.col('pair_id').is_in(apairs)).select(K + ['behavior_family', 'relevance'])
    cj = cand.join(keys.select(K + ['uid', 'label', 'source_class']), on=K, how='left',
                   validate='1:1', maintain_order='left')
    assert cj['uid'].null_count() == 0, \
        ('%d top-20 candidates have no held-out uid' % cj['uid'].null_count())
    assert (cj['label'].to_numpy() == cj['relevance'].to_numpy()).all(), \
        'keys_heldout label disagrees with the audit relevance'
    assert set(cj['source_class'].to_list()) == {'AUDIT', 'POS'}, 'unexpected source_class in scope'
    cand_uid = cj['uid'].to_list()
    cand_rows = np.flatnonzero(in57)                       # positions inside the 3,240 Q rows
    assert len(set(cand_uid)) == len(cand_uid), 'duplicate uid among the candidates'

    # ---- masks over the 3,240 Q rows
    qfam = Q['behavior_family'].to_numpy()
    m = {f: in57 & (qfam == f) for f in (DT, SP, CI)}
    m_all162 = {f: (qfam == f) for f in (DT, SP, CI)}
    mech_scope = m[DT] | m[SP]

    # ---- MAP@5 machinery (identical to step266_fusion_check)
    truths = {k_[0]: set(g['hand_id']) for k_, g in
              d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    keyf = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    aset = set(apairs)
    sel = np.array([i for i, q in enumerate(qids) if q in aset])
    assert len(sel) == N_QUERIES
    qsel = [qids[i] for i in sel]
    hv = np.array([half(ptab[q]) for q in qsel])
    fq = np.array([fams[q] for q in qsel])

    def mapof(s):
        _, mm = summarize(keyf.with_columns(pl.Series('score', dec(s))), truths, fams)
        return np.array([mm[q] for q in qids])[sel]

    # ---- M0 and M1 (the bar).  M1 is built twice: restricted to the 57 queries, and over the
    #      full 162-query production scope, which must give the SAME per-query values on the 57.
    M0 = mapof(BASE)
    v = fuse(BASE, QIX, mech[DT], m[DT], W_MECH)
    M1vec = fuse(v, QIX, mech[SP], m[SP], W_MECH)
    M1 = mapof(M1vec)
    v162 = fuse(BASE, QIX, mech[DT], m_all162[DT], W_MECH)
    M1_162 = mapof(fuse(v162, QIX, mech[SP], m_all162[SP], W_MECH))
    assert np.allclose(M1, M1_162), \
        'restricting the fusion scope to the 57 queries changed their MAP - decoding is not ' \
        'pair-local, so the restriction claim would be wrong'

    rng = np.random.default_rng(BOOT_SEED)
    IDX = rng.integers(0, N_QUERIES, size=(N_BOOT, N_QUERIES))

    print('context loaded in %.1fs: %d queries, %d candidates, MECHPLUS = REF (oof_mechplus.npy)'
          % (time.perf_counter() - t0, N_QUERIES, len(cand_rows)), flush=True)
    return dict(d=d, Q=Q, QIX=QIX, BASE=BASE, mech=mech, mapof=mapof,
                m=m, mech_scope=mech_scope, in57=in57, qfam=qfam,
                cand_rows=cand_rows, cand_uid=cand_uid,
                cand_rel=cj['relevance'].to_numpy().astype(int),
                cand_fam=cj['behavior_family'].to_numpy(),
                apairs=apairs, qsel=qsel, hv=hv, fq=fq, M0=M0, M1=M1, M1vec=M1vec,
                M1_162=M1_162, IDX=IDX)


# --------------------------------------------------------------------------- reporting helpers
def lvl(v, hv):
    return {'map5': round(float(v.mean()), 6),
            'h0': round(float(v[hv == 0].mean()), 6),
            'h1': round(float(v[hv == 1].mean()), 6)}


def dstat(dl, hv, fq, IDX):
    b = dl[IDX].mean(1)
    return {'delta': round(float(dl.mean()), 6),
            'h0': round(float(dl[hv == 0].mean()), 6),
            'h1': round(float(dl[hv == 1].mean()), 6),
            'P_gt_0': round(float((b > 0).mean()), 4),
            'p05': round(float(np.quantile(b, .05)), 6),
            'p50': round(float(np.quantile(b, .50)), 6),
            'p95': round(float(np.quantile(b, .95)), 6),
            **{SHORT[f]: round(float(dl[fq == f].mean()), 6) for f in (DT, SP, CI)},
            'up_down': [int((dl > 0).sum()), int((dl < 0).sum())]}


# --------------------------------------------------------------------------- the score file
def read_scores(path):
    rows = []
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    conf, mal = {}, 0
    for r in rows:
        u = r['uid']
        assert u not in conf, 'duplicate uid in the score file: ' + u
        c = r.get('mean_conf')
        if r.get('malformed') or c is None:
            c = 50.0
            mal += 1
        conf[u] = float(c)
    return conf, mal, len(rows)


def llm_factor(ctx, conf):
    """Full-length (3,240) probability vector: the LLM factor on the candidates, .5 elsewhere."""
    p = np.full(len(ctx['Q']), .5, dtype=np.float64)
    hit = np.zeros(len(ctx['cand_uid']), dtype=bool)
    vals = np.full(len(ctx['cand_uid']), 50.0)
    for i, u in enumerate(ctx['cand_uid']):
        if u in conf:
            vals[i] = conf[u]
            hit[i] = True
    p[ctx['cand_rows']] = np.clip(vals / 100.0, .01, .99)
    p[ctx['cand_rows'][~hit]] = .5                        # neutral for an unscored candidate
    return p, hit, vals


# --------------------------------------------------------------------------- one run
def run_one(ctx, scores_path, w3s, tag):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score

    conf, mal, nrows = read_scores(scores_path)
    unknown = set(conf) - set(ctx['cand_uid'])
    p_llm, hit, vals = llm_factor(ctx, conf)
    cov = float(hit.mean())
    if cov < 1.0:
        print('WARNING: %d of %d top-20 candidates have no score; neutral p = .5 applied'
              % (int((~hit).sum()), len(hit)), flush=True)

    QIX, hv, fq, IDX = ctx['QIX'], ctx['hv'], ctx['fq'], ctx['IDX']
    M0, M1, M1vec = ctx['M0'], ctx['M1'], ctx['M1vec']
    scopes = {'dt_sp': ctx['mech_scope'], 'all_families': ctx['in57']}

    res = {'tag': tag, 'scores_file': str(scores_path), 'score_rows': nrows,
           'malformed_rows': mal, 'unknown_uids_ignored': len(unknown),
           'mechplus_array': 'REF = experiments/step254_mechanical_tensor_20260917/'
                             'oof_mechplus.npy rows 1 (DT) / 2 (SP)',
           'scope': {'queries': N_QUERIES, 'candidates': int(len(ctx['cand_rows'])),
                     'folds': [3, 4],
                     'families': {SHORT[f]: int((np.array(ctx['fq']) == f).sum())
                                  for f in (DT, SP, CI)},
                     'listed_candidates': int(ctx['cand_rel'].sum())},
           'coverage_share': round(cov, 6),
           'n_candidates_unscored_neutral': int((~hit).sum()),
           'llm_conf': {'mean': round(float(vals[hit].mean()), 3) if hit.any() else None,
                        'min': round(float(vals[hit].min()), 3) if hit.any() else None,
                        'max': round(float(vals[hit].max()), 3) if hit.any() else None,
                        'share_clipped_low': round(float((vals[hit] < 1.0).mean()), 4)
                        if hit.any() else None,
                        'share_clipped_high': round(float((vals[hit] > 99.0).mean()), 4)
                        if hit.any() else None},
           'w_mechplus': W_MECH, 'F_prior_shift': F_SHIFT, 'w3_grid': list(w3s),
           'bootstrap': {'draws': N_BOOT, 'seed': BOOT_SEED, 'unit': 'query'}}

    res['M0_base_v1_DTfp3'] = lvl(M0, hv)
    res['M1_base_plus_mechplus'] = lvl(M1, hv)
    res['M1_minus_M0'] = dstat(M1 - M0, hv, fq, IDX)
    res['M1_minus_M0_full162_scope_restricted_to_57'] = dstat(ctx['M1_162'] - M0, hv, fq, IDX)

    res['M2_minus_M1'], res['M2_level'] = {}, {}
    res['M3_minus_M0'] = {}
    for sname, smask in scopes.items():
        for w3 in w3s:
            k = '%s@w3=%g' % (sname, w3)
            M2 = ctx['mapof'](fuse(M1vec, QIX, p_llm, smask, w3))
            res['M2_level'][k] = lvl(M2, hv)
            res['M2_minus_M1'][k] = dstat(M2 - M1, hv, fq, IDX)
            M3 = ctx['mapof'](fuse(ctx['BASE'], QIX, p_llm, smask, w3))
            res['M3_minus_M0'][k] = dstat(M3 - M0, hv, fq, IDX)

    # ---- the LLM factor on its own: listed vs unlisted over the 57 x 20 candidates
    y = ctx['cand_rel']
    s = p_llm[ctx['cand_rows']]
    own = np.array([ctx['mech'][f][r] for f, r in zip(ctx['cand_fam'], ctx['cand_rows'])])
    ap = {'n': int(len(y)), 'n_listed': int(y.sum()),
          'llm_ap': round(float(average_precision_score(y, s)), 5),
          'mechplus_own_family_ap': round(float(average_precision_score(y, own)), 5),
          'base_v1_ap': round(float(average_precision_score(
              y, ctx['BASE'][QIX][ctx['cand_rows']])), 5),
          'spearman_llm_vs_mechplus_own_family': round(float(spearmanr(s, own).statistic), 5)}
    fmask = np.isin(ctx['cand_fam'], [DT, SP])
    ap['spearman_on_the_DT_SP_fusion_scope'] = round(
        float(spearmanr(s[fmask], own[fmask]).statistic), 5)
    for f in (DT, SP, CI):
        mm = ctx['cand_fam'] == f
        if y[mm].sum() and (1 - y[mm]).sum():
            ap['by_family_' + SHORT[f]] = {
                'n': int(mm.sum()),
                'llm_ap': round(float(average_precision_score(y[mm], s[mm])), 5),
                'mechplus_ap': round(float(average_precision_score(y[mm], own[mm])), 5),
                'spearman': round(float(spearmanr(s[mm], own[mm]).statistic), 5)}
    res['llm_standalone'] = ap
    return res


def table(res, w3s):
    """A compact human-readable block for one run."""
    L = []
    L.append('  M0 %.5f   M1 %.5f   (M1-M0 %+.5f  P>0 %.3f  halves %+.5f / %+.5f)'
             % (res['M0_base_v1_DTfp3']['map5'], res['M1_base_plus_mechplus']['map5'],
                res['M1_minus_M0']['delta'], res['M1_minus_M0']['P_gt_0'],
                res['M1_minus_M0']['h0'], res['M1_minus_M0']['h1']))
    L.append('  scope        w3      M2        M2-M1     P(>0)   h0         h1')
    for sname in ('dt_sp', 'all_families'):
        for w3 in w3s:
            k = '%s@w3=%g' % (sname, w3)
            a, b = res['M2_level'][k], res['M2_minus_M1'][k]
            L.append('  %-12s %-6g  %.5f  %+.5f   %.3f   %+.5f  %+.5f'
                     % (sname, w3, a['map5'], b['delta'], b['P_gt_0'], b['h0'], b['h1']))
    L.append('  LLM alone: AP %.4f (mechplus own-family AP %.4f, base v1 AP %.4f), '
             'spearman vs mechplus %.3f, coverage %.4f'
             % (res['llm_standalone']['llm_ap'], res['llm_standalone']['mechplus_own_family_ap'],
                res['llm_standalone']['base_v1_ap'],
                res['llm_standalone']['spearman_llm_vs_mechplus_own_family'],
                res['coverage_share']))
    return '\n'.join(L)


# --------------------------------------------------------------------------- synthetic files
def write_synthetic(ctx):
    """ORACLE / NULL / MECHPLUS-as-LLM score files.  THEY CARRY LABELS - never ship them."""
    SYN.mkdir(parents=True, exist_ok=True)
    keys = pl.read_parquet(KEYS)
    uids, labels = keys['uid'].to_list(), keys['label'].to_numpy().astype(int)
    out = {}

    def dump(name, pairs):
        p = SYN / name
        with open(p, 'w', encoding='utf-8', newline='\n') as fh:
            for u, c in pairs:
                fh.write(json.dumps({'uid': u, 'mean_conf': round(float(c), 4), 'k': 4}) + '\n')
        return p

    out['ORACLE'] = dump('oracle_scores.jsonl', zip(uids, np.where(labels == 1, 90.0, 10.0)))
    rng = np.random.default_rng(BOOT_SEED)
    out['NULL'] = dump('null_scores.jsonl', zip(uids, 100.0 * rng.random(len(uids))))
    own = np.array([ctx['mech'][f][r] for f, r in zip(ctx['cand_fam'], ctx['cand_rows'])])
    out['MECHPLUS_AS_LLM'] = dump('mechplus_as_llm_scores.jsonl',
                                  zip(ctx['cand_uid'], 100.0 * own))
    # a second oracle at the widest confidence range the contract allows: the ceiling of this
    # instrument is set by the SPREAD of mean_conf, not by the judge being right
    out['ORACLE_WIDE'] = dump('oracle_wide_scores.jsonl',
                              zip(uids, np.where(labels == 1, 99.0, 1.0)))
    (SYN / 'README.txt').write_text(
        'Synthetic STEP264 score files written by src/llm_judge/step264_fusion_check.py --validate.\n'
        'oracle_scores.jsonl (90/10) and oracle_wide_scores.jsonl (99/1) CARRY THE LABELS;\n'
        'null_scores.jsonl is Uniform(0,100) with seed 264; all three cover the 6,977 held-out\n'
        'uids.  mechplus_as_llm_scores.jsonl covers only the 1,140 top-20 candidates of the 57\n'
        'audit queries and carries the registered MECHPLUS REF own-family probability x 100.\n'
        'THESE FILES MUST NEVER BE COPIED TO A GPU BOX OR ANYWHERE ELSE.  They exist only to\n'
        'validate the fusion check locally.\n', encoding='utf-8')
    return out


# --------------------------------------------------------------------------- wiring identity
def identity_check(ctx, w3s):
    """THE EXACT WIRING TEST.  Feeding the UNCLIPPED MECHPLUS probability back in as the third
    factor must be algebraically identical to the plain MECHPLUS fusion at a higher weight:

        (p^(1-w) * S^w)^(1-w3) * S^w3 = p^((1-w)(1-w3)) * S^(w(1-w3)+w3)

    and with w = .25 the two exponents are 1 - w_eff and w_eff for w_eff = .25 + .75*w3.  So the
    per-query MAP@5 of M2 must equal that of the direct fusion at w_eff to the last bit.  Unlike
    the MECHPLUS-as-LLM score-file run (which goes through the mandated [.01,.99] clip) this is a
    proof, not an approximation.
    """
    QIX = ctx['QIX']
    p_own = np.where(ctx['m'][DT], ctx['mech'][DT],
                     np.where(ctx['m'][SP], ctx['mech'][SP], .5))
    out = {}
    for w3 in w3s:
        M2 = ctx['mapof'](fuse(ctx['M1vec'], QIX, p_own, ctx['mech_scope'], w3))
        w_eff = W_MECH + (1 - W_MECH) * w3
        v = fuse(ctx['BASE'], QIX, ctx['mech'][DT], ctx['m'][DT], w_eff)
        direct = ctx['mapof'](fuse(v, QIX, ctx['mech'][SP], ctx['m'][SP], w_eff))
        md = float(np.abs(M2 - direct).max())
        assert md == 0.0, ('the fusion is not the production form: composing MECHPLUS at w3=%g '
                           'differs from the direct fusion at w_eff=%g by %g' % (w3, w_eff, md))
        out['w3=%g' % w3] = {'w_eff': round(w_eff, 6), 'max_abs_map_diff': md,
                             'delta_vs_M1': round(float((M2 - ctx['M1']).mean()), 6)}
    return out


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default=None)
    ap.add_argument('--out', default=str(EXP / 'fusion_check264.json'))
    ap.add_argument('--w3', default='.1,.2,.3')
    ap.add_argument('--label-tag', default=None, dest='label_tag')
    ap.add_argument('--validate', action='store_true')
    a = ap.parse_args(argv)
    w3s = [float(x) for x in a.w3.split(',') if x.strip()]
    assert w3s and all(0 < w < 1 for w in w3s), 'the w3 grid must be strictly inside (0, 1)'
    assert a.scores or a.validate, 'pass --scores heldout_scores.jsonl (or --validate)'

    ctx = load_context()

    if a.validate:
        files = write_synthetic(ctx)
        val = {'w3_grid': w3s, 'n_queries': N_QUERIES,
               'mechplus_array': 'REF (oof_mechplus.npy) - the registered production-equivalent OOF',
               'wiring_identity': identity_check(ctx, w3s), 'runs': {}}
        print('WIRING IDENTITY PASS (unclipped MECHPLUS as the third factor == direct fusion at '
              'w_eff, max abs MAP diff 0.0 at every w3)', flush=True)
        lines = []
        for tag, p in files.items():
            r = run_one(ctx, p, w3s, tag)
            val['runs'][tag] = r
            lines.append(tag + '\n' + table(r, w3s))
            print('\n' + lines[-1], flush=True)
        # how high can a perfect judge push MAP through this instrument?
        val['oracle_ceiling_probe'] = {
            k: v for k, v in run_one(ctx, files['ORACLE'], [.5, .7, .9],
                                     'ORACLE_ceiling')['M2_minus_M1'].items()}
        val['expectations'] = {
            'wiring_identity': 'exact: max_abs_map_diff must be 0.0',
            'ORACLE': 'M2 - M1 positive at every w3 and clearly above NULL; the SIZE is bounded '
                      'by the confidence spread, not by the judge being right (see '
                      'oracle_ceiling_probe and ORACLE_WIDE)',
            'NULL': 'M2 - M1 must be NEGATIVE - an uninformative third factor can only add noise '
                    'to a good ranking.  Its magnitude is the break-even cost a real judge has '
                    'to overcome at that w3',
            'MECHPLUS_AS_LLM': 'M2 at w3 is the MECHPLUS fusion at w_eff = .25 + .75*w3, so it '
                               'traces the known w-curve rather than sitting exactly on M1',
            'ORACLE_WIDE': 'the same oracle with mean_conf 99/1 instead of 90/10 - the gap to '
                           'ORACLE is pure dynamic range'}
        val['table'] = lines
        o = EXP / 'fusion_check_validation.json'
        json.dump(val, open(o, 'w', encoding='utf-8'), indent=1)
        print('\nwrote', o, flush=True)
        top = 'dt_sp@w3=%g' % w3s[-1]
        orc = val['runs']['ORACLE']['M2_minus_M1'][top]
        nul = val['runs']['NULL']['M2_minus_M1'][top]
        assert orc['delta'] > .015 and orc['P_gt_0'] > .95, \
            'ORACLE did not lift MAP - the wiring is wrong (%s)' % json.dumps(orc)
        for w3 in w3s:
            k = 'dt_sp@w3=%g' % w3
            assert val['runs']['NULL']['M2_minus_M1'][k]['delta'] < 0, (
                'a Uniform(0,100) third factor IMPROVED MAP at w3=%g - the check is not '
                'measuring what it claims' % w3)
            assert (val['runs']['ORACLE']['M2_minus_M1'][k]['delta']
                    - val['runs']['NULL']['M2_minus_M1'][k]['delta']) > .01, \
                'ORACLE and NULL are not separated at w3=%g' % w3
        print('VALIDATION OK: oracle %+.5f (P>0 %.3f), null %+.5f at w3=%g; wiring identity exact'
              % (orc['delta'], orc['P_gt_0'], nul['delta'], w3s[-1]))
        return

    tag = a.label_tag or Path(a.scores).stem
    r = run_one(ctx, a.scores, w3s, tag)
    json.dump(r, open(a.out, 'w', encoding='utf-8'), indent=1)
    print('\n' + tag + '\n' + table(r, w3s), flush=True)
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()
