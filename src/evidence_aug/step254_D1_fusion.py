"""STEP254 D1: late fusion of the MECHPLUS event probability into production stage-2 scores,
on the cached 162-query evidence audit - exactly the design of step235_C_event_fusion_audit.py.

No network and no stage-2 fit is re-run: STEP235 wrote its 41-view-averaged pre-decoder scores to
outputs_step235_20260917/audit_scores_step235.parquet (column v1_REF), and every STEP235 arm is a
pure post-processing of that vector. This runner reproduces REF = .727099 from it, then replaces
STEP235's event probability (STEP232, 163 raw columns) with MECHPLUS's and re-decodes.

    odds(p) = p / (1 - p);  p_evt = odds(p_family) * F / (1 + odds * F)
    p'      = p_stage2 ** (1 - w) * p_evt ** w        (rows of that family only)

Arms: per family, w in {.25, .5}, F = 10 (STEP235's F).
Inclusion rule (registered): decoded gain >= +.0034 overall AND positive on both table halves.
Audit only. Development phase asserted. Builds no submission. Zero fits.
"""
from __future__ import annotations

import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

from evidence_aug import step165_C_e2e_audit as S165  # noqa: E402

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
OUT = CV / 'experiments/step254_mechanical_tensor_20260917'
S235 = CV / 'outputs_step235_20260917'
K, FAMS, CI = S165.K, S165.FAMS, S165.CI
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}
F_SHIFT = 10.0
REF_EXPECTED = 0.727099


def prior_shift(p, F):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * float(F)
    return o / (1 + o)


def main() -> None:
    t0 = time.perf_counter()
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    D = S165.load()
    d, Q, QIX = D['d'], D['Q'], D['QIX']
    nq = len(Q)
    ph = (pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'phase').collect()
          .join(d.select('hand_id').unique(), on='hand_id'))
    assert ph['phase'].unique().to_list() == ['development'], 'audit rows must be development-phase'

    sc = pl.read_parquet(S235 / 'audit_scores_step235.parquet')
    assert sc.select(K).equals(d.select(K)), 'step235 score file does not align with the audit frame'
    REF = sc['v1_REF'].to_numpy().copy()

    oof = pl.read_parquet(OUT / 'hand_oof_step254.parquet')
    need = K + ['label', 'family'] + [f'mechplus_p_{SHORT[f]}' for f in FAMS]
    j = Q.select(K + ['behavior_family']).join(oof.select(need), on=K, how='left',
                                               validate='1:1', maintain_order='left')
    assert j.height == nq and j.select(K).equals(Q.select(K)), 'event join reordered the audit rows'
    nulls = {c: int(j[c].null_count()) for c in need[2:]}
    assert sum(nulls.values()) == 0, ('event coverage is not 100%', nulls)
    assert (j['label'] == 1).all(), 'an audit row is not a labelled positive pair'
    assert (j['behavior_family'] == j['family']).all(), 'family disagrees'
    PEVT = {f: j[f'mechplus_p_{SHORT[f]}'].to_numpy().astype(np.float64) for f in FAMS}
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}

    dec = D['dec']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptabq = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    key = d.select('pair_id', 'hand_id', 'table_id')
    qids = sorted(truths)
    hv = np.array([half(ptabq[q]) for q in qids])
    fq = np.array([fams[q] for q in qids])

    def mapof(s):
        _, mm = summarize(key.with_columns(pl.Series('score', dec(s))), truths, fams)
        return np.array([mm[q] for q in qids])

    M_REF = mapof(REF)
    ref_ok = abs(float(M_REF.mean()) - REF_EXPECTED) <= 0.002
    print('REF map5', round(float(M_REF.mean()), 6), 'expected', REF_EXPECTED, 'ok', ref_ok, flush=True)
    p_ref_top = REF[QIX] - 10.0

    res = {'n_queries': len(qids), 'ref_map5': round(float(M_REF.mean()), 6),
           'ref_reproduces': bool(ref_ok), 'F': F_SHIFT, 'arms': {}, 'fits_run': 0}
    included = []
    for f in FAMS:
        pe = prior_shift(PEVT[f], F_SHIFT)
        for w in (.25, .5):
            m = isf[f]
            newp = p_ref_top.copy()
            newp[m] = (np.clip(p_ref_top[m], 1e-9, 1.0) ** (1 - w)) * np.clip(pe[m], 1e-9, 1 - 1e-9) ** w
            s = REF.copy()
            s[QIX] = 10.0 + newp
            dl = mapof(s) - M_REF
            a = f'{SHORT[f]}_w{int(w * 100):03d}_F10'
            iso = {SHORT[g]: float(np.abs(dl[fq == g]).max()) for g in FAMS if g != f}
            row = {'family': f, 'w': w, 'rows_changed': int(m.sum()),
                   'decoded_delta': round(float(dl.mean()), 6),
                   'half0': round(float(dl[hv == 0].mean()), 6),
                   'half1': round(float(dl[hv == 1].mean()), 6),
                   'own_family': round(float(dl[fq == f].mean()), 6),
                   'own_family_halves': [round(float(dl[(fq == f) & (hv == h)].mean()), 6) for h in (0, 1)],
                   'queries_up_down': [int((dl > 0).sum()), int((dl < 0).sum())],
                   'query_se': round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 6),
                   'untouched_families_max_abs': iso,
                   'mean_p_evt_on_family': round(float(pe[m].mean()), 6),
                   'mean_p_stage2_on_family': round(float(p_ref_top[m].mean()), 6)}
            row['included'] = bool(row['decoded_delta'] >= .0034 and row['half0'] > 0 and row['half1'] > 0)
            if row['included']:
                included.append(a)
            res['arms'][a] = row
            print(a, {k: row[k] for k in ('decoded_delta', 'half0', 'half1', 'included')}, flush=True)
    res['included_arms'] = included
    res['D1'] = 'PASS' if included else 'FAIL'
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    json.dump(res, open(OUT / 'results254_D1.json', 'w', encoding='utf-8'), indent=1)
    print('D1', res['D1'], res['seconds'], 's')


if __name__ == '__main__':
    main()
