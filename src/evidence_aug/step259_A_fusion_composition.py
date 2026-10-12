"""STEP259-A: does the STEP254 MECHPLUS event-probability fusion COMPOSE with the current production stack (DTfp3)?

Zero fits. Same post-processing as step254_D1_fusion.py (p' = p_stage2^(1-w) * p_evt^w, F = 10), but applied on top of
the STEP251 audit score vector `v1_DTfp3` (outputs_step251_20260917/audit_scores_step251.parquet) instead of `v1_REF`.
SP/CI rows are identical in v1_REF and v1_DTfp3 (DTfp3 touches DT rows only; asserted), so the SP arm is re-measured
only as a check.  Registration: experiments/step259_mech_fusion_production_20260918/registration259_A.json (written first).
Rule (STEP251's stacking rule, unchanged): an arm joins the stack iff the JOINT is above the current stack (DTfp3)
overall and not below it on either table half.  If both DT weights qualify, take w = .25 (smaller departure).
Development phase asserted.  Builds no submission.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import json, sys
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
OUT = CV / 'experiments/step259_mech_fusion_production_20260918'
OOF = CV / 'experiments/step254_mechanical_tensor_20260917/hand_oof_step254.parquet'
S251 = CV / 'outputs_step251_20260917'
K, FAMS = S165.K, S165.FAMS
SHORT = {'directed_transfer': 'DT', 'soft_play': 'SP', 'coordinated_isolation': 'CI'}
F_SHIFT = 10.0


def prior_shift(p, F):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    o = q / (1 - q) * float(F)
    return o / (1 + o)


def main() -> None:
    assert (OUT / 'registration259_A.json').exists(), 'register first'
    from evidence_aug.step65_structural_evidence import summarize
    from evidence_aug.step5_evaluate import half
    from evidence_aug.paths import check_protected
    prot = check_protected()
    D = S165.load()
    d, Q, QIX = D['d'], D['Q'], D['QIX']
    ph = (pl.scan_parquet(DATA / 'hands.parquet').select('hand_id', 'phase').collect()
          .join(d.select('hand_id').unique(), on='hand_id'))
    assert ph['phase'].unique().to_list() == ['development']
    sc = pl.read_parquet(S251 / 'audit_scores_step251.parquet')
    assert sc.select(K).equals(d.select(K))
    REF = sc['v1_REF'].to_numpy().copy()
    BASE = sc['v1_DTfp3'].to_numpy().copy()
    oof = pl.read_parquet(OOF)
    arms_src = ['mechplus', 'mech', 'perm', 'mechplus_not4', 'plus']   # mechplus = registered; the others are diagnostics
    need = K + [f'{a}_p_{SHORT[f]}' for a in arms_src for f in FAMS]
    j = Q.select(K + ['behavior_family']).join(oof.select(need), on=K, how='left', validate='1:1', maintain_order='left')
    assert j.select(K).equals(Q.select(K)) and sum(j[c].null_count() for c in need[2:]) == 0
    isf = {f: (Q['behavior_family'] == f).to_numpy() for f in FAMS}
    fam_rows = {f: (d['behavior_family'] == f).to_numpy() for f in FAMS}
    for f in ('soft_play', 'coordinated_isolation'):
        assert np.array_equal(REF[fam_rows[f]], BASE[fam_rows[f]]), 'DTfp3 moved a non-DT row'
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

    M_REF, M_BASE = mapof(REF), mapof(BASE)
    res = {'ref_map5': round(float(M_REF.mean()), 6), 'base_DTfp3_map5': round(float(M_BASE.mean()), 6),
           'base_minus_ref': round(float((M_BASE - M_REF).mean()), 6), 'arms': {}}
    print('REF', res['ref_map5'], 'BASE(DTfp3)', res['base_DTfp3_map5'], flush=True)

    def fuse(vec, src, f, w):
        p_top = vec[QIX] - 10.0
        pe = prior_shift(j[f'{src}_p_{SHORT[f]}'].to_numpy().astype(np.float64), F_SHIFT)
        m = isf[f]
        newp = p_top.copy()
        newp[m] = (np.clip(p_top[m], 1e-9, 1.0) ** (1 - w)) * np.clip(pe[m], 1e-9, 1 - 1e-9) ** w
        s = vec.copy()
        s[QIX] = 10.0 + newp
        return s

    for src in arms_src:
        for f in FAMS:
            for w in (.25, .5):
                dl = mapof(fuse(BASE, src, f, w)) - M_BASE
                a = f'{src}:{SHORT[f]}_w{int(w * 100):03d}'
                row = {'on': 'v1_DTfp3', 'delta_vs_stack': round(float(dl.mean()), 6),
                       'half0': round(float(dl[hv == 0].mean()), 6), 'half1': round(float(dl[hv == 1].mean()), 6),
                       'own_family': round(float(dl[fq == f].mean()), 6),
                       'up_down': [int((dl > 0).sum()), int((dl < 0).sum())], 'registered': src == 'mechplus'}
                row['joins_stack'] = bool(row['delta_vs_stack'] > 0 and row['half0'] >= 0 and row['half1'] >= 0)
                res['arms'][a] = row
                print(a, row, flush=True)
    comp = BASE.copy()
    parts = []
    for f, ws in (('directed_transfer', (.25, .5)), ('soft_play', (.25,))):
        for w in ws:
            if res['arms'][f'mechplus:{SHORT[f]}_w{int(w * 100):03d}']['joins_stack']:
                comp = fuse(comp, 'mechplus', f, w)
                parts.append(f'{SHORT[f]}_w{int(w * 100):03d}')
                break
    dl = mapof(comp) - M_BASE
    res['composition'] = {'parts': parts, 'delta_vs_stack': round(float(dl.mean()), 6),
                          'half0': round(float(dl[hv == 0].mean()), 6), 'half1': round(float(dl[hv == 1].mean()), 6),
                          'by_family': {SHORT[f]: round(float(dl[fq == f].mean()), 6) for f in FAMS},
                          'n_by_family': {SHORT[f]: int((fq == f).sum()) for f in FAMS}}
    print('COMPOSITION', res['composition'], flush=True)
    assert check_protected() == prot
    json.dump(res, open(OUT / 'results259_A.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
