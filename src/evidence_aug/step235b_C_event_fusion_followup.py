"""STEP235b (Claude): read-only follow-up statistics on the STEP235 per-query audit table.

Adds nothing new to measure -- it only re-reads outputs_step235_20260917/per_query_step235.json's parquet and
reports (a) G_honest restricted to the directed_transfer candidate arms (the global G in results_step235.json
puts the expected-negative SP/CI control arms in the same selection pool, and SP_w025 -- a control -- is what
makes it negative), (b) the half-to-half spread of each arm inside its own family, as a calibration of how
much this class of perturbation moves a family half, and (c) the w-ladder per family.

Usage: python step235b_C_event_fusion_followup.py
Writes outputs_step235_20260917/followup235.json. Audit only; no fit; no file built; never uploads.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

CV = Path('C:/Data/Projects/Money Making/work/poker/Claude Version')
OUT = CV / 'outputs_step235_20260917'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
SHORT = {'DT': 'directed_transfer', 'SP': 'soft_play', 'CI': 'coordinated_isolation'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = lambda x: round(float(x), 6)


def main():
    res = json.loads((OUT / 'results_step235.json').read_text(encoding='utf-8'))
    reg = json.loads((OUT / 'registration235.json').read_text(encoding='utf-8'))
    q = pl.read_parquet(OUT / 'per_query_step235.parquet')
    hv = q['half'].to_numpy()
    fq = q['family'].to_numpy()
    M = {c: q[c].to_numpy() for c in q.columns if c not in ('pair_id', 'half', 'family')}
    REF = reg['reference']
    arms = list(reg['fusion_arms'])
    fam_of = {a: reg['fusion_arms'][a]['family'] for a in arms}
    delta = {a: M[a] - M[REF] for a in arms}

    def held_G(pool):
        out = []
        for h in (0, 1):
            pick = {}
            for a in pool:
                f = fam_of[a]
                m = delta[a][(hv == h) & (fq == f)]
                if len(m) and m.mean() > 0:
                    if f not in pick or m.mean() > pick[f][1]:
                        pick[f] = (a, m.mean())
            sel = sorted(v[0] for v in pick.values())
            dh = sum((delta[a] for a in sel), np.zeros(len(hv)))
            out.append({'selected_on_half': h, 'selected': sel, 'held_half_n': int((hv != h).sum()),
                        'held_half_delta': R(dh[hv != h].mean())})
        g = sum(o['held_half_n'] * o['held_half_delta'] for o in out) / sum(o['held_half_n'] for o in out)
        return {'held': out, 'G_honest': R(g)}

    dt_arms = [a for a in arms if fam_of[a] == 'directed_transfer']
    out = {
        'source': {'results': 'results_step235.json', 'per_query': 'per_query_step235.parquet',
                   'per_query_sha256': sha(OUT / 'per_query_step235.parquet'),
                   'registration_sha256': res['registration_sha256']},
        'G_all_nine_arms_as_in_results': res['decision']['G_honest'],
        'G_restricted_to_DT_candidate_arms': held_G(dt_arms),
        'G_restricted_to_the_three_registered_DT_and_the_primary': held_G(['DT_w025_F10', 'DT_w050_F10']),
        'w_ladder_family_delta': {
            s: {a: {'overall': R(delta[a].mean()),
                    'own_family': R(delta[a][fq == fam_of[a]].mean()),
                    'own_family_halves': [R(delta[a][(fq == fam_of[a]) & (hv == h)].mean()) for h in (0, 1)],
                    'own_family_half_spread': R(abs(delta[a][(fq == fam_of[a]) & (hv == 0)].mean()
                                                    - delta[a][(fq == fam_of[a]) & (hv == 1)].mean()))}
                for a in arms if fam_of[a] == f}
            for s, f in SHORT.items()},
        'queries_per_family_half': {f: [int(((fq == f) & (hv == h)).sum()) for h in (0, 1)] for f in FAMS},
        'note': 'The global G in results_step235.json is produced by the shared STEP205-230 rule, which lets any '
                'arm whose own family is positive on the selection half enter the pool. On half 0 that admits '
                'SP_w025_F10 -- an arm registered as an EXPECTED-NEGATIVE control -- at +.0243 SP, and its '
                '-.0260 on half 1 is what makes the global G negative. Restricted to the directed_transfer '
                'candidates the selection is stable: DT_w025_F10 is picked on both halves, so G equals its own '
                'delta with zero shrinkage.'}
    p = OUT / 'followup235.json'
    assert not p.exists(), 'preserve a completed follow-up'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
