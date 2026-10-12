"""STEP133 descriptive add-on (post-registration; NOT part of the registered reading): upper-half slopes (phi .60/.75/.90 + 1.0),
suggested in CLAUDE message 42 because log-scale learning curves are usually concave. Reads the frozen STEP133 outputs only and reuses
the registered joint resamples exactly (rng 1330, same draw order), verified by reproducing the registered quantiles."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
sys.path.insert(0, ROOT + 'Claude Version/src')
sys.path.insert(0, ROOT + 'Codex Resume/src')
import json, hashlib
import numpy as np
import audit_common as ac
OUT = ROOT + 'Claude Version/experiments/learning_curve_a8_20260913/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
prot0 = ac.check_protected()
RES = json.load(open(OUT + 'step133_results.json', encoding='utf-8'))
Z = np.load(OUT + 'step133_query_ap.npz')
d = ac.load_current()[0]
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
kt = type(next(iter(ptab)))
QORD = [kt(x) for x in Z['qord']]
TI = np.unique(np.array([ptab[q] for q in QORD]), return_inverse=True)[1]
NT = int(TI.max()) + 1
assert NT == RES['bootstrap']['tables']
PHIS, FIT, UP = (.30, .45, .60, .75, .90), (.45, .60, .75, .90), (.60, .75, .90)
RI = {(round(float(a), 2), int(b)): i for i, (a, b) in enumerate(Z['labels'])}
NMEAN = np.array([RES['full']['n_mean']] + [r['n_mean'] for r in RES['runs']])
ARMS, DBAR = RES['arms'], RES['doublings_weighted']
V = {kk: {a: Z[kk + '_' + a] for a in ARMS} for kk in ('raw', 'dec')}
SER = {kk: dict([('S', V[kk]['S'])] + [(a + '-S', V[kk][a] - V[kk]['S']) for a in ARMS if a != 'S']) for kk in V}


def slope(Y, w, sel, pts, with1=True):
    m = Y @ w / w.sum()
    xs = ([np.log2(NMEAN[0])] if with1 else []) + [np.log2(NMEAN[[RI[(x, int(s))] for s in sel[x]]].mean()) for x in pts]
    ys = ([m[0]] if with1 else []) + [m[[RI[(x, int(s))] for s in sel[x]]].mean() for x in pts]
    return float(np.polyfit(xs, ys, 1)[0]), float(m[0])


ALL5, ONE = {x: list(range(5)) for x in PHIS}, np.ones(len(QORD))
rng = np.random.default_rng(1330)
DRAWS = [(np.bincount(rng.integers(0, NT, NT), minlength=NT)[TI].astype(float), {x: rng.integers(0, 5, 5) for x in FIT}) for _ in range(2000)]
qq = lambda a_: [float(np.quantile(a_, .05)), float(np.quantile(a_, .95))]
OUTD, CHECK = {}, []
for kk in SER:
    OUTD[kk] = {}
    for nm, Y in SER[kk].items():
        dl, reg = nm != 'S', RES['curve'][kk][nm]
        s_reg, _ = slope(Y, ONE, ALL5, FIT)
        bj = np.array([slope(Y, w, sel, FIT) for w, sel in DRAWS])
        CHECK.append(abs(s_reg - reg['slope']) < 1e-12 and qq(bj[:, 0]) == reg['slope_q05_q95_joint'])
        su, d1 = slope(Y, ONE, ALL5, UP)
        bu = np.array([slope(Y, w, sel, UP) for w, sel in DRAWS])
        pu = d1 * dl + su * DBAR
        pq = qq(bu[:, 1] * dl + bu[:, 0] * DBAR)
        e = {'slope_registered': reg['slope'], 'slope_upper': su, 'slope_upper_q05_q95_joint': qq(bu[:, 0]), 'slope_low_point_phi30_45_60': slope(Y, ONE, ALL5, (.30, .45, .60), False)[0],
             'proj_registered': reg['proj_production'], 'proj_upper': pu, 'proj_upper_q05_q95': pq, 'composite_upper': .2 * pu}
        if dl and kk == 'dec':
            e['reading_rule_applied_to_upper_descriptive'] = 'UNDERSTATED' if (e['slope_upper_q05_q95_joint'][0] > 0 and pu >= .005) else ('NOT_UNDERSTATED' if pq[1] < .005 else 'INCONCLUSIVE')
        OUTD[kk][nm] = e
R = {'what': 'descriptive add-on, not registered; suggested by CLAUDE message 42', 'results_sha256': sha(OUT + 'step133_results.json'), 'query_ap_sha256': sha(OUT + 'step133_query_ap.npz'),
     'registered_resamples_reproduced': bool(all(CHECK)), 'curves': OUTD, 'protected_unchanged': bool(ac.check_protected() == prot0)}
json.dump(R, open(OUT + 'step133_upper_half_descriptive.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(R, indent=1))
