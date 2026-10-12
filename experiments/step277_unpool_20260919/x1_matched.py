"""X1: a NON-CIRCULAR instrument for the block blend, calibrated against the three public reads.

Why the old one failed (19.229, now attributable after the STEP272 read):
    STEP270 new measurement added linearly   internal +.00090 -> public +.00046   factor +0.51
    STEP272 re-weight an existing member     internal +.00104 -> public +.00020   factor +0.19
    STEP273-275 interactions on risk views   internal +.00316 -> public -.00101   factor -0.32
The ordering is monotone in how much the change leans on the internal label structure. The testbed's
def A / def C negatives are threshold-derived: joined to development_labels.csv, `pos` is 146 REAL
confirmed_target rows, `neg` is 3 real confirmed_non_target rows, and FP/mid/undisclosed/ambiguous (555
rows) are NOT ANNOTATED AT ALL -- they are risk_full cutoffs. So def C was never a circularity guard.

The fix: keep the 146 REAL positives, and compare them only against controls MATCHED ON risk_full. Within a
risk_full stratum risk_full is ~constant, so it cannot drive the separation, and any residual discrimination
is information risk_full does not already carry -- which is exactly the property that transferred at +0.51.

    defM = sum_s w_s * AP(pos, score | stratum s) / sum_s w_s ,  w_s = #pos in s,  s = risk_full quantile bin

CALIBRATION GATE (fixed before looking at any new column): defM must independently reproduce the SIGNS of
all three public reads -- KEY positive, MLP2 positive, the STEP275 interaction block negative. If it fails
that, it is not used to screen anything and this file reports only the failure.
"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '4')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
SC = ('C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/'
      'e91ed0b3-2ad5-45d4-b30c-115132b295b0/scratchpad/pair/')
sys.path.insert(0, C + 'src'); sys.path.insert(0, SC)
import numpy as np, polars as pl
from pu_pairs.step60_multiview_blend import ap, lg
from pu_pairs.step63_softR1_candidate import X_of
import harness as H

F10 = ['SXN3_SHORT3', 'SXN3_ALL5', 'SXN3_LIN30', 'SXN3_LIN300', 'SXN3_FAMMAX',
       'ARCH_XGB3', 'ARCH_ET', 'ARCH_MLP', 'ARCH2_MLP2', 'ARCH2_HGB5']
NBIN = int(sys.argv[1]) if len(sys.argv) > 1 else 5
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
t0 = time.time(); rec = {'nbin': NBIN}

S = pl.concat([pl.read_parquet(C + 'experiments/pu_pairs_20260911/fold%d/scores.parquet' % f,
                               columns=['pair_id', 'shared_hands']) for f in (1, 2)]
              ).unique(subset=['pair_id'], keep='first')
V = pl.read_parquet(SC + 'V_arch3.parquet').filter(pl.col('surface') == 'prefix60').drop('surface')
d = H.load().join(V, on=['pair_id', 'fold'], how='left').join(
    S.select('pair_id', pl.col('shared_hands').alias('sh_full')), on='pair_id', how='left', validate='1:1')
in1, in2, insh = H.groups(d)
n = d.height
X0 = X_of(d, H.FS_M3, 'risk_prefix', 'R1_prefix', 'risk_noT')
KEY = d.select(F10).to_numpy().mean(1)
M2 = d['ARCH2_MLP2'].to_numpy()
noT = lg(np.clip(np.nan_to_num(d['risk_noT'].to_numpy()), 1e-6, 1 - 1e-6))
dom = d['dominated'].cast(pl.Float64).fill_null(0).to_numpy()
sh = np.log(d['shared_hands'].to_numpy().astype(float))
rv = lg(d['risk_prefix'].to_numpy())

ypos = (d['cls'] == 'pos').to_numpy().astype(float)          # 146 REAL confirmed_target rows
rf = d['risk_full'].to_numpy()
ELIG = d['sh_full'].to_numpy() >= 57
CERT = np.ones(n, bool)                                       # fit on every row, as production does
print('rows %d  real positives %d  eligible %d' % (n, int(ypos.sum()), int(ELIG.sum())), flush=True)

qs = np.quantile(rf[ELIG], np.linspace(0, 1, NBIN + 1))
qs[0] -= 1e-9; qs[-1] += 1e-9
binid = np.digitize(rf, qs) - 1
binid = np.clip(binid, 0, NBIN - 1)
for b in range(NBIN):
    m = ELIG & (binid == b)
    print('  bin %d: %4d rows, %3d real pos, risk_full %.4f..%.4f'
          % (b, m.sum(), int(ypos[m].sum()), rf[m].min(), rf[m].max()), flush=True)
rec['bins'] = [{'rows': int((ELIG & (binid == b)).sum()), 'pos': int(ypos[ELIG & (binid == b)].sum())}
               for b in range(NBIN)]


def defM(pred, yy=None):
    """risk_full-stratified AP of the REAL positives; weights = positives per stratum."""
    yy = ypos if yy is None else yy
    num = den = 0.0
    for b in range(NBIN):
        m = ELIG & (binid == b)
        k = yy[m].sum()
        if k < 2 or k == m.sum():
            continue
        num += k * ap(yy[m], pred[m]); den += k
    return float(num / den) if den else float('nan')


def defA_elig(pred):
    from pu_pairs.step62_multiview_candidate import DEFS
    cert = d['cls'].is_in(DEFS['A'][0] + DEFS['A'][1]).to_numpy()
    y = d['cls'].is_in(DEFS['A'][0]).to_numpy().astype(float)
    s = cert & ELIG
    return float(ap(y[s], pred[s]))


def fit(cols, l2, target=None):
    Xm = np.column_stack(cols)
    return H.crossfold(Xm, ypos if target is None else target, CERT, in1, in2, insh, l2)


# ---------------------------------------------------------------- the four production states
ST = {
    'STEP270 (13 in, l2 10)':  ([X0, KEY], 10.0),
    'STEP272 (14 in, l2 10)':  ([X0, KEY, M2], 10.0),
    'STEP275 (19 in, l2 20)':  ([X0, KEY, M2, noT, dom * noT, noT * rv, sh * M2, dom * noT * rv], 20.0),
}
P = {k: fit(c, l) for k, (c, l) in ST.items()}
P['STEP269 (12 in, l2 10)'] = fit([X0], 10.0)
print('\nCALIBRATION -- defM must match the SIGNS of the public reads:\n', flush=True)
print('%-26s %9s %9s %11s %11s' % ('state', 'defM', 'defA', 'd_defM', 'public'), flush=True)
PUB = {'STEP270 (13 in, l2 10)': ('STEP269 (12 in, l2 10)', +.00046),
       'STEP272 (14 in, l2 10)': ('STEP270 (13 in, l2 10)', +.00020),
       'STEP275 (19 in, l2 20)': ('STEP272 (14 in, l2 10)', -.00101)}
cal = {}
for k in ('STEP269 (12 in, l2 10)', 'STEP270 (13 in, l2 10)', 'STEP272 (14 in, l2 10)', 'STEP275 (19 in, l2 20)'):
    m, a = defM(P[k]), defA_elig(P[k])
    if k in PUB:
        ref, pub = PUB[k]
        dm = m - defM(P[ref])
        cal[k] = {'defM': m, 'defA': a, 'd_defM': dm, 'public': pub, 'sign_ok': bool((dm > 0) == (pub > 0))}
        print('%-26s %9.5f %9.5f %+11.5f %+11.5f  %s'
              % (k, m, a, dm, pub, 'OK' if cal[k]['sign_ok'] else 'WRONG SIGN'), flush=True)
    else:
        print('%-26s %9.5f %9.5f %11s %11s' % (k, m, a, '-', '-'), flush=True)
rec['calibration'] = cal
GATE = all(v['sign_ok'] for v in cal.values())
print('\nCALIBRATION GATE: %s' % ('PASS - defM reproduces all three public signs'
                                  if GATE else 'FAIL - defM is not trustworthy, no screening done'), flush=True)
rec['gate'] = bool(GATE)

if GATE:
    # ------------------------------------------------------------ screen the unused columns, LINEARLY
    base = [X0, KEY, M2]
    B = defM(P['STEP272 (14 in, l2 10)'])
    CAND = {}
    for c in ('rand30_mean', 'rand30_min', 'rand30_max', 'rand50_mean', 'rand50_min', 'rand50_max',
              'drop30_min', 'drop50_min', 'drop30_mean', 'drop50_mean',
              'dom_risk', 'share_hands_with_T', 'n_noT', 'risk_rescored', 'n_hands'):
        if c not in d.columns:
            continue
        v = np.nan_to_num(d[c].cast(pl.Float64).fill_null(0).to_numpy().astype(float))
        if c in ('n_noT', 'n_hands', 'share_hands_with_T'):
            CAND[c] = np.log1p(np.clip(v, 0, None))
        elif v.min() >= 0 and v.max() <= 1:
            CAND[c] = lg(np.clip(v, 1e-6, 1 - 1e-6))
        else:
            CAND[c] = v
    print('\nLINEAR single-column screen against STEP272 (defM base %.5f):\n' % B, flush=True)
    print('%-22s %10s %10s %10s' % ('column', 'd_defM', 'd_defA', 'corr_rf'), flush=True)
    arm = {}
    aB = defA_elig(P['STEP272 (14 in, l2 10)'])
    for c, v in CAND.items():
        p = fit(base + [v], 10.0)
        arm[c] = {'d_defM': defM(p) - B, 'd_defA': defA_elig(p) - aB,
                  'corr_rf': float(abs(np.corrcoef(v[ELIG], rf[ELIG])[0, 1]))}
        print('%-22s %+10.5f %+10.5f %10.3f'
              % (c, arm[c]['d_defM'], arm[c]['d_defA'], arm[c]['corr_rf']), flush=True)
    rec['screen'] = arm
    order = sorted(arm, key=lambda z: -arm[z]['d_defM'])
    print('\nbest by defM: %s' % order[:4], flush=True)

    rng = np.random.default_rng(31337)
    nb = []
    for r in range(NREP):
        pm = rng.permutation(n)
        nb.append(max(defM(fit(base + [v[pm]], 10.0)) - B for v in CAND.values()))
        if (r + 1) % 25 == 0:
            print('  null %d/%d running max %+.5f (%.0f s)' % (r + 1, NREP, max(nb), time.time() - t0), flush=True)
    nb = np.array(nb)
    rec['null'] = {'mean': float(nb.mean()), 'q95': float(np.quantile(nb, .95)), 'max': float(nb.max()), 'nrep': NREP}
    print('\nmax-over-family null (%d reps): mean %+.5f q95 %+.5f max %+.5f'
          % (NREP, nb.mean(), np.quantile(nb, .95), nb.max()), flush=True)
    for c in order[:5]:
        rec['screen'][c]['familywise_P'] = float((nb >= arm[c]['d_defM']).mean())
        print('   %-22s d_defM %+.5f  family-wise P = %.3f'
              % (c, arm[c]['d_defM'], rec['screen'][c]['familywise_P']), flush=True)
    keep = [c for c in order if arm[c]['d_defM'] > np.quantile(nb, .95)]
    rec['keep'] = keep
    print('\nabove the family null q95: %s' % keep, flush=True)

rec['seconds'] = round(time.time() - t0, 1)
json.dump(rec, open(SC + 'x1_matched.json', 'w'), indent=1)
print('\ndone %.0f s' % rec['seconds'], flush=True)
