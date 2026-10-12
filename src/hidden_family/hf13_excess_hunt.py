"""HF13 stage 4: with the detection floors known, hunt for excess mass -- and say how much
would have been visible if it were there.

hf12 calibrated the harness against the real families (a planted directed_transfer of exactly
one family's worth lands at percentile .9987; the real family sits at .9966) and produced a
detection floor per mechanism.  That converts every null here into a quantified statement:
not "we found nothing" but "a population of N pairs using mechanism M would have put K of them
in this tail, and we see K_obs".

The null distribution is the host's own 1,488 CONFIRMED NON-TARGETS.  They are the only sample
in the data certified to contain no coordination, so the count of unlabelled pairs above their
q99 / q99.9 is a direct excess-mass estimate:

    expected = n_unlabelled * P(non-target > t)     observed = #{unlabelled > t}

with a Poisson interval on the difference.  Two contaminants are kept in view: the unlabelled
set contains UNLISTED disclosed-family positives (OBS 12 counted ~232 at risk >= .9), and the
non-targets are a host-selected sample, so section A checks the two distributions agree in the
BODY before the tails are compared.

Read-only.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '4'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import CV, write_json

OUT = CV / 'experiments' / 'hidden_family_20260912'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']

# statistic, direction, and the hf12 detection rate into the top 174 at strength 1.0
# (= one real directed_transfer's worth of planted effect)
VALIDATED = [('flow_z', +1, 0.44, 'M0 directed transfer (control)'),
             ('churn', +1, 0.76, 'H4 alternating churn'),
             ('churn_cancel', +1, 0.57, 'H4 alternating churn'),
             ('sv_conc', +1, 0.21, 'H2 shared victim'),
             ('shared_victim', +1, 0.17, 'H2 shared victim'),
             ('copresence', +1, 0.59, 'H5 selection'),
             # n_both is the stratifier itself, so it cannot be tested against a
             # shared-hands-matched null; hf12's power for it is kept for section C only.
             ('avoid_lr', -1, 0.03, 'H1 mutual avoidance'),
             ('chain', +1, 0.02, 'H3 laundering [BLIND CONTROL]')]
# how many hidden-family pairs development would hold at the prevalence OBS 15.2 settled on
N_HIDDEN_DEV = [35, 70, 140]


RNG = np.random.default_rng(20260912)
N_BOOT = 400


def stratified_excess(v, neg, unl, strata, q, n_boot=N_BOOT):
    """Observed minus expected tail count, matched on shared hands.

    Section A showed the host's non-targets carry systematically MORE history than the
    unlabelled population (median 108.5 shared hands against 95), and almost every statistic
    here grows with history, so an unstratified tail comparison measures that selection rather
    than any mechanism.  The threshold is therefore taken inside each shared-hands decile.

    The dominant uncertainty is not Poisson noise on the observed count -- it is the tail
    probability itself, estimated from only 1,488 non-targets.  A q99 threshold rests on ~15 of
    them and a q99.9 on ~1.5, so the interval comes from resampling the non-targets.
    """
    obs = exp = 0.0
    per = []
    for st in np.unique(strata):
        m = strata == st
        ng, un = neg & m, unl & m
        if ng.sum() < 30 or un.sum() < 30:
            continue
        t = np.percentile(v[ng], q)
        obs += float((v[un] > t).sum())
        exp += un.sum() * (1.0 - q / 100.0)     # exact, by the definition of the quantile
        per.append((v[ng].copy(), np.sort(v[un])))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        o = 0.0
        for gv, su in per:
            r = RNG.choice(gv, size=len(gv), replace=True)
            o += len(su) - np.searchsorted(su, np.percentile(r, q), side='right')
        boots[i] = o - exp                      # the THRESHOLD is what is uncertain, so the
    return obs, exp, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    fr = pl.read_parquet(OUT / 'hf11-dev-mechstats-labelled.parquet')
    elig = pl.read_parquet(OUT / 'hf3-dev-eligible.parquet').select(
        'player_1', 'player_2', 'risk_known', 'anom_pct').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('lo'),
        pl.max_horizontal('player_1', 'player_2').alias('hi')).drop('player_1', 'player_2')
    fr = fr.join(elig.unique(subset=['lo', 'hi']), on=['lo', 'hi'], how='left')
    lab = fr['label'].to_numpy()
    fam = fr['behavior_family'].to_numpy()
    unl = lab == -1
    neg = lab == 0
    risk = fr['risk_known'].to_numpy()
    n_u = int(unl.sum())
    print(f'{fr.height} pairs: {int(unl.sum())} unlabelled, {int(neg.sum())} confirmed '
          f'non-targets, {int((lab == 1).sum())} positives')
    print(f'supervised risk available for {int(np.isfinite(risk.astype(float)).sum())} pairs')

    # ---- A. is the non-target sample a fair null in the BODY? ---------------------
    print()
    print('=' * 100)
    print('A.  FAIRNESS OF THE NULL -- non-targets vs unlabelled at the body deciles')
    print('    (if the host picked non-targets non-representatively, the tail test is void)')
    print('=' * 100)
    print(f'{"statistic":<16}' + ''.join(f'{f"q{q}":>12}' for q in (25, 50, 75, 90)))
    fair = {}
    for s, _, _, _ in VALIDATED:
        v = fr[s].to_numpy().astype(float)
        row = []
        for q in (25, 50, 75, 90):
            a, b = np.percentile(v[neg], q), np.percentile(v[unl], q)
            row.append((a, b))
        fair[s] = [(float(a), float(b)) for a, b in row]
        print(f'{s:<16}' + ''.join(f'{a:>6.3f}/{b:<6.3f}' for a, b in row))
    print('  format: non-target / unlabelled.  Close agreement = the null is usable.')

    # ---- B. excess mass in the tails ---------------------------------------------
    print()
    print('=' * 100)
    print('B.  EXCESS MASS -- unlabelled pairs beyond the non-target tail quantiles')
    print('=' * 100)
    nb = fr['n_both'].to_numpy().astype(float)
    # five strata, not ten: the tail threshold has to be estimated INSIDE each stratum, and
    # 1,488 non-targets split ten ways leaves ~149 each -- too few for anything past q95.
    edges = np.percentile(nb[neg], np.linspace(0, 100, 6))
    strata = np.clip(np.searchsorted(edges, nb, side='right') - 1, 0, 4)
    print('    NOTE: `chain` is the blind control -- hf12 showed a planted laundering scheme')
    print('    at 4x strength stays at percentile .63, so any "excess" it reports is the test')
    print('    failing, not a mechanism.  It falsifies every quantile at which it fires.')
    print('    stratified on shared-hands quintiles of the non-target sample: '
          + ', '.join(f'{e:.0f}' for e in edges))
    print(f'{"statistic":<15}{"hypothesis":<26}{"q":>7}{"obs":>7}{"exp":>8}{"excess":>9}'
          f'{"95% CI":>20}{"unrecog":>9}{"sig":>6}')
    rows = []
    for s, sgn, power, hyp in VALIDATED:
        v = fr[s].to_numpy().astype(float) * sgn
        for q in (90.0, 95.0, 99.0, 99.9):
            obs, exp, lo, hi = stratified_excess(v, neg, unl, strata, q)
            sig = 'yes' if lo > 0 else ('  -' if hi > 0 else 'NEG')
            t = np.percentile(v[neg], q)
            sel = unl & (v > t)
            r = np.where(np.isfinite(risk[sel].astype(float)), risk[sel], 0.0).astype(float)
            unrecognised = int((r < 0.5).sum())
            rows.append({'statistic': s, 'hypothesis': hyp, 'q': q, 'observed': obs,
                         'expected': exp, 'excess': obs - exp, 'ci': [lo, hi],
                         'n_unrecognised': unrecognised, 'power_top174': power})
            print(f'{s:<15}{hyp:<26}{q:>7.1f}{obs:>7.0f}{exp:>8.1f}{obs - exp:>+9.1f}'
                  f'{f"[{lo:+.0f}, {hi:+.0f}]":>20}{unrecognised:>9}{sig:>6}')

    # ---- C. what would have been visible? ----------------------------------------
    print()
    print('=' * 100)
    print('C.  POWER -- pairs a hidden family of size N would have put in the top 174')
    print('    (hf12 detection rate at strength 1.0 = one real directed_transfer\'s worth)')
    print('=' * 100)
    print(f'{"hypothesis":<26}{"statistic":<15}{"rate":>7}' +
          ''.join(f'{f"N={n}":>9}' for n in N_HIDDEN_DEV) + f'{"observed top-174":>19}')
    power_rows = []
    for s, sgn, power, hyp in VALIDATED:
        v = fr[s].to_numpy().astype(float) * sgn
        t = np.sort(v[unl])[-174]
        obs174 = int((v[unl] >= t).sum())
        sel = unl & (v >= t)
        r = np.where(np.isfinite(risk[sel].astype(float)), risk[sel], 0.0).astype(float)
        unrec = int((r < 0.5).sum())
        pred = [power * n for n in N_HIDDEN_DEV]
        power_rows.append({'hypothesis': hyp, 'statistic': s, 'rate': power,
                           'predicted': pred, 'observed_top174': obs174,
                           'unrecognised_in_top174': unrec})
        print(f'{hyp:<26}{s:<15}{power:>7.2f}' + ''.join(f'{x:>9.1f}' for x in pred) +
              f'{f"{obs174} ({unrec} unrecognised)":>19}')

    write_json(OUT / 'hf13-excess-hunt.json',
               {'fairness': fair, 'excess': rows, 'power': power_rows,
                'n_unlabelled': n_u, 'n_hidden_dev_scenarios': N_HIDDEN_DEV})
    print()
    print(f'written {OUT / "hf13-excess-hunt.json"}')


if __name__ == '__main__':
    main()
