"""HF14 stage 5: the conjunction test -- mechanism-extreme AND generally anomalous.

hf13 ran into a hard limit: to find ~70 pairs among 172,140 you need the far tail, and the far
tail cannot be calibrated from 1,488 non-targets.  The blind control proved it -- `chain`, which
hf12 showed stays at percentile .63 even at 4x planted strength, reported "significant excess"
at q99.9, which falsifies the test rather than establishing a mechanism.

This test needs no extreme-tail null, because it asks about COMPOSITION rather than count.

The logic.  Being extreme on a mechanism statistic is, on its own, uninformative: 172,140 pairs
produce a tail whatever the world is doing.  But a pair that is extreme BECAUSE it is
coordinating should also be odd in a completely different space -- the 33 partner-conditional
ca_ features -- and a pair that is extreme by chance has no reason to be.  Two facts make that
a fair test:

  * the ca_ Mahalanobis anomaly is UNSUPERVISED, so its power does not depend on the family
    having training labels (OBS 15.3: AUC .992/.998/.999 on the three disclosed families);
  * hf1 measured that even a completely UNTRAINED family keeps its rank -- median percentile
    .92-.96 -- while its supervised risk collapses to .02-.24.

So a genuine fourth family should show up as pairs that are extreme on the mechanism statistic,
high on the ca_ anomaly, and NOT already recognised by the supervised model.  The base rate of
that conjunction is directly measurable in the same shared-hands stratum, so the excess is an
estimate of the hidden population with an honest binomial interval.

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
ANOM_GATE = 0.95
RISK_GATE = 0.50
TOPK = [200, 1000]
RNG = np.random.default_rng(20260912)

VALIDATED = [('flow_z', +1, 0.44, 'M0 directed transfer (control)'),
             ('churn', +1, 0.76, 'H4 alternating churn'),
             ('churn_cancel', +1, 0.57, 'H4 alternating churn'),
             ('sv_conc', +1, 0.21, 'H2 shared victim'),
             ('shared_victim', +1, 0.17, 'H2 shared victim'),
             ('copresence', +1, 0.59, 'H5 selection'),
             ('n_both', +1, 0.69, 'H5 selection'),
             ('avoid_lr', -1, 0.03, 'H1 mutual avoidance'),
             ('chain', +1, 0.02, 'H3 laundering [BLIND CONTROL]')]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    fr = pl.read_parquet(OUT / 'hf11-dev-mechstats-labelled.parquet')
    elig = pl.read_parquet(OUT / 'hf3-dev-eligible.parquet').select(
        'player_1', 'player_2', 'risk_known', 'anom_pct').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('lo'),
        pl.max_horizontal('player_1', 'player_2').alias('hi')).drop('player_1', 'player_2')
    fr = fr.join(elig.unique(subset=['lo', 'hi']), on=['lo', 'hi'], how='inner')
    lab = fr['label'].to_numpy()
    fam = fr['behavior_family'].to_numpy()
    anom = fr['anom_pct'].to_numpy().astype(float)
    risk = fr['risk_known'].to_numpy().astype(float)
    nb = fr['n_both'].to_numpy().astype(float)
    unl, neg, pos = lab == -1, lab == 0, lab == 1
    print(f'eligible development pairs with a ca_ anomaly: {fr.height}')
    print(f'  {int(pos.sum())} labelled positives, {int(neg.sum())} confirmed non-targets, '
          f'{int(unl.sum())} unlabelled')

    # ---- A. what does real coordination look like in the conjunction space? -------
    print()
    print('=' * 100)
    print('A.  REFERENCE PROFILES -- the ca_ anomaly is the second, independent axis')
    print('=' * 100)
    print(f'{"set":<30}{"n":>7}{"med anom pct":>14}{"anom>=.95":>11}{"risk>=.5":>10}')
    ref = {}
    sets = [('labelled positives', pos), ('confirmed non-targets', neg),
            ('unlabelled', unl)] + [(f'  {f}', fam == f) for f in FAMS]
    for name, m in sets:
        r = {'n': int(m.sum()), 'median_anom': float(np.median(anom[m])),
             'frac_anom': float(np.mean(anom[m] >= ANOM_GATE)),
             'frac_risk': float(np.mean(risk[m] >= RISK_GATE))}
        ref[name.strip()] = r
        print(f'{name:<30}{r["n"]:>7}{r["median_anom"]:>14.4f}{r["frac_anom"]:>11.3f}'
              f'{r["frac_risk"]:>10.3f}')
    print()
    print('  Real coordination is overwhelmingly anomalous in ca_ space.  A hidden family is')
    print('  unlabelled, so its supervised risk collapses (hf1), but the ca_ anomaly is')
    print('  UNSUPERVISED and keeps working -- that asymmetry is what this test exploits.')

    # ---- B. the conjunction in each mechanism's tail ------------------------------
    edges = np.percentile(nb[unl], np.linspace(0, 100, 6))
    strata = np.clip(np.searchsorted(edges, nb, side='right') - 1, 0, 4)
    print()
    print('=' * 100)
    print('B.  CONJUNCTION -- of the pairs a mechanism puts in its tail, how many are also')
    print('    anomalous in ca_ space and NOT already recognised by the supervised model?')
    print('=' * 100)
    print(f'{"statistic":<15}{"hypothesis":<32}{"K":>6}{"obs":>6}{"exp":>7}'
          f'{"excess":>9}{"95% CI":>18}')
    rows = []
    for s, sgn, power, hyp in VALIDATED:
        v = fr[s].to_numpy().astype(float) * sgn
        for K in TOPK:
            # take the top K within the unlabelled set, stratified on shared hands so the
            # tail is not just the pairs with the longest history
            sel = np.zeros(fr.height, bool)
            for st in range(5):
                m = np.where(unl & (strata == st))[0]
                k = max(1, int(round(K * len(m) / max(int(unl.sum()), 1))))
                sel[m[np.argsort(-v[m], kind='mergesort')[:k]]] = True
            hit = sel & (anom >= ANOM_GATE) & (risk < RISK_GATE)
            obs = int(hit.sum())
            # base rate of the same conjunction among all unlabelled pairs, per stratum
            exp = 0.0
            for st in range(5):
                m = unl & (strata == st)
                base = float(np.mean((anom[m] >= ANOM_GATE) & (risk[m] < RISK_GATE)))
                exp += base * int((sel & (strata == st)).sum())
            lo, hi = wilson(obs, int(sel.sum()))
            n_sel = int(sel.sum())
            rows.append({'statistic': s, 'hypothesis': hyp, 'K': K, 'n_tail': n_sel,
                         'observed': obs, 'expected': exp, 'excess': obs - exp,
                         'ci_excess': [lo * n_sel - exp, hi * n_sel - exp],
                         'power_top174': power})
            print(f'{s:<15}{hyp:<32}{K:>6}{obs:>6}{exp:>7.1f}{obs - exp:>+9.1f}'
                  f'{f"[{lo * n_sel - exp:+.0f}, {hi * n_sel - exp:+.0f}]":>18}')

    # ---- C. turn the conjunction into a population bound --------------------------
    print()
    print('=' * 100)
    print('C.  WHAT A HIDDEN FAMILY OF N PAIRS WOULD HAVE PRODUCED HERE')
    print('=' * 100)
    p_anom = float(np.mean(anom[pos] >= ANOM_GATE))
    print(f'  a real coordinating pair clears the ca_ gate {p_anom:.1%} of the time (measured')
    print(f'  on the {int(pos.sum())} labelled positives), and hf12 gives the tail-recovery rate')
    print()
    print(f'{"hypothesis":<32}{"stat":<15}{"recovery":>9}{"x anom":>8}'
          f'{"N=35":>8}{"N=70":>8}{"N=140":>8}{"observed excess":>17}')
    bound = []
    for s, sgn, power, hyp in VALIDATED:
        r = [x for x in rows if x['statistic'] == s and x['K'] == 1000][0]
        pred = [power * p_anom * n for n in (35, 70, 140)]
        bound.append({'hypothesis': hyp, 'statistic': s, 'predicted': pred,
                      'observed_excess': r['excess'], 'ci': r['ci_excess']})
        print(f'{hyp:<32}{s:<15}{power:>9.2f}{p_anom:>8.2f}'
              + ''.join(f'{x:>8.1f}' for x in pred)
              + f'{r["excess"]:>+17.1f}')

    write_json(OUT / 'hf14-conjunction.json',
               {'reference': ref, 'conjunction': rows, 'bound': bound,
                'anom_gate': ANOM_GATE, 'risk_gate': RISK_GATE,
                'p_anom_given_positive': p_anom})
    print()
    print(f'written {OUT / "hf14-conjunction.json"}')


if __name__ == '__main__':
    main()
