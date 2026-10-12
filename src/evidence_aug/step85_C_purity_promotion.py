"""Step 85 (C): the purity promotion rule, measured against the REAL headroom, on three surfaces.

author: claude (agent)

Why this exists.  Step 84 found that the raw PU-AP deficit (1 - .7147 = .2853) is almost entirely
89 unlabelled rows at risk >= .9 which OBS 12 / 12.1 showed are the positive population, not errors
("Separability positives vs unlabelled-high: AUC .46 ... one population"; "The .771 'AP' is a lower
bound that treats real colluders as errors").  Censoring them lifts frozen AP to .9766 and the whole
oracle ceiling collapses to +.0234.  So step83's "+.00097 of a .2853 prize" was measured against a
phantom denominator and has to be re-measured against the real one.

Three surfaces, all training-only, no held-fold truth, no evaluation labels:
  raw       -- 164 labelled positives, every other row negative (67,463).  Contaminated; kept only
               for continuity with step83 and OBS 18.8.
  safe      -- unlabelled rows at risk >= .9 CENSORED out of the pool (67,374 rows, 164 positives).
               The conservative reading: we decline to score rows whose truth we do not know.
  presumed  -- unlabelled rows at risk >= .9 RELABELLED POSITIVE (67,463 rows, 253 positives).
               The closest analogue of the competition metric, where every planted colluder is in
               the truth column whether or not it was disclosed to us.  Note the circularity this
               buys: the presumed positives are selected BY the risk column, so frozen AP on this
               surface is optimistically biased and only DELTAS between rankings mean anything.

Interventions:
  w sweep         rank by logit(risk) + w * z(purity)
  promotion grid  rows with risk < r and purity > p are lifted to just above the band
Controls:
  table bootstrap        400 draws resampling tables, for the CI on each delta
  re-salted partition    tables split in two by a fresh hash, 12 salts; report the ACROSS-PARTITION
                         MINIMUM of the delta, which is what survives a split we did not choose
  random descriptor      200 surrogates with purity's exact marginal and purity's Spearman rho to
                         risk, built through a Gaussian copula.  This is the control that answers
                         "are we buying purity, or buying any tie-breaker inside the tail?"

Output: experiments/pair_side_C_20260912/audit-C-purity-promotion.json
"""
import json
import time

import numpy as np
import polars as pl

from evidence_aug.paths import CV, check_protected, write_json

PU = CV / 'experiments' / 'pu_pairs_20260911'
OUT = CV / 'experiments' / 'pair_side_C_20260912'
MIN_SH = 38
RNG = np.random.default_rng(85)
N_BOOT = 400
N_SALT = 12
N_SURR = 200


def log(*a):
    print(*a, flush=True)


def ap(y, s):
    o = np.argsort(-s, kind='stable')
    yy = y[o]
    c = np.cumsum(yy)
    return float((c / np.arange(1, len(yy) + 1) * yy).sum() / max(1.0, yy.sum()))


def rank_normal(x):
    """Blom rank -> standard normal scores."""
    n = len(x)
    r = np.empty(n)
    r[np.argsort(x, kind='stable')] = np.arange(1, n + 1)
    from scipy.special import ndtri                              # scipy ships with the venv sklearn
    return ndtri((r - .375) / (n + .25))


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(float)
    rb = np.argsort(np.argsort(b, kind='stable'), kind='stable').astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load():
    v = (pl.concat([pl.read_parquet(PU / f'fold{f}' / 'scores.parquet') for f in (1, 2)])
         .unique(subset=['pair_id'], keep='first', maintain_order=True)
         .sort('pair_id'))
    return v.filter(pl.col('shared_hands') >= MIN_SH).with_columns(
        (pl.max_horizontal('p1', 'p2', 'p3') /
         (pl.col('p1') + pl.col('p2') + pl.col('p3')).clip(1e-12)).alias('purity'))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def w_blend(r, d, w):
    lr = logit(r)
    z = (d - d.mean()) / (d.std() + 1e-12)
    return lr + w * z


def promote(r, d, r_cut, p_cut):
    """Lift rows with risk < r_cut and descriptor > p_cut to just above the band, order preserved."""
    s = logit(r)
    band = r < r_cut
    hit = band & (d > p_cut)
    if hit.sum():
        top_of_band = s[band].max() if band.sum() else 0.0
        # keep their relative order by descriptor, place the block immediately above the band
        o = np.argsort(d[hit], kind='stable')
        q = np.empty(int(hit.sum()))
        q[o] = (np.arange(int(hit.sum())) + 1) / (hit.sum() + 1)
        s[hit] = top_of_band + 1e-6 + q
    return s


def main():
    t0 = time.time()
    res = {'author': 'claude (agent)', 'step': 'step85_C_purity_promotion',
           'question': 'is the purity tail-rescue real, and what fraction of the real headroom is it',
           'training_only': True,
           'headroom_note': ('step84: raw oracle ceiling +.2853 is a phantom (89 unlabelled real '
                             'colluders at the top); PU-safe frozen AP .9766, real oracle ceiling '
                             '+.0234.  Fractions below are against the PU-safe ceiling.'),
           'protected_before': check_protected()}

    e = load()
    lab = e['label'].to_numpy()
    r = e['risk'].to_numpy()
    d = e['purity'].to_numpy()
    tab = e['table_id'].to_numpy()
    n = len(r)
    unl_high = (lab < 0) & (r >= .9)
    log('rows', n, 'labelled pos', int((lab == 1).sum()), 'unlabelled>=.9', int(unl_high.sum()))

    surfaces = {
        'raw':      {'keep': np.ones(n, dtype=bool), 'y': (lab == 1).astype(float)},
        'safe':     {'keep': ~unl_high,              'y': (lab == 1).astype(float)},
        'presumed': {'keep': np.ones(n, dtype=bool), 'y': ((lab == 1) | unl_high).astype(float)},
    }

    res['surfaces'] = {}
    for name, spec in surfaces.items():
        k = spec['keep']
        yk, rk, dk, tk = spec['y'][k], r[k], d[k], tab[k]
        base = ap(yk, rk)
        # oracle ceiling on this surface: every positive below risk .9 straight to the top
        s_or = rk.copy()
        s_or[(yk == 1) & (rk < .9)] = np.inf
        info = {'rows': int(k.sum()), 'positives': int(yk.sum()),
                'prevalence': round(float(yk.mean()), 5),
                'frozen_ap': round(base, 5),
                'oracle_ap_rescue_all_positives_below_.9': round(ap(yk, s_or), 5),
                'oracle_ceiling': round(ap(yk, s_or) - base, 5),
                'n_positives_below_risk_.9': int(((yk == 1) & (rk < .9)).sum()),
                'spearman_purity_risk': round(spearman(dk, rk), 4)}

        # ---- w sweep and promotion grid
        arms = {}
        for w in (.125, .25, .5, 1, 2, 4):
            arms[f'w{w}'] = w_blend(rk, dk, w)
        for rc in (.9, .99, .999):
            for pc in (.9, .99, .999):
                arms[f'promote_r{rc}_p{pc}'] = promote(rk, dk, rc, pc)
        deltas = {a: round(ap(yk, s) - base, 5) for a, s in arms.items()}
        info['grid_delta_ap'] = deltas
        info['grid_delta_as_fraction_of_oracle_ceiling'] = {
            a: (round(v / info['oracle_ceiling'], 4) if info['oracle_ceiling'] > 0 else None)
            for a, v in deltas.items()}
        best = sorted(deltas, key=lambda a: -deltas[a])[:3]
        info['best_arms'] = best

        # ---- table bootstrap on the best three
        uq = np.unique(tk)
        ix = {t: np.where(tk == t)[0] for t in uq}
        boot = {a: [] for a in best}
        for _ in range(N_BOOT):
            pick = RNG.choice(uq, size=len(uq), replace=True)
            rows = np.concatenate([ix[t] for t in pick])
            yy = yk[rows]
            if yy.sum() < 5:
                continue
            a0 = ap(yy, rk[rows])
            for a in best:
                boot[a].append(ap(yy, arms[a][rows]) - a0)
        info['bootstrap'] = {a: {'mean': round(float(np.mean(v)), 5),
                                 'ci95': [round(float(np.quantile(v, .025)), 5),
                                          round(float(np.quantile(v, .975)), 5)],
                                 'frac_draws_positive': round(float(np.mean(np.array(v) > 0)), 3)}
                             for a, v in boot.items() if v}

        # ---- re-salted partition: the across-partition MINIMUM
        part = {a: [] for a in best}
        for salt in range(N_SALT):
            h = np.array([hash((str(t), salt, 'p85')) % 2 for t in tk])
            halves = []
            for side in (0, 1):
                m = h == side
                if yk[m].sum() < 5:
                    halves = []
                    break
                a0 = ap(yk[m], rk[m])
                halves.append({a: ap(yk[m], arms[a][m]) - a0 for a in best})
            if halves:
                for a in best:
                    part[a].append(min(halves[0][a], halves[1][a]))
        info['across_partition_minimum'] = {
            a: {'median_of_minima': round(float(np.median(v)), 5),
                'worst_minimum': round(float(np.min(v)), 5),
                'frac_salts_both_halves_positive': round(float(np.mean(np.array(v) > 0)), 3)}
            for a, v in part.items() if v}

        # ---- random-descriptor control: purity's marginal, purity's rho to risk
        rho = info['spearman_purity_risk']
        zr = rank_normal(rk)
        marg = np.sort(dk)
        surr = {a: [] for a in best}
        for _ in range(N_SURR):
            z2 = rho * zr + np.sqrt(max(0.0, 1 - rho ** 2)) * RNG.standard_normal(len(zr))
            sd = np.empty(len(zr))
            sd[np.argsort(z2, kind='stable')] = marg            # exact marginal of purity
            for a in best:
                if a.startswith('w'):
                    s = w_blend(rk, sd, float(a[1:]))
                else:
                    _, rc, pc = a.split('_')
                    s = promote(rk, sd, float(rc[1:]), float(pc[1:]))
                surr[a].append(ap(yk, s) - base)
        info['random_descriptor_null'] = {
            a: {'mean': round(float(np.mean(v)), 5),
                'pct95': round(float(np.quantile(v, .95)), 5),
                'observed': deltas[a],
                'p_value_observed_vs_surrogates': round(float(np.mean(np.array(v) >= deltas[a])), 4),
                'rho_check': round(spearman(sd, rk), 3)}
            for a, v in surr.items()}
        res['surfaces'][name] = info
        log(name, 'frozen', info['frozen_ap'], 'ceiling', info['oracle_ceiling'],
            'best', {a: deltas[a] for a in best}, round(time.time() - t0, 1))

    res['protected_after'] = check_protected()
    res['seconds'] = round(time.time() - t0, 1)
    write_json(OUT / 'audit-C-purity-promotion.json', res)
    log('written', OUT / 'audit-C-purity-promotion.json', res['seconds'])


if __name__ == '__main__':
    main()
