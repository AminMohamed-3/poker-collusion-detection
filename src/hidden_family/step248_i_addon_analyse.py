"""STEP248 ADD-ON, stage 2: is there an event that fires for Z and not for A and not for H?

Rules fixed in registration248_addon.json before any of this was computed:
  FIRE          p < 1e-3 (secondary 1e-2), either direction of the pair
  stratification every comparison is made inside shared-hand deciles defined on Z, because
                cnt999 is a COUNT and Z is selected on cnt999 == 0
  Z-SPECIFIC    weighted rate_Z >= 3x rate_H AND >= 2x rate_A AND both stratified Poisson tests
                p < .05/126 AND at least 8 Z pairs fire it
  blind control if any PLACEBO event satisfies the rule, the table is uninterpretable
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_b_events import NAMES, NN
from step248_c_common import OUT

PSIG = 1e-3
PSEC = 1e-2
BONF = 0.05 / NN
MIN_Z = 8


def fire_matrix(cells, groups, thr):
    """pairs x events boolean 'fired in either direction', plus the group and shared hands."""
    f = (cells.filter(pl.col('pval') < thr)
         .group_by(['pair_id', 'event']).agg(pl.col('pval').min().alias('p')))
    ids = groups['pair_id'].to_list()
    pidx = {p: i for i, p in enumerate(ids)}
    M = np.zeros((len(ids), NN), bool)
    for pid, e in zip(f['pair_id'].to_list(), f['event'].to_list()):
        M[pidx[pid], e] = True
    return M


def strat_rates(M, grp, dec, ndec):
    """per group: decile-specific fire rate and pair counts."""
    out = {}
    for g in ('Z', 'A', 'H'):
        m = grp == g
        r = np.zeros((ndec, NN))
        c = np.zeros(ndec)
        for d in range(ndec):
            s = m & (dec == d)
            c[d] = s.sum()
            if c[d]:
                r[d] = M[s].mean(0)
        out[g] = (r, c)
    return out


def main():
    t0 = time.time()
    cells = pl.read_parquet(OUT / 'step248addon-cells.parquet')
    groups = pl.read_parquet(OUT / 'step248addon-groups.parquet')
    grp = groups['grp'].to_numpy()
    nsh = groups['n_shared'].to_numpy()
    res = {}

    zs = nsh[grp == 'Z']
    edges = np.quantile(zs, np.linspace(.1, .9, 9))
    dec = np.clip(np.digitize(nsh, edges), 0, 9)
    ndec = 10
    print('shared hands: Z median %.0f, A median %.0f, H median %.0f' %
          (np.median(zs), np.median(nsh[grp == 'A']), np.median(nsh[grp == 'H'])))
    res['shared_hands'] = {g: {'median': float(np.median(nsh[grp == g])),
                               'q10': float(np.quantile(nsh[grp == g], .1)),
                               'q90': float(np.quantile(nsh[grp == g], .9)),
                               'n': int((grp == g).sum())} for g in ('Z', 'A', 'H')}

    M = fire_matrix(cells, groups, PSIG)
    M2 = fire_matrix(cells, groups, PSEC)
    rates = strat_rates(M, grp, dec, ndec)
    wZ = rates['Z'][1] / rates['Z'][1].sum()

    rows = []
    for e in range(NN):
        obsZ = int(M[grp == 'Z', e].sum())
        r = {}
        for g in ('Z', 'A', 'H'):
            rr, cc = rates[g]
            ok = cc > 0
            w = wZ[ok] / wZ[ok].sum()
            r[g] = float((rr[ok, e] * w).sum())
        expA = float((rates['Z'][1] * rates['A'][0][:, e]).sum())
        expH = float((rates['Z'][1] * rates['H'][0][:, e]).sum())
        pA = float(min(1.0, 2 * poisson.sf(obsZ - 1, max(expA, 1e-9)))) if obsZ else 1.0
        pH = float(min(1.0, 2 * poisson.sf(obsZ - 1, max(expH, 1e-9)))) if obsZ else 1.0
        zsp = (r['Z'] >= 3 * r['H'] and r['Z'] >= 2 * r['A'] and pA < BONF and pH < BONF
               and obsZ >= MIN_Z)
        rows.append({'event': NAMES[e], 'placebo': NAMES[e].startswith('plc_'),
                     'Z_fired': obsZ, 'A_fired': int(M[grp == 'A', e].sum()),
                     'H_fired': int(M[grp == 'H', e].sum()),
                     'rate_Z': r['Z'], 'rate_A': r['A'], 'rate_H': r['H'],
                     'exp_Z_at_A_rate': expA, 'exp_Z_at_H_rate': expH,
                     'p_vs_A': pA, 'p_vs_H': pH, 'z_specific': zsp,
                     'Z_fired_p01': int(M2[grp == 'Z', e].sum()),
                     'A_fired_p01': int(M2[grp == 'A', e].sum()),
                     'H_fired_p01': int(M2[grp == 'H', e].sum())})
    tab = pl.DataFrame(rows).sort('Z_fired', descending=True)
    tab.write_csv(OUT / 'step248addon-event-table.csv')
    zsp = tab.filter('z_specific')
    print(f'\nZ-SPECIFIC events: {zsp.height}'
          + (f' ({", ".join(zsp["event"].to_list())})' if zsp.height else ''))
    print('\n--- events by Z firing count (shared-hand-decile weighted rates) ---')
    print('event                   plc  Zfire Afire Hfire   rateZ   rateA   rateH  expZ@A expZ@H'
          '   p_vs_A   p_vs_H  Zspec')
    for r in tab.head(26).iter_rows(named=True):
        print(f'{r["event"]:>22} {"P" if r["placebo"] else " "}   {r["Z_fired"]:>5} '
              f'{r["A_fired"]:>5} {r["H_fired"]:>5} {r["rate_Z"]:>7.4f} {r["rate_A"]:>7.4f} '
              f'{r["rate_H"]:>7.4f} {r["exp_Z_at_A_rate"]:>6.1f} {r["exp_Z_at_H_rate"]:>6.1f} '
              f'{r["p_vs_A"]:>8.1e} {r["p_vs_H"]:>8.1e}  {"YES" if r["z_specific"] else ""}')
    print('  placebo rows:')
    for r in tab.filter('placebo').iter_rows(named=True):
        print(f'{r["event"]:>22} P   {r["Z_fired"]:>5} {r["A_fired"]:>5} {r["H_fired"]:>5} '
              f'{r["rate_Z"]:>7.4f} {r["rate_A"]:>7.4f} {r["rate_H"]:>7.4f} '
              f'{r["exp_Z_at_A_rate"]:>6.1f} {r["exp_Z_at_H_rate"]:>6.1f} '
              f'{r["p_vs_A"]:>8.1e} {r["p_vs_H"]:>8.1e}  {"YES" if r["z_specific"] else ""}')
    res['z_specific_events'] = zsp['event'].to_list()
    res['placebo_z_specific'] = zsp.filter('placebo')['event'].to_list()

    # ---- how many pairs fire ANYTHING at all -------------------------------------------
    anyf = M.any(1)
    nf = M.sum(1)
    print('\n--- does the group fire at all? ---')
    for g in ('Z', 'A', 'H'):
        m = grp == g
        print(f'  {g}: n={int(m.sum())}, share firing >=1 cell {anyf[m].mean():.3f}, '
              f'median cells {np.median(nf[m]):.0f}, mean cells {nf[m].mean():.2f}')
        res.setdefault('firing', {})[g] = {'n': int(m.sum()),
                                           'share_any': float(anyf[m].mean()),
                                           'mean_cells': float(nf[m].mean())}

    # ---- coherence -----------------------------------------------------------------------
    def jac(Msub, cap=400):
        idx = np.where(Msub.any(1))[0]
        if len(idx) < 2:
            return None
        if len(idx) > cap:
            idx = np.random.default_rng(0).choice(idx, cap, replace=False)
        X = Msub[idx].astype(float)
        inter = X @ X.T
        sz = X.sum(1)
        un = sz[:, None] + sz[None, :] - inter
        iu = np.triu_indices(len(idx), 1)
        return float(np.mean(inter[iu] / np.maximum(un[iu], 1)))
    res['jaccard'] = {g: jac(M[grp == g]) for g in ('Z', 'A', 'H')}
    print('\nmean pairwise Jaccard of fired-event sets: '
          + ', '.join(f'{g} {res["jaccard"][g]}' for g in ('Z', 'A', 'H')))

    # ---- the declared confound splits -----------------------------------------------------
    sh99 = groups['shares_hi99_player'].to_numpy()
    beh = groups['predicted_behavior'].to_numpy()
    z = grp == 'Z'
    print('\n--- Z confound splits ---')
    print(f'  Z sharing a player with a risk>=.99 block pair: '
          f'{sh99[z].mean():.3f}  (A {sh99[grp=="A"].mean():.3f}, H {sh99[grp=="H"].mean():.3f})')
    res['Z_shares_hi99'] = float(sh99[z].mean())
    res['A_shares_hi99'] = float(sh99[grp == 'A'].mean())
    res['H_shares_hi99'] = float(sh99[grp == 'H'].mean())
    for b in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
        mz = z & (beh == b)
        ma = (grp == 'A') & (beh == b)
        print(f'  {b:>22}: Z n={int(mz.sum())} share firing {anyf[mz].mean():.3f} '
              f'median shared {np.median(nsh[mz]):.0f} | A n={int(ma.sum())} '
              f'share firing {anyf[ma].mean():.3f} median shared {np.median(nsh[ma]):.0f}')
        res.setdefault('by_behavior', {})[b] = {
            'Z_n': int(mz.sum()), 'Z_share_firing': float(anyf[mz].mean()),
            'Z_median_shared': float(np.median(nsh[mz])),
            'A_n': int(ma.sum()), 'A_share_firing': float(anyf[ma].mean())}
    for lab, m in (('Z dominated (shares >=.99 player)', z & sh99),
                   ('Z independent', z & ~sh99)):
        print(f'  {lab}: n={int(m.sum())} share firing {anyf[m].mean():.3f} '
              f'median shared {np.median(nsh[m]):.0f} mean cells {nf[m].mean():.2f}')
        res.setdefault('Z_split', {})[lab] = {'n': int(m.sum()),
                                              'share_firing': float(anyf[m].mean()),
                                              'median_shared': float(np.median(nsh[m])),
                                              'mean_cells': float(nf[m].mean())}
    (OUT / 'step248addon-analysis.json').write_text(json.dumps(res, indent=1), encoding='utf-8')

    # ---- strongest Z pairs for the hand reading -------------------------------------------
    best = (cells.filter(pl.col('grp') == 'Z').group_by('pair_id')
            .agg(pl.col('pval').min().alias('min_pval'),
                 (pl.col('pval') < PSIG).sum().alias('n_cells')))
    ev = (cells.filter((pl.col('grp') == 'Z') & (pl.col('pval') < PSIG))
          .group_by('pair_id').agg(pl.col('event_name').unique().alias('events')))
    best = best.join(ev, on='pair_id', how='left').join(
        groups.select('pair_id', 'tidx', 'a', 'b', 'player_a', 'player_b', 'table_id',
                      'n_shared', 'risk_score', 'predicted_behavior', 'shares_hi99_player'),
        on='pair_id', how='left').sort('min_pval')
    best.write_parquet(OUT / 'step248addon-Z-pairs.parquet')
    print(f'\ntop Z pairs by min p-value:')
    for r in best.head(20).iter_rows(named=True):
        print(f'  {r["pair_id"]} risk {r["risk_score"]:.3f} {r["predicted_behavior"][:14]:>14} '
              f'shared {r["n_shared"]:>4} minp {r["min_pval"]:.2e} cells {r["n_cells"]} '
              f'dom {int(r["shares_hi99_player"])} {r["events"]}')
    print(f'\ndone  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
