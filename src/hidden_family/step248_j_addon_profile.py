"""STEP248 ADD-ON, stage 2b: if no event is Z-SPECIFIC, what IS group Z?

Stage 2 returned zero Z-specific events and, for every event Z fires, a rate far BELOW A's.
The obvious competing reading is then that Z is not a different mechanism but group A DILUTED:
the same known-family events at a much lower dose.  This file tests that reading, and gives the
fourth family its last pair-internal chance, by splitting the vocabulary into

    KNOWN  the 20 events with the highest fire rate in A   (defined from the control, not from Z)
    OTHER  the remaining 106 events

and asking separately, inside shared-hand deciles, whether Z has an excess over H in each half.
A fourth family routed into the block with no known-family sharp hand must show its excess in
OTHER.  Low-dose known family shows it in KNOWN.

Everything is decile-stratified on shared hands (Z's deciles), because cnt999 is a COUNT and Z is
selected on cnt999 == 0.  Registered splits (dominated / behaviour / shared-hand quartile) are
recomputed with A and H re-weighted inside each split.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import poisson, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_b_events import NAMES, NN
from step248_c_common import OUT
from step248_i_addon_analyse import PSIG, fire_matrix

N_KNOWN = 20


def strat(mask_target, mask_ref, dec, y, ndec=10):
    """observed count of y in target, and the count expected at the reference's decile rates."""
    obs = float(y[mask_target].sum())
    exp = 0.0
    for d in range(ndec):
        t = mask_target & (dec == d)
        r = mask_ref & (dec == d)
        if t.sum() and r.sum():
            exp += t.sum() * y[r].mean()
        elif t.sum():
            exp += t.sum() * y[mask_ref].mean()      # no reference in this decile: fall back
    return obs, exp


def pois_p(obs, exp):
    if exp <= 0:
        return 0.0 if obs else 1.0
    return float(poisson.sf(obs - 1, exp)) if obs > exp else float(poisson.cdf(obs, exp))


def main():
    t0 = time.time()
    cells = pl.read_parquet(OUT / 'step248addon-cells.parquet')
    groups = pl.read_parquet(OUT / 'step248addon-groups.parquet')
    grp = groups['grp'].to_numpy()
    nsh = groups['n_shared'].to_numpy()
    sh99 = groups['shares_hi99_player'].to_numpy()
    beh = groups['predicted_behavior'].to_numpy()
    M = fire_matrix(cells, groups, PSIG)
    Z, A, H = grp == 'Z', grp == 'A', grp == 'H'
    edges = np.quantile(nsh[Z], np.linspace(.1, .9, 9))
    dec = np.clip(np.digitize(nsh, edges), 0, 9)
    res = {}

    # ---------- is Z above the honest control at all, once shared hands are matched? ----------
    anyf = M.any(1)
    print('--- ANY fire (p<1e-3), shared-hand-decile matched onto Z ---')
    for nm, ref in (('H', H), ('A', A)):
        o, e = strat(Z, ref, dec, anyf)
        print(f'  Z observed {o:.0f} / {Z.sum()}   expected at {nm} rates {e:.1f}   '
              f'ratio {o/max(e,1e-9):.2f}   p {pois_p(o, e):.2e}')
        res[f'anyfire_Z_vs_{nm}'] = {'obs': o, 'exp': e, 'p': pois_p(o, e)}

    # ---------- KNOWN vs OTHER halves of the vocabulary ----------------------------------------
    plc = np.array([n.startswith('plc_') for n in NAMES])
    aorder = np.argsort(-M[A].sum(0))
    known = np.zeros(NN, bool)
    known[aorder[:N_KNOWN]] = True
    known &= ~plc
    other = ~known & ~plc
    print(f'\nKNOWN set (top {N_KNOWN} events by fire count in A): '
          f'{", ".join(np.array(NAMES)[known])}')
    res['known_set'] = list(np.array(NAMES)[known])

    print('\n--- where does Z\'s excess over H live? (decile matched) ---')
    print(f'{"half":>10} {"Zobs":>6} {"expH":>7} {"ratio":>7} {"p":>10} | '
          f'{"Aobs":>6} {"expH":>7} {"ratio":>7}')
    for nm, sub in (('KNOWN', known), ('OTHER', other), ('PLACEBO', plc)):
        y = M[:, sub].any(1)
        o, e = strat(Z, H, dec, y)
        oa, ea = strat(A, H, dec, y)
        print(f'{nm:>10} {o:>6.0f} {e:>7.1f} {o/max(e,1e-9):>7.2f} {pois_p(o,e):>10.2e} | '
              f'{oa:>6.0f} {ea:>7.1f} {oa/max(ea,1e-9):>7.2f}')
        res[f'half_{nm}'] = {'Z_obs': o, 'Z_exp_at_H': e, 'Z_ratio': o / max(e, 1e-9),
                             'Z_p': pois_p(o, e), 'A_obs': oa, 'A_exp_at_H': ea,
                             'A_ratio': oa / max(ea, 1e-9)}
    # cell-level version (counts every significant cell, not just the pair)
    print('\n--- same, counting CELLS rather than pairs ---')
    for nm, sub in (('KNOWN', known), ('OTHER', other), ('PLACEBO', plc)):
        y = M[:, sub].sum(1).astype(float)
        o, e = strat(Z, H, dec, y)
        oa, ea = strat(A, H, dec, y)
        print(f'{nm:>10} {o:>6.0f} {e:>7.1f} {o/max(e,1e-9):>7.2f} {pois_p(o,e):>10.2e} | '
              f'{oa:>6.0f} {ea:>7.1f} {oa/max(ea,1e-9):>7.2f}')
        res[f'cells_{nm}'] = {'Z_obs': o, 'Z_exp_at_H': e, 'Z_ratio': o / max(e, 1e-9),
                              'Z_p': pois_p(o, e)}

    # Z pairs whose ENTIRE fired set lies outside the known half -- the only pair-internal
    # candidates for a fourth family inside the block.
    zi = np.where(Z & anyf)[0]
    pure = [i for i in zi if not M[i, known].any()]
    print(f'\nZ pairs firing >=1 cell: {len(zi)};  of those, firing NOTHING in the KNOWN half: '
          f'{len(pure)}')
    hi = np.where(H & anyf)[0]
    hpure = sum(1 for i in hi if not M[i, known].any())
    print(f'  same purity rate in H: {hpure}/{len(hi)} = {hpure/max(len(hi),1):.3f} '
          f'(Z {len(pure)/max(len(zi),1):.3f})')
    res['Z_firing'] = len(zi)
    res['Z_firing_no_known_event'] = len(pure)
    res['H_purity'] = hpure / max(len(hi), 1)
    ids = groups['pair_id'].to_numpy()
    res['Z_pure_pairs'] = [str(ids[i]) for i in pure]

    # ---------- is Z's PROFILE A's profile? ----------------------------------------------------
    rz = M[Z].mean(0)
    ra = M[A].mean(0)
    rh = M[H].mean(0)
    keep = (rz > 0) | (ra > 0)
    rho = spearmanr(rz[keep], ra[keep]).statistic
    rho_h = spearmanr(rz[keep], rh[keep]).statistic
    cos = float(rz @ ra / max(np.linalg.norm(rz) * np.linalg.norm(ra), 1e-12))
    rng = np.random.default_rng(7)
    ph = rng.permutation(np.where(H)[0])
    h1, h2 = M[ph[:2000]].mean(0), M[ph[2000:]].mean(0)
    rho_hh = spearmanr(h1[keep], h2[keep]).statistic
    print(f'\nevent-profile similarity (126 events):  rho(Z, A) {rho:.3f}   '
          f'rho(Z, H) {rho_h:.3f}   cosine(Z, A) {cos:.3f}   [H-split reference '
          f'rho {rho_hh:.3f}]')
    res['profile'] = {'rho_ZA': float(rho), 'rho_ZH': float(rho_h), 'cos_ZA': cos,
                      'rho_HH': float(rho_hh)}
    dil = float(np.sum(rz) / max(np.sum(ra), 1e-12))
    print(f'Z fires {dil:.3f} x as many cells per pair as A (unstratified mean cells '
          f'{M[Z].sum(1).mean():.2f} vs {M[A].sum(1).mean():.2f})')
    res['dilution'] = dil

    # ---------- registered splits --------------------------------------------------------------
    print('\n--- registered splits of Z (A and H re-matched inside each split) ---')
    print(f'{"split":>34} {"n":>5} {"fire":>5} {"expH":>7} {"ratio":>7} {"p":>10} {"expA":>7}')
    q = np.quantile(nsh[Z], [.25, .5, .75])
    splits = [('dominated (shares >=.99 player)', Z & sh99), ('independent', Z & ~sh99)]
    for b in ('directed_transfer', 'soft_play', 'coordinated_isolation'):
        splits.append((f'behaviour {b}', Z & (beh == b)))
    for k in range(4):
        lo = -1 if k == 0 else q[k - 1]
        hiq = 1e18 if k == 3 else q[k]
        splits.append((f'shared-hand Q{k+1} ({int(max(lo,0))}-'
                       f'{int(hiq) if hiq < 1e17 else 9999})', Z & (nsh > lo) & (nsh <= hiq)))
    for nm, m in splits:
        o, e = strat(m, H, dec, anyf)
        _, ea = strat(m, A, dec, anyf)
        print(f'{nm:>34} {int(m.sum()):>5} {o:>5.0f} {e:>7.1f} {o/max(e,1e-9):>7.2f} '
              f'{pois_p(o,e):>10.2e} {ea:>7.1f}')
        res.setdefault('splits', {})[nm] = {'n': int(m.sum()), 'fire': o, 'exp_H': e,
                                            'ratio': o / max(e, 1e-9), 'p': pois_p(o, e),
                                            'exp_A': ea}
        # OTHER-half excess inside the split: the fourth family's last hiding place
        y = M[:, other].any(1)
        o2, e2 = strat(m, H, dec, y)
        res['splits'][nm]['other_obs'] = o2
        res['splits'][nm]['other_exp_H'] = e2
    print('\n  OTHER-half (non-known events) inside each split:')
    for nm, m in splits:
        d = res['splits'][nm]
        print(f'{nm:>34} {int(m.sum()):>5} {d["other_obs"]:>5.0f} {d["other_exp_H"]:>7.1f} '
              f'{d["other_obs"]/max(d["other_exp_H"],1e-9):>7.2f} '
              f'{pois_p(d["other_obs"], d["other_exp_H"]):>10.2e}')

    (OUT / 'step248addon-profile.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\ndone  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
