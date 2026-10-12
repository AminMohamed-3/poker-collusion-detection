"""Step 68: reconcile the five public readings exactly, then ask the held evidence data why the
struct stage 2 transferred at zero while the r382 hybrid transferred at ~90 %.

P0 is measured.  `submission_S_stage2_r382DTSP.csv` scored .89336 (predicted range .8955-.9014,
so the reading is BELOW its own pre-registration).  Because the composite is
.7 P + .2 E + .1 B and E depends only on the evidence lists, five readings on one split give an
exactly determined linear system:

    S                     = .7 P_S  + .2 E_c1   + .1 B      .89055 (assumed S; see O1)
    M3_stage2_chrono3     = .7 P_M3 + .2 E_c3   + .1 B      .89161
    M4_stage2_chrono3     = .7 P_M4 + .2 E_c3   + .1 B      .89175
    M3_stage2_r382DTSP    = .7 P_M3 + .2 E_r382 + .1 B      .89447
    S_stage2_r382DTSP     = .7 P_S  + .2 E_r382 + .1 B      .89336

Four independent differences fall out with no modelling at all.  Part A prints them against the
pre-registered decision rules.

Part B then re-ledgers the ONE held read (folds 0-2, `pairs.csv`) against the covariate the
evaluation set truncates.  17 found that every pair-side lever after S lived on pairs with fewer
than 38 shared hands.  The evidence side has its own version of the same question: the stage-2
models were trained and scored on development pairs whose candidate sets are much longer than the
evaluation's (plan: median ~121 vs ~78 hands).  If the struct gain sits on long pairs and the
r382 gain does not, length explains the whole transfer failure and E4 is the fix.

Read-only apart from `experiments/public_ledger_20260912/`.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob
import numpy as np
import polars as pl

from evidence_aug.paths import CV, check_protected, write_json

OUT = CV / 'experiments' / 'public_ledger_20260912'
FS2 = CV / 'experiments' / 'family_stage2_20260911'
R382 = CV / 'experiments' / 'family_stage2_r382_20260911'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
RNG = np.random.default_rng(20260912)

# public readings, in the order they were measured
PUB = {'base_chronology_frac': .87625, 'S': .89055, 'M3_stage2_chrono3': .89161,
       'M4_stage2_chrono3': .89175, 'M3_stage2_r382DTSP': .89447, 'S_stage2_r382DTSP': .89336}


def boot(delta, n=4000):
    """paired bootstrap of a mean per-pair delta"""
    d = np.asarray(delta, float)
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def part_a():
    print('=' * 100)
    print('A.  THE FIVE PUBLIC READINGS, SOLVED EXACTLY (one split, additive metric)')
    print('=' * 100)
    S, M3c3, M4c3, M3r, Sr = (PUB['S'], PUB['M3_stage2_chrono3'], PUB['M4_stage2_chrono3'],
                              PUB['M3_stage2_r382DTSP'], PUB['S_stage2_r382DTSP'])
    dP_M3_S = (M3r - Sr) / .7                      # same evidence, different risk
    dE_r382_c3 = (M3r - M3c3) / .2                 # same risk, different evidence
    dE_r382_c1 = (Sr - S) / .2                     # same risk, different evidence
    dE_c3_c1 = dE_r382_c1 - dE_r382_c3             # by difference
    dP_M4_M3 = (M4c3 - M3c3) / .7
    # consistency: the same dP must appear in the chrono3 pair as well
    resid = (M3c3 - S) - (.7 * dP_M3_S + .2 * dE_c3_c1)
    rows = [
        ('pair AP,  M3 order - S order', dP_M3_S, .7, M3r - Sr, 'S_stage2_r382 vs M3_stage2_r382'),
        ('pair AP,  M4 order - M3 order', dP_M4_M3, .7, M4c3 - M3c3, 'M4_stage2_c3 vs M3_stage2_c3'),
        ('evidence MAP@5, r382 hybrid - struct(chrono3)', dE_r382_c3, .2, M3r - M3c3,
         'M3_stage2_r382 vs M3_stage2_c3'),
        ('evidence MAP@5, r382 hybrid - stage1(chrono_frac)', dE_r382_c1, .2, Sr - S,
         'S_stage2_r382 vs S'),
        ('evidence MAP@5, struct(chrono3) - stage1(chrono_frac)', dE_c3_c1, .2,
         .2 * dE_c3_c1, 'by difference of the two above')]
    print(f'{"quantity":<52}{"component":>12}{"composite":>12}   measured by')
    for name, val, w, comp, src in rows:
        print(f'{name:<52}{val:>+12.4f}{comp:>+12.5f}   {src}')
    print(f'\n  identity residual (must be 0 by construction): {resid:+.6f}')

    print()
    print('  Against the held expectations:')
    print(f'    struct stage 2 vs chrono1   held +.0411 [+.0266, +.0563]   public {dE_c3_c1:+.4f}'
          f'   -> transfer {100 * dE_c3_c1 / .0411:+.0f} %')
    print(f'    r382 hybrid  vs struct      held +.0163 [+.0046, +.0277]   public {dE_r382_c3:+.4f}'
          f'   -> transfer {100 * dE_r382_c3 / .0163:+.0f} %')
    print(f'    net evidence vs stage 1     held +.0574                    public {dE_r382_c1:+.4f}'
          f'   -> transfer {100 * dE_r382_c1 / .0574:+.0f} %')

    print()
    print('  Pre-registered decision rules for the P0 upload (plan section 1):')
    r1 = Sr - M3r
    r2 = Sr - S
    v1 = ('S is the pair-side base, M series closed' if r1 > .002 else
          'equal within +-.002 -> keep S (simpler, validated on eligible rows)' if r1 >= -.002 else
          "keep M3's order")
    v2 = ('evidence transferred fully' if r2 >= .008 else
          'partial transfer -> run E4 first' if r2 >= .003 else
          'the struct stage 2 did NOT transfer beyond r382 -> E4 and the chrono1 variant')
    print(f'    read 1  score - .89447 = {r1:+.5f}  ->  {v1}')
    print(f'    read 2  score - .89055 = {r2:+.5f}  ->  {v2}')
    print(f'    pre-registered range for the file itself [.8955, .9014] -> measured .89336: BELOW')

    # component levels, using the probe solution of the base (P .938 / E .608 / B .976)
    P_S = .938 + (PUB['S'] - PUB['base_chronology_frac']) / .7
    lvl = {'P_base': .938, 'E_base': .608, 'B': .976, 'P_S': P_S, 'P_M3': P_S + dP_M3_S,
           'E_chrono3': .608 + dE_c3_c1, 'E_r382': .608 + dE_r382_c1}
    print()
    print('  Implied component levels (probe base P .938 / E .608 / B .976):')
    for k, v in lvl.items():
        print(f'    {k:<12}{v:.4f}')
    best = max(PUB.values())
    need = .93798 - best
    print(f'\n  best public {best:.5f}; leader .93798; gap {need:+.5f}'
          f'  =  {need / .7:+.4f} pair AP  or  {need / .2:+.4f} evidence MAP  (or a mix)')
    return {'differences': {n: v for n, v, _, _, _ in rows}, 'levels': lvl,
            'read1': r1, 'read2': r2, 'verdict_read1': v1, 'verdict_read2': v2,
            'identity_residual': resid, 'public': PUB}


def held_frame():
    """per-pair held MAP@5 for every candidate, plus the candidate-set size of each pair"""
    a = pl.read_csv(FS2 / 'pairs.csv')
    b = pl.read_csv(R382 / 'pairs.csv').filter(pl.col('candidate') != 'chrono1')
    both = pl.concat([a, b.filter(pl.col('candidate') != 'allfam_top20_struct')])
    # hand counts per pair from the frozen chrono1 score frames (all folds share the format)
    sizes = pl.concat([pl.read_parquet(FS2 / 'scores' / f'fold{k}-chrono1.parquet')
                       .group_by('pair_id').len().rename({'len': 'n_hands'}) for k in (0, 1, 2)])
    sizes = sizes.group_by('pair_id').agg(pl.col('n_hands').max())
    wide = both.pivot(values='map5', index=['fold', 'pair_id', 'family', 'predicted'],
                      on='candidate')
    return wide.join(sizes, on='pair_id', how='left')


def eval_sizes():
    """candidate-set size of every one of the 16,060 selected evaluation pairs"""
    files = sorted(glob.glob(str(FS2 / 'eval_structural' / 'batch_*.parquet')))
    q = (pl.concat([pl.scan_parquet(f).select('pair_id') for f in files])
         .group_by('pair_id').len().rename({'len': 'n_hands'}))
    return q.collect()


def part_b(w, ev):
    print()
    print('=' * 100)
    print('B.  THE HELD EVIDENCE GAIN AS A FUNCTION OF CANDIDATE-SET SIZE')
    print('    (evaluation pairs are much shorter than development pairs; if the struct gain is')
    print('     carried by long pairs it cannot transfer)')
    print('=' * 100)
    struct = w['allfam_top20_struct'].to_numpy()
    c1 = w['chrono1'].to_numpy()
    r382 = w['allfam_top20_r382'].to_numpy()
    pred = w['predicted'].to_numpy()
    hyb = np.where(np.isin(pred, ['directed_transfer', 'soft_play']), r382, struct)
    n = w['n_hands'].to_numpy().astype(float)
    fam = w['family'].to_numpy()
    d_struct, d_r382 = struct - c1, hyb - struct
    d_net = hyb - c1

    dv = np.sort(n)
    evn = np.sort(ev['n_hands'].to_numpy().astype(float))
    print(f'  candidate-set size   development held pairs (n={len(dv)}):  '
          f'q10 {np.percentile(dv, 10):.0f}  median {np.median(dv):.0f}  q90 {np.percentile(dv, 90):.0f}')
    print(f'                       evaluation selected pairs (n={len(evn)}):  '
          f'q10 {np.percentile(evn, 10):.0f}  median {np.median(evn):.0f}  q90 {np.percentile(evn, 90):.0f}')

    bands = [(0, 38), (38, 60), (60, 90), (90, 120), (120, 10 ** 9)]
    print()
    print(f'{"size band":<14}{"n held":>8}{"% of eval":>11}'
          f'{"struct-chrono1":>17}{"r382-struct":>14}{"net":>9}')
    per_band = []
    for lo, hi in bands:
        m = (n >= lo) & (n < hi)
        share = float(((evn >= lo) & (evn < hi)).mean())
        if m.sum() == 0:
            continue
        per_band.append({'lo': lo, 'hi': hi, 'n': int(m.sum()), 'eval_share': share,
                         'struct': float(d_struct[m].mean()), 'r382': float(d_r382[m].mean()),
                         'net': float(d_net[m].mean())})
        lab = f'{lo}-{hi if hi < 10 ** 9 else ""}'
        print(f'{lab:<14}{int(m.sum()):>8}{100 * share:>10.1f}%'
              f'{d_struct[m].mean():>+17.4f}{d_r382[m].mean():>+14.4f}{d_net[m].mean():>+9.4f}')

    print()
    print('  pooled, and reweighted so the held pairs match the evaluation size distribution:')
    edges = np.percentile(evn, np.linspace(0, 100, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    bi_dev = np.clip(np.searchsorted(edges, n, side='right') - 1, 0, 9)
    bi_ev = np.clip(np.searchsorted(edges, evn, side='right') - 1, 0, 9)
    p_ev = np.bincount(bi_ev, minlength=10) / len(bi_ev)
    p_dv = np.bincount(bi_dev, minlength=10) / len(bi_dev)
    wgt = np.where(p_dv[bi_dev] > 0, p_ev[bi_dev] / np.maximum(p_dv[bi_dev], 1e-9), 0.0)
    wgt = wgt / wgt.mean()
    out = {'per_band': per_band, 'sizes': {'dev_median': float(np.median(dv)),
                                           'eval_median': float(np.median(evn))}}
    print(f'{"contrast":<28}{"pooled":>10}{"95% CI":>20}{"eligible>=38":>14}'
          f'{"reweighted":>12}{"long>=120":>11}')
    for name, d in [('struct - chrono1', d_struct), ('r382 hybrid - struct', d_r382),
                    ('net (hybrid - chrono1)', d_net)]:
        m, lo, hi = boot(d)
        el = float(d[n >= 38].mean())
        rw = float((d * wgt).mean())
        lg = float(d[n >= 120].mean())
        out[name] = {'pooled': m, 'ci': [lo, hi], 'eligible38': el, 'reweighted': rw, 'long120': lg}
        print(f'{name:<28}{m:>+10.4f}{f"[{lo:+.4f}, {hi:+.4f}]":>20}{el:>+14.4f}'
              f'{rw:>+12.4f}{lg:>+11.4f}')

    print()
    print(f'{"contrast":<28}' + ''.join(f'{f[:14]:>17}' for f in FAMS))
    for name, d in [('struct - chrono1', d_struct), ('r382 hybrid - struct', d_r382),
                    ('net (hybrid - chrono1)', d_net)]:
        cells = []
        for f in FAMS:
            m = fam == f
            cells.append(f'{d[m].mean():+.4f} (n{int(m.sum())})')
        out[name]['by_true_family'] = {f: float(d[fam == f].mean()) for f in FAMS}
        print(f'{name:<28}' + ''.join(f'{c:>17}' for c in cells))

    print()
    print('  the same, restricted to pairs whose candidate set is evaluation-like (38-100 hands):')
    m = (n >= 38) & (n <= 100)
    print(f'{"":28}' + ''.join(f'{f[:14]:>17}' for f in FAMS) + f'{"all":>12}')
    for name, d in [('struct - chrono1', d_struct), ('r382 hybrid - struct', d_r382)]:
        cells = [f'{d[m & (fam == f)].mean():+.4f} (n{int((m & (fam == f)).sum())})' for f in FAMS]
        out[name]['eval_like_38_100'] = float(d[m].mean())
        print(f'{name:<28}' + ''.join(f'{c:>17}' for c in cells) + f'{d[m].mean():>+12.4f}')
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = check_protected()
    a = part_a()
    w = held_frame()
    ev = eval_sizes()
    b = part_b(w, ev)
    write_json(OUT / 'step68-public-ledger.json',
               {'part_a': a, 'part_b': b, 'protected': protected,
                'note': 'S assumed to be the .89055 file (O1 unconfirmed); every quantity that '
                        'depends on it is the struct-vs-stage1 evidence delta.'})
    check_protected()
    print()
    print(f'written {OUT / "step68-public-ledger.json"}')


if __name__ == '__main__':
    main()
