"""Step 74b: E4's gate fired, but against the wrong reference.  What does it look like against the
chain that actually ships, and does the effect behave like a length-shift effect at all?

Step 74 pre-registered `ref_full_chrono3` -- the STRUCT stage-2 design trained on full histories -- as
the reference, and at the decisive prefix .6 seven arms cleared +.010 with a positive lower bound.  The
reference is wrong, and one row proves it: `r382_full_chrono3` "meets the gate" at +.0371.  That arm is
not new and not a candidate.  It is r382-for-all-families on full histories, which was already read on
held folds 0-2 (18.14) and FAILED there on its own isolation family at -.0184; production ships the
hybrid precisely because of that.  A gate an already-rejected arm passes is measuring the r382-vs-struct
design gap, not the truncation.  Everything below is therefore POST-HOC -- written after step 74's
numbers were seen -- and is labelled post-hoc wherever it is recorded.

Three contrasts step 74 did not compute, all on the same fits (same construction, same seeds, so the
per-arm means must reproduce step 74's stored JSON exactly; that assertion is the control):

  (a) VS THE SHIPPING CHAIN.  Every arm paired against `hybrid_full_chrono3` -- r382 for transfer and
      soft play, struct for isolation, trained on full histories, chrono3 -- which is what production
      does today.  This is the only contrast whose sign answers "should we change anything".
  (b) DOSE-RESPONSE, the mechanism test.  (trunc - full) at fixed design and fixed chrono, at every
      truncated prefix.  Stated before computing it: a real length-shift effect is one that GROWS as
      the history gets shorter, because the extrapolation on `chrono_rank` and `n_hands` gets worse.
      If the .5 prefix does not show a larger truncation-training advantage than .6, there is no
      dose-response, and a solitary bump at .6 is a prefix-specific fluctuation rather than a lever.
  (c) THE OTHER HALF OF THE HYPOTHESIS.  (chrono1 - chrono3) at fixed design and fixed training set.
      E4's premise is that the two length-dependent chrono columns are the liability.  If that is true,
      dropping them (chrono1) should help under truncation and help more the shorter the prefix.

Production bar, stricter than step 74's and stated before (a) was computed: an arm is proposed only if
it beats `hybrid_full_chrono3` by more than +.010 with a paired bootstrap lower bound above zero and no
family below -.005, AT BOTH .6 AND .5.  Both, because the true evaluation width is 40/60 = .667 of the
development window and a production lever has to hold across the plausible range rather than at one
point; and because a single prefix is exactly what (b) exists to distrust.

Training-only (the 162 audit pairs, 172 tables outside every held fold).  No held read.  No leaderboard
outcome and no evaluation label is used anywhere.
Output: experiments/label_process_20260911/audit-length-shift-contrasts.json
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_k, '2')
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import check_protected, write_json
from evidence_aug.step14_audit_truncation import OUT, TOP
from evidence_aug.step65_structural_evidence import log
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step74_length_shift import (CI, PREFIXES, REF, build_d0, designs_of, evaluate,
                                              fit_predict, frame_at)

SHIP = 'hybrid_full_chrono3'
BAR = {'beats_shipping_by': .010, 'ci95_lower_gt': 0.0, 'family_floor': -.005,
       'required_at': [.6, .5], 'reference': SHIP,
       'post_hoc': 'written after step 74 reported its gate; recorded as post-hoc everywhere',
       'why_new_reference': 'step 74 used ref_full_chrono3 (struct); production ships the hybrid, and '
                            'r382_full_chrono3 -- already read and rejected on folds 0-2 in 18.14 -- '
                            'clears the old gate, which shows the old gate measures the design gap'}
DOSE = ('a length-shift effect must grow as the prefix shrinks; if (trunc - full) at .5 does not '
        'exceed (trunc - full) at .6, the .6 result is a fluctuation and no lever follows')


def main():
    t0 = time.time()
    check_protected()
    stored = json.loads((OUT / 'audit-length-shift.json').read_text(encoding='utf-8'))
    d0, struct, rcols = build_d0()
    designs = designs_of(struct, rcols)

    res = {'training_only': True, 'post_hoc': True, 'bar': BAR, 'dose_response_rule': DOSE,
           'reproduces': 'evidence_aug/step74_length_shift.py', 'variants': {},
           'vs_shipping': {}, 'dose_response': {}, 'chrono_contrast': {}}
    print('=' * 100)
    print('STEP 74b.  E4 AGAINST THE CHAIN THAT SHIPS, AND WHETHER THE EFFECT IS DOSE-RESPONSIVE')
    print('=' * 100)

    per_pair_all, ctx, full_frame = {}, {}, None
    for p in PREFIXES:
        d = frame_at(d0, p)
        if p == 1.0:
            full_frame = d
        truths = {k[0]: set(g['hand_id']) for k, g in
                  d.filter(pl.col('relevance') == 1).group_by('pair_id')}
        labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
        ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
        ctx[p] = (labels, ptab, sorted(set(ptab.values())))
        print(f'\n  prefix {p}: {len(truths)} pairs, {d.height} hands')

        for dname, cols in designs.items():
            for train_on in ('full', 'trunc'):
                if p == 1.0 and train_on == 'trunc':
                    continue
                tag = f"{dname.split('_')[0]}_{train_on}_{dname.split('_')[1]}"
                summ, per_pair = evaluate(d, fit_predict(d if train_on == 'trunc' else full_frame,
                                                         d, cols, TOP), TOP, truths, labels)
                per_pair_all[(p, tag)] = per_pair
                res['variants'].setdefault(str(p), {})[tag] = round(summ['map5'], 4)
                # control: same construction and seeds, so this must be step 74's number
                was = stored['variants'][str(p)][tag]['map5']
                assert abs(round(summ['map5'], 4) - was) < 1e-9, f'{p} {tag}: {summ["map5"]} vs {was}'

        for train_on in ('full', 'trunc'):
            if p == 1.0 and train_on == 'trunc':
                continue
            for ch in ('chrono3', 'chrono1'):
                a = per_pair_all[(p, f'r382_{train_on}_{ch}')]
                b = per_pair_all[(p, f'ref_{train_on}_{ch}')]
                per_pair_all[(p, f'hybrid_{train_on}_{ch}')] = {
                    q: (b[q] if labels[q] == CI else a[q]) for q in a}
        print(f'    reproduced all {len(designs) * (1 if p == 1.0 else 2)} stored means exactly')

    def contrast(p, base, arm):
        labels, ptab, pools = ctx[p]
        a, b = per_pair_all[(p, base)], per_pair_all[(p, arm)]
        bb = boot(a, b, ptab, pools)
        byf = {}
        for f in FAMILIES:
            fp = [q for q in b if labels[q] == f]
            byf[f] = round(float(np.mean([b[q] - a[q] for q in fp])), 4) if fp else None
        return {'mean_delta': round(bb['mean_delta'], 4), 'ci95': [round(v, 4) for v in bb['ci95']],
                'by_family': byf, 'pairs_better': bb['pairs_better'], 'pairs_worse': bb['pairs_worse'],
                'floor_ok': all(v is None or v >= BAR['family_floor'] for v in byf.values())}

    # ---------- (a) against the shipping chain ----------
    print(f'\n  (a) paired against {SHIP} -- the chain production actually ships')
    arms = sorted({k[1] for k in per_pair_all if k[1] != SHIP})
    for p in PREFIXES:
        print(f'    prefix {p}:')
        for arm in arms:
            if (p, arm) not in per_pair_all:
                continue
            r = contrast(p, SHIP, arm)
            r['meets_bar_here'] = bool(r['mean_delta'] >= BAR['beats_shipping_by']
                                       and r['ci95'][0] > BAR['ci95_lower_gt'] and r['floor_ok'])
            res['vs_shipping'].setdefault(str(p), {})[arm] = r
            print(f"      {arm:<26} {r['mean_delta']:+.4f}  [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
                  f"  {'MEETS' if r['meets_bar_here'] else ''}")

    res['meets_bar_at_both'] = sorted(
        a for a in arms
        if all(res['vs_shipping'].get(str(p), {}).get(a, {}).get('meets_bar_here')
               for p in BAR['required_at']))
    print(f"\n  arms meeting the bar at BOTH {BAR['required_at']}: "
          f"{res['meets_bar_at_both'] or 'none'}")

    # ---------- (b) dose-response: does truncation-training help more as history shortens? ----------
    print(f'\n  (b) dose-response of (trained-truncated - trained-full), same design, same chrono')
    for p in [q for q in PREFIXES if q < 1.0]:
        row, vals = {}, []
        for dn in ('ref', 'r382', 'hybrid'):
            for ch in ('chrono3', 'chrono1'):
                r = contrast(p, f'{dn}_full_{ch}', f'{dn}_trunc_{ch}')
                row[f'{dn}_{ch}'] = r
                if dn != 'hybrid':          # hybrid is a recombination of the other two, not a 5th cell
                    vals.append(r['mean_delta'])
        row['mean_over_four_cells'] = round(float(np.mean(vals)), 4)
        res['dose_response'][str(p)] = row
        print(f"    prefix {p}: mean over the four cells {row['mean_over_four_cells']:+.4f}   "
              + '  '.join(f"{k.replace('_chrono', '/c')} {v['mean_delta']:+.4f}"
                          for k, v in row.items() if k != 'mean_over_four_cells' and 'hybrid' not in k))
    seq = [res['dose_response'][str(p)]['mean_over_four_cells'] for p in (.75, .6, .5)]
    res['dose_response_summary'] = {
        'means_at_075_06_05': seq, 'rule': DOSE,
        'monotone_increasing_as_history_shortens': bool(seq[0] <= seq[1] <= seq[2]),
        'shortest_exceeds_decisive': bool(seq[2] > seq[1])}
    print(f"    .75 / .6 / .5 = {seq[0]:+.4f} / {seq[1]:+.4f} / {seq[2]:+.4f}    "
          f"monotone: {res['dose_response_summary']['monotone_increasing_as_history_shortens']}   "
          f".5 > .6: {res['dose_response_summary']['shortest_exceeds_decisive']}")

    # ---------- (c) do the length-dependent chrono columns actually hurt? ----------
    print(f'\n  (c) (chrono1 - chrono3), same design, same training set -- E4\'s stated premise')
    for p in PREFIXES:
        row, vals = {}, []
        for dn in ('ref', 'r382'):
            for tr in ('full', 'trunc'):
                if p == 1.0 and tr == 'trunc':
                    continue
                r = contrast(p, f'{dn}_{tr}_chrono3', f'{dn}_{tr}_chrono1')
                row[f'{dn}_{tr}'] = r
                vals.append(r['mean_delta'])
        row['mean'] = round(float(np.mean(vals)), 4)
        res['chrono_contrast'][str(p)] = row
        print(f"    prefix {p}: mean {row['mean']:+.4f}   "
              + '  '.join(f"{k} {v['mean_delta']:+.4f}" for k, v in row.items() if k != 'mean'))

    res['reading'] = {
        'old_gate_reference': REF,
        'old_gate_passed_an_already_rejected_arm': 'r382_full_chrono3',
        'arms_meeting_the_shipping_bar_at_both_prefixes': res['meets_bar_at_both'],
        'dose_response_present': res['dose_response_summary']['shortest_exceeds_decisive'],
        'chrono1_helps_under_truncation': bool(res['chrono_contrast']['0.5']['mean'] > 0)}
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-length-shift-contrasts.json', res)
    check_protected()
    log('written', str(OUT / 'audit-length-shift-contrasts.json'))
    print(f"  ({res['seconds']} s)")
    print('DONE')


if __name__ == '__main__':
    main()
