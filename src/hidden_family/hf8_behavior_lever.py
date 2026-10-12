"""HF8: what is emitting `other_coordination` actually worth on the leaderboard?

Behavior MAP is a macro average over the THREE DISCLOSED families only (OBS 11.2): for family
f the ranking score is `risk_score` where `predicted_behavior == f` and 0 otherwise, and the
truth positives are the pairs whose true family is f.  `other_coordination` is not one of the
three, so it never has a component of its own.

That has a non-obvious consequence, and it is the whole operational story:

  * A true `other_coordination` pair is a positive in NO component.  Whatever disclosed family
    we predict for it, it enters that family's ranking with our (high) risk score as a FALSE
    positive, pushing genuine members of that family down.
  * Predicting `other_coordination` sets its score to 0 in all three rankings, so it sinks
    harmlessly to the bottom.

So emitting the string is worth exactly zero by itself, but it REMOVES a false positive from a
disclosed family's ranking.  The gain is real and strictly bounded by the behaviour weight.
The risk is symmetric: predicting `other_coordination` on a pair that IS disclosed-family
zeroes a true positive and costs more than it ever gains, because true members outnumber hidden
ones by ~10:1 at any plausible prevalence.

This script measures both directions at evaluation scale, sweeping prevalence (hf2) and
detector quality, so the decision rests on numbers rather than on the argument above.

Read-only.  No submission is written or modified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import ORIGINAL, CV, write_json

OUT = CV / 'experiments' / 'hidden_family_20260912'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
RNG = np.random.default_rng(20260912)
SUB = CV / 'outputs_multiview_20260911' / 'submission_M4_stage2_chrono3.csv'


def stable_ap(scores, truth, pair_ids):
    order = np.argsort(pair_ids, kind='mergesort')
    s, t = np.asarray(scores)[order], np.asarray(truth)[order]
    idx = np.argsort(-s, kind='mergesort')
    t = t[idx]
    if t.sum() == 0:
        return float('nan')
    tp = np.cumsum(t)
    return float(((tp / np.arange(1, len(t) + 1)) * t).sum() / t.sum())


def behavior_map(risk, pred, true_fam, ids):
    """Host recipe: macro AP over the three disclosed families."""
    aps = []
    for f in FAMS:
        s = np.where(pred == f, risk, 0.0)
        t = (true_fam == f).astype(int)
        aps.append(stable_ap(s, t, ids))
    return float(np.mean(aps)), aps


def main():
    sub = pl.read_csv(SUB, columns=['pair_id', 'risk_score', 'predicted_behavior'])
    ids = np.arange(sub.height)
    risk = sub['risk_score'].to_numpy()
    pred0 = sub['predicted_behavior'].to_numpy()
    print(f'submission {SUB.name}: {sub.height} rows')
    print('  predicted_behavior histogram: ' +
          str(dict(sub['predicted_behavior'].value_counts().iter_rows())))
    print(f'  rows with risk >= .5: {int((risk >= .5).sum())}')

    # ---- build a truth consistent with our own ranking, then vary the hidden share ----
    N = sub.height
    n_pos = 930
    top = np.argsort(-risk, kind='mergesort')[:int(n_pos * 1.12)]   # positives live near the top
    print()
    print('=' * 96)
    print('BEHAVIOR MAP UNDER A HIDDEN FAMILY WE DO OR DO NOT DECLARE')
    print('=' * 96)
    print(f'{"pi":>6}{"n_hidden":>10}{"policy":>34}{"BehaviorMAP":>13}{"composite":>12}')
    rows = []
    for pi in [0.02, 0.05, 0.10, 0.20]:
        base_b = decl_b = mis_b = None
        reps = []
        for _ in range(8):
            pick = RNG.choice(top, size=n_pos, replace=False)
            true_fam = np.array(['none'] * N, dtype=object)
            n_h = int(round(n_pos * pi))
            true_fam[pick[:n_h]] = 'other_coordination'
            rest = pick[n_h:]
            # disclosed truth follows our own predicted mix, so the model is well calibrated
            true_fam[rest] = pred0[rest]
            tf = np.array([str(x) for x in true_fam])

            b0, _ = behavior_map(risk, pred0, tf, ids)          # declare nothing
            pred1 = pred0.copy()
            pred1[pick[:n_h]] = 'other_coordination'            # perfect detection
            b1, _ = behavior_map(risk, pred1, tf, ids)
            pred2 = pred0.copy()                                # detector with 50% precision:
            wrong = RNG.choice(rest, size=n_h, replace=False)   # as many true members mislabelled
            pred2[pick[:n_h]] = 'other_coordination'
            pred2[wrong] = 'other_coordination'
            b2, _ = behavior_map(risk, pred2, tf, ids)
            reps.append((b0, b1, b2))
        b0, b1, b2 = np.mean(reps, axis=0)
        for lab, b in [('declare nothing (current)', b0),
                       ('perfect other_coordination', b1),
                       ('detector at 50% precision', b2)]:
            print(f'{pi:>6.2f}{int(n_pos * pi):>10}{lab:>34}{b:>13.4f}{0.1 * b:>12.4f}')
            rows.append({'pi': pi, 'policy': lab, 'behavior_map': float(b),
                         'composite_contribution': float(0.1 * b)})
        print(f'{"":>6}{"":>10}{"gain from perfect detection":>34}'
              f'{b1 - b0:>+13.4f}{0.1 * (b1 - b0):>+12.4f}')
        print(f'{"":>6}{"":>10}{"gain at 50% precision":>34}'
              f'{b2 - b0:>+13.4f}{0.1 * (b2 - b0):>+12.4f}')
        print()

    print('=' * 96)
    print('CEILING')
    print('=' * 96)
    print('  measured public Behavior MAP = .976, so the entire behaviour component has')
    print('  0.1 x (1 - .976) = +0.0024 composite of headroom left, hidden family included.')
    print('  Nothing in the behaviour column can change a rank that 0.0024 does not change.')

    write_json(OUT / 'hf8-behavior-lever.json',
               {'submission': str(SUB), 'rows': rows,
                'public_behavior_map': 0.976,
                'total_headroom_composite': 0.1 * (1 - 0.976)})
    print()
    print(f'written {OUT / "hf8-behavior-lever.json"}')


if __name__ == '__main__':
    main()
