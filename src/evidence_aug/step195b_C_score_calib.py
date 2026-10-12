"""STEP195b: score the annotation-calibration phase, and gate the extension phase on it.

Each calibration packet shows the annotators' top-3 evidence hands for a pair and hides ranks 4
and 5. The model names 2 more from the remaining candidates. The truth is known.

THE BAR IS NOT RANDOM, IT IS PRODUCTION. Our stage-1 already ranks these candidates; its own next
two picks are the competitor. If the model merely matches production, its labels tell the stage-2
learner nothing it cannot already derive -- that is precisely how STEP137's pseudo-labels came out
NULL. New labels are only worth training on if the judgement behind them beats the model that
would otherwise generate them.

Usage: python step195b_C_score_calib.py PACKETDIR KEYFILE (ANSWERS.json | baseline:production
                                                            | baseline:random | baseline:chrono)
ANSWERS.json maps pair_id -> list of hand_index values, best first.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    pdir, keyfile, arg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    key = json.loads(keyfile.read_text(encoding='utf-8'))['hidden_by_pair']
    rng = np.random.default_rng(1951)

    answers = None
    if not arg.startswith('baseline:'):
        answers = json.loads(Path(arg).read_text(encoding='utf-8'))
        answers = answers.get('answers', answers)

    rows, skipped = [], 0
    for pid, k in key.items():
        pkt = pdir / f'pair_{pid}.json'
        if not pkt.exists():
            continue
        d = json.loads(pkt.read_text(encoding='utf-8'))
        order = k['hand_order']
        pos = {h: i for i, h in enumerate(order)}
        truth = {pos[h] for h in k['hidden']}
        offered = [pos[h] for h in k['candidates_offered']]
        assert truth <= set(offered), f'{pid}: hidden hand not among the offered candidates'

        if answers is not None:
            if pid not in answers:
                skipped += 1
                continue
            pred = [int(x) for x in answers[pid]][:2]
        elif arg == 'baseline:production':
            pred = sorted(offered, key=lambda i: k['s1_rank'][order[i]])[:2]
        elif arg == 'baseline:random':
            pred = list(rng.choice(offered, size=2, replace=False))
        elif arg == 'baseline:chrono':
            pred = sorted(offered)[:2]
        else:
            raise SystemExit(f'unknown baseline {arg}')
        pred = [int(x) for x in pred]
        assert len(set(pred)) == len(pred), f'{pid}: duplicate pick'
        assert all(p in set(offered) for p in pred), f'{pid}: picked a hand that was not offered'
        rows.append((pid, len(set(pred) & truth), len(offered)))

    hits = np.array([r[1] for r in rows], float)
    noff = np.array([r[2] for r in rows], float)
    n = len(rows)
    acc = hits.mean() / 2
    se = (hits / 2).std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    chance = (2.0 / noff).mean()
    print(f'source          : {arg}')
    print(f'pairs scored    : {n}' + (f'   ({skipped} missing from answers)' if skipped else ''))
    print(f'mean candidates offered : {noff.mean():.1f}  -> chance per pick {chance:.4f}')
    print(f'recovery of the 2 hidden: {acc:.4f} +- {se:.4f}   ({int(hits.sum())}/{2*n} hands)')
    print(f'pairs with >=1 recovered: {int((hits > 0).sum())}/{n}')
    print(f'hit histogram           : ' + str({int(k_): int((hits == k_).sum()) for k_ in (0, 1, 2)}))
    print(f'lift over chance        : {acc/chance:.2f}x')


if __name__ == '__main__':
    main()
