"""STEP193b: score a candidate listing rule on the 23 withheld coordinated_isolation pairs.

Every labelled pair lists EXACTLY 5 hands, and a rule returns exactly 5, so precision@5 and
recall@5 are the same number -- which is why STEP115's best CI rule reads .27/.27 rather than two
different figures. The headline is therefore hits/5 averaged over pairs. MAP@5 is reported too
because that is what the competition actually scores.

The rule under test is a python file exposing:

    def rank_hands(pair: dict) -> list[int]
        # pair is the parsed pair_<id>.json; return up to 5 hand_index values, best first

Usage: python step193b_C_score_rule.py PACKETDIR KEYFILE (RULE.py | baseline:NAME)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BASELINES = ['random', 'first5', 'last5', 'biggest_pot', 'biggest_pair_contribution',
             'most_folds_by_others']


def baseline(name, pair, rng):
    hs = pair['hands']
    if name == 'random':
        return list(rng.choice(len(hs), size=min(5, len(hs)), replace=False))
    if name == 'first5':
        return list(range(min(5, len(hs))))
    if name == 'last5':
        return list(range(len(hs)))[-5:][::-1]
    if name == 'biggest_pot':
        key = [-(h['final_pot'] or 0) for h in hs]
    elif name == 'biggest_pair_contribution':
        key = [-sum(s['total_contribution'] or 0 for s in h['seats'] if s['who'] in ('P1', 'P2'))
               for h in hs]
    elif name == 'most_folds_by_others':
        key = [-sum(1 for s in h['seats'] if s['who'].startswith('O') and s['folded']) for h in hs]
    else:
        raise SystemExit(f'unknown baseline {name}; choose from {BASELINES}')
    return list(np.argsort(np.asarray(key, float), kind='mergesort')[:5])


def ap_at_k(pred, truth, k=5):
    hits, s = 0, 0.0
    for i, p in enumerate(pred[:k]):
        if p in truth:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(truth), k) if truth else 0.0


def main():
    pdir, keyfile, rulearg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    key = json.loads(keyfile.read_text(encoding='utf-8'))
    held = key['held_out']
    rng = np.random.default_rng(193)

    fn = None
    if not rulearg.startswith('baseline:'):
        spec = importlib.util.spec_from_file_location('candidate_rule', rulearg)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.rank_hands

    rows = []
    for pid in held:
        pair = json.loads((pdir / f'pair_{pid}.json').read_text(encoding='utf-8'))
        assert pair['split'] == 'WITHHELD' and pair['n_listed'] is None
        # hand_index is positional in the chronological list, and the key is keyed by hand_id; the
        # packet never carries hand_id, so map through position the same way the packet was built.
        listed_ids = set(key['listed_by_pair'][pid])
        # positions of the listed hands: the key preserves chronological order of the pair's hands
        order = key.get('hand_order', {}).get(pid)
        assert order is not None, 'answer key is missing hand_order; regenerate with step193'
        truth = {i for i, h in enumerate(order) if h in listed_ids}
        # coordinated_isolation lists exactly 5 everywhere, so precision@5 and recall coincide and
        # a single number is honest. directed_transfer and soft_play also list 3 and 4, so the two
        # must be reported separately for those families.
        assert 3 <= len(truth) <= 5, f'{pid}: expected 3-5 listed, got {len(truth)}'
        pred = [int(x) for x in (baseline(rulearg.split(':', 1)[1], pair, rng) if fn is None
                                 else fn(pair))]
        assert len(pred) == len(set(pred)) and len(pred) <= 5, f'{pid}: rule returned a bad list'
        assert all(0 <= p < pair['n_hands'] for p in pred), f'{pid}: index out of range'
        rows.append((pid, len(set(pred) & truth), ap_at_k(pred, truth), len(truth), len(pred)))

    hits = np.array([r[1] for r in rows], float)
    aps = np.array([r[2] for r in rows], float)
    ntruth = np.array([r[3] for r in rows], float)
    npred = np.array([r[4] for r in rows], float)
    print(f'rule            : {rulearg}')
    print(f'pairs scored    : {len(rows)}  (withheld, never used to induce)')
    if (ntruth == 5).all():
        print(f'precision@5=recall@5 : {hits.mean()/5:.4f}')
    else:
        print(f'precision@5          : {(hits/np.maximum(npred,1)).mean():.4f}')
        print(f'recall               : {(hits/ntruth).mean():.4f}   '
              f'(list lengths {int(ntruth.min())}-{int(ntruth.max())})')
    print(f'MAP@5                : {aps.mean():.4f}')
    print(f'pairs with >=1 hit   : {int((hits>0).sum())}/{len(rows)}')
    print(f'hit histogram        : ' + str({int(k): int((hits == k).sum()) for k in range(6)}))


if __name__ == '__main__':
    main()
