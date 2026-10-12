"""HF1: how a family the model has never been trained on behaves inside our production system.

`other_coordination` is, by construction, a positive mechanism with zero training labels.
The only honest surrogate available is leave-one-known-family-out (LOFO): the three
`omit_*` confirmation runs of Codex hold out the SAME fold-0 pairs as the all-family
`conditional-action-experiment` screen, so for every disclosed family f the same held
positives can be compared with f IN training (screen) and with f OUT of training (omit_f).

Read-only: consumes `reports/conditional-action-confirmation/results/*` and
`data/development_labels.csv`. Writes only into experiments/hidden_family_20260912/.

No held EVIDENCE truth is read here; evidence numbers are copied from the already-published
metrics.json of the same runs (they were evaluated once, by Codex, in September).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import ORIGINAL, INVERSE, write_json, sha

OUT = Path(__file__).resolve().parents[2] / 'experiments' / 'hidden_family_20260912'
CONF = ORIGINAL / 'reports' / 'conditional-action-confirmation' / 'results'
FAMILIES = ['directed_transfer', 'soft_play', 'coordinated_isolation']


def stable_ap(scores, truth, pair_ids):
    """Host recipe: sort by pair_id, then argsort(-score, kind='mergesort')."""
    order = np.argsort(pair_ids, kind='mergesort')
    s, y = np.asarray(scores)[order], np.asarray(truth)[order]
    idx = np.argsort(-s, kind='mergesort')
    y = y[idx]
    if y.sum() == 0:
        return float('nan')
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


SCREEN_PRED = ORIGINAL / 'reports' / 'conditional-action-experiment' / 'screen-results' / 'pair-predictions.csv'
SCREEN_CLASS = ORIGINAL / 'reports' / 'conditional-behavior-experiment' / 'fold0' / 'pair-predictions.csv'


def load_run(job):
    p = CONF / job / 'pair-predictions.csv'
    return pl.read_csv(p).filter(pl.col('candidate') == 'conditional'), sha(p)


def main():
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    runs, hashes = {}, {}
    for fam in FAMILIES:
        runs[fam], hashes[f'omit_{fam}'] = load_run(f'omit_{fam}')

    # trained (all three families in training) predictions for the same fold-0 held pairs
    screen = pl.read_csv(SCREEN_PRED).filter(pl.col('candidate') == 'conditional')
    hashes['screen_pair_predictions'] = sha(SCREEN_PRED)
    trained_class = pl.read_csv(SCREEN_CLASS).select(
        'pair_id', pl.col('conditional_class').alias('predicted_class'))
    hashes['screen_behavior_classes'] = sha(SCREEN_CLASS)

    held_ids = sorted(screen.filter(pl.col('history') == 'full')['pair_id'].to_list())
    held = labels.filter(pl.col('pair_id').is_in(held_ids))
    assert held.height == len(held_ids), (held.height, len(held_ids))
    # every omit run must be a subset of the same held set (its family + the negatives)
    for fam in FAMILIES:
        assert set(runs[fam]['pair_id']) <= set(held_ids)

    counts = dict(held['behavior_family'].value_counts().iter_rows())
    report = {'held_pairs': len(held_ids), 'held_composition': counts, 'input_sha256': hashes, 'families': {}}
    print(f'held population: {len(held_ids)} pairs  {counts}')

    for history in ['full', 'shortened']:
        print(f'\n{"="*96}\nHISTORY = {history}\n{"="*96}')
        for fam in FAMILIES:
            pos = held.filter(pl.col('behavior_family') == fam)['pair_id'].to_list()
            neg = held.filter(pl.col('label') == 0)['pair_id'].to_list()

            # f NOT in training: the omit_f run (its file holds exactly f's positives + the negatives)
            omitted = runs[fam].filter(pl.col('history') == history).select(
                'pair_id', 'risk', 'predicted_class')
            # f IN training: the all-family screen, restricted to the same pairs
            trained_fr = (screen.filter(pl.col('history') == history)
                          .filter(pl.col('pair_id').is_in(pos + neg))
                          .join(trained_class, on='pair_id', how='left')
                          .select('pair_id', 'risk', 'predicted_class'))
            assert set(omitted['pair_id']) == set(pos) | set(neg)
            assert set(trained_fr['pair_id']) == set(pos) | set(neg)

            def stats(fr):
                d = dict(zip(fr['pair_id'].to_list(), fr['risk'].to_list()))
                c = dict(zip(fr['pair_id'].to_list(), fr['predicted_class'].to_list()))
                rp = np.array([d[p] for p in pos])
                rn = np.array([d[p] for p in neg])
                ap = stable_ap(np.concatenate([rp, rn]),
                               np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]),
                               np.array(pos + neg))
                routing = {}
                for p in pos:
                    routing[INVERSE[c[p]]] = routing.get(INVERSE[c[p]], 0) + 1
                # where do the positives sit in the WHOLE held population by rank
                allr = np.array(list(d.values()))
                pct = np.array([(allr < d[p]).mean() for p in pos])
                return dict(ap=ap, median_risk=float(np.median(rp)),
                            q25=float(np.quantile(rp, .25)), q75=float(np.quantile(rp, .75)),
                            frac_ge_50=float((rp >= .5).mean()), frac_ge_90=float((rp >= .9).mean()),
                            frac_lt_10=float((rp < .1).mean()),
                            median_pct=float(np.median(pct)), routing=routing,
                            median_neg_risk=float(np.median(rn)))

            so = stats(omitted)
            st_mean = stats(trained_fr)
            merged_routing = st_mean['routing']

            report['families'].setdefault(history, {})[fam] = {
                'n_positives': len(pos), 'n_negatives': len(neg),
                'omitted': so, 'trained': st_mean}

            print(f'\n-- {fam}  (n_pos={len(pos)}, n_neg={len(neg)})')
            print(f'   {"":<22}{"TRAINED":>12}{"UNSEEN":>12}   delta')
            for k, lab in [('ap', 'pair AP vs neg'), ('median_risk', 'median risk'),
                           ('frac_ge_90', 'share risk>=.90'), ('frac_ge_50', 'share risk>=.50'),
                           ('frac_lt_10', 'share risk<.10'), ('median_pct', 'median percentile')]:
                print(f'   {lab:<22}{st_mean[k]:>12.4f}{so[k]:>12.4f}   {so[k]-st_mean[k]:+.4f}')
            print(f'   routing when TRAINED : {merged_routing}')
            print(f'   routing when UNSEEN  : {so["routing"]}')

    write_json(OUT / 'hf1-unseen-family-profile.json', report)
    print(f'\nwritten {OUT / "hf1-unseen-family-profile.json"}')


if __name__ == '__main__':
    main()
