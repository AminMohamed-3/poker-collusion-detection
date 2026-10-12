"""Describe incumbent errors using the already-reused audit only; no model fit."""
from audit_common import *
from sklearn.metrics import roc_auc_score

OUT = HERE / 'experiments/error_diagnosis_20260912'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'diagnosis.json').exists()
    d, shared, extra, saved = load_current()
    d = d.with_columns(pl.Series('s2', saved['incumbent'].to_numpy())).with_columns(pl.col('s2').rank('ordinal', descending=True).over('pair_id').alias('s2_rank'))
    rows = []
    for (pid,), q in d.group_by('pair_id', maintain_order=True):
        y = q['relevance'].to_numpy()
        s2 = q['s2'].to_numpy()
        in20 = q['oof_rank'].to_numpy() <= 20
        hits = int(((q['s2_rank'].to_numpy() <= 5) & (y == 1)).sum())
        pos = q.filter((pl.col('relevance') == 1) & (pl.col('oof_rank') <= 20))
        neg = q.filter((pl.col('relevance') == 0) & (pl.col('oof_rank') <= 20))
        wrong = 0
        near = 0
        ncomps = pos.height * neg.height
        if ncomps:
            margins = pos['s2'].to_numpy()[:, None] - neg['s2'].to_numpy()[None, :]
            wrong = int((margins < 0).sum())
            near = int(((margins < 0) & (margins >= -.1)).sum())
        rows.append({'pair_id': pid, 'table_id': q['table_id'][0], 'family': q['behavior_family'][0], 'n': q.height,
                     'listed': int(y.sum()), 'listed_top20': pos.height, 'hits5': hits,
                     'auc_top20': float(roc_auc_score(y[in20], s2[in20])) if ncomps else None,
                     'comparisons': ncomps, 'inversions': wrong, 'near_inversions': near})
    f = pl.DataFrame(rows)
    f.write_parquet(OUT / 'queries.parquet')
    res = {'description': 'Exploratory error diagnosis on reused audit, current full hybrid.',
           'queries': len(rows), 'families': {}, 'source_sha256': sha(PREV / 'scores.parquet')}
    for fam in FAMILIES:
        q = f.filter(pl.col('family') == fam)
        total = int(q['listed'].sum())
        res['families'][fam] = {'queries': q.height, 'listed_hands': total,
            'listed_recall20': float(q['listed_top20'].sum() / total),
            'listed_recall5': float(q['hits5'].sum() / total),
            'mean_within_pair_auc20': float(q['auc_top20'].mean()),
            'inversions': int(q['inversions'].sum()), 'comparisons': int(q['comparisons'].sum()),
            'fraction_inversions_with_probability_gap_below_point1': float(q['near_inversions'].sum() / max(1, q['inversions'].sum())),
            'perfect_top5_queries': int((q['hits5'] == q['listed']).sum())}
    write(OUT / 'diagnosis.json', res)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == '__main__':
    main()
