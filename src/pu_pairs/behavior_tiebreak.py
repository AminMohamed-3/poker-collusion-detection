"""Step 28: behaviour-list tie-break. The official behaviour AP ranks each family's candidates by risk_score, so
the positives (all at risk ~1) are ordered almost arbitrarily relative to each other; a positive whose family was
mis-assigned sits as a false positive anywhere in the top block. Perturbing risk by -eps*(1-confidence) with a
tiny eps orders the top block by family confidence while leaving the pair-AP ordering untouched wherever risk
gaps exceed eps. Measured on the fold-1/fold-2 all-pairs scores (exact parity with the production fold models):
behaviour MAP (macro, official formula) and pair AP, labelled pairs and all pairs (unlabelled as negatives).
Output: Claude Version/experiments/pu_pairs_20260911/behavior-tiebreak.json
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
import numpy as np
import polars as pl

EXP = ORIGINAL / 'Claude Version' / 'experiments' / 'pu_pairs_20260911'
FAM = {1: 'directed_transfer', 2: 'soft_play', 3: 'coordinated_isolation'}


def ap(scores, truth, pair_ids):
    order = np.lexsort((pair_ids, -scores))  # ties -> ascending pair_id, as the host's mergesort over pair_id-sorted rows
    t = truth[order]
    if t.sum() == 0:
        return float('nan')
    hits = np.cumsum(t)
    return float((hits[t == 1] / (np.flatnonzero(t) + 1)).mean())


def behavior_map(df, risk_col):
    out = {}
    pid = df['pair_id'].to_numpy()
    for k, fam in FAM.items():
        s = np.where(df['predicted_class'].to_numpy() == k, df[risk_col].to_numpy(), 0.0)
        truth = (df['behavior_family'] == fam).to_numpy().astype(int)
        out[fam] = ap(s, truth, pid)
    out['macro'] = float(np.mean([out[f] for f in FAM.values()]))
    return out


def main():
    res = {}
    for fold in (1, 2):
        s = pl.read_parquet(EXP / f'fold{fold}' / 'scores.parquet').filter(pl.col('shared_hands') >= 38)
        s = s.with_columns((pl.max_horizontal('p1', 'p2', 'p3') / (pl.col('p1') + pl.col('p2') + pl.col('p3')).clip(lower_bound=1e-12)).alias('conf'))
        s = s.with_columns(pl.col('behavior_family').fill_null('none'))
        r = {}
        for eps in [0.0, 1e-4, 1e-3, 1e-2]:
            s = s.with_columns((pl.col('risk') - eps * (1 - pl.col('conf'))).clip(lower_bound=0.0).alias('risk_adj'))
            lab = s.filter(pl.col('label') >= 0)
            pos = (lab['label'] == 1).to_numpy().astype(int)
            allp = s.with_columns(pl.when(pl.col('label') == 1).then(1).otherwise(0).alias('y'))
            r[str(eps)] = {'behavior_map_labelled': behavior_map(lab, 'risk_adj'), 'behavior_map_all_pairs_unlabelled_as_negative': behavior_map(allp.with_columns(pl.when(pl.col('label') == 1).then(pl.col('behavior_family')).otherwise(pl.lit('none')).alias('behavior_family')), 'risk_adj'),
                           'pair_ap_labelled': ap(lab['risk_adj'].to_numpy(), pos, lab['pair_id'].to_numpy()),
                           'pair_ap_all_unlabelled_as_negative': ap(allp['risk_adj'].to_numpy(), allp['y'].to_numpy(), allp['pair_id'].to_numpy())}
        # reference: family probability itself as the family score (changes pair AP; not submittable as-is)
        lab = s.filter(pl.col('label') >= 0)
        ref = {}
        for k, fam in FAM.items():
            sc = np.where(lab['predicted_class'].to_numpy() == k, lab[f'p{k}'].to_numpy(), 0.0)
            ref[fam] = ap(sc, (lab['behavior_family'] == fam).to_numpy().astype(int), lab['pair_id'].to_numpy())
        ref['macro'] = float(np.mean([ref[f] for f in FAM.values()]))
        r['reference_family_probability_as_score'] = ref
        pos = s.filter(pl.col('label') == 1)
        r['positives'] = {'n': pos.height, 'misassigned': int((pos['predicted_class'].map_elements(lambda k: FAM.get(int(k), 'none'), return_dtype=pl.Utf8) != pos['behavior_family']).sum()),
                          'conf_q10_50': [float(v) for v in np.quantile(pos['conf'].to_numpy(), [.1, .5])], 'risk_q10_50': [float(v) for v in np.quantile(pos['risk'].to_numpy(), [.1, .5])]}
        res[f'fold{fold}'] = r
        print(f'fold{fold}', json.dumps(r, indent=None, default=str)[:3000], flush=True)
    write_json(EXP / 'behavior-tiebreak.json', res)


if __name__ == '__main__':
    main()
