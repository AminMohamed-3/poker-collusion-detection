"""Step 65g: training-only audit ablation for the production stage-2 feature variants (step 65f).
Question: does the all-family top-20 stage-2 need the absolute chronology columns (chrono_rank, n_hands), which shift
under the shorter evaluation histories, or is chrono_frac alone as good? Same protocol as step 65b F1t (per family,
GroupKFold(5, by table), 3 seeds, step-14 CLF, top-20 stage-1 candidates, tail keeps stage-1 order), 162 audit queries.
  F1t_chrono3          struct + chrono_rank, chrono_frac, n_hands + oof, oof_rank   (held-confirmed design)
  F1t_chrono1          struct + chrono_frac + oof, oof_rank
  F1t_chrono3_shift65  chrono3 models applied to rows whose chrono_rank and n_hands are scaled by .65 at prediction
                       time (a crude stand-in for the evaluation length shift; chrono_frac unchanged)
Output: experiments/label_process_20260911/audit-chrono-variants.json"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import write_json
from evidence_aug.frame import FAMILIES
from evidence_aug.step14_audit_truncation import OUT, TOP, CLF
from evidence_aug.step65_structural_evidence import summarize, SEEDS, log
from evidence_aug.step65b_family_stage2 import boot
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


def fam_oof_shift(d, cols, y, groups, shift):
    pred = np.full(d.height, -1e6)
    for fam in FAMILIES:
        mask = ((d['behavior_family'] == fam) & (d['oof_rank'] <= TOP)).to_numpy()
        X = d.select(cols).to_numpy().astype(np.float64)[mask]
        Xs = X.copy()
        for c, factor in shift.items():
            Xs[:, cols.index(c)] *= factor
        p = np.zeros(mask.sum())
        for seed in SEEDS:
            for tr, te in GroupKFold(n_splits=5).split(X, y[mask], groups[mask]):
                clf = lgb.LGBMClassifier(**CLF, random_state=seed); clf.fit(X[tr], y[mask][tr]); p[te] += clf.predict_proba(Xs[te])[:, 1] / len(SEEDS)
        pred[mask] = p
    return pred


def main():
    t0 = time.time()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet'); feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]; sc = ['oof', 'oof_rank']
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows()); ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows()); pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float); groups = d['table_id'].to_numpy(); key = d.select('pair_id', 'hand_id', 'table_id')
    cands_mask = (d['oof_rank'] <= TOP).to_numpy()
    specs = {'F1t_chrono3': (struct + ['chrono_rank', 'chrono_frac', 'n_hands'] + sc, {}), 'F1t_chrono1': (struct + ['chrono_frac'] + sc, {}),
             'F1t_chrono3_shift65': (struct + ['chrono_rank', 'chrono_frac', 'n_hands'] + sc, {'chrono_rank': .65, 'n_hands': .65}),
             'F1t_chrono3_shift40': (struct + ['chrono_rank', 'chrono_frac', 'n_hands'] + sc, {'chrono_rank': .40, 'n_hands': .40})}
    res = {'training_only': True, 'queries': len(truths), 'sets': {}}; maps = {}
    for name, (cols, shift) in specs.items():
        pred = fam_oof_shift(d, cols, y, groups, shift)
        pred = np.where(cands_mask, pred + 10.0, d['oof'].to_numpy() - 1e6)
        s, m = summarize(key.with_columns(pl.Series('score', pred)), truths, labels); res['sets'][name] = s; maps[name] = m
        log(name, json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}}))
    res['paired'] = {}
    for a, b in [('F1t_chrono3', 'F1t_chrono1'), ('F1t_chrono3', 'F1t_chrono3_shift65'), ('F1t_chrono3', 'F1t_chrono3_shift40'), ('F1t_chrono1', 'F1t_chrono3_shift65')]:
        res['paired'][f'{b} minus {a}'] = boot(maps[a], maps[b], ptab, pools)
        log(f'{b} minus {a}', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res['paired'][f'{b} minus {a}'].items()}))
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-chrono-variants.json', res)


if __name__ == '__main__':
    main()
