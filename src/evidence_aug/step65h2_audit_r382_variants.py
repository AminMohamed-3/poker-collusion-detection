"""Step 65h2: training-only follow-up to step 65h. The per-family stage-2 with the 382 stage-1 ranker features added (r382)
gained +.029 [+.007, +.051] on the audit set. Which parts carry it, and is it stable under recipe changes?
  ref             struct + chrono3 + oof, oof_rank                    (F1t_chrono3)
  r382            ref + 382 ranker features                            (step 65h)
  r382_nostruct   chrono3 + oof, oof_rank + 382 (no structural features)
  r382_only       382 + oof, oof_rank (no struct, no chrono)
  r382_ff50       r382 with feature_fraction .5 (column subsampling)
  r382_clf2       r382 with the larger step-65b CLF2 recipe
  r382_5seeds     r382 with 5 seeds (11..15) instead of 3
  r382_pooled     one pooled model over all families (family columns added) with the r382 feature set, top-20
Output: experiments/label_process_20260911/audit-r382-variants.json"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import write_json
from evidence_aug.frame import FAMILIES
from evidence_aug.fold import load_fold
from evidence_aug.step14_audit_truncation import OUT, TOP, CLF
from evidence_aug.step65_structural_evidence import summarize, log
from evidence_aug.step65b_family_stage2 import boot, CLF2
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def fam_oof(d, cols, y, groups, params, seeds, pooled=False):
    pred = np.full(d.height, -1e6)
    fams = [None] if pooled else FAMILIES
    for fam in fams:
        mask = (d['oof_rank'] <= TOP).to_numpy() if fam is None else ((d['behavior_family'] == fam) & (d['oof_rank'] <= TOP)).to_numpy()
        X = d.select(cols).to_numpy().astype(np.float64)[mask]; ys = y[mask]; gs = groups[mask]
        p = np.zeros(len(ys))
        for seed in seeds:
            for tr, te in GroupKFold(n_splits=5).split(X, ys, gs):
                clf = lgb.LGBMClassifier(**params, random_state=seed); clf.fit(X[tr], ys[tr]); p[te] += clf.predict_proba(X[te])[:, 1] / len(seeds)
        pred[mask] = p
    return pred


def main():
    t0 = time.time()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet'); feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]; sc = ['oof', 'oof_rank']; chrono = ['chrono_rank', 'chrono_frac', 'n_hands']
    data, features, _, _ = load_fold(0)
    rcols = [f for f in features if not f.startswith('family_') and f in data.columns and f not in struct + chrono + sc]  # 'pot_bb' exists in both sets; the structural copy is kept (as in step 65h)
    d = d.join(data.select(['pair_id', 'hand_id'] + rcols).with_columns(pl.lit(1).alias('_hit')), on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert d['_hit'].null_count() == 0
    d = d.drop('_hit').with_columns(pl.col(rcols).fill_null(-999.0))
    d = d.with_columns(*[(pl.col('behavior_family') == f).cast(pl.Float64).alias('family_' + f) for f in FAMILIES]); famcols = ['family_' + f for f in FAMILIES]
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows()); ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows()); pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float); groups = d['table_id'].to_numpy(); key = d.select('pair_id', 'hand_id', 'table_id')
    S3 = (11, 12, 13); S5 = (11, 12, 13, 14, 15)
    ref = struct + chrono + sc; r382 = ref + rcols
    specs = {'ref': (ref, CLF, S3, False), 'r382': (r382, CLF, S3, False), 'r382_nostruct': (chrono + sc + rcols, CLF, S3, False), 'r382_only': (sc + rcols, CLF, S3, False),
             'r382_ff50': (r382, dict(CLF, colsample_bytree=.5), S3, False), 'r382_clf2': (r382, CLF2, S3, False), 'r382_5seeds': (r382, CLF, S5, False), 'r382_pooled': (r382 + famcols, CLF, S3, True)}
    res = {'training_only': True, 'queries': len(truths), 'n_ranker_features': len(rcols), 'sets': {}, 'paired_vs_ref': {}, 'paired_vs_r382': {}}; maps = {}
    cands = (d['oof_rank'] <= TOP).to_numpy()
    for name, (cols, params, seeds, pooled) in specs.items():
        t1 = time.time()
        pred = fam_oof(d, cols, y, groups, params, seeds, pooled)
        pred = np.where(cands, pred + 10.0, d['oof'].to_numpy() - 1e6)
        s, m = summarize(key.with_columns(pl.Series('score', pred)), truths, labels); maps[name] = m
        s['auc_within_top20'] = {fam: round(float(roc_auc_score(y[cands & (d['behavior_family'] == fam).to_numpy()], pred[cands & (d['behavior_family'] == fam).to_numpy()])), 4) for fam in FAMILIES}
        s['seconds'] = round(time.time() - t1); res['sets'][name] = s
        log(name, json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}, 'auc20': s['auc_within_top20'], 'sec': s['seconds']}))
    for name in specs:
        for base, store in (('ref', 'paired_vs_ref'), ('r382', 'paired_vs_r382')):
            if name == base:
                continue
            b = boot(maps[base], maps[name], ptab, pools)
            b['by_family'] = {f: round(float(np.mean([maps[name][p] - maps[base][p] for p in maps[base] if labels[p] == f])), 4) for f in FAMILIES}
            res[store][name] = b
            log(f'{name} minus {base}', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}))
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-r382-variants.json', res)
    log('done', res['seconds'])


if __name__ == '__main__':
    main()
