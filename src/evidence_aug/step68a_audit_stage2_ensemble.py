"""Step 68a (plan E1 + E2 + E3): one training-only audit of every remaining stage-2 recipe.

E1  ensembling            ref, r382, r382_ff50, bag_ref, bag_r382, ens_rank, ens_prob, ens_bag
E2  isolation + scope     top-K in {15, 20, 25, 30} on ref; ci_reg, ci_struct_only, ci_mono, ci_inter
E3  LambdaRank stage 2    rank_ref, rank_r382 (and rank_ci spliced into ref)

The audit set is the 162 positive pairs in the 172 tables outside every held fold; labels here are
training-only, so nothing spent from the held budget.  Whatever clears the pre-registered gate
goes into ONE held freeze afterwards (step 65i/65j pattern), which is the only fold-0-2 read this
study gets.

Gates (pre-registered here, before the numbers are printed):
  E1  pooled mean delta vs ref >= +.008 with bootstrap CI lower bound > 0, and coordinated
      isolation >= -.005 for the variant taken forward.
  E2  CI-family variants: gain vs ref in >= 4 of the 5 OOF splits AND pooled CI lower bound > 0.
  E3  same as E1.
The CI branch has extra weight this round: step 69's probe asks whether the CI stage 2 is a net
loss on evaluation, so a CI recipe that survives the audit is the replacement if it is.

Output: experiments/label_process_20260911/audit-stage2-ensemble.json
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from evidence_aug.fold import load_fold
from evidence_aug.frame import FAMILIES
from evidence_aug.paths import write_json
from evidence_aug.step14_audit_truncation import CLF, OUT, TOP
from evidence_aug.step65_structural_evidence import log, summarize
from evidence_aug.step65b_family_stage2 import boot

S3 = (11, 12, 13)
S5 = (11, 12, 13, 14, 15)
BAG = dict(CLF, subsample=.8, subsample_freq=1, colsample_bytree=.5)
CI = 'coordinated_isolation'
GATE = {'E1_pooled_mean_delta_ge': .008, 'ci_family_floor': -.005, 'ci95_lower_gt': 0,
        'E2_splits_better_ge': 4, 'note': 'written before any number in this file was computed'}


def group_sizes(pid):
    _, idx, cnt = np.unique(pid, return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


def fit_fam(X, ys, gs, pid, params, seeds, ranker):
    """5-fold GroupKFold OOF over `seeds`; returns predictions and the split index of every row."""
    p = np.zeros(len(ys))
    split = np.full(len(ys), -1)
    for seed in seeds:
        for si, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, ys, gs)):
            if ranker:
                mdl = lgb.LGBMRanker(objective='lambdarank', random_state=seed, eval_at=[5], **params)
                mdl.fit(X[tr], ys[tr].astype(int), group=group_sizes(pid[tr]))
                z = mdl.predict(X[te])
                z = (z - z.mean()) / (z.std() + 1e-9)          # comparable across folds
                p[te] += z / len(seeds)
            else:
                clf = lgb.LGBMClassifier(**params, random_state=seed)
                clf.fit(X[tr], ys[tr])
                p[te] += clf.predict_proba(X[te])[:, 1] / len(seeds)
            split[te] = si
    return p, split


def run(d, y, groups, per_fam, topk=TOP):
    """per_fam: family -> dict(cols, params, seeds, ranker).  Returns raw predictions + split ids."""
    pred = np.full(d.height, -1e6)
    split = np.full(d.height, -1)
    fam_col = d['behavior_family'].to_numpy()
    rank_col = d['oof_rank'].to_numpy()
    pid_col = d['pair_id'].to_numpy()
    for fam in FAMILIES:
        cfg = per_fam[fam]
        mask = (fam_col == fam) & (rank_col <= topk)
        X = d.select(cfg['cols']).to_numpy().astype(np.float64)[mask]
        p, s = fit_fam(X, y[mask], groups[mask], pid_col[mask], cfg['params'], cfg['seeds'],
                       cfg.get('ranker', False))
        pred[mask], split[mask] = p, s
    return pred, split


def score(d, pred, topk, truths, labels, key):
    cands = (d['oof_rank'] <= topk).to_numpy()
    z = np.where(cands, pred + 10.0, d['oof'].to_numpy() - 1e6)
    return summarize(key.with_columns(pl.Series('score', z)), truths, labels)


def rank_within_pair(d, pred, topk):
    """percentile rank of the prediction inside each pair's candidate set (for rank averaging)"""
    cands = (d['oof_rank'] <= topk).to_numpy()
    out = np.zeros(len(pred))
    pid = d['pair_id'].to_numpy()
    for p in np.unique(pid[cands]):
        m = cands & (pid == p)
        v = pred[m]
        r = np.empty(len(v))
        r[np.argsort(v)] = np.arange(len(v))
        out[m] = r / max(len(v) - 1, 1)
    return out


def main():
    t0 = time.time()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]
    sc, chrono = ['oof', 'oof_rank'], ['chrono_rank', 'chrono_frac', 'n_hands']

    data, features, _, _ = load_fold(0)
    rcols = [f for f in features if not f.startswith('family_') and f in data.columns
             and f not in struct + chrono + sc]
    d = d.join(data.select(['pair_id', 'hand_id'] + rcols).with_columns(pl.lit(1).alias('_hit')),
               on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert d['_hit'].null_count() == 0
    d = d.drop('_hit').with_columns(pl.col(rcols).fill_null(-999.0))

    # E2 interaction block (isolation mechanics: who opens, who folds behind)
    d = d.with_columns([
        (pl.col('pf_open_pair') * (pl.col('pf_others_fold') >= 3).cast(pl.Float64)).alias('x_open_manyfold'),
        (pl.col('pf_open_pair') * pl.col('pf_partner_fold')).alias('x_open_partnerfold'),
        (pl.col('pf_others_fold') * pl.col('pf_opener_pos')).alias('x_fold_openerpos')])
    inter = ['x_open_manyfold', 'x_open_partnerfold', 'x_fold_openerpos']

    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float)
    groups = d['table_id'].to_numpy()
    key = d.select('pair_id', 'hand_id', 'table_id')

    ref = struct + chrono + sc
    r382 = ref + rcols

    def uni(cols, params=CLF, seeds=S3, ranker=False):
        return {f: dict(cols=cols, params=params, seeds=seeds, ranker=ranker) for f in FAMILIES}

    def splice(base, ci_cfg):
        """ref for transfer and soft play, a different recipe for isolation"""
        out = dict(base)
        out[CI] = ci_cfg
        return out

    ci_mono_cols = ref
    mono = [1 if c == 'pf_others_fold' else (-1 if c == 'oof_rank' else 0) for c in ci_mono_cols]
    assert sum(abs(v) for v in mono) == 2, 'monotone anchors missing'
    RNK = dict(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=40, learning_rate=.05,
               n_jobs=2, verbosity=-1)

    base_ref = uni(ref)
    specs = {
        # --- E1 -----------------------------------------------------------------
        'ref': (base_ref, TOP),
        'r382': (uni(r382), TOP),
        'r382_ff50': (uni(r382, dict(CLF, colsample_bytree=.5)), TOP),
        'bag_ref': (uni(ref, BAG, S5), TOP),
        'bag_r382': (uni(r382, BAG, S5), TOP),
        # --- E2 scope -----------------------------------------------------------
        'ref_top15': (base_ref, 15),
        'ref_top25': (base_ref, 25),
        'ref_top30': (base_ref, 30),
        # --- E2 isolation -------------------------------------------------------
        'ci_reg': (splice(base_ref, dict(cols=ref, seeds=S3, params=dict(
            CLF, min_child_samples=80, num_leaves=5, n_estimators=100))), TOP),
        'ci_struct_only': (splice(base_ref, dict(cols=struct + chrono, seeds=S3, params=CLF)), TOP),
        'ci_mono': (splice(base_ref, dict(cols=ci_mono_cols, seeds=S3, params=dict(
            CLF, monotone_constraints=mono))), TOP),
        'ci_inter': (splice(base_ref, dict(cols=ref + inter, seeds=S3, params=CLF)), TOP),
        'ci_bag': (splice(base_ref, dict(cols=ref, seeds=S5, params=BAG)), TOP),
        'ci_r382': (splice(base_ref, dict(cols=r382, seeds=S3, params=CLF)), TOP),
        # --- E3 -----------------------------------------------------------------
        'rank_ref': (uni(ref, RNK, S3, True), TOP),
        'rank_r382': (uni(r382, RNK, S3, True), TOP),
        'rank_ci': (splice(base_ref, dict(cols=ref, seeds=S3, params=RNK, ranker=True)), TOP),
    }

    res = {'training_only': True, 'queries': len(truths), 'gate': GATE,
           'n_ranker_features': len(rcols), 'sets': {}, 'paired_vs_ref': {},
           'ci_splits': {}, 'runtime': {}}
    maps, raw, splits = {}, {}, {}
    for name, (per_fam, k) in specs.items():
        t1 = time.time()
        pred, sp = run(d, y, groups, per_fam, k)
        s, m = score(d, pred, k, truths, labels, key)
        maps[name], raw[name], splits[name] = m, (pred, k), sp
        s['topk'] = k
        res['sets'][name] = s
        res['runtime'][name] = round(time.time() - t1)
        log(name, json.dumps({'map5': round(s['map5'], 4),
                              'by_family': {f[:2]: round(v, 4) for f, v in s['by_family'].items()},
                              'sec': res['runtime'][name]}))

    # --- E1 ensembles: combine the stored ref / r382 predictions ---------------
    for tag, a, b, how in [('ens_rank', 'ref', 'r382', 'rank'), ('ens_prob', 'ref', 'r382', 'prob'),
                           ('ens_bag', 'bag_ref', 'bag_r382', 'rank')]:
        pa, pb = raw[a][0], raw[b][0]
        if how == 'rank':
            z = .5 * (rank_within_pair(d, pa, TOP) + rank_within_pair(d, pb, TOP))
        else:
            z = .5 * (pa + pb)
        s, m = score(d, z, TOP, truths, labels, key)
        maps[tag], s['topk'] = m, TOP
        res['sets'][tag] = s
        log(tag, json.dumps({'map5': round(s['map5'], 4),
                             'by_family': {f[:2]: round(v, 4) for f, v in s['by_family'].items()}}))

    # --- paired bootstrap against ref, plus the per-split CI check -------------
    ci_pairs = [p for p in maps['ref'] if labels[p] == CI]
    sp_ref = {}
    fam_mask = (d['behavior_family'].to_numpy() == CI)
    for p, s in zip(d['pair_id'].to_numpy()[fam_mask], splits['ref'][fam_mask]):
        if s >= 0:
            sp_ref[p] = int(s)
    for name in maps:
        if name == 'ref':
            continue
        b = boot(maps['ref'], maps[name], ptab, pools)
        b['by_family'] = {f: round(float(np.mean([maps[name][p] - maps['ref'][p]
                                                  for p in maps['ref'] if labels[p] == f])), 4)
                          for f in FAMILIES}
        res['paired_vs_ref'][name] = b
        per_split = {}
        for si in range(5):
            ps = [p for p in ci_pairs if sp_ref.get(p) == si]
            if ps:
                per_split[si] = round(float(np.mean([maps[name][p] - maps['ref'][p] for p in ps])), 4)
        res['ci_splits'][name] = {'per_split': per_split,
                                  'splits_better': int(sum(v > 0 for v in per_split.values())),
                                  'n_ci_pairs': len(ci_pairs)}
        log(f'{name} - ref', json.dumps({'mean': round(b['mean_delta'], 4),
                                         'ci95': [round(v, 4) for v in b['ci95']],
                                         'fam': b['by_family'],
                                         'ci_splits_better': res['ci_splits'][name]['splits_better']}))

    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-stage2-ensemble.json', res)
    log('done', res['seconds'])
    print('DONE')


if __name__ == '__main__':
    main()
