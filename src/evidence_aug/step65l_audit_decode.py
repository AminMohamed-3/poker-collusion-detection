"""Step 65l (training-only audit): explicit decoding of the listing rule on top of the stage-2 probabilities.
The record (14.16, 14.22) says the truth list is the first five manipulated hands in (type, time) order: fold-type hands
before showdown-type, chronological within type, capped at 5. The stage-2 classifier learns this only through
chrono_rank / chrono_frac as ordinary features. Here the rule is applied explicitly: with p_g the stage-2 probability of
each top-20 candidate g, a candidate h is listed iff it is manipulated and fewer than five manipulated hands precede it,
    P(listed_h) = p_h * P( sum_{g precedes h} X_g <= 4 ),  X_g ~ Bernoulli(p_g^alpha) independent (Poisson-binomial).
Variants: precedence by time only or by (type, time); alpha in {.5, 1}; p from the chrono-aware hybrid stage 2 (double
counts the chronology; the decode then acts as an extra penalty) or from a chrono-free stage 2 (struct + oof + r382 for
transfer / soft play, struct + oof for isolation) where the explicit rule replaces the learned one; and a geometric blend.
Base = the production hybrid design (r382 for DT/SP, ref for CI, all with chrono3). Output:
experiments/label_process_20260911/audit-decode.json"""
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
from evidence_aug.step65b_family_stage2 import boot
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

CI = 'coordinated_isolation'
SEEDS = (11,)  # CLF has no seed dependence (step 65h2)


def fam_oof_cols(d, cols_by_fam, y, groups, params):
    pred = np.full(d.height, np.nan)
    for fam in FAMILIES:
        mask = ((d['behavior_family'] == fam) & (d['oof_rank'] <= TOP)).to_numpy()
        X = np.nan_to_num(d.select(cols_by_fam[fam]).to_numpy().astype(np.float64)[mask], nan=-999.0); ys = y[mask]; gs = groups[mask]
        p = np.zeros(len(ys))
        for seed in SEEDS:
            for tr, te in GroupKFold(n_splits=5).split(X, ys, gs):
                clf = lgb.LGBMClassifier(**params, random_state=seed); clf.fit(X[tr], ys[tr]); p[te] += clf.predict_proba(X[te])[:, 1] / len(SEEDS)
        pred[mask] = p
    return pred


def poisson_binomial_cdf4(q):
    """P(sum of independent Bernoulli(q_i) <= 4)."""
    dist = np.zeros(6); dist[0] = 1.0  # dist[k] = P(sum == k) for k < 5, dist[5] = P(sum >= 5)
    for x in q:
        new = np.zeros(6)
        new[0] = dist[0] * (1 - x)
        for k in range(1, 5):
            new[k] = dist[k] * (1 - x) + dist[k - 1] * x
        new[5] = dist[5] + dist[4] * x
        dist = new
    return float(dist[:5].sum())


def decode(d, p, cands, by_type, alpha):
    """Return P(listed) for candidate rows (NaN elsewhere)."""
    out = np.full(d.height, np.nan)
    sub = d.select('pair_id', 'chrono_rank', 'pair_sd').with_columns(pl.Series('p', p), pl.Series('row', np.arange(d.height)), pl.Series('c', cands)).filter(pl.col('c'))
    for _, g in sub.group_by('pair_id'):
        typ = (g['pair_sd'].to_numpy() > 0).astype(int) if by_type else np.zeros(g.height, dtype=int)
        t = g['chrono_rank'].to_numpy(); pp = g['p'].to_numpy(); rows = g['row'].to_numpy()
        key = typ * 1e6 + t
        q = np.clip(pp, 0, 1) ** alpha
        for i in range(g.height):
            pre = (key < key[i]) | ((key == key[i]) & (np.arange(g.height) < i))
            out[rows[i]] = pp[i] * poisson_binomial_cdf4(q[pre])
    return out


def main():
    t0 = time.time()
    audit = pl.read_parquet(OUT / 'audit-oof.parquet'); feat = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    d = audit.join(feat, on=['pair_id', 'hand_id'], validate='1:1').sort('pair_id', 'hand_id')
    struct = [c for c in feat.columns if c not in ('pair_id', 'hand_id')]; sc = ['oof', 'oof_rank']; chrono = ['chrono_rank', 'chrono_frac', 'n_hands']
    data, features, _, _ = load_fold(0)
    rcols = [f for f in features if not f.startswith('family_') and f in data.columns and f not in struct + chrono + sc]
    d = d.join(data.select(['pair_id', 'hand_id'] + rcols).with_columns(pl.lit(1).alias('_hit')), on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert d['_hit'].null_count() == 0
    d = d.drop('_hit').with_columns(pl.col(rcols).fill_null(-999.0))
    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows()); ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows()); pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float); groups = d['table_id'].to_numpy(); key = d.select('pair_id', 'hand_id', 'table_id')
    cands = (d['oof_rank'] <= TOP).to_numpy(); oof = d['oof'].to_numpy()
    ref = struct + chrono + sc; r382 = ref + rcols
    hybrid = {f: (ref if f == CI else r382) for f in FAMILIES}
    nochrono = {f: ([c for c in ref if c not in chrono] if f == CI else [c for c in r382 if c not in chrono]) for f in FAMILIES}
    log('fitting chrono-aware hybrid stage 2 (base) and chrono-free stage 2')
    p_base = fam_oof_cols(d, hybrid, y, groups, CLF); p_free = fam_oof_cols(d, nochrono, y, groups, CLF)
    # listing-rule diagnostics on the audit truth (training-only): how often is a listed hand preceded by >= 5 listed hands? (never, if the rule holds)
    diag = {}
    for by_type in (False, True):
        viol = 0; n = 0
        for _, g in d.filter(pl.col('relevance') == 1).group_by('pair_id'):
            typ = (g['pair_sd'].to_numpy() > 0).astype(int) if by_type else np.zeros(g.height, dtype=int)
            k = typ * 1e6 + g['chrono_rank'].to_numpy(); n += 1
            viol += int(g.height > 5)
        diag['by_type' if by_type else 'time_only'] = {'pairs': n, 'lists_longer_than_5': viol}
    diag['candidates_prob_sum_median'] = float(np.median([float(np.nansum(p_base[(d['pair_id'] == p).to_numpy() & cands])) for p in list(truths)[:200]]))
    specs = {}
    specs['base'] = np.where(cands, np.nan_to_num(p_base) + 10.0, oof - 1e6)
    specs['free_nodecode'] = np.where(cands, np.nan_to_num(p_free) + 10.0, oof - 1e6)
    for name, p in (('base', p_base), ('free', p_free)):
        for by_type in (False, True):
            for alpha in (.5, 1.0):
                dec = decode(d, np.nan_to_num(p), cands, by_type, alpha)
                specs[f'dec_{name}_{"type" if by_type else "time"}_a{alpha}'] = np.where(cands, np.nan_to_num(dec) + 10.0, oof - 1e6)
    dec_free = decode(d, np.nan_to_num(p_free), cands, True, 1.0)
    specs['blend_base_decfree'] = np.where(cands, np.sqrt(np.nan_to_num(p_base) * np.nan_to_num(dec_free)) + 10.0, oof - 1e6)
    res = {'training_only': True, 'queries': len(truths), 'diagnostics': diag, 'sets': {}, 'paired_vs_base': {}}; maps = {}
    for name, pred in specs.items():
        s, m = summarize(key.with_columns(pl.Series('score', pred)), truths, labels); maps[name] = m
        s['auc_within_top20'] = {fam: round(float(roc_auc_score(y[cands & (d['behavior_family'] == fam).to_numpy()], pred[cands & (d['behavior_family'] == fam).to_numpy()])), 4) for fam in FAMILIES}
        res['sets'][name] = s
        log(name, json.dumps({'map5': round(s['map5'], 4), 'by_family': {f: round(v, 4) for f, v in s['by_family'].items()}, 'auc20': s['auc_within_top20']}))
    for name in specs:
        if name == 'base':
            continue
        b = boot(maps['base'], maps[name], ptab, pools)
        b['by_family'] = {f: round(float(np.mean([maps[name][p] - maps['base'][p] for p in maps['base'] if labels[p] == f])), 4) for f in FAMILIES}
        res['paired_vs_base'][name] = b
        log(f'{name} minus base', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}))
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-decode.json', res)
    log('done', f'{res["seconds"]} s')


if __name__ == '__main__':
    main()
