"""Step 37: family-agnostic coordination anomaly from the conditional-action (ca_) features (OBSERVATIONS 14.13 -> 14.14).

Motivation: B - P = .038 on the public board points to positives outside the scored families (hidden family, ~40-50
pairs). A supervised model trained on the three disclosed families cannot be expected to rank them. The 33 ca_
features measure how a player acts when the partner is in the hand versus when not (residuals against a behaviour
model, covariance with the partner's actions, call cost, fold benefit, weak pressure, pre-flop information use),
which is what any coordination family must perturb. This script builds an UNSUPERVISED anomaly score from them and
asks, label-free on evaluation:
  1. power on development (folds 1-2 all-pairs frames, 86,922 pairs incl. ~740 labelled positives): does the
     anomaly rank disclosed positives above the population, evenly across the three families? Reference: the
     326-aggregate known-family model trained on the OTHER tables' labelled pairs.
  2. development "unknown anomalies": very anomalous pairs (top .1 %) that the known-family model scores < .1 and
     that are not labelled positive. Are labelled negatives among them at the base rate (noise) or absent? Do they
     share a coherent signature distinct from the three families (cosine similarity of mean z-vectors)?
  3. evaluation: same construction on the 112,540 pairs (ca_ aggregates from outputs_v11_conditional inference
     results). Block (v13 >= .5) vs tail anomaly; tail pairs above the 99.9th percentile; their signature versus
     the development families and versus the development unknown anomalies.
Score: per feature, normal scores (van der Waerden) within shared-hands decile bins (removes the 1/sqrt(n) spread),
then Ledoit-Wolf Mahalanobis distance over the population. Primary feature set: the 33 *_mean aggregates;
secondary: all 132 (mean/std/max/q95). No candidate file is produced here.
Outputs: experiments/hidden_family_20260911/{dev-anomaly.parquet, eval-anomaly.parquet, results.json}
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, glob, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json
from pu_pairs.step32_crossphase_delta_model import EXP as XEXP, PARAMS, X
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = CV / 'experiments' / 'hidden_family_20260911'
PU = CV / 'experiments' / 'pu_pairs_20260911'
INF = ORIGINAL / 'outputs_v11_conditional' / 'inference-package' / 'results'
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
TOP_Q = .999


def normal_scores_binned(M, n_hands, bins=10):
    """M: (n, p) array. Rank-to-normal within shared-hands quantile bins, per column. NaN -> 0."""
    out = np.zeros_like(M, dtype=np.float64)
    edges = np.quantile(n_hands, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.searchsorted(edges, n_hands, side='right')
    for k in range(bins):
        idx = np.where(b == k)[0]
        if len(idx) == 0:
            continue
        sub = M[idx]
        for j in range(M.shape[1]):
            col = sub[:, j].astype(np.float64)
            ok = np.isfinite(col)
            r = np.empty(len(col)); r[:] = np.nan
            if ok.sum() > 1:
                ranks = np.argsort(np.argsort(col[ok], kind='mergesort'), kind='mergesort') + 1.0
                # average ties
                u, inv = np.unique(col[ok], return_inverse=True)
                sums = np.bincount(inv, weights=ranks); cnts = np.bincount(inv)
                ranks = (sums / cnts)[inv]
                r[ok] = norm.ppf(ranks / (ok.sum() + 1.0))
            out[idx, j] = np.nan_to_num(r, nan=0.0)
    return out


def mahalanobis(Z):
    lw = LedoitWolf().fit(Z)
    d2 = lw.mahalanobis(Z)
    return d2, lw


def q(a, qs):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(v) for v in np.quantile(a, qs)] if len(a) else []


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    t0 = time.perf_counter(); OUT.mkdir(exist_ok=True)
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    # ---------------- development: folds 1-2 all-pairs frames ----------------
    files = sorted(glob.glob(str(PU / 'fold1' / 'batch_*-pairs.parquet'))) + sorted(glob.glob(str(PU / 'fold2' / 'batch_*-pairs.parquet')))
    sch = pl.scan_parquet(files[0]).collect_schema().names()
    ca_all = [c for c in sch if c.startswith('ca_')]; ca_mean = [c for c in ca_all if c.endswith('_mean')]
    base_aggs = [c for c in sch if (c.endswith('_mean') or c.endswith('_std')) and not c.startswith('ca_')]
    lab_dev = pl.read_parquet(XEXP / 'lab_on_dev.parquet')
    single = [c for c in lab_dev.columns if c.endswith('_mean') or c.endswith('_std')]
    assert all(c in sch for c in single), 'fold frames lack some single-phase aggregates'
    dev = pl.concat([pl.read_parquet(f, columns=['pair_id', 'table_id', 'shared_hands', *single, *ca_all]) for f in files], how='vertical_relaxed')
    dev = dev.join(lab, on='pair_id', how='left').with_columns(pl.col('label').fill_null(-1))
    fold_tables = set(dev['table_id'].unique())
    # known-family reference model: labelled pairs on the OTHER tables
    tr = lab_dev.filter(~pl.col('table_id').is_in(list(fold_tables))).join(lab, on='pair_id')
    ref = lgb.LGBMClassifier(**PARAMS).fit(X(tr, single), (tr['label'] == 1).to_numpy().astype(int))
    dev = dev.with_columns(pl.Series('risk_known', ref.predict_proba(X(dev, single))[:, 1]))
    n_h = dev['shared_hands'].to_numpy().astype(float)
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'dev': {'pairs': dev.height, 'tables': len(fold_tables), 'labelled_pos': int((dev['label'] == 1).sum()), 'labelled_neg': int((dev['label'] == 0).sum()),
                                                                          'ref_model_training_pairs': tr.height, 'n_ca_mean': len(ca_mean), 'n_ca_all': len(ca_all)}}
    y_pos = (dev['label'] == 1).to_numpy(); y_neg = (dev['label'] == 0).to_numpy(); fam = dev['behavior_family'].to_numpy()
    scores = {}
    for name, cols in [('ca_mean', ca_mean), ('ca_all', ca_all), ('base_aggs', single)]:
        Z = normal_scores_binned(dev.select(cols).to_numpy(), n_h)
        d2, _ = mahalanobis(Z)
        scores[name] = (Z, d2)
        lab_mask = y_pos | y_neg
        rep = {'ap_pos_vs_labelled_neg': float(average_precision_score(y_pos[lab_mask], d2[lab_mask])), 'auc_pos_vs_labelled_neg': float(roc_auc_score(y_pos[lab_mask], d2[lab_mask])),
               'ap_pos_vs_all_others': float(average_precision_score(y_pos, d2)), 'auc_pos_vs_all_others': float(roc_auc_score(y_pos, d2)),
               'by_family_auc_vs_all_others': {f: float(roc_auc_score((fam == f)[~y_pos | (fam == f)], d2[~y_pos | (fam == f)])) for f in FAMS},
               'pos_pct_rank_q': q([(d2 < v).mean() for v in d2[y_pos]], [.1, .25, .5, .75, .9]), 'population_q': q(d2, [.5, .9, .99, .999])}
        res['dev'][name] = rep
        print(json.dumps({'dev_set': name, **rep}), flush=True)
    rk = dev['risk_known'].to_numpy()
    lab_mask = y_pos | y_neg
    res['dev']['reference_known_family_model'] = {'ap_pos_vs_labelled_neg': float(average_precision_score(y_pos[lab_mask], rk[lab_mask])), 'ap_pos_vs_all_others': float(average_precision_score(y_pos, rk)),
                                                  'unlabelled_ge_0.9': int(((dev['label'] == -1).to_numpy() & (rk >= .9)).sum()), 'unlabelled_ge_0.5': int(((dev['label'] == -1).to_numpy() & (rk >= .5)).sum())}
    print(json.dumps(res['dev']['reference_known_family_model']), flush=True)
    # unknown anomalies (primary set)
    Z, d2 = scores['ca_mean']
    thr = float(np.quantile(d2, TOP_Q))
    top = d2 >= thr
    unknown = top & (rk < .1) & ~y_pos
    base_neg_rate = y_neg.sum() / (~y_pos).sum()
    res['dev']['unknown_anomalies'] = {'threshold_d2_q999': thr, 'top_n': int(top.sum()), 'top_labelled_pos': int((top & y_pos).sum()), 'top_known_ge_0.5_unlabelled': int((top & (rk >= .5) & (dev['label'] == -1).to_numpy()).sum()),
                                       'unknown_n': int(unknown.sum()), 'unknown_labelled_neg': int((unknown & y_neg).sum()), 'expected_labelled_neg_if_noise': float(unknown.sum() * base_neg_rate),
                                       'unknown_tables': int(len(set(dev.filter(pl.Series(unknown))['table_id']))), 'unknown_shared_hands_q': q(n_h[unknown], [.1, .5, .9]), 'population_shared_hands_q': q(n_h, [.1, .5, .9])}
    # signatures
    sig = {f: Z[y_pos & (fam == f)].mean(axis=0) for f in FAMS}
    sig_unknown = Z[unknown].mean(axis=0) if unknown.sum() else np.zeros(Z.shape[1])
    sig_neg = Z[y_neg].mean(axis=0)
    # coherence: mean pairwise cosine among unknown anomalies vs among random population pairs
    rng = np.random.default_rng(7329)
    def mean_pair_cos(M, k=2000):
        if len(M) < 2:
            return None
        i = rng.integers(0, len(M), k); j = rng.integers(0, len(M), k); ok = i != j
        a = M[i[ok]]; b = M[j[ok]]
        return float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))
    order = np.argsort(-np.abs(sig_unknown))
    res['dev']['signatures'] = {'cos_unknown_vs_family': {f: cos(sig_unknown, sig[f]) for f in FAMS}, 'cos_family_vs_family': {f'{a}|{b}': cos(sig[a], sig[b]) for a in FAMS for b in FAMS if a < b},
                                'cos_unknown_vs_labelled_neg_mean': cos(sig_unknown, sig_neg), 'unknown_top_features': [{'feature': ca_mean[i], 'mean_z': float(sig_unknown[i])} for i in order[:8]],
                                'family_top_features': {f: [{'feature': ca_mean[i], 'mean_z': float(sig[f][i])} for i in np.argsort(-np.abs(sig[f]))[:5]] for f in FAMS},
                                'coherence_mean_pair_cos': {'unknown': mean_pair_cos(Z[unknown]), 'labelled_pos': mean_pair_cos(Z[y_pos]), 'population': mean_pair_cos(Z[rng.choice(len(Z), 5000, replace=False)]),
                                                            'top_anomalies_known_pos': mean_pair_cos(Z[top & y_pos])}}
    print(json.dumps(res['dev']['unknown_anomalies']), flush=True); print(json.dumps(res['dev']['signatures']), flush=True)
    dev.select('pair_id', 'table_id', 'shared_hands', 'label', 'behavior_family', 'risk_known').with_columns(pl.Series('anom_ca_mean', scores['ca_mean'][1]), pl.Series('anom_ca_all', scores['ca_all'][1]),
                                                                                                             pl.Series('anom_base', scores['base_aggs'][1]), pl.Series('unknown_anomaly', unknown)).write_parquet(OUT / 'dev-anomaly.parquet')
    # ---------------- evaluation ----------------
    ev = pl.concat([pl.read_parquet(f, columns=['pair_id', 'table_id', *ca_all]) for f in sorted(glob.glob(str(INF / 'batch_*-pairs.parquet')))], how='vertical_relaxed')
    meta = pl.read_parquet(XEXP / 'eval-scores-v3.parquet').select('pair_id', 'shared_hands', 'risk', 'delta_score', 'single_on_dev')
    ev = ev.join(meta, on='pair_id', how='inner').sort('pair_id')
    assert ev.height == 112540, ev.height
    n_e = ev['shared_hands'].to_numpy().astype(float)
    Ze = normal_scores_binned(ev.select(ca_mean).to_numpy(), n_e); d2e, _ = mahalanobis(Ze)
    Ze_all = normal_scores_binned(ev.select(ca_all).to_numpy(), n_e); d2e_all, _ = mahalanobis(Ze_all)
    risk = ev['risk'].to_numpy(); blk = risk >= .5; tail = ~blk; lo = risk < .1
    thr_e = float(np.quantile(d2e[tail], TOP_Q)); top_e = tail & (d2e >= thr_e)
    sod = ev['single_on_dev'].to_numpy().astype(float); dl = ev['delta_score'].to_numpy().astype(float)
    sig_e_block = Ze[blk].mean(axis=0); sig_e_top = Ze[top_e].mean(axis=0)
    res['eval'] = {'pairs': ev.height, 'auc_block_vs_tail_ca_mean': float(roc_auc_score(blk, d2e)), 'auc_block_vs_tail_ca_all': float(roc_auc_score(blk, d2e_all)),
                   'auc_hi_vs_lo_ca_mean': float(roc_auc_score(risk[blk | lo] >= .9, d2e[blk | lo])), 'block_pct_rank_in_tail_q': q([(d2e[tail] < v).mean() for v in d2e[blk]], [.1, .25, .5]),
                   'tail_q': q(d2e[tail], [.5, .9, .99, .999]), 'block_q': q(d2e[blk], [.1, .5, .9]),
                   'tail_top_anomalies': {'threshold': thr_e, 'n': int(top_e.sum()), 'tables': int(len(set(ev.filter(pl.Series(top_e))['table_id']))), 'v13_risk_q': q(risk[top_e], [.5, .9, .99]), 'v13_ge_0.1': int((risk[top_e] >= .1).sum()),
                                          'delta_v3_q': q(dl[top_e], [.5, .9]), 'delta_v3_ge_0.5': int(np.nansum(dl[top_e] >= .5)), 'single_on_dev_ge_0.5': int(np.nansum(sod[top_e] >= .5)),
                                          'shared_hands_q': q(n_e[top_e], [.1, .5, .9]), 'tail_shared_hands_q': q(n_e[tail], [.1, .5, .9])},
                   'signatures': {'cos_eval_block_vs_dev_family': {f: cos(sig_e_block, sig[f]) for f in FAMS}, 'cos_eval_tailtop_vs_dev_family': {f: cos(sig_e_top, sig[f]) for f in FAMS},
                                  'cos_eval_tailtop_vs_dev_unknown': cos(sig_e_top, sig_unknown), 'cos_eval_tailtop_vs_eval_block': cos(sig_e_top, sig_e_block),
                                  'eval_tailtop_top_features': [{'feature': ca_mean[i], 'mean_z': float(sig_e_top[i])} for i in np.argsort(-np.abs(sig_e_top))[:8]],
                                  'coherence_mean_pair_cos': {'tail_top': mean_pair_cos(Ze[top_e]), 'block': mean_pair_cos(Ze[blk]), 'tail_population': mean_pair_cos(Ze[tail][rng.choice(tail.sum(), 5000, replace=False)])}}}
    # do the 231 undisclosed development colluders look anomalous in evaluation? (they should not if clean in evaluation)
    und = lo & (sod >= .9)
    res['eval']['undisclosed_dev_colluders_in_eval'] = {'n': int(und.sum()), 'anom_pct_rank_in_tail_q': q([(d2e[tail] < v).mean() for v in d2e[und]], [.25, .5, .75])}
    print(json.dumps(res['eval'], indent=1), flush=True)
    ev.select('pair_id', 'table_id', 'shared_hands', 'risk', 'delta_score', 'single_on_dev').with_columns(pl.Series('anom_ca_mean', d2e), pl.Series('anom_ca_all', d2e_all), pl.Series('tail_top_anomaly', top_e)).write_parquet(OUT / 'eval-anomaly.parquet')
    res['seconds'] = round(time.perf_counter() - t0, 1)
    write_json(OUT / 'results.json', res)


if __name__ == '__main__':
    main()
