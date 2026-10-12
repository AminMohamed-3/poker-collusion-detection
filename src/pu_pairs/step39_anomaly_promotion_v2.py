"""Step 39: anomaly promotion v2 (OBSERVATIONS 14.15) - per-bin covariance, matched-rate development test,
spill-over variants selected on development labels only.

Fixes relative to step 38:
  * covariance per shared-hands decile bin (pooled covariance inflated d^2 for pairs with few hands: tail corr(d^2,
    shared hands) = -.21, 47 of 134 above-threshold pairs in the 40-55 hands bin vs 18 in >= 110);
  * the development "block" is the reference model's top K with K matching the evaluation block RATE (909 / 112,540),
    so the promotion threshold T (median block anomaly) means the same thing in both settings;
  * development unlabelled pair ids are rebuilt (X + sha1(player_1 player_2)[:12], as in build_allpairs_fold) so
    spill-over can be evaluated for every pair;
  * three spill-over variants, chosen by DEVELOPMENT labelled AP only (pre-registered: highest labelled AP after
    promotion; ties -> the most conservative, B):
      A  promote every tail pair with d^2 >= T;
      B  A minus pairs sharing a player with a block pair;
      C  A minus sharing pairs whose d^2 is below that of every block pair they share a player with ("swap" rule:
         a tail pair more anomalous than its block neighbour is kept).
Gates: G1 by-family AUC >= .98 (per-bin covariance); G2 selected variant: development labelled AP and PU-AP not
worse after promotion; G3 evaluation promoted set: |corr(d^2, shared hands)| in the tail <= .05, shared-hands
median ratio >= .8, per-table max <= 5, mean pairwise cosine >= .2.
Prediction if uploaded: public P in [-.006, +.025] of .938 (composite [-.004, +.018] vs .876); E, B unchanged.
Outputs: experiments/hidden_family_20260911/{freeze-v2,promotion-results-v2}.json, promoted-eval-v2.parquet,
         outputs_anomaly_20260911/submission_anomaly_promote_v2.csv (+ validation, receipt-v2.json)
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, glob, time, hashlib, itertools
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
from pu_pairs.step32_crossphase_delta_model import EXP as XEXP, PARAMS, X, sha
from pu_pairs.step37_conditional_anomaly import PU, INF, FAMS, normal_scores_binned, q
from pu_pairs.step38_anomaly_promotion import OUT, OUTD, MIN_SH, TAU_BLOCK, PROMOTE_LO, PROMOTE_HI, mean_pair_cos
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score

EVAL_BLOCK_RATE = 909 / 112540
BINS = 10


def anomaly_binned(frame, cols, bins=BINS):
    n_hands = frame['shared_hands'].to_numpy().astype(float)
    Z = normal_scores_binned(frame.select(cols).to_numpy(), n_hands, bins)
    edges = np.quantile(n_hands, np.linspace(0, 1, bins + 1)[1:-1]); b = np.searchsorted(edges, n_hands, side='right')
    d2 = np.zeros(len(Z))
    for k in range(bins):
        idx = np.where(b == k)[0]
        if len(idx) > 50:
            d2[idx] = LedoitWolf().fit(Z[idx]).mahalanobis(Z[idx])
    return Z, d2


def synthetic_id(p1, p2):
    return 'X' + hashlib.sha1((p1 + p2).encode()).hexdigest()[:12].upper()


def dev_pair_players(tables):
    """All (player_1 < player_2) pairs per development table with their pair ids (labelled id if labelled, else synthetic)."""
    seats = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').select('hand_id', 'player_id')
    hands = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').filter((pl.col('phase') == 'development') & pl.col('table_id').is_in(list(tables))).select('hand_id', 'table_id')
    pt = seats.join(hands, on='hand_id').select('player_id', 'table_id').unique().collect()
    rows = []
    for t, g in pt.group_by('table_id'):
        ps = sorted(g['player_id'].to_list())
        for a, b2 in itertools.combinations(ps, 2):
            rows.append((synthetic_id(a, b2), a, b2))
    syn = pl.DataFrame(rows, schema=['pair_id', 'player_1', 'player_2'], orient='row')
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
    return pl.concat([lab, syn], how='vertical_relaxed').unique(subset=['player_1', 'player_2'], keep='first')


def variants(frame, blk, d2, T, pairs):
    """Masks for variants A, B, C. frame: pair_id column aligned with blk/d2; pairs: pair_id, player_1, player_2."""
    pp = frame.select('pair_id').join(pairs, on='pair_id', how='left')
    p1 = pp['player_1'].to_numpy(); p2 = pp['player_2'].to_numpy()
    above = (~blk) & (d2 >= T)
    # block anomaly per player (max over block pairs containing the player)
    bmax = {}
    for i in np.where(blk)[0]:
        for p in (p1[i], p2[i]):
            if p is not None:
                bmax[p] = max(bmax.get(p, -np.inf), d2[i])
    shares = np.array([(p1[i] in bmax) or (p2[i] in bmax) for i in range(len(d2))])
    swap_keep = np.array([d2[i] > max(bmax.get(p1[i], -np.inf), bmax.get(p2[i], -np.inf)) for i in range(len(d2))])
    return {'A': above, 'B': above & ~shares, 'C': above & (~shares | swap_keep)}, shares, bmax


def rerank(risk, d2, prom, blk):
    new = risk.copy()
    if prom.sum():
        rk = (np.argsort(np.argsort(d2[prom])) + 1) / (prom.sum() + 1); new[prom] = PROMOTE_LO + (PROMOTE_HI - PROMOTE_LO) * rk
    rest = (~blk) & (~prom)
    if rest.sum():
        rr = risk[rest]; new[rest] = PROMOTE_LO * (rr - rr.min()) / (rr.max() - rr.min() + 1e-12) * (1 - 1e-6)
    return new


def main():
    t0 = time.perf_counter(); OUTD.mkdir(exist_ok=True)
    rng = np.random.default_rng(7329)
    protected = check_protected()
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'min_shared_hands': MIN_SH, 'bins': BINS, 'eval_block_rate': EVAL_BLOCK_RATE,
              'threshold_rule': 'median anomaly of the block', 'variant_selection': 'highest development labelled AP after promotion; ties -> B',
              'gates': {'G1_family_auc_min': .98, 'G2': 'selected variant: dev labelled AP and PU-AP not worse', 'G3_corr_abs_max': .05, 'G3_shared_hands_ratio_min': .8, 'G3_table_max': 5, 'G3_coherence_min': .2},
              'prediction_if_uploaded': 'public P in [-.006, +.025] of .938 (composite [-.004, +.018] vs .876); E, B unchanged',
              'inputs': {'v13': sha(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv'), 'base': sha(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv')}}
    write_json(OUT / 'freeze-v2.json', freeze)
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    # ---------------- development ----------------
    files = sorted(glob.glob(str(PU / 'fold1' / 'batch_*-pairs.parquet'))) + sorted(glob.glob(str(PU / 'fold2' / 'batch_*-pairs.parquet')))
    sch = pl.scan_parquet(files[0]).collect_schema().names(); ca_mean = [c for c in sch if c.startswith('ca_') and c.endswith('_mean')]
    lab_dev = pl.read_parquet(XEXP / 'lab_on_dev.parquet'); single = [c for c in lab_dev.columns if c.endswith('_mean') or c.endswith('_std')]
    dev = pl.concat([pl.read_parquet(f, columns=['pair_id', 'table_id', 'shared_hands', *single, *ca_mean]) for f in files], how='vertical_relaxed')
    dev = dev.filter(pl.col('shared_hands') >= MIN_SH).join(lab, on='pair_id', how='left').with_columns(pl.col('label').fill_null(-1)).sort('pair_id')
    fold_tables = set(dev['table_id'].unique())
    pairs_d = dev_pair_players(fold_tables)
    matched = dev.select('pair_id').join(pairs_d, on='pair_id', how='left')['player_1'].is_not_null().mean()
    tr = lab_dev.filter(~pl.col('table_id').is_in(list(fold_tables))).join(lab, on='pair_id')
    ref = lgb.LGBMClassifier(**PARAMS).fit(X(tr, single), (tr['label'] == 1).to_numpy().astype(int))
    risk_d = ref.predict_proba(X(dev, single))[:, 1]
    Zd, d2d = anomaly_binned(dev, ca_mean)
    y_pos = (dev['label'] == 1).to_numpy(); y_neg = (dev['label'] == 0).to_numpy(); fam = dev['behavior_family'].to_numpy(); labm = y_pos | y_neg; unl = (dev['label'] == -1).to_numpy()
    K = int(round(EVAL_BLOCK_RATE * dev.height)); order = np.argsort(-risk_d); blk_d = np.zeros(dev.height, bool); blk_d[order[:K]] = True
    T_d = float(np.median(d2d[blk_d]))
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'dev': {'pairs': dev.height, 'tables': len(fold_tables), 'labelled_pos': int(y_pos.sum()), 'labelled_neg': int(y_neg.sum()), 'pair_players_matched_share': float(matched),
                                                                          'block_K': K, 'block_risk_min': float(risk_d[order[K - 1]]), 'block_labelled_pos': int((blk_d & y_pos).sum()), 'block_labelled_neg': int((blk_d & y_neg).sum()), 'threshold_T': T_d,
                                                                          'corr_d2_shared_hands_tail': float(np.corrcoef(d2d[~blk_d], dev['shared_hands'].to_numpy()[~blk_d])[0, 1])}}
    res['dev']['anomaly'] = {'ap_pos_vs_labelled_neg': float(average_precision_score(y_pos[labm], d2d[labm])), 'auc_pos_vs_all_others': float(roc_auc_score(y_pos, d2d)),
                             'by_family_auc_vs_all_others': {f: float(roc_auc_score((fam == f)[~y_pos | (fam == f)], d2d[~y_pos | (fam == f)])) for f in FAMS}}
    res['dev']['gate_G1'] = bool(all(v >= .98 for v in res['dev']['anomaly']['by_family_auc_vs_all_others'].values()))
    masks, shares_d, bmax_d = variants(dev, blk_d, d2d, T_d, pairs_d)
    ap0 = float(average_precision_score(y_pos[labm], risk_d[labm])); pu0 = float(average_precision_score(y_pos, risk_d))
    res['dev']['reference'] = {'ap_labelled': ap0, 'pu_ap': pu0, 'labelled_pos_in_tail': int((~blk_d & y_pos).sum()), 'tail_pos_above_T': int((~blk_d & y_pos & (d2d >= T_d)).sum()),
                               'tail_pos_sharing_block_player': int((~blk_d & y_pos & shares_d).sum()), 'tail_share_rate': float(shares_d[~blk_d].mean())}
    # swap diagnostic: for tail positives sharing a block player, label of the block neighbour(s) and anomaly comparison
    pp = dev.select('pair_id').join(pairs_d, on='pair_id', how='left'); p1 = pp['player_1'].to_numpy(); p2 = pp['player_2'].to_numpy(); lbl = dev['label'].to_numpy()
    neigh = {}
    for i in np.where(blk_d)[0]:
        for p in (p1[i], p2[i]):
            neigh.setdefault(p, []).append(i)
    swaps = []
    for i in np.where(~blk_d & y_pos & shares_d)[0]:
        js = sorted(set(neigh.get(p1[i], []) + neigh.get(p2[i], [])))
        swaps.append({'tail_pos_d2': round(float(d2d[i]), 1), 'tail_pos_risk': round(float(risk_d[i]), 3), 'block_neighbours': [{'label': int(lbl[j]), 'd2': round(float(d2d[j]), 1), 'risk': round(float(risk_d[j]), 3)} for j in js]})
    res['dev']['swap_diagnostic'] = swaps
    res['dev']['variants'] = {}
    for v, m in masks.items():
        new = rerank(risk_d, d2d, m, blk_d)
        res['dev']['variants'][v] = {'promoted': int(m.sum()), 'labelled_pos_promoted': int((m & y_pos).sum()), 'labelled_neg_promoted': int((m & y_neg).sum()), 'unlabelled_promoted': int((m & unl).sum()),
                                     'ap_labelled_after': float(average_precision_score(y_pos[labm], new[labm])), 'pu_ap_after': float(average_precision_score(y_pos, new)),
                                     'promoted_per_table_max': int(dev.filter(pl.Series(m)).group_by('table_id').len()['len'].max()) if m.sum() else 0}
    best = max(res['dev']['variants'], key=lambda v: (round(res['dev']['variants'][v]['ap_labelled_after'], 6), v == 'B'))
    res['dev']['selected_variant'] = best
    res['dev']['gate_G2'] = bool(res['dev']['variants'][best]['ap_labelled_after'] >= ap0 and res['dev']['variants'][best]['pu_ap_after'] >= pu0)
    print(json.dumps(res['dev'], indent=1), flush=True)
    # ---------------- evaluation ----------------
    ev = pl.concat([pl.read_parquet(f, columns=['pair_id', 'table_id', *ca_mean]) for f in sorted(glob.glob(str(INF / 'batch_*-pairs.parquet')))], how='vertical_relaxed')
    epairs = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2', 'shared_hands')
    v13 = pl.read_csv(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv').select('pair_id', pl.col('risk_score').alias('risk'))
    ev = ev.join(epairs, on='pair_id').join(v13, on='pair_id').sort('pair_id'); assert ev.height == 112540
    risk_e = ev['risk'].to_numpy().astype(float); blk_e = risk_e >= TAU_BLOCK; tail_e = ~blk_e
    Ze, d2e = anomaly_binned(ev, ca_mean); T_e = float(np.median(d2e[blk_e])); sh = ev['shared_hands'].to_numpy().astype(float)
    masks_e, shares_e, _ = variants(ev, blk_e, d2e, T_e, epairs.select('pair_id', 'player_1', 'player_2'))
    res['eval'] = {'pairs': ev.height, 'threshold_T': T_e, 'block_pairs': int(blk_e.sum()), 'auc_block_vs_tail': float(roc_auc_score(blk_e, d2e)), 'corr_d2_shared_hands_tail': float(np.corrcoef(d2e[tail_e], sh[tail_e])[0, 1]),
                   'tail_d2_q': q(d2e[tail_e], [.5, .9, .99, .999]), 'block_d2_q': q(d2e[blk_e], [.1, .5, .9]), 'tail_share_rate': float(shares_e[tail_e].mean()), 'variants': {}}
    for v, m in masks_e.items():
        pt = ev.filter(pl.Series(m)).group_by('table_id').len()
        res['eval']['variants'][v] = {'promoted': int(m.sum()), 'tables': pt.height, 'per_table_max': int(pt['len'].max()) if m.sum() else 0, 'v13_risk_q': q(risk_e[m], [.1, .5, .9]), 'v13_ge_0.1': int((risk_e[m] >= .1).sum()),
                                      'shared_hands_median': float(np.median(sh[m])) if m.sum() else None, 'coherence': mean_pair_cos(Ze[m], rng), 'd2_q': q(d2e[m], [.1, .5, .9])}
    sel = masks_e[best]; pv = res['eval']['variants'][best]
    ratio = pv['shared_hands_median'] / float(np.median(sh[tail_e])) if sel.sum() else 0
    res['eval']['gate_G3'] = {'corr_ok': abs(res['eval']['corr_d2_shared_hands_tail']) <= .05, 'shared_hands_ratio': ratio, 'passed': bool(abs(res['eval']['corr_d2_shared_hands_tail']) <= .05 and ratio >= .8 and pv['per_table_max'] <= 5 and (pv['coherence'] or 0) >= .2)}
    res['eval']['tail_population_coherence'] = mean_pair_cos(Ze[tail_e][rng.choice(tail_e.sum(), 5000, replace=False)], rng)
    print(json.dumps(res['eval'], indent=1), flush=True)
    ev.select('pair_id', 'table_id', 'shared_hands', 'risk').with_columns(pl.Series('anom', d2e), pl.Series('shares_block_player', shares_e), *[pl.Series(f'promoted_{v}', m) for v, m in masks_e.items()]).write_parquet(OUT / 'promoted-eval-v2.parquet')
    # ---------------- candidate (selected variant) ----------------
    base = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'); cols = base.columns
    j = base.join(ev.select('pair_id', 'risk').with_columns(pl.Series('new_risk', rerank(risk_e, d2e, sel, blk_e))), on='pair_id', how='left')
    assert j.height == base.height and float(np.abs(j['risk_score'].to_numpy() - j['risk'].to_numpy()).max()) < 1e-9
    out = j.with_columns(pl.col('new_risk').alias('risk_score')).select(cols); assert out['pair_id'].equals(base['pair_id'])
    blk_old = base.filter(pl.col('risk_score') >= TAU_BLOCK); blk_new = out.filter(pl.col('risk_score') >= TAU_BLOCK)
    assert blk_old.height == blk_new.height and (blk_old['risk_score'].to_numpy() == blk_new['risk_score'].to_numpy()).all()
    for c in cols:
        if c != 'risk_score':
            assert out[c].equals(base[c]), c
    keep = ev.select('pair_id').filter(pl.Series(tail_e & ~sel))
    assert (np.argsort(base.join(keep, on='pair_id')['risk_score'].to_numpy(), kind='stable') == np.argsort(out.join(keep, on='pair_id')['risk_score'].to_numpy(), kind='stable')).all()
    dest = OUTD / 'submission_anomaly_promote_v2.csv'; out.write_csv(dest)
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(dest, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    write_json(OUTD / 'submission_anomaly_promote_v2-validation.json', vrep); assert vrep['valid'], vrep
    res['candidate'] = {'file': str(dest.relative_to(CV)), 'sha256': sha(dest), 'base_sha256': freeze['inputs']['base'], 'variant': best, 'block_pairs_unchanged': blk_old.height, 'promoted_pairs': int(sel.sum()),
                        'validation_valid': bool(vrep['valid']), 'gates_passed': bool(res['dev']['gate_G1'] and res['dev']['gate_G2'] and res['eval']['gate_G3']['passed']), 'prediction': freeze['prediction_if_uploaded'], 'submitted': False}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    write_json(OUT / 'promotion-results-v2.json', res); write_json(OUTD / 'receipt-v2.json', {'freeze': freeze, 'results': res})
    print(json.dumps(res['candidate'], indent=1), flush=True)


if __name__ == '__main__':
    main()
