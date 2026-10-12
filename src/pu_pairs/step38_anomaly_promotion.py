"""Step 38: pre-registered tail-promotion rule from the unsupervised coordination anomaly (OBSERVATIONS 14.14).

Step 37 showed that a label-free Mahalanobis anomaly over the 33 conditional-action means ranks disclosed
development positives at AUC .99+ in every family, and that on evaluation the 112 most anomalous tail pairs form a
coherent group (mean pairwise cosine .41, tail population .00) whose signature matches the detected block (.88).
This script (a) validates the promotion mechanism on development with the labels that are allowed for that,
(b) applies it to evaluation and builds a tail-only candidate. Everything is fixed here before the evaluation
numbers are computed (freeze.json).

Population filter: shared hands >= 40 (the evaluation pair list has none below ~40; development pairs with 1-3
shared hands produced the spurious "unknown anomalies" of step 37).
Anomaly: per-feature normal scores within shared-hands decile bins, Ledoit-Wolf Mahalanobis d^2 (ca_*_mean, 33).
Promotion rule: block = pairs with model risk >= .5 (unchanged). Threshold T = median anomaly of the block ("as
anomalous as a typical detected colluder"). Promote tail pairs with d^2 >= T that do NOT share a player with any
block pair (spill-over: a colluder's conditional features are perturbed in every pair they belong to; ring
transitivity is null, OBSERVATIONS 14.1). Promoted pairs are placed directly below the block in [.45, .5) ordered by
d^2; the rest of the tail keeps its model order in [0, .45).
Gates:
  G1 development power (filtered population): anomaly AUC vs all others >= .98 in each family.
  G2 development mechanism (reference model = 326-aggregate known-family model trained on the other tables, so
     its tail holds labelled positives): among promoted pairs that carry a label, positives >= 50 %; labelled AP
     after promotion >= before; PU-AP (unlabelled as negative) after >= before.
  G3 evaluation promoted set: median shared hands >= .7 x tail median; at most 5 promoted pairs in any table;
     mean pairwise cosine of the promoted signatures >= .2.
Prediction if uploaded: public P in [-.006, +.025] of .938 (composite [-.004, +.018] vs .876); E, B unchanged.
Outputs: experiments/hidden_family_20260911/{freeze,promotion-results}.json, promoted-eval.parquet,
         outputs_anomaly_20260911/submission_anomaly_promote.csv (+ validation, receipt.json)
"""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import sys, json, glob, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected
from pu_pairs.step32_crossphase_delta_model import EXP as XEXP, PARAMS, X, sha
from pu_pairs.step37_conditional_anomaly import PU, INF, FAMS, normal_scores_binned, mahalanobis, q
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = CV / 'experiments' / 'hidden_family_20260911'
OUTD = CV / 'outputs_anomaly_20260911'
MIN_SH = 40
TAU_BLOCK = .5
PROMOTE_LO, PROMOTE_HI = .45, .5


def anomaly(frame, cols):
    Z = normal_scores_binned(frame.select(cols).to_numpy(), frame['shared_hands'].to_numpy().astype(float))
    d2, _ = mahalanobis(Z)
    return Z, d2


def mean_pair_cos(M, rng, k=3000):
    if len(M) < 2:
        return None
    i = rng.integers(0, len(M), k); j = rng.integers(0, len(M), k); ok = i != j
    a = M[i[ok]]; b = M[j[ok]]
    return float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))


def block_players(pairs, block_ids):
    b = pairs.filter(pl.col('pair_id').is_in(block_ids))
    return set(b['player_1'].to_list()) | set(b['player_2'].to_list())


def promote(frame, risk, d2, pairs):
    """Returns (promoted mask, threshold, shares-player mask). frame has pair_id; pairs has pair_id, player_1, player_2."""
    blk = risk >= TAU_BLOCK
    T = float(np.median(d2[blk]))
    bp = block_players(pairs, frame.filter(pl.Series(blk))['pair_id'].to_list())
    pp = frame.select('pair_id').join(pairs, on='pair_id', how='left')
    shares = (pp['player_1'].is_in(list(bp)) | pp['player_2'].is_in(list(bp))).fill_null(False).to_numpy()
    prom = (~blk) & (d2 >= T) & (~shares)
    return prom, T, shares


def rerank(risk, d2, prom):
    """New risk vector: block unchanged; promoted in [.45, .5) by d2; remaining tail scaled into [0, .45) preserving order."""
    new = risk.copy()
    blk = risk >= TAU_BLOCK
    if prom.sum():
        r = d2[prom]; rk = (np.argsort(np.argsort(r)) + 1) / (prom.sum() + 1)
        new[prom] = PROMOTE_LO + (PROMOTE_HI - PROMOTE_LO) * rk
    rest = (~blk) & (~prom)
    if rest.sum():
        rr = risk[rest]; new[rest] = PROMOTE_LO * (rr - rr.min()) / (rr.max() - rr.min() + 1e-12) * (1 - 1e-6)
    return new


def main():
    t0 = time.perf_counter(); OUT.mkdir(exist_ok=True); OUTD.mkdir(exist_ok=True)
    rng = np.random.default_rng(7329)
    protected = check_protected()
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'protected': protected, 'min_shared_hands': MIN_SH, 'tau_block': TAU_BLOCK, 'promote_band': [PROMOTE_LO, PROMOTE_HI],
              'threshold_rule': 'median anomaly of the block (risk >= .5)', 'exclusion': 'tail pairs sharing a player with any block pair',
              'gates': {'G1_family_auc_min': .98, 'G2_promoted_labelled_precision_min': .5, 'G2_ap_not_worse': True, 'G3_shared_hands_ratio_min': .7, 'G3_table_max': 5, 'G3_coherence_min': .2},
              'prediction_if_uploaded': 'public P in [-.006, +.025] of .938 (composite [-.004, +.018] vs .876); E, B unchanged',
              'inputs': {'v13': sha(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv'), 'base': sha(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv')}}
    write_json(OUT / 'freeze.json', freeze)
    lab = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'label', 'behavior_family')
    # ---------------- development ----------------
    files = sorted(glob.glob(str(PU / 'fold1' / 'batch_*-pairs.parquet'))) + sorted(glob.glob(str(PU / 'fold2' / 'batch_*-pairs.parquet')))
    sch = pl.scan_parquet(files[0]).collect_schema().names()
    ca_mean = [c for c in sch if c.startswith('ca_') and c.endswith('_mean')]
    lab_dev = pl.read_parquet(XEXP / 'lab_on_dev.parquet'); single = [c for c in lab_dev.columns if c.endswith('_mean') or c.endswith('_std')]
    dev = pl.concat([pl.read_parquet(f, columns=['pair_id', 'table_id', 'shared_hands', *single, *ca_mean]) for f in files], how='vertical_relaxed')
    dev = dev.filter(pl.col('shared_hands') >= MIN_SH).join(lab, on='pair_id', how='left').with_columns(pl.col('label').fill_null(-1)).sort('pair_id')
    fold_tables = set(dev['table_id'].unique())
    tr = lab_dev.filter(~pl.col('table_id').is_in(list(fold_tables))).join(lab, on='pair_id')
    ref = lgb.LGBMClassifier(**PARAMS).fit(X(tr, single), (tr['label'] == 1).to_numpy().astype(int))
    risk_d = ref.predict_proba(X(dev, single))[:, 1]
    Zd, d2d = anomaly(dev, ca_mean)
    y_pos = (dev['label'] == 1).to_numpy(); y_neg = (dev['label'] == 0).to_numpy(); fam = dev['behavior_family'].to_numpy(); labm = y_pos | y_neg
    # development pairs list for spill-over (all pairs on the fold tables)
    seats = pl.scan_parquet(ORIGINAL / 'data' / 'seats.parquet').select('hand_id', 'player_id')
    hands = pl.scan_parquet(ORIGINAL / 'data' / 'hands.parquet').filter(pl.col('phase') == 'development').select('hand_id', 'table_id')
    dev_pairs = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv').select('pair_id', 'player_1', 'player_2')
    # unlabelled development pairs: pair_id format check -> derive players from pair_id when possible
    sample = dev.filter(pl.col('label') == -1)['pair_id'][:3].to_list()
    unl = dev.filter(pl.col('label') == -1).select('pair_id')
    if all('__' in s or '|' in s or '_' in s for s in sample):
        sep = '__' if '__' in sample[0] else ('|' if '|' in sample[0] else None)
        if sep:
            unl = unl.with_columns(pl.col('pair_id').str.split(sep).list.get(0).alias('player_1'), pl.col('pair_id').str.split(sep).list.get(1).alias('player_2'))
    if 'player_1' not in unl.columns:
        # fall back: players from the all-pairs batch receipts are not stored; use the seats to rebuild (table-level co-seating gives the same pair ids as build_allpairs_fold)
        unl = None
    pairs_d = dev_pairs if unl is None else pl.concat([dev_pairs, unl.select('pair_id', 'player_1', 'player_2')], how='vertical_relaxed')
    res = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'dev': {'pairs': dev.height, 'tables': len(fold_tables), 'labelled_pos': int(y_pos.sum()), 'labelled_neg': int(y_neg.sum()), 'unlabelled_pairs_resolved_for_spillover': None if unl is None else unl.height,
                                                                          'pair_id_sample': sample}}
    res['dev']['anomaly'] = {'ap_pos_vs_labelled_neg': float(average_precision_score(y_pos[labm], d2d[labm])), 'auc_pos_vs_all_others': float(roc_auc_score(y_pos, d2d)),
                             'by_family_auc_vs_all_others': {f: float(roc_auc_score((fam == f)[~y_pos | (fam == f)], d2d[~y_pos | (fam == f)])) for f in FAMS}}
    res['dev']['gate_G1'] = bool(all(v >= .98 for v in res['dev']['anomaly']['by_family_auc_vs_all_others'].values()))
    prom_d, T_d, shares_d = promote(dev, risk_d, d2d, pairs_d)
    new_d = rerank(risk_d, d2d, prom_d)
    tail_d = risk_d < TAU_BLOCK
    res['dev']['reference_model'] = {'ap_labelled_before': float(average_precision_score(y_pos[labm], risk_d[labm])), 'ap_labelled_after': float(average_precision_score(y_pos[labm], new_d[labm])),
                                     'pu_ap_before': float(average_precision_score(y_pos, risk_d)), 'pu_ap_after': float(average_precision_score(y_pos, new_d)),
                                     'labelled_pos_in_tail': int((tail_d & y_pos).sum()), 'labelled_pos_promoted': int((prom_d & y_pos).sum()), 'labelled_neg_promoted': int((prom_d & y_neg).sum()),
                                     'unlabelled_promoted': int((prom_d & (dev['label'] == -1).to_numpy()).sum()), 'promoted_total': int(prom_d.sum()), 'threshold_T': T_d,
                                     'tail_pairs_excluded_for_spillover': int((tail_d & shares_d & (d2d >= T_d)).sum()), 'labelled_pos_in_tail_excluded_for_spillover': int((tail_d & shares_d & (d2d >= T_d) & y_pos).sum()),
                                     'labelled_pos_in_tail_below_T': int((tail_d & y_pos & (d2d < T_d)).sum()), 'promoted_per_table_max': int(dev.filter(pl.Series(prom_d)).group_by('table_id').len()['len'].max()) if prom_d.sum() else 0,
                                     'promoted_unlabelled_risk_known_q': q(risk_d[prom_d & (dev['label'] == -1).to_numpy()], [.5, .9])}
    lp = res['dev']['reference_model']['labelled_pos_promoted']; ln = res['dev']['reference_model']['labelled_neg_promoted']
    res['dev']['gate_G2'] = {'precision_labelled': lp / max(lp + ln, 1), 'passed': bool(lp / max(lp + ln, 1) >= .5 and res['dev']['reference_model']['ap_labelled_after'] >= res['dev']['reference_model']['ap_labelled_before']
                                                                                 and res['dev']['reference_model']['pu_ap_after'] >= res['dev']['reference_model']['pu_ap_before'])}
    print(json.dumps(res['dev'], indent=1), flush=True)
    # ---------------- evaluation ----------------
    ev = pl.concat([pl.read_parquet(f, columns=['pair_id', 'table_id', *ca_mean]) for f in sorted(glob.glob(str(INF / 'batch_*-pairs.parquet')))], how='vertical_relaxed')
    epairs = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2', 'shared_hands')
    v13 = pl.read_csv(ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv').select('pair_id', pl.col('risk_score').alias('risk'))
    ev = ev.join(epairs, on='pair_id').join(v13, on='pair_id').sort('pair_id'); assert ev.height == 112540
    risk_e = ev['risk'].to_numpy().astype(float)
    Ze, d2e = anomaly(ev, ca_mean)
    prom_e, T_e, shares_e = promote(ev, risk_e, d2e, epairs)
    tail_e = risk_e < TAU_BLOCK
    sh = ev['shared_hands'].to_numpy().astype(float)
    per_table = ev.filter(pl.Series(prom_e)).group_by('table_id').len().sort('len', descending=True)
    coh = mean_pair_cos(Ze[prom_e], rng)
    res['eval'] = {'pairs': ev.height, 'threshold_T': T_e, 'block_pairs': int((~tail_e).sum()), 'tail_above_T': int((tail_e & (d2e >= T_e)).sum()), 'excluded_spillover': int((tail_e & (d2e >= T_e) & shares_e).sum()),
                   'promoted': int(prom_e.sum()), 'promoted_tables': per_table.height, 'promoted_per_table_max': int(per_table['len'][0]) if per_table.height else 0,
                   'promoted_v13_risk_q': q(risk_e[prom_e], [.1, .5, .9]), 'promoted_v13_ge_0.1': int((risk_e[prom_e] >= .1).sum()), 'promoted_v13_lt_0.01': int((risk_e[prom_e] < .01).sum()),
                   'promoted_shared_hands_median': float(np.median(sh[prom_e])) if prom_e.sum() else None, 'tail_shared_hands_median': float(np.median(sh[tail_e])),
                   'promoted_d2_q': q(d2e[prom_e], [.1, .5, .9]), 'tail_d2_q': q(d2e[tail_e], [.5, .9, .99, .999]), 'block_d2_q': q(d2e[~tail_e], [.1, .5, .9]),
                   'coherence_mean_pair_cos': {'promoted': coh, 'block': mean_pair_cos(Ze[~tail_e], rng), 'tail_population': mean_pair_cos(Ze[tail_e][rng.choice(tail_e.sum(), 5000, replace=False)], rng)},
                   'tail_share_rate_with_block_player': float(shares_e[tail_e].mean()), 'above_T_share_rate': float(shares_e[tail_e & (d2e >= T_e)].mean())}
    ratio = res['eval']['promoted_shared_hands_median'] / res['eval']['tail_shared_hands_median'] if prom_e.sum() else 0
    res['eval']['gate_G3'] = {'shared_hands_ratio': ratio, 'passed': bool(ratio >= .7 and res['eval']['promoted_per_table_max'] <= 5 and (coh or 0) >= .2)}
    print(json.dumps(res['eval'], indent=1), flush=True)
    ev.select('pair_id', 'table_id', 'shared_hands', 'risk').with_columns(pl.Series('anom', d2e), pl.Series('promoted', prom_e), pl.Series('shares_block_player', shares_e)).write_parquet(OUT / 'promoted-eval.parquet')
    # ---------------- candidate ----------------
    base = pl.read_csv(CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'); cols = base.columns
    j = base.join(ev.select('pair_id', 'risk').with_columns(pl.Series('new_risk', rerank(risk_e, d2e, prom_e))), on='pair_id', how='left')
    assert j.height == base.height and float(np.abs(j['risk_score'].to_numpy() - j['risk'].to_numpy()).max()) < 1e-9
    out = j.with_columns(pl.col('new_risk').alias('risk_score')).select(cols)
    assert out['pair_id'].equals(base['pair_id'])
    blk_old = base.filter(pl.col('risk_score') >= TAU_BLOCK); blk_new = out.filter(pl.col('risk_score') >= TAU_BLOCK)
    assert blk_old.height == blk_new.height and (blk_old['risk_score'].to_numpy() == blk_new['risk_score'].to_numpy()).all()
    for c in cols:
        if c != 'risk_score':
            assert out[c].equals(base[c]), c
    # tail order preserved outside the promoted set
    rest_old = base.join(ev.select('pair_id').filter(pl.Series(tail_e & ~prom_e)), on='pair_id')['risk_score'].to_numpy()
    rest_new = out.join(ev.select('pair_id').filter(pl.Series(tail_e & ~prom_e)), on='pair_id')['risk_score'].to_numpy()
    assert (np.argsort(rest_old, kind='stable') == np.argsort(rest_new, kind='stable')).all()
    dest = OUTD / 'submission_anomaly_promote.csv'; out.write_csv(dest)
    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(dest, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    write_json(OUTD / 'submission_anomaly_promote-validation.json', vrep); assert vrep['valid'], vrep
    res['candidate'] = {'file': str(dest.relative_to(CV)), 'sha256': sha(dest), 'base_sha256': freeze['inputs']['base'], 'block_pairs_unchanged': blk_old.height, 'promoted_pairs': int(prom_e.sum()),
                        'validation_valid': bool(vrep['valid']), 'gates_passed': bool(res['dev']['gate_G1'] and res['dev']['gate_G2']['passed'] and res['eval']['gate_G3']['passed']),
                        'prediction': freeze['prediction_if_uploaded'], 'submitted': False}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    write_json(OUT / 'promotion-results.json', res); write_json(OUTD / 'receipt.json', {'freeze': freeze, 'results': res})
    print(json.dumps(res['candidate'], indent=1), flush=True)


if __name__ == '__main__':
    main()
