"""Step 55: candidate D = candidate C (R1 spill-over demotion, step 49) + a RAW-low tier (OBS 14.34-14.35).
Label-free on evaluation: block pairs (v13 risk >= .5) that are NOT in R1 and whose raw-statistics view (765 non-ca
features, production recipe refit on all labelled pairs, step 53) scores below THR are moved to just below the block,
above the R1 pairs: tier 1 (RAW-low) in [.495, .4999] ordered by RAW, tier 2 (R1) in [.49, .4949] ordered by
risk_noT (as in C). Behaviour and evidence columns stay byte-identical to the measured .87625 base.
Development validation (labels used only here): at THR = .05 the tier holds 2 of 159 labelled positives at full length
(1.3 %), 4.7 % when histories are thinned to the evaluation median (~57 hands, step 54) and 30 unlabelled block pairs
(28 of them in [.5,.9)); the unlabelled [.99,1] pairs (undisclosed positives) behave like the labelled ones.
Scenario sweep (label-free): F block false positives, q2 = FP share of R1, q1 = FP share of tier 1, the rest placed
among the other block pairs with weight (1 - risk)^gamma; cells whose AP_before is consistent with the public P (.938)
give the pre-registered interval. The prediction is written BEFORE the CSV.
Usage: python -m pu_pairs.step55_rawlow_tier_candidate --thr .05 --interval LO HI"""
import os, sys, json, argparse
os.environ['POLARS_MAX_THREADS'] = '2'
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected, sha
sys.path.insert(0, str(ORIGINAL / 'src'))

SP = CV / 'experiments' / 'spillover_20260911'
VD = CV / 'experiments' / 'viewdis_20260911'
BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
OUT = CV / 'outputs_rawlow_20260911'


def ap_block(is_fp, n_pos_total):
    tp = (~is_fp).astype(float)
    cum = np.cumsum(tp); ranks = np.arange(1, len(tp) + 1)
    return float((tp * cum / ranks).sum() / n_pos_total)


def sweep(risk, raw, in1, in2, orders, draws=200, seed=55):
    """orders: dict name -> permutation of block indices (top first). Placement of the non-tier false positives:
    weight (1 - risk)^gamma * (1 - RAW)^beta; beta = 0 is RAW-blind (the step-49a convention), beta = 3 is RAW-informed."""
    n = len(risk); rng = np.random.default_rng(seed)
    others = np.where(~in1 & ~in2)[0]; idx1 = np.where(in1)[0]; idx2 = np.where(in2)[0]
    n1, n2 = len(idx1), len(idx2)
    cells = []
    for beta in (0, 3):
        for F in (200, 250, 300, 350, 400, 450, 500):
            for q2 in (.85, .9, .95, 1.0):
                for q1 in (.5, .65, .8, .9, .95):
                    f1, f2 = int(round(q1 * n1)), int(round(q2 * n2))
                    rest = F - f1 - f2
                    if rest < 0 or rest > len(others):
                        continue
                    for gamma in (2, 3, 4, 6):
                        w = ((1 - risk[others]) + 1e-3) ** gamma * ((1 - raw[others]) + 1e-3) ** beta; w = w / w.sum()
                        acc = {k: [] for k in orders}; b = []
                        for _ in range(draws):
                            is_fp = np.zeros(n, dtype=bool)
                            is_fp[rng.choice(idx1, size=f1, replace=False)] = True
                            is_fp[rng.choice(idx2, size=f2, replace=False)] = True
                            is_fp[rng.choice(others, size=rest, replace=False, p=w)] = True
                            npos = (n - F) / .97
                            b.append(ap_block(is_fp, npos))
                            for k, o in orders.items():
                                acc[k].append(ap_block(is_fp[o], npos))
                        b = float(np.mean(b))
                        cell = {'beta': beta, 'F': F, 'q2_R1': q2, 'q1_tier1': q1, 'gamma': gamma, 'AP_before': round(b, 4)}
                        for k in orders:
                            a = float(np.mean(acc[k])); cell[f'AP_{k}'] = round(a, 4); cell[f'd{k}_composite'] = round(.7 * (a - b), 4)
                        cell['dD_minus_C_composite'] = round(cell['dD_composite'] - cell['dC_composite'], 4)
                        cell['dS_minus_C_composite'] = round(cell['dS_composite'] - cell['dC_composite'], 4)
                        cells.append(cell)
    return cells


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--thr', type=float, default=.05); ap.add_argument('--interval', nargs=2, type=float, default=None)
    ap.add_argument('--dev-summary', default=''); ap.add_argument('--build', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res = pl.read_parquet(SP / 'eval_rescoring.parquet').select('pair_id', 'risk_v13', 'dominated', 'risk_noT', 'band')
    ve = pl.read_parquet(VD / 'view_eval.parquet').select('pair_id', 'risk', 'r_RAW', 'r_CA', 'R1', 'R4any', 'shared_hands')
    d = res.join(ve, on='pair_id', validate='1:1')
    assert d.height == 909 and float((d['risk_v13'] - d['risk']).abs().max()) < 1e-9
    d = d.with_columns((pl.col('dominated') & (pl.col('risk_noT').fill_nan(0) < .1)).alias('T2'))
    assert (d['T2'] == d['R1']).all()
    d = d.with_columns((~pl.col('T2') & (pl.col('r_RAW') < args.thr)).alias('T1')).sort('risk_v13', descending=True)
    risk = d['risk_v13'].to_numpy(); in1 = d['T1'].to_numpy(); in2 = d['T2'].to_numpy()
    o1 = np.where(in1)[0][np.argsort(-d['r_RAW'].to_numpy()[in1], kind='stable')]
    o2 = np.where(in2)[0][np.argsort(-d['risk_noT'].fill_nan(0).to_numpy()[in2], kind='stable')]
    rest = np.where(~in1 & ~in2)[0]
    order_C = np.concatenate([np.where(~in2)[0], o2]); order_D = np.concatenate([rest, o1, o2])
    # S = smooth variant: the remaining block ordered by logit(FULL) + logit(RAW), then tier 1, then R1
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    raw = d['r_RAW'].to_numpy()
    order_S = np.concatenate([rest[np.argsort(-(lg(risk[rest]) + lg(raw[rest])), kind='stable')], o1, o2])
    cells = sweep(risk, raw, in1, in2, {'C': order_C, 'D': order_D, 'S': order_S})
    cons = [c for c in cells if .93 <= c['AP_before'] <= .946]
    def rng_(key, sub): return [min(c[key] for c in sub), max(c[key] for c in sub)] if sub else None
    scen = {'thr': args.thr, 'n_tier1': int(in1.sum()), 'n_tier2_R1': int(in2.sum()), 'tier1_by_band': d.filter(pl.col('T1')).group_by('band').len().sort('band').to_dicts(),
            'tier1_R4any': int(d.filter(pl.col('T1'))['R4any'].sum()), 'tier1_shared_hands_median': float(d.filter(pl.col('T1'))['shared_hands'].median()),
            'grid': {'beta_RAW_placement': [0, 3], 'F': [200, 250, 300, 350, 400, 450, 500], 'q2_R1': [.85, .9, .95, 1.0], 'q1_tier1': [.5, .65, .8, .9, .95], 'gamma': [2, 3, 4, 6], 'draws': 200},
            'consistent_cells_public_P_.93_.946': len(cons)}
    for beta in (0, 3):
        cb = [c for c in cons if c['beta'] == beta]
        scen[f'beta{beta}'] = {'consistent_cells': len(cb), 'F_values': sorted({c['F'] for c in cb}),
                               'dC': rng_('dC_composite', cb), 'dD': rng_('dD_composite', cb), 'dS': rng_('dS_composite', cb),
                               'dD_minus_C': rng_('dD_minus_C_composite', cb), 'dS_minus_C': rng_('dS_minus_C_composite', cb),
                               'by_q1': {str(q): {'dD': rng_('dD_composite', [c for c in cb if c['q1_tier1'] == q]), 'dD_minus_C': rng_('dD_minus_C_composite', [c for c in cb if c['q1_tier1'] == q]),
                                                  'dS_minus_C': rng_('dS_minus_C_composite', [c for c in cb if c['q1_tier1'] == q])} for q in (.5, .65, .8, .9, .95)}}
    scen['cells'] = cells
    write_json(SP.parent / 'viewdis_20260911' / f'eval_scenarios_D_thr{args.thr}.json', scen)
    print(json.dumps({k: v for k, v in scen.items() if k != 'cells'}, indent=1))
    if not args.build:
        return
    T1 = d.filter(pl.col('T1')).sort('r_RAW', descending=True); T2 = d.filter(pl.col('T2')).sort('risk_noT', descending=True, nulls_last=True)
    base = pl.read_csv(BASE)
    assert base.columns == ['pair_id', 'risk_score', 'predicted_behavior', 'evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4', 'evidence_hand_5']
    r = base['risk_score'].to_numpy(); in_sliver = int(((r >= .49) & (r < .5)).sum())
    def spread(x, lo, hi):
        x = np.asarray(x, dtype=float); return lo + (hi - lo) * (x - x.min()) / max(x.max() - x.min(), 1e-9)
    m1 = dict(zip(T1['pair_id'].to_list(), spread(T1['r_RAW'].to_numpy(), .495, .4999).tolist()))
    m2 = dict(zip(T2['pair_id'].to_list(), spread(T2['risk_noT'].fill_nan(0).to_numpy(), .49, .4949).tolist()))
    mapping = {**m1, **m2}; assert len(mapping) == T1.height + T2.height
    DEST = OUT / f'submission_rawlow_tier_thr{args.thr}_plus_R1.csv'
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    pred = {'created_at': now, 'candidate': str(DEST.relative_to(CV)), 'rule': f'D = C (R1: dominated & risk_noT < .1, {T2.height}) + tier 1 (block & not R1 & RAW < {args.thr}, {T1.height})',
            'tier1_by_band': scen['tier1_by_band'], 'tier2_by_band': T2.group_by('band').len().sort('band').to_dicts(),
            'construction': (f'Base = the measured .87625 submission. Tier 1 ({T1.height} pairs) receives risk in [.495, .4999] ordered by the RAW view; tier 2 = the R1 pairs of candidate C ({T2.height}) '
                             f'receives [.49, .4949] ordered by risk_noT; every other row is byte-identical. Existing base pairs in [.49,.5): {in_sliver}.'),
            'hypothesis': ('OBS 14.34-14.35: the raw-statistics view (765 non-ca features) is null on the false-positive mechanisms found in the block (spill-over, short-history noise entering '
                           'through the conditional-action residuals) and stays ~.99 on genuine pairs at evaluation lengths; block pairs whose RAW view is < .05 are mostly false positives, '
                           'so moving them just below the block (above the tail) raises PairAP.'),
            'label_free': 'Evaluation side uses only our own risk, the RAW view trained on development labels (the same labels the production model uses) and the R1 rule. No leaderboard feedback used.',
            'dev_validation_summary': args.dev_summary,
            'prediction': {'interval_composite_delta': args.interval, 'scenario_file': f'experiments/viewdis_20260911/eval_scenarios_D_thr{args.thr}.json',
                           'scenario_ranges': {k: scen[k] for k in ('beta0', 'beta3')}},
            'decision_rule': ('Score - .87625 measures the joint false-positive share of the two tiers. > +.003: at least one mechanism confirmed; C (already built) then separates them. '
                              '<= -.003: the demoted pairs are mostly genuine; both tiers close (C would only be worth uploading if D fails by less than C\'s own interval allows). '
                              'If D and C are both measured, D - C > +.002 confirms the RAW-low tier and the next tier is RAW < .1 (31 more pairs); D - C < -.002 closes it. No re-tuning on outcomes.'),
            'not_uploaded_by_claude': True}
    write_json(OUT / f'prediction_rawlow_thr{args.thr}.json', pred)
    cand = base.with_columns(pl.col('pair_id').replace_strict(mapping, default=None, return_dtype=pl.Float64).alias('new')).with_columns(
        pl.coalesce([pl.col('new'), pl.col('risk_score')]).alias('risk_score')).drop('new')
    cand.write_csv(DEST)
    chk = pl.read_csv(DEST)
    same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score')
    changed = int((chk['risk_score'] != base['risk_score']).sum())
    C = pl.read_csv(CV / 'outputs_spillover_20260911' / 'submission_spillover_demotion_R1.csv')
    superset = int(((chk['risk_score'] != base['risk_score']) & (C['risk_score'] != base['risk_score'])).sum())
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    prot = check_protected()
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'rule': pred['rule'],
               'n_tier1': T1.height, 'n_tier2': T2.height, 'rows_with_changed_risk': changed, 'rows_changed_in_both_C_and_D': superset, 'non_risk_columns_identical_to_base': bool(same_cols),
               'block_ge_.5_after': int((chk['risk_score'] >= .5).sum()), 'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
    write_json(OUT / f'receipt_rawlow_thr{args.thr}.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1))
    print('validation:', json.dumps(vrep)[:600])


if __name__ == '__main__':
    main()
