"""Step 49: candidate C = spill-over demotion. Label-free on evaluation: for block pairs (v13 risk >= .5) that share a player
with a higher-risk block pair and whose risk collapses when the dominating partner's hands are removed (step 48,
experiments/spillover_20260911/eval_rescoring.parquet), move the risk to just below the block, order preserved by the
residual risk. Behaviour and evidence columns stay byte-identical to the measured .87625 base
(outputs_chronology_20260911/submission_chronology_frac.csv). Pre-registered prediction is written BEFORE the CSV.
Usage: python step49_spillover_demotion_candidate.py --rule R1|R2"""
import os, sys, json, argparse
os.environ['POLARS_MAX_THREADS'] = '2'
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected, sha
sys.path.insert(0, str(ORIGINAL / 'src'))

RULES = {
    'R1': ('dominated & risk_noT < .1', lambda d: d['dominated'] & (d['risk_noT'].fill_nan(0) < .1)),
    'R2': ('dominated & cop_T >= .6 & risk_noT < .1', lambda d: d['dominated'] & (d['cop_T'] >= .6) & (d['risk_noT'].fill_nan(0) < .1)),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rule', default='R1'); ap.add_argument('--interval', nargs=2, type=float, default=None)
    ap.add_argument('--dev-summary', default='')
    args = ap.parse_args()
    desc, rule = RULES[args.rule]
    OUT = CV / 'outputs_spillover_20260911'; OUT.mkdir(exist_ok=True)
    BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
    DEST = OUT / f'submission_spillover_demotion_{args.rule}.csv'
    res = pl.read_parquet(CV / 'experiments' / 'spillover_20260911' / 'eval_rescoring.parquet')
    D = res.filter(rule(res)).sort('risk_noT', descending=True, nulls_last=True)
    base = pl.read_csv(BASE)
    assert base.columns == ['pair_id', 'risk_score', 'predicted_behavior', 'evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4', 'evidence_hand_5']
    r = base['risk_score'].to_numpy()
    in_sliver = int(((r >= .49) & (r < .5)).sum())
    rn = D['risk_noT'].fill_nan(0).to_numpy()
    new_risk = .49 + .0099 * (rn - rn.min()) / max(rn.max() - rn.min(), 1e-9)
    mapping = dict(zip(D['pair_id'].to_list(), new_risk.tolist()))
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    pred = {
        'created_at': now, 'candidate': str(DEST.relative_to(CV)), 'rule': f'{args.rule}: {desc}', 'n_demoted': D.height,
        'by_band': D.group_by('band').len().sort('band').to_dicts(),
        'construction': (f'Base = the measured .87625 submission (risk = v13, behaviour = v13, evidence = chronology-frac). The {D.height} demoted pairs receive risk in [.49, .4999], '
                         'ordered by their residual risk without the dominating partner\'s hands; every other row is byte-identical. Existing pairs with base risk in [.49,.5): %d.' % in_sliver),
        'hypothesis': ('Step 48 (OBSERVATIONS 14.30): dominated block pairs whose risk rests entirely on hands in which the dominating partner is dealt are spill-over '
                       'false positives of the partner pair; development truth shows ring positives never share a listed hand (0/53) and genuine pairs keep their risk '
                       'without the partner\'s hands. Moving them just below the block raises PairAP if most are false positives and costs little if some are not '
                       '(they stay above the 111,631 tail pairs).'),
        'label_free': 'Evaluation side uses only our own risk, our own evidence lists, and who was dealt. Development labels were used only to validate the rule (step 48b) and to read the truth-listing property.',
        'dev_validation_summary': args.dev_summary,
        'prediction': {'interval_composite_delta': args.interval, 'scenario_file': 'experiments/spillover_20260911/eval_scenarios.json',
                       'reasoning': 'Scenario sweep over block false-positive counts consistent with public P = .938 and the false-positive share of the demoted set; B and E unchanged.'},
        'decision_rule': 'The score is a measurement of the false-positive share of D. If the composite rises by more than +.003, spill-over is confirmed and the rule extends to the next tier (risk_noT < .5) in a later candidate; if it falls, the dominated pairs were mostly genuine and the lever closes. No re-tuning on the outcome.',
        'not_uploaded_by_claude': True,
    }
    write_json(OUT / f'prediction_{args.rule}.json', pred)
    cand = base.with_columns(pl.col('pair_id').replace_strict(mapping, default=None, return_dtype=pl.Float64).alias('new')).with_columns(
        pl.coalesce([pl.col('new'), pl.col('risk_score')]).alias('risk_score')).drop('new')
    cand.write_csv(DEST)
    chk = pl.read_csv(DEST)
    same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score')
    changed = int((chk['risk_score'] != base['risk_score']).sum())
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    prot = check_protected()
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'rule': f'{args.rule}: {desc}',
               'n_demoted': D.height, 'rows_with_changed_risk': changed, 'non_risk_columns_identical_to_base': bool(same_cols), 'block_ge_.5_after': int((chk['risk_score'] >= .5).sum()),
               'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
    write_json(OUT / f'receipt_{args.rule}.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1))
    print('validation:', json.dumps(vrep)[:600])
    print('protected:', json.dumps(prot)[:300])


if __name__ == '__main__':
    main()