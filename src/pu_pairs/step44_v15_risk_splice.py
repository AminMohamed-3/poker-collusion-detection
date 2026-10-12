"""Step 44: candidate B = Codex's untested v15 bucket risk blend (risk only) spliced onto the .876 chronology candidate
(v13 behavior + chronology-frac evidence). Pre-registered prediction is written BEFORE the CSV is assembled.
Label-free: no development labels, no leaderboard numbers enter the construction. Protected files are read only."""
import sys, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import polars as pl, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_aug.paths import ORIGINAL, CV, write_json, check_protected, sha
OUT = CV / 'outputs_v15splice_20260911'; OUT.mkdir(exist_ok=True)
BASE = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
V15 = ORIGINAL / 'outputs_v15_bucket' / 'submission_bucket_risk_blend.csv'
DEST = OUT / 'submission_v15risk_chrono_evidence.csv'
now = datetime.now(timezone.utc).isoformat(timespec='seconds')
pred = {
    'created_at': now, 'candidate': str(DEST.relative_to(CV)),
    'construction': 'risk_score from outputs_v15_bucket/submission_bucket_risk_blend.csv (Codex v15 equal blend of the v13 risk with a 1,217-feature private-strength-category model, admitted 2026-09-10, never uploaded); predicted_behavior and evidence_hand_1..5 byte-identical to the .876 chronology candidate.',
    'hypothesis': 'Tests Codex v15 risk in isolation against the .876 base: B (.976) and E (.608) are unchanged, so score - .876 = .7 * (P15 - .938).',
    'label_free_diagnostics': 'experiments/v15_diag_20260911/v15_vs_v13_labelfree.json: Spearman .972 vs v13, top-909 overlap 876, 39 pairs leave / 17 enter the >= .5 block, no .5 <-> .1 crossings, >= .5 counts per shared-hands bin unchanged (38-55: 306 -> 304). The blend is a mild reorder of v13 and does not address the length-induced false positives of 14.23.',
    'prediction': {'point_composite_delta': +0.001, 'interval_composite_delta': [-0.006, +0.007], 'implied_P15_interval': [0.929, 0.948],
                   'reasoning': 'Codex development gates: known-fold weighted pair+behavior +.0004 (worst -.0002); withheld-family 1 %-prevalence AP +.023 mean (worst -.017); 4 % of the block changes. Expected effect on public P within +-.01; slight positive prior from the sparse-prevalence gate because public prevalence (~.8 %) is the sparse regime.'},
    'decision_rule': 'Whatever it scores is the measurement of P15. If P15 > .938 by more than .003 the v15 risk becomes the base for any later combination (e.g. with the step-39 promotion re-derived on v15); otherwise v13 stays. No re-tuning on the outcome.',
    'not_uploaded_by_claude': True,
}
write_json(OUT / 'prediction.json', pred)
base = pl.read_csv(BASE); v15 = pl.read_csv(V15)
assert base.columns == v15.columns == ['pair_id', 'risk_score', 'predicted_behavior', 'evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4', 'evidence_hand_5']
j = base.join(v15.select('pair_id', pl.col('risk_score').alias('r15'), pl.col('predicted_behavior').alias('b15')), on='pair_id', how='left')
assert j['r15'].null_count() == 0 and len(j) == 112540
beh_same = int((j['predicted_behavior'] == j['b15']).sum())
cand = j.select('pair_id', pl.col('r15').alias('risk_score'), 'predicted_behavior', 'evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3', 'evidence_hand_4', 'evidence_hand_5')
# keep the base row order byte-for-byte for the non-risk columns
cand.write_csv(DEST)
# checks: non-risk columns identical to base; risk identical to v15 as text is not required (float formatting), but values must match exactly
chk = pl.read_csv(DEST)
same_cols = all((chk[c] == base[c]).all() for c in base.columns if c != 'risk_score')
risk_match = bool(np.array_equal(chk['risk_score'].to_numpy(), v15.join(base.select('pair_id'), on='pair_id', how='inner').sort(pl.col('pair_id').replace_strict(dict(zip(base['pair_id'].to_list(), range(len(base)))), return_dtype=pl.Int64))['risk_score'].to_numpy()))
from validation import validate_submission  # noqa: E402
vrep = validate_submission(DEST, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
prot = check_protected()
receipt = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'dest': str(DEST), 'sha256': sha(DEST), 'base_sha256': sha(BASE), 'v15_sha256': sha(V15),
           'behavior_identical_base_vs_v15': beh_same == 112540, 'non_risk_columns_identical_to_base': bool(same_cols), 'risk_identical_to_v15': risk_match,
           'validation': vrep, 'protected_check': prot, 'prediction_written_first': True, 'do_not_upload_by_claude': True}
write_json(OUT / 'receipt.json', receipt)
print(json.dumps({k: v for k, v in receipt.items() if k not in ('validation', 'protected_check')}, indent=1))
print('validation:', json.dumps(vrep)[:600])
print('protected:', json.dumps(prot)[:300])
