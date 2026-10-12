"""Step 19: build two DIAGNOSTIC probe submissions on top of the scored v13 coverage file.

The public score is 0.7*PairAP + 0.2*EvidenceMAP@5 + 0.1*BehaviorMAP, and we cannot measure any of the three
components locally at leaderboard scale (local pair AP is saturated, the hidden family is unseen). Two probes
that change exactly one component each let the three public components be solved exactly from three scores:
  probe A (blank evidence):  every evidence cell = NO_EVIDENCE   -> S_A = 0.7P + 0.1B
  probe B (blank behavior):  predicted_behavior = other_coordination everywhere (the excluded class) -> S_B = 0.7P + 0.2E
  with S13 = 0.7P + 0.2E + 0.1B:  E = (S13 - S_A)/0.2,  B = (S13 - S_B)/0.1,  P = (S_A + S_B - S13)/0.7
This is diagnosis of WHERE the .077 gap to the leader sits. It uses two of the five daily uploads and does not
tune anything. Nothing here is uploaded by the agent; the user decides.
Output: Claude Version/outputs_probes_20260911/{probe_blank_evidence.csv, probe_blank_behavior.csv, receipt.json}
"""
import os
import sys
import json
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, write_json
sys.path.insert(0, str(ORIGINAL / 'src'))
from validation import validate_submission
import polars as pl

SRC = ORIGINAL / 'outputs_v13_coverage' / 'submission_conditional_coverage.csv'
OUT = ORIGINAL / 'Claude Version' / 'outputs_probes_20260911'
EV = [f'evidence_hand_{i}' for i in range(1, 6)]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pl.read_csv(SRC, schema_overrides={'risk_score': pl.Float64})
    assert base.height == 112540 and base.columns == ['pair_id', 'risk_score', 'predicted_behavior'] + EV
    a = base.with_columns([pl.lit('NO_EVIDENCE').alias(c) for c in EV])
    b = base.with_columns(pl.lit('other_coordination').alias('predicted_behavior'))
    receipt = {'source': str(SRC), 'source_sha256': sha(SRC), 'purpose': 'component diagnosis only; not tuning', 'files': {}}
    for name, frame in [('probe_blank_evidence.csv', a), ('probe_blank_behavior.csv', b)]:
        path = OUT / name
        frame.write_csv(path, float_precision=None)
        rep = validate_submission(path, ORIGINAL / 'data' / 'evaluation_pairs.csv', ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet', ORIGINAL / 'data' / 'sample_submission.csv')
        d = rep.to_dict()
        receipt['files'][name] = {'sha256': sha(path), 'rows': frame.height, 'valid': d.get('valid'), 'errors': {k: v for k, v in d.items() if k != 'valid' and v}}
        print(name, 'valid=', d.get('valid'), flush=True)
    # sanity: risk column identical to source (string-level compare after rewrite)
    back = pl.read_csv(OUT / 'probe_blank_evidence.csv', schema_overrides={'risk_score': pl.Float64})
    receipt['risk_identical_to_source'] = bool((back['risk_score'] == base['risk_score']).all() and (back['pair_id'] == base['pair_id']).all())
    receipt['behavior_identical_in_probe_A'] = bool((back['predicted_behavior'] == base['predicted_behavior']).all())
    receipt['solve'] = {'E': '(S13 - S_A)/0.2', 'B': '(S13 - S_B)/0.1', 'P': '(S_A + S_B - S13)/0.7', 'S13': 0.86005}
    write_json(OUT / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
