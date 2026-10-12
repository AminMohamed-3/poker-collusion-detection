"""Build only the preregistered rank candidate after its independent screen passes."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
BASE = CAMPAIGN/'outputs/submission_sp277_ci276_on_step272.csv'
BASE_SHA = '0493a807d81b7f3cc5ec54a74c86d08c362b72305fceee8e6af9044740990fc9'
DEST = HERE/'submission_fast_rank80_20_on_current.csv'

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exploratory', action='store_true')
    args = parser.parse_args()
    assert not DEST.exists(), 'Preserve any already built candidate'
    result = HERE.parent/'global_pu/global_pu_results.json'
    gate = json.loads(result.read_text())
    primary = gate['primary_gate']
    assert len(primary['fold_gains']) == 2
    recomputed = bool(primary['pooled_gain'] > 0 and all(value > 0 for value in primary['fold_gains'])
                      and primary['bootstrap_lower'] > 0)
    assert primary['pass'] is recomputed, 'Recorded gate must match its numerical inputs'
    deviation = HERE/'EXPLORATORY_DEVIATION.md'
    if not recomputed:
        assert args.exploratory and deviation.exists(), 'Failed gate: explicit documented exploration required'
    verifier = HERE.parent/'global_pu/returned_verified.json'
    assert json.loads(verifier.read_text())['verified'] is True
    assert gate['input_hashes']['returned_verified'] == sha(verifier)
    assert sha(BASE) == BASE_SHA
    with BASE.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    keys = np.asarray([row['pair_id'] for row in rows])
    assert len(keys) == 112540 and len(set(keys)) == len(keys)
    predictions = pl.read_parquet(HERE/'fast_probabilities.parquet')
    assert np.array_equal(predictions['pair_id'].to_numpy(), keys)
    p = predictions.select('p_none','p_dt','p_sp','p_ci').to_numpy().astype(np.float64)
    assert p.shape == (len(keys), 4) and np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1, atol=1e-5)
    replay = np.load(HERE/'replay_first1024.npz', allow_pickle=False)['p']
    assert np.array_equal(replay, p[:1024]), 'Exact fresh-process replay required'
    base = np.asarray([float(row['risk_score']) for row in rows])
    fast = 1-p[:,0]
    r2base = np.rint(2*rankdata(base, method='average')).astype(np.int64)
    r2fast = np.rint(2*rankdata(fast, method='average')).astype(np.int64)
    numerator = 4*r2base+r2fast
    score = numerator/(10*len(keys))
    assert np.isfinite(score).all() and np.all((score >= 0) & (score <= 1))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row, value in zip(rows, score):
        writer.writerow({**row, 'risk_score': format(float(value), '.17g')})
    text = buffer.getvalue()
    restored = list(csv.DictReader(io.StringIO(text)))
    assert len(restored) == len(rows)
    assert np.array_equal(np.asarray([float(row['risk_score']) for row in restored]), score)
    fixed = [name for name in fields if name != 'risk_score']
    assert all(all(old[name] == new[name] for name in fixed) for old, new in zip(rows, restored))
    # The unique integer numerator defines all mathematical ties; float serialization preserves them.
    assert len(np.unique(score)) == len(np.unique(numerator))
    DEST.write_text(text, encoding='utf-8', newline='')
    receipt = {
        'candidate': str(DEST), 'sha256': sha(DEST), 'rows': len(rows),
        'decision_mode': 'explicit_exploratory_information_test' if args.exploratory else 'gate_passed',
        'primary_gate_pass': recomputed, 'primary_gate': primary,
        'exploratory_deviation_sha256': sha(deviation) if args.exploratory else None,
        'reference_sha256': BASE_SHA, 'prediction_sha256': sha(HERE/'fast_probabilities.parquet'),
        'replay_sha256': sha(HERE/'replay_first1024.npz'), 'global_pu_quality_sha256': sha(result),
        'protocol_sha256': sha(HERE/'INTEGRATION_PROTOCOL.md'),
        'builder_sha256': sha(Path(__file__)),
        'full_context_sha256': sha(HERE/'full_context.npz'),
        'context_receipt_sha256': sha(HERE/'context_receipt.json'),
        'runtime_sha256': sha(HERE/'runtime.json'),
        'global_pu_prep_and_metric_amendment_sha256': sha(HERE.parent/'global_pu/PREP_PROTOCOL.md'),
        'risk_formula': '(4 * twice_average_rank(current) + twice_average_rank(Fast)) / (10*N)',
        'unique_risk_scores': len(np.unique(score)),
        'behavior_and_evidence_text_exact': True, 'fresh_replay_exact': True,
        'fresh_replay_scope': 'First 1024 rows only; not a replay of all 112540 predictions.',
        'csv_float_roundtrip_exact': True, 'public_submission_performed': False,
        'scope': 'One fixed information experiment; compatible prefix evidence is not current-system validation.',
    }
    (HERE/'candidate_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
