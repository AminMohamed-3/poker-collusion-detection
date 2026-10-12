"""Read official submission statuses and retain scored receipts locally."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run([str(ROOT / '.venv/Scripts/kaggle.exe'), 'competitions', 'submissions',
        '-c', 'detect-suspicious-value-transfers-in-poker', '--csv'], check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    now = datetime.now(timezone.utc).isoformat()
    pairs = [('outputs_v7_capacity', 'submission_capacity.csv'), ('outputs_v8_equity', 'submission_equity.csv'),
             ('outputs_v9_ensemble', 'submission_ensemble.csv'), ('outputs_v10_tabicl', 'submission_tabicl.csv'),
             ('outputs_v11_conditional', 'submission_conditional_risk.csv'),
             ('outputs_v11_conditional', 'submission_conditional_evidence.csv'),
             ('outputs_v11_conditional', 'submission_conditional_combined.csv'),
             ('outputs_v12_behavior', 'submission_conditional_behavior.csv'),
             ('outputs_v13_coverage', 'submission_conditional_coverage.csv')]
    for directory, filename in pairs:
        matched = [r for r in rows if r['fileName'] == filename]
        if not matched:
            continue
        assert len(matched) == 1, 'Investigate duplicate submission instead of choosing silently'
        row = matched[0]
        out = ROOT / directory
        receipt = {'observed_at': now, 'source': 'Official Kaggle CLI competitions submissions', **row}
        receipt_name = (Path(filename).stem + '.receipt.json') if directory in {'outputs_v11_conditional','outputs_v12_behavior','outputs_v13_coverage'} else 'submission_receipt.json'
        (out / receipt_name).write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ['ref', 'fileName', 'status', 'publicScore']}))
    (ROOT / 'reports/evidence-public-calibration/submissions-observed.json').write_text(json.dumps({'observed_at': now, 'submissions': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
