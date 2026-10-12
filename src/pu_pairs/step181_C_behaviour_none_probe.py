"""Step 181 (Claude): DIAGNOSTIC build that identifies B, the behaviour MAP term of the composite, which has never been measured
(strict bound [.6976, 1]). One column changes: predicted_behavior := 'none' for all 112,540 rows of the .90821 file. 'none' is a legal
label (validation.py BEHAVIORS) and is what data/sample_submission.csv carries for every row. risk_score and evidence_hand_1..5 stay
byte-identical, so under the reconstruction in OBS 19.130 only the .1 term can move and read = .90821 - .1 (B_current - B_none).
THIS FILE IS EXPECTED TO SCORE ABOUT .085 BELOW THE CLEAN BEST and must never be selected as a final submission. Registration
outputs_step181_20260916/prediction_step181.json. Never uploads. Usage: step181_C_behaviour_none_probe.py REGSHA"""
import os
for k in ('OMP_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
import polars as pl

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step181_20260916'
REG = OUT / 'prediction_step181.json'
COLS = ['pair_id', 'risk_score', 'predicted_behavior'] + [f'evidence_hand_{i}' for i in range(1, 6)]


def main():
    t0 = time.perf_counter()
    from evidence_aug.paths import check_protected
    from evidence_aug.step165_C_e2e_audit import sha
    from validation import validate_submission
    reg_sha = sys.argv[1]
    assert sha(REG) == reg_sha, 'registration hash mismatch'
    reg = json.loads(REG.read_text(encoding='utf-8'))
    prot = check_protected()
    bpath = POKER / reg['inputs']['base']['path']
    assert sha(bpath) == reg['inputs']['base']['sha256']
    base = pl.read_csv(bpath, infer_schema=False)
    assert base.columns == COLS and base.height == 112540
    result = base.with_columns(pl.lit('none').alias('predicted_behavior')).select(COLS)
    assert result.select('pair_id', 'risk_score', *COLS[3:]).equals(base.select('pair_id', 'risk_score', *COLS[3:]))
    csv = OUT / 'submission_step181_behaviour_none_probe.csv'
    assert not csv.exists(), 'preserve a built file'
    result.write_csv(csv)
    A, B = bpath.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
    assert len(A) == len(B)
    diff = [k for k in range(len(A)) if A[k] != B[k]]
    for k in diff:                                     # only field 3 may differ, and only to 'none'
        a, b = A[k].split(b','), B[k].split(b',')
        assert a[:2] == b[:2] and a[3:] == b[3:] and b[2] == b'none'
    val = validate_submission(csv, POKER / 'data/evaluation_pairs.csv', POKER / 'data/hands.parquet',
                              POKER / 'data/seats.parquet', POKER / 'data/sample_submission.csv')
    assert not val.errors, val
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'output': str(csv), 'output_sha256': sha(csv), 'base_sha256': sha(bpath),
           'rows_changed': len(diff), 'behaviour_counts_before': dict(base['predicted_behavior'].value_counts().iter_rows()),
           'validation': str(val)[:300], 'risk_and_evidence_identical': True, 'protected_unchanged': True,
           'no_submission_actions': True, 'runner_sha256': sha(Path(__file__)), 'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'receipt_step181.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('DONE', json.dumps({k: rec[k] for k in ('output_sha256', 'rows_changed', 'behaviour_counts_before')}), flush=True)


if __name__ == '__main__':
    main()
