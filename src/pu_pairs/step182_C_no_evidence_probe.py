"""Step 182 (Claude): DIAGNOSTIC build that separates P (pair AP) from E (evidence MAP@5). STEP181 measured B and confirmed the
composite is .7P + .2E + .1B to within .0016 (OBS 19.131), which leaves .7P + .2E = .80971 jointly identified and P and E individually
unknown. Withdrawing every evidence list sets E = 0, so the read is .7P + .1B and P follows to about +-.003. One change: the five
evidence columns become 'NO_EVIDENCE' on all 112,540 rows (what data/sample_submission.csv carries for every row); risk_score and
predicted_behavior stay byte-identical. THIS FILE IS EXPECTED TO SCORE ABOUT .135 BELOW THE CLEAN BEST and must never be selected as a
final submission. Registration outputs_step182_20260916/prediction_step182.json. Never uploads.
Usage: step182_C_no_evidence_probe.py REGSHA"""
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
OUT = CV / 'outputs_step182_20260916'
REG = OUT / 'prediction_step182.json'
EV = [f'evidence_hand_{i}' for i in range(1, 6)]
COLS = ['pair_id', 'risk_score', 'predicted_behavior'] + EV


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
    result = base.with_columns([pl.lit('NO_EVIDENCE').alias(c) for c in EV]).select(COLS)
    assert result.select(COLS[:3]).equals(base.select(COLS[:3]))
    csv = OUT / 'submission_step182_no_evidence_probe.csv'
    assert not csv.exists(), 'preserve a built file'
    result.write_csv(csv)
    A, B = bpath.read_bytes().split(b'\n'), csv.read_bytes().split(b'\n')
    assert len(A) == len(B)
    diff = [k for k in range(len(A)) if A[k] != B[k]]
    for k in diff:                                   # only the five evidence fields may differ, and only to NO_EVIDENCE
        a, b = A[k].split(b','), B[k].split(b',')
        assert a[:3] == b[:3] and set(b[3:]) == {b'NO_EVIDENCE'}
    val = validate_submission(csv, POKER / 'data/evaluation_pairs.csv', POKER / 'data/hands.parquet',
                              POKER / 'data/seats.parquet', POKER / 'data/sample_submission.csv')
    assert not val.errors, val
    assert check_protected() == prot
    rec = {'registration_sha256': reg_sha, 'output': str(csv), 'output_sha256': sha(csv), 'base_sha256': sha(bpath),
           'rows_changed': len(diff), 'validation': str(val)[:300], 'risk_and_behaviour_identical': True,
           'protected_unchanged': True, 'no_submission_actions': True, 'runner_sha256': sha(Path(__file__)),
           'seconds': round(time.perf_counter() - t0, 1)}
    (OUT / 'receipt_step182.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('DONE', json.dumps({k: rec[k] for k in ('output_sha256', 'rows_changed')}), flush=True)


if __name__ == '__main__':
    main()
