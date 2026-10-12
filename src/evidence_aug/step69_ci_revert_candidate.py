"""Step 69: the CI-branch probe candidate (X1) -- stage-2 evidence for predicted transfer and
soft play, stage-1 evidence for predicted coordinated isolation.

Why.  The five public readings decompose exactly (step 68): the r382 hybrid beats the struct
stage 2 by +.0143 evidence MAP, but the struct stage 2 as a whole beats stage 1 by -.0002.  The
hybrid routes predicted CI to the struct model, so the CI lists of `M3_stage2_chrono3` and
`M3_stage2_r382DTSP` are byte-identical (verified here) -- whatever the struct stage 2 does wrong
on evaluation, it does inside the CI branch as well.  Step 68b found the mechanism candidate: the
CI stage 2 rewrites 45 % of the evaluation top-5 lists (top-5 overlap with stage 1 .546) against
23 % on held (.769), while its held gain (+.1249 MAP) is the largest of the three families.

X1 reverts exactly that branch.  Because the two files differ in nothing else, the public
difference is an exact attribution:

    score(X1) - score(S_stage2_r382DTSP) = -0.2 * w_CI * delta_CI

with w_CI the share of evaluation true positives predicted CI and delta_CI the per-pair MAP gain
of the CI stage 2 over stage 1 there.  One upload settles the sign; the mirror candidate (stage-1
DT/SP + struct CI) is algebraically redundant.

No model is fitted.  The evidence columns of predicted-CI rows are copied from the stage-1
submission; every other byte comes from the source candidate.  Read-only apart from
`outputs_ci_revert_20260912/`.
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from evidence_aug.paths import CV, ORIGINAL, check_protected, sha, write_json

OUT = CV / 'outputs_ci_revert_20260912'
EV = [f'evidence_hand_{i}' for i in range(1, 6)]
CI = 'coordinated_isolation'

STAGE1 = CV / 'outputs_chronology_20260911' / 'submission_chronology_frac.csv'
SOURCES = {
    # tag: (source candidate whose DT/SP stage-2 evidence is kept, public score of that source)
    'S_r382DTSP_CIstage1': (CV / 'outputs_stage2_r382_20260911' / 'submission_S_stage2_r382DTSP.csv', .89336),
    'M3_r382DTSP_CIstage1': (CV / 'outputs_stage2_r382_20260911' / 'submission_M3_stage2_r382DTSP.csv', .89447),
}
PRIMARY = 'S_r382DTSP_CIstage1'


def ev_array(d):
    return np.column_stack([d[c].to_numpy().astype(object) for c in EV])


def build(tag, src_path, base_public, s1, s1_ev):
    src = pl.read_csv(src_path)
    assert src['pair_id'].to_list() == s1['pair_id'].to_list(), 'row order differs from stage 1'
    assert (src['predicted_behavior'].to_numpy() == s1['predicted_behavior'].to_numpy()).all()

    src_ev = ev_array(src)
    ci = (src['predicted_behavior'].to_numpy() == CI)
    new_ev = src_ev.copy()
    new_ev[ci] = s1_ev[ci]

    changed = (new_ev != src_ev).any(1)
    kept = (src_ev != s1_ev).any(1) & ~ci
    out = src.with_columns([pl.Series(c, new_ev[:, i]) for i, c in enumerate(EV)])

    # the multiset of listed hands per row must be unchanged in kind (five distinct hand ids)
    assert out.select(EV).null_count().to_numpy().sum() == 0
    assert all(len(set(r)) == 5 for r in new_ev[:2000])

    dest = OUT / f'submission_{tag}.csv'
    out.write_csv(dest)

    sys.path.insert(0, str(ORIGINAL / 'src'))
    from validation import validate_submission  # noqa: E402
    vrep = validate_submission(dest, ORIGINAL / 'data' / 'evaluation_pairs.csv',
                               ORIGINAL / 'data' / 'hands.parquet', ORIGINAL / 'data' / 'seats.parquet',
                               ORIGINAL / 'data' / 'sample_submission.csv').to_dict()
    write_json(OUT / f'{tag}-validation.json', vrep)

    rec = {'tag': tag, 'source': str(src_path.relative_to(CV)), 'source_public': base_public,
           'stage1': str(STAGE1.relative_to(CV)), 'sha256': sha(dest),
           'source_sha256': sha(src_path), 'rows': out.height,
           'rows_predicted_CI': int(ci.sum()),
           'rows_whose_evidence_changed_vs_source': int(changed.sum()),
           'rows_still_carrying_stage2_evidence (DT/SP)': int(kept.sum()),
           'risk_column_identical_to_source': bool(
               np.array_equal(out['risk_score'].to_numpy(), src['risk_score'].to_numpy())),
           'behavior_column_identical_to_source': bool(
               (out['predicted_behavior'].to_numpy() == src['predicted_behavior'].to_numpy()).all()),
           'validation_valid': vrep.get('valid', vrep)}
    print(f'  {tag}: {dest.name}')
    for k, v in rec.items():
        if k not in ('tag',):
            print(f'      {k}: {v}')
    return rec


def main():
    check_protected()
    OUT.mkdir(exist_ok=True)
    s1 = pl.read_csv(STAGE1)
    s1_ev = ev_array(s1)

    ci_n = int((s1['predicted_behavior'].to_numpy() == CI).sum())
    print('=' * 100)
    print('STEP 69.  CI-branch revert candidates (X1)')
    print('=' * 100)
    print(f'  stage-1 evidence source: {STAGE1.name}  (sha {sha(STAGE1)[:8]})')
    print(f'  predicted coordinated_isolation rows: {ci_n} of {s1.height}')

    # cross-check the claim that the hybrid and the struct stage 2 share their CI lists
    ch = pl.read_csv(CV / 'outputs_stage2_20260911' / 'submission_M3_stage2_chrono3.csv')
    hy = pl.read_csv(CV / 'outputs_stage2_r382_20260911' / 'submission_M3_stage2_r382DTSP.csv')
    m = (s1['predicted_behavior'].to_numpy() == CI)
    same_ci = bool((ev_array(ch)[m] == ev_array(hy)[m]).all())
    print(f'  struct(chrono3) and r382-hybrid CI lists identical: {same_ci}')
    assert same_ci, 'the hybrid does not route CI to the struct model -- the probe logic changes'

    # pre-registration BEFORE writing any candidate file
    pred = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'study': 'step69 CI-branch revert probe',
        'primary_candidate': PRIMARY,
        'label_free': ('no evaluation label was used; the only inputs are two existing submission '
                       'files and the predicted_behavior column produced by the frozen v12 behaviour '
                       'model. No model was fitted and no leaderboard outcome entered the construction.'),
        'identity': ('score(X1) - score(source) = -0.2 * w_CI * delta_CI exactly, because the two '
                     'files differ only in the evidence lists of predicted-CI rows'),
        'point_estimate_composite_vs_source': +0.0031,
        'composite_interval_vs_source': [-0.0005, +0.0075],
        'reasoning': ('step 68b puts the evaluation CI top-5 overlap with stage 1 at .546 against '
                      '.769 on held; propagating the held DT/SP transfer rate (~90 %, measured for '
                      'the r382 increment) through the measured struct total of -.0002 MAP leaves '
                      'delta_CI ~ -.048 on evaluation against +.1249 on held, worth +.0031 composite '
                      'to revert.  Under the null (the struct stage 2 transferred at 0 in every '
                      'family) the difference is 0.'),
        'decision_rule': ('>= +.002: the CI branch is the leak; drop the CI stage 2 from every '
                          'candidate and re-target E2 at rebuilding it.  Within +-.002: the CI branch '
                          'is not the leak, the struct shortfall is spread across families, keep the '
                          'CI stage 2 and treat E1/E2/E3 as the only evidence levers.  <= -.002: the '
                          'CI stage 2 does transfer and the DT/SP struct models are the leak instead.'),
        'not_uploaded_by_claude': True,
        'upload_order': [PRIMARY, 'M3_r382DTSP_CIstage1 (hold; only if the primary reads >= +.002)'],
    }
    write_json(OUT / 'prediction.json', pred)
    print(f'\n  pre-registration written: {OUT / "prediction.json"}')
    print(f'  point estimate {pred["point_estimate_composite_vs_source"]:+.4f} composite, '
          f'interval {pred["composite_interval_vs_source"]}')
    print()

    recs = {t: build(t, p, s, s1, s1_ev) for t, (p, s) in SOURCES.items()}
    write_json(OUT / 'receipt.json', {'created_at': pred['created_at'], 'candidates': recs,
                                      'stage1_sha256': sha(STAGE1),
                                      'struct_and_hybrid_share_CI_lists': same_ci})
    check_protected()
    print()
    print(f'  PRIMARY (next upload): {OUT / ("submission_" + PRIMARY + ".csv")}')
    print(f'  sha256 {recs[PRIMARY]["sha256"]}')
    print('DONE')


if __name__ == '__main__':
    main()
