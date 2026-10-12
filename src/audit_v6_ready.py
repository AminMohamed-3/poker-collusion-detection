"""Freeze final v6 readiness and comparison without changing predictions."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import json
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata as metadata
import polars as pl
from features import file_sha256 as sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs_v6'


def main():
    destination = OUT/'ready.json'
    if destination.exists():
        raise FileExistsError('Preserve readiness audit')
    source = ROOT/'outputs_v5/submission_event_tails.csv'
    candidate = OUT/'submission_sequence_evidence.csv'
    before = pl.read_csv(source, infer_schema=False)
    after = pl.read_csv(candidate, infer_schema=False)
    fixed = ['pair_id', 'risk_score', 'predicted_behavior']
    assert before.select(fixed).equals(after.select(fixed))
    assert sha(source) == 'e2ed0f5c25be000437ccb2ae753649b5643cf328e84c545ed9590b57e84d0e76'
    assert sha(candidate) == 'bc309294d0e2feecc2cded72ed0330498485ccea55dfa433c8c6a5ebf05457f7'
    validation_path = OUT/'submission_sequence_evidence.validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['valid'] and not validation['errors']
    evidence = [f'evidence_hand_{i}' for i in range(1, 6)]
    old = before.select(evidence).rows()
    new = after.select(evidence).rows()
    overlaps = [len(set(a) & set(b)) for a, b in zip(old, new, strict=True)]
    changed = [a != b for a, b in zip(old, new, strict=True)]
    risk = after['risk_score'].cast(pl.Float64).to_list()
    cases_path = ROOT/'reports/selected-v6-evidence-cases.json'
    cases = json.loads(cases_path.read_text())
    assert cases['case_count'] == 5 and cases['evidence_records'] == 25
    assert cases['submission_sha256'] == sha(candidate)
    methods = ROOT/'reports/methodology-v6-draft.md'
    words = len(methods.read_text(encoding='utf-8').split())
    assert words <= 1500
    rank_files = sorted((OUT/'rank_batches').glob('batch_*.parquet'))
    ranking = pl.read_parquet(rank_files)
    assert len(rank_files) == 20 and ranking.height == after.height
    assert ranking['pair_id'].n_unique() == after.height
    files = [candidate, source, OUT/'sequence_ranker.joblib', OUT/'fit.json',
             OUT/'submission_sequence_evidence.provenance.json', validation_path,
             cases_path, methods, ROOT/'README-v6.md', Path(__file__)]
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'state': 'Validated exploratory candidate; no public submission or score verified',
        'strict_screen_passed': False, 'strict_confirmation_passed': False,
        'risk_and_behavior_strings_unchanged': True,
        'validation': validation,
        'ranked_pair_hands': int(ranking['shared_hands'].sum()),
        'pairs_with_changed_evidence_order_or_membership': sum(changed),
        'mean_top5_overlap': sum(overlaps)/len(overlaps),
        'risk_at_least_half': {
            'pairs': sum(r >= .5 for r in risk),
            'mean_top5_overlap': sum(o for o, r in zip(overlaps, risk) if r >= .5)/sum(r >= .5 for r in risk),
        },
        'methodology_words': words,
        'versions': {p: metadata.version(p) for p in ['polars', 'numpy', 'lightgbm', 'scikit-learn', 'joblib']},
        'sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
    }
    destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
