"""Fixed locations. The original project is read-only; new artifacts stay in Claude Version."""
from pathlib import Path
import hashlib
import json
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')

ORIGINAL = Path('C:/Data/Projects/Money Making/work/poker')
CV = Path(__file__).resolve().parents[2]
SRC = CV / 'src'
EXP = CV / 'experiments' / 'evidence_augmentation_20260910'
EXPERIMENTS = ORIGINAL / 'experiments'
UTILITY = EXPERIMENTS / 'bet_size_utility_20260910'
SCREEN = ORIGINAL / 'reports' / 'conditional-action-experiment'
BUNDLE = SCREEN / 'screen-results' / 'conditional-evidence.joblib'
BUNDLE_SHA = '331341f7175d8ccc19f0a12792db7742b5e9c8ac4132770885199c08523d16ee'
SAVED_SCORES = {
    'original_roles': SCREEN / 'screen-results' / 'conditional-evidence-scores.parquet',
    'conditional_roles': ORIGINAL / 'reports' / 'conditional-behavior-experiment' / 'fold0' / 'conditional_behavior-scores.parquet',
}
ROLE_FILES = {
    'original_roles': ORIGINAL / 'outputs_v2' / 'validation_pairs.csv',
    'conditional_roles': ORIGINAL / 'reports' / 'conditional-behavior-experiment' / 'fold0' / 'pair-predictions.csv',
}
PROTECTED = {
    'outputs_v13_coverage/submission_conditional_coverage.csv': '418084a4',
    'outputs_v14_evidence_portfolio/submission_history_evidence.csv': '51dc5821',
    'outputs_v14_evidence_portfolio/submission_history_rank_fusion.csv': 'b47adae2',
    'outputs_v14_evidence_portfolio/submission_seed_evidence.csv': 'dc35cb0c',
    'outputs_v14_evidence_portfolio/submission_seed_history_mean.csv': '7955e016',
    'outputs_v14_evidence_portfolio/submission_three_way_mean.csv': 'c49550e6',
    'outputs_v15_bucket/submission_bucket_risk_blend.csv': '14b0d767',
}
LABELS = {'none': 0, 'directed_transfer': 1, 'soft_play': 2, 'coordinated_isolation': 3}
INVERSE = {v: k for k, v in LABELS.items()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_protected():
    """Compare the seven protected submissions against the full hashes recorded by the previous agent."""
    plan = read_json(UTILITY / 'study-plan.json')['protected']
    out = {}
    for rel, prefix in PROTECTED.items():
        digest = sha(ORIGINAL / rel)
        assert digest == plan[rel] and digest.startswith(prefix), rel
        out[rel] = digest
    return out
