"""Consolidate completed-study decisions without changing frozen evaluations."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

H = Path(__file__).resolve().parent
D = H / 'data_expansion'
M = H / 'mhc_local'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

for name, expected in json.loads((D/'profile_repair/auditor-freeze.json').read_text()).items():
    assert sha(D/name) == expected, name
assert sha(D/'evaluate_augmentation.py') == json.loads((D/'augmentation/evaluator-freeze.json').read_text())['sha256']

profile = D/'profile_repair'
legal = json.loads((profile/'pilot-audit.json').read_text())
pair = json.loads((profile/'pair-domain-audit.json').read_text())
coverage = json.loads((profile/'emitted-coverage-audit.json').read_text())
replay = json.loads((profile/'replay-audit.json').read_text())
assert legal['serialized_accounting_checks_passed'] and coverage['coverage_passed']
assert replay['all_serialized_actions_replayed_legally']
admission = {
    'plan_sha256': sha(profile/'pilot-plan.json'),
    'legal_and_emitted_coverage_passed': True,
    'hand_domain_passed': not legal['domain_discrepancy_flag'],
    'pair_domain_auc': pair['auc'],
    'pair_domain_ceiling': pair['diagnostic_threshold'],
    'pair_domain_passed': pair['auc'] <= pair['diagnostic_threshold'],
    'large_scale_admitted': False,
    'detector_utility_tested_for_this_version': False,
    'note': 'Combined final decision supersedes the coarse pilot-audit scale_up_admitted field. Do not equate legal validity or hand-level realism with complete-history realism.',
}
assert not admission['pair_domain_passed']
(profile/'final-admission.json').write_text(json.dumps(admission, indent=2))

receipts = []
for folder in [M/'fold0', M/'omit_directed_transfer', M/'omit_soft_play', M/'omit_coordinated_isolation', D/'augmentation/resampled_control', D/'augmentation/synthetic']:
    for p in sorted(folder.glob('*/receipt.json')):
        r = json.loads(p.read_text())
        model = p.parent/('model.pt' if (p.parent/'model.pt').exists() else 'model.joblib')
        assert sha(model) == r['model_sha256'], model
        assert sha(p.parent/'predictions.npz') == r['predictions_sha256'], p
        assert not r['held_labels_used'], p
        receipts.append({'path': p.relative_to(H).as_posix(), 'sha256': sha(p)})
assert len(receipts) == 52, len(receipts)
plan = json.loads((D/'augmentation/plan.json').read_text())
protected = {name: sha(H.parent/name) for name in plan['protected']}
assert protected == plan['protected']
decisions = json.loads((D/'augmentation/evaluation.json').read_text())['decisions']
assert not any(row['passed_data_screen'] for row in decisions.values())
result = {
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'architecture_fits': 30,
    'augmentation_fits': 22,
    'all_model_and_prediction_hashes_verified': True,
    'receipts': receipts,
    'protected_submissions': protected,
    'production_changed': False,
    'new_kaggle_upload': False,
    'rl_training_started': False,
    'four_pilots_are_related_versions_not_independent_sources': True,
}
(H/'research-completion-20260910.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['receipts','protected_submissions']}, indent=2))
