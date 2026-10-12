from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import tarfile
from conditional_action_features import INPUTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/conditional-action-experiment'


def main():
    files = {p.relative_to(OUT).as_posix(): p for p in (OUT / 'action-cache').glob('*') if p.is_file()}
    assert len(list((OUT / 'action-cache').glob('*.parquet'))) == 20
    files.update({'development_labels.csv': ROOT / 'data/development_labels.csv',
        'held-pairs.csv': ROOT / 'reports/tabicl-screen/held-pairs.csv',
        'run_conditional_policy_screen.py': ROOT / 'src/run_conditional_policy_screen.py',
        'extraction-plan.json': OUT / 'extraction-plan.json'})
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': INPUTS,
        'parameters': {'n_estimators': 100, 'learning_rate': .05, 'num_leaves': 15, 'min_child_samples': 50,
            'reg_lambda': 10, 'n_jobs': 4, 'verbosity': -1, 'random_state': 39117, 'deterministic': True, 'force_col_wise': True},
        'training': 'Confirmed-non-target training-pool actions only, at most 128 sampled rows per pair and 100000 total. Three inner crossfit partitions score training rows; a fourth fit scores the outer-held pools. No pair labels from outer held pools enter any policy model.',
        'features': 'Standardized action residuals and products with conditional partner-strength residuals; fixed scales and bounds. Means and absolute maxima per shared hand. These are descriptive features, not calibrated tests of information leakage or counterfactual utility.',
        'next_gate': 'Fit fixed expanded v5 pair and v8 evidence models. Advance pair features only if full/short AP both nondecreasing and mean gain >=.001, with no sparse-prior decline; advance evidence only for MAP gain >=.005 and no family drop beyond .025. A passing screen requires further pool/family confirmation, not automatic submission.',
        'limitations': 'Reused outer fold; shortening retains the full-observed-history public style references just as existing player baselines do.',
        'files': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}}
    path = OUT / 'policy-plan.json'; assert not path.exists(); path.write_text(json.dumps(plan, indent=2) + '\n')
    archive = OUT / 'policy-bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        t.add(path, arcname='policy-plan.json')
        for n, p in files.items(): t.add(p, arcname=n)
    receipt = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
    (OUT / 'policy-bundle-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n'); print(json.dumps(receipt))


if __name__ == '__main__': main()
