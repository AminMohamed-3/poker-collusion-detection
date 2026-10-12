"""Close the rejected attention experiment without changing its plan or predictions."""
import json,shutil
from datetime import datetime,timezone
from robust_pair_experiment import ROOT,sha
from sequence_evidence_candidate import verify_source_cache

out=ROOT/'reports/attention-reranker-experiment'
assert not (out/'input-audit.json').exists()
verify_source_cache('development')
source=ROOT/'reports/conditional-action-experiment'
mapping={'development_labels.csv':ROOT/'data/development_labels.csv','held-pairs.csv':ROOT/'reports/tabicl-screen/held-pairs.csv','run_conditional_policy_screen.py':ROOT/'src/run_conditional_policy_screen.py'}
inventory={}
for name,expected in json.loads((source/'policy-plan.json').read_text())['files'].items():
    p=mapping.get(name,source/name)
    assert sha(p)==expected,(name,str(p))
    inventory[name]={'local_path':str(p.relative_to(ROOT)),'sha256':expected}
snapshot=out/'source-snapshot'; snapshot.mkdir(exist_ok=False)
sources={}
for p in sorted((ROOT/'src').glob('*.py')):
    shutil.copy2(p,snapshot/p.name); sources[p.name]=sha(snapshot/p.name)
result=json.loads((out/'screen.json').read_text())
assert result['selected_for_confirmation'] is None
audit={'completed_at':datetime.now(timezone.utc).isoformat(),'raw_development_hashes_verified':True,'policy_inputs':inventory,'source_snapshot':sources,'screen_sha256':sha(out/'screen.json'),'disposition':'Rejected. No further fits, subgroup rescue, evaluation inference or submission.'}
(out/'input-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(out/'REPORT.md').write_text('''# Attention reranker: rejected

Six fixed local CPU fits tested a small set-attention correction to existing top-50 evidence rankings. Strict nested pool separation included nine normal-reference crossfits and three first-stage classifiers/rankers. No held labels entered the neural worker. Five structural tests passed; each saved model reproduced its predictions exactly.

Full-history baseline MAP was .525402. Three-seed pairwise, listwise, and their equal ensemble scored .495074, .496704, and .503013. Every candidate failed the frozen full/short gain and family-regression gates. Some shortened-history means improved, but isolation deteriorated. No subgroup rescue or evaluation prediction followed.

The supplied GPU stopped completing bounded CUDA allocation probes; local PyTorch CPU completed all six fits. No driver modification or GPU reset was attempted. The precise host cause is unknown.

All raw development cache hashes and original action-reference input hashes verify. The first inventory attempt used three incorrect relative paths and failed before writing an audit; this corrected audit uses explicit paths and confirms their original hashes. This was a path-resolution failure, not a data-hash mismatch.

Sources, models, predictions, plans, nested exclusion checks and negative results are retained privately. These reused validation partitions do not provide an untouched estimate of future performance.
''',encoding='utf-8')
print(json.dumps({'input_audit_sha256':sha(out/'input-audit.json'),'source_files':len(sources),'policy_inputs':len(inventory),'rejected':True}))
