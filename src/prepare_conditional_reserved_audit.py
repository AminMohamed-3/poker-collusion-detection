"""Lock a less-reused partition for auditing already frozen v11/v12 recipes."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from datetime import datetime,timezone
import json
import shutil
import tarfile
import polars as pl
from robust_pair_experiment import ROOT,sha

OUT=ROOT/'reports/conditional-reserved-audit'


def main():
    OUT.mkdir(exist_ok=False)
    metadata=pl.read_parquet(ROOT/'outputs_v2/pairs_development.parquet').select('pair_id','table_id')
    previous=[ROOT/'reports/tabicl-screen/held-pairs.csv']+[ROOT/'reports/tabicl-confirmation'/f'fold{i}/held-pairs.csv' for i in [1,2]]
    reused=set().union(*[set(pl.read_csv(p)['table_id']) for p in previous])
    held=metadata.filter(~pl.col('table_id').is_in(sorted(reused))).sort('pair_id')
    assert len(reused)==228 and held['table_id'].n_unique()==169
    held.write_csv(OUT/'held-pairs.csv')
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'selection':'All 169 pools outside the union of the three recurring 100-pool partitions, selected without labels, scores or subgroup outcomes.',
        'limits':'This is less-reused confirmation, NOT an untouched holdout. All 397 pools appear in some historical validation record, including the earlier soft-gate experiment. They have also been used as training rows. No independent private-performance estimate is claimed.',
        'held_pairs_sha256':sha(OUT/'held-pairs.csv'),'held_pools':sorted(held['table_id'].unique()),'pairs':held.height,
        'prior_partition_files':{str(p.relative_to(ROOT)):sha(p) for p in previous},
        'frozen_candidates':{'v11_audit_sha256':sha(ROOT/'outputs_v11_conditional/ready-audit.json'),'v12_audit_sha256':sha(ROOT/'outputs_v12_behavior/ready-audit.json')},
        'scope':'Refit the same normal references, v5 tree, v2 family classifier, v10 TabICL recipe, v8 ranker and conditional pair/evidence models using only outer training pools. Compare the four already-frozen changes on full and original pair-thinned 65% histories; no tuning on this partition.',
        'risk_gate':'Conditional risk minus v10 PairAP >=-.003 in both full and shortened histories, mean >0; shortened sparse-prior AP delta >=0.',
        'evidence_gate':'Conditional ranker MAP delta versus equity ranker >=-.005, each family delta >=-.03.',
        'behavior_gate':'New conditional assignment composite delta versus conditional risk/evidence with old assignments >=-.003.',
        'failure_action':'Record any failure and reconsider final-entry preference; never tune or silently discard this partition. Preserve all candidate files and all results.',
        'compute':'Four normal reference classifier/regressor fits, three tree classifiers, two evidence rankers, one unchanged two-member TabICL GPU fit.'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    package=OUT/'policy-package'; package.mkdir(); (package/'action-cache').mkdir()
    old=ROOT/'reports/conditional-action-experiment'; oldplan=json.loads((old/'policy-plan.json').read_text())
    for path in (old/'action-cache').glob('*'):
        if path.is_file(): os.link(path,package/'action-cache'/path.name)
    shutil.copy2(OUT/'held-pairs.csv',package/'held-pairs.csv')
    shutil.copy2(ROOT/'data/development_labels.csv',package/'development_labels.csv')
    shutil.copy2(ROOT/'src/run_conditional_policy_screen.py',package/'run_conditional_policy_screen.py')
    files={p.relative_to(package).as_posix():sha(p) for p in package.rglob('*') if p.is_file()}
    policy={'inputs':oldplan['inputs'],'parameters':oldplan['parameters'],'files':files,'parent_audit_sha256':sha(OUT/'plan.json'),
        'training':'Unchanged nested confirmed-normal policy recipe; all 169 held pools excluded from every reference training fit.'}
    (package/'policy-plan.json').write_text(json.dumps(policy,indent=2)+'\n')
    archive=OUT/'policy-bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for path in package.rglob('*'):
            if path.is_file(): t.add(path,arcname=path.relative_to(package).as_posix())
    receipt={'sha256':sha(archive),'bytes':archive.stat().st_size,'plan_sha256':sha(OUT/'plan.json'),'policy_plan_sha256':sha(package/'policy-plan.json')}
    (OUT/'bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
