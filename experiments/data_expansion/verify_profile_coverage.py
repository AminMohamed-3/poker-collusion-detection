"""Check declared weighting/coverage limits against emitted labels, not intentions."""
from pathlib import Path
from collections import Counter
import hashlib,json
import polars as pl
H=Path(__file__).resolve().parent;D=H/'profile_repair'
g=json.loads((D/'pilot-generation.json').read_text());plan=json.loads((D/'pilot-plan.json').read_text());cells=Counter();families=Counter();total=0;pool_sizes=[]
for row in g['pools']:
    n=row['pairs'];total+=n;pool_sizes.append(n);families[row['policy_family']]+=n
    if row['behavior']=='none' or row['positive_pairs']>0:cells[(row['policy_family'],row['behavior'])]+=1
maximum_pool=max(pool_sizes)/total;maximum_family=max(families.values())/total
result={'labelled_pairs':total,'mechanism_cells_with_emitted_labels':len(cells),'maximum_pool_pair_weight':maximum_pool,'maximum_policy_family_pair_weight':maximum_family,'coverage_passed':len(cells)==16 and maximum_pool<=.04 and maximum_family<=.3,'cells':{str(k):v for k,v in cells.items()},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'note':'Stricter supplement to the older audit: a trace without an emitted eligible positive label does not establish labelled-cell coverage.'}
(D/'emitted-coverage-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
