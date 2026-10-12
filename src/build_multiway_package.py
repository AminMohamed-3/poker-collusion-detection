"""Package only development target histories for fixed multiway descriptors."""
import os
os.environ['POLARS_MAX_THREADS']='2'
from datetime import datetime,timezone
import json
import shutil
import tarfile
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha
from rank_evidence import PARAMS
from multiway_evidence_features import FEATURES,SAMPLES,SEED

OUT=ROOT/'reports/multiway-evidence-experiment'


def main():
    OUT.mkdir(exist_ok=False); package=OUT/'package'; package.mkdir(); (package/'keys').mkdir()
    tests=ROOT/'reports/multiway-feature-tests.log'; assert tests.read_text().rstrip().endswith('OK')
    pairs=pl.read_csv(ROOT/'data/development_labels.csv').filter(pl.col('label')==1).select('pair_id','player_1','player_2')
    keys=pl.read_parquet(ROOT/'reports/sequence-evidence-experiment/development.parquet').select('pair_id','hand_id').join(pairs,on='pair_id',validate='m:1')
    ids=keys.select('hand_id').unique().sort('hand_id').with_columns(pl.Series('batch',np.arange(keys['hand_id'].n_unique())%8))
    keys=keys.join(ids,on='hand_id',validate='m:1'); assert keys.height==45129 and keys['pair_id'].n_unique()==372
    hands=pl.scan_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='development').join(ids.select('hand_id').lazy(),on='hand_id',how='semi').select('hand_id','big_blind','board_cards').collect()
    assert hands.height==ids.height
    hands.write_parquet(package/'hands.parquet')
    pl.scan_parquet(ROOT/'data/seats.parquet').join(ids.select('hand_id').lazy(),on='hand_id',how='semi').select('hand_id','player_id','hole_card_1','hole_card_2').collect().write_parquet(package/'seats.parquet')
    pl.scan_parquet(ROOT/'data/actions.parquet').join(ids.select('hand_id').lazy(),on='hand_id',how='semi').select('hand_id','action_no','street','player_id','action','amount','to_call','pot_before','stack_before').collect().write_parquet(package/'actions.parquet')
    for i in range(8): keys.filter(pl.col('batch')==i).drop('batch').sort('hand_id','pair_id').write_parquet(package/'keys'/f'batch_{i:03}.parquet')
    for name in ['multiway_evidence_features.py','multiway_feature_worker.py','test_multiway_evidence_features.py']:
        shutil.copy2(ROOT/'src'/name,package/name)
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'features':FEATURES,'samples':SAMPLES,'seed':SEED,'parameters':PARAMS,
        'hypothesis':'Observed-card shares against all currently active seats may distinguish ordinary multiway losses from partner-favoring actions better than heads-up proxies.',
        'scope':'Evidence only. Same conditional ranker plus 24 features and their 24 within-pair contexts. Reuse frozen v12 predicted families for baseline and candidate; same 238 trees and parameters.',
        'simulation':'128 fixed uniformly sampled runouts preflop/flop; exact turn/river. All disclosed seat cards block the deck, including folded cards. Only current street board prefix. Equal split among best active hands. No future betting or side pots.',
        'screen_gate':'Fold0 MAP gain >=.005; no family delta below -.025. If passed, freeze candidate before remaining tests.',
        'confirmation_gate':'Three-fold pooled unique-pair gain >=.005, no known-fold delta below -.005, no pooled family delta below -.025. Omitted-family mean >=-.005 and minimum >=-.03. Each 65%/40% coherent-history pooled gain >0, individual fold delta >=-.005, pooled family delta >=-.03.',
        'limitations':'Disclosed-card hindsight descriptors, not actual player information, ranges, causal utility or side-pot values. Unlisted annotations remain unknown. Reused pool splits are not independent private-performance estimates.',
        'math_tests_sha256':sha(tests),'package_source_sha256':sha(__file__),
        'files':{p.relative_to(package).as_posix():sha(p) for p in package.rglob('*') if p.is_file()},'rows':keys.height,'hands':hands.height}
    (package/'config.json').write_text(json.dumps(plan,indent=2)+'\n'); shutil.copy2(package/'config.json',OUT/'plan.json')
    archive=OUT/'bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for path in package.rglob('*'):
            if path.is_file(): t.add(path,arcname=path.relative_to(package).as_posix())
    receipt={'sha256':sha(archive),'bytes':archive.stat().st_size,'plan_sha256':sha(package/'config.json')}
    (OUT/'bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
