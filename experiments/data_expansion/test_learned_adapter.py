"""Compare simulator policy inputs with the established offline feature extractor."""
from pathlib import Path
from itertools import combinations
from dataclasses import replace
import hashlib,json,random,sys
import numpy as np
H=Path(__file__).resolve().parent;D=H/'learned';sys.path.insert(0,str(D))
import generator as g
import learned_policy as lp
from conditional_action_features import extract_hand

original=g.base_decision;captured=[]
def recording(obs,profile,rng):
    captured.append(lp.features(obs,profile)[0])
    return original(obs,profile,rng)
g.base_decision=recording
compared=0
for seed in range(64):
    captured.clear();rng=random.Random(90001+seed);players=[f'P{i}' for i in range(6)]
    profiles=[g.Profile(i%4,.1,0,.1,.05,1,lp.sample_style(rng)) for i in range(6)]
    h,s,a,t=g.simulate_hand(90001+seed,99,seed,players,profiles,[rng.randint(40,250)*2 for _ in players],2,target=('P0','P1'),family=['none','directed_transfer','soft_play','coordinated_isolation'][seed%4])
    pairs=[(str(i),*p) for i,p in enumerate(combinations(players,2))]
    rows=extract_hand(h,s,a,pairs,{p:prof.style for p,prof in zip(players,profiles)})
    assert len(captured)==len(a)
    covered=set()
    for row in rows:
        offline=np.array([row[n] for n in lp.bundle()['inputs']],dtype=np.float32)
        np.testing.assert_array_equal(offline,captured[row['action_no']]);covered.add(row['action_no']);compared+=1
    assert covered==set(range(len(a)))
result={'hands':64,'pair_action_feature_comparisons':compared,'adapter_exact':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256((D/'learned_policy.py').read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((D/'models.joblib').read_bytes()).hexdigest()}
(D/'adapter-parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
