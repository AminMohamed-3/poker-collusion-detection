from pathlib import Path
from dataclasses import replace
import json,sys,hashlib
import numpy as np
H=Path(__file__).resolve().parent;D=H/'profile_repair';sys.path.insert(0,str(D))
from generator import Observation,Profile
from learned_policy import decision
class Capture:
    def choices(self,population,weights):self.weights=weights;return [2]
    def uniform(self,a,b):return (a+b)/2
base=Profile(1,.08,0,.12,.04,1,(.6,.07,.13,.2));tilt=replace(base,aggression=.18,tightness=-.02,noise=.25)
cases=[]
for cards in [('As','Kd'),('7c','6c'),('2s','9d')]:
    obs=Observation(cards,('3s','7d','Jh'),1,4,24,180,8,2,3,1,0,1)
    a=Capture();b=Capture();decision(obs,base,a);decision(obs,tilt,b)
    assert not np.allclose(a.weights,b.weights)
    cases.append({'hole':cards,'base':a.weights,'tilt':b.weights})
result={'passed':True,'public_personality_changes_action_distribution':True,'cases':cases,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(D/'tilt-activation-test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
