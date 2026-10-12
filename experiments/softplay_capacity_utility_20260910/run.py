from pathlib import Path
import subprocess,sys
H=Path(__file__).resolve().parent
for name in ['prepare_trial','train_trial','evaluate_trial']:
    with (H/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(H/(name+'.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name+' complete',flush=True)
