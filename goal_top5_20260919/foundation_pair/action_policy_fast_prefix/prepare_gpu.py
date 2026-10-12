import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2];S=H.parent/'screen';AP=H.parent/'action_policy_fast';T=H.parent/'tabldm_screen';R=P/'goal_top5_20260919/tabpfn35'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 sr=json.loads((H/'SCORE_RECEIPT.json').read_text());assert sr['score_manifest_sha256']==sha(H/'SCORE_MANIFEST.json');sources=[H/'PROTOCOL.md',H/'SCORE_MANIFEST.json',H/'SCORE_RECEIPT.json',H/'run_fast.py',H/'evaluate.py',R/'runtime35.py',R/'tabpfn-v3.5-fast-20260909.safetensors',AP/'fast129_predictions.npz',AP/'FAST129_RECEIPT.json',S/'pf35_predictions.npz',T/'prefix_shared_hands.npz',T/'quality_results.json',*[H/f'context_scored_prefix_fold{f}.npz' for f in [0,1]],*[AP/f'fast129_context_fold{f}.npz' for f in [0,1]],*[S/f'evaluator_fold{f}_prefix60.npz' for f in [0,1]]]
 m={'status':'prepared_no_gpu_no_quality','release':'GPU_RELEASE.json','fits':2,'seed':35935,'members':4,'primary':'4*global doubled rank original + 1*global doubled rank augmented; old route','sources':{str(x.resolve()):sha(x) for x in sources}};(H/'GPU_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
