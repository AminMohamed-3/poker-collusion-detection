import hashlib,json
from pathlib import Path
ROOT=Path(r'C:/Data/Projects/Money Making'); OUT=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_enhanced_screen'; BASE=ROOT/'work/poker/goal_top5_20260919/foundation_pair/tabldm_screen'; SCR=ROOT/'work/poker/goal_top5_20260919/foundation_pair/screen'; RT=ROOT/'work/poker/goal_top5_20260919/tabldm'; CK=RT/'checkpoint/clf_default.ckpt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
contexts=[f'context_fold{f}{s}.npz' for f in (0,1) for s in ('','_prefix60')]
for n in contexts+['prefix_shared_hands.npz']:
 assert sha(OUT/n)==sha(BASE/n),(n,'not byte-exact base-screen input')
external={'pf35_predictions':SCR/'pf35_predictions.npz','base_tabldm_predictions':BASE/'predictions.npz','base_quality_results':BASE/'quality_results.json',**{f'evaluator_fold{f}':SCR/f'evaluator_fold{f}.npz' for f in (0,1)},**{f'evaluator_fold{f}_prefix60':SCR/f'evaluator_fold{f}_prefix60.npz' for f in (0,1)}}
runtime={'PREFLIGHT':RT/'PREFLIGHT.md','synthetic_runtime':RT/'receipts/synthetic_runtime.json','provenance_manifest':RT/'receipts/provenance_manifest.json','classifier_source':RT/'overlay/tabldm/_sklearn/classifier.py','enhanced_preflight_result':RT/'enhanced_preflight/RESULT.json','enhanced_worker_result':RT/'enhanced_preflight/worker_result.json'}
manifest={'status':'PREPARED_NO_POKER_PREDICTIONS','contexts_byte_exact_base_screen':True,'context_files':{n:sha(OUT/n) for n in contexts},'prefix_shared_hands':sha(OUT/'prefix_shared_hands.npz'),'external':{n:sha(p) for n,p in external.items()},'tabldm_runtime':{n:sha(p) for n,p in runtime.items()},'checkpoint_sha256':sha(CK),'interpreter':str(ROOT/'work/poker/Codex Resume/.venv_tabpfn3/Scripts/python.exe'),'pythonpath':str(RT/'overlay'),'settings':{'enhance_candidates':True,'official_enhancement_defaults':True,'expected_candidates':{'main':4,'adaptive_plus':16,'gaussian_rank':16,'total':36},'n_estimators':4,'batch_size':1,'seed':35935,'n_jobs':2,'query_chunk':256,'memory_fraction':.70,'fits':2,'fresh_replay_fold':0,'peak_limit_bytes':3*1024**3,'supervisor_total_seconds':1800},'quality_rule':'Fast+enhanced global doubled ranks versus exact saved Fast+base; old Fast route'}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'RUNNER_FROZEN').write_text('Prepared only. Execution requires root GPU_RELEASE.json bound to freeze.json SHA-256.\n')
files=['PROTOCOL.md','MANIFEST.json','run_tabldm.py','supervise.py','evaluate.py','RUNNER_FROZEN',*contexts,'prefix_shared_hands.npz']
freeze={'status':'FROZEN_AWAITING_ROOT_GPU_RELEASE','files':{n:sha(OUT/n) for n in files},**manifest}
(OUT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print('manifest',sha(OUT/'MANIFEST.json'));print('freeze',sha(OUT/'freeze.json'))
