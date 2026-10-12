"""Local adapter into the immutable Fast-study evaluator; never ship to VM."""
import importlib.util,json,shutil
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;SCR=HERE/'evaluation_scratch';OUT=HERE/'vm_bundle/outputs'
def main():
 assert (HERE/'returned_verified.json').exists(),'Run verify_returned.py before quality evaluation';assert not (HERE/'standard_result.json').exists() and not SCR.exists()
 replay=json.loads((OUT/'replay_first64.json').read_text());assert replay['exact'] and replay['max_abs']==0
 std=np.load(OUT/'standard_predictions.npz',allow_pickle=False);ctx=np.load(BASE/'context.npz',allow_pickle=False);assert np.array_equal(std['pair_ids'],ctx['pair_ids']) and np.array_equal(std['hand_ids'],ctx['hand_ids']) and std['primary'].shape==(940,) and np.isfinite(std['primary']).all() and ((std['primary']>=0)&(std['primary']<=1)).all()
 old=np.load(BASE/'predictions.npz',allow_pickle=False);SCR.mkdir();shutil.copy2(BASE/'context.npz',SCR/'context.npz');shutil.copy2(BASE/'matched64_lgb.npy',SCR/'matched64_lgb.npy');np.savez_compressed(SCR/'predictions.npz',primary=std['primary'],perm=old['perm'],capped=old['capped']);(SCR/'replay-exact.json').write_text('{"exact":true}\n')
 spec=importlib.util.spec_from_file_location('frozen_citation_eval',BASE/'run_study.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=SCR;m.evaluate();r=json.loads((SCR/'result.json').read_text());assert abs(r['baseline_bare_map5']-.7337397119341563)<1e-12 and abs(r['baseline_event_map5']-.7373662551440329)<1e-12
 result={'scope':'standard architecture-only citation screen; oracle-family-conditioned reused audit','baseline_parity':{'bare':r['baseline_bare_map5'],'event':r['baseline_event_map5'],'step277_gain':r['results_vs_event']['lightgbm']['delta_all']},'standard_vs_event':r['results_vs_event']['primary'],'standard_vs_step277':r['comparisons_vs_lightgbm']['primary'],'fast_reference_vs_step277':json.loads((BASE/'result.json').read_text())['comparisons_vs_lightgbm']['primary'],'fixed_citation_weight':.30,'full_query_denominator':'immutable evaluator min(5, all listed relevant hands)','replay_first64_exact':True}
 (HERE/'standard_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
