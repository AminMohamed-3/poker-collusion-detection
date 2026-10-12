import csv,hashlib,json
from pathlib import Path
R=Path(r'C:/Data/Projects/Money Making')
paths={
 'step182':R/'work/poker/Claude Version/outputs_step182_20260916/submission_step182_no_evidence_probe.csv',
 'step259b':R/'work/poker/Claude Version/outputs_step259b_20260918/submission_step259b_geofusion_dt_sp_on_step251.csv',
 'step272':R/'work/poker/Claude Version/outputs_step272_20260919/submission_step272_blockM3xKM_on_step259b.csv',
 'current':R/'work/poker/goal_top5_20260919/outputs/submission_sp277_ci276_on_step272.csv'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with p.open(newline='',encoding='utf-8-sig') as f:
  rows=list(csv.DictReader(f))
 d={r['pair_id']:r for r in rows};assert len(rows)==len(d)==112540
 return d
D={k:load(p) for k,p in paths.items()};keys=set(D['step182']);assert all(set(v)==keys for v in D.values())
def cmp(a,b):
 A,B=D[a],D[b]; ev=[f'evidence_hand_{i}' for i in range(1,6)]
 return {'a':a,'b':b,'key_sets_equal':True,'rows':len(keys),'risk_text_differences':sum(A[k]['risk_score']!=B[k]['risk_score'] for k in keys),'behavior_text_differences':sum(A[k]['predicted_behavior']!=B[k]['predicted_behavior'] for k in keys),'evidence_tuple_differences':sum(any(A[k][c]!=B[k][c] for c in ev) for k in keys),'evidence_cell_differences':sum(A[k][c]!=B[k][c] for k in keys for c in ev)}
sources={
 'step182':R/'work/poker/Claude Version/COORDINATION/CLAUDE_STATUS.json',
 'step259b':R/'work/poker/Claude Version/outputs_step259b_20260918/public_read259b.json',
 'step272':R/'work/poker/Claude Version/COORDINATION/messages/CLAUDE_20260919_89_step272_public_91352_corrects_19229.md',
 'current':R/'work/poker/goal_top5_20260919/outputs/kaggle_result.json'}
out={'files':{k:{'path':str(p.relative_to(R)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in paths.items()},'score_sources':{k:{'path':str(p.relative_to(R)).replace('\\','/'),'sha256':sha(p)} for k,p in sources.items()},'comparisons':[cmp('step182','step259b'),cmp('step259b','step272'),cmp('step272','current')], 'scores':{'step182':.77416,'step259b':.91286,'step272':.91352,'current':.91411}}
# Rounding intervals assume ordinary display rounding to 5 decimals; preserve this explicit conditional.
eps=5e-6;J182=.77416;E259=(.91286-J182)/.2;J272=.91352-.2*E259;Ecur=(.91411-J272)/.2
out['decomposition']={'assumption':'public scores are nearest-1e-5 displays, hence true score in [display-0.000005, display+0.000005); composite=.7P+.1B+.2E; STEP182 no-evidence has E=0', 'P_and_B_separately_identifiable':False,
 'step182_joint_0.7P_plus_0.1B':{'point':J182,'interval':[J182-eps,J182+eps]},
 'step259b_E':{'point':E259,'interval':[E259-2*eps/.2,E259+2*eps/.2]},
 'step272_joint_0.7P_plus_0.1B':{'point':J272,'interval':[J272-3*eps,J272+3*eps],'derivation':'STEP259b and STEP272 evidence text identical, so J272 = score272-score259b+J182'},
 'step272_E':{'point':E259,'interval':[E259-2*eps/.2,E259+2*eps/.2],'derivation':'evidence tuples byte-identical to STEP259b'},
 'current_joint_0.7P_plus_0.1B':{'point':J272,'interval':[J272-3*eps,J272+3*eps],'derivation':'risk and behavior text identical to STEP272'},
 'current_E':{'point':Ecur,'interval':[Ecur-4*eps/.2,Ecur+4*eps/.2]},
 'identified_deltas':{'step259b_to_step272_joint':{'point':.91352-.91286,'interval':[.91352-.91286-2*eps,.91352-.91286+2*eps]},'step272_to_current_E':{'point':(.91411-.91352)/.2,'interval':[(.91411-.91352-2*eps)/.2,(.91411-.91352+2*eps)/.2]}},
 'theoretical_composite_headroom':{'joint_risk_behavior_to_perfect':.8-J272,'evidence_to_perfect_current':.2-.2*Ecur,'warning':'Algebraic ceilings only, not attainable estimates or forecasts. P and B remain inseparable.'}}
O=R/'work/poker/goal_top5_20260919/campaign/component_trace';(O/'trace.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['comparisons'],indent=2));print(json.dumps(out['decomposition'],indent=2))
