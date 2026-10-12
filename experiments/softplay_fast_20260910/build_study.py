"""Freeze a bounded CPU-only soft-play pilot and five-arm tree screen."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).resolve().parent; OLD=H.parent/'history_realism_20260910'; D=H/'repaired'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (H/'study-plan.json').exists()
    D.mkdir()
    for n in ['generator.py','run_pilot.py','learned_policy.py','models.joblib','fit.json','sampler.json','profiles.json','training-calibration.json','balance-process.json','attendance-process.json','conditional_action_features.py','advanced_features.py']:
        shutil.copy2(OLD/'persistent_sessions'/n,D/n)
    source=(D/'generator.py').read_text()
    source=source.replace("HERE.parent/'vendor'","HERE.parents[1]/'history_realism_20260910/vendor'")
    source=source.replace("    own=strength(obs.hole,obs.board); partner=strength(partner_hole,obs.board)","    # Soft play below uses only public state. Private partner strength is used\n    # solely by the inherited directed-transfer controller.\n    own=partner=0.\n    if family=='directed_transfer':\n        own=strength(obs.hole,obs.board); partner=strength(partner_hole,obs.board)")
    start=source.index("    elif family=='soft_play':")
    end=source.index("    elif family=='coordinated_isolation':",start)
    source=source[:start]+'''    elif family=='soft_play':
        # Four declared public-state variants, all subject to the existing
        # episodic/intensity gate. No hidden cards, detector scores or labels.
        from learned_policy import SOFT_VARIANT
        variant=SOFT_VARIANT
        price=obs.to_call/max(obs.pot+obs.to_call,1)
        cap=(.25,.40,.40,.25)[variant]
        if obs.street==0 and price<=cap and (variant!=3 or facing):
            return ('call',size),'partner_low_pressure_preflop'
        if obs.active==2 and obs.to_call==0:
            return ('call',size),'partner_check'
        if facing and obs.to_call>0:
            if action=='raise':return ('call',size),'partner_raise_suppressed'
            if price<=cap and action=='fold':return ('call',size),'partner_low_pressure_call'
        if obs.street>0 and obs.to_call==0 and variant in (0,1):
            return ('call',size),'partner_multiway_check'
''' +source[end:]
    (D/'generator.py').write_text(source)
    with (D/'learned_policy.py').open('a') as f:f.write('\n# Set once for each declared pool by run_pool.\nSOFT_VARIANT=0\n')
    source=(D/'run_pilot.py').read_text().replace("    pool_id=spec['pool_id'];", "    import learned_policy\n    learned_policy.SOFT_VARIANT=spec['soft_variant']\n    pool_id=spec['pool_id'];")
    source=source.replace("'soft_play':{'partner_check','partner_raise_suppressed'}","'soft_play':{'partner_check','partner_raise_suppressed','partner_low_pressure_preflop','partner_low_pressure_call','partner_multiway_check'}")
    source=source.replace('max_workers=4','max_workers=2')
    (D/'run_pilot.py').write_text(source)
    pools=[{'pool_id':256+4*family+variant,'seed':109271+997*(4*family+variant),'policy_family':family,'soft_variant':variant,'behavior':'soft_play','hands':3000} for family in range(4) for variant in range(4)]
    sources={n:sha(D/n) for n in ['generator.py','run_pilot.py','learned_policy.py','models.joblib','profiles.json','balance-process.json','attendance-process.json']}
    (D/'pilot-plan.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'pools':pools,'source_hashes':sources,'training_only':True,'purpose':'Small realism pilot; four related ordinary policies crossed with four public-state soft-play variants.'},indent=2))
    oldplan=json.loads((H.parent/'history_augmentation_20260910/trial/plan.json').read_text())
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'pilot_plan_sha256':sha(D/'pilot-plan.json'),
      'hypothesis':'Partner-conditioned low-pressure continuation and check-back behavior reduce the soft-play action/source mismatch. A realism gain alone is insufficient for augmentation admission.',
      'arms':['original','resampled','old_soft','repaired_soft','real_history_views'],
      'tree_parameters':oldplan['tree_parameters'],'tree_features':1217,'tree_fits':5,
      'added_pair_counts':{'none':120,'directed_transfer':12,'soft_play':40,'coordinated_isolation':8},
      'paired_design':'All augmented arms have the same class counts and 540 additional rows. old_soft/repaired_soft replace only 40 resampled soft pairs, with three grouped views per pair. Other extra pairs identical to resampled. All base rows identical. Equal sample weights. Soft synthetic pairs balanced across eligible pools and selected before validation scoring.',
      'real_history_views':{'fractions':[.50,.65,.85],'seeds':[99173,99179,99181],'positive_rule':'Use random hand-hash subsets. If a positive view has no listed evidence hand, use its original full history and record the fallback. Do not insert evidence hands into random subsets. Existing fixed leave-current-hand-out public baselines are reused.'},
      'boundaries':'Generator, calibration, action references and augmented real views use only the existing 297 training pools. Do not train/calibrate on the 100 outer validation pools. This is an exploratory reuse of those validation pools, not a fresh holdout.',
      'evaluation':'Score once after all five models are frozen. Report full and shortened histories, all classes, and two label-independent hashed halves of the 100 validation pools. Halves and histories are correlated diagnostics, not additional independent datasets. Pool bootstrap intervals, fixed seed 20319, 1000 replicates.',
      'gate':{'realism':'Training-only, grouped soft-play source AUC <=0.85 and below old-source AUC; reduce absolute preflop calling and postflop checking gaps. All emitted labels supported by changed legal actions. At least 12 supported target pairs with every ordinary policy represented.',
       'utility':'Relative to matched resampling, mean log-loss improvement >=0.001, mean Pair AP delta >=0, mean sparse-prevalence AP delta >=0, worst history Pair AP delta >=-0.001; nonnegative mean log-loss improvement in both prespecified pool halves. Compare original-only too. Passing supports replication only; no production promotion.',
       'stop':'If screen fails, stop without hyperparameter search, neural runs, or increased synthesis volume. Report all arms.'},
      'known_limits':['Four related policy models, not independent real player sources.','Only soft-play intervention changes; no broad family coverage claim.','Synthetic duplicates are training weights, not new independent data.','Real history views add no independent players or hands.','Positive evidence annotations are capped, and random-view fallback is label-dependent.','No evidence-ranker training or official combined-score claim.'],
      'protected':oldplan['protected']}
    (H/'study-plan.json').write_text(json.dumps(plan,indent=2))
    (H/'vm-state.json').write_text(json.dumps({'status':'stop request accepted; final provider state not independently observed','accepted_at':'2026-09-10T16:05:56.896431+00:00','instance_id':50489879,'jobs_active_before_stop':0,'transcript_sha256':sha(H/'vm-idle-stop.log'),'current_work':'local CPU only'},indent=2))
    print('Frozen 16-pool pilot and five fixed tree fits.',flush=True)
if __name__=='__main__':main()
