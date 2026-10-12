"""Freeze the next existing population and one unused-seed population together."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).resolve().parent;E=H.parent;P=E/'softplay_capacity_20260910';N=E/'bet_size_confirmation_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((H/'summary.json').read_text())['passes_initial_pilot']
    newseeds={1600811+997*i for i in range(32)};prior=set()
    for f in E.glob('*/**/pilot-plan.json'):prior.update(s['seed'] for s in json.loads(f.read_text())['pools'])
    assert not newseeds&prior
    N.mkdir(exist_ok=False);src=H/'sized';pp0=json.loads((src/'pilot-plan.json').read_text());hashes={}
    for arm in ['b1_sized','fresh_sized']:
        d=N/arm;d.mkdir()
        for name,v in pp0['source_hashes'].items():assert sha(src/name)==v;shutil.copy2(src/name,d/name)
        pp=json.loads(json.dumps(pp0));pp['created_at']=datetime.now(timezone.utc).isoformat()
        if arm=='b1_sized':pp['pools']=json.loads((P/'b1_capacity/pilot-plan.json').read_text())['pools'];pp['paired_existing_control']=str((P/'b1_capacity').resolve())
        else:
            pp['pools']=[{**s,'pool_id':960+i,'scenario':i,'seed':1600811+997*i} for i,s in enumerate(pp['pools'])];pp.pop('paired_existing_control',None);pp.pop('counterfactual_development_replay',None);pp['fresh_seed_confirmation']=True
        pp['unchanged_generation_source_plan_sha256']=sha(src/'pilot-plan.json');(d/'pilot-plan.json').write_text(json.dumps(pp,indent=2));hashes[arm]=sha(d/'pilot-plan.json')
    for name in ['controller.joblib','soft_controller.py','conditional_action_features.py','advanced_features.py','causal_styles.py','audit_batch.py','replay_batch.py','prepare_features.py','run_generation.py','run_analysis.py','causal-style-tests.json']:shutil.copy2(H/name,N/name)
    s=(H/'diagnostics.py').read_text();s=s.replace('Development replay of previously diagnosed scenarios, not fresh-seed confirmation. Same policy source. Original 100 held pools not loaded.','One paired development population and one unused-seed confirmation with unchanged sizing policy. Shared learned source; original outer-held labels not loaded.');(N/'diagnostics.py').write_text(s,encoding='utf-8')
    s=(H/'check_normal.py').read_text().replace("[('control',H.parent/'softplay_capacity_20260910/b0_capacity'),('sized',H/'sized')]","[('b1_control',H.parent/'softplay_capacity_20260910/b1_capacity'),('b1_sized',H/'b1_sized'),('fresh_sized',H/'fresh_sized')]");(N/'check_normal.py').write_text(s,encoding='utf-8')
    plan={'at':datetime.now(timezone.utc).isoformat(),'generator_arms':['b1_sized','fresh_sized'],'generator_plan_sha256':hashes,'batches':[{'arm':'b1_sized','existing_control':str((P/'b1_capacity').resolve()),'pool_ids':list(range(704,736))}],'all_generation_source_bytes_match_initial_pilot':True,'controller_sha256':sha(N/'controller.joblib'),'hands_per_population':96000,'fresh_pool_ids':'960..991','fresh_seed_formula':'1600811+997*i','selection':'The next ordered existing population and one unused population were frozen together. No replacement seeds after outcomes.','gate':'Both populations must pass the unchanged soft-play gates, confirmed-none source AUC<=.85, all legal/causal audits. Initial sizing pilot must remain passed.','progression':'Only both passes allow one frozen detector-utility test: all three size populations, original-only, three predetermined matched resampling draws, three model seeds. No generator choices based on outer results.','protected':json.loads((H/'study-plan.json').read_text())['protected']};(N/'study-plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({'directory':str(N),'plan_sha256':sha(N/'study-plan.json')}))
if __name__=='__main__':main()
