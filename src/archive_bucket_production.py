"""Record a fixed next-day queue and a verified private reproduction archive."""
import json
import shutil
import tarfile
from datetime import datetime,timezone
from pathlib import Path
from robust_pair_experiment import ROOT,sha

OUT=ROOT/'outputs_v15_bucket'; EXP=ROOT/'reports/conditional-bucket-experiment'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    ready=json.loads((OUT/'ready-audit.json').read_text()); assert ready['validation']['valid'] and ready['all_original_feature_and_risk_controls_exact']
    plan=json.loads((OUT/'plan.json').read_text()); assert all(sha(ROOT/p)==h for p,h in plan['protected'].items())
    assert sha(OUT/'submission_bucket_risk_blend.csv')==ready['submission_sha256']
    old=ROOT/'outputs_v14_evidence_portfolio'; oldready=json.loads((old/'ready-audit.json').read_text())
    assert len(oldready['entries'])==5 and all(r['validation']['valid'] and sha(old/r['file'])==r['sha256'] for r in oldready['entries'])
    queue=[OUT/'submission_bucket_risk_blend.csv',*[old/('submission_'+n+'.csv') for n in ['three_way_mean','history_evidence','seed_history_mean','history_rank_fusion']]]
    q={'created_at':datetime.now(timezone.utc).isoformat(),'not_uploaded':True,'automatic_upload_scheduled':False,'expected_allowance_reset_utc':'2026-09-11T00:00:00Z',
       'rationale':'One isolated admitted risk change first, followed by the first four previously frozen evidence comparisons. All compare to v13. The older seed-only evidence recipe remains an immutable reserve. No public outcome selected this order or changes blend weights.',
       'order':[{'position':i+1,'file':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for i,p in enumerate(queue)],
       'reserve':{'file':(old/'submission_seed_evidence.csv').relative_to(ROOT).as_posix(),'sha256':sha(old/'submission_seed_evidence.csv')},
       'prior_queue_is_historical':True,'all_original_five_files_preserved':True,
       'submission_description':'Conditional categorical residual risk v15: fixed equal blend of the incumbent and a 1217-feature pair model using five private-strength categories. Normal references are pool-cross-fitted. Passed known, withheld-family, coherent-history and audit gates. All v13 behavior/evidence fields preserved; complete membership and incumbent-feature/risk replay checks passed.'}
    write(OUT/'next-submission-plan.json',q)
    text='''# A new risk-only candidate

The conditional bucket risk blend passed the frozen admission checks and full production validation. It is ready for the next submission allowance; it has no public score yet. Current scored best remains **0.86005**.

Download: [submission_bucket_risk_blend.csv](submission_bucket_risk_blend.csv).

It averages incumbent risk with a new model of nonlinear action/private-strength dependence, using fixed equal weights on all 112,540 pairs. All v13 behavior labels and evidence lists remain unchanged. No public scores selected features, weights, epochs or family exceptions.

| Development check | Mean gain | Worst gain |
|---|---:|---:|
| Six known full/short cases, .70 PairAP + .10 BehaviorMAP | +0.000399257 | -0.000189134 |
| Six withheld-family cases, hypothetical 1%-prevalence AP | +0.022599364 | -0.016644323 |
| Eight coherent-history/audit cases, weighted pair+behavior | +0.000594868 | -0.000000133 |

These are reused, correlated development comparisons, not a leaderboard forecast. Pure bucket risk and the bucket evidence ranker failed and are excluded. The original and confirmation plans preserve the full research sequence and all limits.

Final fitting uses all 1,860 explicitly labeled development pairs, with full/70%/40% histories weighted equally per pair. Three cross-fitted normal references exclude the scored pool; a final normal reference is used on evaluation gameplay. The pair classifier has 1,217 features. Saved models reproduce predictions after reload.

All twenty evaluation batches rebuild features from disclosed gameplay. Every original conditional aggregate and incumbent risk reproduces exactly before the new risk is accepted. Complete validation covers 112,540 pairs and all 562,700 evidence references, including evaluation phase and both players' hand membership. See [ready-audit.json](ready-audit.json).

The five earlier evidence CSVs are preserved byte-for-byte. The next fixed five-entry queue contains this risk-only file, then the three-way evidence mean, history augmentation, seed/history mean, and history rank fusion. The seed-only evidence file stays in reserve. This introduces a separately controlled risk comparison without mixing its effect with an evidence change. The expected allowance reset is September 11, 00:00 UTC / 03:00 Cairo; no automatic upload is scheduled.

Reproduce with the frozen source snapshot, plan, original dataset and preserved v11/v13 caches. The production scripts fit final references/model, extract the twenty batches and assemble/validate the CSV. The private archive inventories these inputs by hash and contains all new fitted models, features, prediction files, plans, validation records and source. It is an archive verification and inference control, not a clean-environment retraining claim.
'''
    (OUT/'README.md').write_text(text)
    snapshot=OUT/'source-snapshot'; snapshot.mkdir(exist_ok=False)
    for p in sorted((ROOT/'src').glob('*.py')): shutil.copy2(p,snapshot/p.name)
    reports=OUT/'research-records'; reports.mkdir()
    for exp in ['conditional-bucket-experiment','mil-pair-experiment','tabpfn-v2-experiment','event-witness-experiment']:
        source=ROOT/'reports'/exp
        for p in source.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.csv','.md'] and 'package' not in p.parts and 'returned' not in p.parts:
                dest=reports/exp/p.relative_to(source); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['archive.json','inventory.json']]
    inventory={p.relative_to(OUT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}; write(OUT/'inventory.json',inventory)
    archive=ROOT/'outputs_v15_bucket.tar.gz'; assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for p in [*files,OUT/'inventory.json']: t.add(p,arcname=p.relative_to(OUT).as_posix())
    import hashlib
    with tarfile.open(archive) as t:
        members={p.name:p for p in t.getmembers()}; assert set(members)==set(inventory)|{'inventory.json'}
        for name,item in inventory.items(): assert hashlib.sha256(t.extractfile(members[name]).read()).hexdigest()==item['sha256']
        assert json.loads(t.extractfile(members['inventory.json']).read())==inventory
    receipt={'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)+1,'all_archive_members_verified':True,'protected_v13_and_five_v14_files_unchanged':True,'new_submission_sha256':ready['submission_sha256']}
    write(OUT/'archive.json',receipt); print(json.dumps(receipt),flush=True)

if __name__=='__main__': main()
