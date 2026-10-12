"""Private, verifiable handoff for the next submission allowance."""
from datetime import datetime,timezone
import json,shutil
import numpy as np
import polars as pl
from robust_pair_experiment import ROOT,sha
from rank_evidence import EVIDENCE_COLUMNS
from diverse_next_day import OUT as RESEARCH,write

OUT=ROOT/'outputs_v14_evidence_portfolio'
ORDER=['three_way_mean','history_evidence','seed_history_mean','history_rank_fusion','seed_evidence']
LABELS={'three_way_mean':'Three-way score ensemble','history_evidence':'History-augmented ranker','seed_history_mean':'Seed/history score blend','history_rank_fusion':'History/incumbent rank fusion','seed_evidence':'Three-seed ranker ensemble'}

def main():
    assert not (OUT/'archive.json').exists()
    ready=json.loads((OUT/'ready-audit.json').read_text()); entries={r['candidate']:r for r in ready['entries']}; assert len(entries)==5
    gate=json.loads((RESEARCH/'evidence-panel/admission.json').read_text())['gates']
    parent=json.loads((RESEARCH/'plan.json').read_text()); assert all(sha(ROOT/p)==h for p,h in parent['protected'].items())
    oldfit=json.loads((ROOT/'outputs_v11_conditional/fit.json').read_text())
    assert sha(ROOT/'outputs_v11_conditional/policy/development-hand-features.parquet')==oldfit['identity']['policy_features_sha256']
    for p in sorted((ROOT/'reports/evidence-public-calibration').glob('batch_*.json')):
        assert all(sha(ROOT/name)==digest for name,digest in json.loads(p.read_text())['files'].items())
    selected=pl.read_parquet(ROOT/'reports/replay-v13/selected-pairs.parquet'); arrays={}
    for name,r in entries.items():
        assert gate[name]['passed'] and r['validation']['valid'] and sha(OUT/r['file'])==r['sha256']
        arrays[name]=pl.read_csv(OUT/r['file'],infer_schema=False).join(selected,on='pair_id',how='semi').sort('pair_id').select(EVIDENCE_COLUMNS).to_numpy()
    diversity=[{'first':a,'second':b,'selected_pairs':16060,'different_complete_lists':int(np.any(arrays[a]!=arrays[b],axis=1).sum()),'top1_agreement':float(np.mean(arrays[a][:,0]==arrays[b][:,0]))} for i,a in enumerate(ORDER) for b in ORDER[i+1:]]
    queue=[]
    for name in ORDER:
        r=entries[name]; description=f"V14 {LABELS[name]}. Evidence-only candidate on the unchanged v13 16060-pair selection; risk scores and behavior labels identical to v13. Fixed recipes passed pool, omitted-family, coherent-history and audit admission checks. Validation is reused; no private-performance claim. Full 112540-pair / 562700-reference membership validation passed."
        assert len(description)<=500
        queue.append({'candidate':name,'file':r['file'],'sha256':r['sha256'],'description':description,'submitted':False})
    write(OUT/'queue.json',{'created_at':datetime.now(timezone.utc).isoformat(),'entries':queue,'auto_submission_scheduled':False,'expected_next_reset':'2026-09-11T00:00:00Z',
        'order_reason':'Three-way mean first because its audit full/short and all six coherent-history deltas are nonnegative. Other recipes retain mechanistic diversity. This ordering uses reused local diagnostics, not public scores.',
        'resume_checks':['User resumes work; preserve requested pause for today.','Confirm current official quota, competition rules and no duplicate filename/ref.','Verify exact file hashes and full validation receipts.','Upload once, inspect official status; never retry an uncertain upload blindly.','Record all five outcomes unchanged, even losses; do not change weights in response to individual scores.','Keep v13 incumbent until evidence supports final selection; no manual final entry choice yet.']})
    write(OUT/'diversity.json',{'comparisons':diversity,'limits':'Label-free agreement diagnostic. Related models; this is not proof of independent errors or performance.'})
    lines=['# Five prepared evidence submissions','', 'Prepared September 10, 2026; none uploaded. Current scored incumbent remains v13, public .86005, last observed rank 22/129. No daily slots remain; next allowance is expected September 11 at 00:00 UTC / 03:00 Cairo. Work is paused for the day at the user\'s request; no automatic upload is scheduled.','',
        'All five files pass frozen portfolio admission checks and complete membership validation: 112,540 pairs and 562,700 evidence references each. They preserve every v13 risk/behavior cell and all evidence outside the existing 16,060 selected pairs. These are related evidence-ranking recipes using seed ensembles, history augmentation and fixed score/rank mixtures, not five independent model families.','',
        '| Candidate | Full MAP delta | Short MAP delta | Changed lists | CSV |','|---|---:|---:|---:|---|']
    for name in ORDER:
        g=gate[name]; r=entries[name]
        lines.append(f"| {LABELS[name]} | {g['known_pooled']['full']['delta']:+.6f} | {g['known_pooled']['shortened']['delta']:+.6f} | {r['changed_evidence_lists']:,} | [{r['file']}]({r['file']}) |")
    lines+=['', 'Deltas compare with the incumbent on three reused pool splits, averaging overlapping results per unique pair (210 pairs / 141 pools). Both history lengths, three omitted families, six fully rebuilt shorter-history checks and a less-reused 169-pool audit were evaluated. Every full-history bootstrap interval includes zero; these are exploratory candidates, not statistically established gains. The seed-only ensemble has a small full-history loss and is retained only as an admitted diversity comparison.','',
        'The three-way mean is first in the saved queue because its full/short audit deltas and all six coherent-history deltas are positive. History augmentation has the largest known pooled gain but loses .00657 on the full-history audit and .01043 on one coherent test. Those losses remain visible; weights and family gates were not adjusted to remove them.','',
        'Risk variants (pretrained blend, seed ensemble, dense history views, binary objective and their mixtures) failed omitted-family gates and were excluded. The neural attention reranker also failed its preset gates. No thresholds were relaxed to fill the queue. Five admitted evidence recipes were then frozen and evaluated; this adaptive sequence is disclosed in the plans.','',
        'Production refits use 372 annotated development targets / 45,129 pair-hands / 1,817 annotations and 382 gameplay features. Three newly fitted models reproduce predictions after reload. Every one of twenty evaluation batches reproduces incumbent evidence exactly before applying the alternative recipe. The initial validation attempt used a trimmed hands table without the required phase column; its failed schema receipt is preserved. Final validation uses the original phase-bearing hands and seats and passes for all five files.','',
        'The historical internal/public report finds Pearson .688 and Spearman .600 over five matched evidence-only changes, with sign agreement 3/5. Shared baselines, selection history, coverage and distribution differences prevent reliable calibration. No prediction weights were fitted to these scores. See [comparison report](../reports/internal-public-20260910/REPORT.md).','',
        'Reproduction: use the frozen training and inference plans, returned cache receipts and saved source snapshot. `build_diverse_evidence_candidates.py` completed all training/inference; `assemble_diverse_evidence_candidates.py` completes membership validation with the original data. Existing artifacts are immutable. The twenty baseline checks are inference controls, not a clean-environment retraining replay.','',
        'VM stop status is recorded separately in `../reports/diverse-next-day/vast-stop-status.json`. All selected artifacts and all new models are local. Stop preserves the rented disk; do not destroy or recycle the instance.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=['# Research closeout','', 'Five admitted, fully validated files are ready in `outputs_v14_evidence_portfolio`; no new official submission was attempted after the daily quota was exhausted. See its README and immutable queue.','',
        'The seed/history evidence portfolio passed the original admission tolerances. The six alternative risk recipes failed withheld-family stress checks and are not production candidates. All negative metrics and model files remain in this directory. Gates were applied before evaluation prediction generation and were not weakened.','',
        'Primary plans: `plan.json` (seed/dense history portfolio), `binary-plan.json` (independent binary objective), `evidence-panel-plan.json` (fixed evidence mixtures after risk rejection). `admission.json` retains every risk failure; `evidence-panel/admission.json` records all five evidence checks.','',
        'Full and shortened evidence metrics are recomputed separately; pool-averaging avoids double-counting repeated pairs across known splits. Omitted families are stress-test proxies, not the competition\'s unknown fourth family. The 169-pool audit is less reused, not untouched. These qualifications prevent interpreting the local gains as guaranteed private-board gains.','',
        'No further research, upload or VM restart should happen until the user resumes work. The active goal is paused, not completed. A Kaggle prize and first place remain unachieved.']
    (RESEARCH/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    snap=OUT/'source-snapshot'; snap.mkdir(exist_ok=False)
    for p in sorted((ROOT/'src').glob('*.py')): shutil.copy2(p,snap/p.name)
    files=sorted(set([p for p in OUT.rglob('*') if p.is_file()]+[p for p in RESEARCH.rglob('*') if p.is_file()]+[p for p in (ROOT/'reports/internal-public-20260910').rglob('*') if p.is_file()]))
    inventory={str(p.relative_to(ROOT)): {'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    write(OUT/'archive.json',{'created_at':datetime.now(timezone.utc).isoformat(),'files':inventory,'files_count':len(files),'bytes':sum(x['bytes'] for x in inventory.values()),'all_new_artifacts_local':True,'frozen_incumbent_unchanged':True})
    print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in inventory.values()),'five_ready':True,'minimum_different_selected_lists':min(x['different_complete_lists'] for x in diversity),'archive_sha256':sha(OUT/'archive.json')}),flush=True)

if __name__=='__main__': main()
