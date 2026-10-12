from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent; GOAL=HERE.parents[1]
paths=[
 GOAL/'foundation_pair/full_fast/submission_fast_rank80_20_on_current.csv',
 GOAL/'foundation_pair/full_fast/make_rank_candidate.py',
 GOAL/'foundation_pair/full_fast/candidate_receipt.json',
 GOAL/'foundation_pair/full_fast/public_result.json',
 GOAL/'foundation_pair/full_fast/INTEGRATION_PROTOCOL.md',
 GOAL/'foundation_pair/full_fast/EXPLORATORY_DEVIATION.md',
 GOAL/'foundation_pair/full_fast/fast_probabilities.parquet',
 GOAL/'foundation_pair/full_fast/replay_first1024.npz',
 GOAL/'foundation_pair/full_fast/full_context.npz',
 GOAL/'foundation_pair/full_fast/context_receipt.json',
 GOAL/'foundation_pair/full_fast/runtime.json',
 GOAL/'outputs/submission_sp277_ci276_on_step272.csv',
 GOAL/'foundation_pair/global_pu/global_pu_results.json',
 GOAL/'foundation_pair/global_pu/returned_verified.json',
 GOAL/'foundation_pair/global_pu/PREP_PROTOCOL.md',
 GOAL/'tabpfn35/tabpfn-v3.5-fast-20260909.safetensors',
 GOAL/'tabpfn35/source_receipt.json',
 GOAL/'tabpfn35/MODEL_CARD.md',
 GOAL/'reproduction/writeup/case_facts.json',
 GOAL/'reproduction/writeup/case_selection.json',
]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(p.exists() for p in paths)
manifest={'scope':'exact cached current assembly and case provenance; not full raw upstream retraining','assembly_command':'python foundation_pair/full_fast/make_rank_candidate.py --exploratory (in staged copy with destination absent)','working_directory':str(GOAL),'files':[{'path':p.relative_to(GOAL).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths],'limitations':['Fast fresh-process replay covers first 1024 predictions only','STEP251/259b raw upstream training is not consolidated into one command','existing builder intentionally refuses to overwrite its destination']}
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'files':len(paths)},indent=2))
