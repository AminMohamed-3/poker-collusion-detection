"""Predeclare a bounded domain-variation utility study before held scoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, tarfile

H = Path(__file__).resolve().parent
R = H.parents[1]
S = H.parent / 'history_realism_20260910/persistent_sessions'
D = H / 'varied'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2) + '\n')

assert not D.exists()
D.mkdir()
old = json.loads((S / 'pilot-plan.json').read_text())
for name in [*old['source_hashes'], 'training-calibration.json']:
    shutil.copy2(S / name, D / name)
# Change pool-specific nuisance parameters, never the poker engine or labels.
s = (D / 'run_pilot.py').read_text()
s = s.replace("    rng=random.Random(spec['seed']);", "    BALANCE={**globals()['BALANCE'],**spec['balance']}\n    ATTENDANCE={**globals()['ATTENDANCE'],**spec['attendance']}\n    rng=random.Random(spec['seed']);")
s = s.replace("'attendance_process':ATTENDANCE,", "'balance_process':{k:v for k,v in BALANCE.items() if k!='first_stack_by_player'},'attendance_process':ATTENDANCE,")
(D / 'run_pilot.py').write_text(s)
compile(s, str(D / 'run_pilot.py'), 'exec')
regimes = [
    dict(name='long_deep', minimum=30, maximum=300, reset=.005, exit=.018, refresh=.006),
    dict(name='moderate_deep', minimum=35, maximum=275, reset=.010, exit=.025, refresh=.010),
    dict(name='moderate_shallow', minimum=45, maximum=225, reset=.0225, exit=.050, refresh=.022),
    dict(name='short_shallow', minimum=50, maximum=200, reset=.035, exit=.070, refresh=.030),
]
pools = []
for ri, regime in enumerate(regimes):
    for family in range(4):
        for bi, behavior in enumerate(['none', 'directed_transfer', 'soft_play', 'coordinated_isolation']):
            i = ri * 16 + family * 4 + bi
            pools.append(dict(pool_id=64+i, seed=4108117+7919*i, policy_family=family,
                behavior=behavior, hands=3000, regime=regime['name'],
                balance=dict(min_balance_bb=regime['minimum'], max_balance_bb=regime['maximum'], spontaneous_reset_probability=regime['reset']),
                attendance=dict(seat_exit_probability=regime['exit'], table_refresh_probability=regime['refresh'])))
plan = {**old, 'created_at': datetime.now(timezone.utc).isoformat(), 'pools': pools,
    'parent_plan_sha256': sha(S / 'pilot-plan.json'),
    'source_hashes': {name: sha(D / name) for name in old['source_hashes']},
    'design': '64 pools: four balance/session regimes crossed with every ordinary-policy and coordination condition. Fresh seeds; 192000 hands. Legitimate settled balances always persist except explicit resets.',
    'purpose': 'Bounded sensitivity/augmentation diagnostic authorized after the realism repair. Deliberately varied regimes are not claimed to match hidden test distribution.',
    'next_stage': 'Verify cards, chips, labels, reset and seat ledgers, independent replay and unchanged source-discrimination ceiling. No new scaling admission; preserve coverage failures.'}
save(D / 'pilot-plan.json', plan)
study = dict(created_at=datetime.now(timezone.utc).isoformat(),
    arms=['resampled_control', 'calibrated', 'varied'],
    class_pair_caps={'none':200, 'directed_transfer':8, 'soft_play':6, 'coordinated_isolation':8},
    availability_rule='Use min(cap, calibrated availability, varied availability) separately per class, without replacement for synthetic pairs. Abort if any positive class has fewer than four examples. Sample complete full/70%/40% views together. Same class sequence and total rows across arms.',
    selection_seed=918117, regimes=regimes, generation_plan_sha256=sha(D / 'pilot-plan.json'),
    datasets={'calibrated':['persistent_sessions', 'session_confirmation'], 'varied':'64 fresh crossed-regime pools'},
    source_admission='Existing source-AUC ceilings .8 and legal/replay checks apply. Report failures; do not force a varied treatment into fitting if the source checks fail.',
    treatment_contrasts=['calibrated vs resampled_control', 'varied vs resampled_control', 'varied vs calibrated'],
    models=['residual_budget','mhc','single_pass','tree897','tree1217'],
    neural_seeds=[7329,39117,55193], epochs=20,
    gates={'network':'Mean paired logloss reduction >= .001; mean Pair AP delta >= 0; worst history mean AP delta >= -.001.', 'tree':'Mean pair_behavior delta >= .0001; worst >= -.003; mean sparse AP delta >= 0.'},
    frozen_selection='No held-label access until all admitted fits complete. No best-seed choice, weight tuning, public-leaderboard optimization, or new submission.',
    limitations=['Reused original held pools; exploratory evidence only.', 'Confirmation coverage remains 15/16; this bounded utility diagnostic does not retroactively pass the expansion gate.', 'Four related player policies share one training source.', 'Known-family utility cannot replace omitted-family validation before promotion.'],
    preparer_sha256=sha(__file__))
save(H / 'study-plan.json', study)
files=[p for p in D.iterdir() if p.is_file()]
manifest={p.relative_to(H).as_posix():sha(p) for p in files}
save(H/'generation-manifest.json',manifest)
with tarfile.open(H/'generation-package.tar.gz','w:gz') as t:
    for p in files+[H/'generation-manifest.json',H/'study-plan.json']:
        t.add(p,arcname=p.relative_to(H))
print(json.dumps({'pools':len(pools),'hands':192000,'study_plan_sha256':sha(H/'study-plan.json'),'package_bytes':(H/'generation-package.tar.gz').stat().st_size}))
