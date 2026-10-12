"""Forge-E-build eval 4096 label-free build (CPU-only, NO VM/GPU/UPLOAD).
Same seed/prefix exact-postflop enumeration as audit.
Parity: audit 3240 rows reproduce features4096.audit.parquet exactly (max-diff 0).
Eval: 16060 selected pairs top-20 hands (321200 rows) from v1 top20-scores (label-free).
Reuse Codex preflop_equity_precision logic (read-only reference).
Outputs ONLY under .opencode/continuous/tasks/equity_on_v1/.
"""
import os
for _k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['PYTHONUTF8']='1'
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode=True
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import polars as pl

POKER = Path(r'C:/Data/Projects/Money Making/work/poker')
OUTDIR = POKER / '.opencode' / 'continuous' / 'tasks' / 'equity_on_v1'
ROOT_ORIGINAL = POKER / 'src' / 'equity_evidence_features.py'
SAMPLE_SEED = 408331
BASE_SAMPLES, HIGH_SAMPLES = 256, 4096
NAMES = [
    'partner_call_deficit_bb','partner_call_surplus_bb','partner_call_equity_cost_bb',
    'partner_fold_surplus_pot_bb','partner_fold_equity','partner_fold_underdog',
    'late_check_equity','heads_up_check_equity','partner_raise_equity_cost_bb',
    'multiway_raise_low_equity_logcost','partner_call_equity_drop_bb','partner_fold_equity_gain',
]
FEATURES = ['eq_'+n+s for s in ['_sum','_max'] for n in NAMES]
assert len(FEATURES)==24
KEYS = ['pair_id','hand_id']
DATA = POKER / 'data'
CV = POKER / 'Claude Version'
CR = POKER / 'Codex Resume'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for blk in iter(lambda: f.read(1024*1024), b''):
            h.update(blk)
    return h.hexdigest()

def sample_indices(count):
    return np.argsort(np.random.default_rng(SAMPLE_SEED).random((count,48)), axis=1)[:,:5]

def worker_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('original_equity_eval4096', str(ROOT_ORIGINAL))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def swap_test(module):
    cards=module.CARDS
    seats={'a':(cards['As'],cards['Ad']),'b':(cards['Kc'],cards['Kd']),'c':(cards['4c'],cards['5d'])}
    pl_=module.heads_up_equity(seats['a'],seats['b'],[])
    pr=module.heads_up_equity(seats['b'],seats['a'],[])
    assert pl_+pr==1.0, 'preflop paired-swap complement failed'
    preflop_actions=[
        {'action_no':0,'player_id':'a','action':'raise','amount':6,'to_call':2,'pot_before':3,'stack_before':100,'street':'preflop'},
        {'action_no':1,'player_id':'b','action':'call','amount':4,'to_call':4,'pot_before':9,'stack_before':100,'street':'preflop'},
    ]
    lf=module.extract_pair_hand('a','b',2,seats,[],preflop_actions)
    rt=module.extract_pair_hand('b','a',2,seats,[],list(reversed(preflop_actions)))
    assert lf==rt, 'preflop paired-swap invariance failed'
    board=[cards[c] for c in ('2c','3h','7s','9d','Jc')]
    river_actions=[
        {'action_no':2,'player_id':'a','action':'bet','amount':10,'to_call':0,'pot_before':20,'stack_before':100,'street':'river'},
        {'action_no':3,'player_id':'b','action':'call','amount':10,'to_call':10,'pot_before':30,'stack_before':100,'street':'river'},
    ]
    ll=module.extract_pair_hand('a','b',2,seats,board,river_actions)
    rr=module.extract_pair_hand('b','a',2,seats,board,list(reversed(river_actions)))
    assert ll==rr, 'postflop paired-swap invariance failed'

def public_inputs(keys_with_players, module):
    hand_ids=keys_with_players['hand_id'].unique().to_list()
    hands=(pl.scan_parquet(DATA/'hands.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','big_blind','board_cards').collect())
    if hands.height!=len(hand_ids):
        raise RuntimeError(f'missing hands: got {hands.height} want {len(hand_ids)}')
    hand_lookup={h:(b,[module.CARDS[c] for c in (board or '').split()]) for h,b,board in hands.iter_rows()}
    seats=(pl.scan_parquet(DATA/'seats.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','player_id','hole_card_1','hole_card_2').collect())
    seat_lookup={}
    for h,p,c1,c2 in seats.iter_rows():
        seat_lookup.setdefault(h,{})[p]=(module.CARDS[c1],module.CARDS[c2])
    actions=(pl.scan_parquet(DATA/'actions.parquet').filter(pl.col('hand_id').is_in(hand_ids)).select('hand_id','action_no','street','player_id','action','amount','to_call','pot_before','stack_before').sort('hand_id','action_no').collect())
    action_lookup={}
    for r in actions.iter_rows(named=True):
        hh=r.pop('hand_id')
        action_lookup.setdefault(hh,[]).append(r)
    return hand_lookup, seat_lookup, action_lookup, actions.height

def compute_frame(keys_with_players, module, count, verify_postflop=True):
    # keys_with_players: pair_id,hand_id,player_1,player_2 sorted
    indices=sample_indices(count)
    module.PREFLOP_SAMPLES=count
    module._SAMPLE_INDICES=indices
    module._canonical_equity.cache_clear()
    assert np.array_equal(module._SAMPLE_INDICES, sample_indices(count))
    swap_test(module)
    hand_lookup, seat_lookup, action_lookup, action_rows = public_inputs(keys_with_players, module)
    real=module.heads_up_equity
    postflop_track={}
    def tracked(first, second, board):
        v=real(first, second, board)
        if board:
            postflop_track[(tuple(first),tuple(second),tuple(board))]=v
        return v
    module.heads_up_equity=tracked
    rows=[]
    for pid,hid,f,s in keys_with_players.select('pair_id','hand_id','player_1','player_2').iter_rows():
        bl,bd=hand_lookup[hid]
        vals=module.extract_pair_hand(f,s,bl,seat_lookup[hid],bd,action_lookup.get(hid,[]))
        rows.append((pid,hid,*vals))
    frame=(pl.DataFrame(rows, schema=['pair_id','hand_id',*module.FEATURES], orient='row').with_columns(pl.col(module.FEATURES).cast(pl.Float32)).sort('pair_id','hand_id'))
    assert frame.select('pair_id','hand_id').equals(keys_with_players.select('pair_id','hand_id'))
    postflop_exact=0
    if verify_postflop and count==HIGH_SAMPLES:
        module.PREFLOP_SAMPLES=BASE_SAMPLES
        module._SAMPLE_INDICES=sample_indices(BASE_SAMPLES)
        module._canonical_equity.cache_clear()
        # re-evaluate each tracked postflop state with 256 cache; must be identical (exact branch)
        for state, hv in postflop_track.items():
            lv=real(*state)
            if lv!=hv:
                raise AssertionError('postflop equity changed with preflop sample count')
            postflop_exact+=1
        # restore 4096 config for caller (not strictly needed, we clear after)
        module.PREFLOP_SAMPLES=count
        module._SAMPLE_INDICES=indices
        module._canonical_equity.cache_clear()
    # restore original function to avoid cross-batch contamination
    module.heads_up_equity=real
    return frame, action_rows, postflop_exact, len(postflop_track)

def main():
    t0=time.time()
    log=lambda *a: print(*a, f'{time.time()-t0:.0f}s', flush=True)
    assert ROOT_ORIGINAL.exists()
    # verify root source hash matches Claude overlay (same as Codex freeze gate)
    claude_overlay = CV/'src'/'equity_evidence_features.py'
    assert sha(str(ROOT_ORIGINAL))==sha(str(claude_overlay)), 'root vs Claude equity source differ'
    log('sources verified', sha(str(ROOT_ORIGINAL))[:8])
    module=worker_module()
    assert tuple(module.FEATURES)==tuple(FEATURES), 'feature manifest differs'
    assert module.PREFLOP_SAMPLES==BASE_SAMPLES
    assert np.array_equal(module._SAMPLE_INDICES, sample_indices(BASE_SAMPLES)), 'original 256 generator mismatch'
    assert np.array_equal(sample_indices(HIGH_SAMPLES)[:BASE_SAMPLES], sample_indices(BASE_SAMPLES)), '4096 prefix mismatch'
    log('generator prefix verified')
    # --- audit parity ---
    audit_path = OUTDIR/'features4096.audit.parquet'
    assert audit_path.exists()
    audit_expected = pl.read_parquet(str(audit_path)).sort(KEYS)
    assert audit_expected.shape==(3240,26), audit_expected.shape
    audit_keys_src = pl.read_parquet(str(CR/'research'/'preflop_equity_precision_20260912'/'inputs-id-only.parquet')).sort(KEYS)
    assert audit_keys_src.height==3240
    log('audit keys loaded', audit_keys_src.height)
    audit_frame, audit_actions, audit_postflop_exact, audit_tracked = compute_frame(audit_keys_src, module, HIGH_SAMPLES, verify_postflop=True)
    # max-diff vs expected
    maxdiff=0.0
    for c in FEATURES:
        d=float((audit_frame[c].cast(pl.Float64)-audit_expected[c].cast(pl.Float64)).abs().max())
        if d>maxdiff:
            maxdiff=d
    log('audit parity max-diff', maxdiff, 'actions', audit_actions, 'postflop_exact', audit_postflop_exact)
    parity={
        'audit_rows': audit_frame.height,
        'audit_expected_sha256': sha(str(audit_path)),
        'audit_rebuilt_max_abs_diff': maxdiff,
        'audit_parity_exact': bool(maxdiff==0.0),
        'audit_action_rows': audit_actions,
        'postflop_exact_states_verified': audit_postflop_exact,
        'tracked_states': audit_tracked,
        'seed': SAMPLE_SEED,
        'samples': HIGH_SAMPLES,
        'prefix_verified': True,
        'swap_verified': True,
        'root_equity_source_sha256': sha(str(ROOT_ORIGINAL)),
    }
    # write parity now (audit part); eval part appended later
    parity_path = OUTDIR/'parity_eval4096.json'
    parity_path.write_text(json.dumps(parity, indent=2)+'\n', encoding='utf-8')
    if maxdiff!=0.0:
        print('ABORT: audit parity failed', flush=True)
        return
    log('audit parity PASS')
    # --- eval keys (label-free top-20 from v1) ---
    batch_files=sorted((CV/'outputs_v1_view_averaged_exploratory_20260913'/'batches').glob('*-top20-scores.parquet'))
    assert len(batch_files)==20, len(batch_files)
    eval_keys_list=[]
    for bf in batch_files:
        df=pl.read_parquet(str(bf)).select('pair_id','hand_id')
        eval_keys_list.append(df)
    eval_keys_nodup=pl.concat(eval_keys_list)
    log('eval top20 raw rows', eval_keys_nodup.height)
    assert eval_keys_nodup.height==321200, eval_keys_nodup.height
    players=pl.read_csv(str(DATA/'evaluation_pairs.csv'), columns=['pair_id','player_1','player_2'])
    eval_keys=eval_keys_nodup.join(players, on='pair_id', validate='m:1').sort('pair_id','hand_id')
    assert eval_keys.height==321200
    assert eval_keys['pair_id'].n_unique()==16060
    log('eval keys with players', eval_keys.height, 'pairs', eval_keys['pair_id'].n_unique())
    # batch eval build by original v1 batch to bound memory (reuse batch partition)
    # map batch index -> keys subset for that batch's top20 file
    out_frames=[]
    total_actions=0
    total_postflop=0
    for i, bf in enumerate(batch_files):
        sub_keys=pl.read_parquet(str(bf)).select('pair_id','hand_id').join(players, on='pair_id', validate='m:1').sort('pair_id','hand_id')
        log(f'batch {i:02d} keys', sub_keys.height)
        frame, act, pexact, tracked = compute_frame(sub_keys, module, HIGH_SAMPLES, verify_postflop=True)
        log(f'batch {i:02d} done actions {act} postflop_exact {pexact} tracked {tracked}')
        total_actions+=act
        total_postflop+=pexact
        out_frames.append(frame)
        # incremental checkpoint every 5 batches (resume-friendly, still under OUTDIR)
        if (i+1)%5==0:
            ckpt=OUTDIR/f'eval4096.ckpt_{i:02d}.parquet'
            pl.concat(out_frames).write_parquet(str(ckpt))
            log(f'ckpt {ckpt.name} rows', pl.concat(out_frames).height)
    eval_frame=pl.concat(out_frames).sort('pair_id','hand_id')
    assert eval_frame.height==321200
    assert eval_frame['pair_id'].n_unique()==16060
    eval_path=OUTDIR/'eval4096.parquet'
    # never overwrite without check: if exists, verify identical or abort
    if eval_path.exists():
        existing=pl.read_parquet(str(eval_path)).sort(KEYS)
        if existing.equals(eval_frame):
            log('eval4096 already exists and identical, reusing')
        else:
            raise RuntimeError('eval4096.parquet already exists and differs; refusing overwrite')
    else:
        eval_frame.write_parquet(str(eval_path))
    log('eval4096 written', eval_frame.shape)
    # update parity receipt with eval info
    parity.update({
        'eval_rows': eval_frame.height,
        'eval_pairs': int(eval_frame['pair_id'].n_unique()),
        'eval_action_rows_total': total_actions,
        'eval_postflop_exact_states_verified_total': total_postflop,
        'eval_sha256': sha(str(eval_path)),
        'eval_batches': len(batch_files),
        'runtime_s': round(time.time()-t0),
    })
    parity_path.write_text(json.dumps(parity, indent=2)+'\n', encoding='utf-8')
    log('DONE eval4096', parity['eval_sha256'][:8], 'runtime', parity['runtime_s'])

if __name__=='__main__':
    main()
