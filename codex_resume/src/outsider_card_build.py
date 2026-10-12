"""Build actual versus complete-hand card-permutation outsider descriptors."""
import os
for n in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from phevaluator import Card
from marginal_payout_features import inputs,target_keys
from retrospective_strength_features import strength,STREET
from outsider_card_features import FEATURES,aggregate
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'research/outsider_decisions_20260912'
ARMS=('linked','card_control');KEYS=['pair_id','hand_id']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def arrays(hand,seats,pair,events,reverse_seats=False):
    outsiders=[s for s in sorted(seats,key=lambda s:s['seat_no'],reverse=reverse_seats) if s['player_id'] not in pair]
    assert len(outsiders)==4
    index={s['player_id']:i for i,s in enumerate(outsiders)}
    cards=[[Card.to_id(s['hole_card_1']),Card.to_id(s['hole_card_2'])] for s in outsiders]
    board=[Card.to_id(c) for c in (hand['board_cards'] or '').split()]
    cached={}
    actor=[];matrix=[]
    for row in events.iter_rows(named=True):
        n=STREET[row['street']];assert len(board)>=n
        if n not in cached:cached[n]=[strength(c,board[:n]) for c in cards]
        actor.append(index[row['actor']]);matrix.append(cached[n])
    matrix=np.asarray(matrix,dtype=float).reshape(-1,4);actor=np.asarray(actor,dtype=int)
    assert np.array_equal(matrix[np.arange(len(actor)),actor],events['own_strength'].to_numpy())
    return actor,events['pair_aggressors_this_street'].to_numpy(),events['action'].to_numpy(),matrix,events['last_aggressor_strength'].to_numpy(),events['log_pot_bb'].to_numpy()


def same(a,b):
    for arm in ARMS:np.testing.assert_allclose(list(a[arm].values()),list(b[arm].values()),rtol=0,atol=1e-12)


def main():
    start=time.perf_counter();assert not (OUT/'feature-freeze.json').exists(),'write-once feature build'
    coverage=json.loads((OUT/'coverage.json').read_text(encoding='utf-8'))
    assert coverage['status']=='complete_label_free_coverage'
    for p,digest in coverage['source_hashes'].items():assert sha(p)==digest,'coverage source drift'
    ep=OUT/'events.parquet';assert sha(ep)==coverage['event_sha256']
    keys=target_keys();pm,hm,sm,am=inputs(keys)
    all_events=pl.read_parquet(ep).sort(KEYS+['action_no'])
    e=all_events.filter(pl.col('facing_pair'));assert e.height==9524
    assert set(e['pair_aggressors_this_street']).issubset({1,2}) and set(e['action']).issubset({'fold','call','aggr'})
    groups={key:q for key,q in e.group_by(KEYS)};empty=e.head(0)
    sources=[Path(__file__),HERE/'src/outsider_card_features.py',OUT/'test_card_features.py',HERE/'research/outsider_cards_protocol_20260912.md',OUT/'coverage.json',ep]
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'features':list(FEATURES),'arms':list(ARMS),'sources':{str(p):sha(p) for p in sources},'keys':keys.height,'card_permutations_per_hand':24,'scope':'Only public cards, cached public action metadata and frozen shortlist predictions; no relevance/behavior/outer evidence.'}
    write(OUT/'feature-freeze.json',freeze)
    rows={name:[] for name in ARMS};checked=0;future_checked=0
    for pid,hid in keys.iter_rows():
        q=groups.get((pid,hid),empty).sort('action_no')
        args=arrays(hm[hid],sm[hid],pm[pid],q);result=aggregate(*args)
        if checked<20 and q.height:
            same(result,aggregate(*arrays(hm[hid],sm[hid],pm[pid][::-1],q)))
            same(result,aggregate(*arrays(hm[hid],sm[hid],pm[pid],q,True)))
            checked+=1
        if future_checked<20:
            board=(hm[hid]['board_cards'] or '').split()
            earlier=q.filter(pl.col('street')=='flop')
            if len(board)>3 and earlier.height:
                dealt={s[k] for s in sm[hid] for k in ('hole_card_1','hole_card_2')}|set(board)
                replacements=[rank+suit for rank in '23456789TJQKA' for suit in 'cdhs' if rank+suit not in dealt]
                changed=dict(hm[hid],board_cards=' '.join(board[:3]+replacements[:len(board)-3]))
                same(aggregate(*arrays(hm[hid],sm[hid],pm[pid],earlier)),aggregate(*arrays(changed,sm[hid],pm[pid],earlier)))
                future_checked+=1
        for name in ARMS:rows[name].append({'pair_id':pid,'hand_id':hid,**result[name]})
    tables={name:pl.DataFrame(v).select(KEYS+list(FEATURES)).sort(KEYS) for name,v in rows.items()};hashes={}
    for name,t in tables.items():
        assert t.select(KEYS).equals(keys)
        v=t.select(FEATURES).to_numpy();assert np.isfinite(v).all() and (v>=0).all()
        fp=OUT/f'{name}.parquet';t.write_parquet(fp);hashes[name]=sha(fp)
    diff=tables['linked'].select(FEATURES).to_numpy()-tables['card_control'].select(FEATURES).to_numpy()
    receipt={'status':'complete','keys':keys.height,'fields_per_arm':24,'feature_hashes':hashes,'feature_freeze_sha256':sha(OUT/'feature-freeze.json'),'pair_and_outsider_seat_order_fixtures':checked,'future_board_suffix_fixtures':future_checked,'hands_with_different_arms':int(np.any(np.abs(diff)>1e-12,axis=1).sum()),'zero_response_hands':keys.height-e.select(KEYS).unique().height,'events':e.height,'seconds':time.perf_counter()-start,'outer_evidence_truth_read':False}
    assert checked==20 and future_checked==20
    write(OUT/'feature-receipt.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
