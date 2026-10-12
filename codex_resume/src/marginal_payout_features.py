"""Paired single/layered observer payout features, current training-only audit.

This module excludes relevance and behavior labels. Released cards are observer
information; final contributions are verification only. Payouts assume immediate
showdown without future wagering, and must not be called optimal strategic EV.
"""
from __future__ import annotations
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from phevaluator import Card
from contestable_state_features import state_features, target_keys, sha
from marginal_payout_engine import HandPayouts

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
DATA=ROOT/'data'
OUT=HERE/'research'/'marginal_payout_20260912'
KEYS=['pair_id','hand_id']
BASE=[f'{facing}_{choice}_{value}' for facing in ('partner','outsider')
      for choice in ('call','fold') for value in ('own_loss','own_gain','partner_gain','outsider_gain')]
FEATURES=[f'mp_{name}_bb_{stat}' for stat in ('sum','max') for name in BASE]
assert len(FEATURES)==32

def write(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def extract(hand,seats,events,pair,engine):
    # Reuse frozen, independently tested accounting checks before any feature math.
    _, parity=state_features(hand,seats,events,pair)
    players=[s['player_id'] for s in seats]
    assert len(players)==len(set(players))==6
    seat_player={int(s['seat_no']):s['player_id'] for s in seats}
    contributions=dict.fromkeys(players,0)
    button=int(hand['button_seat'])
    contributions[seat_player[(button+1)%6]]=int(hand['small_blind'])
    contributions[seat_player[(button+2)%6]]=int(hand['big_blind'])
    live=set(players); street=None; previous=None
    accum={m:{p:dict.fromkeys(BASE,0.) for p in pair} for m in ('single','layered')}
    stats={'events_used':0,'payout_calls':0,'nonzero_refund_scenarios':0,'different_mode_scenarios':0}
    for a in events:
        if a['street']!=street: street=a['street']; previous=None
        actor=a['player_id']; kind=a['action']; amount=int(a['amount']); to_call=int(a['to_call'])
        assert actor in live and int(a['pot_before'])==sum(contributions.values())
        aggressive=kind in ('bet','raise','all_in') and amount>to_call
        calling=kind in ('call','all_in') and not aggressive
        if actor in pair and set(pair).issubset(live) and previous is not None and to_call>0 and (calling or kind=='fold'):
            assert previous in live and previous!=actor
            partner=pair[1] if actor==pair[0] else pair[0]
            cost=min(to_call,int(a['stack_before']))
            assert cost>0
            if calling: assert amount==cost, (hand['hand_id'],a['action_no'],'call amount',amount,cost)
            else: assert amount==0
            after=dict(contributions); after[actor]+=cost
            results={}
            for mode in ('single','layered'):
                call=engine.payouts(after,live,street,mode=mode)
                fold=engine.payouts(contributions,live-{actor},street,mode=mode)
                cp,fp=call['payouts'],fold['payouts']
                assert set(cp)==set(fp)==set(players)
                assert abs(sum(cp.values())-sum(after.values()))<1e-8*max(1,sum(after.values()))
                assert abs(sum(fp.values())-sum(contributions.values()))<1e-8*max(1,sum(contributions.values()))
                delta={p:cp[p]-fp[p]-(cost if p==actor else 0) for p in players}
                if kind=='fold': delta={p:-v for p,v in delta.items()}
                assert abs(sum(delta.values()))<1e-8*max(1,sum(after.values())), 'nonzero-sum marginal payout'
                outsider_delta=sum(delta[p] for p in players if p not in pair)
                raw={'own_loss':max(0.,-delta[actor]),'own_gain':max(0.,delta[actor]),
                     'partner_gain':max(0.,delta[partner]),'outsider_gain':max(0.,outsider_delta)}
                prefix=('partner' if previous==partner else 'outsider')+'_'+('call' if calling else 'fold')
                for name,value in raw.items(): accum[mode][actor][prefix+'_'+name]+=value/int(hand['big_blind'])
                stats['payout_calls']+=2
                stats['nonzero_refund_scenarios']+=int(any(v>0 for v in call['refunds'].values()))+int(any(v>0 for v in fold['refunds'].values()))
                results[mode]=(cp,fp)
            for scenario in (0,1):
                stats['different_mode_scenarios']+=int(any(abs(results['single'][scenario][p]-results['layered'][scenario][p])>1e-8 for p in players))
            stats['events_used']+=1
        contributions[actor]+=amount
        if aggressive: previous=actor
        if kind=='fold': live.remove(actor)
    output={}
    for mode in ('single','layered'):
        a,b=(accum[mode][p] for p in pair)
        v=[a[n]+b[n] for n in BASE]+[max(a[n],b[n]) for n in BASE]
        assert all(math.isfinite(x) and x>=0 for x in v)
        output[mode]=dict(zip(FEATURES,v))
    return output,dict(parity,**stats)

def inputs(keys):
    pairrows=pl.read_csv(DATA/'development_labels.csv',columns=['pair_id','player_1','player_2'])
    pairrows=pairrows.join(keys.select('pair_id').unique(),on='pair_id',how='semi')
    pairmap={i:(a,b) for i,a,b in pairrows.iter_rows()}
    handids=keys['hand_id'].unique().to_list()
    hands=(pl.scan_parquet(DATA/'hands.parquet').filter(pl.col('hand_id').is_in(handids))
        .select('hand_id','button_seat','small_blind','big_blind','players_dealt','board_cards').collect())
    seats=(pl.scan_parquet(DATA/'seats.parquet').filter(pl.col('hand_id').is_in(handids))
        .select('hand_id','player_id','seat_no','starting_stack','total_contribution','hole_card_1','hole_card_2').collect())
    actions=(pl.scan_parquet(DATA/'actions.parquet').filter(pl.col('hand_id').is_in(handids))
        .sort('hand_id','action_no').collect())
    hm={x['hand_id']:x for x in hands.to_dicts()}
    sm={k[0]:g.to_dicts() for k,g in seats.group_by('hand_id')}
    am={k[0]:g.to_dicts() for k,g in actions.group_by('hand_id')}
    assert len(hm)==len(sm)==len(am)==len(handids)
    assert len(pairmap)==keys['pair_id'].n_unique()
    return pairmap,hm,sm,am

def build(pilot=False):
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    keys=target_keys()
    if pilot:
        handids=keys['hand_id'].unique().sort().head(20)
        keys=keys.filter(pl.col('hand_id').is_in(handids.implode())).sort(*KEYS)
    name='pilot' if pilot else 'full'
    assert not (OUT/f'{name}-freeze.json').exists(), 'write-once extraction'
    sources=[Path(__file__),HERE/'src'/'marginal_payout_engine.py',HERE/'src'/'contestable_state_features.py',HERE/'research'/'marginal_payout_protocol_20260912.md']
    if not pilot:
        assert json.loads((OUT/'pilot-receipt.json').read_text(encoding='utf-8'))['status']=='complete'
        pilotfreeze=json.loads((OUT/'pilot-freeze.json').read_text(encoding='utf-8'))
        assert {str(p):sha(p) for p in sources}==pilotfreeze['sources'], 'source changed after validated pilot'
    write(OUT/f'{name}-freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'features':FEATURES,'sources':{str(p):sha(p) for p in sources},'keys':keys.height,'label_free':True,'pilot':pilot})
    pairmap,hm,sm,am=inputs(keys)
    rows={m:[] for m in ('single','layered')};counts={}; uniquehands=sorted(hm)
    grouped={k[0]:g for k,g in keys.group_by('hand_id')}
    try:
        for j,hid in enumerate(uniquehands):
            hand=hm[hid]; seats=sm[hid];events=am[hid]
            cards={s['player_id']:(Card.to_id(s['hole_card_1']),Card.to_id(s['hole_card_2'])) for s in seats}
            board=[Card.to_id(c) for c in (hand['board_cards'] or '').split()]
            engine=HandPayouts(cards,board)
            for pid,_ in grouped[hid].iter_rows():
                features,stats=extract(hand,seats,events,pairmap[pid],engine)
                if pilot:
                    swapped,_=extract(hand,seats,events,tuple(reversed(pairmap[pid])),engine)
                    assert features==swapped,'pair-swap invariance'
                for mode in rows: rows[mode].append({'pair_id':pid,'hand_id':hid,**features[mode]})
                for k,v in stats.items(): counts[k]=counts.get(k,0)+v
            if (j+1)%200==0: print('Hands',j+1,'of',len(uniquehands),'seconds',round(time.perf_counter()-start,1),flush=True)
        hashes={}
        for mode,v in rows.items():
            frame=pl.DataFrame(v).select(KEYS+FEATURES).sort(*KEYS)
            assert frame.select(KEYS).equals(keys)
            assert np.isfinite(frame.select(FEATURES).to_numpy()).all()
            path=OUT/f'{name}-{mode}.parquet';frame.write_parquet(path);hashes[mode]=sha(path)
        write(OUT/f'{name}-receipt.json',{'status':'complete','keys':keys.height,'hands':len(uniquehands),'fields_per_arm':len(FEATURES),'feature_hashes':hashes,'counts':counts,'seconds':time.perf_counter()-start,'pair_swap_verified_on_pilot':pilot,'label_free':True})
        print(json.dumps(json.loads((OUT/f'{name}-receipt.json').read_text()),indent=2),flush=True)
    except Exception as exc:
        write(OUT/f'{name}-failure.json',{'type':type(exc).__name__,'message':str(exc),'hand_id':hid if 'hid' in locals() else None,'seconds':time.perf_counter()-start})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args();build(args.pilot)
