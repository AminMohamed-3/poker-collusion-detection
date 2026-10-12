"""Fixed matched terminal-board versus action-prefix strength interactions."""
import os
for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,math,hashlib,importlib.util
from pathlib import Path
from itertools import combinations
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from phevaluator import Card,evaluate_cards
from marginal_payout_features import inputs,target_keys
from contestable_state_features import state_features
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OUT=HERE/'research'/'retrospective_strength_20260912'
STREET={'preflop':0,'flop':3,'turn':4,'river':5}
BASE=[f'{ctx}_{act}_{val}' for ctx in ('partner','other') for act in ('call','fold','check','aggr') for val in ('strength','ahead_gap')]
FEATURES=[f'rs_{name}_{stat}' for stat in ('sum','max') for name in BASE]
assert len(FEATURES)==32

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def preflop(c1,c2):
    r1,r2=c1//4+2,c2//4+2;hi,lo=max(r1,r2),min(r1,r2)
    if hi==lo:return .55+hi/32
    return min(.95,max(.05,(hi+lo)/32-max(hi-lo-1,0)/40+.07*(c1%4==c2%4)))
def strength(cards,board):
    if len(board)==0:return preflop(*cards)
    assert len(board) in (3,4,5)
    return (7463-evaluate_cards(*cards,*board))/7462

def tests():
    orig=ROOT/'src'/'advanced_features.py'
    spec=importlib.util.spec_from_file_location('original_strength_parity',orig)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for a,b in combinations(range(52),2):assert preflop(a,b)==module.preflop_strength(a,b)
    # The same flop and different future cards must have exactly the same prefix strength.
    cards=[Card.to_id(c) for c in ('As','Ah')]
    a=[Card.to_id(c) for c in ('2c','7d','9h','Ac','Ks')]
    b=[Card.to_id(c) for c in ('2c','7d','9h','5s','6c')]
    assert strength(cards,a[:3])==strength(cards,b[:3])
    assert strength(cards,a)!=strength(cards,b)
    return {'preflop_original_exact_combinations':1326,'earlier_prefix_suffix_invariance':True,'terminal_sensitive_to_suffix':True}

def extract(hand,seats,events,pair):
    _,parity=state_features(hand,seats,events,pair)
    cards={s['player_id']:(Card.to_id(s['hole_card_1']),Card.to_id(s['hole_card_2'])) for s in seats}
    board=[Card.to_id(c) for c in (hand['board_cards'] or '').split()]
    terminal={p:strength(cards[p],board) for p in pair}
    accum={m:{p:dict.fromkeys(BASE,0.) for p in pair} for m in ('prefix','terminal')}
    live=set(cards);street=None;previous=None;cache={}
    for a in events:
        if a['street']!=street:street=a['street'];previous=None
        n=STREET[street];assert len(board)>=n
        actor=a['player_id'];kind=a['action'];amount=int(a['amount']);tocall=int(a['to_call'])
        aggressive=kind in ('bet','raise','all_in') and amount>tocall
        calling=kind in ('call','all_in') and not aggressive
        if actor in pair and set(pair).issubset(live):
            partner=pair[1] if actor==pair[0] else pair[0]
            action='aggr' if aggressive else 'call' if calling else kind
            assert action in ('aggr','call','fold','check')
            context='partner' if previous==partner else 'other'
            if n not in cache:cache[n]={p:strength(cards[p],board[:n]) for p in pair}
            w=math.log1p(int(a['pot_before'])/int(hand['big_blind']))
            for mode,s in [('prefix',cache[n]),('terminal',terminal)]:
                accum[mode][actor][f'{context}_{action}_strength']+=s[actor]*w
                accum[mode][actor][f'{context}_{action}_ahead_gap']+=max(0.,s[actor]-s[partner])*w
        if aggressive:previous=actor
        if kind=='fold':live.remove(actor)
    result={}
    for mode,v in accum.items():
        a,b=(v[p] for p in pair)
        values=[a[n]+b[n] for n in BASE]+[max(a[n],b[n]) for n in BASE]
        assert all(math.isfinite(x) and x>=0 for x in values)
        result[mode]=dict(zip(FEATURES,values))
    if not board:assert result['prefix']==result['terminal']
    return result,parity

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'freeze.json').exists()
    verification=tests()
    sources=[Path(__file__),ROOT/'src'/'advanced_features.py',HERE/'src'/'marginal_payout_features.py',HERE/'src'/'contestable_state_features.py',HERE/'research'/'retrospective_strength_protocol_20260912.md']
    write(OUT/'freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'features':FEATURES,'sources':{str(p):sha(p) for p in sources},'synthetic_checks':verification,'label_free':True})
    keys=target_keys();pm,hm,sm,am=inputs(keys);rows={m:[] for m in ('prefix','terminal')};counts={};preflop_ending=0
    for i,(pid,hid) in enumerate(keys.iter_rows()):
        features,parity=extract(hm[hid],sm[hid],am[hid],pm[pid])
        if i<20:
            swapped,_=extract(hm[hid],sm[hid],am[hid],tuple(reversed(pm[pid])))
            assert features==swapped
            names={s['player_id']:f'renamed_{j}' for j,s in enumerate(reversed(sm[hid]))}
            renamed_seats=[dict(s,player_id=names[s['player_id']]) for s in sm[hid]]
            renamed_events=[dict(a,player_id=names[a['player_id']]) for a in am[hid]]
            renamed,_=extract(hm[hid],renamed_seats,renamed_events,tuple(names[p] for p in pm[pid]))
            assert renamed==features
        preflop_ending+=int(not (hm[hid]['board_cards'] or '').split())
        for k,v in parity.items():counts[k]=counts.get(k,0)+v
        for mode in rows:rows[mode].append({'pair_id':pid,'hand_id':hid,**features[mode]})
    hashes={}
    for mode,values in rows.items():
        table=pl.DataFrame(values).select(['pair_id','hand_id']+FEATURES).sort('pair_id','hand_id')
        assert table.select('pair_id','hand_id').equals(keys) and np.isfinite(table.select(FEATURES).to_numpy()).all()
        path=OUT/f'{mode}.parquet';table.write_parquet(path);hashes[mode]=sha(path)
    receipt={'status':'complete','keys':keys.height,'fields_per_arm':32,'counts':counts,'preflop_ending_exact_mode_parity_rows':preflop_ending,'pair_swap_and_player_rename_checked_rows':20,'feature_hashes':hashes,'label_free':True}
    write(OUT/'receipt.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
