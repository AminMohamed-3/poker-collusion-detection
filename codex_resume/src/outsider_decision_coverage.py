"""Label-free inventory of outsiders' decisions and disclosed decision strength."""
import os
for n in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import sys,json,time,hashlib,math
from pathlib import Path
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import polars as pl
from phevaluator import Card
from marginal_payout_features import inputs,target_keys
from contestable_state_features import state_features
from retrospective_strength_features import strength,STREET
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
OUT=HERE/'research/outsider_decisions_20260912'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def events_for_hand(hand,seats,events,pair):
    _,parity=state_features(hand,seats,events,pair)
    cards={s['player_id']:[Card.to_id(s['hole_card_1']),Card.to_id(s['hole_card_2'])] for s in seats}
    board=[Card.to_id(c) for c in (hand['board_cards'] or '').split()]
    dealt=[c for v in cards.values() for c in v]+board
    assert len(cards)==6 and len(dealt)==len(set(dealt))
    assert len(board) in (0,3,4,5)
    pset=set(pair);live=set(cards);street=None;last_aggressor=None;pair_aggressors=set();cache={};rows=[]
    assert all(events[i]['action_no']<events[i+1]['action_no'] for i in range(len(events)-1))
    prev_n=-1
    for a in events:
        n=STREET[a['street']];assert n>=prev_n and len(board)>=n;prev_n=n
        if a['street']!=street:street=a['street'];last_aggressor=None;pair_aggressors=set()
        actor=a['player_id'];kind=a['action'];amt=int(a['amount']);tocall=int(a['to_call'])
        assert actor in live
        if kind in ('bet','raise'):assert amt>tocall
        aggr=kind in ('bet','raise','all_in') and amt>tocall
        call=kind in ('call','all_in') and not aggr
        action='aggr' if aggr else 'call' if call else kind
        assert action in ('aggr','call','fold','check')
        if n not in cache:cache[n]={p:strength(c,board[:n]) for p,c in cards.items()}
        if actor not in pset:
            s=cache[n];facing_pair=last_aggressor in pset and tocall>0
            rows.append({'action_no':int(a['action_no']),'actor':actor,'street':street,'action':action,
                'pair_aggressors_this_street':len(pair_aggressors),'live_pair_members':len(pset&live),
                'facing_pair':bool(facing_pair),'last_aggressor_is_pair':bool(last_aggressor in pset),
                'own_strength':s[actor],'pair_max_strength':max(s[p] for p in pair),'pair_min_strength':min(s[p] for p in pair),
                'last_aggressor_strength':s[last_aggressor] if last_aggressor is not None else -1.,
                'log_pot_bb':math.log1p(int(a['pot_before'])/int(hand['big_blind'])),
                'price':min(tocall,int(a['stack_before']))/max(1,int(a['pot_before'])+min(tocall,int(a['stack_before'])))})
        if aggr:
            last_aggressor=actor
            if actor in pset:pair_aggressors.add(actor)
        if kind=='fold':live.remove(actor)
    return rows,parity


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'coverage.json').exists(),'write-once coverage check'
    prev=HERE/'experiments/symmetry_censor_20260912'
    sr=json.loads((prev/'predictions-frozen.json').read_text(encoding='utf-8'))
    assert sha(prev/'scores.parquet')==sr['scores_sha256']
    keys=target_keys();pm,hm,sm,am=inputs(keys);rows=[];parities={};fixtures=0
    for i,(pid,hid) in enumerate(keys.iter_rows()):
        eventrows,parity=events_for_hand(hm[hid],sm[hid],am[hid],pm[pid])
        if i<20:
            swapped,_=events_for_hand(hm[hid],sm[hid],am[hid],pm[pid][::-1]);assert swapped==eventrows;fixtures+=1
        rows.extend(dict(pair_id=pid,hand_id=hid,**r) for r in eventrows)
        for name,v in parity.items():parities[name]=parities.get(name,0)+v
    table=pl.DataFrame(rows).sort('pair_id','hand_id','action_no')
    assert table.select('pair_id','hand_id','action_no').n_unique()==table.height
    assert np.isfinite(table.select('own_strength','pair_max_strength','pair_min_strength','last_aggressor_strength','log_pot_bb','price').to_numpy()).all()
    assert table['own_strength'].min()>=0 and table['own_strength'].max()<=1
    path=OUT/'events.parquet';table.write_parquet(path)
    sources=[Path(__file__),HERE/'src/marginal_payout_features.py',HERE/'src/contestable_state_features.py',HERE/'src/retrospective_strength_features.py',prev/'scores.parquet',prev/'predictions-frozen.json',*[ROOT/'data'/f for f in ('development_labels.csv','hands.parquet','seats.parquet','actions.parquet')]]
    facing=table.filter(pl.col('facing_pair'))
    result={'status':'complete_label_free_coverage','pair_hands':keys.height,'unique_hands':keys['hand_id'].n_unique(),'outsider_action_rows':table.height,'facing_pair_rows':facing.height,'facing_pair_hands':facing.select('pair_id','hand_id').unique().height,'both_members_aggressive_rows':facing.filter(pl.col('pair_aggressors_this_street')==2).height,'by_street_action':facing.group_by('street','action').len().sort('street','action').to_dicts(),'facing_pair_strength_summary':facing.select(pl.col('own_strength').min().alias('min'),pl.col('own_strength').median().alias('median'),pl.col('own_strength').max().alias('max')).to_dicts()[0],'pair_swap_fixtures':fixtures,'parity':parities,'event_sha256':sha(path),'source_hashes':{str(p):sha(p) for p in sources},'seconds':time.perf_counter()-start,'created_at':datetime.now(timezone.utc).isoformat(),'limits':['No relevance/behavior or outer evidence was read; only pair-player lookup columns from label CSV.','Disclosed outsider cards are forensic inputs, not knowledge available to the pair.','Strength is original preflop heuristic or normalized exact made-hand evaluator rank, not equity or action optimality.','Coverage supports feasibility only, not predictive utility.']}
    write(OUT/'coverage.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':main()
