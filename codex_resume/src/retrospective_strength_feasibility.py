"""Label-free information-overlap diagnostic; no relevance or performance claims."""
import os
for name in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='2'
import sys,json,hashlib
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
from phevaluator import Card,evaluate_cards
from marginal_payout_features import inputs,target_keys
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
OUT=HERE/'research'/'retrospective_strength_feasibility_20260912'
STREET={'preflop':0,'flop':3,'turn':4,'river':5}

def preflop(c1,c2):
    r1,r2=c1//4+2,c2//4+2;hi,lo=max(r1,r2),min(r1,r2)
    if hi==lo:return .55+hi/32
    return min(.95,max(.05,(hi+lo)/32-max(hi-lo-1,0)/40+.07*(c1%4==c2%4)))

def stronger(first,second,board):
    if len(board)<3:
        a,b=preflop(*first),preflop(*second)
        return int(a>b)-int(a<b)
    a,b=evaluate_cards(*first,*board),evaluate_cards(*second,*board)
    return int(a<b)-int(a>b)

def main():
    OUT.mkdir(exist_ok=True,parents=True);assert not (OUT/'receipt.json').exists()
    keys=target_keys();pm,hm,sm,am=inputs(keys)
    counts={'pair_hands':keys.height,'pair_hands_with_terminal_postflop_board':0,'pair_hands_with_future_board_at_some_live_pair_action':0,'pair_hands_with_relative_order_change':0,'pair_hands_with_fold_before_terminal_board':0,'live_pair_actions':0,'actions_with_future_board':0,'actions_with_relative_order_change':0,'postflop_actions_with_future_board':0,'postflop_actions_with_relative_order_change':0,'folds_before_terminal_board':0,'folds_with_relative_order_change':0}
    end_lengths={str(n):0 for n in (0,3,4,5)}
    for pid,hid in keys.iter_rows():
        p1,p2=pm[pid];cards={s['player_id']:(Card.to_id(s['hole_card_1']),Card.to_id(s['hole_card_2'])) for s in sm[hid]}
        board=[Card.to_id(c) for c in (hm[hid]['board_cards'] or '').split()];end_lengths[str(len(board))]+=1
        counts['pair_hands_with_terminal_postflop_board']+=int(len(board)>=3)
        live=set(cards);future_any=change_any=fold_any=False
        if len(board)>=3:end_order=stronger(cards[p1],cards[p2],board)
        for a in am[hid]:
            n=STREET[a['street']];assert len(board)>=n
            if a['player_id'] in (p1,p2) and {p1,p2}.issubset(live):
                counts['live_pair_actions']+=1
                if n<len(board):
                    future_any=True;counts['actions_with_future_board']+=1
                    changed=stronger(cards[p1],cards[p2],board[:n])!=end_order
                    counts['actions_with_relative_order_change']+=int(changed);change_any|=changed
                    if n>=3:
                        counts['postflop_actions_with_future_board']+=1
                        counts['postflop_actions_with_relative_order_change']+=int(changed)
                    if a['action']=='fold':
                        fold_any=True;counts['folds_before_terminal_board']+=1
                        counts['folds_with_relative_order_change']+=int(changed)
            if a['action']=='fold':live.remove(a['player_id'])
        counts['pair_hands_with_future_board_at_some_live_pair_action']+=int(future_any)
        counts['pair_hands_with_relative_order_change']+=int(change_any)
        counts['pair_hands_with_fold_before_terminal_board']+=int(fold_any)
    sources=[Path(__file__),HERE/'src'/'marginal_payout_features.py',ROOT/'src'/'advanced_features.py']
    receipt={'status':'complete_label_free','counts':counts,'terminal_board_lengths':end_lengths,'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'preflop_strength':'Same bounded heuristic as existing advanced_features.preflop_strength; not actual equity. Postflop comparisons use exact evaluator.','limitations':['This quantifies information distinction, not usefulness or generalization.','No target relevance, family, metric or outer labels read.','Released terminal cards are legitimate forensic observations, but future outcomes cannot be interpreted as information available to the actor.','No model or submission produced.']}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
