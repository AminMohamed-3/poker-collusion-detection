"""Small no-data semantic fixture for the frozen chronology extractor."""
import hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(HERE))
from conditional_action_features import INPUTS, extract_hand

hand={"hand_id":"H","table_id":"T","board_cards":"2c 3d 4h 5s 6c","button_seat":1,"big_blind":100}
seats=[{"player_id":"A","seat_no":1,"hole_card_1":"As","hole_card_2":"Ah"},
       {"player_id":"B","seat_no":2,"hole_card_1":"Kd","hole_card_2":"Qd"}]
actions=[{"action_no":1,"street":"preflop","player_id":"A","action":"raise","amount":300,"pot_before":150,"stack_before":9900,"to_call":100,"players_active":2},
         {"action_no":2,"street":"preflop","player_id":"B","action":"call","amount":300,"pot_before":450,"stack_before":9900,"to_call":300,"players_active":2},
         {"action_no":3,"street":"flop","player_id":"A","action":"check","amount":0,"pot_before":750,"stack_before":9600,"to_call":0,"players_active":2}]
rows=extract_hand(hand,seats,actions,[("P","A","B")],{"A":[.4,.2,.2,.2],"B":[.1,.3,.4,.2]})
assert len(rows)==3 and [r["action_class"] for r in rows]==[3,2,1]
assert rows[1]["facing_partner"]==1 and rows[2]["facing_partner"]==0
assert rows[0]["prior_actions"]==0 and rows[1]["prior_actions"]==1 and rows[2]["prior_actions"]==2
assert list(rows[0].keys())[8:]==INPUTS
payload=json.dumps(rows,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
(HERE/"PARITY_RECEIPT.json").write_text(json.dumps({"passed":True,"rows":3,"inputs":INPUTS,
 "fixture_sha256":hashlib.sha256(payload).hexdigest(),"source_snapshot_sha256":hashlib.sha256((HERE/'conditional_action_features.py').read_bytes()).hexdigest()},indent=2)+"\n")

