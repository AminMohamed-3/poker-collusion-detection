"""Check that physical-seat remapping preserves a complete game's semantics."""
from pathlib import Path
import sys,json
H=Path(__file__).resolve().parent;D=H/'persistent_sessions';sys.path.insert(0,str(D))
from generator import Profile,simulate_hand
players=[f'P{i}' for i in range(6)];profiles=[Profile(i%4,.1,0,.12,.04,.8) for i in range(6)]
for seed in range(12):
    args=(seed+161181,0,seed,players,profiles,[180,200,280,220,320,240],2)
    baseline=simulate_hand(*args)
    for button in range(6):
        h,s,a,t=simulate_hand(*args,button_seat=button)
        assert h['button_seat']==button and set(r['seat_no'] for r in s)==set(range(6))
        assert [r['player_id'] for r in sorted(s,key=lambda r:(r['seat_no']-button-1)%6)]==players
        for i,row in enumerate(s):
            assert {k:v for k,v in row.items() if k!='seat_no'}=={k:v for k,v in baseline[1][i].items() if k!='seat_no'}
        assert a==baseline[2] and t==baseline[3]
        assert {k:v for k,v in h.items() if k!='button_seat'}=={k:v for k,v in baseline[0].items() if k!='button_seat'}
(D/'physical-seat-tests.json').write_text(json.dumps({'passed':True,'base_hands':12,'remapped_games':72,'cards_actions_payoffs_unchanged':True,'engine_order_reconstructs_from_physical_seats':True},indent=2))
print('72 physical-seat remappings preserve full game semantics.')
