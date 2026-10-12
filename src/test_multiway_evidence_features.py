import unittest
import numpy as np
from phevaluator import Card
from multiway_evidence_features import HandShares,extract_pair_hand,FEATURES

def cards(text): return tuple(Card.to_id(x) for x in text.split())
def action(no,player,kind,amount=0,call=0,street='river'):
    return {'action_no':no,'player_id':player,'action':kind,'amount':amount,'to_call':call,'pot_before':100,'stack_before':200,'street':street}

class MultiwayTests(unittest.TestCase):
    def test_river_winner_and_split(self):
        seats={'a':cards('Ah Ad'),'b':cards('Kh Kd'),'c':cards('Qh Qd')}
        q=HandShares(seats,cards('2c 3c 4s 8h 9d')).shares(set(seats),'river')
        self.assertEqual(q,{'a':1.,'b':0.,'c':0.})
        tied=HandShares(seats,cards('Ts Js Qs Ks As')).shares(set(seats),'river')
        for value in tied.values(): self.assertAlmostEqual(value,1/3)

    def test_future_board_and_renaming_invariance(self):
        seats={'a':cards('Ah Ad'),'b':cards('Kh Kd'),'c':cards('Qh Qd')}
        board=cards('2c 3c 4s 8h 9d'); alternative=cards('2c 3c 4s 7h 6d')
        acts=[action(0,'a','raise',10,0,'flop'),action(1,'b','fold',0,10,'flop')]
        expected=extract_pair_hand('a','b',2,seats,board,acts)
        self.assertEqual(expected,extract_pair_hand('a','b',2,seats,alternative,acts))
        self.assertEqual(expected,extract_pair_hand('b','a',2,seats,board,acts))
        mapping={'a':'x','b':'z','c':'w'}
        renamed={mapping[p]:tuple(reversed(h)) for p,h in reversed(list(seats.items()))}
        changed=[{**a,'player_id':mapping[a['player_id']]} for a in acts]
        self.assertEqual(expected,extract_pair_hand('x','z',2,renamed,board,changed))

    def test_folded_outsider_no_longer_wins(self):
        seats={'a':cards('Kh Kd'),'b':cards('Qh Qd'),'c':cards('Ah Ad')}; board=cards('2c 3c 4s 8h 9d')
        engine=HandShares(seats,board)
        self.assertEqual(engine.shares(set(seats),'river')['a'],0.)
        self.assertEqual(engine.shares({'a','b'},'river')['a'],1.)
        acts=[action(0,'c','fold'),action(1,'a','check'),action(2,'b','check')]
        features=dict(zip(FEATURES,extract_pair_hand('a','b',2,seats,board,acts)))
        self.assertEqual(features['mw_late_check_own_share_sum'],1.)
        self.assertEqual(features['mw_multiway_check_joint_share_sum'],0.)

    def test_fold_benefit_and_unknown_future(self):
        seats={'a':cards('Ah Ad'),'b':cards('Kh Kd'),'c':cards('Qh Qd')}; board=cards('2c 3c 4s 8h 9d')
        acts=[action(0,'b','bet',20),action(1,'a','fold',0,20)]
        features=dict(zip(FEATURES,extract_pair_hand('a','b',2,seats,board,acts)))
        self.assertEqual(features['mw_partner_fold_partner_gain_sum'],1.)
        self.assertEqual(features['mw_partner_fold_outsider_gain_sum'],0.)
        for street,n in [('preflop',0),('flop',3),('turn',4),('river',5)]:
            q=HandShares(seats,board[:n]).shares(set(seats),street)
            self.assertAlmostEqual(sum(q.values()),1.)
        with self.assertRaises(ValueError): HandShares(seats,board[:3]).shares(set(seats),'turn')

if __name__=='__main__': unittest.main(verbosity=2)
