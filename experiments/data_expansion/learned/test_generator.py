"""Engine accounting, reproducibility, information boundary and interventions."""
from dataclasses import fields
import json
from pathlib import Path
import random
import time
import unittest
from generator import Observation, Profile, simulate_hand, base_decision

class SimulationTests(unittest.TestCase):
    def test_information_boundary(self):
        names={f.name for f in fields(Observation)}
        self.assertFalse(names & {'partner_hole','other_holes','deck','future_board','label','family','seed'})
        obs=Observation(('As','Kd'),('3s','7d','Jh'),1,4,24,180,8,2,3,1,0,1)
        profile=Profile(1,.1,0,.1,.04,1)
        self.assertEqual(base_decision(obs,profile,random.Random(8)),base_decision(obs,profile,random.Random(8)))

    def test_reproducible_and_counterfactual_disable(self):
        players=[f'P{i}' for i in range(6)]
        profiles=[Profile(i%4,.12,.01,.12,.04,.8) for i in range(6)]
        args=(329,1,0,players,profiles,[50,70,90,120,200,500],2)
        a=simulate_hand(*args)
        self.assertEqual(a,simulate_hand(*args))
        self.assertEqual(a,simulate_hand(*args,target=('P0','P1'),family='directed_transfer',active_episode=False))

    def test_accounting_diversity_and_interventions(self):
        began=time.perf_counter(); styles=set(); histories=set(); observed=set(); total_actions=0
        players=[f'P{i}' for i in range(6)]
        families=['none','directed_transfer','soft_play','coordinated_isolation']
        for seed in range(256):
            rng=random.Random(seed)
            profiles=[Profile((seed//4+i)%4,rng.uniform(-.04,.19),rng.uniform(-.12,.12),rng.uniform(.03,.2),rng.uniform(.01,.10),rng.uniform(.5,1.3)) for i in range(6)]
            blind=[2,4,10][seed%3]
            stacks=[rng.randint(15,250)*blind for i in range(6)]
            family=families[seed%4]
            h,seats,actions,interventions=simulate_hand(seed,2,seed,players,profiles,stacks,blind,
                target=('P0','P1') if family!='none' else None,family=family,intensity=.85)
            self.assertEqual(sum(r['net_chips'] for r in seats),0)
            self.assertEqual(sum(r['total_contribution'] for r in seats),h['final_pot'])
            self.assertAlmostEqual(sum(r['won_share'] for r in seats),1.)
            for s in seats:
                self.assertGreaterEqual(s['starting_stack']+s['net_chips'],0)
            for action in actions:
                self.assertGreaterEqual(action['amount'],0)
                self.assertLessEqual(action['amount'],action['stack_before'])
                self.assertEqual(action['hand_id'],h['hand_id'])
                styles.add(action['action'])
            if family=='none': self.assertFalse(interventions)
            if interventions: observed.add(family)
            histories.add(tuple((s['hole_card_1'],s['hole_card_2']) for s in seats))
            total_actions+=len(actions)
        self.assertEqual(len(histories),256)
        self.assertEqual(observed,set(families)-{'none'})
        self.assertTrue({'fold','check','call','bet','raise','all_in'}<=styles)
        self.statistics={'hands':256,'actions':total_actions,'seconds':time.perf_counter()-began,'all_mechanisms_exercised':True}
        print(json.dumps(self.statistics),flush=True)

if __name__=='__main__':
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SimulationTests))
    (Path(__file__).parent/'generator-tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'seconds':time.perf_counter()-start},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
