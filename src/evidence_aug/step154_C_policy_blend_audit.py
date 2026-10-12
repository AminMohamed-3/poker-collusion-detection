"""Step 154 (Claude): blend of the production decoder order with the STEP152 out-of-fold policy scores (user request: integrate the
policy into the current best order). No fits: reads experiments/rank_policy_grpo_C_20260914/audit_scores152.parquet.
Blend inside each pair's top 20: (1 - a) z(logit(incumbent - 10)) + a z(policy score), z = within-list standardisation.
Registration: experiments/rank_policy_grpo_C_20260914/registration154.json. Never uploads."""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
import audit_common as ac
from evidence_aug.paths import sha, check_protected

EXP = POKER / 'Claude Version/experiments/rank_policy_grpo_C_20260914'
REG = EXP / 'registration154.json'
K = ['pair_id', 'hand_id']
PRIMARY = ('G0', .5)
GRID = [(arm, a) for arm in ('G0', 'G1', 'L') for a in (.1, .25, .5, .75)]


def z(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)


def main():
    t0 = time.perf_counter()
    reg = json.loads(REG.read_text(encoding='utf-8'))
    for name, v in reg['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], name
    assert sha(Path(__file__)) == reg['runner_sha256']
    assert not (EXP / 'results154.json').exists()
    prot = check_protected()
    d = pl.read_parquet(ac.AUDIT / 'audit-oof.parquet', columns=K + ['table_id', 'behavior_family', 'relevance']).sort(K)
    sc = pl.read_parquet(EXP / 'audit_scores152.parquet').sort(K)
    assert sc.select(K).equals(d.select(K))
    inc = sc['incumbent'].to_numpy()
    top = np.flatnonzero(inc > 5)
    pid = d['pair_id'].to_numpy()[top]
    assert np.all(np.diff(np.unique(pid, return_index=True)[1]) == 20) and len(top) == 3240
    idx = top.reshape(162, 20)
    base = np.log(np.clip(inc[idx] - 10, 1e-15, 1 - 1e-15) / (1 - np.clip(inc[idx] - 10, 1e-15, 1 - 1e-15)))
    pol = {arm: (sc[arm].to_numpy()[idx] - 10) * 1000 for arm in ('G0', 'G1', 'L')}
    replay = ac.evaluate(d, {'incumbent': inc, 'G0': sc['G0'].to_numpy()})['comparisons']['G0']['mean_delta']
    assert abs(replay - (-0.00162)) < 5e-6, replay
    R = lambda x: round(float(x), 6)
    res = {}
    for arm, a in GRID:
        blend = (1 - a) * z(base) + a * z(pol[arm])
        full = inc.copy()
        full[idx.reshape(-1)] = 10 + blend.reshape(-1) / 1000
        c = ac.evaluate(d, {'incumbent': inc, 'b': full})['comparisons']['b']
        changed = float(np.mean([set(np.argsort(-b)[:5]) != set(np.argsort(-o)[:5]) for b, o in zip(blend, base)]))
        res[f'{arm}_a{a}'] = {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                              'fam': {k[:2]: R(x) for k, x in c['by_family'].items()}, 'halves': {k: R(x) for k, x in c['halves'].items()},
                              'ac_gate': bool(c['gate_pass']), 'top5_set_changed_share': R(changed)}
        print(f'{arm}_a{a}', json.dumps(res[f'{arm}_a{a}']), flush=True)
    p = res[f'{PRIMARY[0]}_a{PRIMARY[1]}']
    small = p['mean'] >= .006 and p['ci95'][0] > -.004 and min(p['fam'].values()) >= -.01 and min(p['halves'].values()) >= 0
    verdict = 'PASS' if p['ac_gate'] else ('SMALL' if small else 'FAIL')
    assert check_protected() == prot
    out = {'registration_sha256': sha(REG), 'primary': f'{PRIMARY[0]}_a{PRIMARY[1]}', 'verdict': verdict, 'grid': res,
           'grid_passing_ac_gate': [k for k, v in res.items() if v['ac_gate']], 'held_reads': 9, 'seconds': round(time.perf_counter() - t0, 1)}
    (EXP / 'results154.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('VERDICT', verdict, flush=True)


if __name__ == '__main__':
    main()
