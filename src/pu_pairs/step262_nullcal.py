"""STEP262 diagnostic (NOT part of registration262.json, NOT a gate): the null distribution of the
harness rule when an UNINFORMATIVE eleventh column is added to the FINAL2 tail stack.

Ten independent standard-normal type columns are put on the registered scale and measured exactly as
an arm would be.  This says how large a mean dPP90 has to be before it means anything, and how often
a pure-noise column would pass the three harness conditions.  It selects nothing and changes no gate;
it is reported as context for the arm numbers.  Development labels only.

Usage:  python step262_nullcal.py [N]
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '3')
import sys, json, time
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl                                        # noqa: E402
from pu_pairs import step262_repro_tail as H                            # noqa: E402
from pu_pairs import step262_arms as A                                  # noqa: E402
from pu_pairs import step159c_C_combine_dev as S159                     # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.time()
    dest = H.OUT + 'nullcal262.json'
    assert not os.path.exists(dest), dest
    u = pl.read_parquet(H.OUT + 'universe262.parquet').select('pair_id', 'fold', 'surface')
    V = H.stack_V('arch3')
    D0 = H.build_D(V)
    M = A.scale_map(D0)
    cur = H.measure(D0, H.BASE)
    out = {'note': 'diagnostic added after registration262.json; selects nothing, gates nothing',
           'n_draws': n, 'FINAL2_16_abs': cur, 'draws': []}
    for i in range(n):
        raw = np.random.default_rng(900000 + i).normal(size=u.height)
        D = H.build_D(V, extra=[A.typecol(u, raw, 'NULL', M)])
        cand = H.measure(D, H.BASE + ['NULL'])
        r = S159.rule({p: {'a': cand[p], 'b': cur[p]} for p in cand}, 'a', 'b')
        r['seed'] = 900000 + i
        out['draws'].append(r)
        print(i, json.dumps(r), flush=True)
    d90 = np.array([r['mean_dPP90'] for r in out['draws']])
    dcl = np.array([r['mean_dPPcls'] for r in out['draws']])
    out['summary'] = {'mean_dPP90_mean': float(d90.mean()), 'mean_dPP90_sd': float(d90.std(ddof=1)),
                      'mean_dPP90_min': float(d90.min()), 'mean_dPP90_max': float(d90.max()),
                      'mean_dPPcls_mean': float(dcl.mean()), 'mean_dPPcls_sd': float(dcl.std(ddof=1)),
                      'n_passing_harness_rule': int(sum(r['passes'] for r in out['draws'])),
                      'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(dest, 'w', encoding='utf-8'), indent=1)
    print('SUMMARY', json.dumps(out['summary']), flush=True)


if __name__ == '__main__':
    main()
