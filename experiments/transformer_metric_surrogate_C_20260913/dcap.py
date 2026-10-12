import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, hashlib
sys.dont_write_bytecode = True
S = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/cf441075-1502-4bb5-9e98-d102782e30e6/scratchpad/s109/'
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
from audit_common import *
from evidence_aug.step104_C_typed_decoder_build import decode, chronology, align, less_than_five, OUT as O104, AUDIT_SCORES, KEYS
prot = check_protected()
print('registration sha256', hashlib.sha256(open(S + 'PREDECLARE_step111C_double_cap.md', 'rb').read()).hexdigest())
d = pl.read_parquet(AUDIT / 'audit-oof.parquet', columns=KEYS + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(KEYS)
sc = pl.read_parquet(AUDIT_SCORES, columns=KEYS + ['table_id', 'B0', 'prefix_strength']).sort(KEYS)
fb = chronology(sc.select(KEYS + ['table_id', 'B0']).rename({'B0': 'score'}))
fs = fb.with_columns(pl.Series('score', sc['prefix_strength'].to_numpy()))
t1 = align(fb, pl.read_parquet(O104 / 'audit-type-attributes.parquet'))
B0, decS = fb['score'].to_numpy(), decode(fs, t1)


def decode_alpha(frame, t1, alpha):
    score = frame['score'].to_numpy()
    out = score.copy()
    pid, cr = frame['pair_id'].to_numpy(), frame['chrono_rank'].to_numpy()
    idx = np.flatnonzero(score > 5)
    _, first = np.unique(pid[idx], return_index=True)
    for g in np.split(idx, first[1:]):
        p = np.clip(score[g] - 10, 0, 1)
        order = np.lexsort((cr[g], ~t1[g]))
        q = np.empty(len(g))
        q[order] = less_than_five(p[order])
        out[g] = 10 + p * q ** alpha + 1e-9 * p
    return out


A = (0.0, 0.5, 1.0, 2.0)
arr = {a: decode_alpha(fs, t1, a) for a in A}
print('replay alpha 1 == decode(S*)', bool(np.array_equal(arr[1.0], decS)))
assert np.array_equal(arr[1.0], decS)
keep = ('mean_delta', 'ci95', 'by_family', 'halves', 'pairs_better', 'pairs_worse')
out = {'study': 'STEP111C_double_cap', 'audit_only': True, 'held_read': False}
for base, inc in (('typed_S_star', decS), ('B0', B0)):
    names = {f'alpha_{a}': arr[a] for a in A if not (base == 'typed_S_star' and a == 1.0)}
    rep = evaluate(d, {'incumbent': inc, **names})
    out['sets_' + base] = {k: round(v['map5'], 6) for k, v in rep['sets'].items()}
    for n in names:
        c = json.loads(json.dumps({k: rep['comparisons'][n][k] for k in keep}, default=lambda o: o.item()))
        out[f'{n}_vs_{base}'] = c
        print(f'{n} vs {base}: {c["mean_delta"]:+.4f} [{c["ci95"][0]:+.4f}, {c["ci95"][1]:+.4f}]', {k[:5]: round(v, 4) for k, v in c['by_family'].items()},
              {k: round(v, 4) for k, v in c['halves'].items()}, c['pairs_better'], c['pairs_worse'], flush=True)
    print('sets', out['sets_' + base])
h, dbl, und = out['alpha_0.5_vs_typed_S_star'], None, out['alpha_2.0_vs_typed_S_star']
out['double_count_supported'] = h['mean_delta'] >= .010 and h['ci95'][0] > 0
out['under_weight_supported'] = und['mean_delta'] >= .010 and und['ci95'][0] > 0
out['protected_unchanged'] = check_protected() == prot
open(S + 'dcap111C.json', 'w', encoding='utf-8').write(json.dumps(out, indent=1))
print('double count supported', out['double_count_supported'], '| under-weight supported', out['under_weight_supported'], '| protected unchanged', out['protected_unchanged'])
