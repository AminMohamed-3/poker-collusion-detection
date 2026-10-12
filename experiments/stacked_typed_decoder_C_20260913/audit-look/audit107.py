"""Step 107 audit look (Claude, training-only): typed decoder over S* = B0 + prefix_strength. No held read."""
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
from audit_common import *
from evidence_aug.step104_C_typed_decoder_build import decode, chronology, align, OUT as O104, AUDIT_SCORES, KEYS
S = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s107/'
prot = check_protected()
d = pl.read_parquet(AUDIT / 'audit-oof.parquet', columns=KEYS + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(KEYS)
sc = pl.read_parquet(AUDIT_SCORES, columns=KEYS + ['table_id', 'B0', 'prefix_strength']).sort(KEYS)
assert sc.select(KEYS).equals(d.select(KEYS)) and np.array_equal(sc['B0'].to_numpy() > 5, sc['prefix_strength'].to_numpy() > 5)
fb = chronology(sc.select(KEYS + ['table_id', 'B0']).rename({'B0': 'score'}))
assert fb.select(KEYS).equals(sc.select(KEYS))
fs = fb.with_columns(pl.Series('score', sc['prefix_strength'].to_numpy()))
t1 = align(fb, pl.read_parquet(O104 / 'audit-type-attributes.parquet'))
print('T1 rows', int(t1.sum()), 'attrs sha', sha(O104 / 'audit-type-attributes.parquet')[:16], flush=True)
arms = {'B0': fb['score'].to_numpy(), 'S_star': fs['score'].to_numpy(), 'typed_B0': decode(fb, t1), 'typed_S_star': decode(fs, t1),
        'time_only_S_star': decode(fs, np.zeros(len(t1), bool)), 'shuffled_type_S_star': decode(fs, t1, np.random.default_rng(104))}
COMP = [('typed_S_star', 'B0'), ('typed_S_star', 'S_star'), ('typed_S_star', 'typed_B0'), ('S_star', 'B0'), ('typed_B0', 'B0'),
        ('time_only_S_star', 'S_star'), ('shuffled_type_S_star', 'S_star')]
out = {}
for cand, base in COMP:
    c = evaluate(d, {'incumbent': arms[base], cand: arms[cand]})['comparisons'][cand]
    r = json.loads(json.dumps({k: c[k] for k in ('mean_delta', 'ci95', 'by_family', 'halves', 'pairs_better', 'pairs_worse', 'gate_pass')}, default=lambda o: o.item()))
    out[f'{cand}_vs_{base}'] = r
    print(f'{cand}_vs_{base}', round(r['mean_delta'], 6), [round(v, 6) for v in r['ci95']], {k[:5]: round(v, 4) for k, v in r['by_family'].items()},
          {k: round(v, 4) for k, v in r['halves'].items()}, r['pairs_better'], r['pairs_worse'], 'gate', r['gate_pass'], flush=True)
out['audit_looks_added'] = len(COMP)
out['protected_unchanged'] = check_protected() == prot
json.dump(out, open(S + 'audit107.json', 'w', encoding='utf-8'), indent=2)
print('protected unchanged', out['protected_unchanged'])
