"""Step 109 audit looks (Claude, training-only; no held read): transformer rerankers over the step-104 typed decoder on S*, trained on
differentiable surrogates of Evidence MAP@5, grouped 5-fold cross-fitting (pair's first table) averaged over 5 fold seeds x 2 inits."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys, json, time
sys.dont_write_bytecode = True
S = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/cf441075-1502-4bb5-9e98-d102782e30e6/scratchpad/s109/'
sys.path.insert(0, S)
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Claude Version/src')
from audit_common import *
from evidence_aug.step104_C_typed_decoder_build import decode, chronology, align, OUT as O104, AUDIT_SCORES, KEYS
from evidence_aug import step106_features as F6
import tfm109 as M
R106 = 'C:/Data/Projects/Money Making/work/poker/Claude Version/experiments/residual_reranker_C_20260913/'
LOOKS = sys.argv[1].split(',')
CFG = json.loads(sys.argv[2]) if len(sys.argv) > 2 else dict(d=16, h=2, L=1, m=32, steps=300, lr=3e-3, wd=1e-2)
TAG = sys.argv[3] if len(sys.argv) > 3 else 'base'
OPT = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
print('options', OPT, flush=True)

prot = check_protected()
d = pl.read_parquet(AUDIT / 'audit-oof.parquet', columns=KEYS + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(KEYS)
sc = pl.read_parquet(AUDIT_SCORES, columns=KEYS + ['table_id', 'B0', 'prefix_strength']).sort(KEYS)
fb = chronology(sc.select(KEYS + ['table_id', 'B0']).rename({'B0': 'score'}))
assert fb.select(KEYS).equals(d.select(KEYS)) and sc.select(KEYS).equals(d.select(KEYS))
fs = fb.with_columns(pl.Series('score', sc['prefix_strength'].to_numpy()))
t1 = align(fb, pl.read_parquet(O104 / 'audit-type-attributes.parquet'))
B0, decS = fb['score'].to_numpy(), decode(fs, t1)
x = F6.candidate_features(fs.with_columns(d['relevance']), pl.read_parquet(R106 + 'audit-reranker-attributes.parquet'), t1, decS)
x = x.sort('pair_id', 'chrono_rank')
pids = x['pair_id'].unique(maintain_order=True).to_list()
Q, T = len(pids), 20
assert x.height == Q * T and (x['trank'].to_numpy().reshape(Q, T) == np.arange(1, T + 1)).all()
pS = np.clip(x['p'].to_numpy(), 0, 1)
X = np.column_stack([F6.matrix(x), x['T1'].cast(pl.Float64).to_numpy(), M.logit(x['dec'].to_numpy()), M.logit(pS)]).reshape(Q, T, -1)
Y = x['relevance'].to_numpy().astype(float).reshape(Q, T)
P = np.tile(np.arange(T), (Q, 1))
nt = dict(d.group_by('pair_id').agg(pl.col('relevance').sum().alias('k')).iter_rows())
NQ = np.array([min(5., float(nt[q])) for q in pids])
tabd = dict(x.group_by('pair_id').agg(pl.col('table_id').min().alias('t')).iter_rows())
tab = np.array([tabd[q] for q in pids])
ug = np.unique(tab)
OFF = {'dec': M.logit(x['dec'].to_numpy()).reshape(Q, T), 'p': M.logit(pS).reshape(Q, T)}
print('queries', Q, 'features', X.shape[2], 'relevant in top-20', int(Y.sum()), 'truth total', int(sum(nt.values())), 'cfg', CFG, flush=True)
print('exact AP@5 on top-20 tokens: dec offset', round(M.ap5_exact(OFF['dec'], Y, NQ), 6), '| p offset', round(M.ap5_exact(OFF['p'], Y, NQ), 6), flush=True)


def cv(loss, off, seeds=tuple(OPT.get('seeds', (0, 1, 2, 3, 4))), inits=(0, 1)):
    YT = Y.copy()
    if OPT.get('shuffle'):
        rs = np.random.RandomState(777)
        for q in range(Q):
            YT[q] = Y[q][rs.permutation(T)]
        assert (YT.sum(1) == Y.sum(1)).all()
    oof, ins = np.zeros((Q, T)), []
    for sd in seeds:
        perm = np.random.RandomState(sd).permutation(len(ug))
        fmap = dict(zip(ug[perm], np.arange(len(ug)) % 5))
        fo = np.array([fmap[t] for t in tab])
        for k in range(5):
            tr, te = fo != k, fo == k
            Ft = X[tr].reshape(-1, X.shape[2])
            mu, sdv = Ft.mean(0), Ft.std(0) + 1e-9
            Xtr, Xte = np.clip((X[tr] - mu) / sdv, -5, 5), np.clip((X[te] - mu) / sdv, -5, 5)
            for ini in inits:
                prm = M.train(Xtr, P[tr], off[tr], YT[tr], NQ[tr], loss=loss, seed=1000 * sd + 10 * k + ini, **CFG)
                te_m = off[te] + M.forward(prm, Xte, P[te], CFG['L'], CFG['h'])[0]
                oof[te] += te_m / (len(seeds) * len(inits))
                if sd == 0 and ini == 0:
                    fit = off[tr] + M.forward(prm, Xtr, P[tr], CFG['L'], CFG['h'])[0]
                    ins.append(tuple(round(M.ap5_exact(*a), 4) for a in ((fit, Y[tr], NQ[tr]), (off[tr], Y[tr], NQ[tr]), (te_m, Y[te], NQ[te]), (off[te], Y[te], NQ[te]))))
    return oof, ins


def compare(name, arr):
    out = {}
    for bn, base in (('B0', B0), ('typed_S_star', decS)):
        c = evaluate(d, {'incumbent': base, name: arr})['comparisons'][name]
        r = json.loads(json.dumps({k: c[k] for k in ('mean_delta', 'ci95', 'by_family', 'halves', 'pairs_better', 'pairs_worse', 'gate_pass')}, default=lambda o: o.item()))
        out[bn] = r
        print(f'  {name} vs {bn}', round(r['mean_delta'], 6), [round(v, 6) for v in r['ci95']], {k[:5]: round(v, 4) for k, v in r['by_family'].items()},
              {k: round(v, 4) for k, v in r['halves'].items()}, r['pairs_better'], r['pairs_worse'], 'gate', r['gate_pass'], flush=True)
    return out


rep = evaluate(d, {'incumbent': B0, 'typed_S_star': decS})
print('replay typed_S_star vs B0', round(rep['comparisons']['typed_S_star']['mean_delta'], 6), '| sets', json.dumps(rep['sets'], default=lambda o: o.item())[:500], flush=True)
results = {'looks': LOOKS, 'cfg': CFG, 'tag': TAG}
for look in LOOKS:
    loss, offn = look.split('_')
    t0 = time.time()
    oof, ins = cv(loss, OFF[offn])
    print(look, 'seconds', round(time.time() - t0, 1), '| seed-0 folds (train fit, train offset, test fit, test offset) AP@5 on tokens:', ins, flush=True)
    results[look] = compare(f'tf_{look}', F6.to_scores(fs, x, oof.reshape(-1)))
    np.save(S + f'oof_{TAG}_{look}.npy', oof)
results['protected_unchanged'] = check_protected() == prot
json.dump(results, open(S + f'aud109_{TAG}.json', 'w', encoding='utf-8'), indent=2)
print('protected unchanged', results['protected_unchanged'], flush=True)
