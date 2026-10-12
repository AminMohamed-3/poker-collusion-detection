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
