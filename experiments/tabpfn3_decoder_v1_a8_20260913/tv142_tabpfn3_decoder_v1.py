"""STEP142 (poker-1b, formerly poker-a8): TabPFN-3 (T) with the typed decoder and with v1, on the training-only audit.
Registration: PREDECLARE_step142_tabpfn3_decoder_v1.md. No fits: Codex's saved T audit scores (read-only) and step 124's saved v1 scores.
Decoder and metric as in tv138_v1_scale_curve.py. Audit look only: no held read, nothing kept."""
REG_SHA = 'b194174f05f9adae9c19dcd0bef76b5257a250cd433717bfb35f54920e55877b'
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '2'
import sys
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
sys.path.insert(0, ROOT + 'Claude Version/src')
sys.path.insert(0, ROOT + 'Codex Resume/src')
import json, time, hashlib
import numpy as np, polars as pl
import audit_common as ac
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align
from evidence_aug.step65_structural_evidence import summarize
CVS, CR = ROOT + 'Claude Version/', ROOT + 'Codex Resume/'
OUT = CVS + 'experiments/tabpfn3_decoder_v1_a8_20260913/'
E124 = CVS + 'experiments/view_averaging_incumbent_C_20260913/'
E138 = CVS + 'experiments/v1_scale_curve_a8_20260913/'
ET = CR + 'experiments/tabpfn3_evidence_20260912/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REG = sha(OUT + 'PREDECLARE_step142_tabpfn3_decoder_v1.md')
assert REG == REG_SHA, 'registration changed'
assert sha(E124 + 'audit_v1_scores.parquet') == '1f157a9d15ccfb51030c46bf9969004fc24d0f27faef63e8ec480c2b3576525a'
assert sha(E138 + 'results_v1_scale_curve.json') == '83a477f6861aaa54903d2f0bd096c5bd556ae2b69475bdff74a6584259b9f5f1'
assert sha(ET + 'scores.parquet') == '8fb03bfb4a4cb28a89ad7c36fb408d08c02b9876b97ea9d636afd12b21c44058'
K, FAM = ['pair_id', 'hand_id'], ac.FAMILIES
T0 = time.time()
log = lambda *a: print(*a, f'{time.time() - T0:.0f}s', flush=True)
prot0 = ac.check_protected()
d, shared, rcols, saved = ac.load_current()
top = (d['oof_rank'] <= 20).to_numpy()
e = pl.read_parquet(ET + 'evaluation-frame.parquet')
s = pl.read_parquet(ET + 'scores.parquet')
assert s.select(K).equals(e.select(K))
j = d.select(K + ['oof', 'oof_rank', 'relevance', 'table_id']).join(e.join(s.drop('table_id'), on=K, validate='1:1'), on=K, how='left', validate='1:1', suffix='_e').sort(K)
assert j.select(K).equals(d.select(K))
T = j['tabpfn3_default'].to_numpy().astype(float)
G = {'a_frame_equal': bool(j['oof_e'].null_count() == 0 and float((j['oof'] - j['oof_e']).abs().max()) == 0.0
                           and (j['oof_rank'].cast(pl.Float64) == j['oof_rank_e']).all() and (j['relevance'] == j['relevance_e']).all()
                           and (j['table_id'] == j['table_id_e']).all()),
     'a_T_top20_gt5': int((T[top] > 5).sum()), 'a_T_tail_lt': int((T[~top] < -1e5).sum()), 'n_tail': int((~top).sum())}
v124 = pl.read_parquet(E124 + 'audit_v1_scores.parquet')
assert v124.select(K).equals(d.select(K))
V1 = v124['v1'].to_numpy().astype(float)
TV = V1.copy()
TV[top] = (T[top] + V1[top]) / 2
fr0 = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', V1)))
assert fr0.select(K).equals(d.select(K))
t1 = align(fr0, pl.read_parquet(CVS + 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
dec = lambda p: decode(fr0.with_columns(pl.Series('score', p)), t1)
truths = {k_[0]: set(g_['hand_id']) for k_, g_ in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
fams = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
key = d.select('pair_id', 'hand_id', 'table_id')
QORD = sorted(summarize(key.with_columns(pl.Series('score', d['oof'].to_numpy())), truths, fams)[1])
assert len(QORD) == 162
famq = np.array([fams[q_] for q_ in QORD])


def ap(p):
    S_, M_ = summarize(key.with_columns(pl.Series('score', p)), truths, fams)
    return float(S_['map5']), np.array([M_[q_] for q_ in QORD])


DV = dec(V1)
mT = ap(T)[0]
JT = json.load(open(ET + 'results.json', encoding='utf-8'))['sets']['tabpfn3_default']['map5']
R138 = json.load(open(E138 + 'results_v1_scale_curve.json', encoding='utf-8'))['runs'][0]['map5']
A = {'V': ap(DV), 'v1_raw': ap(V1)}
G['b_T_raw_replay'] = bool(abs(mT - JT) < 1e-9)
G['b_dec_v1_exact'] = bool(np.array_equal(DV, v124['dec_v1'].to_numpy()))
G['b_v1_map5_replay'] = bool(abs(A['V'][0] - R138['dec']['v1']) < 1e-12 and abs(A['v1_raw'][0] - R138['raw']['v1']) < 1e-12)
RES = {'registration_sha256': REG, 'gates': G}
dump = lambda: json.dump(RES, open(OUT + 'results_step142.json', 'w', encoding='utf-8'), indent=1)
log('gates', G, 'T raw', mT, 'codex', JT)
if not (G['a_frame_equal'] and G['a_T_top20_gt5'] == 3240 and G['a_T_tail_lt'] == G['n_tail'] and G['b_T_raw_replay'] and G['b_dec_v1_exact'] and G['b_v1_map5_replay']):
    dump(); sys.exit('ABORT: gates')
A.update({'TV': ap(dec(TV)), 'dT': ap(dec(T)), 'T_raw': ap(T), 'TV_raw': ap(TV)})
TI = np.unique(np.array([ptab[q_] for q_ in QORD]), return_inverse=True)[1]
NT = int(TI.max()) + 1
rng = np.random.default_rng(1420)
TW = np.stack([np.bincount(rng.integers(0, NT, NT), minlength=NT)[TI].astype(float) for _ in range(2000)])
qq = lambda a_: [float(np.quantile(a_, .05)), float(np.quantile(a_, .95))]


def cmp(a, b):
    D = A[a][1] - A[b][1]
    return {'mean': float(D.mean()), 'q05_q95': qq(TW @ D / TW.sum(axis=1)), 'by_family': {f: float(D[famq == f].mean()) for f in FAM},
            'better_worse': [int((D > 1e-12).sum()), int((D < -1e-12).sum())]}


C = {'TV-V': cmp('TV', 'V'), 'dT-V': cmp('dT', 'V'), 'dT-T_raw': cmp('dT', 'T_raw'), 'TV_raw-v1_raw': cmp('TV_raw', 'v1_raw'), 'T_raw-v1_raw': cmp('T_raw', 'v1_raw')}
keep = lambda c: c['mean'] > 0 and c['q05_q95'][0] > 0 and min(c['by_family'].values()) >= -.01
reading = 'KEEP_CANDIDATE_TV' if keep(C['TV-V']) else ('KEEP_CANDIDATE_T' if keep(C['dT-V']) else 'NO_CHANGE')
G['c_protected_unchanged'] = bool(ac.check_protected() == prot0)
corr = float(np.corrcoef(T[top], V1[top])[0, 1])
RES.update(map5={a: A[a][0] for a in A}, by_family={a: {f: float(A[a][1][famq == f].mean()) for f in FAM} for a in A}, comparisons=C,
           reading=reading, top20_prob_corr_T_v1=corr, runtime_s=round(time.time() - T0))
dump()
np.savez(OUT + 'query_ap_step142.npz', qord=np.array(QORD), **{a: A[a][1] for a in A})
log('map5', {a: round(A[a][0], 4) for a in A}, 'corr', round(corr, 3))
log('READING', reading, {c: {k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in C[c].items() if k_ != 'by_family'} for c in C})
log('by_family', {c: {f: round(v_, 4) for f, v_ in C[c]['by_family'].items()} for c in C}, 'gates', G)
