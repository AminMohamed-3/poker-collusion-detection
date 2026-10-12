"""STEP241 -- a LEARNED event-type model in place of the hard T1 rule of the typed decoder (audit, development only).

Production decoder: order the top-20 rows (T1 first, then time), multiply p by P(fewer than five listed before).
T1 = "this hand's chip-loser folded to the partner with equity > .5" (STEP104), off for coordinated isolation.
Measured today (STEP239/240): T1 is true on only 79 % of directed_transfer block-1 listed hands and 63 % of
soft_play block-1 hands, and true on 19 % of DT block-2 hands.

Here: tau = P(first-type | hand), LightGBM, trained on listed hands whose block is known from the list itself
(two-block lists: hands before the time descent = block 1, after = block 2; five-long single-block lists = block 1;
unlisted sharp hands later than the last listed block-2 hand of a two-block pair = second-type by the listing rule).
Cross-fitted by table over all positive pairs, so every audit query is typed by a model that never saw its table.
Arms (registered in registration241.json BEFORE this script measured anything):
  HARD  : T1 := tau > .5            (DT and SP; CI untouched)
  SOFT  : Monte-Carlo expectation of the decoder over types ~ Bernoulli(tau), 400 draws (DT and SP)   <- PRIMARY
  SOFT_DT, SOFT_SP : the same for one family at a time.
Stage-2 probabilities are the saved REF p (outputs_step235_20260917/audit_scores_step235.parquet): no stage-2 or
network fit, so the only new randomness is the type model (seeded).
"""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
sys.path.insert(0, str(CV / 'src'))
OUT = CV / 'experiments' / 'step241_learned_type_20260917'
OUT.mkdir(parents=True, exist_ok=True)
REG = OUT / 'registration241.json'
assert REG.exists(), 'write the registration first'
SHARP = 0.4662
FAMS2 = ['directed_transfer', 'soft_play']
CI = 'coordinated_isolation'


def less_than_five(ps):
    dist, out = np.zeros(6), np.empty(len(ps))
    dist[0] = 1.
    for i, p in enumerate(ps):
        out[i] = dist[:5].sum()
        new = dist * (1 - p)
        new[1:] += dist[:-1] * p
        new[5] += dist[5] * p
        dist = new
    return out


def decode(p, t1, cr):
    order = np.lexsort((cr, ~t1))
    q = np.empty(len(p)); q[order] = less_than_five(p[order])
    return p * q + 1e-9 * p


def ap(sc, rel, ntrue):
    o5 = np.argsort(-sc, kind='stable')[:5]; r = rel[o5]; hits = np.cumsum(r)
    return float((hits / np.arange(1, len(o5) + 1) * r).sum() / min(5, ntrue))


# ---------------------------------------------------------------- frames
lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv')
players = {r[0]: (r[1], r[2]) for r in lab.select('pair_id', 'player_1', 'player_2').iter_rows()}
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'started_at', 'phase'])
ev = pl.read_csv(ROOT / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
o = pl.read_parquet(CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet')
raw = pl.read_parquet(str(ROOT / 'outputs_v2' / 'hands_development' / 'batch_*.parquet'))
feat = [c for c in raw.columns if c not in ('pair_id', 'hand_id', 'table_id') and raw[c].dtype.is_numeric()]
pos = o.filter(pl.col('label') == 1).join(hands, on='hand_id', how='left').join(ev, on=['pair_id', 'hand_id'], how='left')
assert set(pos['phase'].unique().to_list()) == {'development'}
pos = pos.with_columns(t=pl.col('started_at').dt.epoch('s')).drop('started_at')

# ---------------------------------------------------------------- type labels from the lists themselves
rows = []
for (pid,), g in pos.group_by(['pair_id']):
    fam = g['behavior_family'][0]
    if fam == CI:
        continue
    L = g.filter(pl.col('listed') == 1).sort('evidence_rank')
    lt = L['t'].to_list(); n = len(lt)
    cut = next((k for k in range(1, n) if lt[k] < lt[k - 1]), n)
    if cut < n:
        for h, b in zip(L['hand_id'].to_list(), [1] * cut + [0] * (n - cut)):
            rows.append((pid, h, fam, b, 'listed'))
        lim = max(lt[cut:])
        S = g.filter((pl.col('listed') == 0) & (pl.col('t') > lim) & (pl.col('p_fmax') > SHARP))
        for h in S['hand_id'].to_list():
            rows.append((pid, h, fam, 0, 'later_sharp'))
    elif n == 5:
        for h in L['hand_id'].to_list():
            rows.append((pid, h, fam, 1, 'listed_single_block'))
Y = pl.DataFrame(rows, schema=['pair_id', 'hand_id', 'family', 'y', 'src'], orient='row')
print('type-label rows', Y.group_by('family', 'src', 'y').len().sort('family', 'src', 'y'))

# ---------------------------------------------------------------- rule attributes (STEP104) for label rows + audit rows
from evidence_aug import step104_C_typed_decoder_build as S104
A = pl.read_parquet(CV / 'outputs_step235_20260917' / 'audit_scores_step235.parquet').select('pair_id', 'hand_id', 'v1_REF').filter(pl.col('v1_REF') > -1e5)
Q = pl.read_parquet(CV / 'outputs_step235_20260917' / 'per_query_step235.parquet').select('pair_id', 'family', 'half', 'REF')
cache = OUT / 'rule_attributes.parquet'
if cache.exists():
    ATT = pl.read_parquet(cache)
else:
    keys = pl.concat([Y.select('pair_id', 'hand_id'), A.select('pair_id', 'hand_id')]).unique().sort('pair_id', 'hand_id')
    hp = dict(hands.select('hand_id', 'phase').join(keys.select('hand_id').unique(), on='hand_id').iter_rows())
    assert set(hp.values()) == {'development'}
    ATT = S104.type_attributes(keys, players)
    ATT.write_parquet(cache)
old = pl.read_parquet(CV / 'experiments' / 'typed_listing_decoder_C_20260913' / 'audit-type-attributes.parquet')
chk = A.select('pair_id', 'hand_id').join(ATT, on=['pair_id', 'hand_id'], how='left').join(
    old.select('pair_id', 'hand_id', pl.col('T1').alias('T1_old')), on=['pair_id', 'hand_id'], how='left')
assert (chk['T1'] == chk['T1_old']).all(), 'rule attributes do not reproduce the production T1'
ATT = ATT.with_columns(f_partner=(pl.col('facing') == 'partner').cast(pl.Float64), f_out=(pl.col('facing') == 'outsider').cast(pl.Float64),
                       f_nofold=(pl.col('facing') == 'nofold').cast(pl.Float64), f_none=(pl.col('facing') == 'none').cast(pl.Float64),
                       fold_street=pl.col('fold_street').cast(pl.Float64), T1f=pl.col('T1').cast(pl.Float64))
XA = ['f_partner', 'f_out', 'f_nofold', 'f_none', 'fold_street', 'folder_equity', 'T1f']
fold_of = dict(o.select('pair_id', 'fold').unique().iter_rows())


def design(keys):
    return keys.join(raw.select(['pair_id', 'hand_id'] + feat), on=['pair_id', 'hand_id'], how='left').join(
        ATT.select(['pair_id', 'hand_id'] + XA), on=['pair_id', 'hand_id'], how='left')


P = dict(n_estimators=300, num_leaves=7, learning_rate=.03, min_child_samples=15, subsample=.8, subsample_freq=1,
         colsample_bytree=.5, reg_lambda=5.0, n_jobs=1, verbosity=-1, random_state=241)
DY = design(Y)
DA = design(A.join(Q.select('pair_id', 'family'), on='pair_id', how='left'))
tau = np.full(DA.height, np.nan)
fa = np.array([fold_of[p] for p in DA['pair_id'].to_list()]); fy = np.array([fold_of[p] for p in DY['pair_id'].to_list()])
fam_a = DA['family'].to_numpy(); fam_y = DY['family'].to_numpy()
oof_y = np.full(DY.height, np.nan)
yv = DY['y'].to_numpy()
XY = DY.select(feat + XA).to_numpy().astype(float); XAu = DA.select(feat + XA).to_numpy().astype(float)
for f in FAMS2:
    for k in range(5):
        tr = (fam_y == f) & (fy != k)
        m = lgb.LGBMClassifier(**P).fit(XY[tr], yv[tr])
        te = (fam_a == f) & (fa == k)
        if te.any():
            tau[te] = m.predict_proba(XAu[te])[:, 1]
        ty = (fam_y == f) & (fy == k)
        if ty.any():
            oof_y[ty] = m.predict_proba(XY[ty])[:, 1]
from sklearn.metrics import roc_auc_score
diag = {}
for f in FAMS2:
    mk = fam_y == f
    rule = DY['T1f'].to_numpy()[mk]
    diag[f] = {'n': int(mk.sum()), 'share_type1': float(yv[mk].mean()), 'oof_auc_learned': float(roc_auc_score(yv[mk], oof_y[mk])),
               'oof_acc_learned_at_.5': float(((oof_y[mk] > .5) == yv[mk]).mean()), 'acc_rule_T1': float((rule == yv[mk]).mean())}
print(json.dumps(diag, indent=1))

# ---------------------------------------------------------------- decode the audit
DA = DA.with_columns(pl.Series('tau', tau)).join(hands.select('hand_id', 'started_at'), on='hand_id', how='left').with_columns(
    t=pl.col('started_at').dt.epoch('s')).drop('started_at').join(ev, on=['pair_id', 'hand_id'], how='left')
ntrue = dict(ev.group_by('pair_id').len().iter_rows())
ARMS = ['REF', 'HARD', 'SOFT', 'SOFT_DT', 'SOFT_SP']
per = []
for (pid,), g in DA.group_by(['pair_id']):
    g = g.sort('t'); fam = g['family'][0]
    p = np.clip(g['v1_REF'].to_numpy() - 10, 0, 1); cr = np.arange(g.height, dtype=float)
    rel = g['evidence_rank'].is_not_null().to_numpy().astype(float)
    t1 = g['T1f'].to_numpy().astype(bool) & (fam != CI)
    ref = decode(p, t1, cr)
    sc = {a: ref for a in ARMS}
    if fam != CI:
        tu = g['tau'].to_numpy()
        hard = decode(p, tu > .5, cr)
        rng = np.random.default_rng(int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16))
        soft = np.mean([decode(p, rng.random(len(tu)) < tu, cr) for _ in range(400)], axis=0)
        sc = {'REF': ref, 'HARD': hard, 'SOFT': soft, 'SOFT_DT': soft if fam == 'directed_transfer' else ref,
              'SOFT_SP': soft if fam == 'soft_play' else ref}
    row = {'pair_id': pid, 'family': fam}
    for a in ARMS:
        row[a] = ap(sc[a], rel, ntrue[pid])
    per.append(row)
R = pl.DataFrame(per).join(Q.select('pair_id', 'half', pl.col('REF').alias('REF_recorded')), on='pair_id')
assert float((R['REF'] - R['REF_recorded']).abs().max()) == 0.0, 'REF not reproduced'
R.write_parquet(OUT / 'per_query.parquet')
rng = np.random.default_rng(241)
res = {'type_model': diag, 'REF': float(R['REF'].mean())}
for a in ARMS[1:]:
    dl = (R[a] - R['REF']).to_numpy(); fm = R['family'].to_numpy(); hv = R['half'].to_numpy()
    bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(4000)]
    res[a] = {'map5': float(R[a].mean()), 'delta': float(dl.mean()), 'halves': [float(dl[hv == 0].mean()), float(dl[hv == 1].mean())],
              'by_family': {f: {'delta': float(dl[fm == f].mean()), 'halves': [float(dl[(fm == f) & (hv == 0)].mean()), float(dl[(fm == f) & (hv == 1)].mean())],
                                'up': int((dl[fm == f] > 0).sum()), 'down': int((dl[fm == f] < 0).sum())} for f in FAMS2},
              'boot90': [float(np.quantile(bs, .05)), float(np.quantile(bs, .95))]}
(OUT / 'results241.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in res.items() if k != 'type_model'}, indent=1))
