"""STEP240 -- how much is the decoder's TYPE flag worth?  Oracle bounds on the 162-query audit, no fitting.

Takes the pre-decoder probability p of every audit top-20 row (STEP235 saved v1 = p + 10), checks that re-decoding with the production T1 flag reproduces REF per query, then
re-decodes with oracle variants of the flag.  Development phase only.  Output: experiments/step240_type_oracle_20260917/
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
OUT = CV / 'experiments' / 'step240_type_oracle_20260917'
OUT.mkdir(parents=True, exist_ok=True)
SHARP = 0.4662


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


def invert(s, t1, cr):
    """decoded s = p * q in the order (T1 first, then time) -> p."""
    order = np.lexsort((cr, ~t1))
    p = np.empty(len(s)); dist = np.zeros(6); dist[0] = 1.
    for i in order:
        q = dist[:5].sum()
        p[i] = min(max(s[i] / q, 0.), 1.) if q > 1e-12 else 0.
        new = dist * (1 - p[i]); new[1:] += dist[:-1] * p[i]; new[5] += dist[5] * p[i]; dist = new
    return p


def decode(p, t1, cr):
    order = np.lexsort((cr, ~t1))
    q = np.empty(len(p)); q[order] = less_than_five(p[order])
    return p * q + 1e-9 * p


def ap5(score, rel):
    o = np.argsort(-score, kind='stable')[:5]
    r = rel[o]; hits = np.cumsum(r)
    return float((hits / np.arange(1, len(o) + 1) * r).sum() / min(5, rel.sum()))


a = pl.read_parquet(CV / 'outputs_step235_20260917' / 'audit_scores_step235.parquet').select('pair_id', 'hand_id', 'v1_REF').filter(pl.col('v1_REF') > -1e5)
q = pl.read_parquet(CV / 'outputs_step235_20260917' / 'per_query_step235.parquet').select('pair_id', 'family', 'half', 'REF')
T = pl.read_parquet(CV / 'experiments' / 'typed_listing_decoder_C_20260913' / 'audit-type-attributes.parquet').select('pair_id', 'hand_id', 'T1')
o = pl.read_parquet(CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet').select('pair_id', 'hand_id', 'p_fmax')
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'started_at', 'phase'])
ev = pl.read_csv(ROOT / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
ev_all = ev.join(hands, on='hand_id', how='left').with_columns(t=pl.col('started_at').dt.epoch('s')).drop('started_at')
d = a.join(T, on=['pair_id', 'hand_id'], how='left').join(o, on=['pair_id', 'hand_id'], how='left').join(
    hands, on='hand_id', how='left').join(ev, on=['pair_id', 'hand_id'], how='left').join(q, on='pair_id', how='inner')
assert set(d['phase'].unique().to_list()) == {'development'} and d['T1'].null_count() == 0
d = d.with_columns(t=pl.col('started_at').dt.epoch('s')).drop('started_at')
print('rows', d.height, 'pairs', d['pair_id'].n_unique())
# block of every listed hand, from the FULL list (not only the rows inside the top-20)
blk = {}
two_block, last_b2_t = {}, {}
for (pid,), g in ev_all.group_by(['pair_id']):
    g = g.sort('evidence_rank'); lt = g['t'].to_list(); n = len(lt)
    cut = next((k for k in range(1, n) if lt[k] < lt[k - 1]), n)
    for h, b in zip(g['hand_id'].to_list(), [1] * cut + [2] * (n - cut)):
        blk[(pid, h)] = b
    two_block[pid] = cut < n
    last_b2_t[pid] = max(lt[cut:]) if cut < n else None
ARMS = ['rule (REF check)', 'oracle on listed', 'oracle on listed + late sharp unlisted = type 2', 'oracle on listed + all unlisted = type 2',
        'no types (time only)', 'all unlisted keep rule; listed all type 1']
per = []
for (pid,), g in d.group_by(['pair_id']):
    g = g.sort('t')
    fam_ci = g['family'][0] == 'coordinated_isolation'
    t1 = g['T1'].to_numpy().astype(bool) & (not fam_ci)   # production: T1 off for coordinated isolation
    cr = np.arange(g.height, dtype=float)
    rel = g['evidence_rank'].is_not_null().to_numpy().astype(float)
    nlisted_total = sum(1 for k in blk if k[0] == pid)
    p = np.clip(g['v1_REF'].to_numpy() - 10, 0, 1)   # the saved v1 scores are PRE-decoder probabilities + 10
    hid = g['hand_id'].to_list(); pf = g['p_fmax'].to_numpy(); tt = g['t'].to_numpy()
    listed = rel > 0
    ob = np.array([blk.get((pid, h), 0) for h in hid])
    v = {}
    v[ARMS[0]] = t1
    x = t1.copy(); x[listed] = ob[listed] == 1; v[ARMS[1]] = x
    y = x.copy()
    if two_block[pid]:
        y[(~listed) & (pf > SHARP)] = False
    v[ARMS[2]] = y
    z = x.copy(); z[~listed] = False; v[ARMS[3]] = z
    v[ARMS[4]] = np.zeros(len(t1), bool)
    w = t1.copy(); w[listed] = True; v[ARMS[5]] = w
    row = {'pair_id': pid, 'family': g['family'][0], 'half': g['half'][0], 'REF_recorded': g['REF'][0]}
    # AP denominators use the full list length, as the audit does (listed hands outside the top-20 count as misses)
    for k, flag in v.items():
        sc = decode(p, flag, cr)
        o5 = np.argsort(-sc, kind='stable')[:5]; r = rel[o5]; hits = np.cumsum(r)
        row[k] = float((hits / np.arange(1, len(o5) + 1) * r).sum() / min(5, nlisted_total))
    per.append(row)
P = pl.DataFrame(per)
P.write_parquet(OUT / 'per_query.parquet')
res = {'reproduction_max_abs_diff_per_query': float((P[ARMS[0]] - P['REF_recorded']).abs().max()),
       'REF_recorded_mean': float(P['REF_recorded'].mean())}
for k in ARMS:
    res[k] = {'all': float(P[k].mean()), 'delta_all': float((P[k] - P[ARMS[0]]).mean()),
              'by_family': {f: [round(float(P.filter(pl.col('family') == f)[k].mean()), 4),
                                round(float((P.filter(pl.col('family') == f)[k] - P.filter(pl.col('family') == f)[ARMS[0]]).mean()), 4)]
                            for f in ['directed_transfer', 'soft_play', 'coordinated_isolation']},
              'halves_delta': [round(float((P.filter(pl.col('half') == h)[k] - P.filter(pl.col('half') == h)[ARMS[0]]).mean()), 4) for h in (0, 1)]}
(OUT / 'oracle.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print(json.dumps(res, indent=1))
