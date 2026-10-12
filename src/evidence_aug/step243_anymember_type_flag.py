"""STEP243 -- a corrected hard first-type flag for the typed decoder (audit, development phase only, no fitting).

Production T1 (STEP104): the member with the lower net chips in the hand folded facing the partner, equity > .5.
ANY flag: EITHER member folded facing the partner's aggression with equity against the partner above a threshold.
Registration: experiments/step243_anymember_type_20260917/registration243.json (written first).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
sys.path.insert(0, str(CV / 'src'))
from evidence_aug import step104_C_typed_decoder_build as S104
OUT = CV / 'experiments' / 'step243_anymember_type_20260917'
assert (OUT / 'registration243.json').exists(), 'register first'
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


lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv')
players = {r[0]: (r[1], r[2]) for r in lab.select('pair_id', 'player_1', 'player_2').iter_rows()}
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'started_at', 'phase', 'board_cards'])
ev = pl.read_csv(ROOT / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
A = pl.read_parquet(CV / 'outputs_step235_20260917' / 'audit_scores_step235.parquet').select('pair_id', 'hand_id', 'v1_REF').filter(pl.col('v1_REF') > -1e5)
Q = pl.read_parquet(CV / 'outputs_step235_20260917' / 'per_query_step235.parquet').select('pair_id', 'family', 'half', 'REF')
T = pl.read_parquet(CV / 'experiments' / 'typed_listing_decoder_C_20260913' / 'audit-type-attributes.parquet').select('pair_id', 'hand_id', 'T1')
A = A.join(Q.select('pair_id', 'family'), on='pair_id', how='left').join(T, on=['pair_id', 'hand_id'], how='left').join(
    hands.select('hand_id', 'started_at', 'phase'), on='hand_id', how='left')
assert set(A['phase'].unique().to_list()) == {'development'} and A['T1'].null_count() == 0
A = A.with_columns(t=pl.col('started_at').dt.epoch('s')).drop('started_at', 'phase')

cache = OUT / 'any_member_fold_equity.parquet'
if cache.exists():
    E = pl.read_parquet(cache)
else:
    hid = A.filter(pl.col('family') != CI)['hand_id'].unique().to_list()
    boards = dict(hands.filter(pl.col('hand_id').is_in(hid)).select('hand_id', 'board_cards').iter_rows())
    seats = defaultdict(dict)
    for r in pl.scan_parquet(ROOT / 'data' / 'seats.parquet').filter(pl.col('hand_id').is_in(hid)).select(
            'hand_id', 'player_id', 'folded', 'hole_card_1', 'hole_card_2').collect().iter_rows(named=True):
        seats[r['hand_id']][r['player_id']] = r
    acts = defaultdict(list)
    for r in pl.scan_parquet(ROOT / 'data' / 'actions.parquet').filter(pl.col('hand_id').is_in(hid)).select(
            'hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call').collect().sort('hand_id', 'action_no').iter_rows(named=True):
        acts[r['hand_id']].append(r)
    rows = []
    for pid, h in A.filter(pl.col('family') != CI).select('pair_id', 'hand_id').iter_rows():
        a, b = players[pid]; best = -1.0
        for me, pa in ((a, b), (b, a)):
            s = seats[h][me]
            if not s['folded']:
                continue
            st, last, other_in = S104.fold_context(acts[h], me, pa)
            if last == pa and other_in:
                board = (boards[h] or '').split()[:S104.BOARD_LEN[st]]
                eq = S104.equity([s['hole_card_1'], s['hole_card_2']], [seats[h][pa]['hole_card_1'], seats[h][pa]['hole_card_2']], board, 243)
                best = max(best, eq)
        rows.append((pid, h, best))
    E = pl.DataFrame(rows, schema=['pair_id', 'hand_id', 'any_fold_eq'], orient='row')
    E.write_parquet(cache)
A = A.join(E, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('any_fold_eq').fill_null(-1.0)).join(ev, on=['pair_id', 'hand_id'], how='left')
ntrue = dict(ev.group_by('pair_id').len().iter_rows())
# agreement of the flags on the audit rows
ag = A.filter(pl.col('family') != CI).with_columns(any50=pl.col('any_fold_eq') > .5)
print('flag agreement on DT+SP audit rows:', ag.group_by('T1', 'any50').len().sort('T1', 'any50').to_dicts())
ARMS = ['REF', 'ANY50', 'ANY40', 'ANY50_DT', 'ANY50_SP']
per = []
for (pid,), g in A.group_by(['pair_id']):
    g = g.sort('t'); fam = g['family'][0]
    p = np.clip(g['v1_REF'].to_numpy() - 10, 0, 1); cr = np.arange(g.height, dtype=float)
    rel = g['evidence_rank'].is_not_null().to_numpy().astype(float)
    t1 = g['T1'].to_numpy().astype(bool) & (fam != CI)
    ref = decode(p, t1, cr); sc = {a: ref for a in ARMS}
    if fam != CI:
        e = g['any_fold_eq'].to_numpy()
        a50, a40 = decode(p, e > .5, cr), decode(p, e > .4, cr)
        sc = {'REF': ref, 'ANY50': a50, 'ANY40': a40, 'ANY50_DT': a50 if fam == 'directed_transfer' else ref,
              'ANY50_SP': a50 if fam == 'soft_play' else ref}
    row = {'pair_id': pid, 'family': fam}
    for a in ARMS:
        row[a] = ap(sc[a], rel, ntrue[pid])
    per.append(row)
R = pl.DataFrame(per).join(Q.select('pair_id', 'half', pl.col('REF').alias('REF_recorded')), on='pair_id')
assert float((R['REF'] - R['REF_recorded']).abs().max()) == 0.0, 'REF not reproduced'
R.write_parquet(OUT / 'per_query.parquet')
rng = np.random.default_rng(243)
res = {'REF': float(R['REF'].mean())}
for a in ARMS[1:]:
    dl = (R[a] - R['REF']).to_numpy(); fm = R['family'].to_numpy(); hv = R['half'].to_numpy()
    bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(4000)]
    res[a] = {'map5': float(R[a].mean()), 'delta': float(dl.mean()), 'halves': [float(dl[hv == 0].mean()), float(dl[hv == 1].mean())],
              'by_family': {f: {'delta': float(dl[fm == f].mean()), 'halves': [float(dl[(fm == f) & (hv == 0)].mean()), float(dl[(fm == f) & (hv == 1)].mean())],
                                'up': int((dl[fm == f] > 0).sum()), 'down': int((dl[fm == f] < 0).sum())} for f in ['directed_transfer', 'soft_play', CI]},
              'boot90': [float(np.quantile(bs, .05)), float(np.quantile(bs, .95))]}
(OUT / 'results243.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
for a in ARMS[1:]:
    v = res[a]
    print(a, 'delta %+.4f halves %s boot90 %s' % (v['delta'], [round(x, 4) for x in v['halves']], [round(x, 4) for x in v['boot90']]))
    for f, w in v['by_family'].items():
        print('    %-24s %+.4f halves %s up/down %d/%d' % (f, w['delta'], [round(x, 4) for x in w['halves']], w['up'], w['down']))
