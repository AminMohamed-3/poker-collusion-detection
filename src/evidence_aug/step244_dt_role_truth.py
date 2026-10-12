"""STEP244 -- directed_transfer: is production's DIRECTION (net chip flow over the pool) the true role?  Development only.

True donor of a positive DT pair = the member who folds to the partner while ahead in the listed hands (STEP242:
the same member in every such hand of a pair, 87 / 87).  Compare with production's dt_sign donor and with an
event-based estimator that needs no labels (majority folder over ALL shared hands where a member folds to the
partner with equity > .5).  Then split the audit AP by whether production's direction is right.
Output: experiments/step244_dt_role_truth_20260917/
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
OUT = CV / 'experiments' / 'step244_dt_role_truth_20260917'
OUT.mkdir(parents=True, exist_ok=True)

lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv').filter((pl.col('label') == 1) & (pl.col('behavior_family') == 'directed_transfer'))
players = {r[0]: (r[1], r[2]) for r in lab.select('pair_id', 'player_1', 'player_2').iter_rows()}
o = pl.read_parquet(CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet').filter(
    pl.col('pair_id').is_in(list(players))).select('pair_id', 'hand_id', 'listed', 'p_fmax')
hid = o['hand_id'].unique().to_list()
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'phase', 'board_cards', 'big_blind']).filter(pl.col('hand_id').is_in(hid))
assert set(hands['phase'].unique().to_list()) == {'development'} and hands.height == len(hid)
boards = dict(hands.select('hand_id', 'board_cards').iter_rows()); bbs = dict(hands.select('hand_id', 'big_blind').iter_rows())
cache = OUT / 'member_fold_facts.parquet'
if cache.exists():
    M = pl.read_parquet(cache)
else:
    seats = defaultdict(dict)
    for r in pl.scan_parquet(ROOT / 'data' / 'seats.parquet').filter(pl.col('hand_id').is_in(hid)).select(
            'hand_id', 'player_id', 'folded', 'hole_card_1', 'hole_card_2', 'net_chips').collect().iter_rows(named=True):
        seats[r['hand_id']][r['player_id']] = r
    acts = defaultdict(list)
    for r in pl.scan_parquet(ROOT / 'data' / 'actions.parquet').filter(pl.col('hand_id').is_in(hid)).select(
            'hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call').collect().sort('hand_id', 'action_no').iter_rows(named=True):
        acts[r['hand_id']].append(r)
    rows = []
    for pid, h in o.select('pair_id', 'hand_id').iter_rows():
        a, b = players[pid]
        rec = {'pair_id': pid, 'hand_id': h, 'n1_bb': seats[h][a]['net_chips'] / (bbs[h] or 1), 'n2_bb': seats[h][b]['net_chips'] / (bbs[h] or 1)}
        for tag, me, pa in (('1', a, b), ('2', b, a)):
            s = seats[h][me]; eq = None
            if s['folded']:
                st, last, other_in = S104.fold_context(acts[h], me, pa)
                if last == pa and other_in:
                    board = (boards[h] or '').split()[:S104.BOARD_LEN[st]]
                    eq = S104.equity([s['hole_card_1'], s['hole_card_2']], [seats[h][pa]['hole_card_1'], seats[h][pa]['hole_card_2']], board, 244)
            rec['ftp_eq_' + tag] = eq
        rows.append(rec)
    M = pl.DataFrame(rows, schema={'pair_id': pl.String, 'hand_id': pl.String, 'n1_bb': pl.Float64, 'n2_bb': pl.Float64, 'ftp_eq_1': pl.Float64, 'ftp_eq_2': pl.Float64})
    M.write_parquet(cache)
D = o.join(M, on=['pair_id', 'hand_id'], how='left')
D = D.with_columns(a1=(pl.col('ftp_eq_1') > .5).fill_null(False), a2=(pl.col('ftp_eq_2') > .5).fill_null(False))
pair = D.group_by('pair_id').agg(
    true1=(pl.col('a1') & (pl.col('listed') == 1)).sum(), true2=(pl.col('a2') & (pl.col('listed') == 1)).sum(),
    ev1=pl.col('a1').sum(), ev2=pl.col('a2').sum(),
    flow=((pl.col('n2_bb') - pl.col('n1_bb')) / 2).sum(), absflow=((pl.col('n2_bb') - pl.col('n1_bb')) / 2).abs().sum(), n=pl.len())
pair = pair.with_columns(
    truth=pl.when((pl.col('true1') > 0) & (pl.col('true2') == 0)).then(1).when((pl.col('true2') > 0) & (pl.col('true1') == 0)).then(2)
            .when((pl.col('true1') == 0) & (pl.col('true2') == 0)).then(0).otherwise(-1),
    prod=pl.when(pl.col('flow') >= 0).then(1).otherwise(2),          # flow >= 0: player_2 is the recipient, so player_1 is the donor
    evt=pl.when(pl.col('ev1') > pl.col('ev2')).then(1).when(pl.col('ev2') > pl.col('ev1')).then(2).otherwise(0),
    margin=(pl.col('flow').abs() / pl.col('absflow')))
res = {'dt_pairs': pair.height, 'truth_counts': dict(pair.group_by('truth').len().sort('truth').iter_rows())}
K = pair.filter(pl.col('truth') > 0)
res['pairs_with_known_donor'] = K.height
res['production_direction_correct'] = float((K['prod'] == K['truth']).mean())
res['event_majority_correct'] = float((K['evt'] == K['truth']).mean())
res['event_majority_undecided'] = float((K['evt'] == 0).mean())
res['both_wrong'] = float(((K['prod'] != K['truth']) & (K['evt'] != K['truth'])).mean())
res['production_correct_by_margin_tercile'] = [float((g['prod'] == g['truth']).mean()) for g in
                                                [K.sort('margin')[i * K.height // 3:(i + 1) * K.height // 3] for i in range(3)]]
# audit AP by whether production's direction is right
Q = pl.read_parquet(CV / 'outputs_step235_20260917' / 'per_query_step235.parquet').filter(pl.col('family') == 'directed_transfer').select('pair_id', 'REF', 'half')
A = Q.join(pair.select('pair_id', 'truth', 'prod', 'evt', 'margin'), on='pair_id', how='left')
res['audit'] = {'dt_queries': A.height,
                'known_donor': int((A['truth'] > 0).sum()),
                'AP_direction_right': [int(((A['truth'] > 0) & (A['prod'] == A['truth'])).sum()), float(A.filter((pl.col('truth') > 0) & (pl.col('prod') == pl.col('truth')))['REF'].mean())],
                'AP_direction_wrong': [int(((A['truth'] > 0) & (A['prod'] != A['truth'])).sum()), float(A.filter((pl.col('truth') > 0) & (pl.col('prod') != pl.col('truth')))['REF'].mean() or 0)],
                'AP_no_type1_listed': [int((A['truth'] == 0).sum()), float(A.filter(pl.col('truth') == 0)['REF'].mean() or 0)]}
pair.write_parquet(OUT / 'pair_roles.parquet')
(OUT / 'role_truth.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
print(json.dumps(res, indent=1))
