"""STEP242 -- what actually separates block 1 from block 2 of a truth list?  (development phase only)

Two-block lists give exact block labels for listed hands.  Build plain, role-resolved action facts per listed hand
and let shallow decision trees look for a near-deterministic rule.  If a depth-2/3 tree reaches ~100 %, that is the
generator's block definition and the decoder's T1 flag can be replaced by it.
Output: experiments/step242_block_rule_20260917/
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, GroupKFold

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
sys.path.insert(0, str(CV / 'src'))
from evidence_aug import step104_C_typed_decoder_build as S104
OUT = CV / 'experiments' / 'step242_block_rule_20260917'
OUT.mkdir(parents=True, exist_ok=True)
STREET = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}

lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv').filter(pl.col('label') == 1)
players = {r[0]: (r[1], r[2]) for r in lab.select('pair_id', 'player_1', 'player_2').iter_rows()}
famof = dict(lab.select('pair_id', 'behavior_family').iter_rows())
tabof = {}
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'table_id', 'started_at', 'phase', 'board_cards', 'final_pot', 'big_blind', 'players_dealt', 'players_at_showdown', 'button_seat'])
ev = pl.read_csv(ROOT / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank').join(
    hands.select('hand_id', 'started_at', 'phase'), on='hand_id', how='left')
assert set(ev['phase'].unique().to_list()) == {'development'}
ev = ev.with_columns(t=pl.col('started_at').dt.epoch('s')).drop('started_at')
rows = []
for (pid,), g in ev.group_by(['pair_id']):
    g = g.sort('evidence_rank'); lt = g['t'].to_list(); n = len(lt)
    cut = next((k for k in range(1, n) if lt[k] < lt[k - 1]), n)
    if cut < n and famof[pid] != 'coordinated_isolation':
        for h, b in zip(g['hand_id'].to_list(), [1] * cut + [2] * (n - cut)):
            rows.append((pid, h, b))
K = pl.DataFrame(rows, schema=['pair_id', 'hand_id', 'block'], orient='row')
hid = K['hand_id'].unique().to_list()
H = {r['hand_id']: r for r in hands.filter(pl.col('hand_id').is_in(hid)).drop('started_at').iter_rows(named=True)}
seats = defaultdict(dict)
for r in pl.scan_parquet(ROOT / 'data' / 'seats.parquet').filter(pl.col('hand_id').is_in(hid)).collect().iter_rows(named=True):
    seats[r['hand_id']][r['player_id']] = r
acts = defaultdict(list)
for r in pl.scan_parquet(ROOT / 'data' / 'actions.parquet').filter(pl.col('hand_id').is_in(hid)).collect().sort('hand_id', 'action_no').iter_rows(named=True):
    acts[r['hand_id']].append(r)
# overall chip flow of each member inside the pair's listed+all shared hands is not needed: use the hand itself and the
# pair's development-phase total to resolve roles
allseat = pl.scan_parquet(ROOT / 'data' / 'seats.parquet').join(
    pl.scan_parquet(ROOT / 'data' / 'hands.parquet').filter(pl.col('phase') == 'development').select('hand_id'), on='hand_id').select(
    'hand_id', 'player_id', 'net_chips').collect()
mem = set(p for ab in players.values() for p in ab)
tot = dict(allseat.filter(pl.col('player_id').is_in(list(mem))).group_by('player_id').agg(pl.col('net_chips').sum()).iter_rows())


def member_facts(hidx, me, partner):
    se = seats[hidx]; A = acts[hidx]; bb = H[hidx]['big_blind'] or 1
    s = se[me]; out = {}
    out['folded'] = float(bool(s['folded'])); out['showdown'] = float(bool(s['went_to_showdown']))
    out['net_bb'] = s['net_chips'] / bb; out['contrib_bb'] = s['total_contribution'] / bb
    out['won'] = float((s['won_share'] or 0) > 0)
    mine = [x for x in A if x['player_id'] == me]
    out['n_raise'] = sum(S104.aggressive(x) for x in mine); out['n_call'] = sum(x['action'] == 'call' for x in mine)
    out['n_check'] = sum(x['action'] == 'check' for x in mine); out['n_act'] = len(mine)
    out['vpip'] = float(any(x['street'] == 'preflop' and (x['action'] == 'call' or S104.aggressive(x)) for x in mine))
    st, last, other_in = S104.fold_context(A, me, partner) if s['folded'] else (-1, None, False)
    out['fold_street'] = float(st)
    out['fold_to_partner'] = float(s['folded'] and last == partner); out['fold_to_outsider'] = float(s['folded'] and last not in (None, partner))
    out['fold_unforced'] = float(s['folded'] and last is None)
    out['partner_in_at_fold'] = float(other_in)
    eq = np.nan
    if s['folded'] and other_in and se[partner]['hole_card_1']:
        board = (H[hidx]['board_cards'] or '').split()[:S104.BOARD_LEN[st]]
        eq = S104.equity([s['hole_card_1'], s['hole_card_2']], [se[partner]['hole_card_1'], se[partner]['hole_card_2']], board, 7)
    out['eq_vs_partner_at_fold'] = eq
    # last call facing the partner's bet
    last_ag, c_p, c_p_amt = None, 0, 0.
    street = None
    for x in A:
        if x['street'] != street:
            street, last_ag = x['street'], None
        if x['player_id'] == me and x['action'] == 'call' and last_ag == partner:
            c_p += 1; c_p_amt += (x['amount'] or 0) / bb
        if S104.aggressive(x):
            last_ag = x['player_id']
    out['calls_of_partner'] = c_p; out['calls_of_partner_bb'] = c_p_amt
    return out


feat_rows = []
for pid, h, b in K.iter_rows():
    a, c = players[pid]
    lo, wi = (a, c) if tot.get(a, 0) <= tot.get(c, 0) else (c, a)      # overall chip-loser / chip-winner of the pair
    r = {'pair_id': pid, 'hand_id': h, 'block': b, 'family': famof[pid], 'table_id': H[h]['table_id']}
    for nm, me, pa in (('L', lo, wi), ('W', wi, lo)):
        for k, v in member_facts(h, me, pa).items():
            r[nm + '_' + k] = v
    bb = H[h]['big_blind'] or 1
    r['pot_bb'] = H[h]['final_pot'] / bb; r['n_dealt'] = H[h]['players_dealt']; r['n_showdown'] = H[h]['players_at_showdown']
    r['last_street'] = max(STREET[x['street']] for x in acts[h])
    r['both_showdown'] = r['L_showdown'] * r['W_showdown']
    r['hand_loser_is_L'] = float(r['L_net_bb'] <= r['W_net_bb'])
    r['outsider_won'] = float(r['L_won'] + r['W_won'] == 0)
    feat_rows.append(r)
F = pl.DataFrame(feat_rows)
F.write_parquet(OUT / 'listed_block_facts.parquet')
res = {}
for fam in ['directed_transfer', 'soft_play']:
    f = F.filter(pl.col('family') == fam)
    cols = [c for c in f.columns if c not in ('pair_id', 'hand_id', 'block', 'family', 'table_id')]
    X = np.nan_to_num(f.select(cols).to_numpy().astype(float), nan=-1.)
    y = (f['block'].to_numpy() == 1).astype(int); grp = f['table_id'].to_numpy()
    res[fam] = {'n': int(len(y)), 'share_block1': float(y.mean())}
    print('\n=====', fam, 'n', len(y), 'block-1 share %.3f' % y.mean())
    for depth in (1, 2, 3, 4):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=10, random_state=0)
        cv = cross_val_score(t, X, y, cv=GroupKFold(5), groups=grp).mean()
        t.fit(X, y)
        res[fam]['depth%d' % depth] = {'cv_acc': float(cv), 'train_acc': float(t.score(X, y))}
        print('depth %d: cv acc %.3f train acc %.3f' % (depth, cv, t.score(X, y)))
        if depth == 3:
            print(export_text(t, feature_names=cols))
    # one-feature table: best single split accuracy per feature
    one = []
    for j, c in enumerate(cols):
        t = DecisionTreeClassifier(max_depth=1).fit(X[:, [j]], y)
        one.append((c, t.score(X[:, [j]], y), float(np.nanmean(X[y == 1, j])), float(np.nanmean(X[y == 0, j]))))
    one.sort(key=lambda r: -r[1])
    print('%-28s %6s %10s %10s' % ('single feature', 'acc', 'mean b1', 'mean b2'))
    for r in one[:14]:
        print('%-28s %6.3f %10.3f %10.3f' % r)
    res[fam]['single'] = one[:14]
(OUT / 'block_rule.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
