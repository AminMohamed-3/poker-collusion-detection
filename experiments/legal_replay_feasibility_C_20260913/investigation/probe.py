"""s103 probe (read-only).
A: replay/conservation check. Full (incl. payout) on a hashed development sample that EXCLUDES the audit
   top-20 hands; public-only (actions + seat ids/stacks + total_contribution parity) on the audit top-20 hands.
B: label-free focal-action eligibility on the 3,240 audit top-20 rows (actions, seat ids/stacks, blinds only).
C: donor corpus: non-query labelled pairs inside the audit tables (ids only) + same eligibility counts.
Never reads label, relevance, behavior_family or evidence columns."""
import sys; sys.dont_write_bytecode = True
import os
for k in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import hashlib, json, time
from collections import Counter, defaultdict
import polars as pl
from phevaluator import evaluate_cards

R = 'C:/Data/Projects/Money Making/work/poker/'
D = R + 'data/'
OUT = 'C:/Users/USER/AppData/Local/Temp/claude/C--Data-Projects-Money-Making-work-poker/2902d2e3-d89a-4488-96d3-ba0785bfb759/scratchpad/s103_agent/'
STREETS = ('preflop', 'flop', 'turn', 'river'); SIDX = {s: i for i, s in enumerate(STREETS)}
BOARD_LEN = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
ACOLS = ['hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'amount_to', 'pot_before', 'stack_before', 'to_call', 'players_active']
T0 = time.time(); TIMES = {}
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)
try:
    import psutil; PROC = psutil.Process()
    def rss(): return round(PROC.memory_info().rss / 2**20)
except Exception:
    def rss(): return None

def load(ids, hcols, scols, acols):
    s = pl.Series('hand_id', sorted(ids))
    H = pl.scan_parquet(D + 'hands.parquet').filter(pl.col('hand_id').is_in(s)).select(hcols).collect(engine='streaming')
    S = pl.scan_parquet(D + 'seats.parquet').filter(pl.col('hand_id').is_in(s)).select(scols).collect(engine='streaming')
    A = pl.scan_parquet(D + 'actions.parquet').filter(pl.col('hand_id').is_in(s)).select(acols).collect(engine='streaming').sort('hand_id', 'action_no')
    hmap = {r['hand_id']: r for r in H.iter_rows(named=True)}
    smap = defaultdict(list); amap = defaultdict(list)
    for r in S.iter_rows(named=True): smap[r['hand_id']].append(r)
    for r in A.iter_rows(named=True): amap[r['hand_id']].append(r)
    return hmap, smap, amap, (H.height, S.height, A.height)

def side_pots(contrib, live, strength, order):
    pay = dict.fromkeys(contrib, 0); odd = 0; prev = 0
    for lv in sorted(set(x for x in contrib.values() if x > 0)):
        layer = sum(max(0, min(x, lv) - prev) for x in contrib.values())
        elig = [p for p in live if contrib[p] >= lv]
        if not elig:
            top = max(contrib[p] for p in live); elig = [p for p in live if contrib[p] == top]
        best = min(strength[p] for p in elig)
        win = sorted((p for p in elig if strength[p] == best), key=order)
        share, rem = divmod(layer, len(win))
        for w in win: pay[w] += share
        if rem: pay[win[0]] += rem; odd += rem
        prev = lv
    return pay, odd

def replay(h, srows, ev, full):
    c = Counter(); v = Counter()
    n = len(srows)
    byp = {r['player_id']: r for r in srows}
    seatp = {int(r['seat_no']): r['player_id'] for r in srows}
    seat_of = {p: int(r['seat_no']) for p, r in byp.items()}
    c['hands'] += 1; c['dealt_%s_seatrows_%s' % (h['players_dealt'], n)] += 1
    if sorted(seatp) != list(range(n)) or len(byp) != n or n != h['players_dealt'] or n < 2:
        c['seat_layout_bad'] += 1; return c, v
    if not ev:
        c['no_actions'] += 1; return c, v
    nums = [e['action_no'] for e in ev]
    c['action_no_start_%s' % nums[0]] += 1
    if nums != list(range(nums[0], nums[0] + len(nums))): v['action_no_gap'] += 1
    btn, sba, bba = int(h['button_seat']), int(h['small_blind']), int(h['big_blind'])
    sbs, bbs = ((btn + 1) % n, (btn + 2) % n) if n > 2 else (btn, (btn + 1) % n)
    sbp, bbp = seatp[sbs], seatp[bbs]
    start = {p: int(r['starting_stack']) for p, r in byp.items()}
    contrib = dict.fromkeys(byp, 0); sbet = dict.fromkeys(byp, 0)
    for p, amt in ((sbp, sba), (bbp, bba)):
        x = min(amt, start[p]); contrib[p] += x; sbet[p] += x
        if x < amt: c['blind_posted_short'] += 1
    folded = set(); allin = {p for p in byp if start[p] > 0 and contrib[p] == start[p]}; acted = set()
    S = {'street': 'preflop', 'cur': max(sbet.values()), 'inc': bba, 'prev': bbs}
    def can_act(p): return p not in folded and p not in allin
    def needs(p): return can_act(p) and (p not in acted or sbet[p] < S['cur'])
    def closed():
        live = [p for p in byp if p not in folded]
        if len(live) <= 1: return True
        lc = [p for p in live if p not in allin]
        if not lc: return True
        if len(lc) == 1 and sbet[lc[0]] >= S['cur']: return True
        return not any(needs(p) for p in byp)
    def next_actor(frm):
        for k in range(1, n + 1):
            p = seatp[(frm + k) % n]
            if needs(p): return p
        return None
    for i, e in enumerate(ev):
        st = e['street']
        if st != S['street']:
            if st not in SIDX: v['unknown_street'] += 1; break
            if not closed(): v['street_change_while_round_open'] += 1
            d = SIDX[st] - SIDX[S['street']]
            if d <= 0: v['street_order_backwards'] += 1
            elif d > 1: v['street_skipped'] += 1
            c['street_changes'] += 1
            S.update(street=st, cur=0, inc=bba, prev=btn)
            for p in sbet: sbet[p] = 0
            acted.clear()
        if closed(): v['action_with_no_pending_decision'] += 1
        actor, a = e['player_id'], e['action']
        amt, tc = int(e['amount']), int(e['to_call'])
        c['actions'] += 1
        if actor not in byp: v['actor_not_seated'] += 1; break
        if actor != next_actor(S['prev']): v['turn_order'] += 1
        if actor in folded: v['actor_already_folded'] += 1
        if actor in allin: v['actor_already_allin'] += 1
        cur = S['cur']; pot = sum(contrib.values()); stk = start[actor] - contrib[actor]; raw = cur - sbet[actor]
        if int(e['pot_before']) != pot: v['pot_before'] += 1
        if int(e['stack_before']) != stk: v['stack_before'] += 1
        c['to_call_eq_raw'] += tc == raw; c['to_call_eq_capped_by_stack'] += tc == min(raw, stk)
        if tc != raw and tc != min(raw, stk): v['to_call'] += 1
        live = [p for p in byp if p not in folded]
        c['players_active_eq_unfolded'] += int(e['players_active']) == len(live)
        c['players_active_eq_unfolded_not_allin'] += int(e['players_active']) == sum(1 for p in live if p not in allin)
        at = int(e['amount_to'])
        c['amount_to_eq_street_total_after'] += at == sbet[actor] + amt
        c['amount_to_eq_hand_total_after'] += at == contrib[actor] + amt
        c['amount_to_eq_amount'] += at == amt
        c['act_' + a] += 1
        if amt < 0 or amt > stk: v['amount_bounds'] += 1
        if a == 'check':
            if raw != 0 or amt != 0: v['illegal_check'] += 1
        elif a == 'fold':
            if amt != 0: v['fold_nonzero_amount'] += 1
            if raw <= 0: c['fold_when_check_available'] += 1
        elif a == 'call':
            if raw <= 0: v['call_with_nothing_to_call'] += 1
            elif amt != min(raw, stk): v['call_amount'] += 1
            if amt == stk: c['call_for_exact_whole_stack'] += 1
        elif a == 'bet':
            if cur != 0: v['bet_facing_bet'] += 1
            if amt <= 0: v['bet_nonpositive'] += 1
            elif amt < bba and amt < stk: v['bet_below_big_blind'] += 1
            if amt == stk: c['bet_for_exact_whole_stack'] += 1
        elif a == 'raise':
            new = sbet[actor] + amt
            if cur <= 0: v['raise_without_bet'] += 1
            if new <= cur: v['raise_not_above_current'] += 1
            elif new - cur < S['inc'] and amt < stk: v['raise_below_min_increment'] += 1
            if amt == stk: c['raise_for_exact_whole_stack'] += 1
        elif a == 'all_in':
            if amt != stk: v['all_in_amount_not_stack'] += 1
            c['all_in_as_call' if amt <= raw else 'all_in_as_raise'] += 1
        else:
            v['unknown_action'] += 1
        new = sbet[actor] + amt
        contrib[actor] += amt; sbet[actor] = new
        if a == 'fold': folded.add(actor)
        if new > cur:
            if new - cur >= S['inc']: S['inc'] = new - cur
            S['cur'] = new
        acted.add(actor)
        if actor not in folded and start[actor] > 0 and contrib[actor] == start[actor]: allin.add(actor)
        S['prev'] = seat_of[actor]
    live = [p for p in byp if p not in folded]
    fold_end = len(live) == 1; last = S['street']
    if not fold_end:
        if not closed(): v['hand_ends_with_decision_pending'] += 1
        if last != 'river' and sum(1 for p in live if p not in allin) >= 2: v['hand_ends_before_river_two_can_bet'] += 1
    c[('end_fold_' if fold_end else 'end_showdown_') + last] += 1
    for p, r in byp.items():
        if int(r['total_contribution']) != contrib[p]: v['final_contribution'] += 1
    if full:
        board = (h['board_cards'] or '').split()
        exp_len = 5 if not fold_end else BOARD_LEN[last]
        if len(board) != exp_len: v['board_length'] += 1; c['board_len_%d_expected_%d' % (len(board), exp_len)] += 1
        if int(h['final_pot']) != sum(int(r['total_contribution']) for r in srows): v['final_pot_vs_contributions'] += 1
        if sum(int(r['net_chips']) for r in srows) != 0: v['net_chips_sum_nonzero'] += 1
        for p, r in byp.items():
            if bool(r['folded']) != (p in folded): v['folded_flag'] += 1
            if bool(r['went_to_showdown']) != ((not fold_end) and p not in folded): v['showdown_flag'] += 1
        if int(h['players_at_showdown']) != sum(1 for r in srows if r['went_to_showdown']): v['players_at_showdown'] += 1
        obs = {p: int(r['net_chips']) + int(r['total_contribution']) for p, r in byp.items()}
        exp = None; odd = 0
        if fold_end:
            exp = {p: (sum(contrib.values()) if p == live[0] else 0) for p in byp}
        else:
            try:
                strength = {p: evaluate_cards(byp[p]['hole_card_1'], byp[p]['hole_card_2'], *board) for p in live}
                exp, odd = side_pots(contrib, live, strength, lambda p: (seat_of[p] - btn - 1) % n)
                c['showdown_evaluated'] += 1
            except Exception:
                v['showdown_not_evaluable'] += 1
        if exp is not None:
            dmax = max(abs(exp[p] - obs[p]) for p in byp)
            if dmax == 0: c['payout_exact'] += 1
            elif odd and dmax <= odd: c['payout_oddchip_assignment_only'] += 1
            else: v['payout_mismatch'] += 1
            fp = max(int(h['final_pot']), 1)
            c['won_share_eq_received_over_final_pot'] += all(abs(float(byp[p]['won_share'] or 0) - obs[p] / fp) < 1e-6 for p in byp)
    cons = ('pot_before', 'stack_before', 'final_contribution', 'final_pot_vs_contributions', 'net_chips_sum_nonzero')
    c['hand_pot_stack_conservation_pass'] += not any(v[k] for k in cons)
    c['hand_turn_street_legality_pass'] += not any(v[k] for k in list(v) if k not in cons)
    c['hand_all_checks_pass'] += not v
    for k, x in list(v.items()):
        c['viol_events_' + k] += x; c['viol_hands_' + k] += 1
    return c, v

def is_callish(e): return e['action'] == 'call' or (e['action'] == 'all_in' and int(e['amount']) <= int(e['to_call']))
def is_agg(e): return e['action'] in ('bet', 'raise') or (e['action'] == 'all_in' and int(e['amount']) > int(e['to_call']))

def elig(srows, ev, p1, p2):
    pair = {p1, p2}; seated = {r['player_id'] for r in srows}
    if not pair <= seated: return None
    folded = set(); street = None; last_agg = None
    rf, rc, rr, hc, hb, outs = [], [], [], [], [], []
    opener = None; open_idx = None; out_folds_after_open = 0
    for i, e in enumerate(ev):
        a, p = e['action'], e['player_id']
        if e['street'] != street: street = e['street']; last_agg = None
        agg, callish = is_agg(e), is_callish(e)
        live_before = seated - folded
        if p in pair:
            q = p2 if p == p1 else p1
            if q not in folded:
                if last_agg == q:
                    if a == 'fold': rf.append(i)
                    elif callish: rc.append(i)
                    elif agg: rr.append(i)
                if street != 'preflop' and live_before == pair:
                    if a == 'check': hc.append(i)
                    elif agg and last_agg != q: hb.append(i)
        else:
            outs.append(i)
        if street == 'preflop':
            if opener is None and agg: opener, open_idx = p, i
            elif opener is not None and p not in pair and a == 'fold': out_folds_after_open += 1
        if agg: last_agg = p
        if a == 'fold': folded.add(p)
    n = len(ev); last = n - 1
    def swap_exact(f):
        if f != last: return False
        return is_callish(ev[f]) or (ev[f]['action'] == 'fold' and ev[f]['street'] == 'river')
    foc = {}
    if len(rf) + len(rc) == 1 and not rr: foc['DT_single_response_to_partner_aggression'] = (rf + rc)[0]
    if len(rf) + len(rc) + len(hc) == 1 and not rr and not hb: foc['SP_single_passive_vs_partner'] = (rf + rc + hc)[0]
    if opener in pair and out_folds_after_open >= 3: foc['CI_member_open_ge3_outsider_folds'] = open_idx
    row = {'n_actions': n, 'n_resp_fold': len(rf), 'n_resp_call': len(rc), 'n_resp_raise': len(rr), 'n_hu_check': len(hc),
           'n_hu_bet': len(hb), 'n_outsider_actions': len(outs), 'member_open': int(opener in pair)}
    for k, f in foc.items():
        row[k] = 1; row[k + '__terminal'] = int(f == last); row[k + '__swap_exact'] = int(swap_exact(f))
        row[k + '__trunc_removed_frac'] = (n - f) / n; row[k + '__action'] = ev[f]['action']; row[k + '__street'] = ev[f]['street']
        row[k + '__outsider_action_same_street'] = int(any(ev[j]['street'] == ev[f]['street'] for j in outs))
    row['last_action_outsider_swap_exact'] = int(n > 0 and ev[last]['player_id'] not in pair and swap_exact(last))
    return row

FOC = ['DT_single_response_to_partner_aggression', 'SP_single_passive_vs_partner', 'CI_member_open_ge3_outsider_folds']
def summarize(E):
    out = {'rows': E.height, 'queries': E['pair_id'].n_unique(), 'tables': E['table_id'].n_unique(),
           'rows_any_response_to_partner_aggression': int(((E['n_resp_fold'] + E['n_resp_call']) >= 1).sum()),
           'rows_ge2_responses_to_partner_aggression': int(((E['n_resp_fold'] + E['n_resp_call']) >= 2).sum()),
           'rows_any_member_raise_vs_partner': int((E['n_resp_raise'] >= 1).sum()),
           'rows_member_open': int(E['member_open'].sum()),
           'rows_last_action_outsider_swap_exact': int(E['last_action_outsider_swap_exact'].sum()),
           'n_actions_quantiles_10_25_50_75_90': [float(E['n_actions'].quantile(q)) for q in (.1, .25, .5, .75, .9)]}
    for k in FOC:
        if k not in E.columns: out[k] = {'rows': 0}; continue
        e = E.with_columns(pl.col(k).fill_null(0), pl.col(k + '__swap_exact').fill_null(0))
        pq_ = e.group_by('pair_id').agg(pl.col(k).sum().alias('n'), pl.col(k + '__swap_exact').sum().alias('x'))
        sub = e.filter(pl.col(k) == 1)
        out[k] = {'rows': sub.height, 'share': round(sub.height / E.height, 4),
                  'queries_ge1': int((pq_['n'] >= 1).sum()), 'queries_ge3': int((pq_['n'] >= 3).sum()), 'queries_ge5': int((pq_['n'] >= 5).sum()),
                  'median_per_query': float(pq_['n'].median()), 'tables_ge1': int(sub['table_id'].n_unique()),
                  'terminal_rows': int(sub[k + '__terminal'].sum()), 'swap_exact_rows': int(sub[k + '__swap_exact'].sum()),
                  'queries_swap_exact_ge1': int((pq_['x'] >= 1).sum()), 'queries_swap_exact_ge3': int((pq_['x'] >= 3).sum()),
                  'tables_swap_exact_ge1': int(sub.filter(pl.col(k + '__swap_exact') == 1)['table_id'].n_unique()),
                  'mean_frac_actions_removed_by_truncation': round(float(sub[k + '__trunc_removed_frac'].mean()), 3),
                  'rows_outsider_action_same_street': int(sub[k + '__outsider_action_same_street'].sum()),
                  'focal_action': dict(sub[k + '__action'].value_counts().iter_rows()),
                  'focal_street': dict(sub[k + '__street'].value_counts().iter_rows())}
    return out

res = {}
phase_counts = dict(pl.scan_parquet(D + 'hands.parquet').group_by('phase').len().collect().iter_rows())
sc_path = R + 'Codex Resume/experiments/symmetry_censor_20260912/scores.parquet'
sc = pl.read_parquet(sc_path, columns=['pair_id', 'hand_id', 'stage1_only']).sort('pair_id', 'hand_id')
top = (sc.with_columns(pl.col('stage1_only').rank('ordinal', descending=True).over('pair_id').alias('r'))
         .filter(pl.col('r') <= 20).select('pair_id', 'hand_id'))
assert top.height == 3240 and top['pair_id'].n_unique() == 162
audit_all_hands = set(sc['hand_id'].to_list()); audit_top_hands = set(top['hand_id'].to_list())
tid = pl.scan_parquet(D + 'hands.parquet').filter(pl.col('hand_id').is_in(pl.Series(sorted(audit_all_hands)))).select('hand_id', 'table_id', 'phase').collect()
assert set(tid['phase'].unique()) == {'development'}
top = top.join(tid.select('hand_id', 'table_id'), on='hand_id', how='left', validate='m:1')
audit_tables = set(tid['table_id'].to_list()); query_pairs = set(top['pair_id'].to_list())
res['keys'] = {'phase_hand_counts': phase_counts, 'audit_pair_hands': sc.height, 'audit_distinct_hands': len(audit_all_hands),
               'audit_top20_rows': top.height, 'audit_top20_distinct_hands': len(audit_top_hands), 'audit_tables': len(audit_tables)}
log(res['keys'], 'rss', rss())
t = time.time()
dev_ids = pl.scan_parquet(D + 'hands.parquet').filter(pl.col('phase') == 'development').select('hand_id').collect()['hand_id'].to_list()
sample = sorted(h for h in dev_ids if h not in audit_top_hands and int(hashlib.sha256(('s103:' + h).encode()).hexdigest()[:8], 16) % 50 == 0)
n_dev = len(dev_ids); del dev_ids
HF = ['hand_id', 'button_seat', 'small_blind', 'big_blind', 'players_dealt', 'board_cards', 'final_pot', 'players_at_showdown']
SF = ['hand_id', 'player_id', 'seat_no', 'starting_stack', 'hole_card_1', 'hole_card_2', 'total_contribution', 'net_chips', 'folded', 'went_to_showdown', 'won_share']
hmap, smap, amap, shp = load(sample, HF, SF, ACOLS)
TIMES['A_load_sample_s'] = round(time.time() - t, 1); log('sample loaded', len(sample), shp, 'rss', rss())
t = time.time(); tot = Counter(); EX = defaultdict(list)
for hid in sample:
    cc, vv = replay(hmap[hid], smap[hid], amap.get(hid, []), True); tot.update(cc)
    for k in vv:
        if len(EX[k]) < 3: EX[k].append(hid)
TIMES['A_replay_sample_s'] = round(time.time() - t, 1)
res['A_dev_sample_full'] = {'development_hands': n_dev, 'rule': "sha256('s103:'+hand_id)[:8] % 50 == 0, audit top-20 hands excluded", **dict(sorted(tot.items()))}
res['A_dev_sample_violation_examples'] = dict(EX)
del hmap, smap, amap
log('A sample done', 'rss', rss())
t = time.time()
HP = ['hand_id', 'button_seat', 'small_blind', 'big_blind', 'players_dealt']
SP = ['hand_id', 'player_id', 'seat_no', 'starting_stack', 'total_contribution']
hmap, smap, amap, shp = load(audit_top_hands, HP, SP, ACOLS)
TIMES['A_load_audit_s'] = round(time.time() - t, 1)
t = time.time(); tot = Counter()
for hid in sorted(audit_top_hands): tot.update(replay(hmap[hid], smap[hid], amap.get(hid, []), False)[0])
TIMES['A_replay_audit_public_only_s'] = round(time.time() - t, 2)
res['A_audit_top20_public_only'] = dict(sorted(tot.items()))
log('A audit done', 'rss', rss())
pm = {a: (b, c) for a, b, c in pl.read_csv(D + 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2']).select('pair_id', 'player_1', 'player_2').iter_rows()}
t = time.time(); rows = []
for pid, hid, tb in top.select('pair_id', 'hand_id', 'table_id').iter_rows():
    r = elig(smap[hid], amap.get(hid, []), *pm[pid]); assert r is not None
    r.update(pair_id=pid, hand_id=hid, table_id=tb); rows.append(r)
TIMES['B_eligibility_3240_rows_s'] = round(time.time() - t, 2)
E = pl.DataFrame(rows, infer_schema_length=None); E.write_parquet(OUT + 'eligibility_audit_top20.parquet')
res['B_audit_top20'] = summarize(E)
del hmap, smap, amap
log('B done', 'rss', rss())
json.dump(res, open(OUT + 'probe_results_AB.json', 'w'), indent=1, default=str)
t = time.time(); ca = None
for base in (R, R + 'Codex Resume/', R + 'Claude Version/'):
    p = base + 'reports/conditional-action-experiment/policy-screen/hand-features.parquet'
    if os.path.exists(p): ca = p; break
res['C_ca_hand_features_path'] = ca
if ca:
    allpairs = pl.scan_parquet(ca).select('pair_id', 'hand_id', 'table_id').collect()
    AT = pl.Series(sorted(audit_tables)); QP = pl.Series(sorted(query_pairs))
    inaud = allpairs.filter(pl.col('table_id').is_in(AT))
    donors = inaud.filter(~pl.col('pair_id').is_in(QP)); qin = inaud.filter(pl.col('pair_id').is_in(QP))
    res['C_labelled_all'] = {'pairs': allpairs['pair_id'].n_unique(), 'pair_hands': allpairs.height, 'tables': allpairs['table_id'].n_unique()}
    res['C_audit_tables'] = {'labelled_pairs': inaud['pair_id'].n_unique(), 'query_pairs': qin['pair_id'].n_unique(), 'query_pair_hands': qin.height,
                             'non_query_pairs': donors['pair_id'].n_unique(), 'non_query_pair_hands': donors.height,
                             'non_query_distinct_hands': donors['hand_id'].n_unique()}
    del allpairs
    qplayers = defaultdict(set)
    for q in query_pairs:
        qplayers[pm[q][0]].add(q); qplayers[pm[q][1]].add(q)
    dhands = sorted(set(donors['hand_id'].to_list())); rows = []; chunk = 12000
    for s0 in range(0, len(dhands), chunk):
        part = dhands[s0:s0 + chunk]
        _, smap, amap, _ = load(part, ['hand_id'], ['hand_id', 'player_id', 'seat_no'], ['hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call'])
        for pid, hid, tb in donors.filter(pl.col('hand_id').is_in(pl.Series(part))).iter_rows():
            seated = {r['player_id'] for r in smap[hid]}
            qs = set().union(*(qplayers[x] for x in seated if x in qplayers))
            r = elig(smap[hid], amap.get(hid, []), *pm[pid])
            if r is None: continue
            r.update(pair_id=pid, hand_id=hid, table_id=tb, query_pair_coseated=int(any(pm[q][0] in seated and pm[q][1] in seated for q in qs)))
            rows.append(r)
        del smap, amap; log('C chunk', s0, 'rss', rss())
    DE = pl.DataFrame(rows, infer_schema_length=None); DE.write_parquet(OUT + 'eligibility_donors_audit_tables.parquet')
    res['C_donor_eligibility'] = summarize(DE)
    res['C_donor_eligibility']['pair_hands_with_query_pair_coseated'] = int(DE['query_pair_coseated'].sum())
    for k in FOC:
        if k in DE.columns:
            res['C_donor_eligibility'][k]['rows_without_query_pair_coseated'] = DE.filter((pl.col(k) == 1) & (pl.col('query_pair_coseated') == 0)).height
    TIMES['C_total_s'] = round(time.time() - t, 1)
res['times'] = TIMES; res['rss_mb_end'] = rss()
json.dump(res, open(OUT + 'probe_results.json', 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
