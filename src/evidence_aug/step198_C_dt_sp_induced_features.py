"""STEP198 phase A: are the DT and SP induced rules pointing at defects production actually has?

Two isolated agents induced listing rules from gameplay packets alone (STEP193b). Held out, never
seen during induction: DT 0.5946 precision@5 over 37 pairs, SP 0.5879 / recall 0.6460 over 33 pairs,
against random 0.027 and 0.042. SP beats its STEP115 reference of 0.57/0.54; DT is below its 0.67.

Neither rule is a submission. What is being screened is two claims about production's ENCODING of
events it already detects, from Codex Resume/src/retrospective_strength_features.py (the 32 rs_*
columns S165 loads as pf_cols):

  DT  production anchors CORRECTLY -- context = 'partner' if previous == partner else 'other', where
      previous is the street's last aggressor. This is NOT the pf_others_fold anchor bug again. The
      defect claimed is the FINAL COLLAPSE: values = [a[n]+b[n]] + [max(a[n],b[n])] sums and maxes
      across the two pair members, so every rs_* column is symmetric and cannot say WHICH member
      acted. The induced rule says evidence is directional -- donor = the pair's net loser -- and
      that the direction-blind version is anti-signal (partner folds of any kind: 0.73 listed vs
      1.57 unlisted). dt_any_ahead_fold below is the direction-blind control that lets this FAIL.

  SP  the event is right and the WEIGHTING is wrong: production accumulates s[actor] * log1p(pot/bb),
      and the induced rule measured both weightings as harmful (pot: to_call/pot 0.392 listed vs
      0.391 unlisted; strength: max-hole ranking 0.3071 against a 0.3515 base that ignores it).
      Weaker hypothesis than DT and registered as such -- cs_call_partner_active_count is already
      unweighted, so a tree may form the conjunction from existing columns.

AUC IS A SCREEN THAT CAN ONLY STOP THIS STEP, NEVER JUSTIFY A BUILD. OBS 19.147: +.0052 of stage-1
AUC bought +.0005 of decoded MAP.

Production's own strength() is imported, not reimplemented -- the same evaluator that built the
counterpart columns. Codex Resume stays read-only; bytecode is off.

Audit only. No evaluation pair, no held fold, never uploads.

Usage: python step198_C_dt_sp_induced_features.py REGSHA
"""
import os
import sys
sys.dont_write_bytecode = True          # Codex Resume is READ-ONLY: never leave .pyc behind
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step198_20260917'
REG = OUT / 'registration198.json'
FEAT = OUT / 'induced_features.parquet'
STREET = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}

DT_NEW = ['dt_donor_ahead_fold', 'dt_donor_ahead_fold_gap', 'dt_donor_fold_to_recip',
          'dt_donor_loses_showdown', 'dt_any_ahead_fold']
SP_NEW = ['sp_calls_partner_no_raise', 'sp_calls_partner_count']
NEW = DT_NEW + SP_NEW

# production's nearest counterparts, measured identically
DT_REF = ['rs_partner_fold_ahead_gap_max', 'rs_partner_fold_ahead_gap_sum',
          'rs_partner_fold_strength_max', 'v2_soft_ahead_fold_max', 'v2_soft_ahead_fold_sum']
SP_REF = ['rs_partner_call_strength_max', 'rs_partner_call_strength_sum',
          'rs_partner_call_ahead_gap_max', 'cs_call_partner_active_count',
          'cs_aggr_partner_active_count', 'v2_partner_ahead_call_count_max']


def hand_features(events, seats, board, p1, p2, donor):
    """One (pair, hand). events ordered; seats maps player_id -> dict; board is a list of card ids.

    Mirrors production's iteration conventions exactly so the comparison is like-for-like:
    `previous` is the street's last aggressor and resets each street; aggressive is
    amount > to_call; calling is call/all_in that is not aggressive; features only accumulate while
    BOTH pair members are still live (production's set(pair).issubset(live) guard).
    """
    from retrospective_strength_features import strength
    recip = p2 if donor == p1 else p1
    cards = {p: seats[p]['cards'] for p in (p1, p2)}
    pair = (p1, p2)
    f = dict.fromkeys(NEW, 0.0)
    live = {s for s in seats}
    street, previous, cache = None, None, {}
    sp_calls = sp_raises = 0
    for a in events:
        if a['street'] != street:
            street, previous = a['street'], None
        n = STREET[a['street']]
        if len(board) < n:
            break                                   # hand ended before this street was dealt
        actor, kind = a['player_id'], a['action']
        amount, tocall = int(a['amount'] or 0), int(a['to_call'] or 0)
        aggressive = kind in ('bet', 'raise', 'all_in') and amount > tocall
        calling = kind in ('call', 'all_in') and not aggressive
        if actor in pair and set(pair).issubset(live):
            partner = p2 if actor == p1 else p1
            facing_partner = previous == partner
            if n not in cache:
                cache[n] = {p: strength(cards[p], board[:n]) for p in pair}
            s = cache[n]
            if facing_partner:
                if calling:
                    sp_calls += 1
                elif aggressive:
                    sp_raises += 1
                if kind == 'fold' and n > 0:                      # postflop only
                    ahead = s[actor] > s[partner]
                    if actor == donor:
                        f['dt_donor_fold_to_recip'] = 1.0
                        if ahead:
                            f['dt_donor_ahead_fold'] = 1.0
                            f['dt_donor_ahead_fold_gap'] = max(
                                f['dt_donor_ahead_fold_gap'], s[actor] - s[partner])
                    if ahead:
                        f['dt_any_ahead_fold'] = 1.0              # direction-blind control
        if aggressive:
            previous = actor
        if kind == 'fold':
            live.discard(actor)
    f['sp_calls_partner_count'] = float(sp_calls)
    f['sp_calls_partner_no_raise'] = float(sp_calls >= 1 and sp_raises == 0)
    if seats[p1]['sd'] and seats[p2]['sd'] and len(board) == 5:
        from retrospective_strength_features import strength as _st
        sd = {p: _st(cards[p], board) for p in (p1, p2)}
        f['dt_donor_loses_showdown'] = float(sd[donor] < sd[recip])
    return f


def main():
    reg_sha = sys.argv[1]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == reg_sha, 'registration hash mismatch'
    from phevaluator import Card
    from evidence_aug.step167_C_lp_production import modules
    from evidence_aug.paths import check_protected
    from sklearn.metrics import roc_auc_score
    prot = check_protected()
    S108, _ = modules()
    CSP, KEYS = S108.CSP, S108.KEYS
    t0 = time.perf_counter()

    frame = CSP.development_frame()[0]
    keys = frame.select(KEYS + ['relevance', 'behavior_family']).unique(subset=KEYS)
    lab = pl.read_csv(CSP.ROOT / 'data/development_labels.csv',
                      columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    members = sorted({q for v in pp.values() for q in v})
    print(f'{keys.height} (pair,hand) rows, {len(pp)} pairs, {len(members)} flagged players', flush=True)

    # DIRECTION: donor = lower total net chips over ALL shared hands of the pair, not just the
    # candidate pool. That is the rule's definition and the pool is a subset.
    sm = (pl.scan_parquet(CSP.ROOT / 'data/seats.parquet')
          .filter(pl.col('player_id').is_in(members))
          .select('hand_id', 'player_id', 'net_chips').collect())
    byplayer = {}
    for hid, q, nc in sm.iter_rows():
        byplayer.setdefault(q, {})[hid] = nc
    donor_of = {}
    for pid, (a, b) in pp.items():
        ha, hb = byplayer.get(a, {}), byplayer.get(b, {})
        shared = ha.keys() & hb.keys()
        na = sum(ha[h] for h in shared)
        nb = sum(hb[h] for h in shared)
        donor_of[pid] = a if na <= nb else b
    print(f'direction set for {len(donor_of)} pairs  {round(time.perf_counter()-t0)}s', flush=True)
    del sm, byplayer

    hids = keys['hand_id'].unique().to_list()
    hd = (pl.scan_parquet(CSP.ROOT / 'data/hands.parquet')
          .filter(pl.col('hand_id').is_in(hids)).select('hand_id', 'board_cards').collect())
    board_of = {h: [Card.to_id(c) for c in (b or '').split()] for h, b in hd.iter_rows()}
    st = (pl.scan_parquet(CSP.ROOT / 'data/seats.parquet')
          .filter(pl.col('hand_id').is_in(hids) & pl.col('player_id').is_in(members))
          .select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2', 'went_to_showdown').collect())
    seat_of = {}
    for hid, q, c1, c2, sd in st.iter_rows():
        seat_of.setdefault(hid, {})[q] = {'cards': (Card.to_id(c1), Card.to_id(c2)), 'sd': bool(sd)}
    ac = (pl.scan_parquet(CSP.ROOT / 'data/actions.parquet')
          .filter(pl.col('hand_id').is_in(hids))
          .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call')
          .sort('hand_id', 'action_no').collect())
    ev_of = {}
    for hid, _, street, q, a, amt, tc in ac.iter_rows():
        ev_of.setdefault(hid, []).append(
            {'street': street, 'player_id': q, 'action': a, 'amount': amt, 'to_call': tc})
    print(f'loaded {ac.height} actions over {len(hids)} hands  {round(time.perf_counter()-t0)}s', flush=True)
    del ac, st, hd

    rows = []
    for pid, hid, _, _ in keys.iter_rows():
        p1, p2 = pp[pid]
        seats = seat_of.get(hid, {})
        if p1 not in seats or p2 not in seats:
            f = dict.fromkeys(NEW, 0.0)
        else:
            f = hand_features(ev_of.get(hid, []), seats, board_of.get(hid, []), p1, p2, donor_of[pid])
        f['pair_id'], f['hand_id'] = pid, hid
        rows.append(f)
    feat = pl.DataFrame(rows).select(KEYS + NEW)
    assert feat.height == keys.height
    assert np.isfinite(feat.select(NEW).to_numpy()).all(), 'non-finite feature'
    OUT.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(FEAT)
    print(f'wrote {FEAT.name} {feat.shape}  {round(time.perf_counter()-t0)}s', flush=True)

    # ---- the screen -------------------------------------------------------------------------
    # The 32 rs_* columns are NOT on the CSP development frame -- S165 joins them in separately.
    # The audit query frame carries every production column AND is the frame phase B measures on,
    # so the screen runs there: like-for-like, and no second definition of a production column.
    from evidence_aug import step165_C_e2e_audit as S165
    Q = S165.load()['Q']
    j = Q.select(KEYS + ['relevance', 'behavior_family'] + DT_REF + SP_REF).join(
        feat, on=KEYS, how='left', validate='1:1')
    assert j.select(NEW).null_count().to_numpy().sum() == 0, 'audit rows missing from the build'
    res = {'registration_sha256': reg_sha, 'rows': j.height,
           'note': ('single-feature AUC vs relevance, 162-query audit frame (top-20 rows), per '
                    'family. A SCREEN ONLY -- it can stop this step, never justify a build.'),
           'families': {}}
    for fam, cand, ref in (('directed_transfer', DT_NEW, DT_REF), ('soft_play', SP_NEW, SP_REF)):
        sub = j.filter(pl.col('behavior_family') == fam)
        y = sub['relevance'].to_numpy().astype(int)
        blk = {'n_rows': int(sub.height), 'n_listed': int(y.sum()), 'candidates': {}, 'production': {}}
        for c in cand + ref:
            x = sub[c].to_numpy().astype(float)
            a = float(roc_auc_score(y, x)) if len(np.unique(x)) > 1 else 0.5
            tgt = blk['candidates'] if c in cand else blk['production']
            tgt[c] = {'auc': round(a, 4), 'listed_mean': round(float(x[y == 1].mean()), 4),
                      'unlisted_mean': round(float(x[y == 0].mean()), 4)}
        bc = max(blk['candidates'], key=lambda c: blk['candidates'][c]['auc'])
        bp = max(blk['production'], key=lambda c: blk['production'][c]['auc'])
        blk['best_candidate'] = {'col': bc, 'auc': blk['candidates'][bc]['auc']}
        blk['best_production'] = {'col': bp, 'auc': blk['production'][bp]['auc']}
        blk['gap'] = round(blk['candidates'][bc]['auc'] - blk['production'][bp]['auc'], 4)
        blk['survives_phase_A'] = bool(blk['gap'] > 0)
        res['families'][fam] = blk
    d = res['families']['directed_transfer']['candidates']
    res['DIRECTION_TEST'] = {
        'donor_anchored_auc': d['dt_donor_ahead_fold']['auc'],
        'direction_blind_control_auc': d['dt_any_ahead_fold']['auc'],
        'gap': round(d['dt_donor_ahead_fold']['auc'] - d['dt_any_ahead_fold']['auc'], 4),
        'reading': ('the registered DT hypothesis is that DIRECTION is the payload production throws '
                    'away in its final sum/max collapse. If donor-anchored does not beat the '
                    'direction-blind control, that hypothesis is wrong however good the raw AUC is.')}
    res['seconds'] = round(time.perf_counter() - t0, 1)
    assert check_protected() == prot
    (OUT / 'results198_screen.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
