"""Step 70 (plan E7): mechanism-specific structural features for directed transfer and soft play.

The ~70 features of step 65 were built for the coordinated-isolation gap and are pre-flop shaped:
who opened, over how many limpers, what the partner did, how many others continued.  They describe
*who was in the pot*.  Almost nothing in them describes the two mechanisms the other two labels are
named after -- chips moving from one member to the other, and members declining to bet into each
other.  E7 asks whether that omission is why stage 2's structural gain did not transfer.

This step adds a separate `f2_` block, keyed by (pair_id, hand_id), and never touches
`step65_structural_evidence.build`: the production caches under
`experiments/family_stage2_20260911/eval_structural/` depend on the old column set exactly.

Directed transfer (chips move A -> B in this hand)
  f2_opposite_signs            the two members' net chips have opposite signs
  f2_transfer_bb               min(|A net|, |B net|) when the signs oppose, else 0 -- the amount
                               that could have changed hands between them
  f2_transfer_frac_pot         the same as a fraction of the final pot
  f2_loser_put_bb              what the losing member put in
  f2_loser_share_of_pot        the losing member's contribution / final pot
  f2_members_only_showdown     both members reached showdown
  f2_member_bet_partner_fold   a member's aggressive action was the last one before the partner folded
  f2_partner_fold_strong       that partner folded holding Chen >= 8
  f2_loser_called_streets      streets on which the losing member called and never raised
  f2_winner_was_behind_session the member who won this hand is the one who was behind between the two
                               over the pair's EARLIER hands (chronology from started_at, never ids)
  f2_session_net_gap_prev      that prior running gap, signed: winner's prior net minus loser's
Soft play (members decline value against each other)
  f2_both_active_streets       streets on which neither member had folded
  f2_checked_through_members   streets both members saw with no aggression from either
  f2_member_aggr_partner_in    aggressive actions by a member while the partner was still in
  f2_member_aggr_partner_out   the same after the partner had folded
  f2_aggr_ratio_in_out         (in + 1) / (out + 1) -- soft play should sit below 1
  f2_member_check_strong       a member checked a street holding Chen >= 10 with the partner in
  f2_late_street_bets          bets and raises on turn and river with both members still in
  f2_sd_won_low_aggression     a member won at showdown with fewer than two late aggressive actions
  f2_pot_vs_table_median       final pot / the median final pot of that table's audited hands
  f2_pot_vs_pair_median        final pot / the median final pot of that pair's own hands

Then the step-65h2 / 68a training-only harness: per-family GroupKFold(5) OOF classifiers over the
top-20 stage-1 candidates, `ref` and `r382` with and without the block, paired bootstrap by table.
Development labels are used for validation only; no evaluation label exists and no leaderboard
outcome enters anything here.

Pre-registered gate (written before any number below was computed): a design joins the next held
freeze only if its paired delta against its own base is >= +.010 in some family with that family's
bootstrap lower bound > 0, and no family below -.005.

Outputs: experiments/label_process_20260911/audit-f2-features.parquet and audit-f2.json
"""
import os

os.environ['POLARS_MAX_THREADS'] = '2'
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from evidence_aug.frame import FAMILIES
from evidence_aug.paths import ORIGINAL, check_protected, write_json
from evidence_aug.step14_audit_truncation import CLF, OUT, TOP
from evidence_aug.step65_structural_evidence import chen, log
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step68a_audit_stage2_ensemble import S3, run, score

GATE = {'family_delta_ge': .010, 'family_ci95_lower_gt': 0.0, 'other_family_floor': -.005,
        'note': 'written before any number in this file was computed'}


def is_aggr(a):
    """a = (street, player_id, action, amount, amount_to, to_call, players_active)"""
    return a[2] in ('raise', 'bet') or (a[2] == 'all_in' and (a[3] or 0) > (a[5] or 0))


def f2_hand(acts, seats, p1, p2, bb, pot):
    role = {p1: 'A', p2: 'B'}
    sa, sb = seats[p1], seats[p2]
    na, nb = sa['net_chips'] / bb, sb['net_chips'] / bb
    f = {}
    opp = (na > 0) != (nb > 0) and abs(na) > 1e-9 and abs(nb) > 1e-9
    f['f2_opposite_signs'] = float(opp)
    f['f2_transfer_bb'] = float(min(abs(na), abs(nb))) if opp else 0.0
    f['f2_transfer_frac_pot'] = f['f2_transfer_bb'] * bb / pot if pot > 0 else 0.0
    loser, winner = (sa, sb) if na < nb else (sb, sa)
    f['f2_loser_put_bb'] = loser['total_contribution'] / bb
    f['f2_loser_share_of_pot'] = loser['total_contribution'] / pot if pot > 0 else 0.0
    f['f2_members_only_showdown'] = float(bool(sa['went_to_showdown']) and bool(sb['went_to_showdown']))
    ca = chen(sa['hole_card_1'], sa['hole_card_2'])
    cb = chen(sb['hole_card_1'], sb['hole_card_2'])

    last_aggr = None                      # player id of the most recent aggressive action
    bet_then_partner_fold = 0.0
    partner_fold_strong = 0.0
    aggr_in, aggr_out, late_bets = 0, 0, 0
    check_strong = 0.0
    folded = set()
    loser_id, winner_id = loser['player_id'], winner['player_id']
    loser_called, loser_raised = set(), False
    aggr_streets, both_streets = set(), set()
    late_aggr_winner = 0
    for a in acts:
        st, pid, act = a[0], a[1], a[2]
        both_in = (p1 not in folded) and (p2 not in folded)
        if both_in:
            both_streets.add(st)
        if is_aggr(a):
            if pid in role:
                partner = p2 if pid == p1 else p1
                if partner in folded:
                    aggr_out += 1
                else:
                    aggr_in += 1
                    if st in ('turn', 'river'):
                        late_bets += 1
                if pid == loser_id:
                    loser_raised = True
                if pid == winner_id and st in ('turn', 'river'):
                    late_aggr_winner += 1
                aggr_streets.add(st)
            last_aggr = pid
        elif act == 'call':
            if pid == loser_id:
                loser_called.add(st)
        elif act == 'fold':
            if pid in role and last_aggr is not None and last_aggr == (p2 if pid == p1 else p1):
                bet_then_partner_fold = 1.0
                if (ca if pid == p1 else cb) >= 8:
                    partner_fold_strong = 1.0
            folded.add(pid)
        elif act == 'check' and pid in role and both_in:
            if (ca if pid == p1 else cb) >= 10:
                check_strong = 1.0
    f['f2_member_bet_partner_fold'] = bet_then_partner_fold
    f['f2_partner_fold_strong'] = partner_fold_strong
    f['f2_loser_called_streets'] = 0.0 if loser_raised else float(len(loser_called))
    f['f2_both_active_streets'] = float(len(both_streets))
    f['f2_checked_through_members'] = float(len(both_streets - aggr_streets))
    f['f2_member_aggr_partner_in'] = float(aggr_in)
    f['f2_member_aggr_partner_out'] = float(aggr_out)
    f['f2_aggr_ratio_in_out'] = (aggr_in + 1.0) / (aggr_out + 1.0)
    f['f2_member_check_strong'] = check_strong
    f['f2_late_street_bets'] = float(late_bets)
    won_sd = bool(winner['went_to_showdown']) and winner['won_share'] > 0
    f['f2_sd_won_low_aggression'] = float(won_sd and late_aggr_winner < 2)
    f['_pot_bb'] = pot / bb
    f['_netA'], f['_netB'] = float(na), float(nb)
    return f


def build2(audit, pair_players=None):
    """audit needs pair_id, hand_id, chrono_rank (chrono_rank comes from started_at, not from ids)."""
    t0 = time.time()
    labels = (pair_players if pair_players is not None
              else pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
              .select('pair_id', 'player_1', 'player_2'))
    pairs = dict(labels.filter(pl.col('pair_id').is_in(audit['pair_id'].unique().implode()))
                 .select('pair_id', pl.struct('player_1', 'player_2')).iter_rows())
    hids = audit['hand_id'].unique().implode()
    hands = (pl.read_parquet(ORIGINAL / 'data' / 'hands.parquet',
                             columns=['hand_id', 'table_id', 'big_blind', 'final_pot'])
             .filter(pl.col('hand_id').is_in(hids)))
    hinfo = {r['hand_id']: r for r in hands.iter_rows(named=True)}
    seats = pl.read_parquet(ORIGINAL / 'data' / 'seats.parquet').filter(pl.col('hand_id').is_in(hids))
    srows = {}
    for r in seats.iter_rows(named=True):
        srows.setdefault(r['hand_id'], {})[r['player_id']] = r
    acts = (pl.scan_parquet(ORIGINAL / 'data' / 'actions.parquet').filter(pl.col('hand_id').is_in(hids))
            .sort('hand_id', 'action_no').collect())
    arows = {}
    for hid, st, pid, act, amt, amt_to, tc, pa in acts.select(
            'hand_id', 'street', 'player_id', 'action', 'amount', 'amount_to', 'to_call',
            'players_active').iter_rows():
        arows.setdefault(hid, []).append((st, pid, act, amt, amt_to, tc, pa))
    log('f2 loaded', hands.height, 'hands', acts.height, 'actions', round(time.time() - t0))

    tab_med = dict(hands.with_columns((pl.col('final_pot') / pl.col('big_blind')).alias('pb'))
                   .group_by('table_id').agg(pl.col('pb').median()).iter_rows())
    rows = []
    for pid, hid in audit.select('pair_id', 'hand_id').iter_rows():
        h = hinfo[hid]
        pp = pairs[pid]
        f = f2_hand(arows.get(hid, []), srows[hid], pp['player_1'], pp['player_2'],
                    h['big_blind'], h['final_pot'])
        f['pair_id'], f['hand_id'] = pid, hid
        f['_tabmed'] = tab_med[h['table_id']]
        rows.append(f)
    feat = pl.DataFrame(rows).join(audit.select('pair_id', 'hand_id', 'chrono_rank'),
                                   on=['pair_id', 'hand_id'], validate='1:1')
    feat = feat.with_columns((pl.col('_pot_bb') / pl.col('_tabmed')).alias('f2_pot_vs_table_median'))
    med = feat.group_by('pair_id').agg(pl.col('_pot_bb').median().alias('_pm'))
    feat = (feat.join(med, on='pair_id')
            .with_columns((pl.col('_pot_bb') / pl.col('_pm')).alias('f2_pot_vs_pair_median')))

    # who was ahead between the two members over the pair's earlier hands (exclusive running sum)
    feat = feat.sort('pair_id', 'chrono_rank')
    feat = feat.with_columns([
        (pl.col('_netA').cum_sum().over('pair_id') - pl.col('_netA')).alias('_cumA'),
        (pl.col('_netB').cum_sum().over('pair_id') - pl.col('_netB')).alias('_cumB')])
    winA = pl.col('_netA') > pl.col('_netB')
    gap = (pl.when(winA).then(pl.col('_cumA') - pl.col('_cumB'))
           .otherwise(pl.col('_cumB') - pl.col('_cumA')))
    feat = feat.with_columns([gap.alias('f2_session_net_gap_prev'),
                              (gap < 0).cast(pl.Float64).alias('f2_winner_was_behind_session')])
    feat = feat.drop([c for c in feat.columns if c.startswith('_')] + ['chrono_rank'])
    log('f2 features', feat.shape, round(time.time() - t0))
    return feat


def main():
    t0 = time.time()
    check_protected()
    from evidence_aug.fold import load_fold
    audit = pl.read_parquet(OUT / 'audit-oof.parquet')
    base = pl.read_parquet(OUT / 'audit-structural-features.parquet')
    cache = OUT / 'audit-f2-features.parquet'
    if cache.exists():
        f2 = pl.read_parquet(cache)
    else:
        f2 = build2(audit)
        f2.write_parquet(cache)
    f2cols = [c for c in f2.columns if c.startswith('f2_')]

    d = (audit.join(base, on=['pair_id', 'hand_id'], validate='1:1')
         .join(f2.select(['pair_id', 'hand_id'] + f2cols), on=['pair_id', 'hand_id'], validate='1:1')
         .sort('pair_id', 'hand_id'))
    struct = [c for c in base.columns if c not in ('pair_id', 'hand_id')]
    sc, chrono = ['oof', 'oof_rank'], ['chrono_rank', 'chrono_frac', 'n_hands']
    data, features, _, _ = load_fold(0)
    rcols = [f for f in features if not f.startswith('family_') and f in data.columns
             and f not in struct + chrono + sc]
    d = d.join(data.select(['pair_id', 'hand_id'] + rcols).with_columns(pl.lit(1).alias('_hit')),
               on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert d['_hit'].null_count() == 0
    d = d.drop('_hit').with_columns(pl.col(rcols).fill_null(-999.0))
    d = d.with_columns(pl.col(f2cols).cast(pl.Float64).fill_nan(0.0).fill_null(0.0))

    truths = {k[0]: set(g['hand_id']) for k, g in d.filter(pl.col('relevance') == 1).group_by('pair_id')}
    labels = dict(d.select('pair_id', 'behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id', 'table_id').unique().iter_rows())
    pools = sorted(set(ptab.values()))
    y = d['relevance'].to_numpy().astype(float)
    groups = d['table_id'].to_numpy()
    key = d.select('pair_id', 'hand_id', 'table_id')

    # 1. how well each new column separates listed from unlisted INSIDE the top-20, by family
    cands = (d['oof_rank'] <= TOP).to_numpy()
    fam_col = d['behavior_family'].to_numpy()
    sep = {}
    for fam in FAMILIES:
        m = cands & (fam_col == fam)
        yy = y[m]
        row = {}
        for c in f2cols:
            x = d[c].to_numpy().astype(float)[m]
            row[c] = (round(float(roc_auc_score(yy, x)), 4)
                      if len(np.unique(yy)) > 1 and len(np.unique(x)) > 1 else None)
        sep[fam] = dict(sorted(row.items(),
                               key=lambda kv: -abs((kv[1] if kv[1] is not None else .5) - .5)))
        log('AUC top6 within top-20, ' + fam,
            json.dumps({k: v for k, v in list(sep[fam].items())[:6]}))

    # 2. the designs
    ref, r382 = struct + chrono + sc, struct + chrono + sc + rcols
    specs = {'ref': ref, 'ref_f2': ref + f2cols, 'r382': r382, 'r382_f2': r382 + f2cols,
             'f2_only': chrono + sc + f2cols}
    maps, sets = {}, {}
    for name, cols in specs.items():
        t1 = time.time()
        per_fam = {f: dict(cols=cols, params=CLF, seeds=S3, ranker=False) for f in FAMILIES}
        pred, _ = run(d, y, groups, per_fam, TOP)
        s, m = score(d, pred, TOP, truths, labels, key)
        maps[name], sets[name] = m, s
        log(name, json.dumps({'map5': round(s['map5'], 4),
                              'fam': {f[:2]: round(v, 4) for f, v in s['by_family'].items()},
                              'n_features': len(cols), 'sec': round(time.time() - t1)}))

    # 3. paired bootstraps, pooled and per family
    res = {'training_only': True, 'gate': GATE, 'queries': len(truths), 'n_f2': len(f2cols),
           'f2_columns': f2cols, 'auc_within_top20': sep, 'sets': sets, 'paired': {}}
    for a, b in [('ref', 'ref_f2'), ('r382', 'r382_f2'), ('ref', 'f2_only')]:
        bt = boot(maps[a], maps[b], ptab, pools)
        bt['by_family'] = {}
        for f in FAMILIES:
            fp = [p for p in maps[a] if labels[p] == f]
            fb = boot({p: maps[a][p] for p in fp}, {p: maps[b][p] for p in fp},
                      ptab, sorted({ptab[p] for p in fp}))
            bt['by_family'][f] = {'mean_delta': round(fb['mean_delta'], 4),
                                  'ci95': [round(v, 4) for v in fb['ci95']],
                                  'n_pairs': len(fp)}
        res['paired'][f'{b} - {a}'] = bt
        log(f'{b} - {a}', json.dumps({'mean': round(bt['mean_delta'], 4),
                                      'ci95': [round(v, 4) for v in bt['ci95']],
                                      'fam': {f[:2]: [v['mean_delta'], v['ci95']]
                                              for f, v in bt['by_family'].items()}}))

    # 4. the gate, evaluated mechanically
    verdict = {}
    for k, bt in res['paired'].items():
        passes = [f for f, v in bt['by_family'].items()
                  if v['mean_delta'] >= GATE['family_delta_ge']
                  and v['ci95'][0] > GATE['family_ci95_lower_gt']]
        floor_ok = all(v['mean_delta'] >= GATE['other_family_floor'] for v in bt['by_family'].values())
        verdict[k] = {'families_passing': passes, 'no_family_below_floor': floor_ok,
                      'gate_met': bool(passes) and floor_ok}
        log('GATE ' + k, json.dumps(verdict[k]))
    res['verdict'] = verdict
    res['seconds'] = round(time.time() - t0)
    write_json(OUT / 'audit-f2.json', res)
    check_protected()
    log('done', res['seconds'])
    print('DONE')


if __name__ == '__main__':
    main()
