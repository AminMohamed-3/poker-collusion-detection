"""STEP205: complete the causal re-anchoring that STEP194 started, and ask whether the causal preflop
block helps directed_transfer and soft_play stage-2 as well as coordinated_isolation.

Source of the idea: production's own code, not the STEP193 packets (OBS 19.158 -- those packets contain
evaluation-phase hands and are not used for anything new). `step65_structural_evidence.hand_features`
anchors EVERY preflop response feature on `opener_ix`, the first aggressor of any kind:

    pf_others_raise                 outsider raises after the opener -- outsider-opened hands count raises
                                    against the OUTSIDER, exactly the pf_others_fold defect STEP194 fixed
    pf_partner_{fold,call,raise,none}  defined only when a pair member OPENS; when an outsider opens and a
                                    pair member three-bets, the partner's response is reported as 'none'
    pf_partner_folded_before_open   same anchor
    pf_open_size_bb                 the size of whoever opened, outsider or not

The development-only re-check of STEP194 (clean_rederivation_devphase_only.json) shows the anchor matters:
the two anchors disagree on 40.5 % of coordinated-isolation hands. NEW2 re-anchors the rest of the
block on the PAIR's first preflop aggression, `ix`:

    pf_others_raise_after_pair_agg        outsider aggressive actions after ix
    pf_partner_{fold,call,raise,silent}_after_pair_agg   the OTHER member's first action after ix
    pf_partner_folded_before_pair_agg     the other member folded before ix
    pf_pair_agg_to_bb, pf_pair_agg_to_pot the size of the pair's aggression
    pf_outsiders_live_at_pair_agg         outsiders still in the hand when the pair aggresses
    pf_others_fold_share_after_pair_agg   after-folds / outsiders live at ix

When the pair never aggresses preflop every NEW2 column is 0 (STEP194's flag pf_pair_aggresses_preflop
separates that case).

Phases:
  features   build NEW2 for the 45,129 development (pair, hand) rows -- keys taken from STEP194's parquet --
             asserting phase == 'development' on every hand. No labels are read.
  run REGSHA the registered audit (see registration205.json).

Audit only. No evaluation pair, no held fold, never uploads.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
OUT = CV / 'outputs_step205_20260917'
REG = OUT / 'registration205.json'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT2 = OUT / 'causal_completion_dev.parquet'
KEYS = ['pair_id', 'hand_id']
NEW4 = ['pf_others_fold_after_pair_agg', 'pf_others_fold_before_pair_agg',
        'pf_pair_aggresses_preflop', 'pf_others_call_after_pair_agg']
NEW2 = ['pf_others_raise_after_pair_agg',
        'pf_partner_fold_after_pair_agg', 'pf_partner_call_after_pair_agg',
        'pf_partner_raise_after_pair_agg', 'pf_partner_silent_after_pair_agg',
        'pf_partner_folded_before_pair_agg',
        'pf_pair_agg_to_bb', 'pf_pair_agg_to_pot',
        'pf_outsiders_live_at_pair_agg', 'pf_others_fold_share_after_pair_agg']


def agg(action, amount, to_call):
    """Identical to step194's definition (and step65's)."""
    return action in ('raise', 'bet') or (action == 'all_in' and (amount or 0) > (to_call or 0))


def completion_features(acts, p1, p2, bb):
    """acts: ordered (street, player_id, action, amount, amount_to, to_call, players_active, pot_before)
    for ONE hand's preflop. Anchor: the pair's first preflop aggression, as in step194."""
    pf = [a for a in acts if a[0] == 'preflop']
    pair = (p1, p2)
    f = dict.fromkeys(NEW2, 0.0)
    ix = None
    for i, a in enumerate(pf):
        if a[1] in pair and agg(a[2], a[3], a[5]):
            ix = i
            break
    if ix is None:
        return f
    actor = pf[ix][1]
    partner = p2 if actor == p1 else p1
    partner_folded_before = any(a[1] == partner and a[2] == 'fold' for a in pf[:ix])
    f['pf_partner_folded_before_pair_agg'] = float(partner_folded_before)
    resp = None
    raises = folds = 0
    for a in pf[ix + 1:]:
        if a[1] == partner and resp is None:
            resp = ('raise' if agg(a[2], a[3], a[5])
                    else 'fold' if a[2] == 'fold'
                    else 'call' if a[2] in ('call', 'all_in', 'check') else 'other')
        if a[1] not in pair:
            if agg(a[2], a[3], a[5]):
                raises += 1
            elif a[2] == 'fold':
                folds += 1
    f['pf_others_raise_after_pair_agg'] = float(raises)
    f['pf_partner_fold_after_pair_agg'] = float(resp == 'fold')
    f['pf_partner_call_after_pair_agg'] = float(resp == 'call')
    f['pf_partner_raise_after_pair_agg'] = float(resp == 'raise')
    f['pf_partner_silent_after_pair_agg'] = float(resp is None)
    f['pf_pair_agg_to_bb'] = float((pf[ix][4] or 0) / bb)
    pot = pf[ix][7] or 0
    f['pf_pair_agg_to_pot'] = float((pf[ix][4] or 0) / pot) if pot > 0 else 0.0
    # players_active counts everyone still in before this action, the actor included
    live_out = max(int(pf[ix][6] or 0) - 1 - (0 if partner_folded_before else 1), 0)
    f['pf_outsiders_live_at_pair_agg'] = float(live_out)
    f['pf_others_fold_share_after_pair_agg'] = float(folds / live_out) if live_out > 0 else 0.0
    return f


def build_features(keys, pair_players, phase, out_path):
    """Shared by the development build here and any evaluation build later. `phase` is ASSERTED on every
    hand -- OBS 19.158: a docstring is not a check."""
    t0 = time.perf_counter()
    hids = keys['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'phase', 'big_blind').collect())
    assert hands.height == len(hids), 'hands missing'
    bad = hands.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    bb = dict(hands.select('hand_id', 'big_blind').iter_rows())
    acts = (pl.scan_parquet(DATA / 'actions.parquet')
            .filter(pl.col('hand_id').is_in(hids) & (pl.col('street') == 'preflop'))
            .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'amount_to',
                    'to_call', 'players_active', 'pot_before')
            .sort('hand_id', 'action_no').collect())
    print(f'{keys.height} rows, {len(hids)} {phase} hands, {acts.height} preflop actions  '
          f'{round(time.perf_counter() - t0)}s', flush=True)
    arows = {}
    for hid, _, st, pid, a, amt, amt_to, tc, pa, pot in acts.iter_rows():
        arows.setdefault(hid, []).append((st, pid, a, amt, amt_to, tc, pa, pot))
    rows = []
    for pid, hid in keys.select(KEYS).iter_rows():
        f = completion_features(arows.get(hid, []), *pair_players[pid], bb[hid])
        f['pair_id'], f['hand_id'] = pid, hid
        rows.append(f)
    feat = pl.DataFrame(rows).select(KEYS + NEW2)
    assert feat.height == keys.height
    assert np.isfinite(feat.select(NEW2).to_numpy()).all(), 'non-finite feature'
    feat.write_parquet(out_path)
    print(f'wrote {out_path.name} {feat.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return feat


def phase_features():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT2.exists(), 'preserve a built feature file'
    keys = pl.read_parquet(FEAT4).select(KEYS)
    assert keys.height == 45129
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    feat = build_features(keys, pp, 'development', FEAT2)
    # consistency with STEP194: both anchor on the same ix, so 'no pair aggression' must agree
    f4 = pl.read_parquet(FEAT4).join(feat, on=KEYS, validate='1:1')
    noagg = f4['pf_pair_aggresses_preflop'].to_numpy() == 0
    assert (f4.select(NEW2).to_numpy()[noagg] == 0).all(), 'NEW2 non-zero where the pair never aggresses'
    assert ((f4['pf_partner_fold_after_pair_agg'] + f4['pf_partner_call_after_pair_agg']
             + f4['pf_partner_raise_after_pair_agg'] + f4['pf_partner_silent_after_pair_agg'])
            .to_numpy()[~noagg] <= 1).all()
    print(feat.select(NEW2).describe(), flush=True)
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'sha256': hashlib.sha256(
        FEAT2.read_bytes()).hexdigest(), 'pair_aggresses_share': round(float(1 - noagg.mean()), 4)}
    (OUT / 'receipt_features205.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    elif sys.argv[1] == 'run':
        from evidence_aug.step205b_C_causal_completion_audit import main as run_main
        run_main(sys.argv[2])
    else:
        raise SystemExit('usage: features | run REGSHA')
