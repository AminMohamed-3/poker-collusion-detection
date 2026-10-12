"""STEP216 features: the STEP208 partner-context strength block split by the HAND's roles instead of the pool's.

STEP208 orders the two members by pool direction (who receives chips over the pair's whole pool). Inside a
single hand the flow can run the other way: on the 3,240 audit rows, 74 % of hands have exactly one member
winning, and in 35 % of those it is the pool DONOR who wins. The chip-dump signature is per hand ("the member
who lost THIS hand called the partner's bet weakly"), so a hand-level order may place it better.

Hand roles: if exactly one member won the hand, loser/winner are those two; otherwise the pool roles are kept
(donor = loser). Implemented as a per-hand sign handed to STEP208's gated extractor
(sign_h = dt_sign, flipped where the pool donor won alone), then renamed.
Label-free: `won` is public hand data (production already uses won_sum/max/min).

Columns: dt_hlose_rs_<8 partner base>, dt_hwin_rs_<8 partner base>  (16).
Usage: python step216_C_dt_hand_role_strength.py features
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

CV = POKER / 'Claude Version'
DATA = POKER / 'data'
OUT = CV / 'outputs_step216_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT_ROLE = CV / 'outputs_step206_20260917/dt_roles_dev.parquet'
FEAT_RS = CV / 'outputs_step208_20260917/dt_role_strength_dev.parquet'
FEAT_H = OUT / 'dt_hand_role_strength_dev.parquet'
KEYS = ['pair_id', 'hand_id']


def hand_sign(roles):
    """roles: KEYS + dt_sign + dt_donor_won + dt_recip_won -> KEYS + dt_sign (hand-level)."""
    flip = (pl.col('dt_donor_won') == 1) & (pl.col('dt_recip_won') == 0)
    return roles.select(KEYS + [pl.when(flip).then(-pl.col('dt_sign')).otherwise(pl.col('dt_sign')).alias('dt_sign')])


def rename_map(bases):
    m = {}
    for b in bases:
        m['dt_donor_rs_' + b] = 'dt_hlose_rs_' + b
        m['dt_recip_rs_' + b] = 'dt_hwin_rs_' + b
    return m


def build(roles, pair_players, phase, out_path, check_rs=None):
    from evidence_aug import step208_C_dt_role_strength as S208
    RSF = S208._rsf()
    bases = [b for b in RSF.BASE if b.startswith('partner_')]
    tmp = out_path.with_name('_tmp_' + out_path.name)
    full, gate = S208.build(hand_sign(roles), pair_players, phase, tmp, check_rs=check_rs)
    tmp.unlink()
    rm = rename_map(bases)
    out = full.select(KEYS + list(rm)).rename(rm)
    out.write_parquet(out_path)
    return out, gate


def phase_features():
    from evidence_aug import step165_C_e2e_audit as S165
    from evidence_aug import step208_C_dt_role_strength as S208
    RSF = S208._rsf()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_H.exists(), 'preserve a built feature file'
    roles = pl.read_parquet(FEAT_ROLE).select(KEYS + ['dt_sign', 'dt_donor_won', 'dt_recip_won'])
    assert roles.height == 45129 and roles.select(KEYS).equals(pl.read_parquet(FEAT4).select(KEYS))
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    D = S165.load()
    check = D['Q'].select(KEYS + list(D['pf_cols']))
    assert sorted(D['pf_cols']) == sorted(RSF.FEATURES)
    out, gate = build(roles, pp, 'development', FEAT_H, check_rs=check)
    # consistency with STEP208: where the hand sign equals the pool sign the values must be identical
    hs = hand_sign(roles)
    same = roles.select(KEYS + ['dt_sign']).join(hs.rename({'dt_sign': 'h'}), on=KEYS).filter(pl.col('dt_sign') == pl.col('h'))
    s208 = pl.read_parquet(FEAT_RS)
    j = same.select(KEYS).join(s208, on=KEYS).join(out, on=KEYS)
    bases = [b for b in RSF.BASE if b.startswith('partner_')]
    cons = max(float(np.abs(j['dt_donor_rs_' + b].to_numpy() - j['dt_hlose_rs_' + b].to_numpy()).max()) for b in bases)
    flipped = roles.height - same.height
    print('rows with hand sign == pool sign', same.height, 'flipped', flipped, 'consistency max abs', cons, flush=True)
    assert cons == 0.0
    qk = D['Q'].select(KEYS).join(hs, on=KEYS).join(roles.select(KEYS + ['dt_sign']).rename({'dt_sign': 'p'}), on=KEYS)
    rec = {'rows': out.height, 'phase_asserted': 'development', 'gate_vs_production_rs_max_abs': max(gate.values()),
           'gate_rows': check.height, 'flipped_rows_pool': flipped,
           'flipped_rows_audit_top20': int((qk['dt_sign'] != qk['p']).sum()),
           'consistency_with_step208_max_abs': cons, 'columns': [c for c in out.columns if c not in KEYS],
           'positive_share': {c: round(float((out[c] > 0).mean()), 4) for c in out.columns if c not in KEYS},
           'sources': {str(p.relative_to(CV)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (FEAT_ROLE, FEAT_RS)},
           'sha256': hashlib.sha256(FEAT_H.read_bytes()).hexdigest()}
    (OUT / 'receipt_features216.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('columns', 'positive_share')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
