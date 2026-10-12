"""STEP206c: production's per-player hand columns re-expressed by the pair's ROLE (pool donor / recipient).

step65 hand_features gives A_chen, A_put_bb, A_folded, A_won, A_pos (and B_*), where A/B follow the pair
LIST, not the transfer. STEP206 supplies the direction; this supplies the same five per-player quantities for
the donor and the recipient, so directed_transfer stage-2 can ask "did the donor put chips in with a weak
hand and the recipient collect" without learning the pair's orientation from scratch.

Definitions are production's (step65): chen imported, pos = (seat_no - button_seat) % (max seat_no in the
hand + 1), put = total_contribution / big_blind, folded = seats.folded, won = won_share > 0.
Direction s is the STEP206 one: +1 when player_2 is the pool's net recipient. Phase is ASSERTED on every hand.

Columns: dt_donor_{chen,put_bb,folded,won,pos}, dt_recip_{chen,put_bb,folded,won,pos}, and dt_sign (kept in the
file for production use; not a stage-2 column).

features  development: keys = STEP194's 45,129 rows. GATES: s equals the sign implied by STEP206's
          dt_flow_along_bb on every row where the nets differ; on every audit row the role columns equal
          production's A_/B_ columns picked by s, exactly.
"""
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
OUT = CV / 'outputs_step206_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT_DIR = OUT / 'dt_direction_dev.parquet'
FEAT_ROLE = OUT / 'dt_roles_dev.parquet'
KEYS = ['pair_id', 'hand_id']
Q5 = ['chen', 'put_bb', 'folded', 'won', 'pos']
ROLE = ['dt_donor_' + q for q in Q5] + ['dt_recip_' + q for q in Q5]


def build_roles(pool_keys, pair_players, phase, out_path, keep_keys=None):
    """pool_keys: EVERY (pair, hand) of each pair's pool (the sign needs the whole pool).
    keep_keys: rows to write (default all)."""
    from evidence_aug.step65_structural_evidence import chen
    t0 = time.perf_counter()
    hids = pool_keys['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'phase', 'big_blind', 'button_seat').collect())
    assert hands.height == len(hids), 'hands missing'
    bad = hands.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    allseats = (pl.scan_parquet(DATA / 'seats.parquet').filter(pl.col('hand_id').is_in(hids))
                .select('hand_id', 'player_id', 'seat_no', 'hole_card_1', 'hole_card_2', 'net_chips',
                        'total_contribution', 'folded', 'won_share').collect())
    nseat = allseats.group_by('hand_id').agg((pl.col('seat_no').max() + 1).alias('nseat'))
    players = sorted({p for pr in pair_players.values() for p in pr})
    seats = allseats.filter(pl.col('player_id').is_in(players))
    pp = pl.DataFrame({'pair_id': list(pair_players), 'p1': [v[0] for v in pair_players.values()],
                       'p2': [v[1] for v in pair_players.values()]})
    x = (pool_keys.select(KEYS).join(pp, on='pair_id', how='left', validate='m:1')
         .join(hands.select('hand_id', 'big_blind', 'button_seat'), on='hand_id', how='left', validate='m:1')
         .join(seats.select('hand_id', 'player_id', 'net_chips').rename({'player_id': 'p1', 'net_chips': 'n1'}),
               on=['hand_id', 'p1'], how='left', validate='m:1')
         .join(seats.select('hand_id', 'player_id', 'net_chips').rename({'player_id': 'p2', 'net_chips': 'n2'}),
               on=['hand_id', 'p2'], how='left', validate='m:1'))
    assert x.select('n1', 'n2').null_count().to_numpy().sum() == 0, 'a pair member is not seated'
    x = x.with_columns(((pl.col('n2') / pl.col('big_blind') - pl.col('n1') / pl.col('big_blind')) / 2).alias('flow'))
    x = x.with_columns(pl.when(pl.col('flow').sum().over('pair_id') >= 0).then(1.0).otherwise(-1.0).alias('dt_sign'))
    if keep_keys is not None:
        x = keep_keys.select(KEYS).join(x, on=KEYS, how='left', validate='1:1', maintain_order='left')
        assert x['dt_sign'].null_count() == 0
    x = x.with_columns(pl.when(pl.col('dt_sign') > 0).then(pl.col('p1')).otherwise(pl.col('p2')).alias('donor'),
                       pl.when(pl.col('dt_sign') > 0).then(pl.col('p2')).otherwise(pl.col('p1')).alias('recip'))
    s = seats.join(nseat, on='hand_id').join(hands.select('hand_id', 'big_blind', 'button_seat'), on='hand_id')
    s = s.with_columns(
        pl.struct('hole_card_1', 'hole_card_2').map_elements(
            lambda r: chen(r['hole_card_1'], r['hole_card_2']), return_dtype=pl.Float64).alias('chen'),
        (pl.col('total_contribution') / pl.col('big_blind')).alias('put_bb'),
        pl.col('folded').cast(pl.Float64).alias('folded_f'),
        (pl.col('won_share') > 0).cast(pl.Float64).alias('won'),
        ((pl.col('seat_no') - pl.col('button_seat')) % pl.col('nseat')).cast(pl.Float64).alias('pos'))
    s = s.select('hand_id', 'player_id', 'chen', 'put_bb', pl.col('folded_f').alias('folded'), 'won', 'pos')
    for role, pref in (('donor', 'dt_donor_'), ('recip', 'dt_recip_')):
        x = x.join(s.rename({'player_id': role, **{q: pref + q for q in Q5}}), on=['hand_id', role],
                   how='left', validate='m:1')
    out = x.select(KEYS + ['dt_sign'] + ROLE)
    assert out.select(ROLE).null_count().to_numpy().sum() == 0
    assert np.isfinite(out.select(ROLE).to_numpy()).all()
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out


def phase_features():
    from evidence_aug import step165_C_e2e_audit as S165
    assert not FEAT_ROLE.exists(), 'preserve a built feature file'
    keys = pl.read_parquet(FEAT4).select(KEYS)
    assert keys.height == 45129
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    feat = build_roles(keys, pp, 'development', FEAT_ROLE)
    # gate 1: the sign agrees with STEP206's direction wherever this hand's nets differ
    dr = pl.read_parquet(FEAT_DIR).join(feat.select(KEYS + ['dt_sign']), on=KEYS, validate='1:1')
    raw = ((dr['B'] - dr['A']) / 2).to_numpy()
    nz = raw != 0
    implied = np.sign(dr['dt_flow_along_bb'].to_numpy()[nz] / raw[nz])
    g1 = int((implied != dr['dt_sign'].to_numpy()[nz]).sum())
    # gate 2: role columns are production's A_/B_ columns picked by the sign, on every audit row
    d = S165.load()['d']
    j = d.select(KEYS + ['A_' + q for q in Q5] + ['B_' + q for q in Q5]).join(feat, on=KEYS, how='left',
                                                                             validate='1:1')
    sp = j['dt_sign'].to_numpy() > 0
    g2 = {}
    for q in Q5:
        a, b = j['A_' + q].to_numpy(), j['B_' + q].to_numpy()
        g2['dt_donor_' + q] = float(np.abs(np.where(sp, a, b) - j['dt_donor_' + q].to_numpy()).max())
        g2['dt_recip_' + q] = float(np.abs(np.where(sp, b, a) - j['dt_recip_' + q].to_numpy()).max())
    print('gate sign disagreements', g1, 'rows checked', int(nz.sum()), flush=True)
    print('gate role vs production', json.dumps(g2), flush=True)
    assert g1 == 0, 'sign differs from STEP206 direction'
    assert max(g2.values()) < 1e-9, 'role columns differ from production A_/B_ columns'
    print(feat.select(ROLE).describe(), flush=True)
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_sign_disagreements': g1,
           'gate_sign_rows_checked': int(nz.sum()), 'gate_role_vs_production_max_abs': g2,
           'audit_rows_checked': j.height, 'donor_is_player_1_share': round(float((feat['dt_sign'] > 0).mean()), 4),
           'sha256': hashlib.sha256(FEAT_ROLE.read_bytes()).hexdigest()}
    (OUT / 'receipt_features206c.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
