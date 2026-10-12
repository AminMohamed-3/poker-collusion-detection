"""STEP206: give directed_transfer stage-2 the pair's DIRECTION.

Production's per-hand chip columns are symmetric or ordered by the pair list, never by role:
A_net_bb / B_net_bb follow player_1 / player_2, pair_net_bb sums them and pair_net_gap is an absolute value.
The only within-pair transform, `context_` (rank_evidence.context), subtracts the pair's own mean -- exactly
the quantity that says who is losing to whom. So stage-2 cannot tell a hand that moves chips ALONG the
pair's transfer from one that moves them AGAINST it, although a directed-transfer listing is by definition
about the former.

Direction is taken from the pair's own POOL (every shared hand of the pair within ONE phase, which is what
production's pool already is), held fixed across the 41 views like the network outputs. Only the pair
list's own player_1 / player_2 are used, within phase -- the same use production already makes for the
typed decoder. Phase is ASSERTED on every hand (OBS 19.158).

Columns (all per (pair, hand); `s` = +1 when player_2 is the pool's net recipient, else -1):
    dt_flow_along_bb        s * (B_net_bb - A_net_bb) / 2        chips moved along the pair's direction
    dt_donor_net_bb         the pool donor's net in this hand (bb)
    dt_recip_net_bb         the pool recipient's net in this hand (bb)
    dt_flow_share           dt_flow_along_bb / sum over the pool of |flow|
    dt_dir_margin           |sum flow| / sum |flow| over the pool   (pair-level: how one-sided the pair is)

Phases:
  features   development: keys = STEP194's 45,129 rows (the full development pools); A/B nets must equal
             production's A_net_bb / B_net_bb on every audit row.
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
FEAT_DEV = OUT / 'dt_direction_dev.parquet'
KEYS = ['pair_id', 'hand_id']
DIR = ['dt_flow_along_bb', 'dt_donor_net_bb', 'dt_recip_net_bb', 'dt_flow_share', 'dt_dir_margin']


def build_direction(pool_keys, pair_players, phase, out_path, keep_keys=None):
    """pool_keys: EVERY (pair, hand) of each pair's pool -- direction needs the whole pool.
    keep_keys: the rows to write (default: all pool rows)."""
    t0 = time.perf_counter()
    hids = pool_keys['hand_id'].unique().to_list()
    hands = (pl.scan_parquet(DATA / 'hands.parquet').filter(pl.col('hand_id').is_in(hids))
             .select('hand_id', 'phase', 'big_blind').collect())
    assert hands.height == len(hids), 'hands missing'
    bad = hands.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    players = sorted({p for pr in pair_players.values() for p in pr})
    seats = (pl.scan_parquet(DATA / 'seats.parquet')
             .filter(pl.col('hand_id').is_in(hids) & pl.col('player_id').is_in(players))
             .select('hand_id', 'player_id', 'net_chips').collect())
    pp = pl.DataFrame({'pair_id': list(pair_players), 'p1': [v[0] for v in pair_players.values()],
                       'p2': [v[1] for v in pair_players.values()]})
    x = (pool_keys.select(KEYS).join(pp, on='pair_id', how='left', validate='m:1')
         .join(hands.select('hand_id', 'big_blind'), on='hand_id', how='left', validate='m:1')
         .join(seats.rename({'player_id': 'p1', 'net_chips': 'n1'}), on=['hand_id', 'p1'], how='left', validate='m:1')
         .join(seats.rename({'player_id': 'p2', 'net_chips': 'n2'}), on=['hand_id', 'p2'], how='left', validate='m:1'))
    assert x.select('n1', 'n2', 'big_blind').null_count().to_numpy().sum() == 0, 'a pair member is not seated'
    x = x.with_columns((pl.col('n1') / pl.col('big_blind')).alias('A'), (pl.col('n2') / pl.col('big_blind')).alias('B'))
    x = x.with_columns(((pl.col('B') - pl.col('A')) / 2).alias('flow'))
    x = x.with_columns(pl.col('flow').sum().over('pair_id').alias('tot'),
                       pl.col('flow').abs().sum().over('pair_id').alias('absum'))
    x = x.with_columns(pl.when(pl.col('tot') >= 0).then(1.0).otherwise(-1.0).alias('s'))
    x = x.with_columns(
        (pl.col('s') * pl.col('flow')).alias('dt_flow_along_bb'),
        pl.when(pl.col('s') > 0).then(pl.col('A')).otherwise(pl.col('B')).alias('dt_donor_net_bb'),
        pl.when(pl.col('s') > 0).then(pl.col('B')).otherwise(pl.col('A')).alias('dt_recip_net_bb'),
        pl.when(pl.col('absum') > 0).then(pl.col('s') * pl.col('flow') / pl.col('absum')).otherwise(0.0)
        .alias('dt_flow_share'),
        pl.when(pl.col('absum') > 0).then(pl.col('tot').abs() / pl.col('absum')).otherwise(0.0)
        .alias('dt_dir_margin'))
    out = x.select(KEYS + ['A', 'B'] + DIR)
    if keep_keys is not None:
        out = keep_keys.select(KEYS).join(out, on=KEYS, how='left', validate='1:1', maintain_order='left')
        assert out.select(DIR).null_count().to_numpy().sum() == 0
    assert np.isfinite(out.select(DIR).to_numpy()).all()
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out


def phase_features():
    from evidence_aug import step165_C_e2e_audit as S165
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_DEV.exists(), 'preserve a built feature file'
    keys = pl.read_parquet(FEAT4).select(KEYS)
    assert keys.height == 45129
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    feat = build_direction(keys, pp, 'development', FEAT_DEV)
    # gate: my A/B nets are production's A_net_bb / B_net_bb on every audit row
    d = S165.load()['d'].select(KEYS + ['A_net_bb', 'B_net_bb'])
    j = d.join(feat, on=KEYS, how='left', validate='1:1')
    ga = float(np.abs(j['A'] - j['A_net_bb']).max())
    gb = float(np.abs(j['B'] - j['B_net_bb']).max())
    print('gate A/B nets vs production: max abs', ga, gb, flush=True)
    assert ga < 1e-9 and gb < 1e-9, 'net definition differs from production'
    print(feat.select(DIR).describe(), flush=True)
    rec = {'rows': feat.height, 'phase_asserted': 'development', 'gate_A_B_vs_production': [ga, gb],
           'sha256': hashlib.sha256(FEAT_DEV.read_bytes()).hexdigest()}
    (OUT / 'receipt_features206.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
