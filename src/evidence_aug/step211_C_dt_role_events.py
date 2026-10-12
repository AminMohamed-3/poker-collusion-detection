"""STEP211 features: production's v2 opponent-aware EVENT block (src/advanced_features.py, the 56 `v2_*_sum` /
`v2_*_max` per-hand columns inside the ranker block) split by the pair's pool role, as STEP208 did for the
strength block.

Production computes 28 per-actor event sums (`directional_events`, once with player_1 acting and once with
player_2) and then keeps only their sum and max. This calls the SAME production functions
(strength_table, directional_events -- imported, not retyped) on the requested hands, and orders the two
per-actor results by STEP206's pool direction. GATE: donor + recipient and max(donor, recipient) reproduce
production's v2_*_sum and v2_*_max EXACTLY on every audit row. Phase is ASSERTED on every hand.

Columns: dt_donor_<event>, dt_recip_<event> for the 28 events (56).
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
OUT = CV / 'outputs_step211_20260917'
FEAT4 = CV / 'outputs_step194_20260916/causal_fold_features.parquet'
FEAT_ROLE = CV / 'outputs_step206_20260917/dt_roles_dev.parquet'
FEAT_EV = OUT / 'dt_role_events_dev.parquet'
KEYS = ['pair_id', 'hand_id']


def per_actor(keys_players):
    """keys_players: pair_id, hand_id, player_1, player_2. Returns (first, second): per-actor v2 event sums
    with player_1 / player_2 acting, exactly as advanced_features.build computes them."""
    import advanced_features as AF
    ids = keys_players.select('hand_id').unique()
    h = pl.scan_parquet(DATA / 'hands.parquet').join(ids.lazy(), on='hand_id').collect()
    s = pl.scan_parquet(DATA / 'seats.parquet').join(ids.lazy(), on='hand_id').collect()
    a = pl.scan_parquet(DATA / 'actions.parquet').join(ids.lazy(), on='hand_id').collect().sort('hand_id', 'action_no')
    a = a.join(h.select('hand_id', 'big_blind'), on='hand_id').with_columns(
        (pl.col('action').is_in(['bet', 'raise', 'all_in']) &
         (pl.col('amount') > pl.col('to_call'))).alias('aggressive'))
    a = a.with_columns(pl.when('aggressive').then(pl.col('player_id')).otherwise(None)
                       .forward_fill().shift(1).over('hand_id', 'street').alias('previous_aggressor'))
    folds = a.filter(pl.col('action') == 'fold').group_by('hand_id', 'player_id').agg(
        pl.col('action_no').min().alias('fold_no'))
    si = AF.strength_table(s, h).join(s.select('hand_id', 'player_id', 'net_chips'), on=['hand_id', 'player_id'])
    si = si.join(folds, on=['hand_id', 'player_id'], how='left').with_columns(pl.col('fold_no').fill_null(1000000))
    first = AF.directional_events(keys_players, a, si, 'player_1', 'player_2')
    second = AF.directional_events(keys_players, a, si, 'player_2', 'player_1')
    return h, first, second


def build(keys_sign, pair_players, phase, out_path, check_v2=None):
    t0 = time.perf_counter()
    pp = pl.DataFrame({'pair_id': list(pair_players), 'player_1': [v[0] for v in pair_players.values()],
                       'player_2': [v[1] for v in pair_players.values()]})
    kp = keys_sign.select(KEYS).join(pp, on='pair_id', how='left', validate='m:1')
    assert kp['player_1'].null_count() == 0
    h, first, second = per_actor(kp)
    ph = h.select('hand_id', 'phase')
    assert ph.height == keys_sign['hand_id'].n_unique(), 'hands missing'
    bad = ph.filter(pl.col('phase') != phase).height
    assert bad == 0, f'{bad} hands are not {phase}-phase'
    ev = [c for c in first.columns if c.startswith('v2_')]
    assert len(ev) == 28
    j = (keys_sign.select(KEYS + ['dt_sign'])
         .join(first, on=KEYS, how='left', validate='1:1')
         .join(second, on=KEYS, how='left', validate='1:1', suffix='_p2'))
    j = j.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in ev] +
                       [pl.col(c + '_p2').fill_null(0).cast(pl.Float32) for c in ev])
    pos = pl.col('dt_sign') > 0
    out = j.select(KEYS + [pl.when(pos).then(pl.col(c)).otherwise(pl.col(c + '_p2')).alias('dt_donor_' + c[3:])
                           for c in ev]
                   + [pl.when(pos).then(pl.col(c + '_p2')).otherwise(pl.col(c)).alias('dt_recip_' + c[3:])
                      for c in ev])
    assert out.height == keys_sign.height
    assert np.isfinite(out.drop(KEYS).to_numpy()).all()
    gate = None
    if check_v2 is not None:
        g = check_v2.join(j, on=KEYS, how='inner', validate='1:1')
        assert g.height == check_v2.height
        gate = {}
        for c in ev:
            x1, x2 = g[c].to_numpy(), g[c + '_p2'].to_numpy()
            gate[c + '_sum'] = float(np.abs((x1 + x2) - g[c + '_sum'].to_numpy()).max())
            gate[c + '_max'] = float(np.abs(np.maximum(x1, x2) - g[c + '_max'].to_numpy()).max())
        print('gate vs production v2_ (max abs):', max(gate.values()), 'rows', g.height, flush=True)
        assert max(gate.values()) == 0.0, 'role split does not reproduce production v2_ columns'
    out.write_parquet(out_path)
    print(f'wrote {out_path.name} {out.shape}  {round(time.perf_counter() - t0)}s', flush=True)
    return out, gate


def phase_features():
    from evidence_aug import step165_C_e2e_audit as S165
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT_EV.exists(), 'preserve a built feature file'
    ks = pl.read_parquet(FEAT_ROLE).select(KEYS + ['dt_sign'])
    assert ks.height == 45129
    assert ks.select(KEYS).equals(pl.read_parquet(FEAT4).select(KEYS))
    lab = pl.read_csv(DATA / 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
    pp = {r['pair_id']: (r['player_1'], r['player_2']) for r in lab.iter_rows(named=True)}
    D = S165.load()
    v2 = [c for c in D['rcols'] if c.startswith('v2_')]
    assert len(v2) == 56
    check = D['Q'].select(KEYS + v2)
    feat, gate = build(ks, pp, 'development', FEAT_EV, check_v2=check)
    rec = {'rows': feat.height, 'columns': feat.drop(KEYS).columns, 'phase_asserted': 'development',
           'gate_rows': check.height, 'gate_vs_production_v2_max_abs': max(gate.values()),
           'sha256': hashlib.sha256(FEAT_EV.read_bytes()).hexdigest()}
    (OUT / 'receipt_features211.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k != 'columns'}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
