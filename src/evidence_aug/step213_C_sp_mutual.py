"""STEP213 features: MUTUAL (minimum over the two members) versions of production's per-member strength (rs_)
and event (v2_) accumulators, for soft_play.

Soft play is a two-sided behaviour: both members decline to contest each other. Production keeps each
per-member accumulator as sum and max over the two members; min = sum - max is implied but a depth-3 tree
cannot subtract two columns, so "BOTH checked strong hands facing each other" is not directly visible.
The per-member values already exist (STEP208 and STEP211 files, gated to reproduce production exactly);
min is symmetric, so the role order in those files does not matter.

GATE: on every audit row, min equals production's (sum - max) to 1e-6 (float32 sources).
Columns: sp_min_rs_<16 base>, sp_min_<28 events> (44).
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
OUT = CV / 'outputs_step213_20260917'
F_RS = CV / 'outputs_step208_20260917/dt_role_strength_dev.parquet'
F_V2 = CV / 'outputs_step211_20260917/dt_role_events_dev.parquet'
FEAT = OUT / 'sp_mutual_dev.parquet'
KEYS = ['pair_id', 'hand_id']


def derive(rs, v2):
    """rs: KEYS + dt_{donor,recip}_rs_<base>; v2: KEYS + dt_{donor,recip}_<event>. Returns KEYS + mins."""
    rb = [c[len('dt_donor_rs_'):] for c in rs.columns if c.startswith('dt_donor_rs_')]
    vb = [c[len('dt_donor_'):] for c in v2.columns if c.startswith('dt_donor_')]
    j = rs.join(v2, on=KEYS, how='inner', validate='1:1')
    assert j.height == rs.height == v2.height
    out = j.select(KEYS + [pl.min_horizontal('dt_donor_rs_' + b, 'dt_recip_rs_' + b).cast(pl.Float64).alias('sp_min_rs_' + b)
                           for b in rb]
                   + [pl.min_horizontal('dt_donor_' + e, 'dt_recip_' + e).cast(pl.Float64).alias('sp_min_' + e)
                      for e in vb])
    return out, rb, vb


def phase_features():
    from evidence_aug import step165_C_e2e_audit as S165
    OUT.mkdir(parents=True, exist_ok=True)
    assert not FEAT.exists()
    rs, v2 = pl.read_parquet(F_RS), pl.read_parquet(F_V2)
    out, rb, vb = derive(rs, v2)
    assert len(rb) == 16 and len(vb) == 28 and out.height == 45129
    D = S165.load()
    j = D['Q'].select(KEYS + ['rs_' + b + s for b in rb for s in ('_sum', '_max')]
                      + ['v2_' + e + s for e in vb for s in ('_sum', '_max')]).join(out, on=KEYS, validate='1:1')
    gate = {}
    for b in rb:
        gate['rs_' + b] = float(np.abs((j['rs_' + b + '_sum'] - j['rs_' + b + '_max']).to_numpy() - j['sp_min_rs_' + b].to_numpy()).max())
    for e in vb:
        gate['v2_' + e] = float(np.abs((j['v2_' + e + '_sum'] - j['v2_' + e + '_max']).to_numpy() - j['sp_min_' + e].to_numpy()).max())
    print('gate min vs production (sum - max), max abs:', max(gate.values()), 'rows', j.height, flush=True)
    assert max(gate.values()) < 1e-5
    out.write_parquet(FEAT)
    rec = {'rows': out.height, 'columns': out.drop(KEYS).columns, 'gate_rows': j.height,
           'gate_max_abs': max(gate.values()),
           'sources': {str(F_RS.relative_to(CV)): hashlib.sha256(F_RS.read_bytes()).hexdigest(),
                       str(F_V2.relative_to(CV)): hashlib.sha256(F_V2.read_bytes()).hexdigest()},
           'nonzero_share': {c: round(float((out[c] != 0).mean()), 4) for c in out.drop(KEYS).columns},
           'sha256': hashlib.sha256(FEAT.read_bytes()).hexdigest()}
    (OUT / 'receipt_features213.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('columns', 'nonzero_share')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'features':
        phase_features()
    else:
        raise SystemExit('usage: features')
