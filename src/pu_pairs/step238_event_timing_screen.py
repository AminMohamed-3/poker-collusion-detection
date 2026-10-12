"""STEP238 -- timing of injected hands inside a colluding pair's shared history (development phase only).

Questions: are listed events spaced like a constant-rate (geometric) process, or in bursts / on a schedule?
Do sharp unlisted events (p_fmax above the honest 99.9 % quantile) sit next to listed ones?
Is there an onset (first event late in the history)?  Output: experiments/step238_event_timing_20260917/timing.json
"""
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
OUT = CV / 'experiments' / 'step238_event_timing_20260917'
OUT.mkdir(parents=True, exist_ok=True)
SHARP = 0.4662

h = pl.read_parquet(CV / 'experiments' / 'step232_event_detector_20260917' / 'hand_oof.parquet')
hands = pl.read_parquet(ROOT / 'data' / 'hands.parquet', columns=['hand_id', 'started_at', 'phase'])
h = h.join(hands, on='hand_id', how='left')
assert h['phase'].null_count() == 0 and set(h['phase'].unique().to_list()) == {'development'}
ev = pl.read_csv(ROOT / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id', 'evidence_rank')
h = h.join(ev, on=['pair_id', 'hand_id'], how='left')
h = h.sort('pair_id', 'started_at').with_columns(
    idx=pl.int_range(pl.len()).over('pair_id'), n=pl.len().over('pair_id'))
pos = h.filter(pl.col('label') == 1)
res = {}
rng = np.random.default_rng(238)
for fam in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
    P = pos.filter(pl.col('behavior_family') == fam)
    gaps_same, first_frac, last_frac, sharp_near, sharp_null, n_sharp_before_first = [], [], [], [], [], []
    between_sharp, between_total = 0, 0
    cv_list = []
    for pid, g in P.group_by('pair_id'):
        g = g.sort('idx')
        n = g['n'][0]
        L = g.filter(pl.col('listed') == 1).sort('evidence_rank')
        li = L['idx'].to_numpy()
        if len(li) < 2:
            continue
        # blocks: a descent in time inside the list marks the start of block 2
        cut = next((k for k in range(1, len(li)) if li[k] < li[k - 1]), len(li))
        b1, b2 = li[:cut], li[cut:]
        for b in (b1, b2):
            if len(b) >= 2:
                gaps_same.extend(np.diff(b).tolist())
        first_frac.append(li.min() / n)
        last_frac.append(li.max() / n)
        if len(b1) >= 3:
            d = np.diff(b1)
            cv_list.append(float(d.std() / d.mean()))
        S = g.filter((pl.col('listed') == 0) & (pl.col('p_fmax') > SHARP))['idx'].to_numpy()
        n_sharp_before_first.append(int((S < li.min()).sum()))
        for s in S:
            sharp_near.append(int(np.abs(li - s).min()))
            sharp_null.append(int(np.abs(li - rng.integers(0, n)).min()))
        # unlisted hands strictly inside block 1's span: by the listing rule none is a first-type event
        if len(b1) >= 2:
            inside = g.filter((pl.col('idx') > b1.min()) & (pl.col('idx') < b1.max()) & (pl.col('listed') == 0))
            between_total += inside.height
            between_sharp += int((inside['p_fmax'] > SHARP).sum())
    gs = np.array(gaps_same)
    res[fam] = {
        'pairs': P['pair_id'].n_unique(),
        'same_block_gap': {'n': int(len(gs)), 'mean': float(gs.mean()), 'median': float(np.median(gs)),
                           'share_gap_1': float((gs == 1).mean()), 'share_gap_le_3': float((gs <= 3).mean()),
                           'geometric_share_gap_1_at_same_mean': float(1 / gs.mean()),
                           'geometric_share_le_3': float(1 - (1 - 1 / gs.mean()) ** 3),
                           'quantiles_10_50_90': [float(x) for x in np.quantile(gs, [.1, .5, .9])]},
        'block1_gap_cv_mean (geometric ~ 1, regular ~ 0, bursty > 1)': float(np.mean(cv_list)) if cv_list else None,
        'first_listed_position_frac': {'mean': float(np.mean(first_frac)), 'q10_50_90': [float(x) for x in np.quantile(first_frac, [.1, .5, .9])]},
        'last_listed_position_frac': {'mean': float(np.mean(last_frac)), 'q10_50_90': [float(x) for x in np.quantile(last_frac, [.1, .5, .9])]},
        'sharp_unlisted_before_first_listed_per_pair': float(np.mean(n_sharp_before_first)),
        'sharp_unlisted_distance_to_nearest_listed': {'median': float(np.median(sharp_near)) if sharp_near else None,
                                                      'null_median (random hand)': float(np.median(sharp_null)) if sharp_null else None,
                                                      'share_within_2': float(np.mean(np.array(sharp_near) <= 2)) if sharp_near else None,
                                                      'null_share_within_2': float(np.mean(np.array(sharp_null) <= 2)) if sharp_null else None},
        'unlisted_inside_block1_span': {'hands': between_total, 'sharp': between_sharp,
                                        'sharp_rate': between_sharp / max(between_total, 1)},
    }
# wall-clock: are events tied to sessions?  gap in minutes between consecutive shared hands, listed vs not
pos2 = pos.sort('pair_id', 'started_at').with_columns(
    dt_prev=(pl.col('started_at') - pl.col('started_at').shift(1).over('pair_id')))
try:
    pos2 = pos2.with_columns(dtm=pl.col('dt_prev').dt.total_seconds() / 60)
except Exception:
    pos2 = pos2.with_columns(dtm=pl.col('dt_prev').cast(pl.Float64))
res['minutes_since_previous_shared_hand'] = {
    'listed_median': pos2.filter(pl.col('listed') == 1)['dtm'].median(),
    'unlisted_median': pos2.filter(pl.col('listed') == 0)['dtm'].median(),
    'sharp_unlisted_median': pos2.filter((pl.col('listed') == 0) & (pl.col('p_fmax') > SHARP))['dtm'].median()}
(OUT / 'timing.json').write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
print(json.dumps(res, indent=1, default=str))
