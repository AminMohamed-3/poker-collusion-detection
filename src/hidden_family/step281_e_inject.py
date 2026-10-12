"""STEP281 stage E: plant the mechanism, so "not found" can be turned into a number.

hf12's lesson was that a detection floor quoted without a calibrated positive control is
worthless -- its first sweep was ~1000x too weak and would have reported a detectable mechanism
as blind.  So the dose here is calibrated against the REAL coordinated_isolation family: the
planted pair has to reach the score the real family reaches before any exclusion is quoted.

The planted mechanism is a JOINT SQUEEZE, the victim-mediated pattern the disclosed families do
not contain: in k of the pair's shared hands both members enter and act aggressively, a third
player puts money in and folds facing that aggression, and the pot goes to the pair.  It is
planted directly in the statistics' own space, so it is the STRONGEST form the mechanism can
take -- the floor it gives is optimistic, and the exclusion it supports is therefore
conservative in the right direction.
"""
import json
import os

os.environ['POLARS_MAX_THREADS'] = '6'
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
OUT = ROOT / 'Claude Version' / 'experiments' / 'step281_victim_instrument_20260919'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step281_a_victim_stats import DATA, action_summary, hand_frame, table_stats  # noqa: E402
from step281_a2_cross_null import build_matrices  # noqa: E402
from step281_c_discover import CELLS, FAMILIES, Scorer, strat  # noqa: E402

DOSES = [1, 2, 3, 5, 8, 12]
N_TABLES = 80
POT_BB = 6.0          # chips moved per planted hand, in big blinds


def plant(M, a, b, k, rng):
    """Manufacture k joint-squeeze hands for the pair (a, b) inside one table."""
    W, V = M['W'], M['V']
    cand = np.where((W[:, a] > 0) & (W[:, b] > 0) & ((V[:, a] == 0) | (V[:, b] == 0)))[0]
    if len(cand) < k:
        return None
    hs = rng.choice(cand, k, replace=False)
    out = {key: (val.copy() if isinstance(val, np.ndarray) else val) for key, val in M.items()}
    for h in hs:
        others = [c for c in np.where(W[h] > 0)[0] if c not in (a, b)]
        if not others:
            continue
        c = int(rng.choice(others))
        out['V'][h, a] = out['V'][h, b] = 1.0
        out['V'][h, c] = 1.0
        out['n3'][h, a] = max(out['n3'][h, a], 1.0)
        out['n3'][h, b] = max(out['n3'][h, b], 1.0)
        out['ff'][h, c] = 1.0                       # the victim folds facing the aggression
        out['cfold'][h, c] = max(out['cfold'][h, c], POT_BB / 2)
        out['net'][h, a] += POT_BB / 2
        out['net'][h, b] += POT_BB / 2
        out['net'][h, c] -= POT_BB
    out['U'] = out['W'] * (1 - out['V'])
    return out


def main():
    dev_ref = pl.read_parquet(OUT / 'victim_development_trunc2000.parquet') \
                .filter(pl.col('n_shared') >= 38)
    lab = pl.read_csv(ROOT / 'data' / 'development_labels.csv').with_columns(
        pl.min_horizontal('player_1', 'player_2').alias('p1'),
        pl.max_horizontal('player_1', 'player_2').alias('p2'))
    fam = dev_ref.join(lab.select('p1', 'p2', 'behavior_family'), on=['p1', 'p2'], how='left')[
        'behavior_family'].fill_null('unlabelled').to_numpy()
    sc = Scorer(dev_ref, CELLS)
    base = sc(dev_ref)
    ranked = np.sort(base)

    def pct(v):
        return np.searchsorted(ranked, v, side='left') / len(ranked)

    real = {f: float(np.median(pct(base[fam == f]))) for f in FAMILIES}
    thr1000 = np.quantile(base, 1 - 1000 / 112_540)
    thr300 = np.quantile(base, 1 - 300 / 112_540)
    print('median score percentile of the REAL families:',
          {k: round(v, 4) for k, v in real.items()})

    hands = hand_frame('development', 2000)
    tables = sorted(hands['table_id'].unique().to_list())[:N_TABLES]
    hb = hands.filter(pl.col('table_id').is_in(tables))
    summ, _ = action_summary(hb.select('hand_id'))
    seats = (pl.scan_parquet(DATA / 'seats.parquet')
             .select('hand_id', 'player_id', 'net_chips', 'total_contribution', 'folded')
             .join(hb.select('hand_id', 'table_id', 'big_blind').lazy(), on='hand_id',
                   how='inner')
             .join(summ.lazy(), on=['hand_id', 'player_id'], how='left')
             .collect(engine='streaming')
             .with_columns(pl.col('vpip').fill_null(0), pl.col('n_aggr3').fill_null(0),
                           pl.col('fold_faced').fill_null(0)))

    rng = np.random.default_rng(281)
    rows = []
    mats = {t: build_matrices(seats, t) for t in tables}
    for dose in DOSES:
        pctl, hit1000, hit300 = [], 0, 0
        for t in tables:
            M = mats[t]
            P = len(M['players'])
            sh = M['W'].T @ M['W']
            iu = np.triu_indices(P, 1)
            ok = np.where(sh[iu] >= 38)[0]
            if not len(ok):
                continue
            pick = int(rng.choice(ok))
            a, b = int(iu[0][pick]), int(iu[1][pick])
            Mp = plant(M, a, b, dose, rng)
            if Mp is None:
                continue
            C = np.zeros((P, P))
            st = table_stats(Mp['V'], Mp['W'], Mp['net'], Mp['cfold'], Mp['ff'], Mp['n3'],
                             C, t, Mp['players'])
            row = st.filter((pl.col('p1') == M['players'][a])
                            & (pl.col('p2') == M['players'][b]))
            if not row.height:
                continue
            s = float(sc(row)[0])
            pctl.append(pct(s))
            hit1000 += s >= thr1000
            hit300 += s >= thr300
        rows.append({'dose_hands': dose, 'n': len(pctl),
                     'median_percentile': round(float(np.median(pctl)), 4),
                     'share_at_1000_budget': round(hit1000 / max(len(pctl), 1), 3),
                     'share_at_300_budget': round(hit300 / max(len(pctl), 1), 3)})
        print(rows[-1], flush=True)

    ci = real['coordinated_isolation']
    cal = [r for r in rows if r['median_percentile'] >= ci]
    res = {'real_family_median_percentile': real, 'injection': rows,
           'dose_matching_real_CI': cal[0]['dose_hands'] if cal else None,
           'detection_at_that_dose': cal[0] if cal else None}
    json.dump(res, open(OUT / 'inject281.json', 'w'), indent=2)
    print('\ndose reproducing the real CI family:',
          res['dose_matching_real_CI'], '->', res['detection_at_that_dose'])


if __name__ == '__main__':
    main()
