"""STEP248 ADD-ON, stage 3: read the hands.

(1) The 26 group-Z pairs whose entire significant-cell set lies OUTSIDE the known-family half of
    the vocabulary -- after stage 2b these are the only pair-internal candidates left for a
    fourth family inside the STEP207 block.  Are their events coherent?
(2) Render and read >= 40 hands from the 20 Z pairs with the strongest cells (the registered
    fallback, since no event was Z-SPECIFIC), marking both members.
(3) Render 20 hands from group A (known-family control) for contrast.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step248_b_events import NAMES, NN, SCRATCH
from step248_c_common import OUT
from step248_e_hands import flagged_hands, render_hand
from step248_i_addon_analyse import PSIG, fire_matrix

N_Z_PAIRS = 20
HANDS_PER_Z = 3
N_A_PAIRS = 5
HANDS_PER_A = 4
SEED = 248251


def main():
    t0 = time.time()
    cells = pl.read_parquet(OUT / 'step248addon-cells.parquet')
    groups = pl.read_parquet(OUT / 'step248addon-groups.parquet')
    prof = json.loads((OUT / 'step248addon-profile.json').read_text(encoding='utf-8'))
    known = set(prof['known_set'])
    pure = set(prof['Z_pure_pairs'])
    sig = cells.filter(pl.col('pval') < PSIG)
    res = {}

    # ---- (1) the pure-OTHER Z pairs ---------------------------------------------------------
    ps = sig.filter(pl.col('pair_id').is_in(list(pure)))
    cnt = (ps.group_by('event_name').agg(pl.col('pair_id').n_unique().alias('pairs'))
           .sort('pairs', descending=True))
    print(f'--- the {len(pure)} Z pairs with NO known-family event: which events do they fire? ---')
    for r in cnt.head(18).iter_rows(named=True):
        aa = int(sig.filter((pl.col('grp') == 'A') &
                            (pl.col('event_name') == r['event_name']))['pair_id'].n_unique())
        hh = int(sig.filter((pl.col('grp') == 'H') &
                            (pl.col('event_name') == r['event_name']))['pair_id'].n_unique())
        print(f'  {r["event_name"]:>24} fired by {r["pairs"]:>3} of {len(pure)} pure-Z pairs   '
              f'(A {aa}/308, H {hh}/4000)')
    res['pure_events'] = cnt.head(18).to_dicts()
    top3 = cnt.head(3)['event_name'].to_list()
    cov = int(ps.filter(pl.col('event_name').is_in(top3))['pair_id'].n_unique())
    print(f'  top-3 event coverage of the pure set: {cov}/{len(pure)} = {cov/len(pure):.2f}')
    print(f'  distinct events fired by the pure set: {cnt.height} over '
          f'{int(ps["pair_id"].n_unique())} pairs, {ps.height} cells')
    res['pure_top3_coverage'] = cov / len(pure)
    res['pure_distinct_events'] = int(cnt.height)

    # ---- (2)+(3) pick the pairs and extract their hands --------------------------------------
    zp = pl.read_parquet(OUT / 'step248addon-Z-pairs.parquet').head(N_Z_PAIRS)
    rng = np.random.default_rng(SEED)
    ag = groups.filter(pl.col('grp') == 'A')
    ap = ag[np.sort(rng.choice(ag.height, N_A_PAIRS, replace=False))]
    key = groups.select('pair_id', 'tidx', 'a', 'b', 'grp', 'player_a', 'player_b', 'table_id',
                        'n_shared', 'risk_score', 'predicted_behavior', 'shares_hi99_player',
                        'evidence_hand_1', 'evidence_hand_2', 'evidence_hand_3',
                        'evidence_hand_4', 'evidence_hand_5')
    picked = pl.concat([zp.select('pair_id'), ap.select('pair_id')]).join(key, on='pair_id')
    want = defaultdict(set)
    ev = sig.join(picked.select('pair_id', 'tidx', 'a', 'b'), on='pair_id', how='inner')
    for r in ev.iter_rows(named=True):
        x, y = (r['a'], r['b']) if r['direction'] == 0 else (r['b'], r['a'])
        want[(r['tidx'], x, y)].add(r['event'])
    print(f'\nextracting hands for {len(want)} directed cells over {picked.height} pairs '
          f'[{time.time()-t0:.0f}s]', flush=True)
    fh = flagged_hands('evaluation', want)
    hd = pl.read_parquet(SCRATCH / 'evaluation' / 'hands.parquet')
    fh = fh.join(hd.select('tidx', 'hloc', 'hidx', 'hand_id'), on=['tidx', 'hloc'], how='left')
    fh = fh.with_columns(pl.Series('event_name', [NAMES[i] for i in fh['event'].to_list()]))
    fh.write_parquet(OUT / 'step248addon-flagged-hands.parquet')

    # ---- (b) do the mechanism hands coincide with the STEP207 evidence hands? ----------------
    print('\n--- Z pairs: overlap of the instrument\'s hands with the 5 STEP207 evidence hands ---')
    ov = []
    for r in picked.filter(pl.col('grp') == 'Z').iter_rows(named=True):
        sub = fh.filter((pl.col('tidx') == r['tidx']) &
                        (pl.col('A').is_in([r['a'], r['b']])) &
                        (pl.col('B').is_in([r['a'], r['b']])))
        mh = set(sub['hand_id'].to_list())
        evh = [r[f'evidence_hand_{i}'] for i in range(1, 6)]
        k = sum(1 for e in evh if e in mh)
        ranks = [i + 1 for i, e in enumerate(evh) if e in mh]
        ov.append({'pair_id': r['pair_id'], 'n_mech_hands': len(mh), 'overlap': k,
                   'ranks': ranks, 'behaviour': r['predicted_behavior']})
    ovf = pl.DataFrame(ov)
    print(f'  {ovf.height} Z pairs, median mechanism hands {ovf["n_mech_hands"].median():.0f}; '
          f'overlap counts: ' +
          ', '.join(f'{k}:{int((ovf["overlap"]==k).sum())}' for k in range(6)))
    print(f'  mean overlap {ovf["overlap"].mean():.2f} of 5; share with ZERO overlap '
          f'{(ovf["overlap"]==0).mean():.2f}')
    res['evidence_overlap'] = {'n_pairs': ovf.height,
                               'mean_overlap': float(ovf['overlap'].mean()),
                               'zero_share': float((ovf['overlap'] == 0).mean()),
                               'counts': {str(k): int((ovf['overlap'] == k).sum())
                                          for k in range(6)},
                               'median_mech_hands': float(ovf['n_mech_hands'].median())}
    ovf.write_parquet(OUT / 'step248addon-evidence-overlap.parquet')

    # ---- render -------------------------------------------------------------------------------
    txt = []
    nz = na = 0
    for r in picked.sort(['grp', 'pair_id']).iter_rows(named=True):
        sub = fh.filter((pl.col('tidx') == r['tidx']) & (pl.col('A').is_in([r['a'], r['b']])) &
                        (pl.col('B').is_in([r['a'], r['b']]))).sort('hloc')
        if not sub.height:
            continue
        cap = HANDS_PER_Z if r['grp'] == 'Z' else HANDS_PER_A
        evs = sorted(set(sub['event_name'].to_list()))
        txt.append('=' * 100)
        txt.append(f'GROUP {r["grp"]} {r["pair_id"]} table {r["table_id"]} '
                   f'players {r["player_a"]}/{r["player_b"]} (local {r["a"]},{r["b"]}) '
                   f'shared {r["n_shared"]} risk207 {r["risk_score"]:.4f} '
                   f'predicted {r["predicted_behavior"]} '
                   f'dominated {int(r["shares_hi99_player"])}')
        txt.append(f'  significant events: {evs}')
        txt.append(f'  mechanism hands: {sub["hand_id"].n_unique()}   '
                   f'STEP207 evidence hands: '
                   f'{[r[f"evidence_hand_{i}"] for i in range(1,6)]}')
        for hidx in sub['hidx'].unique(maintain_order=True).to_list()[:cap]:
            ee = sorted(set(sub.filter(pl.col('hidx') == hidx)['event_name'].to_list()))
            txt.append(f'  >>> events on this hand: {ee}')
            txt.append(render_hand('evaluation', int(hidx), mark={r['a'], r['b']}))
            txt.append('')
            if r['grp'] == 'Z':
                nz += 1
            else:
                na += 1
    (OUT / 'step248addon-hand-readings.txt').write_text('\n'.join(txt), encoding='utf-8')
    print(f'\nrendered {nz} Z hands and {na} A hands -> step248addon-hand-readings.txt '
          f'[{time.time()-t0:.0f}s]')
    res['rendered'] = {'Z_hands': nz, 'A_hands': na}
    (OUT / 'step248addon-hands.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
