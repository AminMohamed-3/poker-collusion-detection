"""STEP237 step 1-2: sample invisible / visible listed hands and matched honest hands, render them.

Groups per family: 40 invisible listed (p_fmax < .0852 = honest 99% quantile), 40 visible listed
(p_fmax > .4662 = honest 99.9% quantile), 80 honest hands of confirmed non-target pairs matched
one-for-one on a coarse type key: (players_dealt, last street reached, both members acted on the
flop, a member folded facing the partner as last aggressor). Development phase is asserted.
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '1')

import numpy as np
import polars as pl

from step237_render import HandBook

ROOT = r'C:\Data\Projects\Money Making\work\poker'
DATA = ROOT + r'\data'
OOF = ROOT + r'\Claude Version\experiments\step232_event_detector_20260917\hand_oof.parquet'
OUT = ROOT + r'\Claude Version\experiments\step237_invisible_events_20260917'
INVIS, VIS = .0852, .4662
SIDX = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}


def action_frame(hand_ids: pl.DataFrame) -> pl.DataFrame:
    a = (pl.scan_parquet(DATA + r'\actions.parquet').join(hand_ids.lazy(), on='hand_id')
         .collect().sort('hand_id', 'action_no'))
    a = a.with_columns(
        (pl.col('action').is_in(['bet', 'raise', 'all_in']) & (pl.col('amount') > pl.col('to_call')))
        .alias('aggressive'))
    a = a.with_columns(pl.when('aggressive').then(pl.col('player_id')).otherwise(None)
                       .forward_fill().shift(1).over('hand_id', 'street').alias('prev_agg'))
    return a


def coarse_keys(ph: pl.DataFrame, a: pl.DataFrame) -> pl.DataFrame:
    """ph: pair_id, hand_id, player_1, player_2. Returns the coarse type key per pair-hand."""
    last = a.group_by('hand_id').agg(
        pl.col('street').replace_strict(SIDX, return_dtype=pl.Int64).max().alias('last_street'))
    flop = (a.filter(pl.col('street') == 'flop').select('hand_id', 'player_id').unique()
            .with_columns(pl.lit(1).alias('on_flop')))
    fold_p = (a.filter((pl.col('action') == 'fold') & pl.col('prev_agg').is_not_null())
              .select('hand_id', 'player_id', 'prev_agg').unique()
              .with_columns(pl.lit(1).alias('fold_to')))
    out = ph
    for side, other in ((1, 2), (2, 1)):
        out = out.join(flop.rename({'player_id': f'player_{side}', 'on_flop': f'flop_{side}'}),
                       on=['hand_id', f'player_{side}'], how='left')
        out = out.join(fold_p.rename({'player_id': f'player_{side}', 'prev_agg': f'player_{other}',
                                      'fold_to': f'foldp_{side}'}),
                       on=['hand_id', 'player_1', 'player_2'], how='left')
    out = out.join(last, on='hand_id', how='left')
    hands = pl.read_parquet(DATA + r'\hands.parquet', columns=['hand_id', 'players_dealt', 'phase'])
    out = out.join(hands, on='hand_id', how='left')
    assert (out['phase'] == 'development').all(), 'non-development hand in the pair-hand table'
    return out.with_columns(
        pl.col('flop_1', 'flop_2', 'foldp_1', 'foldp_2').fill_null(0),
    ).with_columns(
        (pl.col('flop_1') * pl.col('flop_2')).alias('both_flop'),
        pl.max_horizontal('foldp_1', 'foldp_2').alias('fold_to_partner'),
    ).with_columns(
        (pl.col('players_dealt').cast(pl.Utf8) + '|' + pl.col('last_street').cast(pl.Utf8) + '|'
         + pl.col('both_flop').cast(pl.Utf8) + '|' + pl.col('fold_to_partner').cast(pl.Utf8))
        .alias('ckey'))


def main() -> None:
    oof = pl.read_parquet(OOF)
    lab = pl.read_csv(DATA + r'\development_labels.csv').select('pair_id', 'player_1', 'player_2')
    ph = oof.join(lab, on='pair_id')
    a = action_frame(ph.select('hand_id').unique())
    keys = coarse_keys(ph, a)
    keys.write_parquet(OUT + r'\coarse_keys.parquet')
    rng = np.random.default_rng(237)
    honest = keys.filter(pl.col('label') == 0)
    hon_by_key: dict[str, list[int]] = {}
    for i, k in enumerate(honest['ckey'].to_list()):
        hon_by_key.setdefault(k, []).append(i)
    for v in hon_by_key.values():
        rng.shuffle(v)
    cursor: dict[str, int] = {}
    # role resolution: A = member losing more chips over the pair's shared hands
    seats = (pl.scan_parquet(DATA + r'\seats.parquet')
             .join(ph.select('hand_id').unique().lazy(), on='hand_id')
             .select('hand_id', 'player_id', 'net_chips').collect())
    nets = {}
    for side in (1, 2):
        j = (ph.select('pair_id', 'hand_id', pl.col(f'player_{side}').alias('player_id'))
             .join(seats, on=['hand_id', 'player_id'], how='left')
             .group_by('pair_id', 'player_id').agg(pl.col('net_chips').sum()))
        for r in j.iter_rows(named=True):
            nets.setdefault(r['pair_id'], {})[r['player_id']] = r['net_chips'] or 0
    role = {}
    for r in lab.iter_rows(named=True):
        n = nets.get(r['pair_id'], {})
        p1, p2 = r['player_1'], r['player_2']
        role[r['pair_id']] = (p1, p2) if (n.get(p1, 0), p1) <= (n.get(p2, 0), p2) else (p2, p1)

    picks = []
    for fam in ('soft_play', 'coordinated_isolation', 'directed_transfer'):
        for grp, cond in (('invisible', pl.col('p_fmax') < INVIS), ('visible', pl.col('p_fmax') > VIS)):
            pool = keys.filter((pl.col('listed') == 1) & (pl.col('behavior_family') == fam) & cond)
            idx = rng.permutation(pool.height)[:40]
            sub = pool[np.sort(idx)]
            for r in sub.iter_rows(named=True):
                picks.append({'family': fam, 'group': grp, 'pair_id': r['pair_id'],
                              'hand_id': r['hand_id'], 'p_fmax': r['p_fmax'], 'ckey': r['ckey']})
                pool_idx = hon_by_key.get(r['ckey'], [])
                c = cursor.get(r['ckey'], 0)
                if c < len(pool_idx):
                    cursor[r['ckey']] = c + 1
                    hr = honest[pool_idx[c]].to_dicts()[0]
                    picks.append({'family': fam, 'group': 'honest', 'pair_id': hr['pair_id'],
                                  'hand_id': hr['hand_id'], 'p_fmax': hr['p_fmax'],
                                  'ckey': hr['ckey']})
    pk = pl.DataFrame(picks)
    pk.write_csv(OUT + r'\sample_manifest.csv')
    print(pk.group_by('family', 'group').len().sort('family', 'group'))
    book = HandBook(pk['hand_id'].unique().to_list())
    for fam in pk['family'].unique().to_list():
        for grp in ('invisible', 'visible', 'honest'):
            sub = pk.filter((pl.col('family') == fam) & (pl.col('group') == grp))
            if not sub.height:
                continue
            blocks = []
            for i, r in enumerate(sub.iter_rows(named=True)):
                A, B = role[r['pair_id']]
                head = (f'### {fam} / {grp} #{i + 1}  p_fmax={r["p_fmax"]:.4f}  ckey={r["ckey"]}'
                        f'  (A = chip-losing member)')
                blocks.append(book.render(r['hand_id'], A, B, head))
            path = OUT + f'\\hands_{fam}_{grp}.txt'
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(blocks) + '\n')
            print('wrote', path, sub.height)


if __name__ == '__main__':
    main()
