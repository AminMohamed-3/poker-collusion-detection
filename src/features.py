"""Gameplay-only pair/hand features, built in bounded batches of player pools."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
RANKS = {r: i + 2 for i, r in enumerate('23456789TJQKA')}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_identity(phase, batch_size, pools):
    source = 'development_labels.csv' if phase == 'development' else 'evaluation_pairs.csv'
    return {
        'phase': phase, 'batch_size': batch_size,
        'code_sha256': file_sha256(Path(__file__)),
        'polars_version': pl.__version__,
        'inputs': {name: file_sha256(DATA / name) for name in
                   ['hands.parquet', 'seats.parquet', 'actions.parquet', source]},
        'batches': {f'batch_{batch:03}.parquet': pools[start:start + batch_size]
                    for batch, start in enumerate(range(0, len(pools), batch_size))},
    }


def write_manifest(cache, manifest):
    temporary = cache / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(cache / 'manifest.json')


def verify_adoption(cache, identity, pairs):
    """Check cache structure; feature-value correctness needs independent review."""
    ids = []
    for name, tables in identity['batches'].items():
        frame = pl.read_parquet(cache / name, columns=['pair_id', 'hand_id', 'table_id'])
        if set(frame['table_id']) != set(tables):
            raise ValueError(f'Cannot adopt {name}: table membership differs from this batch layout')
        ids.append(frame)
    raw = pl.concat(ids)
    if raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() != raw.height:
        raise ValueError('Cannot adopt cache: duplicate pair-hand rows')
    actual = raw.select('pair_id', 'table_id').unique()
    expected = pairs.select('pair_id', 'table_id').unique()
    if (actual.join(expected, on=['pair_id', 'table_id'], how='anti').height or
            expected.join(actual, on=['pair_id', 'table_id'], how='anti').height):
        raise ValueError('Cannot adopt cache: pair/table coverage differs from source pairs')
    if identity['phase'] == 'evaluation':
        expected_counts = pl.read_csv(DATA / 'evaluation_pairs.csv').select('pair_id', 'shared_hands')
        counts = raw.group_by('pair_id').len().join(expected_counts, on='pair_id')
        if counts.filter(pl.col('len') != pl.col('shared_hands')).height:
            raise ValueError('Cannot adopt cache: evaluation shared-hand counts differ')
    return {'pair_hands': raw.height, 'pairs': raw['pair_id'].n_unique(),
            'tables': raw['table_id'].n_unique(), 'batch_layout_and_coverage_checked': True}


def prepare_cache(cache, identity, pairs, *, adopt=False, note=None, check=False):
    """Fail closed on untracked, stale, altered, or incompatible cache files."""
    path = cache / 'manifest.json'
    files = {p.name: p for p in cache.glob('*.parquet')}
    expected = set(identity['batches'])
    if set(files) - expected:
        raise ValueError('Unexpected Parquet batches in cache; move the cache aside and rebuild')
    if path.exists():
        if adopt:
            raise ValueError('Cache already has a manifest; adoption cannot override it')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('format_version') != 1 or manifest.get('identity') != identity:
            raise ValueError('Cache fingerprint mismatch (code, inputs, batch size, pools, or Polars); '
                             'restore matching inputs/code or move the cache aside and rebuild')
        if set(manifest['files']) != set(files):
            raise ValueError('Cache files differ from the manifest; refusing partial or untracked reuse')
        for name, fingerprint in manifest['files'].items():
            if file_sha256(files[name]) != fingerprint:
                raise ValueError(f'Cache content hash mismatch: {name}')
    else:
        if check:
            raise ValueError('Cache has no manifest; build a fresh cache or explicitly adopt verified files')
        if files and not adopt:
            raise ValueError('Existing cache has no manifest. After independent verification, use '
                             '--adopt-cache --adoption-note "describe the verification" once')
        if adopt and (not note or not note.strip()):
            raise ValueError('--adopt-cache requires a nonempty --adoption-note')
        if adopt and set(files) != expected:
            raise ValueError('Adoption requires the complete expected batch set')
        verification = verify_adoption(cache, identity, pairs) if adopt else None
        manifest = {
            'format_version': 1, 'identity': identity,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'provenance': {'mode': 'adopted_existing' if adopt else 'generated',
                           'note': note if adopt else None, 'verification': verification},
            'files': {name: file_sha256(file) for name, file in files.items()},
        }
        write_manifest(cache, manifest)
    if check and set(files) != expected:
        raise ValueError('Cache is compatible but incomplete; run the feature build to finish it')
    return manifest


def build(phase: str, batch_size: int = 20, limit_batches: int | None = None,
          *, adopt_cache=False, adoption_note=None, check_cache=False):
    if batch_size < 1 or (limit_batches is not None and limit_batches < 0):
        raise ValueError('Batch size must be positive and batch limit must be nonnegative')
    if (adopt_cache or check_cache) and limit_batches is not None:
        raise ValueError('Cache adoption/checking cannot be combined with --limit-batches')
    OUT.mkdir(exist_ok=True)
    cache = OUT / f'hands_{phase}'
    cache.mkdir(exist_ok=True)
    hands = pl.read_parquet(DATA / 'hands.parquet').filter(pl.col('phase') == phase)
    # Table IDs are only grouping/leakage-control keys, never features.
    table_map = (
        pl.scan_parquet(DATA / 'seats.parquet').select('hand_id', 'player_id')
        .join(hands.lazy().select('hand_id', 'table_id'), on='hand_id')
        .select('player_id', 'table_id').unique().collect(engine='streaming')
    )
    source = 'development_labels.csv' if phase == 'development' else 'evaluation_pairs.csv'
    pairs = pl.read_csv(DATA / source).select('pair_id', 'player_1', 'player_2')
    pairs = pairs.join(table_map.rename({'player_id': 'player_1'}), on='player_1')
    pools = sorted(pairs['table_id'].unique().to_list())
    identity = cache_identity(phase, batch_size, pools)
    manifest = prepare_cache(cache, identity, pairs, adopt=adopt_cache,
                             note=adoption_note, check=check_cache)
    if adopt_cache or check_cache:
        print(f'{phase}: {"adopted" if adopt_cache else "verified"} {len(manifest["files"])} '
              'cached batches; no feature or summary files changed', flush=True)
        return
    for batch, start in enumerate(range(0, len(pools), batch_size)):
        if limit_batches is not None and batch >= limit_batches:
            break
        dest = cache / f'batch_{batch:03}.parquet'
        if dest.exists():
            print(f'{phase} batch {batch}: cached', flush=True)
            continue
        these = pools[start:start + batch_size]
        h = hands.filter(pl.col('table_id').is_in(these))
        p = pairs.filter(pl.col('table_id').is_in(these))
        s = pl.scan_parquet(DATA / 'seats.parquet').join(
            h.lazy().select('hand_id', 'big_blind', 'final_pot', 'players_at_showdown'),
            on='hand_id', how='inner').collect(engine='streaming')
        a = pl.scan_parquet(DATA / 'actions.parquet').join(
            h.lazy().select('hand_id', 'big_blind'), on='hand_id', how='inner'
        ).collect(engine='streaming').sort('hand_id', 'action_no')
        a = a.with_columns(
            (pl.col('action').is_in(['bet', 'raise', 'all_in']) &
             (pl.col('amount') > pl.col('to_call'))).alias('aggressive'),
            (pl.col('amount') / pl.col('big_blind')).alias('amount_bb'),
            (pl.col('to_call') / (pl.col('pot_before') + pl.col('to_call')).clip(lower_bound=1)).alias('pot_odds'),
        ).with_columns(
            pl.when('aggressive').then(pl.col('player_id')).otherwise(None)
            .forward_fill().shift(1).over('hand_id', 'street').alias('previous_aggressor')
        )
        action_stats = a.group_by('hand_id', 'player_id').agg(
            pl.len().alias('n_actions'),
            pl.col('aggressive').sum().alias('n_aggressive'),
            (pl.col('action') == 'check').sum().alias('n_check'),
            (pl.col('action') == 'call').sum().alias('n_call'),
            ((pl.col('action') == 'check') & (pl.col('street') != 'preflop')).sum().alias('late_checks'),
            pl.when('aggressive').then('amount_bb').otherwise(0).sum().alias('agg_bb'),
            pl.when(pl.col('action').is_in(['call','all_in']) & ~pl.col('aggressive')).then('amount_bb').otherwise(0).sum().alias('call_bb'),
            pl.when(pl.col('action') == 'fold').then('pot_odds').otherwise(0).max().alias('fold_pot_odds'),
            pl.col('players_active').min().alias('min_active'),
        )
        interactions = a.filter(pl.col('previous_aggressor').is_not_null()).group_by(
            'hand_id', 'player_id', 'previous_aggressor').agg(
                (pl.col('action') == 'fold').sum().alias('opp_fold'),
                pl.col('aggressive').sum().alias('opp_raise'),
                (~pl.col('aggressive') & pl.col('action').is_in(['call','all_in'])).sum().alias('opp_call'),
                pl.when(~pl.col('aggressive')).then('amount_bb').otherwise(0).sum().alias('opp_call_bb'),
                pl.when(pl.col('action') == 'fold').then('pot_odds').otherwise(0).max().alias('opp_fold_odds'),
        )
        s = s.join(action_stats, on=['hand_id', 'player_id'], how='left').with_columns(
            (pl.col('total_contribution') / pl.col('big_blind')).alias('contribution_bb'),
            (pl.col('net_chips') / pl.col('big_blind')).alias('net_bb'),
            (pl.col('starting_stack') / pl.col('big_blind')).alias('stack_bb'),
            pl.col('hole_card_1').str.slice(0, 1).replace_strict(RANKS).cast(pl.Float64).alias('rank1'),
            pl.col('hole_card_2').str.slice(0, 1).replace_strict(RANKS).cast(pl.Float64).alias('rank2'),
            (pl.col('hole_card_1').str.slice(1) == pl.col('hole_card_2').str.slice(1)).cast(pl.Float64).alias('suited'),
        ).with_columns(
            ((pl.max_horizontal('rank1', 'rank2') + pl.min_horizontal('rank1', 'rank2') * .5) / 21 +
             (pl.col('rank1') == pl.col('rank2')).cast(pl.Float64) * .7 + pl.col('suited') * .1).alias('hole_strength')
        )
        player_base = s.group_by('player_id').agg(
            pl.col('n_aggressive').mean().alias('baseline_aggressive'),
            pl.col('n_check').mean().alias('baseline_check'),
            pl.col('contribution_bb').mean().alias('baseline_contribution'),
            pl.col('net_bb').mean().alias('baseline_net'),
            pl.col('folded').mean().alias('baseline_fold'),
        )
        s = s.join(player_base, on='player_id').with_columns(
            (pl.col('n_aggressive') - pl.col('baseline_aggressive')).alias('aggression_residual'),
            (pl.col('n_check') - pl.col('baseline_check')).alias('check_residual'),
            (pl.col('contribution_bb') - pl.col('baseline_contribution')).alias('contribution_residual'),
            (pl.col('net_bb') - pl.col('baseline_net')).alias('net_residual'),
        )
        columns = ['contribution_bb', 'net_bb', 'stack_bb', 'folded', 'went_to_showdown',
                   'won_share', 'n_actions', 'n_aggressive', 'n_check', 'n_call', 'late_checks',
                   'agg_bb', 'call_bb', 'fold_pot_odds', 'min_active', 'hole_strength',
                   'aggression_residual', 'check_residual', 'contribution_residual', 'net_residual']
        s1 = s.select('hand_id', 'player_id', *columns).rename({'player_id': 'player_1', **{c:c+'_1' for c in columns}})
        s2 = s.select('hand_id', 'player_id', *columns).rename({'player_id': 'player_2', **{c:c+'_2' for c in columns}})
        joined = p.join(s1, on='player_1').join(s2, on=['player_2', 'hand_id']).join(
            h.select('hand_id', 'big_blind', 'final_pot', 'players_at_showdown'), on='hand_id')
        oppcols = ['opp_fold','opp_raise','opp_call','opp_call_bb','opp_fold_odds']
        for side, other in [(1,2), (2,1)]:
            inter = interactions.rename({'player_id': f'player_{side}', 'previous_aggressor': f'player_{other}',
                                         **{c: c+f'_{side}' for c in oppcols}})
            joined = joined.join(inter, on=['hand_id','player_1','player_2'], how='left')
        joined = joined.with_columns(pl.col(oppcols[0]+'_1').fill_null(0))
        symmetric = []
        for c in columns + oppcols:
            c1, c2 = pl.col(c+'_1').cast(pl.Float64).fill_null(0), pl.col(c+'_2').cast(pl.Float64).fill_null(0)
            symmetric += [(c1+c2).alias(c+'_sum'), (c1-c2).abs().alias(c+'_gap'),
                          pl.min_horizontal(c1,c2).alias(c+'_min'), pl.max_horizontal(c1,c2).alias(c+'_max')]
        joined = joined.with_columns(symmetric).with_columns(
            (pl.col('final_pot')/pl.col('big_blind')).alias('pot_bb'),
            ((pl.col('net_bb_1')>0) & (pl.col('net_bb_2')<0) | (pl.col('net_bb_2')>0) & (pl.col('net_bb_1')<0)).cast(pl.Float64).alias('opposite_outcomes'),
            (pl.col('folded_1') & pl.col('folded_2')).cast(pl.Float64).alias('both_fold'),
            ((pl.col('won_share_1')+pl.col('won_share_2'))).alias('joint_win_share'),
            (pl.col('hole_strength_1') * pl.col('opp_fold_1').fill_null(0) +
             pl.col('hole_strength_2') * pl.col('opp_fold_2').fill_null(0)).alias('strong_fold_to_partner'),
            (pl.col('hole_strength_1') * pl.col('late_checks_1') +
             pl.col('hole_strength_2') * pl.col('late_checks_2')).alias('strong_late_checks'),
        )
        names = [e.meta.output_name() for e in symmetric]
        extra = ['pot_bb','players_at_showdown','opposite_outcomes','both_fold','joint_win_share','strong_fold_to_partner','strong_late_checks']
        result = joined.select('pair_id','hand_id','table_id', *names, *extra)
        result = result.with_columns(pl.exclude('pair_id','hand_id','table_id').cast(pl.Float32).fill_null(0))
        temporary = dest.with_suffix('.parquet.tmp')
        result.write_parquet(temporary)
        temporary.replace(dest)
        manifest['files'][dest.name] = file_sha256(dest)
        write_manifest(cache, manifest)
        print(f'{phase} batch {batch}: {result.height} pair-hands, {result.width-3} features', flush=True)
        del a, s, joined, result, s1, s2, action_stats, interactions
        gc.collect()
    if limit_batches is None:
        raw = pl.scan_parquet([cache / name for name in identity['batches']])
        values = [c for c in raw.collect_schema().names() if c not in ['pair_id','hand_id','table_id']]
        aggs = [pl.len().alias('shared_hands')]
        for c in values:
            aggs += [pl.col(c).mean().alias(c+'_mean'), pl.col(c).std().alias(c+'_std'),
                     pl.col(c).max().alias(c+'_max'), pl.col(c).quantile(.95).alias(c+'_q95')]
        summary = raw.group_by('pair_id','table_id').agg(aggs).collect(engine='streaming')
        summary.write_parquet(OUT/f'pairs_{phase}.parquet')
        print(f'{phase}: wrote {summary.height} pair summaries', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development','evaluation'])
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--limit-batches', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--adopt-cache', action='store_true', help='Record a verified complete legacy cache once; do not rebuild')
    mode.add_argument('--check-cache', action='store_true', help='Verify a complete cache without changing its files')
    parser.add_argument('--adoption-note', help='Independent verification record required with --adopt-cache')
    args = parser.parse_args()
    if args.adoption_note and not args.adopt_cache:
        parser.error('--adoption-note requires --adopt-cache')
    try:
        build(args.phase, args.batch_size, args.limit_batches, adopt_cache=args.adopt_cache,
              adoption_note=args.adoption_note, check_cache=args.check_cache)
    except ValueError as exc:
        parser.exit(2, f'{exc}\n')
