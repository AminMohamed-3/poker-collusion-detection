"""Extract development action references in bounded existing pool batches."""
import os
os.environ['POLARS_MAX_THREADS'] = '4'
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import polars as pl
from conditional_action_features import INPUTS, extract_hand, action_class, card_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/conditional-action-experiment'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True); cache = OUT / 'action-cache'; cache.mkdir(exist_ok=True)
    plan_path = OUT / 'extraction-plan.json'
    identity = {'sources': {p: sha(ROOT / p) for p in ['src/conditional_action_features.py', 'src/build_conditional_action_cache.py',
        'data/hands.parquet', 'data/seats.parquet', 'data/actions.parquet', 'data/development_labels.csv']},
        'inputs': INPUTS, 'scope': 'Development gameplay only; public/own-card input, private partner strength is a regression response, and actual action is a classification response.',
        'style': 'Each player action distribution from all other development hands. No pair labels enter style calculation. This describes observed batch behavior rather than online forecasting.'}
    if plan_path.exists(): assert json.loads(plan_path.read_text())['identity'] == identity
    else: plan_path.write_text(json.dumps({'created_at': datetime.now(timezone.utc).isoformat(), 'identity': identity}, indent=2) + '\n')
    hands = pl.read_parquet(ROOT / 'data/hands.parquet').filter(pl.col('phase') == 'development').select(
        'hand_id', 'table_id', 'board_cards', 'button_seat', 'big_blind')
    phase_ids = hands.select('hand_id')
    actions = pl.scan_parquet(ROOT / 'data/actions.parquet').join(phase_ids.lazy(), on='hand_id', how='semi')
    cls = pl.when(pl.col('action') == 'fold').then(0).when(pl.col('action') == 'check').then(1).when(
        (pl.col('action') == 'call') | ((pl.col('action') == 'all_in') & (pl.col('amount') <= pl.col('to_call')))).then(2).otherwise(3)
    totals = actions.with_columns(cls.alias('kind')).group_by('player_id').agg(*[(pl.col('kind') == k).sum().alias('n' + str(k)) for k in range(4)]).collect()
    totals.write_parquet(OUT / 'public-action-style-counts.parquet')
    count_lookup = {r[0]: np.array(r[1:], dtype=np.int64) for r in totals.iter_rows()}
    pairs = pl.read_csv(ROOT / 'data/development_labels.csv').select('pair_id', 'player_1', 'player_2')
    for raw_path in sorted((ROOT / 'outputs_v2/hands_development').glob('batch_*.parquet')):
        dest = cache / raw_path.name; receipt = dest.with_suffix('.json')
        if receipt.exists():
            assert json.loads(receipt.read_text())['sha256'] == sha(dest)
            continue
        start = time.perf_counter()
        keys = pl.read_parquet(raw_path, columns=['pair_id', 'hand_id']).join(pairs, on='pair_id', validate='m:1')
        ids = keys.select('hand_id').unique()
        h = hands.join(ids, on='hand_id', how='semi')
        s = pl.scan_parquet(ROOT / 'data/seats.parquet').join(ids.lazy(), on='hand_id', how='semi').collect()
        a = actions.join(ids.lazy(), on='hand_id', how='semi').collect().sort('hand_id', 'action_no')
        hand_pairs = defaultdict(list); hand_seats = defaultdict(list); hand_actions = defaultdict(list)
        for pair, hand, first, second in keys.iter_rows(): hand_pairs[hand].append((pair, first, second))
        for row in s.iter_rows(named=True): hand_seats[row['hand_id']].append(row)
        for row in a.iter_rows(named=True): hand_actions[row['hand_id']].append(row)
        rows = []
        for hand in h.iter_rows(named=True):
            hand_id = hand['hand_id']; observed = hand_actions[hand_id]
            current_counts = defaultdict(lambda: np.zeros(4, dtype=np.int64))
            for action in observed: current_counts[action['player_id']][action_class(action)] += 1
            styles = {}
            for seat in hand_seats[hand_id]:
                player = seat['player_id']; counts = count_lookup.get(player, np.zeros(4, dtype=np.int64)) - current_counts[player]
                assert counts.min() >= 0
                styles[player] = ((counts + 1) / (counts.sum() + 4)).tolist()
            rows.extend(extract_hand(hand, hand_seats[hand_id], observed, hand_pairs[hand_id], styles))
        frame = pl.DataFrame(rows).with_columns(pl.col(INPUTS + ['partner_strength', 'observed_log_amount_bb']).cast(pl.Float32))
        assert frame.select(pl.struct('pair_id', 'direction', 'hand_id', 'action_no').n_unique()).item() == frame.height
        frame.write_parquet(dest)
        report = {'sha256': sha(dest), 'rows': frame.height, 'pairs': frame['pair_id'].n_unique(), 'seconds': time.perf_counter()-start}
        receipt.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps({'batch': dest.name, **report}), flush=True)
        card_features.cache_clear()


if __name__ == '__main__': main()
