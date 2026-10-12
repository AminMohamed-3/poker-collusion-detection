"""Verify accelerated extraction against every saved scalar development row."""
from __future__ import annotations
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import json
import time
from pathlib import Path
import numpy as np
import polars as pl
from sequence_evidence_fast import extract
from sequence_evidence_features import NAMES
from features import file_sha256 as sha

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/sequence-evidence-experiment'


def main():
    started = time.perf_counter()
    destination = REPORT/'fast-verification.json'
    if destination.exists():
        raise FileExistsError('Preserve completed verification')
    scalar = pl.read_parquet(REPORT/'development.parquet').sort('pair_id','hand_id')
    labels = pl.read_csv(ROOT/'data/development_labels.csv')
    keys = scalar.select('pair_id','hand_id').join(labels.select('pair_id','player_1','player_2'), on='pair_id')
    ids = keys.select('hand_id').unique().lazy()
    hands = pl.scan_parquet(ROOT/'data/hands.parquet').join(ids,on='hand_id').collect()
    seats = pl.scan_parquet(ROOT/'data/seats.parquet').join(ids,on='hand_id').collect()
    actions = pl.scan_parquet(ROOT/'data/actions.parquet').join(ids,on='hand_id').collect()
    before = time.perf_counter()
    fast = extract(keys, hands, seats, actions).sort('pair_id','hand_id')
    calculation_seconds = time.perf_counter()-before
    assert fast.select('pair_id','hand_id').equals(scalar.select('pair_id','hand_id'))
    names = ['seq_'+n for n in NAMES]
    a, b = scalar.select(names).to_numpy(), fast.select(names).to_numpy()
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
    # Verify a real subset with pair order reversed and source rows shuffled.
    subset = keys.head(1000).with_columns(pl.col('player_2').alias('player_1'), pl.col('player_1').alias('player_2'))
    selected = subset.select('hand_id').unique()
    reversed_result = extract(subset.reverse(), hands.join(selected,on='hand_id').reverse(), seats.join(selected,on='hand_id').reverse(), actions.join(selected,on='hand_id').reverse()).sort('pair_id','hand_id')
    expected = fast.join(subset.select('pair_id','hand_id'),on=['pair_id','hand_id']).sort('pair_id','hand_id')
    np.testing.assert_allclose(expected.select(names).to_numpy(), reversed_result.select(names).to_numpy(), rtol=2e-6, atol=2e-6)
    source_paths = [Path(__file__), ROOT/'src/sequence_evidence_fast.py', ROOT/'src/sequence_evidence_features.py', REPORT/'development.parquet']
    result = {'passed': True, 'pair_hands': scalar.height, 'features': len(names), 'cells': a.size,
        'different_float32_cells': int(np.sum(a != b)), 'max_absolute_difference': float(np.max(np.abs(a-b))),
        'source_row_and_pair_order_invariance': True, 'extraction_seconds': calculation_seconds,
        'seconds': time.perf_counter()-started, 'sha256': {str(p): sha(p) for p in source_paths}}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
