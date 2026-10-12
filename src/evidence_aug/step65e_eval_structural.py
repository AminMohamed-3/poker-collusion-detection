"""Step 65e: structural relational-action features (step 65 `hand_features`) for every hand of the 16,060 selected
evaluation pairs (the pairs whose evidence lists the submissions carry), batch by batch, label-free.
Output: Claude Version/experiments/family_stage2_20260911/eval_structural/batch_XXX.parquet + receipt json."""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import ORIGINAL, CV, sha, write_json
from evidence_aug.step65_structural_evidence import build
import polars as pl

OUT = CV / 'experiments' / 'family_stage2_20260911' / 'eval_structural'


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pl.read_parquet(ORIGINAL / 'reports' / 'replay-v13' / 'selected-pairs.parquet').select('pair_id')
    players = pl.read_csv(ORIGINAL / 'data' / 'evaluation_pairs.csv').select('pair_id', 'player_1', 'player_2').join(selected, on='pair_id', how='semi')
    assert players.height == 16060
    files = {}; total = 0
    for path in sorted((ORIGINAL / 'outputs_v2' / 'hands_evaluation').glob('batch_*.parquet')):
        dest = OUT / (path.stem + '.parquet')
        keys = pl.read_parquet(path, columns=['pair_id', 'hand_id']).join(selected, on='pair_id', how='semi')
        if dest.exists():
            files[dest.name] = sha(dest); total += keys.height; continue
        feat = build(keys, players)
        assert feat.height == keys.height
        feat.write_parquet(dest); files[dest.name] = sha(dest); total += feat.height
        print(json.dumps({'batch': path.stem, 'hands': feat.height, 'seconds': round(time.time() - t0)}), flush=True)
    write_json(OUT / 'receipt.json', {'files': files, 'selected_hands': total, 'selected_pairs': 16060, 'seconds': round(time.time() - t0)})


if __name__ == '__main__':
    main()
