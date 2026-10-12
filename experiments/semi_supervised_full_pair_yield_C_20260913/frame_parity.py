"""Claude stop check, run after `screen.py prepare` and before `screen.py score`.

The full-corpus `full` and `v70` frames restricted to the old sample pairs must equal the frozen
Codex sample frames exactly: same pairs, same id columns, same 765 features (NaN-equal). Any
difference means corpus size leaked into the frames, and scoring must not run.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent
PARENT = Path('C:/Data/Projects/Money Making/work/poker/Codex Resume/experiments/semi_supervised_pair_yield_20260913')
IDS = ['pair_id', 'table_id', 'player_1', 'player_2', 'label', 'behavior_family']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

dest = OUT / 'old-frame-parity.json'
assert not dest.exists(), 'parity already recorded'
frozen = json.loads((OUT / 'frames-frozen.json').read_text(encoding='utf-8'))
old_frozen = json.loads((PARENT / 'frames-frozen.json').read_text(encoding='utf-8'))
features = frozen['features']
assert features == old_frozen['features'] and len(features) == 765
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'author': 'Claude', 'views': {}}
ok = True
for name in ('full', 'v70'):
    assert sha(OUT / f'{name}.parquet') == frozen['outputs'][name]['sha256']
    assert sha(PARENT / f'{name}.parquet') == old_frozen['outputs'][name]['sha256']
    new = pl.read_parquet(OUT / f'{name}.parquet')
    old = pl.read_parquet(PARENT / f'{name}.parquet').sort('pair_id')
    sub = new.join(old.select('pair_id'), on='pair_id', how='semi').sort('pair_id')
    same_rows = sub.height == old.height and sub['pair_id'].equals(old['pair_id'])
    same_order = sub.columns == old.columns
    ids_equal = same_rows and all(sub[c].cast(pl.Utf8).equals(old[c].cast(pl.Utf8)) for c in IDS)
    A = old.select(features).to_numpy().astype(np.float64)
    B = sub.select(features).to_numpy().astype(np.float64) if same_rows else None
    if same_rows:
        eq = np.isclose(A, B, rtol=0, atol=0, equal_nan=True)
        bad = [features[j] for j in np.where(~eq.all(axis=0))[0]]
        mad = float(np.nanmax(np.abs(A - B))) if A.size else 0.0
    else:
        bad, mad = ['ROW MISMATCH'], None
    exact = same_rows and same_order and ids_equal and not bad
    ok = ok and exact
    report['views'][name] = {'old_rows': old.height, 'matched_rows': sub.height, 'same_pairs': same_rows,
                             'same_column_order': same_order, 'ids_equal': ids_equal,
                             'unequal_feature_columns': bad[:50], 'n_unequal_feature_columns': len(bad),
                             'max_abs_difference': mad, 'exact': exact}
report['all_exact'] = ok
report['decision'] = 'score may run' if ok else 'STOP: do not run score; investigate frame drift first'
dest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report), flush=True)
sys.exit(0 if ok else 1)
