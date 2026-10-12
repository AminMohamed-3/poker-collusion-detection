"""STEP230 (Claude): within-pair percentile ranks of the strong ADDED stage-2 blocks, computed over each pair's
top-20 candidate rows (the rows stage 2 scores).

Why: step 78 (OBS 19.3) found within-pair relative encoding beat plain columns in all four cells tested, and the
production r382 block carries within-pair z-scores for the stage-1 features -- but every block added since
STEP204 (causal preflop columns, DT role strength, DT outsider cards) enters stage 2 raw.

Encoding (one definition for audit and production): for column c and pair p with n top-20 rows,
    wr_<c> = (rank_average(c within p) - 1) / (n - 1),   0.5 when n == 1.
Blocks:
    DT: the 16 DTs16 columns (STEP208) + the 24 DTod columns (STEP221/222)   -> 40 wr_ columns
    CI: the 4 causal (STEP194) + 10 completion (STEP205) preflop columns     -> 14 wr_ columns
Usage: python step230_C_within_pair_rank.py audit    (writes outputs_step230_20260917/wr_blocks_audit.parquet)
Development-phase audit rows only (asserted). No labels are read by this file.
"""
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
sys.dont_write_bytecode = True

CV = POKER / 'Claude Version'
KEYS = ['pair_id', 'hand_id']
OUT = CV / 'outputs_step230_20260917'
DT_SRC = {'outputs_step208_20260917/dt_role_strength_dev.parquet': 16,
          'outputs_step222_20260917/outsider_linked_dev.parquet': 24}
CI_SRC = {'outputs_step194_20260916/causal_fold_features.parquet': 4,
          'outputs_step205_20260917/causal_completion_dev.parquet': 10}
PREFIX = 'wr_'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rank_block(frame, cols):
    """frame: exactly the top-20 rows of each pair (KEYS + cols). Returns KEYS + wr_ columns, same row order."""
    assert frame.select(cols).null_count().sum_horizontal().item() == 0, 'nulls in a ranked column'
    n = pl.len().over('pair_id')
    ex = [pl.when(n > 1).then((pl.col(c).rank('average').over('pair_id') - 1) / (n - 1))
          .otherwise(0.5).cast(pl.Float64).alias(PREFIX + c) for c in cols]
    return frame.select(KEYS + ex)


REG228 = CV / 'outputs_step228_20260917/registration228.json'
OD_CODEX = '../Codex Resume/research/outsider_decisions_20260912/linked.parquet'


def source_columns(src):
    """The exact column lists the STEP228 reference uses (the dev files carry further columns)."""
    ff = json.loads(REG228.read_text(encoding='utf-8'))['feature_files']
    cols = []
    for rel, k in src.items():
        c = ff[OD_CODEX] if rel == 'outputs_step222_20260917/outsider_linked_dev.parquet' else ff[rel]
        assert len(c) == k and set(c) <= set(pl.read_parquet_schema(CV / rel)), rel
        cols.append((rel, list(c)))
    return cols


def audit():
    from evidence_aug import step165_C_e2e_audit as S165
    D = S165.load()
    Q = D['Q'].select(KEYS)
    sizes = Q.group_by('pair_id').len()['len']
    assert Q.height == 3240 and sizes.min() == 20 and sizes.max() == 20, (Q.height, sizes.min(), sizes.max())
    ph = (pl.scan_parquet(POKER / 'data/hands.parquet').select('hand_id', 'phase').collect()
          .join(Q.select('hand_id').unique(), on='hand_id'))
    assert ph['phase'].unique().to_list() == ['development']
    f = Q
    blocks = {}
    for fam, src in (('directed_transfer', DT_SRC), ('coordinated_isolation', CI_SRC)):
        names = []
        for rel, cols in source_columns(src):
            k0 = f.select(KEYS)
            f = f.join(pl.read_parquet(CV / rel).select(KEYS + cols), on=KEYS, how='left', validate='1:1',
                       maintain_order='left')
            assert f.select(KEYS).equals(k0)
            names += cols
        blocks[fam] = names
    assert len(blocks['directed_transfer']) == 40 and len(blocks['coordinated_isolation']) == 14
    allc = blocks['directed_transfer'] + blocks['coordinated_isolation']
    assert len(set(allc)) == len(allc)
    r = rank_block(f.select(KEYS + allc), allc)
    assert r.select(KEYS).equals(Q)
    wr = [c for c in r.columns if c not in KEYS]
    assert r.select(wr).null_count().sum_horizontal().item() == 0
    OUT.mkdir(exist_ok=True)
    dest = OUT / 'wr_blocks_audit.parquet'
    assert not dest.exists()
    r.write_parquet(dest)
    rec = {'file': str(dest.relative_to(CV)), 'sha256': sha(dest), 'rows': r.height,
           'columns': {fam: [PREFIX + c for c in v] for fam, v in blocks.items()},
           'sources_sha256': {rel: sha(CV / rel) for rel in list(DT_SRC) + list(CI_SRC)},
           'builder_sha256': sha(Path(__file__)),
           'distinct_values_per_column_min': int(min(r[c].n_unique() for c in wr)),
           'constant_columns': [c for c in wr if r[c].n_unique() == 1],
           'note': 'audit rows = D[Q] of step165 load (162 development pairs x 20 top rows); DTod values from the '
                   'STEP222 port, identical to Codex linked.parquet on these rows (receipt_features222.json)'}
    (OUT / 'wr_blocks_audit_receipt.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps({k: rec[k] for k in ('sha256', 'rows', 'distinct_values_per_column_min', 'constant_columns')}))


if __name__ == '__main__':
    assert sys.argv[1] == 'audit'
    audit()
