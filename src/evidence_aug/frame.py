"""Truth-free construction of the 382-feature evidence frame for original target pairs.

Nothing here reads held-pool rows of data/development_evidence.csv. Relevance for
TRAINING pools is attached separately by `training_relevance`, which drops every
row outside the requested training pools before returning.
"""
import sys
from evidence_aug.paths import ORIGINAL, SRC, ROLE_FILES, INVERSE, sha
sys.path.insert(0, str(SRC))
import numpy as np
import polars as pl
from rank_evidence import context, family_columns, matrix, FAMILIES, KEYS, PARAMS  # noqa: F401
from sequence_evidence_features import NAMES as _SEQ_NAMES
from equity_evidence_features import FEATURES as EQ

SEQ = ['seq_' + n for n in _SEQ_NAMES]
CACHES = {
    'raw': ORIGINAL / 'outputs_v2' / 'hands_development',
    'sequence': ORIGINAL / 'reports' / 'sequence-evidence-experiment' / 'development.parquet',
    'equity': ORIGINAL / 'reports' / 'equity-evidence-experiment' / 'development.parquet',
    'conditional': ORIGINAL / 'reports' / 'conditional-action-experiment' / 'policy-screen' / 'hand-features.parquet',
}


def cache_identity():
    files = sorted(CACHES['raw'].glob('batch_*.parquet')) + [CACHES['sequence'], CACHES['equity'], CACHES['conditional']]
    return {str(p.relative_to(ORIGINAL)).replace('\\', '/'): sha(p) for p in files}


def assemble(raw, seq, eq, ca, families):
    """Shared 382-feature assembly for original and synthetic pair-hands.

    raw: 3 keys + 163 raw columns; seq/eq: pair_id, hand_id + features; ca: pair_id, hand_id (+table_id) + ca_ features;
    families: pair_id, behavior_family.
    """
    base = [c for c in raw.columns if c not in KEYS]
    events = [c for c in base if c.startswith('v2_')]
    data = raw.join(families.select('pair_id', 'behavior_family'), on='pair_id', validate='m:1')
    data = context(data, events)
    data = context(data.join(seq, on=['pair_id', 'hand_id'], validate='1:1'), SEQ)
    data = context(data.join(eq, on=['pair_id', 'hand_id'], validate='1:1'), EQ)
    caf = [c for c in ca.columns if c.startswith('ca_')]
    data = data.join(ca.drop([c for c in ['table_id'] if c in ca.columns]), on=['pair_id', 'hand_id'], how='left', validate='1:1')
    assert data.select(pl.sum_horizontal(pl.col(caf).null_count())).item() == 0
    data = context(data, caf).sort('pair_id', 'hand_id')
    features = (base + ['context_' + c for c in events] + ['family_' + f for f in FAMILIES]
                + SEQ + ['context_' + c for c in SEQ] + EQ + ['context_' + c for c in EQ] + caf + ['context_' + c for c in caf])
    assert len(features) == 382 and len(base) == 163 and len(events) == 56 and len(caf) == 33
    return data, features, caf


def load_original():
    """All hands of the 372 confirmed target pairs with the 382 ranker inputs (no relevance)."""
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet')))
    assert raw.select(pl.struct('pair_id', 'hand_id').n_unique()).item() == raw.height
    targets = labels.filter(pl.col('label') == 1).select('pair_id', 'behavior_family')
    raw = raw.join(targets.select('pair_id'), on='pair_id', how='semi')
    data, features, caf = assemble(raw, pl.read_parquet(CACHES['sequence']), pl.read_parquet(CACHES['equity']),
                                   pl.read_parquet(CACHES['conditional']), targets)
    assert data['pair_id'].n_unique() == 372
    return data, features, caf


def held_pools():
    """The 100 outer validation pools, derived from the fixed role file (no evidence truth)."""
    roles_frame = pl.read_csv(ROLE_FILES['original_roles'])
    held = pl.read_csv(ORIGINAL / 'reports' / 'tabicl-screen' / 'held-pairs.csv')
    assert set(roles_frame['pair_id']) == set(held['pair_id'])
    pools = sorted(held['table_id'].unique())
    assert len(pools) == 100
    return pools


def roles(kind):
    """Fixed predicted families for the 457 held pairs. Never uses true families."""
    if kind == 'original_roles':
        frame = pl.read_csv(ROLE_FILES[kind]).select('pair_id', 'predicted_behavior')
    else:
        frame = pl.read_csv(ROLE_FILES[kind]).select(
            'pair_id', pl.col('conditional_class').map_elements(lambda i: INVERSE[int(i)], return_dtype=pl.String).alias('predicted_behavior'))
    assert frame.height == 457 and frame['pair_id'].n_unique() == 457
    assert set(frame['predicted_behavior']) <= set(FAMILIES)
    return frame


def training_relevance(pools):
    """Listed-evidence relevance restricted to the given TRAINING pools.

    Reads the evidence file but drops every row that is not a training-pool pair,
    so held-pool truth never leaves this function.
    """
    raw_keys = pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet')), columns=['pair_id', 'table_id']).unique()
    allowed = raw_keys.filter(pl.col('table_id').is_in(list(pools))).select('pair_id')
    evidence = pl.read_csv(ORIGINAL / 'data' / 'development_evidence.csv').select('pair_id', 'hand_id')
    evidence = evidence.join(allowed, on='pair_id', how='semi')
    assert evidence.unique().height == evidence.height
    return evidence.with_columns(pl.lit(1, dtype=pl.Int64).alias('relevance'))


def attach_relevance(frame, evidence):
    out = frame.join(evidence, on=['pair_id', 'hand_id'], how='left').with_columns(pl.col('relevance').fill_null(0))
    assert out.height == frame.height
    return out


def score(model, frame, features):
    return model.predict(matrix(frame, features), num_threads=2)
