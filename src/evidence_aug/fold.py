"""Per-fold truth-free frames for the three known held partitions (fold0 screen, fold1/fold2 confirmation).

Each fold has its own pool-separated conditional-action reference features, its own held pools,
its own fixed predicted families and its own saved 382-feature 'conditional' evidence ranker.
"""
import polars as pl
from evidence_aug.paths import ORIGINAL, SCREEN, INVERSE, sha
from evidence_aug.frame import CACHES, assemble, held_pools as fold0_pools, roles as fold0_roles, FAMILIES

CONF = ORIGINAL / 'reports' / 'conditional-action-confirmation'
TABICL = ORIGINAL / 'reports' / 'tabicl-confirmation'


def fold_paths(fold):
    if fold == 0:
        return {'ca': CACHES['conditional'], 'held_pairs': ORIGINAL / 'reports' / 'tabicl-screen' / 'held-pairs.csv',
                'bundle': SCREEN / 'screen-results' / 'conditional-evidence.joblib',
                'saved_scores': ORIGINAL / 'reports' / 'conditional-behavior-experiment' / 'fold0' / 'conditional_behavior-scores.parquet'}
    d = CONF / 'results' / f'fold{fold}'
    return {'ca': CONF / f'conditional-fold{fold}-v1' / 'policy-screen' / 'hand-features.parquet', 'held_pairs': TABICL / f'fold{fold}' / 'held-pairs.csv',
            'bundle': d / 'conditional-evidence.joblib', 'saved_scores': d / 'conditional-evidence-scores.parquet', 'roles': d / 'pair-predictions.csv'}


def fold_pools(fold):
    if fold == 0:
        return fold0_pools()
    pools = sorted(pl.read_csv(fold_paths(fold)['held_pairs'])['table_id'].unique())
    assert len(pools) == 100
    return pools


def fold_roles(fold):
    """Fixed full-history predicted family of the fold's 'conditional' pair candidate."""
    if fold == 0:
        return fold0_roles('conditional_roles')
    frame = pl.read_csv(fold_paths(fold)['roles']).filter((pl.col('history') == 'full') & (pl.col('candidate') == 'conditional'))
    frame = frame.select('pair_id', pl.col('predicted_class').cast(pl.Int64).replace_strict(INVERSE, return_dtype=pl.String).alias('predicted_behavior'))
    assert frame['pair_id'].n_unique() == frame.height and set(frame['predicted_behavior']) <= set(FAMILIES) | {'none'}
    return frame


def load_fold(fold):
    """All hands of the 372 target pairs with the fold's conditional reference features; plus held pools."""
    labels = pl.read_csv(ORIGINAL / 'data' / 'development_labels.csv')
    raw = pl.read_parquet(sorted(CACHES['raw'].glob('batch_*.parquet')))
    targets = labels.filter(pl.col('label') == 1).select('pair_id', 'behavior_family')
    raw = raw.join(targets.select('pair_id'), on='pair_id', how='semi')
    paths = fold_paths(fold)
    ca = pl.read_parquet(paths['ca'])
    data, features, caf = assemble(raw, pl.read_parquet(CACHES['sequence']), pl.read_parquet(CACHES['equity']), ca, targets)
    pools = fold_pools(fold)
    identity = {'ca_sha256': sha(paths['ca']), 'held_pairs_sha256': sha(paths['held_pairs']), 'bundle_sha256': sha(paths['bundle']), 'saved_scores_sha256': sha(paths['saved_scores'])}
    return data, features, pools, identity
