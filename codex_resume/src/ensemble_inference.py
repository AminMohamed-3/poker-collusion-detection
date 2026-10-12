"""Reloadable inference for the frozen equal-methods ensemble, without labels."""
from audit_common import *
sys.path.insert(0, str(HERE / 'vendor/catboost'))
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRanker
from symmetry_censor_audit import swapped
from pair_comparator_audit import tournament
from method_ensemble_audit import ranks


class BinaryBooster:
    def __init__(self, path):
        self.model = lgb.Booster(model_file=str(path))

    def predict_proba(self, x):
        p = self.model.predict(x, num_threads=2)
        return np.column_stack([1. - p, p])


def predict_saved(frame, directory):
    """Frame supplies named features and query IDs; no evidence labels accepted."""
    assert 'relevance' not in frame.columns
    recipe = json.loads((directory / 'recipe.json').read_text(encoding='utf-8'))
    for name, digest in recipe['model_sha256'].items():
        assert sha(directory / name) == digest
    cols = recipe['columns']
    x = frame.select(cols).to_numpy().astype(float)
    assert np.isfinite(x).all()
    xs = swapped(x, cols)
    pid = frame['pair_id'].to_numpy()
    base = BinaryBooster(directory / 'incumbent.txt')
    p = base.predict_proba(x)[:, 1]
    raw = {'incumbent': p, 'swap_average': .5 * (p + base.predict_proba(xs)[:, 1])}
    for name in ['symmetric_training', 'censor_half', 'censor_quarter', 'symmetric_censor_half']:
        model = BinaryBooster(directory / f'{name}.txt')
        pp = model.predict_proba(x)[:, 1]
        raw[name] = .5 * (pp + model.predict_proba(xs)[:, 1]) if name.startswith('symmetric') else pp
    for name, ranker in [('cat_logloss', False), ('cat_map', True)]:
        model = CatBoostRanker() if ranker else CatBoostClassifier()
        model.load_model(str(directory / f'{name}.cbm'))
        raw[name] = model.predict(x, thread_count=2) if ranker else model.predict_proba(x, thread_count=2)[:, 1]
    for name, context_n in [('pair_delta', 0), ('pair_context', recipe['shared_n'])]:
        raw[name] = tournament(BinaryBooster(directory / f'{name}.txt'), x, pid, context_n)
    groups = {'incumbent': ranks(pid, p)}
    for name, (_, arms) in recipe['groups'].items():
        groups[name] = np.mean([ranks(pid, raw[a]) for a in arms], axis=0)
    return p, sum(recipe['weights'][k] * groups[k] for k in groups)
